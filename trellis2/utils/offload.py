"""
CPU offload helpers for running TRELLIS.2 on GPUs that cannot hold a full DiT.

TRELLIS.2 is a cascade of independently trained modules (sparse-structure DiT,
shape SLat DiTs, texture SLat DiTs, VAEs). `low_vram` already keeps unused
modules on CPU. That is not enough when a *single* 1.3B bf16 DiT (~2.6 GB of
weights) plus activations exceeds the card.

Sequential block offload keeps transformer/conv blocks on CPU (optionally
pinned) and moves one block to the GPU for each forward, then moves it back.
"""
from __future__ import annotations

import gc
import os
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn


_OFFLOAD_HOOK_ATTR = "_trellis2_block_offload_hooks"
_OFFLOAD_FLAG_ATTR = "block_offload"
_OFFLOAD_DEVICE_ATTR = "_trellis2_offload_exec_device"


def gpu_total_memory_gb(device: Optional[Union[str, torch.device, int]] = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        device = torch.device(device)
        index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)


def gpu_free_memory_gb(device: Optional[Union[str, torch.device, int]] = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        device = torch.device(device)
        index = device.index if device.index is not None else torch.cuda.current_device()
    free, _total = torch.cuda.mem_get_info(index)
    return free / (1024 ** 3)


def recommend_block_offload(threshold_gb: float = 12.0) -> bool:
    """Enable layer offload on cards that cannot hold a 1.3B bf16 DiT plus activations."""
    total = gpu_total_memory_gb()
    if total <= 0:
        return False
    return total <= threshold_gb


def recommend_pipeline_type(threshold_gb: float = 8.0) -> Optional[str]:
    """
    On very small GPUs, stay on the 512^3 models. Those already completed on a
    4 GB RTX 3050; the 1024 cascade is what blows past VRAM.
    """
    total = gpu_total_memory_gb()
    if total > 0 and total < threshold_gb:
        return "512"
    return None


def module_nbytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def host_memory_gb() -> Tuple[float, float]:
    """Return (available_gb, total_gb) from /proc/meminfo, or (0, 0)."""
    try:
        info = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = int(value.strip().split()[0]) / (1024 ** 2)
        total = info.get("MemTotal", 0.0)
        available = info.get("MemAvailable", info.get("MemFree", 0.0))
        return available, total
    except Exception:
        return 0.0, 0.0


def recommend_pin_memory() -> bool:
    """Pinned weights cannot be swapped. Only pin when plenty of host RAM remains."""
    env = os.environ.get("TRELLIS_PIN_MEMORY", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    available, total = host_memory_gb()
    if total and total < 24:
        return False
    if available and available < 12:
        return False
    return False


def log_host_memory(tag: str) -> None:
    available, total = host_memory_gb()
    if not total:
        return
    print(f"[RAM] {tag}: available={available:.1f}GiB / total={total:.1f}GiB")


def log_vram(tag: str) -> None:
    flag = os.environ.get("TRELLIS_VRAM_LOG", "").strip().lower()
    if flag not in ("1", "true", "yes"):
        return
    if not torch.cuda.is_available():
        print(f"[VRAM] {tag}: CUDA unavailable")
        return
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    free = gpu_free_memory_gb() * 1024
    print(f"[VRAM] {tag}: allocated={allocated:.0f}MiB reserved={reserved:.0f}MiB free={free:.0f}MiB")


def release_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def iter_offload_blocks(model: nn.Module) -> Iterator[nn.Module]:
    """Yield leaf blocks from `model.blocks` (flat or nested ModuleList)."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return
    for item in blocks:
        if isinstance(item, nn.ModuleList):
            for sub in item:
                if isinstance(sub, nn.Module):
                    yield sub
        elif isinstance(item, nn.Module):
            yield item


def pin_module(module: nn.Module) -> None:
    """Pin CPU parameters so later HtoD copies can be non-blocking."""
    module.to("cpu")
    for p in module.parameters():
        if p.device.type == "cpu" and not p.is_pinned():
            try:
                p.data = p.data.pin_memory()
            except RuntimeError:
                pass
    for b in module.buffers():
        if b.device.type == "cpu" and b.is_floating_point() and not b.is_pinned():
            try:
                b.data = b.data.pin_memory()
            except RuntimeError:
                pass


def _move_module(module: nn.Module, device: Union[str, torch.device]) -> None:
    device = torch.device(device)
    kwargs = {}
    if device.type == "cuda":
        kwargs["non_blocking"] = True
    module.to(device, **kwargs)


def _pre_hook(exec_device: Union[str, torch.device]):
    def hook(module: nn.Module, _inputs):
        _move_module(module, exec_device)
    return hook


def _post_hook():
    def hook(module: nn.Module, _inputs, output):
        _move_module(module, "cpu")
        return output
    return hook


def disable_module_block_offload(model: nn.Module) -> None:
    hooks: List = getattr(model, _OFFLOAD_HOOK_ATTR, None) or []
    for hook in hooks:
        hook.remove()
    if hasattr(model, _OFFLOAD_HOOK_ATTR):
        delattr(model, _OFFLOAD_HOOK_ATTR)
    if hasattr(model, _OFFLOAD_DEVICE_ATTR):
        delattr(model, _OFFLOAD_DEVICE_ATTR)
    setattr(model, _OFFLOAD_FLAG_ATTR, False)


def enable_module_block_offload(
    model: nn.Module,
    execution_device: Union[str, torch.device],
    pin_memory: Optional[bool] = None,
) -> int:
    """
    Install forward hooks so each block is copied to `execution_device` just
    before its forward and copied back to CPU afterwards.

    Returns the number of hooked blocks.
    """
    blocks = list(iter_offload_blocks(model))
    if not blocks:
        setattr(model, _OFFLOAD_FLAG_ATTR, False)
        return 0

    prev_device = getattr(model, _OFFLOAD_DEVICE_ATTR, None)
    if getattr(model, _OFFLOAD_HOOK_ATTR, None) and str(prev_device) == str(execution_device):
        setattr(model, _OFFLOAD_FLAG_ATTR, True)
        return len(blocks)

    disable_module_block_offload(model)

    if pin_memory is None:
        pin_memory = recommend_pin_memory()

    hooks = []
    nbytes = 0
    for block in blocks:
        if pin_memory:
            pin_module(block)
        else:
            block.to("cpu")
        nbytes += module_nbytes(block)
        hooks.append(block.register_forward_pre_hook(_pre_hook(execution_device)))
        hooks.append(block.register_forward_hook(_post_hook()))

    setattr(model, _OFFLOAD_HOOK_ATTR, hooks)
    setattr(model, _OFFLOAD_DEVICE_ATTR, torch.device(execution_device))
    setattr(model, _OFFLOAD_FLAG_ATTR, True)
    return len(blocks)


def move_non_block_modules(model: nn.Module, device: Union[str, torch.device]) -> None:
    """Move everything except `blocks` onto `device` (embeddings, I/O layers)."""
    device = torch.device(device)
    for name, child in model.named_children():
        if name == "blocks":
            continue
        child.to(device)
    for name, buf in list(model.named_buffers(recurse=False)):
        if buf.device != device:
            model._buffers[name] = buf.to(device)
    for _name, param in model.named_parameters(recurse=False):
        if param.device != device:
            param.data = param.data.to(device)


def stage_model(
    model,
    device: Union[str, torch.device],
    on_gpu: bool,
    block_offload: bool,
) -> None:
    """Place a pipeline submodule on GPU or CPU, optionally with block offload."""
    if model is None:
        return
    if on_gpu:
        log_vram(f"before stage {type(model).__name__}")
        already = bool(getattr(model, _OFFLOAD_HOOK_ATTR, None)) and str(
            getattr(model, _OFFLOAD_DEVICE_ATTR, None)
        ) == str(torch.device(device))
        if block_offload and isinstance(model, nn.Module) and hasattr(model, "blocks"):
            n_blocks = enable_module_block_offload(model, device, pin_memory=recommend_pin_memory())
            move_non_block_modules(model, device)
            if n_blocks and not already and os.environ.get("TRELLIS_OFFLOAD_LOG", "1").strip().lower() not in ("0", "false", "no"):
                gb = module_nbytes(model) / (1024 ** 3)
                print(
                    f"[TRELLIS.2] Block CPU offload: {type(model).__name__} "
                    f"({n_blocks} blocks, ~{gb:.2f} GB weights stay in RAM)"
                )
        elif hasattr(model, "to"):
            model.to(device)
        log_vram(f"after stage {type(model).__name__}")
    else:
        if block_offload and isinstance(model, nn.Module) and hasattr(model, "blocks"):
            # Keep hooked blocks on (pinned) CPU; only evict embeddings / I/O.
            move_non_block_modules(model, "cpu")
        elif hasattr(model, "cpu"):
            model.cpu()
        elif hasattr(model, "to"):
            model.to("cpu")
        release_cuda_memory()
        log_vram(f"after unload {type(model).__name__}")


def models_for_pipeline_type(pipeline_type: str) -> List[str]:
    """Subset of checkpoint keys needed for a given `pipeline.run(..., pipeline_type=)`."""
    ss = [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
    ]
    shape_dec = ["shape_slat_decoder"]
    tex_dec = ["tex_slat_decoder"]
    mapping = {
        "512": ss + ["shape_slat_flow_model_512"] + shape_dec + ["tex_slat_flow_model_512"] + tex_dec,
        "1024": ss + ["shape_slat_flow_model_1024"] + shape_dec + ["tex_slat_flow_model_1024"] + tex_dec,
        "1024_cascade": ss
        + ["shape_slat_flow_model_512", "shape_slat_flow_model_1024"]
        + shape_dec
        + ["tex_slat_flow_model_1024"]
        + tex_dec,
        "1536_cascade": ss
        + ["shape_slat_flow_model_512", "shape_slat_flow_model_1024"]
        + shape_dec
        + ["tex_slat_flow_model_1024"]
        + tex_dec,
    }
    if pipeline_type not in mapping:
        raise ValueError(f"Unknown pipeline_type {pipeline_type!r}")
    return mapping[pipeline_type]


def drop_module(obj) -> None:
    """Drop a model from RAM (hooks, parameters, then GC)."""
    if obj is None:
        return
    if isinstance(obj, nn.Module):
        disable_module_block_offload(obj)
    if hasattr(obj, "cpu"):
        try:
            obj.cpu()
        except Exception:
            pass
    del obj
    release_cuda_memory()


class LazyModelMap(dict):
    """Dict that loads a checkpoint the first time a key is accessed."""

    def __init__(self, load_fn, specs: dict):
        super().__init__()
        self._load_fn = load_fn
        self._specs = dict(specs)

    def __contains__(self, key):
        return key in self._specs or super().__contains__(key)

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if key not in self._specs:
            raise KeyError(key)
        model = self._load_fn(key, self._specs[key])
        super().__setitem__(key, model)
        return model

    def pop(self, key, default=None):
        self._specs.pop(key, None)
        if super().__contains__(key):
            return super().pop(key)
        return default


def drop_models(store: dict, keys: Iterable[str]) -> List[str]:
    """Delete named entries from a pipeline model dict and free host memory."""
    dropped = []
    for key in keys:
        model = store.pop(key, None)
        if model is None:
            continue
        drop_module(model)
        dropped.append(key)
    if dropped:
        available, total = host_memory_gb()
        extra = f" (RAM {available:.1f}/{total:.1f} GiB free)" if total else ""
        print(f"[TRELLIS.2] Freed models from RAM: {', '.join(dropped)}{extra}")
    return dropped
