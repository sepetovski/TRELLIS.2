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


def recommend_lr_tokens() -> Optional[int]:
    """
    Occupied-voxel cap for the 512 shape-SLat pass.

    `T.png` is a few thousand voxels and is never capped. On a 4 GB card
    the shape-SLat MLP dies with `device not ready` once occupancy gets
    too dense: 3783 voxels (fairy) finished, 6712 (house2) died on step 0.
    The cap stays at 4096. Interior voxels are dropped before the shell is
    thinned. Override with TRELLIS_LR_TOKENS (a count, or `full` to keep
    every voxel — that can TDR the card).
    """
    env = os.environ.get("TRELLIS_LR_TOKENS", "").strip().lower()
    if env in ("full", "off", "none"):
        return None
    if env:
        return int(env)
    total = gpu_total_memory_gb()
    if total <= 0:
        return None
    if total < 6:
        return 4096
    if total < 8:
        return 8192
    return None


def recommend_sequential_cfg() -> bool:
    """Park the CFG positive prediction on CPU before the negative forward.

    Off by default. Token capping is what keeps 4 GB sampling alive; the extra
    synchronize on every step made DiT sampling several times slower.
    Enable with TRELLIS_SEQ_CFG=1 if a dense occupancy still TDRs.
    """
    env = os.environ.get("TRELLIS_SEQ_CFG", "").strip().lower()
    return env in ("1", "true", "yes")


# Shape VAE conv at 1,988,316 voxels finished on a 4 GB card. The texture
# VAE runs the same conv afterwards, while the mesh is still resident, and
# dies in the flex_gemm neighbor map (`device not ready`). Stay under that
# peak. 449,867 parents still all keep a child; extra children fill the rest.
TEX_DECODE_VOXEL_CAP = 1_500_000


def recommend_tex_decode_voxels() -> Optional[int]:
    """
    Max children the texture VAE may spawn in one upsample on a small GPU.

    None means keep every positive subdivision logit. Override with
    TRELLIS_TEX_VOXELS (a count, or `full` to disable).
    """
    env = os.environ.get("TRELLIS_TEX_VOXELS", "").strip().lower()
    if env in ("full", "off", "none"):
        return None
    if env:
        return int(env)
    total = gpu_total_memory_gb()
    if total > 0 and total < 6:
        return TEX_DECODE_VOXEL_CAP
    return None


def limit_subdiv_feats(feats: torch.Tensor, max_out: Optional[int]) -> torch.Tensor:
    """
    Thin [N, 8] subdivision logits so at most `max_out` children stay positive.

    Each row is one parent voxel and each column is one of 8 children.
    SparseChannel2Spatial keeps children with logit > 0. When that would
    exceed `max_out`, keep the strongest child of as many parents as possible,
    then spend the rest of the budget on the next-highest positive logits.
    """
    if feats is None or max_out is None or max_out < 0:
        return feats
    if feats.ndim != 2 or feats.shape[0] == 0 or feats.shape[-1] == 0:
        return feats
    if not feats.is_floating_point():
        return feats
    positive = feats > 0
    n_pos = int(positive.sum().item())
    if n_pos <= max_out:
        return feats
    if max_out == 0:
        return feats.masked_fill(positive, 0)

    masked = feats.masked_fill(~positive, float("-inf"))
    best_logit, best_idx = masked.max(dim=-1)
    has_child = torch.isfinite(best_logit)
    n_parents = int(has_child.sum().item())
    keep = torch.zeros_like(positive)
    if n_parents <= max_out:
        parent_rows = has_child.nonzero(as_tuple=False).flatten()
        keep[parent_rows, best_idx[parent_rows]] = True
        budget = max_out - n_parents
        if budget > 0:
            rest = positive & ~keep
            n_rest = int(rest.sum().item())
            if n_rest > 0:
                rest_logits = feats.masked_fill(~rest, float("-inf")).reshape(-1)
                k = min(budget, n_rest)
                flat = torch.topk(rest_logits, k, largest=True, sorted=False).indices
                keep.view(-1)[flat] = True
    else:
        parent_score = best_logit.masked_fill(~has_child, float("-inf"))
        top_parents = torch.topk(parent_score, max_out, largest=True, sorted=False).indices
        keep[top_parents, best_idx[top_parents]] = True

    drop = positive & ~keep
    return feats.masked_fill(drop, 0)


def recommend_upsample_voxels() -> Optional[int]:
    """
    Max voxels allowed during cascade VAE C2S upsample.

    0 means skip the 4-level VAE upsample and integer-scale LR occupancy
    (required on 4 GB — that C2S is what TDRs dragon.png after the 512 pass).
    None means run the full VAE upsample. Override with TRELLIS_UPSAMPLE_VOXELS
    (`0`, a count, or `full`).
    """
    env = os.environ.get("TRELLIS_UPSAMPLE_VOXELS", "").strip().lower()
    if env in ("full", "off", "none"):
        return None
    if env:
        return int(env)
    total = gpu_total_memory_gb()
    if total > 0 and total < 8:
        return 0
    return None


def scale_coords_upsample(coords: torch.Tensor, upsample_times: int) -> torch.Tensor:
    """Integer-scale occupancy coords as if `upsample_times` 2× VAE C2S ran."""
    if coords is None or upsample_times <= 0:
        return coords
    out = coords.clone()
    out[:, 1:] = out[:, 1:] * (2 ** int(upsample_times))
    return out


def dilate_occupancy_coords(coords: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Replace each occupancy voxel with an `factor`³ block (no VAE)."""
    if coords is None or factor <= 1:
        return coords
    device = coords.device
    rng = torch.arange(int(factor), device=device, dtype=coords.dtype)
    offs = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1).reshape(-1, 3)
    k = offs.shape[0]
    batch = coords[:, :1].repeat_interleave(k, dim=0)
    xyz = coords[:, 1:].repeat_interleave(k, dim=0) * int(factor) + offs.repeat(coords.shape[0], 1)
    return torch.cat([batch, xyz], dim=1).contiguous()


def _drop_interior_voxels(coords: torch.Tensor) -> torch.Tensor:
    """Drop voxels whose six face neighbors are also occupied. The shell stays."""
    if coords.shape[0] < 64:
        return coords
    xyz = coords[:, 1:].detach().cpu().tolist()
    occupied = set(map(tuple, xyz))
    keep = []
    for index, point in enumerate(xyz):
        x, y, z = point
        interior = (
            (x + 1, y, z) in occupied and (x - 1, y, z) in occupied
            and (x, y + 1, z) in occupied and (x, y - 1, z) in occupied
            and (x, y, z + 1) in occupied and (x, y, z - 1) in occupied
        )
        if not interior:
            keep.append(index)
    if len(keep) == coords.shape[0]:
        return coords
    index = torch.tensor(keep, dtype=torch.long, device=coords.device)
    return coords.index_select(0, index)


def cap_sparse_coords(coords: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Keep at most `max_tokens` occupancy voxels while covering the same volume."""
    if coords is None or max_tokens is None or coords.shape[0] <= max_tokens:
        return coords
    peeled = _drop_interior_voxels(coords)
    if peeled.shape[0] < coords.shape[0]:
        print(
            f"[TRELLIS.2] Dropped {coords.shape[0] - peeled.shape[0]} interior "
            f"voxels, {peeled.shape[0]} left on the shell."
        )
        coords = peeled
    if coords.shape[0] <= max_tokens:
        return coords
    batch = coords[:, :1]
    xyz = coords[:, 1:]
    selected = coords
    cell = 2
    while selected.shape[0] > max_tokens and cell <= 32:
        q = torch.cat([batch, xyz // cell], dim=1)
        inverse = torch.unique(q, dim=0, return_inverse=True)[1]
        order = torch.argsort(inverse, stable=True)
        inv_sorted = inverse[order]
        mask = torch.ones(inv_sorted.shape[0], dtype=torch.bool, device=coords.device)
        mask[1:] = inv_sorted[1:] != inv_sorted[:-1]
        selected = coords[order[mask].sort().values]
        cell *= 2
    if selected.shape[0] > max_tokens:
        n = selected.shape[0]
        idx = torch.linspace(0, n - 1, steps=max_tokens, device=selected.device)
        idx = idx.round().long().unique()[:max_tokens]
        selected = selected[idx]
    return selected.contiguous()


def park_activation(x):
    """Move a sampler prediction to CPU so the next forward can reuse VRAM."""
    if x is None:
        return x
    if hasattr(x, "cpu"):
        return x.cpu()
    return x


def unpark_activation(x, like):
    """Copy a parked prediction back to the device of `like`."""
    if x is None:
        return x
    device = None
    if torch.is_tensor(like):
        device = like.device
    elif hasattr(like, "feats") and torch.is_tensor(like.feats):
        device = like.feats.device
    elif hasattr(like, "device"):
        device = like.device
    if device is None or not hasattr(x, "to"):
        return x
    return x.to(device)


def ensure_cuda_ready() -> None:
    """Fail fast if a previous `device not ready` left the CUDA context dead."""
    if not torch.cuda.is_available():
        return
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "CUDA is dead (`device not ready`). A reboot is not enough if WSL "
            "kept the GPU. In PowerShell run: wsl --shutdown\n"
            "Then reopen Ubuntu and retry."
        ) from exc


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


def is_nested_block_model(model: nn.Module) -> bool:
    """VAEs store `blocks` as a list of resolution stages, not a flat DiT."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return False
    return any(isinstance(item, nn.ModuleList) for item in blocks)


def should_dit_block_offload(model: nn.Module) -> bool:
    """Only stream large flat DiTs. 36 tiny VAE convs on a 4 GB card just fault the driver."""
    if not isinstance(model, nn.Module) or not hasattr(model, "blocks"):
        return False
    if is_nested_block_model(model):
        return False
    return module_nbytes(model) >= int(1.2 * 1024 ** 3)


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
        if block_offload and is_nested_block_model(model):
            move_non_block_modules(model, device)
            for stage in model.blocks:
                stage.to("cpu")
            if hasattr(model, "low_vram"):
                model.low_vram = True
            if os.environ.get("TRELLIS_OFFLOAD_LOG", "1").strip().lower() not in ("0", "false", "no"):
                print(
                    f"[TRELLIS.2] VAE stage offload: {type(model).__name__} "
                    f"({len(model.blocks)} resolution levels, one on GPU at a time)"
                )
        elif block_offload and should_dit_block_offload(model):
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
            move_non_block_modules(model, "cpu")
            if is_nested_block_model(model):
                for stage in model.blocks:
                    stage.to("cpu")
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

    def unload(self, key) -> bool:
        """Drop a loaded instance but keep the spec so it can be loaded again."""
        if super().__contains__(key):
            model = super().pop(key)
            drop_module(model)
            return True
        return False


def unload_models(store: dict, keys: Iterable[str]) -> List[str]:
    """Free RAM for models that will be needed again later (keeps checkpoint specs)."""
    unloaded = []
    for key in keys:
        if isinstance(store, LazyModelMap):
            if store.unload(key):
                unloaded.append(key)
            continue
        model = store.get(key)
        if model is None:
            continue
        drop_module(model)
        store[key] = None
        unloaded.append(key)
    if unloaded:
        available, total = host_memory_gb()
        extra = f" (RAM {available:.1f}/{total:.1f} GiB free)" if total else ""
        print(f"[TRELLIS.2] Unloaded (reloadable): {', '.join(unloaded)}{extra}")
    return unloaded


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
