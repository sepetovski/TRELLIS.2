from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
from .. import models
from ..utils import offload


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()
        self.block_offload = getattr(self, "block_offload", "auto")

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_file: str = "pipeline.json",
        models_to_load: Optional[List[str]] = None,
    ) -> "Pipeline":
        """
        Load a pretrained model.

        Args:
            path: Local directory or Hugging Face repo id.
            config_file: Pipeline config filename.
            models_to_load: Optional subset of checkpoint keys. Use this to skip
                1024-resolution DiTs when running the 512 pipeline on a small GPU.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        names = models_to_load
        if names is None and hasattr(cls, 'model_names_to_load'):
            names = cls.model_names_to_load

        _models = {}
        for k, v in args['models'].items():
            if names is not None and k not in names:
                continue
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                _models[k] = models.from_pretrained(v)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    def _resolve_block_offload(self) -> bool:
        flag = getattr(self, "block_offload", "auto")
        if flag == "auto":
            enabled = offload.recommend_block_offload()
            self.block_offload = enabled
            if enabled:
                print(
                    "[TRELLIS.2] GPU has <=12 GB VRAM. Enabling sequential "
                    "transformer-block CPU offload (one layer on GPU at a time)."
                )
            return enabled
        return bool(flag)

    def _stage_model(self, model, on_gpu: bool) -> None:
        if not getattr(self, "low_vram", True):
            if on_gpu and hasattr(model, "to"):
                model.to(self.device)
            return
        offload.stage_model(
            model,
            self.device,
            on_gpu=on_gpu,
            block_offload=self._resolve_block_offload(),
        )

    @contextmanager
    def _model_on_device(self, model):
        self._stage_model(model, True)
        try:
            yield model
        finally:
            self._stage_model(model, False)

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))