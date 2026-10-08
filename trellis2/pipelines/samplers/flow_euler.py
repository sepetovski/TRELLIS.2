from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


def flow_time_schedule(steps: int, rescale_t: float = 1.0, t_start: float = 1.0) -> list:
    """Warped timestep schedule from ``t_start`` down to 0.

    ``t_start=1`` is the original full schedule (pure noise). Smaller values
    skip the early steps so a real latent can be noised only part-way and
    then denoised — the img2img / refine path. ``steps`` is the full-schedule
    length; the returned list is the suffix that still has to be integrated,
    with the first entry forced to ``t_start``.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    t_start = float(t_start)
    if t_start < 0.0 or t_start > 1.0:
        raise ValueError(f"t_start must be in [0, 1], got {t_start}")
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    if t_start >= float(t_seq[0]) - 1e-8:
        return t_seq.tolist()
    if t_start <= 1e-8:
        return [0.0]
    tail = [float(t) for t in t_seq if float(t) < t_start - 1e-8]
    return [t_start, *tail]


def _is_varlen(x) -> bool:
    return hasattr(x, "feats") and hasattr(x, "replace")


def _randn_like(x):
    if _is_varlen(x):
        return x.replace(torch.randn_like(x.feats))
    return torch.randn_like(x)


def _blend_repaint(sample, pinned, repaint_mask):
    """Keep ``sample`` where ``repaint_mask`` is True; keep ``pinned`` elsewhere."""

    def mix(feats, pinned_feats, mask):
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.to(device=feats.device)
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        if mask.shape[0] != feats.shape[0]:
            raise ValueError(
                f"repaint_mask has {mask.shape[0]} rows, latent has {feats.shape[0]}"
            )
        while mask.ndim < feats.ndim:
            mask = mask.unsqueeze(-1)
        pinned_feats = pinned_feats.to(device=feats.device, dtype=feats.dtype)
        return torch.where(mask, feats, pinned_feats)

    if _is_varlen(sample):
        pinned_feats = pinned.feats if _is_varlen(pinned) else pinned
        return sample.replace(mix(sample.feats, pinned_feats, repaint_mask))
    return mix(sample, pinned, repaint_mask)


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _diffuse(self, x_0, t: float, eps):
        """q(x_t | x_0) for the rectified-flow schedule used at training time."""
        t = float(t)
        return (1.0 - t) * x_0 + (self.sigma_min + (1.0 - self.sigma_min) * t) * eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        store_intermediates: bool = False,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            store_intermediates: If True, also compute and return pred_x_0.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0 (only if store_intermediates).
        """
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        ret = edict({"pred_x_prev": pred_x_prev})
        if store_intermediates:
            pred_x_0, _pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
            ret.pred_x_0 = pred_x_0
        return ret

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        store_intermediates: bool = False,
        t_start: float = 1.0,
        x_0: Optional[Any] = None,
        repaint_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor. When ``x_0`` is set, this is the
                epsilon used to noise it (random if omitted).
            cond: conditional information.
            steps: The number of steps in the *full* 1→0 schedule. A
                ``t_start`` below 1 runs only the remaining suffix.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            store_intermediates: If True, keep pred_x_t / pred_x_0 for every step
                on device. Off by default — those tensors are unused by the
                pipeline and accumulate VRAM across steps.
            t_start: Noise level to start from. 1 is pure noise (generation).
                0 returns ``x_0`` without calling the model.
            x_0: Clean latent to noise up to ``t_start`` (refine / img2img).
            repaint_mask: Per-token bool, True = let the model change this
                token, False = pin it to ``x_0`` at every step (RePaint).
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        if repaint_mask is not None and x_0 is None:
            raise ValueError("repaint_mask requires x_0, the clean latent to keep outside the mask")
        t_start = float(t_start)
        if t_start < 0.0 or t_start > 1.0:
            raise ValueError(f"t_start must be in [0, 1], got {t_start}")
        if x_0 is not None and t_start <= 1e-8:
            return edict({"samples": x_0, "pred_x_t": [], "pred_x_0": []})

        eps = None
        if x_0 is not None:
            eps = noise if noise is not None else _randn_like(x_0)
            sample = self._diffuse(x_0, t_start, eps)
            schedule_start = t_start
        else:
            if t_start < 1.0 - 1e-8:
                raise ValueError("t_start < 1 requires x_0 (the clean latent to noise from)")
            sample = noise
            eps = noise
            schedule_start = 1.0

        t_seq = flow_time_schedule(steps, rescale_t, schedule_start)
        ret = edict({"samples": sample, "pred_x_t": [], "pred_x_0": []})
        if len(t_seq) < 2:
            return ret
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(len(t_seq) - 1))
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(
                model, sample, t, t_prev, cond,
                store_intermediates=store_intermediates,
                **kwargs,
            )
            sample = out.pred_x_prev
            if repaint_mask is not None:
                if t_prev <= 1e-8:
                    pinned = x_0
                else:
                    pinned = self._diffuse(x_0, t_prev, eps)
                sample = _blend_repaint(sample, pinned, repaint_mask)
            if store_intermediates:
                ret.pred_x_t.append(out.pred_x_prev)
                ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)
