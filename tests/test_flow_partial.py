import unittest

import torch

from trellis2.pipelines.samplers.flow_euler import (
    FlowEulerGuidanceIntervalSampler,
    FlowEulerSampler,
    flow_time_schedule,
)


class _FakeSparse:
    def __init__(self, feats):
        self.feats = feats
        self.shape = (1,)
        self.device = feats.device

    def replace(self, feats):
        return _FakeSparse(feats)

    def _other(self, other):
        return other.feats if isinstance(other, _FakeSparse) else other

    def __add__(self, other):
        return _FakeSparse(self.feats + self._other(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return _FakeSparse(self.feats - self._other(other))

    def __mul__(self, other):
        return _FakeSparse(self.feats * self._other(other))

    def __rmul__(self, other):
        return self.__mul__(other)


def _perfect(x0, eps, sigma_min):
    velocity = (1.0 - sigma_min) * eps - x0

    def model(x, t, cond=None, **kwargs):
        return velocity

    return model


class FlowPartialTests(unittest.TestCase):
    def test_full_schedule_matches_the_old_linspace(self):
        steps, rescale = 12, 3.0
        got = flow_time_schedule(steps, rescale, 1.0)
        seq = torch.linspace(1, 0, steps + 1).numpy()
        seq = rescale * seq / (1 + (rescale - 1) * seq)
        self.assertEqual(len(got), steps + 1)
        self.assertAlmostEqual(got[0], float(seq[0]), places=6)
        self.assertAlmostEqual(got[-1], 0.0, places=6)

    def test_partial_schedule_starts_at_t_start_and_ends_at_zero(self):
        got = flow_time_schedule(4, 1.0, 0.6)
        self.assertAlmostEqual(got[0], 0.6, places=6)
        self.assertAlmostEqual(got[-1], 0.0, places=6)
        self.assertTrue(all(got[i] > got[i + 1] for i in range(len(got) - 1)))
        self.assertEqual(flow_time_schedule(4, 1.0, 0.0), [0.0])

    def test_zero_velocity_generation_is_unchanged(self):
        sampler = FlowEulerSampler(1e-5)
        noise = torch.randn(2, 4)
        calls = {"n": 0}

        def model(x, t, cond=None, **kwargs):
            calls["n"] += 1
            return torch.zeros_like(x)

        out = sampler.sample(model, noise, steps=4, rescale_t=1.0, verbose=False)
        self.assertEqual(calls["n"], 4)
        torch.testing.assert_close(out.samples, noise)

    def test_strength_zero_skips_the_model(self):
        sampler = FlowEulerSampler(1e-5)
        x0 = torch.randn(2, 4)
        calls = {"n": 0}

        def model(x, t, cond=None, **kwargs):
            calls["n"] += 1
            return torch.ones_like(x)

        out = sampler.sample(model, torch.randn_like(x0), x_0=x0, t_start=0.0, steps=4, verbose=False)
        self.assertEqual(calls["n"], 0)
        torch.testing.assert_close(out.samples, x0)

    def test_perfect_model_reconstructs_from_partial_noise(self):
        sigma = 1e-5
        sampler = FlowEulerSampler(sigma)
        x0 = torch.randn(2, 4)
        eps = torch.randn_like(x0)
        out = sampler.sample(
            _perfect(x0, eps, sigma),
            eps,
            x_0=x0,
            t_start=0.5,
            steps=8,
            rescale_t=3.0,
            verbose=False,
        )
        torch.testing.assert_close(out.samples, x0 + sigma * eps, atol=1e-4, rtol=1e-4)

    def test_repaint_pins_known_tokens_against_a_bad_model(self):
        sampler = FlowEulerSampler(1e-5)
        x0 = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        mask = torch.tensor([[True, True, False, False], [False, True, False, True]])

        def model(x, t, cond=None, **kwargs):
            return torch.ones_like(x) * 50

        out = sampler.sample(
            model, torch.randn_like(x0), x_0=x0, t_start=1.0,
            repaint_mask=mask, steps=3, verbose=False,
        )
        pinned = out.samples[~mask]
        torch.testing.assert_close(pinned, x0[~mask])
        self.assertTrue(bool((out.samples[mask] - x0[mask]).abs().mean() > 0.1))

    def test_repaint_on_a_duck_typed_sparse_tensor(self):
        sampler = FlowEulerSampler(1e-5)
        x0 = _FakeSparse(torch.tensor([[1.0], [2.0], [3.0]]))
        eps = _FakeSparse(torch.zeros(3, 1))
        mask = torch.tensor([False, True, False])

        def model(x, t, cond=None, **kwargs):
            return _FakeSparse(torch.ones_like(x.feats) * 7)

        out = sampler.sample(
            model, eps, x_0=x0, t_start=1.0, repaint_mask=mask, steps=2, verbose=False,
        )
        self.assertIsInstance(out.samples, _FakeSparse)
        torch.testing.assert_close(out.samples.feats[0, 0], torch.tensor(1.0))
        torch.testing.assert_close(out.samples.feats[2, 0], torch.tensor(3.0))
        self.assertGreater(float(out.samples.feats[1, 0].abs()), 0.1)

    def test_guidance_sampler_accepts_partial_kwargs(self):
        sigma = 1e-5
        sampler = FlowEulerGuidanceIntervalSampler(sigma)
        x0 = torch.randn(1, 4)
        eps = torch.randn_like(x0)
        out = sampler.sample(
            _perfect(x0, eps, sigma),
            eps,
            cond=torch.zeros(1, 2),
            neg_cond=torch.zeros(1, 2),
            steps=4,
            rescale_t=1.0,
            guidance_strength=1.0,
            guidance_interval=(0.6, 1.0),
            verbose=False,
            x_0=x0,
            t_start=0.5,
        )
        torch.testing.assert_close(out.samples, x0 + sigma * eps, atol=1e-4, rtol=1e-4)

    def test_repaint_requires_x0(self):
        sampler = FlowEulerSampler(1e-5)
        with self.assertRaises(ValueError):
            sampler.sample(lambda *a, **k: None, torch.zeros(1, 2), repaint_mask=torch.tensor([True, False]), verbose=False)


if __name__ == "__main__":
    unittest.main()
