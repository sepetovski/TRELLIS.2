import unittest

from trellis2.utils.bake_limits import (
    RAISED_DECIMATION_TARGET,
    LOW_VRAM_TEXTURE_SIZE,
    FULL_TEXTURE_SIZE,
    bake_size_attempts,
    default_texture_size,
    is_cuda_oom,
    is_cuda_context_dead,
)


class BakeLimitTests(unittest.TestCase):
    def test_raised_defaults(self):
        self.assertEqual(RAISED_DECIMATION_TARGET, 1_000_000)
        self.assertEqual(LOW_VRAM_TEXTURE_SIZE, 2048)
        self.assertEqual(FULL_TEXTURE_SIZE, 4096)
        self.assertEqual(default_texture_size(4.0), 2048)
        self.assertEqual(default_texture_size(24.0), 4096)

    def test_attempts_step_down_without_growing(self):
        self.assertEqual(
            bake_size_attempts(1_000_000, 4096),
            [(1_000_000, 4096), (500_000, 2048), (100_000, 1024)],
        )
        self.assertEqual(
            bake_size_attempts(1_000_000, 2048),
            [(1_000_000, 2048), (500_000, 2048), (100_000, 1024)],
        )

    def test_request_below_a_rung_is_not_raised(self):
        self.assertEqual(
            bake_size_attempts(200_000, 2048),
            [(200_000, 2048), (100_000, 1024)],
        )

    def test_smallest_bake_is_a_single_attempt(self):
        self.assertEqual(bake_size_attempts(100_000, 1024), [(100_000, 1024)])

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            bake_size_attempts(0, 1024)

    def test_oom_vs_dead_context(self):
        oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        dead = RuntimeError("CUDA driver error: device not ready")
        self.assertTrue(is_cuda_oom(oom))
        self.assertFalse(is_cuda_oom(dead))
        self.assertTrue(is_cuda_context_dead(dead))
        self.assertFalse(is_cuda_context_dead(oom))

        class OutOfMemoryError(RuntimeError):
            pass

        self.assertTrue(is_cuda_oom(OutOfMemoryError("")))


if __name__ == "__main__":
    unittest.main()
