import unittest

from trellis2.utils.bake_limits import (
    RAISED_DECIMATION_TARGET,
    RAISED_TEXTURE_SIZE,
    bake_size_attempts,
    is_cuda_oom,
)


class BakeLimitTests(unittest.TestCase):
    def test_raised_defaults_match_full_export(self):
        self.assertEqual(RAISED_DECIMATION_TARGET, 1_000_000)
        self.assertEqual(RAISED_TEXTURE_SIZE, 4096)

    def test_attempts_step_down_without_growing(self):
        self.assertEqual(
            bake_size_attempts(1_000_000, 4096),
            [(1_000_000, 4096), (500_000, 2048), (100_000, 1024)],
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

    def test_oom_detection_ignores_other_cuda_faults(self):
        self.assertTrue(is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))
        self.assertFalse(is_cuda_oom(RuntimeError("CUDA driver error: device not ready")))

        class OutOfMemoryError(RuntimeError):
            pass

        self.assertTrue(is_cuda_oom(OutOfMemoryError("")))


if __name__ == "__main__":
    unittest.main()
