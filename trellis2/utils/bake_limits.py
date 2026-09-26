"""Face-count and texture-size choices for the GLB bake.

The 512³ shape sample is left alone. What looked janky was the export:
100k faces and a 1024 texture.

On a 4 GB card a 4096 texture dies at "Sampling attributes" with
`CUDA driver error: device not ready`. That kills the CUDA context, so a
smaller retry in the same process cannot run. Default there is 1,000,000
faces and a 2048 texture (still a raise from 100k/1024). Full 4096 stays
the large-GPU bake, or an explicit TRELLIS_TEXTURE_SIZE=4096.
"""

RAISED_DECIMATION_TARGET = 1_000_000
LOW_VRAM_TEXTURE_SIZE = 2048
FULL_TEXTURE_SIZE = 4096
# Kept for older imports; 4 GB callers should use default_texture_size().
RAISED_TEXTURE_SIZE = LOW_VRAM_TEXTURE_SIZE

_FALLBACKS = (
    (500_000, 2048),
    (100_000, 1024),
)


def default_texture_size(total_gb: float) -> int:
    """2048 on GPUs under 8 GB; 4096 otherwise."""
    if total_gb > 0 and total_gb < 8:
        return LOW_VRAM_TEXTURE_SIZE
    return FULL_TEXTURE_SIZE


def bake_size_attempts(decimation_target: int, texture_size: int):
    """Largest bake first. Later entries are never bigger than the request."""
    faces = int(decimation_target)
    tex = int(texture_size)
    if faces < 1 or tex < 1:
        raise ValueError(
            f"decimation_target and texture_size must be positive, got {faces}, {tex}"
        )
    attempts = []
    for candidate_faces, candidate_tex in ((faces, tex), *_FALLBACKS):
        choice = (min(int(candidate_faces), faces), min(int(candidate_tex), tex))
        if choice not in attempts:
            attempts.append(choice)
    return attempts


def is_cuda_oom(exc: BaseException) -> bool:
    """True only for an out-of-memory failure, not other CUDA faults."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def is_cuda_context_dead(exc: BaseException) -> bool:
    """True when the driver dropped the context. Retrying in-process cannot work."""
    msg = str(exc).lower()
    return "device not ready" in msg or "cuda driver error" in msg
