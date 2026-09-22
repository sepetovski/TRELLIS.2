"""Face-count and texture-size choices for the GLB bake.

The 512³ shape sample is left alone. What looked janky was the export:
100k faces and a 1024 texture. These defaults match the full example's
bake (1,000,000 / 4096). Remesh stays off — it rebuilds the shape, which
is the part that already looks right.

If the large bake does not fit in VRAM, callers walk `bake_size_attempts`
and retry a smaller one.
"""

# Same numbers as example.py on a large GPU.
RAISED_DECIMATION_TARGET = 1_000_000
RAISED_TEXTURE_SIZE = 4096

# Tried only when the requested bake is larger and runs out of VRAM.
_FALLBACKS = (
    (500_000, 2048),
    (100_000, 1024),
)


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
