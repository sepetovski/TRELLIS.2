"""Load OpenEXR HDRIs used as environment maps.

``opencv-python`` 5.x wheels are built without OpenEXR. ``cv2.imread`` then
returns an empty image and ``cv2.cvtColor`` raises ``!_src.empty()``. This
helper keeps the OpenCV path when the codec is present and otherwise reads
the file with the ``OpenEXR`` package.
"""
import os
from typing import Optional

import numpy as np

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2


def read_exr(path: str) -> np.ndarray:
    """Read an OpenEXR image as float32 RGB with shape ``(H, W, 3)``.

    Paths may be relative to the current working directory or to the
    repository root.
    """
    resolved = _resolve_path(path)
    if resolved is None:
        raise FileNotFoundError(
            f"HDRI not found: {path}. Run from the TRELLIS.2 repository root, "
            f"or pass a path that exists."
        )

    image = _read_exr_cv2(resolved)
    if image is None:
        image = _read_exr_openexr(resolved)
    if image is None:
        raise RuntimeError(_missing_codec_message(resolved))
    return image


def _resolve_path(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    candidate = os.path.join(repo_root, path)
    if os.path.isfile(candidate):
        return candidate
    return None


def _read_exr_cv2(path: str) -> Optional[np.ndarray]:
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.size == 0:
        return None
    if bgr.ndim == 2:
        bgr = np.repeat(bgr[:, :, None], 3, axis=2)
    elif bgr.shape[-1] == 4:
        bgr = bgr[:, :, :3]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.float32)


def _read_exr_openexr(path: str) -> Optional[np.ndarray]:
    try:
        import OpenEXR
    except ImportError:
        return None

    channels = OpenEXR.File(path).channels()
    for key in ('RGB', 'RGBA'):
        channel = channels.get(key)
        if channel is None:
            continue
        pixels = np.asarray(channel.pixels)
        if pixels.ndim == 3 and pixels.shape[-1] >= 3:
            return np.ascontiguousarray(pixels[..., :3], dtype=np.float32)

    if all(key in channels for key in ('R', 'G', 'B')):
        planes = [np.asarray(channels[key].pixels, dtype=np.float32) for key in ('R', 'G', 'B')]
        return np.ascontiguousarray(np.stack(planes, axis=-1))
    return None


def _missing_codec_message(path: str) -> str:
    version = getattr(cv2, '__version__', 'unknown')
    return (
        f"Could not read OpenEXR HDRI '{path}'.\n"
        f"OpenCV {version} returned an empty image. opencv-python 5.x wheels ship "
        f"without the OpenEXR codec, so assets/hdri/*.exr never load and the demo "
        f"stops in cv2.cvtColor.\n"
        f"Install one of these, then run the app again:\n"
        f'    pip install "opencv-python-headless>=4.10,<5"\n'
        f"    pip install OpenEXR"
    )
