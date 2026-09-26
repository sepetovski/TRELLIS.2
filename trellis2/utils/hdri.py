"""Load a lat-long HDRI for the preview video.

`assets/hdri/*.exr` are DWAB-compressed. OpenCV 5's imread returns an empty
image for that compression, and `cvtColor` then fails with `!_src.empty()`.
The GLB does not use this file. Only the turntable video does.
"""
import numpy as np


def studio_latlong(height: int = 256, width: int = 512) -> np.ndarray:
    """Small HDR lat-long: cool sky, dark ground, one sun. v=0 is up."""
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    sky = np.array([0.55, 0.62, 0.75], dtype=np.float32)
    ground = np.array([0.12, 0.11, 0.10], dtype=np.float32)
    image = ground * y[..., None] + sky * (1.0 - y[..., None])
    du = (x - 0.5 + 0.5) % 1.0 - 0.5
    dv = y - 0.28
    sun = np.exp(-(du * du / 0.004 + dv * dv / 0.008)).astype(np.float32)
    image = image + sun[..., None] * np.array([8.0, 7.2, 6.0], dtype=np.float32)
    return np.ascontiguousarray(image)


def _usable(image) -> bool:
    if image is None or not hasattr(image, "ndim"):
        return False
    if image.ndim < 2 or image.size == 0:
        return False
    return min(image.shape[0], image.shape[1]) > 0


def _to_rgb_float(image, bgr: bool) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel HDRI, got shape {image.shape}")
    image = np.ascontiguousarray(image.astype(np.float32, copy=False))
    if bgr:
        image = image[..., ::-1]
    return np.ascontiguousarray(image)


def _read_cv2(path: str):
    try:
        import cv2
    except ImportError:
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if not _usable(image):
        return None
    # OpenCV is BGR. Integer files are 0-255; EXR is already linear float.
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    return _to_rgb_float(image, bgr=True)


def _read_imageio(path: str):
    try:
        import imageio.v3 as iio
    except ImportError:
        return None
    try:
        image = iio.imread(path)
    except Exception:
        return None
    if not _usable(image):
        return None
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    return _to_rgb_float(image, bgr=False)


def _read_openexr(path: str):
    try:
        import Imath
        import OpenEXR
    except ImportError:
        return None
    try:
        exr = OpenEXR.InputFile(path)
        header = exr.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        pixel = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = [np.frombuffer(exr.channel(name, pixel), dtype=np.float32) for name in "RGB"]
        image = np.stack([channel.reshape(height, width) for channel in channels], axis=-1)
    except Exception:
        return None
    if not _usable(image):
        return None
    return _to_rgb_float(image, bgr=False)


def load_latlong_rgb(path: str) -> np.ndarray:
    """HxWx3 float32 RGB. Uses a built-in studio light if the EXR cannot be decoded."""
    for reader in (_read_cv2, _read_imageio, _read_openexr):
        image = reader(path)
        if image is not None:
            return image
    print(
        f"Could not decode HDRI {path}. OpenCV cannot read this DWAB EXR "
        "(cvtColor then sees an empty image). Preview video will use a built-in studio light. "
        "The GLB does not need this file."
    )
    return studio_latlong()


def preview_settings(total_gb: float):
    """(resolution, frames, ssaa). Smaller on 4 GB so the turntable does not TDR."""
    if total_gb and total_gb < 8:
        return 512, 60, 1
    return 1024, 120, 2
