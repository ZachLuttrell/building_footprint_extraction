"""Model loading, tiled prediction and output encoding for the Streamlit app."""
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.io import MemoryFile
except ImportError:  # GeoTIFF support is optional; plain images still work
    rasterio = None

PATCH_SIZE = 256
OVERLAP = 64

# Larger images can exhaust the memory of a free Streamlit Cloud instance.
MAX_PIXELS = 25_000_000

MODEL_FILE_ID = "1MB7DOQq6--oIYF6TWdn7kisjXWnPI1E4"
MODEL_PATH = Path(__file__).resolve().parent / "unet_vgg_14.keras"

# Class weights unet_vgg_14 was trained with (weighted binary cross-entropy).
# Building pixels counted ~14x more than background, which pushes raw scores up.
CLASS_WEIGHTS = (0.5355597809300489, 7.530414514976497)
POS_WEIGHT_RATIO = CLASS_WEIGHTS[1] / CLASS_WEIGHTS[0]


# ------------------------------------------------------
# Model
# ------------------------------------------------------
def load_model(path=MODEL_PATH):
    """Download the trained model once, then load it for inference only."""
    import gdown
    import tensorflow as tf

    if not Path(path).exists():
        gdown.download(id=MODEL_FILE_ID, output=str(path), quiet=False)
    # compile=False skips the training loss and metrics, so no custom objects are needed.
    return tf.keras.models.load_model(path, compile=False)


def calibrate(scores):
    """Convert raw model scores into approximate probabilities of a building.

    Under the weighted loss the model learns q = r*p / (r*p + 1 - p) for a true
    probability p, where r is the positive:negative weight ratio, so
    p = q / (q + r*(1 - q)). Treat the result as approximate.
    """
    q = np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
    return q / (q + POS_WEIGHT_RATIO * (1.0 - q) + 1e-12)


# ------------------------------------------------------
# Reading images
# ------------------------------------------------------
def _stretch_to_uint8(bands):
    """Stretch each band's 2nd-98th percentile to 0-255, ignoring zero (no-data) pixels."""
    out = np.zeros(bands.shape, dtype=np.uint8)
    for i, band in enumerate(bands):
        valid = band[band > 0]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, (2, 98))
        scaled = (band.astype(np.float32) - lo) / max(hi - lo, 1e-6)
        out[i] = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
    return out


class ImageTooLargeError(ValueError):
    pass


def _check_size(width, height, max_pixels):
    if max_pixels and width * height > max_pixels:
        raise ImageTooLargeError(
            f"{width:,} x {height:,} pixels is larger than this app can process "
            f"({max_pixels:,} pixels). Please crop the image and try again.")


def _rgb_band_indexes(src):
    """1-based indexes of the red, green and blue bands, defaulting to the first three."""
    interp = list(src.colorinterp)
    wanted = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    if all(colour in interp for colour in wanted):
        return [interp.index(colour) + 1 for colour in wanted]
    return [1, 2, 3]


def read_image(data, filename, max_pixels=MAX_PIXELS):
    """Return (rgb, georef, note) for an uploaded file.

    rgb is an (H, W, 3) uint8 array. georef holds the CRS and transform when the
    file is a georeferenced GeoTIFF, otherwise None. note explains any conversion.
    Raises ImageTooLargeError before decoding images over max_pixels.
    """
    ext = os.path.splitext(filename)[1].lower()
    if rasterio is not None and ext in (".tif", ".tiff"):
        try:
            with MemoryFile(data) as mem, mem.open() as src:
                if src.count >= 3:
                    _check_size(src.width, src.height, max_pixels)
                    bands = src.read(_rgb_band_indexes(src))
                    georef = {"crs": src.crs, "transform": src.transform} if src.crs else None
                    note = None
                    if bands.dtype != np.uint8:
                        bands = _stretch_to_uint8(bands)
                        note = (f"{src.dtypes[0]} image stretched to 8-bit for the model; "
                                "results depend on this rendering.")
                    return np.moveaxis(bands, 0, -1), georef, note
        except ImageTooLargeError:
            raise
        except Exception:
            pass  # not something rasterio can read; fall back to Pillow
    image = Image.open(io.BytesIO(data))
    _check_size(image.width, image.height, max_pixels)
    return np.array(image.convert("RGB")), None, None


# ------------------------------------------------------
# Tiled prediction
# ------------------------------------------------------
def _positions(length, size, step):
    """Start offsets that cover [0, length) with windows of `size`."""
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, step))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def _blend_window(size):
    """Weights that favour each tile's centre, so overlapping tiles blend without seams."""
    ramp = np.sin(np.pi * (np.arange(size) + 0.5) / size)
    return (0.05 + 0.95 * np.outer(ramp, ramp)).astype(np.float32)


def predict_scores(model, rgb, batch_size=16, progress=None):
    """Run the model over overlapping tiles and blend them into one (H, W) score map."""
    image = rgb
    height, width = image.shape[:2]
    pad_h, pad_w = max(0, PATCH_SIZE - height), max(0, PATCH_SIZE - width)
    if pad_h or pad_w:  # images smaller than one tile
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    padded_h, padded_w = image.shape[:2]

    step = PATCH_SIZE - OVERLAP
    coords = [(y, x) for y in _positions(padded_h, PATCH_SIZE, step)
              for x in _positions(padded_w, PATCH_SIZE, step)]
    window = _blend_window(PATCH_SIZE)
    total = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight = np.zeros((padded_h, padded_w), dtype=np.float32)

    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        batch = np.stack([image[y:y + PATCH_SIZE, x:x + PATCH_SIZE] for y, x in chunk])
        # The model was trained on RGB scaled to [0, 1]; convert one batch at a time to save memory.
        preds = np.asarray(model.predict_on_batch(batch.astype(np.float32) / 255.0))[..., 0]
        for (y, x), pred in zip(chunk, preds, strict=True):
            total[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += pred * window
            weight[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += window
        if progress:
            progress(min(1.0, (start + len(chunk)) / len(coords)))

    return (total / np.maximum(weight, 1e-6))[:height, :width]


# ------------------------------------------------------
# Outputs
# ------------------------------------------------------
def overlay(rgb, mask, color=(255, 140, 0), alpha=0.55):
    """Tint building pixels on top of the original image."""
    out = rgb.astype(np.float32)
    hit = mask.astype(bool)
    out[hit] = out[hit] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return out.astype(np.uint8)


def building_area_m2(prob, georef):
    """Rough area covered by buildings: the sum of building probabilities times the pixel area.

    Only available when the image is georeferenced in a projected CRS measured in metres.
    """
    if not georef or georef["crs"] is None or not georef["crs"].is_projected:
        return None
    if (georef["crs"].linear_units or "").lower() not in ("metre", "meter", "m"):
        return None
    t = georef["transform"]
    return float(prob.sum()) * abs(t.a * t.e - t.b * t.d)


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def geotiff_bytes(array, georef):
    """Single-band uint8 GeoTIFF that keeps the input's CRS and transform."""
    profile = {"driver": "GTiff", "height": array.shape[0], "width": array.shape[1], "count": 1,
               "dtype": "uint8", "crs": georef["crs"], "transform": georef["transform"],
               "compress": "deflate"}
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(array, 1)
        return mem.read()
