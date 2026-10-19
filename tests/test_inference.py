"""Checks for inference.py that run without TensorFlow or the trained model.

Run with: pip install pytest && pytest
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference  # noqa: E402

UTM_18N = "EPSG:32618"
TRANSFORM = from_origin(400000, 4000000, 10, 10)


class RedChannelModel:
    """Stands in for the network: predicts each pixel's red value."""

    def predict_on_batch(self, batch):
        return batch[..., :1]


def geotiff(bands, crs=UTM_18N, colorinterp=None):
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0],
                      dtype=bands.dtype, crs=crs, transform=TRANSFORM) as dst:
            dst.write(bands)
            if colorinterp:
                dst.colorinterp = colorinterp
        return mem.read()


def test_calibrate_inverts_the_weighted_loss_bias():
    p = np.linspace(0.01, 0.99, 50)
    r = inference.POS_WEIGHT_RATIO
    raw = r * p / (r * p + 1 - p)
    np.testing.assert_allclose(inference.calibrate(raw), p, rtol=1e-4)
    assert inference.calibrate(0.5) == pytest.approx(1 / (1 + r), rel=1e-4)


@pytest.mark.parametrize("shape", [(100, 120), (256, 256), (300, 517)])
def test_predict_scores_covers_every_pixel_without_seams(shape):
    rgb = np.random.default_rng(0).integers(0, 256, shape + (3,), dtype=np.uint8)
    scores = inference.predict_scores(RedChannelModel(), rgb, batch_size=4)
    assert scores.shape == shape
    np.testing.assert_allclose(scores, rgb[..., 0] / 255.0, atol=1e-5)


def test_read_image_png_has_no_georeference():
    buffer = io.BytesIO()
    Image.new("RGBA", (40, 30), (10, 20, 30, 255)).save(buffer, format="PNG")
    rgb, georef, note = inference.read_image(buffer.getvalue(), "scene.png")
    assert rgb.shape == (30, 40, 3) and rgb.dtype == np.uint8
    assert georef is None and note is None


def test_read_image_geotiff_keeps_location_and_band_order():
    bgr = np.stack([np.full((20, 30), value, np.uint8) for value in (30, 20, 10)])
    data = geotiff(bgr, colorinterp=[ColorInterp.blue, ColorInterp.green, ColorInterp.red])
    rgb, georef, note = inference.read_image(data, "scene.tif")
    assert rgb[0, 0].tolist() == [10, 20, 30]
    assert georef["crs"] == rasterio.crs.CRS.from_string(UTM_18N)
    assert georef["transform"] == TRANSFORM
    assert note is None


def test_read_image_stretches_16_bit_images():
    bands = np.random.default_rng(1).integers(500, 3000, (3, 20, 30)).astype(np.uint16)
    rgb, _, note = inference.read_image(geotiff(bands), "scene.tif")
    assert rgb.dtype == np.uint8 and rgb.max() == 255
    assert "uint16" in note


def test_read_image_rejects_huge_images_before_decoding():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100)).save(buffer, format="PNG")
    with pytest.raises(inference.ImageTooLargeError):
        inference.read_image(buffer.getvalue(), "big.png", max_pixels=10_000)


def test_building_area_needs_a_metric_projection():
    prob = np.full((100, 100), 0.5, np.float32)
    assert inference.building_area_m2(prob, {"crs": rasterio.crs.CRS.from_string(UTM_18N),
                                             "transform": TRANSFORM}) == pytest.approx(500_000)
    assert inference.building_area_m2(prob, {"crs": rasterio.crs.CRS.from_epsg(4326),
                                             "transform": TRANSFORM}) is None
    assert inference.building_area_m2(prob, None) is None


def test_geotiff_output_round_trips():
    mask = np.zeros((20, 30), np.uint8)
    mask[5:10, 5:10] = 255
    georef = {"crs": rasterio.crs.CRS.from_string(UTM_18N), "transform": TRANSFORM}
    with MemoryFile(inference.geotiff_bytes(mask, georef)) as mem, mem.open() as src:
        assert src.crs == georef["crs"] and src.transform == TRANSFORM
        np.testing.assert_array_equal(src.read(1), mask)
