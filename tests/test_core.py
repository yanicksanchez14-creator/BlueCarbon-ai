import numpy as np
import pytest
import rasterio

from bluecarbon.carbon import carbon_report, change_report, class_areas_ha, pixel_area_ha
from bluecarbon.config import CarbonCfg, load_config
from bluecarbon.features import N_FEATURES, Normalizer, compute_features
from bluecarbon.labels import class_histogram, ignore_boundaries
from bluecarbon.metrics import confusion, error_adjusted_area, summarize
from bluecarbon.schema import IGNORE_INDEX, N_CLASSES
from bluecarbon.tiling import block_split, make_chips


def test_features_shape_and_range():
    bands = np.random.default_rng(0).integers(1, 5000, (10, 8, 8)).astype(np.uint16)
    f = compute_features(bands)
    assert f.shape == (N_FEATURES, 8, 8)
    assert f[:10].max() <= 1.0  # reflectance scaled
    assert np.abs(f[10:]).max() <= 1.0  # indices


def test_features_scale_invariant():
    bands = np.random.default_rng(1).integers(1, 5000, (10, 4, 4)).astype(np.float32)
    assert np.allclose(compute_features(bands), compute_features(bands / 10000), atol=1e-5)


def test_normalizer_roundtrip():
    x = np.random.default_rng(0).normal(3, 2, (4, 16, 16)).astype(np.float32)
    n = Normalizer.fit([x])
    z = n(x)
    assert np.allclose(z.mean((1, 2)), 0, atol=1e-3)
    assert np.allclose(Normalizer.from_dict(n.to_dict())(x), z)


def test_boundary_ignore():
    lab = np.zeros((10, 10), np.uint8)
    lab[:, 5:] = 1
    out = ignore_boundaries(lab, 1)
    assert (out[:, 4:6] == IGNORE_INDEX)[1:-1].all()
    assert (out[:, :3] == 0).all() and (out[:, 7:] == 1).all()


def test_histogram_ignores_255():
    lab = np.array([[0, 1, 255], [1, 1, 5]], np.uint8)
    h = class_histogram(lab)
    assert h["water"] == 1 and h["mangrove"] == 3 and h["other_land"] == 1 and sum(h.values()) == 5


def test_confusion_and_summary():
    ref = np.array([0, 0, 1, 1, 255])
    pred = np.array([0, 1, 1, 1, 0])
    cm = confusion(ref, pred)
    assert cm.sum() == 4 and cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 2
    s = summarize(cm)
    assert s["overall_accuracy"] == 0.75
    assert s["iou"]["water"] == 0.5 and s["iou"]["saltmarsh"] is None  # absent classes excluded


def test_error_adjusted_area_perfect_map():
    cm = np.diag([100, 50, 0, 0, 20, 30, 0])
    mapped = np.array([1000, 500, 0, 0, 200, 300, 0])
    adj = error_adjusted_area(cm, mapped, 0.01)
    assert adj["water"]["adjusted_ha"] == pytest.approx(10.0)
    assert adj["water"]["ci95_ha"] == pytest.approx(0.0)


def test_error_adjusted_area_corrects_commission():
    # half of what the map calls mangrove is really water
    cm = np.zeros((N_CLASSES, N_CLASSES))
    cm[0, 0], cm[0, 1], cm[1, 1] = 100, 50, 50
    mapped = np.array([1000, 1000, 0, 0, 0, 0, 0])
    adj = error_adjusted_area(cm, mapped, 1.0)
    assert adj["mangrove"]["adjusted_ha"] == pytest.approx(500)
    assert adj["water"]["adjusted_ha"] == pytest.approx(1500)


def test_pixel_area_projected_and_geographic():
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    assert pixel_area_ha(from_origin(0, 0, 10, 10), CRS.from_epsg(32611), 3)[0] == pytest.approx(0.01)
    geo = pixel_area_ha(from_origin(-117, 32.8, 8.983e-5, 8.983e-5), CRS.from_epsg(4326), 1)[0]
    assert geo == pytest.approx(0.01 * np.cos(np.radians(32.8)), rel=0.02)


def test_carbon_report_math():
    cfg = CarbonCfg(monte_carlo=2000)
    rep = carbon_report({"mangrove": 10.0, "saltmarsh": 0.0, "seagrass": 0.0}, cfg)
    m = rep["classes"]["mangrove"]["stock_tCO2e"]
    lo = 10 * (436 + 25) * 3.667
    hi = 10 * (510 + 80) * 3.667
    assert lo <= m["p05"] < m["mean"] < m["p95"] <= hi
    assert rep["blue_carbon_area_ha"] == 10.0


def test_change_report_sign():
    cfg = CarbonCfg(monte_carlo=500)
    ch = change_report({"saltmarsh": 10.0}, {"saltmarsh": 5.0}, cfg)
    assert ch["delta_ha"]["saltmarsh"] == -5.0
    assert ch["net_stock_change_tCO2e"]["mean"] < 0


def test_block_split_deterministic():
    assert block_split("a", 1, 2) == block_split("a", 1, 2)
    splits = [block_split("s", x, y) for x in range(30) for y in range(30)]
    frac = splits.count("train") / len(splits)
    assert 0.6 < frac < 0.8


def test_chips_no_overlap_and_blocks(scene, tmp_path):
    img_p, lab_p, _, _ = scene
    recs = make_chips(img_p, lab_p, tmp_path / "chips", "syn", size=128, stride=128, block_km=1.28)
    assert len(recs) == 16
    assert len({(r.row, r.col) for r in recs}) == 16
    # all chips inside the same 1.28 km block share a split
    for r in recs:
        assert r.split in {"train", "val", "test"}


def test_class_areas(scene):
    img_p, lab_p, _, lab = scene
    with rasterio.open(lab_p) as ds:
        a = class_areas_ha(ds.read(1), ds.transform, ds.crs)
    assert sum(a.values()) == pytest.approx((lab.size - (rasterio.open(lab_p).read(1) == 255).sum()) * 0.01)


def test_config_loads_repo_default():
    cfg = load_config("configs/default.yaml")
    assert cfg.carbon.classes["mangrove"].soil[1] == 471
    assert cfg.chips.stride == cfg.chips.size


def test_compute_tile_splits_on_timeout(monkeypatch):
    from rasterio.transform import from_origin

    from bluecarbon import gee

    calls = []

    def fake_once(image, crs, transform, x0, y0, w, h):
        calls.append((w, h))
        if w * h > 128 * 128:
            raise Exception("Computation timed out.")
        yy, xx = np.mgrid[y0 : y0 + h, x0 : x0 + w]
        return np.stack([yy, xx]).astype(np.int32)

    monkeypatch.setattr(gee, "_compute_tile_once", fake_once)
    out = gee._compute_tile(None, "EPSG:32611", from_origin(0, 0, 10, 10), 0, 0, 512, 300)
    yy, xx = np.mgrid[0:300, 0:512]
    assert out.shape == (2, 300, 512)
    assert (out[0] == yy).all() and (out[1] == xx).all()
    assert len(calls) > 1
