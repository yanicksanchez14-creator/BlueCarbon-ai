"""End-to-end: chips -> train -> checkpoint -> predict -> report -> demo bundle, on synthetic data."""

import json

import numpy as np
import rasterio

from bluecarbon.config import load_config
from bluecarbon.demo import export_scene
from bluecarbon.model import load_checkpoint
from bluecarbon.predict import predict_raster
from bluecarbon.report import scene_report
from bluecarbon.tiling import make_chips, read_index, write_index
from bluecarbon.train import train

from .conftest import make_scene


def test_end_to_end(tmp_path):
    cfg = load_config(None, {
        "model": {"encoder": "resnet18", "encoder_weights": None},
        "train": {"epochs": 3, "batch_size": 4, "num_workers": 0, "lr": 3e-3, "device": "cpu", "amp": False},
        "carbon": {"monte_carlo": 200},
    })
    recs = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        d = tmp_path / f"site{i}"
        d.mkdir()
        make_scene(d / "image.tif", d / "label.tif", size=256, seed=i)
        recs += make_chips(d / "image.tif", d / "label.tif", tmp_path / "chips", f"site{i}", size=128, stride=128,
                           force_split=split)
    write_index(recs, tmp_path / "chips" / "index.json")
    recs = read_index(tmp_path / "chips" / "index.json")

    res = train(cfg, recs, tmp_path / "model", log=lambda *_: None)
    assert (tmp_path / "model" / "model.pt").exists()
    assert "test" in res and res["test"]["mIoU"] is not None
    assert json.loads((tmp_path / "model" / "metrics.json").read_text())["best_epoch"] >= 1

    model, norm, ck = load_checkpoint(tmp_path / "model" / "model.pt")
    assert ck["metrics"]["test_confusion"] is not None

    scene = tmp_path / "scene"
    scene.mkdir()
    make_scene(scene / "image.tif", scene / "label.tif", size=200, seed=9)  # non-multiple of tile size
    predict_raster(model, norm, scene / "image.tif", scene / "pred.tif", tile=128, overlap=32, tta=True)
    with rasterio.open(scene / "pred.tif") as ds:
        cls, conf = ds.read(1), ds.read(2)
        assert cls.shape == (200, 200) and ds.crs.to_epsg() == 32611
    assert cls.max() < 6 and conf.max() <= 100

    rep = scene_report(scene / "pred.tif", cfg.carbon, ck["metrics"]["test_confusion"])
    assert np.isclose(sum(rep["areas_ha"].values()), 200 * 200 * 0.01)
    assert "error_adjusted_areas_ha" in rep

    out = export_scene(scene, tmp_path / "demo" / "syn", "Synthetic", cfg, tmp_path / "model" / "model.pt")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["kind"] == "single" and (out / "rgb.png").exists() and (out / "classes.png").exists()
    (s, w), (n, e) = meta["bounds"]
    assert s < n and w < e


def test_spectral_model(tmp_path):
    from bluecarbon.predictors import load_predictor, model_card
    from bluecarbon.spectral import train_spectral

    recs = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        d = tmp_path / f"site{i}"
        d.mkdir()
        make_scene(d / "image.tif", d / "label.tif", size=256, seed=i)
        recs += make_chips(d / "image.tif", d / "label.tif", tmp_path / "chips", f"site{i}", size=128, stride=128,
                           force_split=split)
    res = train_spectral(recs, tmp_path / "m", per_class_per_chip=200, n_estimators=40, log=lambda *_: None)
    assert res["test"]["mIoU"] > 0.8  # synthetic classes are spectrally separable
    pr = load_predictor(tmp_path / "m" / "spectral.json")
    with rasterio.open(tmp_path / "site3" / "image.tif") as s:
        cls, conf = pr.predict(s.read())
    assert cls.shape == (256, 256) and conf.max() <= 1.0
    assert model_card(tmp_path / "m" / "spectral.json")["kind"] == "spectral-lgbm"
