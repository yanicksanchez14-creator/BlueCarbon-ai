"""Command-line interface: `bluecarbon --help`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from .config import Config, load_config

app = typer.Typer(add_completion=False, help="BlueCarbon-AI: map blue carbon ecosystems from Sentinel-2.")

CfgOpt = typer.Option("configs/default.yaml", "--config", "-c", help="Pipeline config YAML")
SitesOpt = typer.Option("configs/sites.yaml", "--sites", help="Sites YAML")


def _cfg(path: str) -> Config:
    return load_config(path if Path(path).exists() else None)


def _sites(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


@app.command()
def fetch(config: str = CfgOpt, sites: str = SitesOpt, only: list[str] = typer.Option(None, help="Site names"),
          service_account: Path = typer.Option(None, help="Service account key JSON")):
    """Download Sentinel-2 composites + fused reference labels for every site (Earth Engine)."""
    from . import gee
    from .features import S2_BANDS
    from .labels import postprocess_label_file

    cfg, s = _cfg(config), _sites(sites)
    gee.init(cfg.project, service_account.read_text() if service_account else None)
    year = s["year"]
    for site in s["sites"]:
        if only and site["name"] not in only:
            continue
        d = cfg.work / "sites" / site["name"]
        region = gee.bbox_geometry(site["bbox"])
        img = gee.s2_composite(region, f"{year}-01-01", f"{year + 1}-01-01", cfg)
        typer.echo(f"[{site['name']}] imagery ...")
        gee.download(img, site["bbox"], d / "image.tif", cfg, "uint16", 0, S2_BANDS)
        typer.echo(f"[{site['name']}] labels ...")
        gee.download(gee.reference_labels(region, cfg), site["bbox"], d / "label.tif", cfg, "uint8", 255, ["label"])
        hist = postprocess_label_file(d / "label.tif", cfg.labels.boundary_ignore_px, cfg.labels.overrides)
        (d / "meta.json").write_text(json.dumps({**site, "year": year, "label_px": hist}, indent=2))
        typer.echo(f"[{site['name']}] label pixels: {hist}")


@app.command()
def chips(config: str = CfgOpt, sites: str = SitesOpt):
    """Tile site rasters into chips with a spatial-block split (test sites fully held out)."""
    from .tiling import make_chips, write_index

    cfg, s = _cfg(config), _sites(sites)
    c = cfg.chips
    out = cfg.work / "chips"
    recs = []
    for site in s["sites"]:
        d = cfg.work / "sites" / site["name"]
        if not (d / "image.tif").exists():
            typer.echo(f"skip {site['name']} (not fetched)")
            continue
        r = make_chips(d / "image.tif", d / "label.tif", out, site["name"], c.size, c.stride, c.min_labeled_frac,
                       c.block_km, c.split, c.seed, "test" if site.get("role") == "test" else None)
        typer.echo(f"{site['name']}: {len(r)} chips")
        recs += r
    write_index(recs, out / "index.json")
    counts = {k: sum(r.split == k for r in recs) for k in ("train", "val", "test")}
    typer.echo(f"total {len(recs)} chips {counts}")


@app.command()
def train(config: str = CfgOpt, epochs: int = typer.Option(None), device: str = typer.Option(None)):
    """Train the segmentation model on the chip index."""
    from .tiling import read_index
    from .train import train as run

    cfg = _cfg(config)
    if epochs:
        cfg.train.epochs = epochs
    if device:
        cfg.train.device = device
    res = run(cfg, read_index(cfg.work / "chips" / "index.json"), cfg.work / "model")
    typer.echo(json.dumps({k: res[k] for k in ("best_epoch", "val", "test") if k in res}, indent=2))


@app.command()
def predict(image: Path, model: Path = typer.Option(..., "--model", "-m"), out: Path = typer.Option(None),
            config: str = CfgOpt):
    """Predict a habitat map (GeoTIFF: band 1 class, band 2 confidence %)."""
    from .model import load_checkpoint, resolve_device
    from .predict import predict_raster

    cfg = _cfg(config)
    net, norm, _ = load_checkpoint(model, resolve_device(cfg.train.device))
    out = out or image.with_name(image.stem + "_pred.tif")
    p = cfg.predict
    predict_raster(net, norm, image, out, p.tile, p.overlap, p.tta)
    typer.echo(f"wrote {out}")


@app.command()
def report(prediction: Path, model: Path = typer.Option(None, "--model", "-m",
                                                        help="Adds error-adjusted areas from test confusion"),
           config: str = CfgOpt, out: Path = typer.Option(None)):
    """Area + carbon stock / sequestration report for a prediction raster."""
    import torch

    from .report import scene_report, to_markdown, write_report

    cfg = _cfg(config)
    cm = None
    if model:
        ck = torch.load(model, map_location="cpu", weights_only=False)
        cm = ck.get("metrics", {}).get("test_confusion")
    rep = scene_report(prediction, cfg.carbon, cm)
    write_report(rep, out or prediction.parent, prediction.stem + "_report")
    typer.echo(to_markdown(rep))


@app.command()
def scene(bbox: list[float] = typer.Option(..., help="lon_min lat_min lon_max lat_max"),
          start: str = typer.Option(...), end: str = typer.Option(...),
          model: Path = typer.Option(..., "--model", "-m"), name: str = "scene", config: str = CfgOpt,
          service_account: Path = typer.Option(None)):
    """Fetch any area + date range, map it and report carbon (Earth Engine)."""
    from . import gee
    from .features import S2_BANDS

    cfg = _cfg(config)
    gee.init(cfg.project, service_account.read_text() if service_account else None)
    d = cfg.work / "scenes" / name
    img = gee.s2_composite(gee.bbox_geometry(bbox), start, end, cfg)
    gee.download(img, bbox, d / "image.tif", cfg, "uint16", 0, S2_BANDS)
    predict(d / "image.tif", model, d / "pred.tif", config)
    report(d / "pred.tif", model, config, d)


@app.command()
def case_study(model: Path = typer.Option(..., "--model", "-m"), config: str = CfgOpt, sites: str = SitesOpt,
               service_account: Path = typer.Option(None)):
    """Before/after change analysis for the `case_study` block in sites.yaml."""
    from . import gee
    from .features import S2_BANDS
    from .report import change_scene_report, write_report

    cfg, s = _cfg(config), _sites(sites)
    cs = s["case_study"]
    gee.init(cfg.project, service_account.read_text() if service_account else None)
    d = cfg.work / "scenes" / cs["name"]
    for key, (a, b) in cs["periods"].items():
        img = gee.s2_composite(gee.bbox_geometry(cs["bbox"]), a, b, cfg)
        gee.download(img, cs["bbox"], d / f"{key}_image.tif", cfg, "uint16", 0, S2_BANDS)
        predict(d / f"{key}_image.tif", model, d / f"{key}_pred.tif", config)
    rep = change_scene_report(d / "t0_pred.tif", d / "t1_pred.tif", cfg.carbon)
    write_report(rep, d, "change_report")
    typer.echo(json.dumps(rep["change"], indent=2))


@app.command()
def export_demo(scene_dir: Path, name: str = typer.Option(...), title: str = typer.Option(...),
                out: Path = typer.Option(Path("demo_data")), model: Path = typer.Option(None, "--model", "-m"),
                config: str = CfgOpt, description: str = ""):
    """Package a predicted scene (image.tif + pred.tif, or t0_/t1_ pairs) for the Streamlit demo."""
    from .demo import export_scene

    p = export_scene(scene_dir, out / name, title, _cfg(config), model, description)
    typer.echo(f"wrote {p}")


if __name__ == "__main__":
    app()
