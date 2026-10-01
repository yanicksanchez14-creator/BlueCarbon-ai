"""Regional seagrass survey maps, fetched from public web services and burned into the labels.

Global seagrass labels (Allen Coral Atlas) only cover tropical reefs, so meadows in murky or temperate
water (Florida's Gulf coast, Moreton Bay) were unlabelled. Official survey polygons fix that:

  arcgis  an ArcGIS REST feature layer, queried by bounding box (e.g. FWC "Seagrass Statewide")
  wfs     an OGC WFS layer, downloaded as GeoJSON (e.g. Seamap Australia, Moreton Bay 2015)

Polygons are burned as seagrass only over pixels that are open water or unlabelled water, never over
land, mangrove or salt marsh. A survey that cannot be reached is skipped with a message, so a network
hiccup never stops a training run.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .schema import IGNORE_INDEX, KEY_TO_ID

UA = {"User-Agent": "BlueCarbon-AI (https://github.com/yanicksanchez14-creator/bluecarbon-ai)"}


def _get_json(url: str, params: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _intersects(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def fetch_arcgis(layer_url: str, bbox: list[float], where: str = "1=1", page: int = 1000) -> list[dict]:
    """GeoJSON features (EPSG:4326) of an ArcGIS feature layer that intersect bbox, all pages."""
    feats, offset = [], 0
    while True:
        d = _get_json(layer_url.rstrip("/") + "/query", {
            "where": where, "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": 4326,
            "returnGeometry": "true", "maxAllowableOffset": 0.00003, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": page,
        })
        if "error" in d:
            raise RuntimeError(str(d["error"])[:200])
        batch = d.get("features", [])
        feats += batch
        more = d.get("exceededTransferLimit") or d.get("properties", {}).get("exceededTransferLimit")
        if not batch or (len(batch) < page and not more):
            return feats
        offset += len(batch)


def fetch_wfs(url: str, layer: str) -> list[dict]:
    d = _get_json(url, {"service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": layer,
                        "outputFormat": "application/json", "srsName": "EPSG:4326"}, timeout=300)
    return d.get("features", [])


def seagrass_features(feats: list[dict]) -> tuple[list[dict], str]:
    """Keep seagrass polygons. If some text field names habitats, keep the rows that say seagrass
    (and not 'no seagrass' / 'absent'); otherwise the whole layer is seagrass."""
    if not feats:
        return [], "no features"
    props = [f.get("properties") or {} for f in feats]
    for key in sorted({k for p in props for k in p}):
        vals = [str(p.get(key, "")).lower() for p in props]
        if any("seagrass" in v for v in vals) and not all("seagrass" in v for v in vals):
            keep = [f for f, v in zip(feats, vals, strict=True)
                    if "seagrass" in v and not any(n in v for n in ("no seagrass", "absent", "non-seagrass"))]
            return keep, f"kept {len(keep)}/{len(feats)} features where '{key}' mentions seagrass"
    return feats, f"all {len(feats)} features treated as seagrass"


def burn_seagrass(label: np.ndarray, image: np.ndarray, transform, crs, feats: list[dict]) -> tuple[np.ndarray, int]:
    """Set seagrass inside the polygons, over water / unlabelled water pixels only."""
    from rasterio import features as rfeatures
    from rasterio.warp import transform_geom

    from .features import compute_features

    geoms = [transform_geom("EPSG:4326", crs, f["geometry"]) for f in feats if f.get("geometry")]
    if not geoms:
        return label, 0
    inside = rfeatures.rasterize([(g, 1) for g in geoms], out_shape=label.shape, transform=transform,
                                 fill=0, dtype="uint8").astype(bool)
    mndwi = compute_features(image)[12]  # MNDWI > 0: the pixel looks like water in the image
    wet = (label == KEY_TO_ID["water"]) | ((label == IGNORE_INDEX) & (mndwi > 0))
    m = inside & wet
    out = label.copy()
    out[m] = KEY_TO_ID["seagrass"]
    return out, int(m.sum())


def apply_surveys(label_path: str | Path, image_path: str | Path, bbox: list[float], surveys: list[dict],
                  report: dict | None = None, log=print) -> int:
    """Burn every survey that overlaps bbox into label.tif. Returns the number of seagrass pixels added."""
    import rasterio

    todo = [s for s in surveys if _intersects(bbox, s.get("extent", [-180, -90, 180, 90]))]
    if not todo:
        return 0
    with rasterio.open(image_path) as src:
        image = src.read()
    total = 0
    with rasterio.open(label_path, "r+") as ds:
        lab = ds.read(1)
        for s in todo:
            try:
                if s["kind"] == "arcgis":
                    feats = fetch_arcgis(s["url"], bbox, s.get("where", "1=1"))
                else:
                    feats = [f for f in fetch_wfs(s["url"], s["layer"]) if f.get("geometry")]
                feats, how = seagrass_features(feats)
                lab, n = burn_seagrass(lab, image, ds.transform, ds.crs, feats)
                total += n
                log(f"  seagrass survey {s['name']}: {how}; {n * 0.01:,.0f} ha labelled as seagrass")
                if report is not None and n:
                    report.setdefault("label_sources", []).append(s["name"])
            except Exception as e:  # never let an unreachable survey stop the run
                log(f"  seagrass survey {s['name']} skipped ({str(e)[:120]})")
        ds.write(lab, 1)
    return total
