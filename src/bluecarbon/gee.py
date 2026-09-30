"""Google Earth Engine: Sentinel-2 composites, fused reference labels, and tiled download.

Downloads use `ee.data.computePixels` in fixed-size tiles written straight into a GeoTIFF
in the local UTM zone, so any area size works (the v1 app used getDownloadURL, which fails
above ~32 MB, and exported in EPSG:4326 which broke area calculations).
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .config import Config
from .features import S2_BANDS
from .schema import IGNORE_INDEX, KEY_TO_ID

try:  # optional dependency
    import ee
except ImportError:  # pragma: no cover
    ee = None


def _require_ee():
    if ee is None:
        raise ImportError("earthengine-api is not installed: pip install 'bluecarbon[gee]'")


def init(project: str, service_account_json: str | None = None) -> None:
    """Initialize Earth Engine with user credentials or a service-account key (JSON text)."""
    _require_ee()
    if service_account_json:
        import json

        info = json.loads(service_account_json)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=service_account_json)
        ee.Initialize(creds, project=project or info.get("project_id"))
    else:
        ee.Initialize(project=project)


# --------------------------------------------------------------------------- geometry
def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def utm_grid(bbox: list[float], scale: float) -> tuple[str, rasterio.Affine, int, int]:
    """Snap a lon/lat bbox to a UTM pixel grid. Returns (crs, transform, width, height)."""
    from rasterio.warp import transform_bounds

    lon0, lat0, lon1, lat1 = bbox
    epsg = utm_epsg((lon0 + lon1) / 2, (lat0 + lat1) / 2)
    crs = f"EPSG:{epsg}"
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", crs, lon0, lat0, lon1, lat1)
    x0 = math.floor(x0 / scale) * scale
    y1 = math.ceil(y1 / scale) * scale
    w = int(math.ceil((x1 - x0) / scale))
    h = int(math.ceil((y1 - y0) / scale))
    return crs, from_origin(x0, y1, scale, scale), w, h


# --------------------------------------------------------------------------- imagery
def s2_composite(region, start: str, end: str, cfg: Config):
    """Cloud Score+ masked median composite of S2 L2A bands (uint16 reflectance x1e4)."""
    _require_ee()
    ic = cfg.imagery
    s2 = ee.ImageCollection(ic.collection).filterBounds(region).filterDate(start, end)
    cs = ee.ImageCollection(ic.cloud_score_collection)
    linked = s2.linkCollection(cs, [ic.cloud_score_band])
    masked = linked.map(lambda im: im.updateMask(im.select(ic.cloud_score_band).gte(ic.clear_threshold)))
    return masked.select(S2_BANDS).median().toUint16().set("n_images", s2.size())


def image_count(region, start: str, end: str, cfg: Config) -> int:
    _require_ee()
    return (
        ee.ImageCollection(cfg.imagery.collection).filterBounds(region).filterDate(start, end).size().getInfo()
    )


# --------------------------------------------------------------------------- labels
def _asset_bands(asset_id: str) -> list[str] | None:
    """Band names of an image / first image of a collection, or None if the asset is unavailable."""
    try:
        info = ee.data.getAsset(asset_id)
    except Exception:
        return None
    try:
        img = ee.Image(asset_id) if info.get("type") == "IMAGE" else ee.ImageCollection(asset_id).first()
        return img.bandNames().getInfo()
    except Exception:
        return None


def tidal_zone(cfg: Config):
    """Where the tide actually reaches. Salt marsh is only labelled inside this zone.

    Primary: Murray et al. (2022) tidal wetland probability (tidal flat + marsh + mangrove, 30 m).
    Fallback: low-lying (<= 3 m) land within 1 km of open water.
    """
    lc = cfg.labels
    bands = _asset_bands(lc.tidal_wetland)
    if bands and lc.tidal_wetland_band in bands:
        return ee.Image(lc.tidal_wetland).select(lc.tidal_wetland_band).gte(lc.tidal_wetland_min_prob), "murray-gic"
    dem = ee.Image(lc.dem).select("elevation")
    wc = ee.ImageCollection(lc.worldcover).first().select("Map")
    near_water = wc.eq(80).focalMax(radius=1000, kernelType="circle", units="meters")
    return dem.lte(3).And(near_water), "elevation-fallback"


def reference_labels(region, cfg: Config, report: dict | None = None):
    """Fuse published global products into the class schema (uint8, 255 = ignore).

    Priority (later overrides earlier):
      WorldCover base -> Murray tidal flats -> tidal-zone herbaceous vegetation = salt marsh
      -> WorldCover mangrove -> Allen Coral Atlas / survey-polygon seagrass.
    Freshwater wetland (WorldCover 90 outside the tidal zone, e.g. Everglades sawgrass) is ignored.
    """
    _require_ee()
    lc = cfg.labels
    wc = ee.ImageCollection(lc.worldcover).first().select("Map")
    ign = IGNORE_INDEX
    k = KEY_TO_ID
    used = []

    # WorldCover: 10 tree,20 shrub,30 grass,40 crop,50 built,60 bare,70 snow,80 water,90 herb. wetland,
    #             95 mangrove,100 moss/lichen
    base = wc.remap(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        [k["other_land"]] * 6 + [ign, k["water"], ign, k["mangrove"], k["other_land"]],
        ign,
    )
    used.append("worldcover")

    intertidal = ee.ImageCollection(lc.intertidal).sort("system:time_start", False).first().select(0)
    lab = base.where(intertidal.eq(1).And(wc.neq(95)), k["tidal_flat"])
    used.append("murray-tidal-flats")

    tz, tz_src = tidal_zone(cfg)
    used.append(tz_src)
    dem = ee.Image(lc.dem).select("elevation")
    herb = wc.eq(90).Or(wc.eq(30)).Or(wc.eq(20))
    marsh = herb.And(tz.unmask(0)).And(dem.lte(lc.marsh_max_elev_m)).And(wc.neq(95))
    lab = lab.where(marsh, k["saltmarsh"])
    lab = lab.where(wc.eq(95), k["mangrove"])

    if _asset_bands(lc.reef_habitat):
        benthic = ee.Image(lc.reef_habitat).select("benthic")
        lab = lab.where(benthic.eq(lc.seagrass_value), k["seagrass"])
        used.append("allen-coral-atlas")
    for fc_id in lc.seagrass_vectors:
        try:
            fc = ee.FeatureCollection(fc_id).filterBounds(region)
            sg = ee.Image(0).paint(fc, 1).selfMask()
            lab = lab.where(sg.unmask(0).eq(1).And(wc.eq(80).Or(lab.eq(ign))), k["seagrass"])
            used.append(fc_id)
        except Exception:
            pass
    if report is not None:
        report["label_sources"] = used
    return lab.unmask(ign).clip(region).toUint8().rename("label")


# --------------------------------------------------------------------------- download
def _compute_tile(image, crs: str, transform: rasterio.Affine, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    tx = transform @ rasterio.Affine.translation(x0, y0)
    req = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {
                "scaleX": tx.a,
                "shearX": tx.b,
                "translateX": tx.c,
                "shearY": tx.d,
                "scaleY": tx.e,
                "translateY": tx.f,
            },
            "crsCode": crs,
        },
    }
    arr = ee.data.computePixels(req)
    if isinstance(arr, bytes | bytearray):  # older clients return raw .npy bytes
        arr = np.load(io.BytesIO(arr))
    return np.stack([arr[n] for n in arr.dtype.names])


def download(image, bbox: list[float], out_path: str | Path, cfg: Config, dtype: str = "uint16",
             nodata: int | None = 0, band_names: list[str] | None = None, progress=None) -> Path:
    """Download an ee.Image over `bbox` to a tiled, compressed GeoTIFF in local UTM."""
    _require_ee()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crs, transform, W, H = utm_grid(bbox, cfg.imagery.scale_m)
    band_names = band_names or image.bandNames().getInfo()
    t = cfg.imagery.tile_px
    tiles = [(x, y) for y in range(0, H, t) for x in range(0, W, t)]
    profile = dict(driver="GTiff", width=W, height=H, count=len(band_names), dtype=dtype, crs=crs,
                   transform=transform, nodata=nodata, compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.descriptions = tuple(band_names)
        for i, (x, y) in enumerate(tiles):
            w, h = min(t, W - x), min(t, H - y)
            block = _compute_tile(image, crs, transform, x, y, w, h).astype(dtype)
            dst.write(block, window=rasterio.windows.Window(x, y, w, h))
            if progress:
                progress((i + 1) / len(tiles))
    return out_path


def bbox_geometry(bbox: list[float]):
    _require_ee()
    return ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
