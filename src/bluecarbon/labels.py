"""Local label post-processing: boundary buffering, user polygon overrides, class stats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio import features as rfeatures

from .schema import CLASS_KEYS, IGNORE_INDEX, N_CLASSES


def ignore_boundaries(label: np.ndarray, px: int) -> np.ndarray:
    """Set pixels within `px` of a class boundary to IGNORE.

    Global products are 10-30 m and edges are the least reliable pixels, so we don't
    train on them. Pure numpy (no scipy/opencv dependency).
    """
    if px <= 0:
        return label
    out = label.copy()
    edge = np.zeros(label.shape, bool)
    for dy in range(-px, px + 1):
        for dx in range(-px, px + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(label, dy, 0), dx, 1)
            edge |= shifted != label
    # np.roll wraps around; don't invent edges on the image border
    edge[:px, :] = edge[-px:, :] = False
    edge[:, :px] = edge[:, -px:] = False
    out[edge & (label != IGNORE_INDEX)] = IGNORE_INDEX
    return out


def apply_overrides(label: np.ndarray, transform, crs, vector_path: str | Path) -> np.ndarray:
    """Burn user polygons (column `class_id`) over the reference labels."""
    import geopandas as gpd  # optional dependency, only needed for overrides

    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        return label
    if "class_id" not in gdf.columns:
        raise ValueError(f"{vector_path} needs an integer `class_id` column")
    gdf = gdf.to_crs(crs)
    shapes = [(g, int(c)) for g, c in zip(gdf.geometry, gdf.class_id, strict=True) if g is not None]
    burned = rfeatures.rasterize(shapes, out_shape=label.shape, transform=transform, fill=IGNORE_INDEX,
                                 dtype="uint8")
    out = label.copy()
    m = burned != IGNORE_INDEX
    out[m] = burned[m]
    return out


def class_histogram(label: np.ndarray) -> dict[str, int]:
    counts = np.bincount(label[label != IGNORE_INDEX].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    return {k: int(c) for k, c in zip(CLASS_KEYS, counts, strict=True)}


def postprocess_label_file(path: str | Path, boundary_px: int, overrides: str | None = None) -> dict[str, int]:
    with rasterio.open(path, "r+") as ds:
        lab = ds.read(1)
        if overrides:
            lab = apply_overrides(lab, ds.transform, ds.crs, overrides)
        lab = ignore_boundaries(lab, boundary_px)
        ds.write(lab, 1)
    return class_histogram(lab)
