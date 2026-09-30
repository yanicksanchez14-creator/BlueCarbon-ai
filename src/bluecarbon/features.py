"""Sentinel-2 bands -> model input features.

Imagery is stored on disk as the raw surface-reflectance bands (uint16, scale 1e4) so
spectral indices are always computed the same way at training and inference time.
"""

from __future__ import annotations

import numpy as np

# Sentinel-2 L2A bands used by the model (10 m and 20 m bands; 60 m atmospheric bands dropped).
S2_BANDS: list[str] = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
REFLECTANCE_SCALE = 10_000.0

# name -> (band_a, band_b) for normalized differences (a - b) / (a + b)
INDICES: dict[str, tuple[str, str]] = {
    "NDVI": ("B8", "B4"),  # vegetation vigour
    "NDWI": ("B3", "B8"),  # open water (McFeeters)
    "MNDWI": ("B3", "B11"),  # water vs. built/soil (Xu)
    "NDMI": ("B8", "B11"),  # canopy moisture; separates mangrove from dry upland
}

FEATURE_NAMES: list[str] = S2_BANDS + list(INDICES)
N_FEATURES = len(FEATURE_NAMES)

# Ancillary "context" layers stored alongside the imagery (ancillary.tif, int16):
#   elevation  NASADEM metres
#   tidal_prob Murray et al. tidal wetland probability, 0-100
#   abs_lat    |latitude| x 100
# Elevation and tidal probability carry what a single image cannot show: whether the tide reaches
# a pixel (salt marsh vs freshwater marsh). Latitude is stored but is NOT a model input: with a
# handful of training sites the model used it as a site ID ("no mangrove north of 25 deg") and
# missed the mangroves of held-out Tampa Bay and Moreton Bay. The mangrove latitude range is
# applied as an explicit rule instead (priors.py).
ANCILLARY_BANDS: list[str] = ["elevation", "tidal_prob", "abs_lat"]
ANCILLARY_FEATURES: list[str] = ["ELEV", "TIDAL"]
N_ANC_FEATURES = len(ANCILLARY_FEATURES)
FEATURE_NAMES_ANC: list[str] = FEATURE_NAMES + ANCILLARY_FEATURES


def ancillary_features(anc: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(anc.astype(np.float32))
    elev = np.clip(a[0], -10, 60) / 60.0
    tidal = np.clip(a[1], 0, 100) / 100.0
    return np.stack([elev, tidal]).astype(np.float32)


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = a + b
    out = np.zeros_like(a, dtype=np.float32)
    np.divide(a - b, den, out=out, where=np.abs(den) > 1e-6)
    return np.clip(out, -1.0, 1.0)


def compute_features(bands: np.ndarray, band_names: list[str] | None = None,
                     anc: np.ndarray | None = None) -> np.ndarray:
    """(C, H, W) raw S2 bands -> (N_FEATURES, H, W) float32 reflectance + indices.

    Accepts either scaled integers (0..10000) or reflectance floats (0..1). NaNs become 0.
    """
    band_names = band_names or S2_BANDS
    if bands.shape[0] != len(band_names):
        raise ValueError(f"Expected {len(band_names)} bands, got {bands.shape[0]}")
    x = np.nan_to_num(bands.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if np.nanmax(x) > 2.0:  # integer-scaled reflectance
        x = x / REFLECTANCE_SCALE
    idx = {n: i for i, n in enumerate(band_names)}
    missing = [b for b in S2_BANDS if b not in idx]
    if missing:
        raise ValueError(f"Missing bands: {missing}")
    refl = np.stack([x[idx[b]] for b in S2_BANDS])
    ind = np.stack([_nd(x[idx[a]], x[idx[b]]) for a, b in INDICES.values()])
    parts = [refl, ind]
    if anc is not None:
        if anc.shape[1:] != bands.shape[1:]:
            raise ValueError(f"ancillary shape {anc.shape} does not match imagery {bands.shape}")
        parts.append(ancillary_features(anc))
    return np.concatenate(parts).astype(np.float32)


def valid_mask(bands: np.ndarray) -> np.ndarray:
    """Pixels with real data in every band (not nodata / not masked cloud)."""
    finite = np.isfinite(bands).all(axis=0)
    nonzero = (np.nan_to_num(bands) != 0).any(axis=0)
    return finite & nonzero


class Normalizer:
    """Per-feature standardization. Stats travel inside the model checkpoint."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, feature_arrays: list[np.ndarray], masks: list[np.ndarray] | None = None) -> Normalizer:
        n = feature_arrays[0].shape[0]
        s = np.zeros(n, np.float64)
        s2 = np.zeros(n, np.float64)
        cnt = 0
        for i, f in enumerate(feature_arrays):
            m = masks[i] if masks is not None else np.ones(f.shape[1:], bool)
            v = f[:, m].astype(np.float64)
            s += v.sum(1)
            s2 += (v**2).sum(1)
            cnt += v.shape[1]
        if cnt == 0:
            raise ValueError("No valid pixels to fit normalizer")
        mean = s / cnt
        std = np.sqrt(np.maximum(s2 / cnt - mean**2, 1e-12))
        return cls(mean, std)

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        return ((feats - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> Normalizer:
        return cls(np.array(d["mean"]), np.array(d["std"]))
