"""One interface over both model families (U-Net checkpoints and spectral LightGBM models)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Predictor:
    kind: str
    meta: dict

    def predict(self, bands: np.ndarray, tile: int = 256, overlap: int = 64, tta: bool = True,
                progress=None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class TorchPredictor(Predictor):
    kind = "unet"

    def __init__(self, path: str | Path, device: str = "auto"):
        from .model import load_checkpoint, resolve_device

        self.model, self.norm, ck = load_checkpoint(path, resolve_device(device))
        self.meta = {"arch": ck["arch"], "encoder": ck["encoder"], "kind": self.kind,
                     "metrics": ck.get("metrics", {}), "extra": ck.get("extra", {})}

    def predict(self, bands, tile=256, overlap=64, tta=True, progress=None):
        from .predict import predict_array

        return predict_array(self.model, self.norm, bands, tile, overlap, tta, progress=progress)


class SpectralPredictor(Predictor):
    kind = "spectral-lgbm"

    def __init__(self, path: str | Path):
        from .spectral import SpectralModel

        self.model = SpectralModel.load(path)
        self.meta = {**self.model.info, "metrics": self.model.metrics, "extra": self.model.extra}

    def predict(self, bands, tile=256, overlap=64, tta=True, progress=None):
        out = self.model.predict(bands)
        if progress:
            progress(1.0)
        return out


def is_spectral(path: str | Path) -> bool:
    p = Path(path)
    if p.suffix == ".json":
        return True
    with open(p, "rb") as f:
        return f.read(1) == b"{"


def load_predictor(path: str | Path, device: str = "auto") -> Predictor:
    return SpectralPredictor(path) if is_spectral(path) else TorchPredictor(path, device)


def model_card(path: str | Path) -> dict:
    """Metadata for display without loading weights into a network."""
    if is_spectral(path):
        d = json.loads(Path(path).read_text())
        return {"arch": "LightGBM", "encoder": "spectral-context", "kind": d["kind"], "metrics": d.get("metrics", {}),
                "extra": d.get("extra", {})}
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {"arch": ck["arch"], "encoder": ck["encoder"], "kind": "unet", "metrics": ck.get("metrics", {}),
            "extra": ck.get("extra", {})}
