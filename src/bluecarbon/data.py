"""PyTorch dataset over .npz chips."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import Normalizer, compute_features, valid_mask
from .schema import IGNORE_INDEX, N_CLASSES
from .tiling import ChipRecord


def load_chip(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["image"], z["label"]


class ChipDataset(Dataset):
    def __init__(self, records: list[ChipRecord], normalizer: Normalizer, augment: bool = False):
        self.records = records
        self.norm = normalizer
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        img, lab = load_chip(self.records[i].path)
        x = self.norm(compute_features(img))
        y = lab.astype(np.int64)
        if self.augment:
            x, y = _augment(x, y)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def _augment(x: np.ndarray, y: np.ndarray, rng=np.random):
    """Dihedral (8 orientations) + mild per-chip spectral jitter.

    Spectral jitter mimics atmosphere / sun-angle differences between sites.
    """
    k = rng.randint(4)
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    if rng.rand() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    gain = 1 + rng.normal(0, 0.05, size=(x.shape[0], 1, 1)).astype(np.float32)
    bias = rng.normal(0, 0.05, size=(x.shape[0], 1, 1)).astype(np.float32)
    return x * gain + bias, y


def fit_normalizer(records: list[ChipRecord], max_chips: int = 400, seed: int = 0) -> Normalizer:
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(records))[:max_chips]
    feats, masks = [], []
    for i in pick:
        img, _ = load_chip(records[i].path)
        feats.append(compute_features(img))
        masks.append(valid_mask(img))
    return Normalizer.fit(feats, masks)


def class_frequencies(records: list[ChipRecord]) -> np.ndarray:
    counts = np.zeros(N_CLASSES, np.int64)
    for r in records:
        _, lab = load_chip(r.path)
        counts += np.bincount(lab[lab != IGNORE_INDEX].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    return counts
