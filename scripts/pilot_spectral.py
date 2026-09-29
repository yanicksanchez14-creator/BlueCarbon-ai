"""Build models/pilot_spectral_mission_bay_2018.json, the demo's default model.

Same honest protocol as scripts/pilot_mission_bay.py: leave-one-2.56 km-block-out spatial
cross-validation on the 2018 Mission Bay hand labels, pooled confusion matrix, then a final fit
on every labelled pixel. Run from the repo root:  python scripts/pilot_spectral.py  (~2 min, CPU)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from bluecarbon.features import valid_mask
from bluecarbon.metrics import confusion, summarize
from bluecarbon.schema import IGNORE_INDEX, N_CLASSES
from bluecarbon.spectral import SpectralModel, pixel_features

D = Path("data/pilot")
BP = 256
FOLDS = [(0, 0), (0, 1), (1, 1), (1, 0)]


def main():
    with rasterio.open(D / "mission_bay_2018_image.tif") as s:
        bands = s.read()
    with rasterio.open(D / "mission_bay_2018_label.tif") as s:
        lab = s.read(1)
    X = pixel_features(bands)
    Xf = X.reshape(X.shape[0], -1).T
    y = lab.ravel()
    H, W = lab.shape
    block = ((np.arange(H)[:, None] // BP), (np.arange(W)[None, :] // BP))
    ok = ((y != IGNORE_INDEX) & valid_mask(bands).ravel())

    pooled = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    for by, bx in FOLDS:
        test = ((block[0] == by) & (block[1] == bx)).ravel()
        m = SpectralModel().fit(Xf[ok & ~test], y[ok & ~test])
        # predict the whole held-out block so smoothing sees real neighbours
        rows = slice(by * BP, min((by + 1) * BP, H))
        cols = slice(bx * BP, min((bx + 1) * BP, W))
        pred, _ = m.predict(bands[:, rows, cols])
        cm = confusion(lab[rows, cols], pred)
        pooled += cm
        print(f"fold {(by, bx)}: IoU {summarize(cm)['iou']}")
    s = summarize(pooled)
    print("pooled:", {k: s[k] for k in ("mIoU", "macro_f1", "overall_accuracy", "kappa")}, s["iou"])
    cv = {"pooled_test": s, "pooled_confusion": pooled.tolist(), "folds": 4,
          "protocol": "leave-one-2.56km-block-out spatial CV, Mission Bay 2018"}
    (D / "cv_metrics_spectral.json").write_text(json.dumps(cv, indent=2))

    final = SpectralModel().fit(Xf[ok], y[ok])
    final.metrics = {"test": s, "test_confusion": pooled.tolist()}
    final.extra = {"name": "pilot-spectral-mission-bay-2018", "pilot": True, "evaluation": cv["protocol"],
                   "training_data": "Mission Bay 2018 Sentinel-2 composite with the 2025 QGIS hand labels "
                                    "(salt marsh, seagrass, land) plus MNDWI water"}
    final.save("models/pilot_spectral_mission_bay_2018.json")


if __name__ == "__main__":
    main()
