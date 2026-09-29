"""Reproduce the pilot model shipped in runs/pilot/pilot_unet_mission_bay_2018.pt.

Data: data/pilot/ holds the 2018 Mission Bay Sentinel-2 composite and the original 2025
QGIS hand labels, converted to the v2 class schema (see docs/AUDIT.md for the conversion).

1. Leave-one-block-out spatial cross-validation over four 2.56 km blocks -> cv_metrics.json
2. Final model on three blocks (fourth = validation) -> the checkpoint used by the demo app

Run from the repo root:  python scripts/pilot_mission_bay.py   (CPU is fine: ~5 min)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bluecarbon.config import load_config
from bluecarbon.metrics import summarize
from bluecarbon.model import load_checkpoint, save_checkpoint
from bluecarbon.tiling import make_chips
from bluecarbon.train import train

D = Path("data/pilot")
OUT = Path("runs/pilot")
BP = 256  # block size in pixels (2.56 km)
CHIP = 96
BLOCKS = [(0, 0), (0, 1), (1, 1), (1, 0)]


def block(r):
    return r.row // BP, r.col // BP


def main():
    recs = make_chips(D / "mission_bay_2018_image.tif", D / "mission_bay_2018_label.tif", OUT / "chips",
                      "mission_bay_2018", size=CHIP, stride=32, min_labeled_frac=0.3, block_km=2.56)
    # stride < chip size is fine *within* a block; chips that straddle two blocks are dropped
    recs = [r for r in recs if block(r) == ((r.row + CHIP - 1) // BP, (r.col + CHIP - 1) // BP)]
    cfg = load_config(None, {
        "model": {"encoder": "resnet18", "encoder_weights": None},
        "train": {"epochs": 30, "batch_size": 16, "num_workers": 0, "lr": 2e-3, "device": "auto",
                  "amp": False, "patience": 30},
    })

    pooled = np.zeros((6, 6), np.int64)
    for k, tb in enumerate(BLOCKS):
        vb = BLOCKS[(k + 1) % 4]
        for r in recs:
            r.split = "test" if block(r) == tb else "val" if block(r) == vb else "train"
        res = train(cfg, recs, OUT / f"cv{k}", log=lambda *a: None)
        pooled += np.array(res["test_confusion"])
        print(f"fold {k} test block {tb}: mIoU {res['test']['mIoU']}")
    cv = {"pooled_test": summarize(pooled), "pooled_confusion": pooled.tolist(), "folds": 4,
          "protocol": "leave-one-2.56km-block-out spatial CV, Mission Bay 2018, 96px chips"}
    (D / "cv_metrics.json").write_text(json.dumps(cv, indent=2))
    print("pooled:", {k: cv["pooled_test"][k] for k in ("mIoU", "macro_f1", "overall_accuracy", "kappa")})

    for r in recs:
        r.split = "val" if block(r) == (0, 0) else "train"
    train(cfg, recs, OUT / "final", log=print)
    model, norm, ck = load_checkpoint(OUT / "final" / "model.pt")
    model = model.half()  # halves the file size; weights are cast back to fp32 on load
    save_checkpoint(Path("runs/pilot/pilot_unet_mission_bay_2018.pt"), model, ck["arch"], ck["encoder"], norm,
                    {"val": ck["metrics"]["val"], "test": cv["pooled_test"], "test_confusion": cv["pooled_confusion"]},
                    {"name": "pilot-mission-bay-2018", "pilot": True, "evaluation": cv["protocol"],
                     "training_data": "Mission Bay 2018 Sentinel-2 composite, 2025 QGIS hand labels "
                                      "(salt marsh, seagrass, land) + MNDWI water"})
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
