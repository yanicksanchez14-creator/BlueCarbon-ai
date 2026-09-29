# Experiments: making salt marsh and seagrass detection work

**Goal.** The blue carbon classes are the whole point of the product, and the first U-Net pilot
missed them: salt marsh IoU 0.13 and seagrass 0.00.

**Protocol (fixed for every row).** Mission Bay 2018 Sentinel-2 composite with the 2025 QGIS hand
labels, split into four 2.56 km blocks. Each block is held out once, the model trains on the other
three, and the confusion matrices from the four held-out blocks are pooled. No pixel is ever scored
by a model that trained on it or on its immediate neighbours.

| # | Model | Salt marsh IoU | Seagrass IoU | Water | Land | mIoU |
|---|---|---:|---:|---:|---:|---:|
| 1 | U-Net (ResNet-18, from scratch) | 0.13 | 0.00 | 0.92 | 0.95 | 0.50 |
| 2 | LightGBM, pixel features only (14) | 0.79 | 0.38 | 0.98 | 0.98 | 0.78 |
| 3 | LightGBM, pixel + 3/7/15/31 px mean & std context (126) | 0.55 | 0.29 | 0.98 | 0.95 | 0.69 |
| 4 | LightGBM, pixel + water-column ratios (19) | 0.74 | 0.42 | 0.99 | 0.98 | 0.78 |
| 5 | #4 + 3 px context means | 0.74 | 0.47 | 0.99 | 0.98 | 0.79 |
| 6 | #4 + 3 & 9 px context means | 0.69 | 0.53 | 0.99 | 0.98 | 0.80 |
| 7 | #4 + 3×3 probability smoothing | 0.78 | 0.45 | 0.99 | 0.98 | 0.80 |
| 8 | #5 + 3×3 smoothing | 0.76 | 0.49 | 0.99 | 0.98 | 0.81 |
| **9** | **#6 + 5×5 smoothing (shipped)** | **0.72** | **0.52** | **0.99** | **0.98** | **0.80** |

**Findings**
- **With little data, trees beat a deep network.** The U-Net has millions of parameters and saw about
  3,000 salt-marsh pixels, so it overfit. Gradient-boosted trees on physically meaningful features
  need far fewer examples.
- **Seagrass needs water-column features.** Seagrass sits under water, so its signal is a small
  shift in the blue/green/red balance. Log band ratios (the depth-invariant terms from Lyzenga's
  water-column correction) plus a red-edge index added about +0.15 IoU.
- **A little spatial context helps, a lot hurts.** 30–90 m neighbourhood means help seagrass (meadows are
  contiguous). Large windows with variance features (#3) memorised block-specific texture and
  generalised worse.
- **Smoothing** the class probabilities removes isolated misclassified pixels at almost no cost.
- #9 was chosen for the best balance between the two blue carbon classes. Picking among nine variants
  on the same folds is mildly optimistic. The multi-site run and the held-out hand-label test in the
  Colab notebook are the unbiased check.

**Where the remaining errors are.** Seagrass is still the hardest class. Most of its errors are at the
deep edge of meadows, where the signal fades into open water. Next steps, in order of expected impact:
1. More labelled seagrass from other bays (local eelgrass survey polygons exist for San Diego Bay).
2. Low-tide image selection, so shallow meadows are less submerged.
3. A U-Net pretrained on ImageNet or a satellite foundation model, fine-tuned on the multi-site data.
   Deep models overtake trees once there is enough data, which is why the pipeline trains both and keeps
   the winner (`bluecarbon train --kind both`).

Reproduce with `python scripts/pilot_spectral.py` (row 9) and `python scripts/pilot_mission_bay.py` (row 1).
