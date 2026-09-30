"""Class schema shared by labels, training, inference, carbon accounting and the app.

v1 used 0 for both "water" and "unlabeled", which silently taught the model that every
unannotated pixel was water. v2 reserves 255 for "ignore" and never trains on it.
"""

from __future__ import annotations

from dataclasses import dataclass

IGNORE_INDEX = 255


@dataclass(frozen=True)
class HabitatClass:
    id: int
    key: str
    name: str
    color: str  # hex, used for maps and charts (validated for CVD separation; 'other' is a neutral)
    blue_carbon: bool


CLASSES: tuple[HabitatClass, ...] = (
    HabitatClass(0, "water", "Open water", "#1f5fbf", False),
    HabitatClass(1, "mangrove", "Mangrove", "#0b7a3e", True),
    HabitatClass(2, "saltmarsh", "Salt marsh", "#9bb52a", True),
    HabitatClass(3, "seagrass", "Seagrass", "#14a3a0", True),
    HabitatClass(4, "tidal_flat", "Tidal flat / bare", "#c07a2c", False),
    HabitatClass(5, "other_land", "Other land", "#8b9199", False),
    # Inland (non-tidal) marsh and swamp. Stores carbon but is not blue carbon, so it is mapped
    # and reported separately and excluded from the blue carbon totals.
    HabitatClass(6, "freshwater", "Freshwater wetland", "#a0529b", False),
)

N_CLASSES = len(CLASSES)
CLASS_KEYS = [c.key for c in CLASSES]
BLUE_CARBON_KEYS = [c.key for c in CLASSES if c.blue_carbon]
KEY_TO_ID = {c.key: c.id for c in CLASSES}


def compatible(classes: list[str]) -> bool:
    """Older models trained on a prefix of the current schema still load (missing classes get p=0)."""
    return list(classes) == CLASS_KEYS[: len(classes)]


def pad_confusion(cm):
    import numpy as np

    cm = np.asarray(cm)
    if cm.shape[0] >= N_CLASSES:
        return cm
    out = np.zeros((N_CLASSES, N_CLASSES), cm.dtype)
    out[: cm.shape[0], : cm.shape[1]] = cm
    return out


def class_by_key(key: str) -> HabitatClass:
    return CLASSES[KEY_TO_ID[key]]


def palette_rgb() -> list[tuple[int, int, int]]:
    out = []
    for c in CLASSES:
        h = c.color.lstrip("#")
        out.append(tuple(int(h[i : i + 2], 16) for i in (0, 2, 4)))
    return out
