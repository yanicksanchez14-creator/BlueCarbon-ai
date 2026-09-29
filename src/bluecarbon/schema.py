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
    color: str  # hex, used for maps and charts
    blue_carbon: bool


CLASSES: tuple[HabitatClass, ...] = (
    HabitatClass(0, "water", "Open water", "#2b6cb0", False),
    HabitatClass(1, "mangrove", "Mangrove", "#1b7f3b", True),
    HabitatClass(2, "saltmarsh", "Salt marsh", "#9acd32", True),
    HabitatClass(3, "seagrass", "Seagrass", "#20b2aa", True),
    HabitatClass(4, "tidal_flat", "Tidal flat / bare", "#d2b48c", False),
    HabitatClass(5, "other_land", "Other land", "#9e9e9e", False),
)

N_CLASSES = len(CLASSES)
CLASS_KEYS = [c.key for c in CLASSES]
BLUE_CARBON_KEYS = [c.key for c in CLASSES if c.blue_carbon]
KEY_TO_ID = {c.key: c.id for c in CLASSES}


def class_by_key(key: str) -> HabitatClass:
    return CLASSES[KEY_TO_ID[key]]


def palette_rgb() -> list[tuple[int, int, int]]:
    out = []
    for c in CLASSES:
        h = c.color.lstrip("#")
        out.append(tuple(int(h[i : i + 2], 16) for i in (0, 2, 4)))
    return out
