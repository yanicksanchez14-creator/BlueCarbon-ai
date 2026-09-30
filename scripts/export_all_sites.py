"""Turn every fetched site into a demo page for the web app (run in Colab after training).

    python scripts/export_all_sites.py            # uses runs/default/model/best.*

Training sites are labelled as such on their page, because maps of places the model trained on
look better than they would on a new coastline. Held-out sites are the honest showcase.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

WORK = Path("runs/default")

TITLES = {
    "mission_bay_ca": ("Mission Bay, San Diego", "California, USA"),
    "south_sd_bay_ca": ("South San Diego Bay", "California, USA"),
    "south_sf_bay_ca": ("South San Francisco Bay", "California, USA"),
    "ten_thousand_isl_fl": ("Ten Thousand Islands", "Florida, USA"),
    "florida_bay_fl": ("Florida Bay, Everglades", "Florida, USA"),
    "andros_bahamas": ("Andros Island", "The Bahamas"),
    "laguna_terminos_mx": ("Laguna de Términos", "Campeche, Mexico"),
    "sundarbans_bd": ("Sundarbans", "Bangladesh"),
    "wadden_sea_de": ("Wadden Sea", "Schleswig-Holstein, Germany"),
    "moreton_bay_au": ("Moreton Bay", "Queensland, Australia"),
    "tampa_bay_fl": ("Tampa Bay", "Florida, USA"),
    "florida_keys_fl": ("Lower Florida Keys", "Florida, USA"),
    "belize_lagoon_bz": ("Belize Barrier Reef lagoon", "Belize"),
    "exuma_bahamas": ("Exuma Cays", "The Bahamas"),
    "hinchinbrook_au": ("Hinchinbrook Island", "Queensland, Australia"),
    "chwaka_bay_tz": ("Chwaka Bay", "Zanzibar, Tanzania"),
    "safaga_eg": ("Safaga", "Red Sea, Egypt"),
}


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    model = next(p for p in (WORK / "model").glob("best.*") if p.suffix != ".txt")
    cfg = yaml.safe_load(Path("configs/sites.yaml").read_text())
    for s in cfg["sites"]:
        d = WORK / "sites" / s["name"]
        if not (d / "image.tif").exists():
            print(f"skip {s['name']} (not downloaded)")
            continue
        title, region = TITLES.get(s["name"], (s["name"], ""))
        held_out = s.get("role") == "test"
        note = s.get("note", "").removeprefix("Held-out: ")
        desc = (f"{note}. Sentinel-2 {cfg['year']} composite. "
                + ("This area was held out of training, so the map shows how the AI does on a coastline it has "
                   "never seen." if held_out else
                   "This area was part of the training data, so the map looks better than it would on a new coastline."))
        run("bluecarbon", "predict", str(d / "image.tif"), "-m", str(model), "--out", str(d / "pred.tif"))
        run("bluecarbon", "export-demo", str(d), "--name", s["name"], "--title", title, "-m", str(model),
            "--description", desc)
        meta_p = Path("demo_data") / s["name"] / "meta.json"
        meta = json.loads(meta_p.read_text())
        meta.update(region=region, period=f"Full year {cfg['year']}", held_out=held_out)
        meta_p.write_text(json.dumps(meta, indent=2))
    print("done:", sorted(p.name for p in Path("demo_data").iterdir()))


if __name__ == "__main__":
    sys.exit(main())
