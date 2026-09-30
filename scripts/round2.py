"""Round 2: rebuild labels with the improved rules, retrain, and export every demo page.

Run in the same Colab session after the first run (the satellite images are reused):

    !git pull -q && python scripts/round2.py

Takes about 45-75 minutes on an L4/T4 GPU and writes round2_results.zip.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("runs/default")


def run(*args: str) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    run("bluecarbon", "fetch", "--labels-only")
    shutil.rmtree(WORK / "chips", ignore_errors=True)
    run("bluecarbon", "chips")
    run("bluecarbon", "train", "--kind", "both")
    best = next(p for p in (WORK / "model").glob("best.*") if p.suffix != ".txt")
    print((WORK / "model" / "best.txt").read_text())

    run("bluecarbon", "case-study", "-m", str(best))
    scene = WORK / "scenes" / "mission_bay_change"
    (scene / "periods.json").write_text(json.dumps({"t0": "Summer 2018", "t1": "Summer 2024"}))
    shutil.rmtree("demo_data", ignore_errors=True)
    run("bluecarbon", "export-demo", str(scene), "--name", "mission_bay_change",
        "--title", "Mission Bay, San Diego (2018 → 2024)", "-m", str(best),
        "--description", "Salt marsh and seagrass in Mission Bay, summer 2018 compared with summer 2024. "
                         "This bay was never used in training.")
    meta_p = Path("demo_data/mission_bay_change/meta.json")
    meta = json.loads(meta_p.read_text())
    meta.update(region="California, USA", period="2018 → 2024", held_out=True)
    meta_p.write_text(json.dumps(meta, indent=2))
    run(sys.executable, "scripts/export_all_sites.py")

    out = Path("round2_results")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    shutil.copy(best, out / best.name)
    shutil.copy(WORK / "model" / "best.txt", out / "best.txt")
    for f in ("spectral_metrics.json", "metrics.json"):
        if (WORK / "model" / f).exists():
            shutil.copy(WORK / "model" / f, out / f)
    labels = {p.parent.name: json.loads(p.read_text()).get("label_px") for p in (WORK / "sites").glob("*/meta.json")}
    (out / "label_summary.json").write_text(json.dumps(labels, indent=2))
    shutil.copytree("demo_data", out / "demo_data")
    shutil.make_archive("round2_results", "zip", ".", str(out))
    print("\nDone: round2_results.zip. Download it with: from google.colab import files; "
          "files.download('round2_results.zip')")


if __name__ == "__main__":
    main()
