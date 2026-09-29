"""BlueCarbon — coastal blue carbon mapping from Sentinel-2.

Runs with zero credentials on the bundled demo sites. With Earth Engine credentials in
Streamlit secrets (GEE_SERVICE_ACCOUNT), the Analyze tab maps any coastline on demand.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bluecarbon.config import load_config  # noqa: E402
from bluecarbon.schema import BLUE_CARBON_KEYS, CLASSES  # noqa: E402

DEMO_DIR = ROOT / "demo_data"
DEFAULT_MODEL = ROOT / "models" / "pilot_mission_bay_2018.pt"
REPO = "https://github.com/yanicksanchez14-creator/bluecarbon-ai"
MAX_AREA_KM2 = 60
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Imagery © Esri, Maxar, Earthstar Geographics"

CFG = load_config(ROOT / "configs" / "default.yaml")
CLS = {c.key: c for c in CLASSES}

LOGO = """<svg width="34" height="34" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0e5a67"/>
<stop offset="1" stop-color="#082f3d"/></linearGradient></defs>
<rect width="40" height="40" rx="10" fill="url(#g)"/>
<path d="M20 8c5.5 3.2 7.6 8.4 5.2 13.2-1.4 2.7-3.3 3.8-5.2 4.3-1.9-.5-3.8-1.6-5.2-4.3C12.4 16.4 14.5 11.2 20 8z" fill="#7fd6c2"/>
<path d="M20 11v14" stroke="#0b3f4c" stroke-width="1.4" stroke-linecap="round"/>
<path d="M7 28.5c2.2 0 2.2-1.6 4.4-1.6s2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 3.8 1.6"
 fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
<path d="M7 33c2.2 0 2.2-1.6 4.4-1.6s2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 3.8 1.6"
 fill="none" stroke="#ffffff" stroke-opacity=".45" stroke-width="2" stroke-linecap="round"/></svg>"""

st.set_page_config(page_title="BlueCarbon · Coastal carbon mapping", page_icon=":material/eco:", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------- design system
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
:root{
  --ink:#0b1f2a; --ink-2:#344651; --muted:#6a7a84; --line:#e4eaed; --line-2:#eef2f4;
  --bg:#f5f7f8; --card:#ffffff; --brand:#0e5a67; --brand-ink:#083744; --accent:#14a3a0;
  --accent-soft:#e6f5f3; --warn-bg:#fff6e8; --warn-ink:#8a5a00; --radius:14px;
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
  font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }
.stApp{background:var(--bg); color:var(--ink);}
header[data-testid="stHeader"]{background:transparent; height:0;}
div[data-testid="stToolbar"]{right:1rem;}
.block-container{padding:1.2rem 2.2rem 3rem; max-width:1320px;}
h1,h2,h3,h4{color:var(--ink); letter-spacing:-.015em;}

/* top bar */
.bc-top{display:flex; align-items:center; justify-content:space-between; padding:.35rem 0 1.1rem;
  border-bottom:1px solid var(--line); margin-bottom:1.6rem;}
.bc-brand{display:flex; align-items:center; gap:.7rem;}
.bc-word{font-weight:700; font-size:1.18rem; color:var(--ink); letter-spacing:-.02em;}
.bc-word span{color:var(--accent);}
.bc-tag{font-size:.72rem; font-weight:600; color:var(--brand); background:var(--accent-soft);
  border:1px solid #cfe9e5; padding:2px 8px; border-radius:999px; margin-left:.35rem;}
.bc-links a{color:var(--ink-2); text-decoration:none; font-size:.88rem; font-weight:500; margin-left:1.4rem;}
.bc-links a:hover{color:var(--brand);}

/* hero */
.bc-hero{display:grid; grid-template-columns:1.35fr 1fr; gap:2rem; align-items:end; margin-bottom:1.6rem;}
.bc-eyebrow{font-size:.74rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--accent);}
.bc-hero h1{font-size:2.35rem; line-height:1.12; font-weight:700; margin:.35rem 0 .6rem; padding:0;}
.bc-hero p{color:var(--ink-2); font-size:1.02rem; line-height:1.6; margin:0; max-width:620px;}
.bc-facts{display:flex; gap:.6rem; flex-wrap:wrap; justify-content:flex-end;}
.bc-fact{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.6rem .85rem; min-width:118px;}
.bc-fact b{display:block; font-size:1.05rem; color:var(--ink);}
.bc-fact small{color:var(--muted); font-size:.74rem;}

/* tabs */
div[data-baseweb="tab-list"]{gap:.25rem; border-bottom:1px solid var(--line);}
button[data-baseweb="tab"]{padding:.55rem .95rem !important; border-radius:10px 10px 0 0;}
button[data-baseweb="tab"] p{font-weight:600 !important; font-size:.92rem !important; color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--brand-ink);}
div[data-baseweb="tab-highlight"]{background:var(--brand) !important; height:2px;}
div[data-baseweb="tab-border"]{display:none;}

/* cards */
.bc-card{background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:1.1rem 1.2rem;}
.bc-kpis{display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin:.4rem 0 1.1rem;}
.bc-kpi .l{font-size:.78rem; color:var(--muted); font-weight:500;}
.bc-kpi .v{font-size:1.65rem; font-weight:700; color:var(--ink); margin:.25rem 0 .1rem; letter-spacing:-.02em;}
.bc-kpi .v small{font-size:.85rem; font-weight:500; color:var(--muted); margin-left:.25rem;}
.bc-kpi .s{font-size:.76rem; color:var(--muted);}
.bc-range{position:relative; height:6px; border-radius:6px; background:var(--line-2); margin:.55rem 0 .3rem;}
.bc-range i{position:absolute; top:0; bottom:0; border-radius:6px; background:#9fd9d1;}
.bc-range em{position:absolute; top:-3px; width:3px; height:12px; border-radius:2px; background:var(--brand);}
.bc-section{font-size:.74rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  margin:1.6rem 0 .6rem;}
.bc-site h3{margin:0; font-size:1.25rem;}
.bc-site .meta{color:var(--muted); font-size:.84rem; margin:.15rem 0 .6rem;}
.bc-site p{color:var(--ink-2); font-size:.92rem; line-height:1.55; margin:0;}
.bc-legend{display:flex; flex-direction:column; gap:.42rem; margin-top:.2rem;}
.bc-legend .row{display:grid; grid-template-columns:14px 1fr auto auto; gap:.2rem .6rem; align-items:center; font-size:.88rem;}
.bc-legend .bar{grid-column:2 / 5; height:5px; background:var(--line-2); border-radius:4px; overflow:hidden; margin-bottom:.25rem;}
.bc-legend .bar i{display:block; height:100%; border-radius:4px;}
.bc-legend .sw{width:12px; height:12px; border-radius:3px;}
.bc-legend .n{color:var(--ink);} .bc-legend .a{font-variant-numeric:tabular-nums; color:var(--ink); font-weight:600;}
.bc-legend .p{font-variant-numeric:tabular-nums; color:var(--muted); font-size:.8rem; min-width:42px; text-align:right;}
.bc-legend .bc{font-size:.66rem; font-weight:600; color:var(--brand); background:var(--accent-soft);
  padding:1px 6px; border-radius:999px; margin-left:.35rem;}
.bc-badge{display:inline-flex; align-items:center; gap:.35rem; font-size:.74rem; font-weight:600; padding:3px 10px;
  border-radius:999px; background:var(--warn-bg); color:var(--warn-ink); border:1px solid #f3d9a8;}
.bc-note{font-size:.8rem; color:var(--muted); line-height:1.5;}
.bc-map iframe{border-radius:var(--radius); border:1px solid var(--line) !important;}
.bc-hint{font-size:.78rem; color:var(--muted); margin-top:.35rem;}

/* methodology */
.bc-steps{display:grid; grid-template-columns:repeat(5,1fr); gap:.8rem; margin:.4rem 0 1.4rem;}
.bc-step{background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:1rem;}
.bc-step .k{font-family:'JetBrains Mono', monospace; font-size:.72rem; color:var(--accent); font-weight:500;}
.bc-step h4{margin:.3rem 0 .35rem; font-size:.98rem;}
.bc-step p{margin:0; font-size:.82rem; line-height:1.5; color:var(--ink-2);}
.bc-doc{max-width:880px;}
.bc-doc h3{font-size:1.15rem; margin:1.6rem 0 .5rem;}
.bc-doc p, .bc-doc li{color:var(--ink-2); font-size:.94rem; line-height:1.65;}
.bc-table{width:100%; border-collapse:collapse; font-size:.88rem; background:var(--card); border:1px solid var(--line);
  border-radius:var(--radius); overflow:hidden;}
.bc-table th{background:#f8fafb; text-align:left; font-weight:600; color:var(--ink-2); padding:.6rem .8rem;
  border-bottom:1px solid var(--line); font-size:.78rem; text-transform:uppercase; letter-spacing:.05em;}
.bc-table td{padding:.6rem .8rem; border-bottom:1px solid var(--line-2); color:var(--ink); vertical-align:top;}
.bc-table tr:last-child td{border-bottom:none;}
.bc-table td.num{text-align:right; font-variant-numeric:tabular-nums;}
.bc-formula{font-family:'JetBrains Mono', monospace; font-size:.86rem; background:var(--card); border:1px solid var(--line);
  border-left:3px solid var(--accent); border-radius:8px; padding:.8rem 1rem; color:var(--ink); margin:.6rem 0;}
.bc-refs li{font-size:.84rem;}
.bc-footer{margin-top:3rem; padding-top:1.2rem; border-top:1px solid var(--line); color:var(--muted); font-size:.8rem;
  display:flex; justify-content:space-between;}
.bc-footer a{color:var(--ink-2);}

/* widgets */
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stSegmentedControl"] label
  {font-size:.78rem !important; color:var(--muted) !important; font-weight:500 !important;}
div[data-baseweb="select"] > div{border-radius:10px; border-color:var(--line); background:var(--card);}
.stButton button[kind="primary"]{background:var(--brand); border:none; border-radius:10px; font-weight:600;}
.stButton button[kind="primary"]:hover{background:var(--brand-ink);}
div[data-testid="stExpander"]{border:1px solid var(--line); border-radius:var(--radius); background:var(--card);}
@media (max-width: 900px){
  .bc-hero{grid-template-columns:1fr;} .bc-facts{justify-content:flex-start;}
  .bc-kpis{grid-template-columns:repeat(2,1fr);} .bc-steps{grid-template-columns:1fr 1fr;}
  .block-container{padding:1rem 1rem 2rem;}
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- helpers
def fmt(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "–"
    a = abs(x)
    if a >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if a >= 1e4:
        return f"{x / 1e3:,.1f}k"
    if a >= 100:
        return f"{x:,.0f}"
    return f"{x:,.{max(digits, 1)}f}"


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def rgba_uri(rgba: np.ndarray) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fit_view(bounds, width_px: int = 820) -> tuple[list[float], int]:
    (s, w), (n, e) = bounds
    span = max(e - w, (n - s) * 1.25, 1e-4)
    zoom = int(math.floor(math.log2(width_px * 360 / (256 * span))))
    return [(s + n) / 2, (w + e) / 2], max(3, min(zoom, 16))


def make_map(bounds, height: int = 540):
    center, zoom = fit_view(bounds)
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, zoom_control=True)
    folium.TileLayer(ESRI, attr=ESRI_ATTR, name="Satellite", max_zoom=19).add_to(m)
    Fullscreen(position="topright").add_to(m)
    return m


def overlay(m, uri, bounds, opacity=1.0):
    folium.raster_layers.ImageOverlay(uri, bounds=bounds, opacity=opacity, interactive=False, zindex=2).add_to(m)


def outline(m, bounds):
    (s, w), (n, e) = bounds
    folium.Rectangle([[s, w], [n, e]], color="#ffffff", weight=1.2, fill=False, dash_array="4 4", opacity=0.8).add_to(m)


@st.cache_data
def list_sites() -> dict[str, Path]:
    out = {}
    for f in sorted(DEMO_DIR.glob("*/meta.json")):
        out[json.loads(f.read_text())["title"]] = f.parent
    return out


def best_areas(report: dict) -> dict[str, float]:
    adj = report.get("error_adjusted_areas_ha")
    return {k: (adj[k]["adjusted_ha"] if adj else report["areas_ha"][k]) for k in CLS}


def range_bar(p05: float, mean: float, p95: float) -> str:
    hi = p95 * 1.15 if p95 > 0 else 1
    lo_pct, hi_pct, m_pct = 100 * p05 / hi, 100 * p95 / hi, 100 * mean / hi
    return (f'<div class="bc-range"><i style="left:{lo_pct:.1f}%;width:{max(hi_pct - lo_pct, 1):.1f}%"></i>'
            f'<em style="left:calc({m_pct:.1f}% - 1px)"></em></div>')


def kpi_cards(report: dict) -> str:
    cb = report["carbon"]
    s, q = cb["total_stock_tCO2e"], cb["total_sequestration_tCO2e_per_yr"]
    a = best_areas(report)
    bc_area = sum(a[k] for k in BLUE_CARBON_KEYS)
    v = cb["indicative_annual_value_usd"]
    parts = [
        ("Blue carbon habitat", fmt(bc_area, 1), "ha",
         " · ".join(f"{CLS[k].name} {fmt(a[k], 1)}" for k in BLUE_CARBON_KEYS if a[k] > 0.05) or "none detected", ""),
        ("Carbon stock", fmt(s["mean"]), "tCO₂e", f"90% interval {fmt(s['p05'])} – {fmt(s['p95'])}",
         range_bar(s["p05"], s["mean"], s["p95"])),
        ("Annual sequestration", fmt(q["mean"], 1), "tCO₂e / yr", f"90% interval {fmt(q['p05'], 1)} – {fmt(q['p95'], 1)}",
         range_bar(q["p05"], q["mean"], q["p95"])),
        ("Indicative credit value", "$" + fmt(v["mid"]), "/ yr", f"${fmt(v['low'])} – ${fmt(v['high'])} at $15–40 per tCO₂e", ""),
    ]
    cards = "".join(
        f'<div class="bc-card bc-kpi"><div class="l">{label}</div><div class="v">{val}<small>{u}</small></div>{bar}'
        f'<div class="s">{sub}</div></div>' for label, val, u, sub, bar in parts)
    return f'<div class="bc-kpis">{cards}</div>'


def legend_html(report: dict) -> str:
    a = best_areas(report)
    tot = sum(a.values()) or 1
    rows = []
    for c in CLASSES:
        tag = '<span class="bc">blue carbon</span>' if c.blue_carbon else ""
        rows.append(f'<div class="row"><span class="sw" style="background:{c.color}"></span>'
                    f'<span class="n">{c.name}{tag}</span><span class="a">{fmt(a[c.key], 1)} ha</span>'
                    f'<span class="p">{100 * a[c.key] / tot:.1f}%</span><span></span>'
                    f'<span class="bar"><i style="width:{max(100 * a[c.key] / tot, 0.4 if a[c.key] > 0 else 0):.1f}%;'
                    f'background:{c.color}"></i></span></div>')
    return f'<div class="bc-legend">{"".join(rows)}</div>'


def carbon_table(report: dict) -> str:
    rows = []
    for k, v in report["carbon"]["classes"].items():
        s, q = v["stock_tCO2e"], v["sequestration_tCO2e_per_yr"]
        rows.append(f'<tr><td><span class="sw" style="display:inline-block;width:10px;height:10px;border-radius:3px;'
                    f'background:{CLS[k].color};margin-right:8px"></span>{CLS[k].name}</td>'
                    f'<td class="num">{fmt(v["area_ha"], 1)}</td><td class="num">{fmt(s["mean"])}</td>'
                    f'<td class="num">{fmt(s["p05"])} – {fmt(s["p95"])}</td><td class="num">{fmt(q["mean"], 1)}</td></tr>')
    if not rows:
        rows = ['<tr><td colspan="5">No blue carbon habitat detected.</td></tr>']
    return ('<table class="bc-table"><thead><tr><th>Habitat</th><th style="text-align:right">Area (ha)</th>'
            '<th style="text-align:right">Stock (tCO₂e)</th><th style="text-align:right">90% interval</th>'
            '<th style="text-align:right">Sequestration (tCO₂e/yr)</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def model_panel(model: dict | None) -> None:
    if not model:
        return
    t = model.get("test") or {}
    badge = ('<span class="bc-badge">● Pilot model · single-site training data</span>' if model.get("pilot")
             else '<span class="bc-badge" style="background:#e6f5f3;color:#0e5a67;border-color:#cfe9e5">'
                  '● Multi-site model</span>')
    st.markdown(f'{badge}<div style="margin:.7rem 0 .2rem;font-weight:600">{model.get("name", "model")}</div>'
                f'<div class="bc-note">{model["arch"]} · {model["encoder"]} encoder · 14 spectral inputs</div>',
                unsafe_allow_html=True)
    if t:
        rows = "".join(
            f'<tr><td>{CLS[k].name}</td><td class="num">{v:.2f}</td><td class="num">{t["f1"][k]:.2f}</td>'
            f'<td class="num">{t["support_px"][k]:,}</td></tr>' for k, v in t["iou"].items() if v is not None)
        st.markdown(
            f'<div style="display:flex;gap:1.6rem;margin:.9rem 0">'
            f'<div><div class="bc-note">mIoU</div><div style="font-size:1.35rem;font-weight:700">{t["mIoU"]:.2f}</div></div>'
            f'<div><div class="bc-note">Macro F1</div><div style="font-size:1.35rem;font-weight:700">{t["macro_f1"]:.2f}</div></div>'
            f'<div><div class="bc-note">Cohen\'s κ</div><div style="font-size:1.35rem;font-weight:700">{t["kappa"]:.2f}</div></div></div>'
            f'<table class="bc-table"><thead><tr><th>Class</th><th style="text-align:right">IoU</th>'
            f'<th style="text-align:right">F1</th><th style="text-align:right">Test pixels</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    notes = " ".join(x for x in [model.get("evaluation") and f"Evaluated with {model['evaluation']}.",
                                 model.get("training_data") and f"Trained on {model['training_data']}."] if x)
    if notes:
        st.markdown(f'<div class="bc-note" style="margin-top:.7rem">{notes}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- chrome
st.markdown(
    f'<div class="bc-top"><div class="bc-brand">{LOGO}<div class="bc-word">Blue<span>Carbon</span></div>'
    f'<span class="bc-tag">v2</span></div><div class="bc-links"><a href="{REPO}" target="_blank">GitHub</a>'
    f'<a href="{REPO}/blob/main/docs/METHODOLOGY.md" target="_blank">Docs</a>'
    f'<a href="https://colab.research.google.com/github/yanicksanchez14-creator/bluecarbon-ai/blob/main/notebooks/'
    f'train_colab.ipynb" target="_blank">Train the model</a></div></div>'
    '<div class="bc-hero"><div><div class="bc-eyebrow">Coastal carbon intelligence</div>'
    "<h1>Mapping the ocean's carbon sinks from space</h1>"
    "<p>BlueCarbon segments mangrove, salt marsh and seagrass in 10&nbsp;m Sentinel-2 imagery with a deep "
    "learning model, then estimates the carbon they store, with the uncertainty shown, not hidden.</p></div>"
    '<div class="bc-facts"><div class="bc-fact"><b>10 m</b><small>Sentinel-2 resolution</small></div>'
    '<div class="bc-fact"><b>6 classes</b><small>3 blue carbon habitats</small></div>'
    '<div class="bc-fact"><b>IPCC Tier 1</b><small>Monte Carlo intervals</small></div></div></div>',
    unsafe_allow_html=True,
)

tab_explore, tab_analyze, tab_method = st.tabs(["Explore sites", "Analyze an area", "Methodology"])

# ----------------------------------------------------------------------------- explore
VIEWS = {"Habitats": "classes", "Blue carbon": "bluecarbon", "Satellite": None, "False color": "falsecolor"}

with tab_explore:
    sites = list_sites()
    if not sites:
        st.warning("No demo data found in demo_data/.")
    else:
        c1, c2, c3 = st.columns([1.3, 1.6, 1])
        title = c1.selectbox("Site", list(sites))
        view = c2.segmented_control("Layer", list(VIEWS), default="Habitats", key="view") or "Habitats"
        opacity = c3.slider("Overlay opacity", 0.0, 1.0, 0.8, 0.05)
        d = sites[title]
        meta = json.loads((d / "meta.json").read_text())
        report = meta["report"] if meta["kind"] == "single" else meta["t1"]

        st.markdown(kpi_cards(report), unsafe_allow_html=True)

        left, right = st.columns([0.64, 0.36], gap="large")
        with left:
            m = make_map(meta["bounds"])
            tag = "" if meta["kind"] == "single" else "_t1"
            if view == "False color":
                overlay(m, png_uri(d / f"falsecolor{tag}.png"), meta["bounds"], 1.0)
            else:
                overlay(m, png_uri(d / f"rgb{tag}.png"), meta["bounds"], 1.0)
                if VIEWS[view]:
                    overlay(m, png_uri(d / f"{VIEWS[view]}{tag}.png"), meta["bounds"], opacity)
            outline(m, meta["bounds"])
            st.markdown('<div class="bc-map">', unsafe_allow_html=True)
            st_folium(m, height=540, use_container_width=True, returned_objects=[], key=f"map_{title}_{view}_{opacity}")
            st.markdown("</div>", unsafe_allow_html=True)
            hints = {
                "Habitats": "Model output over the Sentinel-2 composite. Toggle layers above.",
                "Blue carbon": "Only mangrove, salt marsh and seagrass: the habitats counted in the carbon estimate.",
                "Satellite": "Cloud-masked Sentinel-2 median composite (true color).",
                "False color": "Near-infrared / red / green. Healthy vegetation shows up red, the same view used "
                               "to hand-label marsh in QGIS.",
            }
            st.markdown(f'<div class="bc-hint">{hints[view]}</div>', unsafe_allow_html=True)
        with right:
            where = " · ".join(x for x in [meta.get("region"), meta.get("period")] if x)
            st.markdown(f'<div class="bc-site"><h3>{meta["title"]}</h3><div class="meta">{where}</div>'
                        f'<p>{meta.get("description", "")}</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="bc-section">Habitat composition</div>', unsafe_allow_html=True)
            st.markdown(legend_html(report), unsafe_allow_html=True)
            if report.get("error_adjusted_areas_ha"):
                st.markdown('<div class="bc-note">Areas are error-adjusted with the model\'s held-out confusion '
                            "matrix (Olofsson et al., 2014), which corrects the raw map's over- and under-counting.</div>",
                            unsafe_allow_html=True)

        b1, b2 = st.columns([0.58, 0.42], gap="large")
        with b1:
            st.markdown('<div class="bc-section">Carbon by habitat</div>', unsafe_allow_html=True)
            st.markdown(carbon_table(report), unsafe_allow_html=True)
            st.markdown(f'<div class="bc-note" style="margin-top:.6rem">{report["carbon"]["method"]}. Stock = soil '
                        "organic carbon to 1 m + living biomass. Credit value uses annual sequestration only. Tier 1 "
                        "values are global averages: an order-of-magnitude screen, not a crediting methodology.</div>",
                        unsafe_allow_html=True)
            if meta["kind"] == "change":
                ch = meta["change"]["change"]
                st.markdown('<div class="bc-section">Change</div>', unsafe_allow_html=True)
                st.markdown(f"Net blue carbon stock change: **{fmt(ch['net_stock_change_tCO2e']['mean'])} tCO₂e**")
        with b2:
            st.markdown('<div class="bc-section">Model</div>', unsafe_allow_html=True)
            with st.container(border=True):
                model_panel(meta.get("model"))


# ----------------------------------------------------------------------------- analyze
def secret(name: str) -> str | None:
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    v = v or os.environ.get(name)
    if isinstance(v, dict) or hasattr(v, "to_dict"):
        v = json.dumps(dict(v))
    return v


@st.cache_resource
def get_model():
    from bluecarbon.model import load_checkpoint

    url, path = secret("MODEL_URL"), DEFAULT_MODEL
    if url:
        import urllib.request

        path = Path(tempfile.gettempdir()) / "bluecarbon_model.pt"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
    return load_checkpoint(path)


@st.cache_resource
def init_gee(sa_json: str):
    from bluecarbon import gee

    gee.init(secret("GEE_PROJECT") or json.loads(sa_json).get("project_id") or CFG.project, sa_json)
    return True


with tab_analyze:
    sa = secret("GEE_SERVICE_ACCOUNT")
    if not sa:
        st.markdown(
            """
<div class="bc-card" style="max-width:820px;margin-top:.6rem">
  <div class="bc-eyebrow">On-demand analysis</div>
  <h3 style="margin:.35rem 0 .4rem">Map any coastline in about a minute</h3>
  <p style="color:var(--ink-2);line-height:1.6;margin:0 0 .9rem">Draw a box anywhere on Earth, pick a season, and
  BlueCarbon pulls a fresh cloud-free Sentinel-2 composite from Google Earth Engine, runs the model and returns a habitat
  map, carbon report and downloadable GeoTIFF.</p>
  <div class="bc-note">This deployment hasn't been connected to Earth Engine yet, so the live pipeline is switched off.
  Everything in <b>Explore sites</b> works without it. The same analysis runs locally with
  <code>bluecarbon scene --bbox … --start … --end … -m model.pt</code>.</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="bc-note" style="margin:.4rem 0 .8rem">Draw a rectangle over a coastline '
                    f"(up to {MAX_AREA_KM2} km²), choose the season, then run.</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        start = c1.date_input("Start", pd.Timestamp("2024-05-01"))
        end = c2.date_input("End", pd.Timestamp("2024-09-30"))
        m = folium.Map(location=[32.78, -117.22], zoom_start=12, tiles=None)
        folium.TileLayer(ESRI, attr=ESRI_ATTR, name="Satellite").add_to(m)
        Draw(draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False,
                           "circlemarker": False, "rectangle": {"shapeOptions": {"color": "#14a3a0"}}},
             edit_options={"edit": False}).add_to(m)
        out = st_folium(m, height=480, use_container_width=True, key="draw")
        feat = (out or {}).get("last_active_drawing")
        c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        run = c3.button("Run analysis", type="primary", disabled=feat is None, width="stretch")
        if run and feat:
            import rasterio
            from rasterio.warp import Resampling

            from bluecarbon import gee
            from bluecarbon.demo import _to_mercator
            from bluecarbon.features import S2_BANDS
            from bluecarbon.predict import predict_array
            from bluecarbon.report import scene_report
            from bluecarbon.viz import class_rgba, true_color

            coords = np.array(feat["geometry"]["coordinates"][0])
            bbox = [coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()]
            km2 = ((bbox[2] - bbox[0]) * 111.32 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
                   * (bbox[3] - bbox[1]) * 110.57)
            if km2 > MAX_AREA_KM2:
                st.error(f"That box is {km2:,.0f} km². Please draw one under {MAX_AREA_KM2} km².")
                st.stop()
            init_gee(sa)
            model, norm, ck = get_model()
            with st.status("Running pipeline…", expanded=True) as status:
                tmp = Path(tempfile.mkdtemp())
                region = gee.bbox_geometry(bbox)
                n = gee.image_count(region, str(start), str(end), CFG)
                st.write(f"{n} Sentinel-2 scenes found. Building a cloud-masked median composite…")
                if n == 0:
                    status.update(label="No imagery for that period", state="error")
                    st.stop()
                bar = st.progress(0.0)
                gee.download(gee.s2_composite(region, str(start), str(end), CFG), bbox, tmp / "image.tif", CFG,
                             "uint16", 0, S2_BANDS, progress=bar.progress)
                st.write("Segmenting habitats…")
                with rasterio.open(tmp / "image.tif") as src:
                    bands, prof = src.read(), src.profile
                cls, _ = predict_array(model, norm, bands, tile=CFG.predict.tile, overlap=CFG.predict.overlap)
                prof.update(count=1, dtype="uint8", nodata=255)
                with rasterio.open(tmp / "pred.tif", "w", **prof) as dst:
                    dst.write(cls, 1)
                rep = scene_report(tmp / "pred.tif", CFG.carbon, ck["metrics"].get("test_confusion"))
                status.update(label="Analysis complete", state="complete", expanded=False)
            mb, bnds = _to_mercator(tmp / "image.tif", list(range(1, 11)), Resampling.bilinear, 0)
            mc, _ = _to_mercator(tmp / "pred.tif", [1], Resampling.nearest, 255)
            st.markdown(kpi_cards(rep), unsafe_allow_html=True)
            l2, r2 = st.columns([0.64, 0.36], gap="large")
            with l2:
                m2 = make_map(bnds, 520)
                overlay(m2, rgba_uri(true_color(mb)), bnds)
                overlay(m2, rgba_uri(class_rgba(mc[0], 200)), bnds, 0.8)
                st_folium(m2, height=520, use_container_width=True, returned_objects=[], key="result")
            with r2:
                st.markdown('<div class="bc-section">Habitat composition</div>', unsafe_allow_html=True)
                st.markdown(legend_html(rep), unsafe_allow_html=True)
                d1, d2 = st.columns(2)
                d1.download_button("Habitat GeoTIFF", (tmp / "pred.tif").read_bytes(), "bluecarbon_habitats.tif",
                                   width="stretch")
                d2.download_button("Report (JSON)", json.dumps(rep, indent=2), "bluecarbon_report.json",
                                   width="stretch")
            st.markdown(carbon_table(rep), unsafe_allow_html=True)

# ----------------------------------------------------------------------------- methodology
with tab_method:
    steps = [
        ("01", "Acquire", "Sentinel-2 L2A surface reflectance, masked with Cloud Score+ and reduced to a seasonal median."),
        ("02", "Label", "Reference labels fused from ESA WorldCover, Murray tidal flats and the Allen Coral Atlas."),
        ("03", "Learn", "U-Net with a ResNet encoder on 10 bands + 4 spectral indices, trained with Dice + CE loss."),
        ("04", "Map", "Overlapping tiles blended with a smooth window and flip test-time augmentation."),
        ("05", "Account", "Error-adjusted areas × IPCC Tier 1 carbon factors, with Monte Carlo 90% intervals."),
    ]
    st.markdown('<div class="bc-steps">' + "".join(
        f'<div class="bc-step"><div class="k">{k}</div><h4>{t}</h4><p>{p}</p></div>' for k, t, p in steps) + "</div>",
        unsafe_allow_html=True)

    cc = CFG.carbon.classes
    carbon_rows = "".join(
        f'<tr><td>{CLS[k].name}</td><td class="num">{v.soil[1]:.0f} <span class="bc-note">({v.soil[0]:.0f}–{v.soil[2]:.0f})</span></td>'
        f'<td class="num">{v.biomass[1]:.0f} <span class="bc-note">({v.biomass[0]:g}–{v.biomass[2]:g})</span></td>'
        f'<td class="num">{v.accumulation[1]:.2f} <span class="bc-note">({v.accumulation[0]:g}–{v.accumulation[2]:g})</span></td></tr>'
        for k, v in cc.items())

    st.markdown(
        f"""
<div class="bc-doc">
<h3>Imagery</h3>
<p>Each scene is a per-pixel median of every Sentinel-2 L2A acquisition in the chosen season
(<code>COPERNICUS/S2_SR_HARMONIZED</code>) after removing cloud and shadow with Google's Cloud Score+
(<code>cs_cdf ≥ 0.6</code>). Ten bands (B2–B8A, B11, B12) are exported at 10&nbsp;m in the local UTM zone,
so every pixel has a true ground area. The model also receives four indices: NDVI (vegetation), NDWI and
MNDWI (water), and NDMI (canopy moisture, which separates mangrove from dry upland).</p>

<h3>Reference labels</h3>
<p>Training labels come from independent, peer-reviewed global products, not from thresholds on the
model's own input bands. Pixels within one pixel of a class boundary are excluded, because edges are where these
products are least reliable. Local survey polygons (for example eelgrass surveys) can override any source.</p>
<table class="bc-table"><thead><tr><th>Class</th><th>Source</th><th>Rule</th></tr></thead><tbody>
<tr><td>Open water · Other land</td><td>ESA WorldCover 2021 (10 m)</td><td>Classes 80 · 10–60, 100</td></tr>
<tr><td>Mangrove</td><td>ESA WorldCover 2021</td><td>Class 95</td></tr>
<tr><td>Salt marsh</td><td>ESA WorldCover + NASADEM</td><td>Herbaceous wetland (90) below 5 m elevation</td></tr>
<tr><td>Tidal flat</td><td>Murray et al., global intertidal</td><td>Tidal flat classification</td></tr>
<tr><td>Seagrass</td><td>Allen Coral Atlas benthic map</td><td>Seagrass class (tropical coverage)</td></tr>
</tbody></table>

<h3>Model and evaluation</h3>
<p>A U-Net with a ResNet-34 encoder (<code>segmentation-models-pytorch</code>) and a 14-channel input stem, trained
with cross-entropy plus Dice loss and square-root inverse-frequency class weights, AdamW with a one-cycle
schedule, mixed precision and early stopping on validation mIoU.</p>
<p>Evaluation is built so the model can't score well by memorizing. Chips never overlap, whole 5&nbsp;km blocks are
assigned to a single split, and two complete estuaries (Moreton Bay, Tampa Bay) are never seen in training. The
headline metrics are per-class IoU and F1. Overall accuracy is reported but not emphasized: a scene that is 70% water
can score 90% accuracy while missing every marsh pixel.</p>

<h3>Area and carbon accounting</h3>
<p>Raw pixel counts are biased toward whatever the model over-predicts. Areas are therefore corrected with the
stratified estimator of Olofsson et al. (2014), using the held-out confusion matrix. Carbon is then estimated per
habitat:</p>
<div class="bc-formula">stock (tCO₂e) = area (ha) × [ soil C to 1 m + living biomass C ] (tC/ha) × 44/12<br>
sequestration (tCO₂e/yr) = area (ha) × soil C accumulation (tC/ha/yr) × 44/12</div>
<p>Each coefficient is drawn from a triangular distribution over its published range, jointly with the area
uncertainty (5,000 Monte Carlo draws), and results are reported as a mean with a 90% interval. Tier 1 defaults
(IPCC 2013 Wetlands Supplement):</p>
<table class="bc-table"><thead><tr><th>Habitat</th><th style="text-align:right">Soil C, tC/ha</th>
<th style="text-align:right">Biomass C, tC/ha</th><th style="text-align:right">Accumulation, tC/ha/yr</th></tr></thead>
<tbody>{carbon_rows}</tbody></table>
<p>The indicative credit value is based on annual sequestration at $15–40 per tCO₂e. Standing stock is not creditable
on its own.</p>

<h3>Limitations</h3>
<ul>
<li>Tier 1 factors are global averages. They are suited to screening and prioritization, not to issuing credits,
which requires field-measured stocks.</li>
<li>Seagrass is spectrally close to water, and global seagrass labels cover only tropical reefs. Temperate
meadows need local survey data.</li>
<li>Tides change what is exposed in intertidal zones, and a median composite averages across tidal states.</li>
<li>Reference products carry their own errors, which the model partly learns. The confidence intervals treat pixels
as independent samples, so they understate the true uncertainty.</li>
</ul>

<h3>References</h3>
<ul class="bc-refs">
<li>IPCC (2014). <i>2013 Supplement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Wetlands</i>, Chapter 4.</li>
<li>Olofsson, P. et al. (2014). Good practices for estimating area and assessing accuracy of land change. <i>Remote Sensing of Environment</i> 148.</li>
<li>Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200.</li>
<li>Murray, N. J. et al. (2019). The global distribution and trajectory of tidal flats. <i>Nature</i> 565.</li>
<li>Allen Coral Atlas (2022). Imagery, maps and monitoring of the world's tropical coral reefs.</li>
<li>Pasquarella, V. et al. (2023). Cloud Score+: comprehensive cloud and cloud-shadow detection for Sentinel-2.</li>
</ul>
</div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="bc-footer"><span>BlueCarbon v2 · Built by Yanick Sanchez</span>'
    f'<span><a href="{REPO}" target="_blank">Source on GitHub</a> · MIT License</span></div>',
    unsafe_allow_html=True,
)
