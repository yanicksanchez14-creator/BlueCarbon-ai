"""BlueCarbon-AI demo app.

Runs with zero credentials on the bundled demo sites. If Earth Engine credentials are
configured (st.secrets["GEE_SERVICE_ACCOUNT"] = service-account JSON), the "Analyze an area"
tab maps any coastline on demand.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import altair as alt
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
MAX_AREA_KM2 = 60

st.set_page_config(page_title="BlueCarbon-AI", page_icon="🌊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1400px;}
      .bc-hero h1 {font-size: 2.1rem; margin-bottom: .1rem;}
      .bc-hero p {color: #5b6b7a; margin-top: 0; font-size: 1.02rem;}
      .bc-legend span {display:inline-flex; align-items:center; margin-right:14px; font-size:.88rem;}
      .bc-legend i {width:12px; height:12px; border-radius:3px; display:inline-block; margin-right:6px;}
      .bc-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem;
                 background:#fff4e5; color:#8a5300; border:1px solid #f5c97a;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

CFG = load_config(ROOT / "configs" / "default.yaml")
CLS = {c.key: c for c in CLASSES}


# ----------------------------------------------------------------------------- helpers
def fmt(x: float, unit: str = "") -> str:
    if x is None:
        return "–"
    a = abs(x)
    s = f"{x / 1e6:,.2f}M" if a >= 1e6 else f"{x / 1e3:,.1f}k" if a >= 1e4 else f"{x:,.0f}" if a >= 100 else f"{x:,.1f}"
    return f"{s} {unit}".strip()


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def rgba_uri(rgba: np.ndarray) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def legend(keys=None):
    items = [c for c in CLASSES if keys is None or c.key in keys]
    html = "".join(f'<span><i style="background:{c.color}"></i>{c.name}</span>' for c in items)
    st.markdown(f'<div class="bc-legend">{html}</div>', unsafe_allow_html=True)


def base_map(bounds, height=560):
    (s, w), (n, e) = bounds
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer("Esri.WorldImagery", name="Esri World Imagery").add_to(m)
    folium.TileLayer("CartoDB.Positron", name="Light basemap").add_to(m)
    m.fit_bounds([[s, w], [n, e]])
    Fullscreen().add_to(m)
    return m


def overlay(m, uri, bounds, name, opacity=1.0, show=True):
    folium.raster_layers.ImageOverlay(uri, bounds=bounds, name=name, opacity=opacity, show=show,
                                      interactive=False, zindex=2).add_to(m)


@st.cache_data
def list_sites():
    out = {}
    for d in sorted(DEMO_DIR.glob("*/meta.json")):
        meta = json.loads(d.read_text())
        out[meta["title"]] = d.parent
    return out


def area_frame(report: dict) -> pd.DataFrame:
    rows = []
    adj = report.get("error_adjusted_areas_ha")
    for c in CLASSES:
        r = {"Habitat": c.name, "key": c.key, "Mapped (ha)": report["areas_ha"][c.key],
             "Blue carbon": "yes" if c.blue_carbon else ""}
        if adj:
            r["Adjusted (ha)"] = adj[c.key]["adjusted_ha"]
            r["± 95% CI (ha)"] = adj[c.key]["ci95_ha"]
        rows.append(r)
    return pd.DataFrame(rows)


def area_chart(df: pd.DataFrame):
    col = "Adjusted (ha)" if "Adjusted (ha)" in df else "Mapped (ha)"
    d = df[df[col] > 0].copy()
    d["color"] = d["key"].map(lambda k: CLS[k].color)
    return (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=3, height=18)
        .encode(
            y=alt.Y("Habitat:N", sort="-x", title=None),
            x=alt.X(f"{col}:Q", title="Area (ha)"),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=["Habitat", alt.Tooltip(f"{col}:Q", format=",.1f")],
        )
        .properties(height=36 * len(d) + 20)
    )


def carbon_panel(report: dict):
    cb = report["carbon"]
    s, q = cb["total_stock_tCO2e"], cb["total_sequestration_tCO2e_per_yr"]
    bc_area = sum(report.get("error_adjusted_areas_ha", {}).get(k, {}).get("adjusted_ha", report["areas_ha"][k])
                  for k in BLUE_CARBON_KEYS)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Blue carbon habitat", fmt(bc_area, "ha"))
    c2.metric("Carbon stock", fmt(s["mean"], "tCO₂e"), help=f"90% interval {fmt(s['p05'])} – {fmt(s['p95'])} tCO₂e")
    c3.metric("Sequestration", fmt(q["mean"], "tCO₂e/yr"),
              help=f"90% interval {fmt(q['p05'])} – {fmt(q['p95'])} tCO₂e per year")
    v = cb["indicative_annual_value_usd"]
    c4.metric("Indicative credit value", f"${fmt(v['mid'])}/yr",
              help=f"${fmt(v['low'])} – ${fmt(v['high'])} per year at $15–40/tCO₂e. Based on annual "
                   "sequestration only, not standing stock. Not a valuation.")
    rows = []
    for k, v in cb["classes"].items():
        rows.append({"Habitat": CLS[k].name, "Area used (ha)": v["area_ha"],
                     "Stock (tCO₂e)": v["stock_tCO2e"]["mean"],
                     "Stock 90% interval": f"{fmt(v['stock_tCO2e']['p05'])} – {fmt(v['stock_tCO2e']['p95'])}",
                     "Sequestration (tCO₂e/yr)": v["sequestration_tCO2e_per_yr"]["mean"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Area used (ha)": st.column_config.NumberColumn(format="%.1f"),
                                    "Stock (tCO₂e)": st.column_config.NumberColumn(format="%.0f"),
                                    "Sequestration (tCO₂e/yr)": st.column_config.NumberColumn(format="%.1f")})
    basis = "error-adjusted" if report.get("carbon_area_basis") == "error_adjusted" else "mapped"
    st.caption(f"{cb['method']}. Areas: {basis}. Tier 1 global defaults give an order-of-magnitude "
               "estimate; carbon crediting requires field-measured stocks.")


def model_card(model: dict | None):
    if not model:
        return
    t = model.get("test") or {}
    pilot = model.get("pilot")
    if pilot:
        st.markdown('<span class="bc-badge">Pilot model: single-bay training data</span>', unsafe_allow_html=True)
    st.markdown(f"**{model.get('name', 'model')}**: {model['arch']} / {model['encoder']}")
    if t:
        c1, c2, c3 = st.columns(3)
        c1.metric("mIoU", f"{t['mIoU']:.2f}")
        c2.metric("Macro F1", f"{t['macro_f1']:.2f}")
        c3.metric("Overall acc.", f"{t['overall_accuracy']:.2f}")
        iou = pd.DataFrame([{"Habitat": CLS[k].name, "IoU": v, "F1": t["f1"][k], "Pixels": t["support_px"][k]}
                            for k, v in t["iou"].items() if v is not None])
        st.dataframe(iou, hide_index=True, width="stretch",
                     column_config={"IoU": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                    "F1": st.column_config.NumberColumn(format="%.2f")})
    if model.get("evaluation"):
        st.caption(f"Evaluation: {model['evaluation']}.")
    if model.get("training_data"):
        st.caption(f"Training data: {model['training_data']}.")


# ----------------------------------------------------------------------------- header
st.markdown(
    '<div class="bc-hero"><h1>🌊 BlueCarbon-AI</h1>'
    "<p>Deep-learning maps of mangrove, salt marsh and seagrass from Sentinel-2, with carbon estimates "
    "that show their uncertainty.</p></div>",
    unsafe_allow_html=True,
)

tab_demo, tab_live, tab_method = st.tabs(["🗺️ Explore sites", "🛰️ Analyze an area", "📖 Methodology"])

# ----------------------------------------------------------------------------- demo tab
with tab_demo:
    sites = list_sites()
    if not sites:
        st.warning("No demo data found in demo_data/.")
    else:
        left, right = st.columns([0.62, 0.38], gap="large")
        with right:
            title = st.selectbox("Site", list(sites))
            d = sites[title]
            meta = json.loads((d / "meta.json").read_text())
            st.write(meta.get("description", ""))
            opacity = st.slider("Habitat layer opacity", 0.0, 1.0, 0.75, 0.05)
        with left:
            m = base_map(meta["bounds"])
            if meta["kind"] == "single":
                overlay(m, png_uri(d / "rgb.png"), meta["bounds"], "Sentinel-2 composite")
                overlay(m, png_uri(d / "classes.png"), meta["bounds"], "All habitats", opacity)
                overlay(m, png_uri(d / "bluecarbon.png"), meta["bounds"], "Blue carbon only", opacity, show=False)
            else:
                overlay(m, png_uri(d / "rgb_t1.png"), meta["bounds"], "Sentinel-2 (after)")
                overlay(m, png_uri(d / "classes_t0.png"), meta["bounds"], "Habitats (before)", opacity, show=False)
                overlay(m, png_uri(d / "classes_t1.png"), meta["bounds"], "Habitats (after)", opacity)
                overlay(m, png_uri(d / "change.png"), meta["bounds"], "Blue carbon change", 1.0, show=False)
            folium.LayerControl(collapsed=False).add_to(m)
            st_folium(m, height=560, use_container_width=True, returned_objects=[], key=f"map_{title}")
            legend()
        with right:
            report = meta["report"] if meta["kind"] == "single" else meta["t1"]
            st.subheader("Habitat area")
            df = area_frame(report)
            st.altair_chart(area_chart(df), width="stretch")
            with st.expander("Area table (mapped vs. error-adjusted)"):
                st.dataframe(df.drop(columns=["key"]), hide_index=True, width="stretch")
                st.caption("Error-adjusted areas correct the map's biases using the model's held-out confusion "
                           "matrix (Olofsson et al., 2014).")
        st.subheader("Carbon")
        carbon_panel(report)
        if meta["kind"] == "change":
            st.subheader("Change")
            ch = meta["change"]["change"]
            st.dataframe(pd.DataFrame([{"Habitat": CLS[k].name, "Δ area (ha)": v}
                                       for k, v in ch["delta_ha"].items()]), hide_index=True)
            n = ch["net_stock_change_tCO2e"]
            st.metric("Net blue carbon stock change", fmt(n["mean"], "tCO₂e"))
        with st.expander("Model card", expanded=False):
            model_card(meta.get("model"))

# ----------------------------------------------------------------------------- live tab
def gee_secret() -> str | None:
    try:
        v = st.secrets.get("GEE_SERVICE_ACCOUNT")
    except Exception:
        v = None
    v = v or os.environ.get("GEE_SERVICE_ACCOUNT")
    if isinstance(v, dict):
        v = json.dumps(dict(v))
    return v


@st.cache_resource
def get_model():
    from bluecarbon.model import load_checkpoint

    url = None
    try:
        url = st.secrets.get("MODEL_URL")
    except Exception:
        pass
    url = url or os.environ.get("MODEL_URL")
    path = DEFAULT_MODEL
    if url:
        import urllib.request

        path = Path(tempfile.gettempdir()) / "bluecarbon_model.pt"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
    return load_checkpoint(path)


@st.cache_resource
def init_gee(secret: str):
    from bluecarbon import gee

    project = None
    try:
        project = st.secrets.get("GEE_PROJECT")
    except Exception:
        pass
    gee.init(project or json.loads(secret).get("project_id") or CFG.project, secret)
    return True


with tab_live:
    secret = gee_secret()
    if not secret:
        st.info(
            "Live analysis needs Google Earth Engine credentials, which aren't configured on this "
            "deployment. Everything in **Explore sites** works without them.\n\n"
            "To enable it, add a service-account key as the `GEE_SERVICE_ACCOUNT` secret; see "
            "the README section *Deploy the live demo*. Locally, the same flow is available as "
            "`bluecarbon scene --bbox ... --start ... --end ... -m model.pt`."
        )
    else:
        st.write(f"Draw a rectangle on a coastline (max {MAX_AREA_KM2} km²), pick a season and run the model.")
        c1, c2, c3 = st.columns([1, 1, 1])
        start = c1.date_input("Start", pd.Timestamp("2024-05-01"))
        end = c2.date_input("End", pd.Timestamp("2024-09-30"))
        m = folium.Map(location=[32.78, -117.22], zoom_start=12, tiles="Esri.WorldImagery")
        Draw(draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False,
                           "circlemarker": False, "rectangle": True}, edit_options={"edit": False}).add_to(m)
        out = st_folium(m, height=480, use_container_width=True, key="draw")
        feat = (out or {}).get("last_active_drawing")
        run = c3.button("Run analysis", type="primary", disabled=feat is None, width="stretch")
        if run and feat:
            coords = np.array(feat["geometry"]["coordinates"][0])
            bbox = [coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()]
            lat = np.radians((bbox[1] + bbox[3]) / 2)
            km2 = (bbox[2] - bbox[0]) * 111.32 * np.cos(lat) * (bbox[3] - bbox[1]) * 110.57
            if km2 > MAX_AREA_KM2:
                st.error(f"That area is {km2:,.0f} km². Please draw something under {MAX_AREA_KM2} km².")
                st.stop()
            import rasterio

            from bluecarbon import gee
            from bluecarbon.features import S2_BANDS
            from bluecarbon.predict import predict_array
            from bluecarbon.report import scene_report
            from bluecarbon.viz import class_rgba, true_color

            init_gee(secret)
            model, norm, ck = get_model()
            with st.status("Running pipeline…", expanded=True) as status:
                tmp = Path(tempfile.mkdtemp())
                region = gee.bbox_geometry(bbox)
                n = gee.image_count(region, str(start), str(end), CFG)
                st.write(f"Found {n} Sentinel-2 scenes; building a cloud-masked median composite…")
                if n == 0:
                    status.update(label="No imagery for that period", state="error")
                    st.stop()
                bar = st.progress(0.0)
                img = gee.s2_composite(region, str(start), str(end), CFG)
                gee.download(img, bbox, tmp / "image.tif", CFG, "uint16", 0, S2_BANDS, progress=bar.progress)
                st.write("Segmenting habitats…")
                with rasterio.open(tmp / "image.tif") as src:
                    bands, prof = src.read(), src.profile
                cls, conf = predict_array(model, norm, bands, tile=CFG.predict.tile, overlap=CFG.predict.overlap)
                prof.update(count=1, dtype="uint8", nodata=255)
                with rasterio.open(tmp / "pred.tif", "w", **prof) as dst:
                    dst.write(cls, 1)
                rep = scene_report(tmp / "pred.tif", CFG.carbon, ck["metrics"].get("test_confusion"))
                status.update(label="Done", state="complete")
            from rasterio.warp import Resampling

            from bluecarbon.demo import _to_mercator

            mb, bnds = _to_mercator(tmp / "image.tif", list(range(1, 11)), Resampling.bilinear, 0)
            mc, _ = _to_mercator(tmp / "pred.tif", [1], Resampling.nearest, 255)
            m2 = base_map(bnds, 520)
            overlay(m2, rgba_uri(true_color(mb)), bnds, "Sentinel-2 composite")
            overlay(m2, rgba_uri(class_rgba(mc[0], 200)), bnds, "Habitats", 0.8)
            folium.LayerControl(collapsed=False).add_to(m2)
            st_folium(m2, height=520, use_container_width=True, returned_objects=[], key="result")
            legend()
            st.altair_chart(area_chart(area_frame(rep)), width="stretch")
            carbon_panel(rep)
            d1, d2 = st.columns(2)
            d1.download_button("Download habitat GeoTIFF", (tmp / "pred.tif").read_bytes(), "bluecarbon_habitats.tif")
            d2.download_button("Download report (JSON)", json.dumps(rep, indent=2), "bluecarbon_report.json")

# ----------------------------------------------------------------------------- method tab
with tab_method:
    st.markdown((ROOT / "docs" / "METHODOLOGY.md").read_text() if (ROOT / "docs" / "METHODOLOGY.md").exists()
                else "See docs/METHODOLOGY.md")
