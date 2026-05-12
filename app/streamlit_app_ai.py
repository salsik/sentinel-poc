from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src.risk_score_ai import compute_risk_score

DATA = ROOT / "data" / "sites_with_ai_predictions.csv"

DATA = ROOT / "data" / "corridor_sites_with_ai_predictions.csv"

st.set_page_config(page_title="Syria Climate & Child Health Risk POC", layout="wide")
st.title("Open Climate & Health Risk Intelligence for Children in Syria")
st.caption("UNICEF Venture Fund proof-of-concept | AI predictions are preliminary and require local validation")

if not DATA.exists():
    st.error(f"Missing file: {DATA}")
    st.info(
        "Expected file: data/sites_with_ai_predictions.csv. "
        "Run the extraction script after creating the AI prediction GeoTIFF."
    )
    st.stop()

raw = pd.read_csv(DATA)
df = compute_risk_score(raw)

if df.empty:
    st.error("The data file loaded, but no rows have valid latitude/longitude values.")
    st.stop()

# Sidebar
st.sidebar.header("POC data status")
st.sidebar.write("**Input file:**", str(DATA.relative_to(ROOT)))
st.sidebar.write("**Rows loaded:**", len(raw))
st.sidebar.write("**Rows mapped:**", len(df))
st.sidebar.caption(
    "Current model: Random Forest trained on Sentinel-2 features and ESA WorldCover labels; "
    "site-level scores are preliminary."
)

# Top metrics
high_count = int((df["risk_level"] == "High").sum())
child_sites = int(df["site_type"].str.lower().isin(["school", "clinic", "hospital", "doctors"]).sum())
mean_risk = df["risk_score"].mean()
mean_conf = pd.to_numeric(df.get("ai_confidence", pd.Series(dtype=float)), errors="coerce").mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mapped sites", len(df))
col2.metric("High-risk sites", high_count)
col3.metric("Schools/clinics", child_sites)
col4.metric("Mean AI confidence", "N/A" if pd.isna(mean_conf) else f"{mean_conf:.2f}")

# Risk map
st.subheader("AI climate-health risk map")

# Center the map around available points
center_lat = float(df["latitude"].mean())
center_lon = float(df["longitude"].mean())

# Color: higher risk = more red. Green channel decreases with risk.
map_df = df.copy()
map_df["risk_red"] = (map_df["risk_score"] * 2.55).clip(0, 255)
map_df["risk_green"] = (180 - map_df["risk_score"] * 1.2).clip(40, 180)
map_df["risk_radius"] = (map_df["risk_score"] * 45 + 80).clip(80, 5000)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[longitude, latitude]",
    get_radius="risk_radius",
    get_fill_color="[risk_red, risk_green, 80, 180]",
    pickable=True,
)
view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=9, pitch=0)
st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            "text": (
                "{site_name}\n"
                "Type: {site_type}\n"
                "Risk: {risk_score} ({risk_level})\n"
                "AI land cover: {ai_landcover_class}\n"
                "AI confidence: {ai_confidence}"
            )
        },
    ),
    use_container_width=True,
)

left, right = st.columns([1.05, 1])

with left:
    st.subheader("Risk components")
    selected = st.selectbox("Select site", df["site_name"].astype(str).tolist())
    row = df[df["site_name"].astype(str) == selected].iloc[0]

    comp = pd.DataFrame(
        {
            "component": [
                "AI land stress",
                "Water stress placeholder",
                "Drought stress placeholder",
                "Heat stress placeholder",
            ],
            "score": [
                float(row.get("vegetation_stress", row.get("ai_land_stress_score", 0.0))),
                float(row.get("water_stress", 0.5)),
                float(row.get("drought_stress", 0.5)),
                float(row.get("heat_stress", 0.5)),
            ],
        }
    )
    st.plotly_chart(
        px.bar(
            comp,
            x="component",
            y="score",
            range_y=[0, 1],
            title=f"{selected}: preliminary risk components",
        ),
        use_container_width=True,
    )
    st.caption(
        "Water, drought, and heat are neutral placeholders unless those columns are added to the CSV. "
        "The current AI signal comes mainly from predicted land-cover / land-stress class."
    )

with right:
    st.subheader("AI prediction summary")
    st.write(f"**{selected}** is classified as **{row.risk_level} risk** with a score of **{row.risk_score:.1f}/100**.")
    st.write(f"**AI land-cover class:** `{row.get('ai_landcover_class', 'unknown')}`")

    conf = row.get("ai_confidence", None)
    if pd.notna(conf):
        st.write(f"**AI confidence:** {float(conf):.2f}")
        if float(conf) < 0.45:
            st.warning("Low confidence: prioritize local validation before using this prediction.")
        elif float(conf) < 0.70:
            st.info("Medium confidence: useful for screening, but still needs local validation.")
        else:
            st.success("Higher confidence: suitable for prioritization discussion, not final operational decisions.")

    if row.risk_level == "High":
        st.warning("Suggested action: prioritize local validation, WASH/nutrition review, and heat-safety planning for this site.")
    elif row.risk_level == "Medium":
        st.info("Suggested action: monitor satellite indicators and collect local feedback before seasonal shocks intensify.")
    elif row.risk_level == "Low":
        st.success("Suggested action: continue monitoring. No immediate high-risk flag in the current POC indicators.")
    else:
        st.info("Suggested action: insufficient confidence/data. Review manually.")

    st.caption("This is not a medical diagnosis or emergency decision system. Human review is required.")

# Charts
st.subheader("Prototype analytics")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    if "ai_landcover_class" in df.columns:
        st.plotly_chart(
            px.histogram(df, x="ai_landcover_class", color="risk_level", title="AI land-cover classes by risk level"),
            use_container_width=True,
        )
with chart_col2:
    if "ai_confidence" in df.columns:
        st.plotly_chart(
            px.histogram(df, x="ai_confidence", nbins=20, title="AI confidence distribution"),
            use_container_width=True,
        )

st.subheader("AI prediction dataset")
preferred_cols = [
    "site_id",
    "site_name",
    "site_type",
    "latitude",
    "longitude",
    "ai_landcover_class",
    "ai_confidence",
    "ai_land_stress_score",
    "risk_score",
    "risk_level",
    #"source",
    #"validation_status",
]
visible_cols = [c for c in preferred_cols if c in df.columns]
other_cols = [c for c in df.columns if c not in visible_cols]
st.dataframe(df[visible_cols + other_cols].sort_values("risk_score", ascending=False), use_container_width=True)
