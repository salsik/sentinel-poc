from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

DATA = ROOT / "data" / "corridor_sites_with_ai_predictions_v2.csv"

st.set_page_config(page_title="Syria Climate & Child Health Risk POC", layout="wide")
st.title("Open Climate & Health Risk Intelligence for Children in Syria")
st.caption("proof-of-concept | AI predictions are preliminary and require local validation")

if not DATA.exists():
    st.error(f"Missing file: {DATA}")
    st.info(
        "Expected file: data/corridor_sites_with_ai_predictions_v2.csv. "
        "Run scripts/predict_sites_from_feature_csv_v2.py first."
    )
    st.stop()

raw = pd.read_csv(DATA)

def to_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = raw.copy()
df = to_numeric(df, [
    "latitude", "longitude", "ai_confidence", "ai_land_stress_score",
    "vegetation_stress", "water_stress", "drought_stress", "heat_stress",
    "site_sensitivity", "ai_climate_health_risk_score", "risk_score",
])

# Backward-compatible fallback if an older CSV is loaded.
if "risk_score" not in df.columns and "ai_climate_health_risk_score" in df.columns:
    df["risk_score"] = (pd.to_numeric(df["ai_climate_health_risk_score"], errors="coerce") * 100).round(1)
if "risk_level" not in df.columns and "ai_risk_category" in df.columns:
    df["risk_level"] = df["ai_risk_category"].astype(str).str.title()

# Keep only mappable rows.
df = df.dropna(subset=["latitude", "longitude"])

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

if "ai_confidence" in df.columns:
    min_conf = st.sidebar.slider("Minimum AI confidence", 0.0, 1.0, 0.0, 0.05)
    df = df[df["ai_confidence"].fillna(0) >= min_conf]
    st.sidebar.write("**Rows after confidence filter:**", len(df))

if "site_type" in df.columns:
    site_types = sorted(df["site_type"].dropna().astype(str).unique().tolist())
    selected_types = st.sidebar.multiselect("Site types", site_types, default=site_types)
    df = df[df["site_type"].astype(str).isin(selected_types)]

if df.empty:
    st.warning("No rows after filters.")
    st.stop()

# Top metrics
high_count = int((df["risk_level"] == "High").sum()) if "risk_level" in df.columns else 0
medium_count = int((df["risk_level"] == "Medium").sum()) if "risk_level" in df.columns else 0
child_sites = int(df["site_type"].str.lower().isin(["school", "clinic", "hospital", "doctors"]).sum())
mean_risk = df["risk_score"].mean()
mean_conf = df["ai_confidence"].mean() if "ai_confidence" in df.columns else None

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Mapped sites", len(df))
col2.metric("High-risk sites", high_count)
col3.metric("Medium-risk sites", medium_count)
col4.metric("Schools/clinics", child_sites)
col5.metric("Mean AI confidence", "N/A" if pd.isna(mean_conf) else f"{mean_conf:.2f}")

# Risk map
st.subheader("AI climate-health risk map")

center_lat = float(df["latitude"].mean())
center_lon = float(df["longitude"].mean())

map_df = df.copy()
map_df["risk_red"] = (map_df["risk_score"] * 2.55).clip(0, 255)
map_df["risk_green"] = (190 - map_df["risk_score"] * 1.45).clip(35, 190)
map_df["risk_radius"] = (map_df["risk_score"] * 38 + 70).clip(70, 4500)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[longitude, latitude]",
    get_radius="risk_radius",
    get_fill_color="[risk_red, risk_green, 80, 180]",
    pickable=True,
)
view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=8.5, pitch=0)
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
                "Vegetation stress (NDVI)",
                "Water stress (NDWI)",
                "Drought stress (NDMI)",
                "Heat stress placeholder",
                "Site sensitivity",
            ],
            "score": [
                float(row.get("ai_land_stress_score", 0.5)),
                float(row.get("vegetation_stress", 0.5)),
                float(row.get("water_stress", 0.5)),
                float(row.get("drought_stress", 0.5)),
                float(row.get("heat_stress", 0.5)),
                float(row.get("site_sensitivity", 0.5)),
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
        "Vegetation, water, and drought stress are calculated from Sentinel-2 indices. "
        "Heat remains neutral until a climate-temperature layer is added."
    )

with right:
    st.subheader("AI prediction summary")
    st.write(f"**{selected}** is classified as **{row.risk_level} risk** with a score of **{row.risk_score:.1f}/100**.")
    st.write(f"**AI land-cover class:** `{row.get('ai_landcover_class', 'unknown')}`")

    if all(c in row for c in ["NDVI", "NDWI", "NDMI"]):
        st.write(f"**NDVI / NDWI / NDMI:** {row.get('NDVI'):.3f} / {row.get('NDWI'):.3f} / {row.get('NDMI'):.3f}")

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

chart_col3, chart_col4 = st.columns(2)
with chart_col3:
    if "risk_level" in df.columns:
        st.plotly_chart(
            px.histogram(df, x="risk_level", title="Risk category distribution"),
            use_container_width=True,
        )
with chart_col4:
    if "risk_score" in df.columns:
        st.plotly_chart(
            px.histogram(df, x="risk_score", nbins=25, title="Risk score distribution"),
            use_container_width=True,
        )

st.subheader("AI prediction dataset")
preferred_cols = [
    "site_id",
    "site_name",
    "site_type",
    "latitude",
    "longitude",
    "NDVI",
    "NDWI",
    "NDMI",
    "ai_landcover_class",
    "ai_confidence",
    "ai_land_stress_score",
    "vegetation_stress",
    "water_stress",
    "drought_stress",
    "heat_stress",
    "site_sensitivity",
    "risk_score",
    "risk_level",
    "source",
    "validation_status",
]
visible_cols = [c for c in preferred_cols if c in df.columns]
other_cols = [c for c in df.columns if c not in visible_cols]
st.dataframe(df[visible_cols + other_cols].sort_values("risk_score", ascending=False), use_container_width=True)
