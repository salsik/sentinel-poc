

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
#if str(ROOT) not in sys.path:
#    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src.risk_score import compute_risk_score

DATA = ROOT / "data" / "sample_sites.csv"

DATA = ROOT / "data" / "sites_with_risk_scores.csv"

DATA = ROOT / "data" / "sites_with_ai_predictions.csv"

st.set_page_config(page_title="Syria Climate & Child Health Risk POC", layout="wide")
st.title("Open Climate & Health Risk Intelligence for Children in Syria")
st.caption("10-day UNICEF Venture Fund proof-of-concept | Demo data until local validation is added")

if not DATA.exists():
    st.error("Missing data/sites_with_risk_scores.csv. Run: python src/sample_data_generator.py")
    st.stop()

raw = pd.read_csv(DATA)
df = compute_risk_score(raw)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Demo sites", len(df))
col2.metric("High-risk sites", int((df.risk_level == "High").sum()))
col3.metric("Schools/clinics", int(df.site_type.isin(["school", "clinic"]).sum()))
col4.metric("Mean risk score", f"{df.risk_score.mean():.1f}/100")

st.subheader("Risk map")
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[longitude, latitude]",
    get_radius="risk_score * 55 +2", # to prevet 0 cases
    get_fill_color="[risk_score * 2.55, 120, 90, 170]",
    pickable=True,
)
view_state = pdk.ViewState(latitude=34.1, longitude=36.75, zoom=7.1, pitch=0)
st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "{site_name}\nType: {site_type}\nRisk: {risk_score} ({risk_level})"},
        #height=650,
    ),
    use_container_width=True,
)


left, right = st.columns([1.05, 1])
with left:
    st.subheader("Risk components")
    selected = st.selectbox("Select site", df["site_name"].tolist())
    row = df[df.site_name == selected].iloc[0]
    comp = pd.DataFrame({
        "component": ["Vegetation stress", "Water stress", "Heat stress", "Drought stress"],
        "score": [row.vegetation_stress, row.water_stress, row.heat_stress, row.drought_stress],
    })
    st.plotly_chart(px.bar(comp, x="component", y="score", range_y=[0, 1], title=f"{selected}: risk components"), use_container_width=True)

with right:
    st.subheader("Suggested action note")
    st.write(f"**{selected}** is classified as **{row.risk_level} risk** with a score of **{row.risk_score}/100**.")
    if row.risk_level == "High":
        st.warning("Prioritize local validation, WASH/nutrition review, and heat-safety planning for this site.")
    elif row.risk_level == "Medium":
        st.info("Monitor satellite indicators and collect local feedback before seasonal shocks intensify.")
    else:
        st.success("Continue monitoring. No immediate high-risk flag in the current demo indicators.")
    st.caption("This is not a medical diagnosis or emergency decision system. Human review is required.")

st.subheader("Demo dataset")
st.dataframe(df.sort_values("risk_score", ascending=False), use_container_width=True)


"""


pip install -r requirements.txt
streamlit run app/streamlit_app.py

"""