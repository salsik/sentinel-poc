# Open Climate & Health Risk Intelligence for Children in Syria

An open-source AI/geospatial dashboard that maps child-centered climate-health vulnerability for communities, schools, and health facilities in Syria using satellite, climate, water, vegetation, and local validation data.

## Demo

Live POC demo:

https://mazadi-sentinel-poc.streamlit.app/

## How to run

Install the project requirements:

```bash
pip install -r requirements.txt
````

Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_app_ai_v2.py
```

## POC scope

Pilot area: Homs-Damascus corridor, Syria.

Prototype layers:

1. Vegetation stress: NDVI / EVI from Sentinel-2
2. Water stress: NDWI / NDMI and surface-water indicators
3. Heat / drought exposure: NASA POWER or ERA5-derived temperature and precipitation indicators
4. Child-centered vulnerability: school, clinic, and community risk scores

## What this repo contains

* `app/streamlit_app_ai_v2.py` - demo dashboard using sample data
* `src/risk_score_ai.py` - transparent scoring logic
* `src/sample_data_generator.py` - generates synthetic demo points
* `scripts/gee_sentinel_layers.js` - Google Earth Engine starter script for NDVI, NDWI, NDMI

## Open-source plan

Code: MIT License or Apache 2.0.
Documentation: CC-BY 4.0.
Sensitive local data and any child-related personal data must not be published.

## Data protection note

This POC does not collect personal data from children. All child-centered indicators are aggregated at school, clinic, or community level and should be published only with appropriate privacy safeguards.


