# Open Climate & Health Risk Intelligence for Children in Syria

A proof-of-concept for the UNICEF Venture Fund Climate & Health 2026 call.

## One-line concept
An open-source AI/geospatial dashboard that maps child-centered climate-health vulnerability for communities, schools, and health facilities in Syria using satellite, climate, water, vegetation, and local validation data.

## POC scope
Pilot area: Homs-Damascus corridor, Syria.

Prototype layers:
1. Vegetation stress: NDVI / EVI from Sentinel-2
2. Water stress: NDWI / NDMI and surface-water indicators
3. Heat / drought exposure: NASA POWER or ERA5-derived temperature and precipitation indicators
4. Child-centered vulnerability: school, clinic, and community risk scores

## What this repo contains
- `app/streamlit_app.py` - demo dashboard using sample data
- `src/risk_score.py` - transparent scoring logic
- `src/sample_data_generator.py` - generates synthetic demo points
- `scripts/gee_sentinel_layers.js` - Google Earth Engine starter script for NDVI, NDWI, NDMI
- `docs/10_day_sprint.md` - implementation plan
- `docs/data_sources.md` - open data sources and licensing notes
- `docs/model_card_draft.md` - draft model card for UNICEF
- `docs/video_script.md` - 2-minute pitch video structure
- `docs/unicef_form_answers_draft.md` - draft answers aligned with the application form

## Open-source plan
Code: MIT License or Apache 2.0.  
Documentation: CC-BY 4.0.  
Sensitive local data and any child-related personal data must not be published.

## Data protection note
This POC does not collect personal data from children. All child-centered indicators are aggregated at school, clinic, or community level and should be published only with appropriate privacy safeguards.
