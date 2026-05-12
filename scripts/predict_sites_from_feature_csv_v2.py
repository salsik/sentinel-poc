"""
Predict site-level AI land-cover and climate-health risk scores from a
Google Earth Engine site-feature CSV.

Input:
  - A CSV exported from GEE with site metadata + Sentinel-2 features
  - A trained model bundle: models/worldcover_rf_model.joblib

Output:
  - data/corridor_sites_with_ai_predictions.csv for Streamlit

This version improves scoring by:
  - using NDVI to calculate vegetation stress
  - using NDWI to calculate water stress
  - using NDMI to calculate drought/moisture stress
  - avoiding the old aggressive multiplication by site type
  - keeping heat stress as a neutral placeholder until climate data is added
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]

CLASS_MAP = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    80: "permanent_water",
}

# Less aggressive land-context scores.
# Built-up is exposure context, not automatically extreme climate risk.
LAND_STRESS_BY_CLASS = {
    "permanent_water": 0.15,
    "tree_cover": 0.20,
    "cropland": 0.35,
    "grassland": 0.40,
    "shrubland": 0.50,
    "built_up": 0.45,
    "bare_sparse_vegetation": 0.85,
    "unknown": 0.55,
}


def clamp01(value):
    if pd.isna(value):
        return np.nan
    return float(max(0.0, min(1.0, value)))


def class_name_from_code(code):
    if pd.isna(code):
        return "unknown"
    try:
        return CLASS_MAP.get(int(code), "unknown")
    except Exception:
        return "unknown"


def risk_from_land_class_name(class_name):
    return LAND_STRESS_BY_CLASS.get(str(class_name).lower(), 0.55)


def vegetation_stress_from_ndvi(ndvi):
    """Lower NDVI = more vegetation stress.

    For this dryland POC:
    NDVI >= 0.45 -> low stress
    NDVI <= 0.05 -> high stress
    """
    if pd.isna(ndvi):
        return 0.50
    return clamp01((0.45 - float(ndvi)) / 0.40)


def water_stress_from_ndwi(ndwi):
    """Lower NDWI = more water/wetness stress.

    NDWI >= 0.10 -> low stress
    NDWI <= -0.30 -> high stress
    """
    if pd.isna(ndwi):
        return 0.50
    return clamp01((0.10 - float(ndwi)) / 0.40)


def drought_stress_from_ndmi(ndmi):
    """Lower NDMI = more moisture/drought stress.

    NDMI >= 0.20 -> low stress
    NDMI <= -0.30 -> high stress
    """
    if pd.isna(ndmi):
        return 0.50
    return clamp01((0.20 - float(ndmi)) / 0.50)


def site_sensitivity_score(site_type):
    """Child-centered site sensitivity as an additive component, not multiplier."""
    site_type = str(site_type).lower()
    if site_type in ["clinic", "hospital", "doctors"]:
        return 0.80
    if site_type == "school":
        return 0.70
    return 0.50


def risk_category(score_0_1):
    if pd.isna(score_0_1):
        return "unknown"
    if score_0_1 >= 0.75:
        return "high"
    if score_0_1 >= 0.50:
        return "medium"
    return "low"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Site feature CSV exported from GEE"
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "worldcover_rf_model.joblib"),
        help="Trained model bundle"
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "corridor_sites_with_ai_predictions.csv"),
        help="Output CSV for Streamlit"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    bundle = load(args.model)

    model = bundle["model"]
    features = bundle["features"]

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing model feature columns: {missing}")

    # Keep rows with valid feature values.
    X = df[features].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)

    print(f"Total rows: {len(df)}")
    print(f"Rows with valid model features: {valid.sum()}")
    print(f"Rows with missing features: {(~valid).sum()}")

    df["ai_landcover_code"] = np.nan
    df["ai_landcover_class"] = "unknown"
    df["ai_confidence"] = np.nan

    if valid.sum() > 0:
        X_valid = X.loc[valid, features]

        pred = model.predict(X_valid)
        df.loc[valid, "ai_landcover_code"] = pred.astype(int)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_valid)
            df.loc[valid, "ai_confidence"] = proba.max(axis=1)

    df["ai_landcover_class"] = df["ai_landcover_code"].apply(class_name_from_code)
    df["ai_land_stress_score"] = df["ai_landcover_class"].apply(risk_from_land_class_name)

    # Calculate actual stress components from Sentinel-2 features.
    # These are no longer neutral 0.5 placeholders.
    df["vegetation_stress"] = df["NDVI"].apply(vegetation_stress_from_ndvi) if "NDVI" in df.columns else 0.50
    df["water_stress"] = df["NDWI"].apply(water_stress_from_ndwi) if "NDWI" in df.columns else 0.50
    df["drought_stress"] = df["NDMI"].apply(drought_stress_from_ndmi) if "NDMI" in df.columns else 0.50

    # Keep heat as neutral until a climate dataset is added.
    # Later, replace this with ERA5/NASA POWER-derived heat exposure.
    df["heat_stress"] = 0.50
    df["site_sensitivity"] = df["site_type"].apply(site_sensitivity_score)

    # Balanced UNICEF POC score: additive, not multiplier.
    df["ai_climate_health_risk_score"] = (
        0.30 * df["ai_land_stress_score"] +
        0.25 * df["vegetation_stress"] +
        0.20 * df["water_stress"] +
        0.10 * df["drought_stress"] +
        0.05 * df["heat_stress"] +
        0.10 * df["site_sensitivity"]
    ).clip(0, 1)

    df["ai_risk_category"] = df["ai_climate_health_risk_score"].apply(risk_category)

    # Also add dashboard-friendly naming.
    df["risk_score"] = (df["ai_climate_health_risk_score"] * 100).round(1)
    df["risk_level"] = df["ai_risk_category"].map({
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "unknown": "Unknown",
    }).fillna("Unknown")

    preferred_cols = [
        "site_id",
        "site_name",
        "site_type",
        "latitude",
        "longitude",
        "source",
        "validation_status",
        "NDVI",
        "NDWI",
        "NDMI",
        "EVI",
        "SAVI",
        "BSI",
        "ai_landcover_code",
        "ai_landcover_class",
        "ai_confidence",
        "ai_land_stress_score",
        "vegetation_stress",
        "water_stress",
        "drought_stress",
        "heat_stress",
        "site_sensitivity",
        "ai_climate_health_risk_score",
        "ai_risk_category",
        "risk_score",
        "risk_level",
    ]

    available_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in available_cols]
    df = df[available_cols + other_cols]

    df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Wrote: {args.out}")
    print("Risk distribution:")
    print(df["risk_level"].value_counts(dropna=False))
    print("\nSample predictions:")
    print(df[[
        "site_id",
        "site_name",
        "site_type",
        "ai_landcover_class",
        "ai_confidence",
        "vegetation_stress",
        "water_stress",
        "drought_stress",
        "risk_score",
        "risk_level",
    ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()



"""


python scripts/predict_sites_from_feature_csv_v2.py --csv data/syria_2025_site_features_for_ai_inference.csv --out data/corridor_sites_with_ai_predictions_v2.csv



"""