
"""

Python script: predict directly from site feature CSV


this converts the output inference values from GEE to preddictions SV ddirectly

ouput we can take directly to Streamlit without needing to do raster sampling again,\\
 and also allows us to use the confidence values from the model if available.
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

def risk_from_land_class(class_code):
    if pd.isna(class_code):
        return 0.55

    class_code = int(class_code)

    if class_code == 80:
        return 0.20  # water nearby / water body
    if class_code in [10, 30, 40]:
        return 0.25  # vegetation/cropland
    if class_code == 20:
        return 0.45  # shrubland / semi-dry
    if class_code == 50:
        return 0.60  # built-up
    if class_code == 60:
        return 0.90  # bare/sparse vegetation

    return 0.55

def site_weight(site_type):
    site_type = str(site_type).lower()

    if site_type in ["clinic", "hospital", "doctors"]:
        return 1.30
    if site_type == "school":
        return 1.20
    return 1.00

def risk_category(score):
    if pd.isna(score):
        return "unknown"
    if score >= 0.67:
        return "high"
    if score >= 0.34:
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

    # Keep rows with valid feature values
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

    df["ai_landcover_class"] = df["ai_landcover_code"].apply(
        lambda x: CLASS_MAP.get(int(x), "unknown") if not pd.isna(x) else "unknown"
    )

    df["ai_land_stress_score"] = df["ai_landcover_code"].apply(risk_from_land_class)

    # We can strengthen this later by adding heat/drought climate data.
    # For now, land stress + site sensitivity is the POC score.
    df["ai_climate_health_risk_score"] = (
        df["ai_land_stress_score"] * df["site_type"].apply(site_weight)
    ).clip(0, 1)

    df["ai_risk_category"] = df["ai_climate_health_risk_score"].apply(risk_category)

    # Clean columns for Streamlit
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
        "ai_climate_health_risk_score",
        "ai_risk_category",
    ]

    available_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in available_cols]
    df = df[available_cols + other_cols]

    df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Wrote: {args.out}")
    print(df[[
        "site_id",
        "site_name",
        "site_type",
        "ai_landcover_class",
        "ai_confidence",
        "ai_climate_health_risk_score",
        "ai_risk_category"
    ]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()



"""



python scripts/predict_sites_from_feature_csv.py --csv data/syria_2025_site_features_for_ai_inference.csv --out data/corridor_sites_with_ai_predictions.csv


"""