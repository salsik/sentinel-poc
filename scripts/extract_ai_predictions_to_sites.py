from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

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
    class_code = int(class_code) if not pd.isna(class_code) else 0

    if class_code == 80:   # water
        return 0.20
    if class_code in [10, 30, 40]:  # tree, grass, crop
        return 0.25
    if class_code == 20:   # shrubland
        return 0.45
    if class_code == 50:   # built-up
        return 0.60
    if class_code == 60:   # bare/sparse vegetation
        return 0.90

    return 0.55

def risk_category(score):
    if pd.isna(score):
        return "unknown"
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"

def site_weight(site_type):
    site_type = str(site_type).lower()

    if site_type in ["clinic", "hospital", "doctors"]:
        return 1.30
    if site_type == "school":
        return 1.20
    return 1.00

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tif",
        default=str(ROOT / "data" / "syria_ai_landcover_prediction.tif"),
        help="AI prediction GeoTIFF: band 1 = class, band 2 = confidence"
    )
    parser.add_argument(
        "--sites",
        default=str(ROOT / "data" / "sites_from_osm.csv"),
        help="CSV with latitude and longitude"
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "sites_with_ai_predictions.csv"),
        help="Output CSV for Streamlit"
    )
    args = parser.parse_args()

    sites = pd.read_csv(args.sites)

    if "latitude" not in sites.columns or "longitude" not in sites.columns:
        raise ValueError("Sites CSV must include latitude and longitude columns.")

    with rasterio.open(args.tif) as src:
        lon = sites["longitude"].astype(float).tolist()
        lat = sites["latitude"].astype(float).tolist()

        if src.crs and str(src.crs) != "EPSG:4326":
            xs, ys = transform("EPSG:4326", src.crs, lon, lat)
        else:
            xs, ys = lon, lat

        samples = list(src.sample(list(zip(xs, ys))))

    pred_classes = []
    confidences = []

    for sample in samples:
        class_code = sample[0]
        confidence = sample[1] if len(sample) > 1 else np.nan

        if class_code == 0:
            pred_classes.append(np.nan)
            confidences.append(np.nan)
        else:
            pred_classes.append(int(class_code))
            confidences.append(float(confidence))

    sites["ai_landcover_code"] = pred_classes
    sites["ai_landcover_class"] = [
        CLASS_MAP.get(int(c), "unknown") if not pd.isna(c) else "unknown"
        for c in pred_classes
    ]
    sites["ai_confidence"] = confidences

    sites["ai_land_stress_score"] = sites["ai_landcover_code"].apply(risk_from_land_class)

    sites["ai_climate_health_risk_score"] = (
        sites["ai_land_stress_score"] * sites["site_type"].apply(site_weight)
    ).clip(0, 1)

    sites["ai_risk_category"] = sites["ai_climate_health_risk_score"].apply(risk_category)

    sites.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Wrote: {args.out}")
    print(sites[
        [
            "site_id",
            "site_name",
            "site_type",
            "ai_landcover_class",
            "ai_confidence",
            "ai_climate_health_risk_score",
            "ai_risk_category",
        ]
    ].head(20).to_string(index=False))

if __name__ == "__main__":
    main()



"""
python scripts/extract_ai_predictions_to_sites.py 
  --tif data/syria_ai_landcover_prediction.tif 
  --sites data/sites_from_osm.csv 
  --out data/sites_with_ai_predictions.csv

"""