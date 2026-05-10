from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def risk_from_indices(ndvi, ndwi, ndmi):
    """
    Convert spectral indicators to simple 0-1 stress indicators.

    Higher score = higher risk/stress.

    These thresholds are intentionally simple for POC.
    They must be validated locally before operational use.
    """

    ndvi = safe_float(ndvi)
    ndwi = safe_float(ndwi)
    ndmi = safe_float(ndmi)

    # Vegetation stress:
    # NDVI high means healthier vegetation, so stress is inverse.
    # NDVI <= 0.10 => high stress
    # NDVI >= 0.50 => low stress
    if np.isnan(ndvi):
        vegetation_stress = np.nan
    else:
        vegetation_stress = clamp01((0.50 - ndvi) / 0.40)

    # Water stress:
    # NDWI high usually indicates water/wetness.
    # NDWI <= -0.20 => high water stress
    # NDWI >= 0.20 => low water stress
    if np.isnan(ndwi):
        water_stress = np.nan
    else:
        water_stress = clamp01((0.20 - ndwi) / 0.40)

    # Moisture/drought stress:
    # NDMI high means more moisture, so drought stress is inverse.
    # NDMI <= -0.20 => high drought stress
    # NDMI >= 0.30 => low drought stress
    if np.isnan(ndmi):
        drought_stress = np.nan
    else:
        drought_stress = clamp01((0.30 - ndmi) / 0.50)

    # Placeholder heat stress until climate data is added.
    # Keep simple for first POC.
    heat_stress = 0.50

    # Site sensitivity can be added later based on site type.
    return vegetation_stress, water_stress, drought_stress, heat_stress

def risk_category(score):
    if pd.isna(score):
        return "unknown"
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"

def site_weight(site_type):
    if site_type == "clinic":
        return 1.30
    if site_type == "school":
        return 1.20
    return 1.00

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tif",
        required=True,
        help="Path to exported GeoTIFF from Google Earth Engine"
    )
    parser.add_argument(
        "--sites",
        default=str(ROOT / "data" / "sites_from_osm.csv"),
        help="Input CSV with latitude and longitude columns"
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "sites_with_risk_scores.csv"),
        help="Output CSV"
    )
    args = parser.parse_args()

    sites = pd.read_csv(args.sites)

    required_cols = {"latitude", "longitude"}
    missing = required_cols - set(sites.columns)
    if missing:
        raise ValueError(f"Sites CSV missing required columns: {missing}")

    with rasterio.open(args.tif) as src:
        print("GeoTIFF CRS:", src.crs)
        print("Band count:", src.count)
        print("Bounds:", src.bounds)

        # Expected GEE band order:
        # 1 NDVI, 2 NDWI, 3 NDMI, 4 EVI, 5 SAVI
        band_names = ["NDVI", "NDWI", "NDMI", "EVI", "SAVI"]

        # Convert lon/lat WGS84 points to raster CRS if needed.
        lon = sites["longitude"].astype(float).tolist()
        lat = sites["latitude"].astype(float).tolist()

        if src.crs and str(src.crs) != "EPSG:4326":
            xs, ys = transform("EPSG:4326", src.crs, lon, lat)
        else:
            xs, ys = lon, lat

        coords = list(zip(xs, ys))

        sampled = list(src.sample(coords))

    # Add sampled band values
    for b_idx, b_name in enumerate(band_names):
        values = []
        for pix in sampled:
            if b_idx < len(pix):
                val = pix[b_idx]
                if val == src.nodata:
                    val = np.nan
                values.append(float(val))
            else:
                values.append(np.nan)
        sites[b_name] = values

    # Compute risk fields
    vegetation_stress = []
    water_stress = []
    drought_stress = []
    heat_stress = []
    overall_risk = []
    categories = []

    for _, row in sites.iterrows():
        veg, wat, dro, heat = risk_from_indices(row.get("NDVI"), row.get("NDWI"), row.get("NDMI"))

        weight = site_weight(row.get("site_type", "community"))

        # UNICEF-friendly transparent risk formula
        base = (
            0.30 * veg +
            0.30 * wat +
            0.25 * dro +
            0.15 * heat
        )

        score = clamp01(base * weight)

        vegetation_stress.append(veg)
        water_stress.append(wat)
        drought_stress.append(dro)
        heat_stress.append(heat)
        overall_risk.append(score)
        categories.append(risk_category(score))

    sites["vegetation_stress"] = vegetation_stress
    sites["water_stress"] = water_stress
    sites["drought_stress"] = drought_stress
    sites["heat_stress"] = heat_stress
    sites["overall_risk"] = overall_risk
    sites["risk_category"] = categories

    sites.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote scored sites to {args.out}")
    print(sites[[
        "site_id",
        "site_name",
        "site_type",
        "NDVI",
        "NDWI",
        "NDMI",
        "overall_risk",
        "risk_category"
    ]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()

"""

python scripts/extract_site_values.py --tif "data/syria_ai_landcover_prediction.tif"

"""