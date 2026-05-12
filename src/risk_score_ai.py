"""Risk scoring utilities for the Syria UNICEF POC.

This module supports two input formats:

1. New AI prediction format:
   site_id, site_name, site_type, latitude, longitude, source, validation_status,
   ai_landcover_code, ai_landcover_class, ai_confidence,
   ai_land_stress_score, ai_climate_health_risk_score, ai_risk_category

2. Older spectral-index format:
   id/site_id, site_name, site_type, latitude, longitude, NDVI, NDWI, NDMI,
   EVI, SAVI, vegetation_stress, water_stress, drought_stress, heat_stress,
   overall_risk, risk_category

The returned dataframe always includes the normalized columns used by Streamlit:
   site_id, site_name, site_type, latitude, longitude,
   risk_score, risk_level,
   vegetation_stress, water_stress, drought_stress, heat_stress,
   ai_landcover_class, ai_confidence
"""

from __future__ import annotations

import numpy as np
import pandas as pd


AI_SCORE_COL = "ai_climate_health_risk_score"
AI_CATEGORY_COL = "ai_risk_category"


def minmax(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0..1 while handling constant or missing values."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _site_weight(site_type: object) -> float:
    """Increase priority for child-serving facilities."""
    t = str(site_type).strip().lower()
    if t in {"clinic", "hospital", "doctors", "health_facility", "health facility"}:
        return 1.30
    if t == "school":
        return 1.20
    return 1.00


def _risk_level_from_score(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 66:
        return "High"
    if score >= 33:
        return "Medium"
    return "Low"


def _normalize_risk_category(value: object) -> str:
    v = str(value).strip().lower()
    if v == "high":
        return "High"
    if v == "medium":
        return "Medium"
    if v == "low":
        return "Low"
    return "Unknown"


def _ensure_identity_columns(out: pd.DataFrame) -> pd.DataFrame:
    """Normalize ID and basic display columns."""
    if "site_id" not in out.columns and "id" in out.columns:
        out["site_id"] = out["id"]
    if "site_id" not in out.columns:
        out["site_id"] = [f"site_{i+1:03d}" for i in range(len(out))]
    if "site_name" not in out.columns:
        out["site_name"] = out["site_id"].astype(str)
    if "site_type" not in out.columns:
        out["site_type"] = "community"
    return out


def _compute_from_ai_predictions(out: pd.DataFrame) -> pd.DataFrame:
    """Use already-generated AI risk columns from sites_with_ai_predictions.csv."""
    # ai_climate_health_risk_score is expected in 0..1.
    raw_score = pd.to_numeric(out[AI_SCORE_COL], errors="coerce").fillna(0)

    # Be defensive: if someone provides 0..100, keep it; if 0..1, scale to 0..100.
    if raw_score.max() <= 1.0:
        out["risk_score"] = (raw_score * 100).clip(0, 100).round(1)
    else:
        out["risk_score"] = raw_score.clip(0, 100).round(1)

    if AI_CATEGORY_COL in out.columns:
        out["risk_level"] = out[AI_CATEGORY_COL].apply(_normalize_risk_category)
    else:
        out["risk_level"] = out["risk_score"].apply(_risk_level_from_score)

    # For the component chart, approximate the main components from available AI fields.
    # The AI CSV does not contain the old NDVI/NDWI component columns.
    if "vegetation_stress" not in out.columns:
        out["vegetation_stress"] = pd.to_numeric(
            out.get("ai_land_stress_score", pd.Series([0.0] * len(out))), errors="coerce"
        ).fillna(0)

    if "water_stress" not in out.columns:
        # No separate water score in new AI CSV; keep neutral until we merge spectral features.
        out["water_stress"] = 0.5
    if "drought_stress" not in out.columns:
        # No separate drought score in new AI CSV; keep neutral until climate/NDMI layer is added.
        out["drought_stress"] = 0.5
    if "heat_stress" not in out.columns:
        # No heat data yet; keep neutral placeholder and label it clearly in dashboard.
        out["heat_stress"] = 0.5

    if "ai_landcover_class" not in out.columns:
        out["ai_landcover_class"] = "unknown"
    if "ai_confidence" not in out.columns:
        out["ai_confidence"] = np.nan

    return out


def _compute_from_spectral_indices(out: pd.DataFrame) -> pd.DataFrame:
    """Fallback for older NDVI/NDWI/NDMI CSVs."""
    # If stress columns are already present, use them. Otherwise derive from indices.
    if "vegetation_stress" not in out.columns:
        if "NDVI" in out.columns:
            out["vegetation_stress"] = 1 - minmax(out["NDVI"])
        else:
            out["vegetation_stress"] = 0.5

    if "water_stress" not in out.columns:
        if "NDWI" in out.columns:
            out["water_stress"] = 1 - minmax(out["NDWI"])
        else:
            out["water_stress"] = 0.5

    if "drought_stress" not in out.columns:
        if "NDMI" in out.columns:
            out["drought_stress"] = 1 - minmax(out["NDMI"])
        else:
            out["drought_stress"] = 0.5

    if "heat_stress" not in out.columns:
        out["heat_stress"] = 0.5

    for col in ["vegetation_stress", "water_stress", "drought_stress", "heat_stress"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.5).clip(0, 1)

    # Use existing overall_risk if present; otherwise calculate transparent score.
    if "overall_risk" in out.columns:
        raw_score = pd.to_numeric(out["overall_risk"], errors="coerce").fillna(0)
        if raw_score.max() <= 1.0:
            out["risk_score"] = (raw_score * 100).clip(0, 100).round(1)
        else:
            out["risk_score"] = raw_score.clip(0, 100).round(1)
    else:
        base = (
            0.30 * out["water_stress"]
            + 0.25 * out["vegetation_stress"]
            + 0.25 * out["heat_stress"]
            + 0.20 * out["drought_stress"]
        )
        weight = out["site_type"].apply(_site_weight)
        out["risk_score"] = np.clip(base * weight * 100, 0, 100).round(1)

    if "risk_category" in out.columns:
        out["risk_level"] = out["risk_category"].apply(_normalize_risk_category)
    else:
        out["risk_level"] = out["risk_score"].apply(_risk_level_from_score)

    if "ai_landcover_class" not in out.columns:
        out["ai_landcover_class"] = "not_available"
    if "ai_confidence" not in out.columns:
        out["ai_confidence"] = np.nan

    return out


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with dashboard-ready risk fields.

    If the new AI columns are present, they are used directly. Otherwise the
    older spectral-index scoring logic is applied.
    """
    out = df.copy()
    out = _ensure_identity_columns(out)

    if AI_SCORE_COL in out.columns:
        out = _compute_from_ai_predictions(out)
    else:
        out = _compute_from_spectral_indices(out)

    # Final cleanup for dashboard.
    for col in ["latitude", "longitude"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["latitude", "longitude"])
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce").fillna(0).clip(0, 100)
    out["risk_level"] = out["risk_level"].fillna("Unknown")

    return out


if __name__ == "__main__":
    sample = pd.read_csv("data/sites_with_ai_predictions.csv")
    scored = compute_risk_score(sample)
    scored.to_csv("data/sites_with_ai_predictions_scored.csv", index=False)
    print(scored[["site_id", "site_name", "site_type", "risk_score", "risk_level"]].head())
