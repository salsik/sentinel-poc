"""Transparent climate-health risk scoring for the Syria UNICEF POC.

This module intentionally starts with a simple interpretable score rather than a
black-box model. It can later be replaced or calibrated with field validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def minmax(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0..1 while handling constant values."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute child-centered climate-health risk.

    Required columns:
      - ndvi: higher = healthier vegetation; low values increase nutrition/livelihood risk
      - ndwi: higher = more surface water/moisture; low values increase WASH risk
      - temp_c: recent mean/max temperature; high values increase heat risk
      - drought_index: higher = more drought exposure
      - child_site_weight: 1.0 community, 1.2 school, 1.3 clinic by default

    Returns a copy with component scores and total risk score from 0..100.
    """
    out = df.copy()
    out["vegetation_stress"] = 1 - minmax(out["NDVI"])
    out["water_stress"] = 1 - minmax(out["NDWI"])
    out["heat_stress"] = minmax(out["heat_stress"])
    out["drought_stress"] = minmax(out["drought_stress"])

    base = (
        0.30 * out["water_stress"]
        + 0.25 * out["vegetation_stress"]
        + 0.25 * out["heat_stress"]
        + 0.20 * out["drought_stress"]
    )
    weight = out.get("child_site_weight", 1.0)
    out["risk_score"] = np.clip(base * weight * 100, 0, 100).round(1)
    out["risk_level"] = pd.cut(
        out["risk_score"],
        bins=[-0.1, 33, 66, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return out


if __name__ == "__main__":
    sample = pd.read_csv("data/sample_sites.csv")
    scored = compute_risk_score(sample)
    scored.to_csv("data/sample_sites_scored.csv", index=False)
    print(scored[["name", "site_type", "risk_score", "risk_level"]].head())
