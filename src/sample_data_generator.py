"""Generate synthetic sample points for the UNICEF POC demo.

The coordinates are approximate demo points between Homs and Damascus and do not
represent verified schools, clinics, or communities. Replace with vetted local
partner data before submission or clearly label as demo data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from risk_score import compute_risk_score

rng = np.random.default_rng(42)

# Approximate corridor bounds between Homs and Damascus.
N = 36
lat = rng.uniform(33.50, 34.75, N)
lon = rng.uniform(36.30, 37.25, N)
site_types = rng.choice(["community", "school", "clinic"], size=N, p=[0.55, 0.30, 0.15])
weight_map = {"community": 1.0, "school": 1.2, "clinic": 1.3}

# Synthetic indicators for demo only. Replace with satellite/climate-derived values.
df = pd.DataFrame({
    "id": [f"SY-DEMO-{i+1:03d}" for i in range(N)],
    "name": [f"Demo Site {i+1}" for i in range(N)],
    "site_type": site_types,
    "lat": lat,
    "lon": lon,
    "ndvi": rng.uniform(0.08, 0.62, N),
    "ndwi": rng.uniform(-0.28, 0.26, N),
    "temp_c": rng.uniform(30, 43, N),
    "drought_index": rng.uniform(0.15, 0.95, N),
})
df["child_site_weight"] = df["site_type"].map(weight_map)
scored = compute_risk_score(df)
scored.to_csv("data/sample_sites.csv", index=False)
print("Wrote data/sample_sites.csv")
