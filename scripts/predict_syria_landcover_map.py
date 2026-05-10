from pathlib import Path
import argparse
import numpy as np
import rasterio
from joblib import load

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", required=True, help="Syria feature stack GeoTIFF exported from GEE")
    parser.add_argument("--model", default=str(ROOT / "models" / "worldcover_rf_model.joblib"))
    parser.add_argument("--out", default=str(ROOT / "data" / "syria_ai_landcover_prediction.tif"))
    args = parser.parse_args()

    bundle = load(args.model)
    model = bundle["model"]
    features = bundle["features"]

    with rasterio.open(args.tif) as src:
        arr = src.read()
        profile = src.profile.copy()
        height = src.height
        width = src.width

        if arr.shape[0] != len(features):
            raise ValueError(
                f"Expected {len(features)} bands, got {arr.shape[0]}. "
                f"Check GEE band order."
            )

        X = arr.reshape(arr.shape[0], -1).T

        # Handle nan pixels
        valid = np.isfinite(X).all(axis=1)
        pred = np.zeros(X.shape[0], dtype=np.uint8)
        confidence = np.zeros(X.shape[0], dtype=np.float32)

        print(f"Valid pixels: {valid.sum()} / {len(valid)}")

        if valid.sum() > 0:
            pred_valid = model.predict(X[valid])
            pred[valid] = pred_valid.astype(np.uint8)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X[valid])
                confidence[valid] = proba.max(axis=1).astype(np.float32)

        pred_img = pred.reshape(height, width)
        conf_img = confidence.reshape(height, width)

        profile.update(
            count=2,
            dtype=rasterio.float32,
            nodata=0,
            compress="deflate"
        )

        with rasterio.open(args.out, "w", **profile) as dst:
            dst.write(pred_img.astype(np.float32), 1)
            dst.write(conf_img.astype(np.float32), 2)
            dst.set_band_description(1, "ai_predicted_worldcover_class")
            dst.set_band_description(2, "ai_confidence")

    print(f"Wrote prediction map to: {args.out}")

if __name__ == "__main__":
    main()


"""

python scripts/predict_syria_landcover_map.py --tif "data/syria_sentinel_feature_stack_for_inference.tif"


"""