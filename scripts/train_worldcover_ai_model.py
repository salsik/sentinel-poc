from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "BLUE", "GREEN", "RED",
    "RE1", "RE2", "RE3",
    "NIR", "SWIR1", "SWIR2",
    "NDVI", "NDWI", "NDMI", "NDBI",
    "EVI", "SAVI", "BSI"
]

WORLD_COVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Training sample CSV exported from GEE")
    parser.add_argument("--out-model", default=str(MODELS_DIR / "worldcover_rf_model.joblib"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Clean columns in case Earth Engine exports unusual types
    needed = FEATURES + ["label"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[needed].replace([np.inf, -np.inf], np.nan).dropna()

    # Remove classes that are irrelevant or too rare if needed
    df = df[df["label"].isin([10, 20, 30, 40, 50, 60, 80])]

    df["label_name"] = df["label"].map(WORLD_COVER_CLASSES)

    print("Class distribution:")
    print(df["label_name"].value_counts())

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    labels_sorted = sorted(y.unique().tolist())
    target_names = [WORLD_COVER_CLASSES[int(x)] for x in labels_sorted]

    report_text = classification_report(
        y_test,
        preds,
        labels=labels_sorted,
        target_names=target_names
    )

    report_dict = classification_report(
        y_test,
        preds,
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True
    )

    cm = confusion_matrix(y_test, preds, labels=labels_sorted)

    print(f"\nAccuracy against ESA WorldCover labels: {acc:.3f}")
    print(report_text)

    dump(
        {
            "model": model,
            "features": FEATURES,
            "class_map": WORLD_COVER_CLASSES,
            "note": "Model trained on ESA WorldCover labels, not local field labels."
        },
        args.out_model
    )

    with open(REPORTS_DIR / "worldcover_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "important_note": "Metrics are against ESA WorldCover labels, not independent Syria ground truth.",
            "accuracy_against_worldcover_labels": acc,
            "features": FEATURES,
            "classes_used": {str(k): v for k, v in WORLD_COVER_CLASSES.items() if k in labels_sorted},
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist()
        }, f, indent=2)

    print(f"\nSaved model to: {args.out_model}")
    print(f"Saved metrics to: {REPORTS_DIR / 'worldcover_model_metrics.json'}")

if __name__ == "__main__":
    main()




"""

python scripts\train_worldcover_ai_model.py --csv "data/regional_sentinel_worldcover_training_samples.csv"

"""