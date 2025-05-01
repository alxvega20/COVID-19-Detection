
# src/prep_data.py
"""
Usage:
    python -m src.prep_data --root data/raw/Coronahack-Chest-XRay-Dataset
"""

import argparse, pathlib, pandas as pd
from sklearn.model_selection import train_test_split

def build_manifests(root: pathlib.Path):
    meta = pd.read_csv(root.parent / "Chest_xray_Corona_Metadata.csv")

    # keep frontal views & covid|normal labels
    meta = meta.query("View in ['PA', 'AP']")
    meta = meta[meta.Label.isin(['COVID-19', 'Normal'])].copy()
    meta["target"] = meta.Label.map({"Normal": 0, "COVID-19": 1})

    # attach absolute paths
    def _full(row):
        sub = "train" if row.Dataset_type.upper() == "TRAIN" else "test"
        return root / sub / row.X_ray_image_name
    meta["filepath"] = meta.apply(_full, axis=1)

    # stratified 70/15/15 split
    train_df, temp_df = train_test_split(
        meta, test_size=0.30, stratify=meta.target, random_state=42)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df.target, random_state=42)

    interim = root.parent / "interim"
    interim.mkdir(exist_ok=True)
    train_df.to_csv(interim/"train_manifest.csv", index=False)
    val_df.to_csv(interim/"val_manifest.csv", index=False)
    test_df.to_csv(interim/"test_manifest.csv", index=False)
    print("✓ wrote manifests to", interim)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    args = p.parse_args()
    build_manifests(args.root.resolve())
