#!/usr/bin/env python3
"""
split_dataset.py
-----------------
Stratified train/val split of a folder-per-class image dataset (the output
of crop_dataset.py) into:

    data/
        train/<CLASS>/...
        val/<CLASS>/...

Usage:
    python split_dataset.py --input ./classification_data --output ./data --val-frac 0.15
"""
import argparse
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    in_root = Path(args.input)
    out_root = Path(args.output)
    train_root = out_root / "train"
    val_root = out_root / "val"

    classes = sorted(p.name for p in in_root.iterdir() if p.is_dir())
    print("Classes:", classes)

    for cls in classes:
        files = sorted((in_root / cls).glob("*.jpg"))
        random.shuffle(files)
        n_val = max(1, int(len(files) * args.val_frac))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for subset_root, subset_files in [(train_root, train_files), (val_root, val_files)]:
            dst_dir = subset_root / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in subset_files:
                shutil.copy2(f, dst_dir / f.name)

        print(f"{cls}: train={len(train_files)} val={len(val_files)}")


if __name__ == "__main__":
    main()
