#!/usr/bin/env python3
"""
crop_dataset.py
----------------
Converts the raw RV-PBS whole-slide dataset (one folder per dominant class,
each containing full smear photos + a CVAT annotations.xml with per-cell
polygon/box labels) into a single-cell classification dataset:

    classification_data/
        NEUTROPHILS/   IMG_1234_0.jpg  IMG_1234_1.jpg  ...
        LYMPHOCYTES/   ...
        BASOPHILS/     ...
        ...

Each saved crop is a plain bounding-box region around ONE annotated WBC,
taken straight from the original photo (with a small padding margin) and
with NO alpha-masking. That's a deliberate choice: a masked, black-background
cutout looks nothing like what a real upload will look like, so training on
masked crops would generalize poorly. A natural bbox crop (with its real
background) is much closer to what NucleiMapper's frontend will eventually
hand the model.

Per-shape labels (inside annotations.xml) are used directly, so this also
works unmodified on the TEST_MIXED folder, which contains slides with more
than one cell type per image - making it a natural, untouched test set.

Usage:
    # crop the 10 training class folders
    python crop_dataset.py --input /path/to/RV-PBS-main --output ./classification_data

    # crop the held-out mixed test folder the same way
    python crop_dataset.py --input /path/to/RV-PBS-main --output ./test_mixed_data --folders "TEST_MIXED"
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from tqdm import tqdm

# label used *inside* annotations.xml (lowercase) -> canonical class folder name
LABEL_TO_CLASS = {
    "band cell": "BAND CELLS",
    "basophil": "BASOPHILS",
    "blast cell": "BLAST CELLS",
    "eosinophil": "EOSINOPHILS",
    "lymphocyte": "LYMPHOCYTES",
    "metamyelocyte": "METAMYELOCYTES",
    "monocyte": "MONOCYTES",
    "myelocyte": "MYELOCYTE",
    "neutrophil": "NEUTROPHILS",
    "promyelocyte": "PROMYELOCYTES",
}
SKIP_LABELS = {"bg", "background"}

DEFAULT_SOURCE_FOLDERS = [
    "BAND CELLS", "BASOPHILS", "BLAST CELLS", "EOSINOPHILS", "LYMPHOCYTES",
    "METAMYELOCYTES", "MONOCYTES", "MYELOCYTE", "NEUTROPHILS", "PROMYELOCYTES",
]

PAD = 12  # pixels of extra context kept around each annotated cell


def shape_bbox(shape_el):
    tag = shape_el.tag
    if tag == "polygon":
        pts = shape_el.attrib["points"]
        xs, ys = [], []
        for pair in pts.split(";"):
            x, y = pair.split(",")
            xs.append(float(x))
            ys.append(float(y))
        return min(xs), min(ys), max(xs), max(ys)
    if tag == "box":
        a = shape_el.attrib
        return float(a["xtl"]), float(a["ytl"]), float(a["xbr"]), float(a["ybr"])
    return None


def crop_folder(folder: Path, out_root: Path) -> int:
    ann_path = folder / "annotations.xml"
    if not ann_path.exists():
        print(f"  ! no annotations.xml in {folder}, skipping")
        return 0
    root = ET.parse(ann_path).getroot()
    saved = 0
    for image_el in root.iter("image"):
        img_name = image_el.attrib["name"]
        img_path = folder / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        shapes = list(image_el.iter("polygon")) + list(image_el.iter("box"))
        stem = Path(img_name).stem
        for i, shape_el2 in enumerate(shapes):
            label = shape_el2.attrib.get("label", "").strip().lower()
            if label in SKIP_LABELS:
                continue
            class_name = LABEL_TO_CLASS.get(label)
            if class_name is None:
                continue
            bbox = shape_bbox(shape_el2)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            x0 = max(0, int(x0) - PAD)
            y0 = max(0, int(y0) - PAD)
            x1 = min(w, int(x1) + PAD)
            y1 = min(h, int(y1) + PAD)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img[y0:y1, x0:x1]
            out_dir = out_root / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{stem}_{i}.jpg"), crop)
            saved += 1
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the RV-PBS-main folder")
    ap.add_argument("--output", required=True, help="Output folder for the cropped dataset")
    ap.add_argument("--folders", nargs="*", default=None,
                     help="Specific subfolders to process (default: the 10 class folders)")
    args = ap.parse_args()

    input_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    folders = args.folders or DEFAULT_SOURCE_FOLDERS
    total = 0
    for fname in tqdm(folders, desc="folders"):
        folder = input_root / fname
        if not folder.is_dir():
            print(f"  ! missing folder: {folder}")
            continue
        n = crop_folder(folder, out_root)
        print(f"{fname}: {n} crops")
        total += n
    print(f"\nTotal crops written: {total}  ->  {out_root}")


if __name__ == "__main__":
    main()
