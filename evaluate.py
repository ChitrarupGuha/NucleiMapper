#!/usr/bin/env python3
"""
evaluate.py - full evaluation report for a trained checkpoint.

This exists because overall accuracy lies by omission. A model can post
90% overall accuracy on an imbalanced dataset while one rare class (the
one with 4 support samples instead of 100) is sitting at 20% recall - and
the single headline number will never tell you that. This script prints
and saves everything needed to actually catch that:

    - Confusion matrix (printed as a table AND saved as a PNG heatmap)
    - Precision / Recall / F1 per class
    - Per-class accuracy (one-vs-rest: TP+TN over everything - NOT the
      same number as recall, see per_class_accuracy() below)
    - An explicit warning list of any class under a recall threshold

Run from the backend/ directory:

    python evaluate.py --data ./data/val --checkpoint ./checkpoints/best_model.pt \
        --class-names ./checkpoints/class_names.json --out ./eval_reports

NOTE: RV-PBS's "TEST_MIXED" folder has no annotations.xml (it's unlabeled
demo slides) - it can't be used as a labeled test set. Evaluate against a
held-out labeled split instead (e.g. data/val/).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix

from common.model import load_checkpoint
from common.transforms import get_eval_transforms


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """One-vs-rest accuracy per class: (TP + TN) / total.

    This is a genuinely different number from recall. Recall only asks
    "of this class's own samples, how many did we get right?". Per-class
    accuracy also rewards correctly keeping OTHER classes' samples out of
    this class - so a class the model rarely predicts (low recall) but
    also rarely confuses other classes into can still score deceptively
    high here. Both numbers are reported because each hides a different
    failure mode; neither replaces the confusion matrix itself.
    """
    total = cm.sum()
    accs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        accs.append((tp + tn) / total)
    return np.array(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--class-names", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", default="./eval_reports",
                     help="Where to save eval_report.json and confusion_matrix.png")
    ap.add_argument("--flag-below", type=float, default=0.5,
                     help="Warn about any class whose recall falls below this fraction")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = json.loads(Path(args.class_names).read_text())
    model = load_checkpoint(args.checkpoint, len(class_names), arch=args.arch, device=device)

    ds = datasets.ImageFolder(args.data, transform=get_eval_transforms(args.image_size))
    assert ds.classes == class_names, f"class mismatch: {ds.classes} vs {class_names}"
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            pred = out.argmax(1).cpu()
            y_true += y.tolist()
            y_pred += pred.tolist()
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    pca = per_class_accuracy(cm)
    overall_acc = float((y_true == y_pred).mean())

    # ---- printed table ----
    print(f"\nOverall accuracy: {overall_acc:.4f}  ({int((y_true == y_pred).sum())}/{len(y_true)})\n")
    header = f"{'CLASS':<16}{'PRECISION':>10}{'RECALL':>10}{'F1':>10}{'PER-CLASS ACC':>16}{'SUPPORT':>10}"
    print(header)
    print("-" * len(header))
    warnings = []
    for i, c in enumerate(class_names):
        r = report_dict[c]
        print(f"{c:<16}{r['precision']:>10.2f}{r['recall']:>10.2f}{r['f1-score']:>10.2f}"
              f"{pca[i]:>16.2f}{int(r['support']):>10}")
        if r['recall'] < args.flag_below:
            warnings.append((c, r['recall'], int(r['support'])))
    print("-" * len(header))
    macro, weighted = report_dict['macro avg'], report_dict['weighted avg']
    print(f"{'macro avg':<16}{macro['precision']:>10.2f}{macro['recall']:>10.2f}{macro['f1-score']:>10.2f}")
    print(f"{'weighted avg':<16}{weighted['precision']:>10.2f}{weighted['recall']:>10.2f}{weighted['f1-score']:>10.2f}")

    if warnings:
        print(f"\n⚠ WARNING: {len(warnings)} class(es) below {args.flag_below:.0%} recall despite "
              f"{overall_acc:.0%} overall accuracy. Don't trust the headline number alone:")
        for c, r, n in warnings:
            print(f"   - {c}: {r:.0%} recall (support={n})")
    else:
        print(f"\nNo class fell below {args.flag_below:.0%} recall.")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    col_w = max(len(c) for c in class_names) + 2
    print(" " * col_w + "".join(f"{c[:8]:>10}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"{c:<{col_w}}" + "".join(f"{v:>10}" for v in cm[i]))

    # ---- save artifacts ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_json = {
        "overall_accuracy": overall_acc,
        "per_class": {
            c: {
                "precision": report_dict[c]["precision"],
                "recall": report_dict[c]["recall"],
                "f1": report_dict[c]["f1-score"],
                "per_class_accuracy": float(pca[i]),
                "support": int(report_dict[c]["support"]),
            } for i, c in enumerate(class_names)
        },
        "macro_avg": macro,
        "weighted_avg": weighted,
        "confusion_matrix": cm.tolist(),
        "class_order": class_names,
        "weak_classes_below_threshold": [{"class": c, "recall": r, "support": n} for c, r, n in warnings],
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report_json, indent=2))
    print(f"\nFull report saved to {out_dir / 'eval_report.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (overall acc={overall_acc:.2%})")
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        print(f"Confusion matrix heatmap saved to {out_dir / 'confusion_matrix.png'}")
    except ImportError:
        print("matplotlib not installed - skipping confusion_matrix.png (pip install matplotlib to get it)")


if __name__ == "__main__":
    main()
