#!/usr/bin/env python3
"""
train.py - fine-tunes a torchvision CNN backbone on the cropped RV-PBS
classification dataset produced by data_prep/crop_dataset.py + split_dataset.py.

Run from the backend/ directory (so the `common` package resolves):

    python train.py --data ./data --epochs 25 --arch resnet18 --out ./checkpoints

Outputs (in --out):
    best_model.pt      - state_dict with the highest val accuracy seen
    last_model.pt       - state_dict from the final epoch
    class_names.json    - ordered list of class names matching the model's output indices
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from common.model import build_model
from common.transforms import get_train_transforms, get_eval_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder containing train/ and val/ subfolders")
    ap.add_argument("--out", default="./checkpoints")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    ap.add_argument("--resume", default=None, help="Path to an existing checkpoint to continue training from")
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True,
                     help="Start from ImageNet weights (default). Needs internet access "
                          "to download.pytorch.org the first time.")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                     help="Train from random initialization instead (use this if your "
                          "network blocks download.pytorch.org).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = Path(args.data)
    train_ds = datasets.ImageFolder(data_root / "train", transform=get_train_transforms(args.image_size))
    val_ds = datasets.ImageFolder(data_root / "val", transform=get_eval_transforms(args.image_size))

    class_names = train_ds.classes
    assert class_names == val_ds.classes, "train/val class folders must match exactly"
    print("Classes:", class_names)

    # RV-PBS is naturally imbalanced (e.g. ~150 Blast Cell crops vs ~28 Basophil
    # crops) - inverse-frequency class weights stop the model from just
    # always predicting the majority classes.
    counts = [0] * len(class_names)
    for _, y in train_ds.samples:
        counts[y] += 1
    print("Train counts:", dict(zip(class_names, counts)))
    weights = torch.tensor([1.0 / c if c > 0 else 0.0 for c in counts], dtype=torch.float32)
    weights = weights / weights.sum() * len(class_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(len(class_names), arch=args.arch, pretrained=args.pretrained).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))

    best_acc = 0.0
    if args.resume:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        best_acc = correct / max(1, total)
        print(f"Resumed model's current val_acc: {best_acc:.4f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running_loss / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(1, total)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> saved new best ({best_acc:.4f})")

    torch.save(model.state_dict(), out_dir / "last_model.pt")
    print(f"\nDone. Best val acc: {best_acc:.4f}")
    print(f"Checkpoints + class_names.json saved to {out_dir}")


if __name__ == "__main__":
    main()
