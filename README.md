# NucleiMapper AI Backend

This is the real, trained-CNN backend for NucleiMapper, sitting behind the
existing heuristic JS pipeline:

```
Website → Upload Image → Python Backend → Model → Prediction → Result
   (HTML)      (browser)      (FastAPI)     (ResNet18)   (JSON)   (UI card)
```

It classifies a single white blood cell crop into one of the 10 **RV-PBS**
classes: `BAND CELLS`, `BASOPHILS`, `BLAST CELLS`, `EOSINOPHILS`,
`LYMPHOCYTES`, `METAMYELOCYTES`, `MONOCYTES`, `MYELOCYTE`, `NEUTROPHILS`,
`PROMYELOCYTES`.

This whole pipeline was test-run end to end while building it (crop ->
split -> a few real training batches -> checkpoint -> FastAPI -> a real
`curl` upload returning a JSON prediction), so the wiring is confirmed
working. You still need to run a full training job yourself — see Step 3.

---

## 0. Setup

```bash
cd backend
pip install -r requirements.txt
```

(Use a venv if you like. PyTorch will try to download ImageNet-pretrained
ResNet18 weights from `download.pytorch.org` the first time you train —
make sure that's reachable from wherever you run training.)

---

## 1. Crop the raw RV-PBS slides into single-cell images

Your downloaded `RV-PBS-main` folder contains whole smear photos
(4032×3024) plus a CVAT `annotations.xml` per class folder, with one
polygon/box per annotated WBC. `crop_dataset.py` reads every shape's own
label and bbox-crops it straight out of the original photo — **no alpha
masking**, so the crop keeps its natural background. That matters: a
masked, black-background cutout looks nothing like a real upload, and a
model trained on those generalizes poorly. A plain bbox crop (with a little
padding) is much closer to what a real user will actually hand the model.

```bash
python data_prep/crop_dataset.py \
    --input  /path/to/RV-PBS-main \
    --output ./classification_data
```

Note: `TEST_MIXED` has **no** `annotations.xml` — it's unlabeled demo
slides, not a labeled test set. Don't try to crop it for evaluation; use a
held-out split of the 10 labeled folders instead (Step 2 does this for
you).

Expect roughly 700-800 crops in total across the 10 classes from the
~585 labeled slides (more than one WBC is sometimes annotated per slide).
The classes are naturally imbalanced (Blast Cells and Neutrophils have
noticeably more crops than Basophils or Metamyelocytes) — `train.py`
already compensates for this with inverse-frequency class weights, but
more data for the rare classes will always help.

## 2. Split into train/val

```bash
python data_prep/split_dataset.py \
    --input  ./classification_data \
    --output ./data \
    --val-frac 0.15
```

This produces `./data/train/<CLASS>/...` and `./data/val/<CLASS>/...`,
stratified per class.

## 3. Train

```bash
python train.py --data ./data --epochs 25 --arch resnet18 --out ./checkpoints
```

- `--arch resnet18` (default) or `--arch efficientnet_b0`
- `--pretrained` (default) starts from ImageNet weights; `--no-pretrained`
  trains from random initialization (use this if your network blocks
  `download.pytorch.org`)
- `--workers N` controls DataLoader worker processes (set to `0` on
  low-core machines — worker spin-up overhead can dominate runtime on a
  single-core box)
- `--resume PATH` continues training from an existing checkpoint instead
  of starting over
- Fine-tunes from ImageNet weights, AdamW + cosine LR schedule
- Class-weighted cross-entropy to counter the dataset imbalance
- Saves `checkpoints/best_model.pt` (best val accuracy), `last_model.pt`,
  and `checkpoints/class_names.json` (the class-index ordering the model
  was trained with — required at inference time)
- With only ~700 crops total, expect val accuracy to be a useful signal
  but not state-of-the-art out of the box. The dataset's own paper uses
  domain adaptation across multiple source datasets to push accuracy
  further — a good next step once this pipeline is working end to end.

## 3.5. Evaluate (don't skip this)

Overall accuracy is not enough on an imbalanced dataset like this. Run the
full evaluation report — confusion matrix, precision/recall/F1, **and
per-class accuracy** — plus an automatic warning for any class whose
recall falls below 50% (because overall accuracy alone hides exactly that):

```bash
python evaluate.py --data ./data/val \
    --checkpoint ./checkpoints/best_model.pt \
    --class-names ./checkpoints/class_names.json \
    --out ./eval_reports
```

This prints a full per-class table to the terminal, writes
`eval_reports/eval_report.json` (the same numbers, machine-readable), and
saves `eval_reports/confusion_matrix.png` — a labeled heatmap, because a
table of numbers is harder to scan for "which row doesn't sit on its own
diagonal" than a glance at a heatmap is.

Note the distinction between two metrics that look similar but aren't:
- **Recall** — of this class's own samples, how many did the model get right?
- **Per-class accuracy** — one-vs-rest (TP+TN)/total — also rewards
  correctly keeping *other* classes' samples out of this one.

A class can score well on one and badly on the other; that's why both are
reported instead of collapsing to a single per-class number.

## 4. Run the API server

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

- `GET /health` → `{"status": "ok", "classes": [...]}`
- `POST /predict` (multipart form, field name `file`) → 
  ```json
  {
    "class": "NEUTROPHILS",
    "confidence": 81.42,
    "probabilities": { "BAND CELLS": 2.1, "BASOPHILS": 0.4, ... }
  }
  ```

Environment variables (all optional):
| var | default |
|---|---|
| `CHECKPOINT_PATH` | `./checkpoints/best_model.pt` |
| `CLASS_NAMES_PATH` | `./checkpoints/class_names.json` |
| `MODEL_ARCH` | `resnet18` |

## 5. Frontend

`frontend_integration/nucleimapper_v3_ai_backend.html` has had its
**heuristic classification logic removed entirely.** The old `classify()`
function — the math that looked at lobe count, concavity, elongation,
cytoplasm color ratio etc. and guessed a cell type — is gone. So is the
fallback that quietly kept showing that guess if the AI backend was
unreachable.

What's left, by design:

- **Detection (kept):** simple color-threshold masking still finds the
  WBC region and draws the red dotted box — that's localization, not a
  diagnosis, so it stays.
- **Visualization panels (kept):** Isolated Nucleus, Binary mask, Granule
  Density Map are still computed the same way — visual aids for a human
  to look at, not a decision the app makes for you.
- **Nucleus Area / Lobe Count / Circularity (kept):** still measured and
  displayed in Extracted Features, but purely as numbers — nothing reads
  them anymore to pick a cell type.
- **Classification (changed):** the box starts unlabeled and the
  Classification Result card shows **"Awaiting Backend…"** the moment
  detection finishes. The AI backend gets the original-resolution crop of
  that detected box (the scale-fix from before — model trained on tight
  single-cell crops, not full slide photos). Only when it actually
  responds does the box get a label and the result card get a real class
  + confidence.
- **If the backend is unreachable (changed):** the card explicitly says
  **"Backend Not Connected"** in red, with the reason and a reminder to
  start `uvicorn`. No class is shown, no guess is made. Pipeline step 4
  ("AI Backend Classification") turns red instead of green.

`AI_BACKEND_URL` is a constant near the top of the script — edit it
directly if you're not running `uvicorn` on `localhost:8000`.

---

## Results from an actual training run (included in this package)

`checkpoints/best_model.pt` in this package isn't a placeholder — it's a
real checkpoint trained on the cropped RV-PBS data, from random
initialization (no ImageNet pretraining was available in the environment
this was trained in — see note below), ResNet18, image size 160, ~21
cumulative epochs:

```
              precision  recall  f1-score  support
BAND CELLS         0.70    0.78      0.74        9
BASOPHILS          1.00    1.00      1.00        4
BLAST CELLS        0.95    0.82      0.88       22
EOSINOPHILS        0.92    0.92      0.92       13
LYMPHOCYTES        0.75    1.00      0.86        9
METAMYELOCYTES     0.18    0.50      0.27        4
MONOCYTES          0.83    0.62      0.71        8
MYELOCYTE          1.00    0.21      0.35       14
NEUTROPHILS        0.83    0.88      0.86       17
PROMYELOCYTES      0.50    0.75      0.60        8

accuracy                             0.75      108
```

**75% overall validation accuracy** (vs. 10% chance on a 10-class problem) —
but that single number hides a real problem. Running the full
`evaluate.py` report (see Step 3.5) on this checkpoint surfaces it
immediately:

```
⚠ WARNING: 1 class(es) below 50% recall despite 75% overall accuracy.
Don't trust the headline number alone:
   - MYELOCYTE: 21% recall (support=14)
```

MYELOCYTE's *per-class accuracy* is 90% (it rarely causes false positives
in other classes), but its *recall* is only 21% — most actual Myelocyte
crops get predicted as something else (mostly Metamyelocyte, also Band
Cells and Promyelocytes). That's exactly the gap between the two metrics
described in Step 3.5: a class can look fine on one and be quietly
failing on the other. This makes biological sense too — Myelocyte and
Metamyelocyte are adjacent stages in the same granulocyte maturation
continuum and look genuinely similar, and both have the fewest training
crops (28 and 32 respectively) to learn the distinction from.

To push past this:
- Train with `--pretrained` (default) instead of `--no-pretrained` on a
  machine that can reach `download.pytorch.org` — ImageNet features
  should help the most exactly on these visually-similar, data-poor
  classes.
- More crops for Basophils/Metamyelocytes would help more than
  architecture changes at this point.
- `--resume ./checkpoints/best_model.pt` lets you keep extending training
  in short bursts instead of restarting from scratch (useful on a slow
  CPU-only machine, or just to try a different LR partway through):
  ```bash
  python train.py --data ./data --epochs 4 --resume ./checkpoints/best_model.pt \
      --lr 5e-5 --out ./checkpoints
  ```
  Note: each call restarts the cosine LR schedule from `--lr`, so use a
  progressively smaller `--lr` on each resumed call rather than the
  original training LR, or you'll see a temporary accuracy dip at the
  start of every resumed run.

---

## Project layout

```
backend/
├── requirements.txt
├── common/
│   ├── model.py          # build_model() / load_checkpoint()
│   └── transforms.py     # shared train/eval image transforms
├── data_prep/
│   ├── crop_dataset.py   # raw RV-PBS slides -> single-cell crops
│   └── split_dataset.py  # crops -> stratified train/val split
├── train.py               # fine-tune ResNet18/EfficientNet-B0
├── evaluate.py             # confusion matrix, precision/recall/F1, per-class accuracy, weak-class warnings
├── api/
│   ├── inference.py       # Predictor class (shared by app.py & scripts)
│   └── app.py              # FastAPI server (/health, /predict)
└── frontend_integration/
    └── nucleimapper_v3_ai_backend.html   # your frontend + AI panel wired in
```
