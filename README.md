> ⚠️ **Proprietary Software Licensing Notice**
>
> NucleiMapper is proprietary software. This repository is publicly accessible solely for portfolio, educational, and research demonstration purposes.
>
> The source code is provided for viewing only. No permission is granted to use, copy, modify, merge, publish, distribute, sublicense, create derivative works from, or commercially exploit any part of this repository without prior written permission from the copyright holder.
>
> Viewing this repository on GitHub does **not** grant any license or other rights to reuse the source code. Please refer to the accompanying `LICENSE` file for the complete legal terms and conditions.


# 🧬 NucleiMapper

**NucleiMapper is a PyTorch-based white blood cell morphology classifier trained on real peripheral blood smear images, combining CNN inference with nucleus and granule feature visualization.**

It connects a browser-based morphology interface to a Python AI backend:

```text
Blood Smear Image
        ↓
WBC Localization
        ↓
Original-Resolution Cell Crop
        ↓
FastAPI Backend
        ↓
ResNet18 / EfficientNet-B0
        ↓
10-Class WBC Prediction
        ↓
Confidence + Morphology UI
```

> \*\*Research and educational project only. NucleiMapper is not a medical diagnostic device and must not be used for clinical decision-making.\*\*

\---

## 🔬 What NucleiMapper Does

NucleiMapper classifies a detected white blood cell crop into one of 10 RV-PBS morphology classes:

* Band Cells
* Basophils
* Blast Cells
* Eosinophils
* Lymphocytes
* Metamyelocytes
* Monocytes
* Myelocytes
* Neutrophils
* Promyelocytes

The frontend also visualizes:

* isolated nucleus
* binary nucleus mask
* granule density map
* nucleus area
* estimated lobe count
* circularity

These morphology measurements are **visual and descriptive features**. They do not determine the final class prediction. Classification is performed by the trained CNN backend.

\---

## 🧠 AI Pipeline

```text
RV-PBS Slides + CVAT Annotations
              ↓
       crop\_dataset.py
              ↓
      Single-WBC Crops
              ↓
       split\_dataset.py
              ↓
         Train / Val
              ↓
           train.py
              ↓
  Weighted CNN Classification
              ↓
       best\_model.pt
              ↓
     FastAPI /predict API
              ↓
       NucleiMapper UI
```

The training pipeline supports:

* PyTorch and torchvision
* ResNet18
* EfficientNet-B0
* ImageNet transfer learning
* inverse-frequency class weighting
* weighted cross-entropy loss
* AdamW optimizer
* cosine annealing learning-rate scheduling
* checkpoint resume
* best-validation checkpoint tracking

Class weighting is used because the RV-PBS crops are naturally imbalanced across WBC classes.

\---

## 📊 Evaluation Results

A real ResNet18 training run achieved:

> \*\*75% overall validation accuracy on 108 held-out cell crops.\*\*

```text
CLASS             PRECISION   RECALL   F1-SCORE   SUPPORT
BAND CELLS           0.70      0.78      0.74        9
BASOPHILS            1.00      1.00      1.00        4
BLAST CELLS          0.95      0.82      0.88       22
EOSINOPHILS          0.92      0.92      0.92       13
LYMPHOCYTES          0.75      1.00      0.86        9
METAMYELOCYTES       0.18      0.50      0.27        4
MONOCYTES            0.83      0.62      0.71        8
MYELOCYTE            1.00      0.21      0.35       14
NEUTROPHILS          0.83      0.88      0.86       17
PROMYELOCYTES        0.50      0.75      0.60        8

ACCURACY                                  0.75      108
```

### ⚠️ Known Weakness: Myelocyte Recall

The model reached **21% recall for Myelocytes** despite 75% overall accuracy.

This is intentionally documented rather than hidden behind the headline metric.

Myelocytes were frequently confused with nearby granulocytic maturation stages, especially Metamyelocytes. The small number of training crops for these visually similar classes is a major limitation.

`evaluate.py` reports:

* precision
* recall
* F1-score
* confusion matrix
* one-vs-rest per-class accuracy
* automatic warnings for classes below 50% recall

The project therefore treats overall accuracy as **one metric, not proof of reliable classification across every class**.

\---

## 📁 Project Structure

```text
NucleiMapper/
├── api/
│   ├── app.py
│   └── inference.py
│
├── common/
│   ├── model.py
│   └── transforms.py
│
├── data\_prep/
│   ├── crop\_dataset.py
│   └── split\_dataset.py
│
├── eval\_reports/
│
├── frontend\_integration/
│   └── nucleimapper\_v3\_ai\_backend.html
│
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── LICENSE
```

Training datasets, virtual environments, and model checkpoints are intentionally excluded from the repository.

\---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ChitrarupGuha/NucleiMapper.git
cd NucleiMapper
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\\Scripts\\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

\---

## 🧫 Dataset Preparation

NucleiMapper uses the **RV-PBS (Ramakrishna Vivekananda Peripheral Blood Smear) dataset** as its training data source.

The dataset itself is **not redistributed in this repository**.

Obtain RV-PBS from its original project, then crop the annotated WBC regions:

```bash
python data\_prep/crop\_dataset.py \\
    --input /path/to/RV-PBS-main \\
    --output ./classification\_data
```

`crop\_dataset.py` reads the CVAT annotations and extracts bounding-box crops while retaining the natural smear background.

The crop is not alpha-masked because artificial black-background cutouts differ significantly from expected real-world image inputs.

### Create the Train / Validation Split

```bash
python data\_prep/split\_dataset.py \\
    --input ./classification\_data \\
    --output ./data \\
    --val-frac 0.15
```

The result is a stratified directory structure:

```text
data/
├── train/
│   └── <CLASS>/
└── val/
    └── <CLASS>/
```

\---

## 🚂 Train the Model

```bash
python train.py \\
    --data ./data \\
    --epochs 25 \\
    --arch resnet18 \\
    --out ./checkpoints
```

Available architectures:

```text
resnet18
efficientnet\_b0
```

ImageNet pretrained weights are enabled by default.

To train from random initialization:

```bash
python train.py --data ./data --no-pretrained
```

To resume from an existing checkpoint:

```bash
python train.py \\
    --data ./data \\
    --epochs 4 \\
    --resume ./checkpoints/best\_model.pt \\
    --lr 5e-5 \\
    --out ./checkpoints
```

The training pipeline saves:

```text
checkpoints/
├── best\_model.pt
├── last\_model.pt
└── class\_names.json
```

`best\_model.pt` tracks the highest validation accuracy observed during the run.

\---

## 📈 Evaluate the Model

```bash
python evaluate.py \\
    --data ./data/val \\
    --checkpoint ./checkpoints/best\_model.pt \\
    --class-names ./checkpoints/class\_names.json \\
    --out ./eval\_reports
```

Generated outputs include:

```text
eval\_reports/
├── eval\_report.json
└── confusion\_matrix.png
```

The evaluator automatically warns when a class has recall below 50%.

This matters because strong overall accuracy can coexist with severe class-specific failure.

\---

## 🚀 Run the FastAPI Backend

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "classes": \["BAND CELLS", "BASOPHILS", "..."]
}
```

### Prediction

```http
POST /predict
```

Upload an image using multipart form data with the field name `file`.

Example response:

```json
{
  "class": "EOSINOPHILS",
  "confidence": 63.87,
  "probabilities": {
    "BAND CELLS": 1.24,
    "BASOPHILS": 0.51,
    "EOSINOPHILS": 63.87
  }
}
```

### Optional Environment Variables

|Variable|Default|
|-|-|
|`CHECKPOINT\_PATH`|`./checkpoints/best\_model.pt`|
|`CLASS\_NAMES\_PATH`|`./checkpoints/class\_names.json`|
|`MODEL\_ARCH`|`resnet18`|

\---

## 🖥️ Frontend Integration

The NucleiMapper frontend no longer performs heuristic WBC classification.

The browser still handles:

* WBC region localization
* nucleus visualization
* binary mask generation
* granule density visualization
* morphology feature display

The detected WBC crop is sent to the FastAPI backend for CNN inference.

If the backend is unavailable, the UI explicitly displays:

```text
Backend Not Connected
```

No fallback class or heuristic prediction is silently shown.

This separation prevents the interface from presenting a rule-based guess as an AI model result.

\---

## 🧪 Current Limitations

NucleiMapper currently has several important limitations:

* trained on a relatively small classification dataset
* class imbalance remains significant
* low Myelocyte recall
* visually adjacent maturation stages are difficult to separate
* validation results are not equivalent to external clinical validation
* the model has not been approved or evaluated as a medical device

Future work may include transfer learning experiments, additional rare-class data, external dataset evaluation, calibration analysis, and improved WBC localization.

\---

## 📚 Dataset Attribution

NucleiMapper uses the **RV-PBS (Ramakrishna Vivekananda Peripheral Blood Smear) dataset** for model training.

Original project:

**Jimut123/RV-PBS**

Dataset authors:

* Jimut Bahan Pal
* Aniket Bhattacharyea
* Debasis Banerjee
* Br. Tamal Maharaj

This repository does **not** redistribute the RV-PBS dataset. Users should obtain the data from the original source and comply with its applicable license terms.

### Paper

> \*Advancing instance segmentation and WBC classification in peripheral blood smear through domain adaptation: A study on PBC and the novel RV-PBS datasets.\*

**Expert Systems with Applications, 2024**

DOI: `10.1016/j.eswa.2024.123660`

If you use RV-PBS in your own research, please follow the citation guidance provided by the original dataset authors.

\---

## 📜 License

Copyright © 2026 Chitrarup Guha.
All Rights Reserved.

NucleiMapper is proprietary software.

No permission is granted to use, copy, modify, merge, publish, distribute,
sublicense, sell, create derivative works from, or commercially exploit any
part of this repository without prior written permission from the copyright holder.

Viewing the source code on GitHub does **not** grant permission to reuse it.

The complete legal terms are provided in the accompanying `LICENSE` file.

### Dataset Notice

NucleiMapper does **not** claim ownership of the RV-PBS dataset.

The RV-PBS dataset remains the intellectual property of its original authors and
must be obtained separately from its official source under its own licensing terms.

This repository intentionally does **not** redistribute the dataset.

## 👨‍💻 Author

**Chitrarup Guha**

Built as a computer vision and medical morphology exploration project combining a BCA software background with laboratory-science knowledge.

\---

**NucleiMapper** 🔬🧬  
*Mapping morphology. Measuring model limits.*



## License

This project is proprietary software.

The source code is publicly visible for demonstration and portfolio purposes only.

No permission is granted to use, copy, modify, distribute,
or create derivative works without explicit written permission.
