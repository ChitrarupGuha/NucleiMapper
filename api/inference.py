"""
inference.py - single-image prediction helper, shared by the FastAPI app
(app.py) and any CLI/testing script. Keeping this separate from app.py means
you can also script batch predictions without spinning up a server.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from common.model import load_checkpoint
from common.transforms import get_eval_transforms


class Predictor:
    def __init__(self, checkpoint_path: str, class_names_path: str,
                 arch: str = "resnet18", image_size: int = 224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = json.loads(Path(class_names_path).read_text())
        self.model = load_checkpoint(checkpoint_path, len(self.class_names), arch=arch, device=self.device)
        self.transform = get_eval_transforms(image_size)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
        idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return {
            "class": self.class_names[idx],
            "confidence": round(probs[idx] * 100, 2),
            "probabilities": {c: round(p * 100, 2) for c, p in zip(self.class_names, probs)},
        }
