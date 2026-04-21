from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from dataset import preprocess_bgr
from models import TinyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image prediction with a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--image", type=str, required=True, help="Input BMP image path.")
    parser.add_argument("--topk", type=int, default=3, help="How many classes to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = Path(args.image)

    if image_path.suffix.lower() != ".bmp":
        raise ValueError(f"Only BMP images are supported for prediction, got: {image_path.suffix}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    model = TinyClassifier(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    tensor = preprocess_bgr(
        image=image,
        input_size=int(checkpoint["input_size"]),
        mean=list(checkpoint["mean"]),
        std=list(checkpoint["std"]),
        augment=False,
    ).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(args.topk, len(class_names))
    scores, indices = torch.topk(probs, k=topk)
    print(f"Image: {image_path.resolve()}")
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        print(f"Top{rank}: {class_names[index]} ({score:.4f})")


if __name__ == "__main__":
    main()
