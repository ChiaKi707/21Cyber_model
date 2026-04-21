from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from dataset import preprocess_bgr
from models import TinyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image prediction with a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Input BMP image path.")
    group.add_argument("--image-dir", type=str, help="Directory containing BMP images for batch prediction.")
    parser.add_argument("--topk", type=int, default=3, help="How many classes to print.")
    return parser.parse_args()


def collect_bmp_images(image_path: str | None, image_dir: str | None) -> list[Path]:
    if image_path is not None:
        path = Path(image_path)
        if path.suffix.lower() != ".bmp":
            raise ValueError(f"Only BMP images are supported for prediction, got: {path.suffix}")
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return [path]

    folder = Path(image_dir)  # type: ignore[arg-type]
    if not folder.exists():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory path, got: {folder}")

    image_paths = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".bmp")
    if not image_paths:
        raise ValueError(f"No BMP images found under: {folder}")
    return image_paths


def load_model(checkpoint_path: str, device: torch.device) -> tuple[TinyClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TinyClassifier(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_one_image(
    model: TinyClassifier,
    checkpoint: dict,
    image_path: Path,
    device: torch.device,
    topk: int,
) -> tuple[list[float], list[int]]:
    if image_path.suffix.lower() != ".bmp":
        raise ValueError(f"Only BMP images are supported for prediction, got: {image_path.suffix}")

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

    class_names = checkpoint["class_names"]
    topk = min(topk, len(class_names))
    scores, indices = torch.topk(probs, k=topk)
    return scores.tolist(), indices.tolist()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = collect_bmp_images(args.image, args.image_dir)
    model, checkpoint = load_model(args.checkpoint, device)
    class_names = checkpoint["class_names"]
    predicted_counts = {class_name: 0 for class_name in class_names}

    print(f"Device: {device}")
    print(f"Total BMP images: {len(image_paths)}")

    for image_path in image_paths:
        scores, indices = predict_one_image(
            model=model,
            checkpoint=checkpoint,
            image_path=image_path,
            device=device,
            topk=args.topk,
        )
        predicted_counts[class_names[indices[0]]] += 1

        print(f"Image: {image_path.resolve()}")
        for rank, (score, index) in enumerate(zip(scores, indices), start=1):
            print(f"Top{rank}: {class_names[index]} ({score:.4f})")
        print("-" * 60)

    print("Prediction summary:")
    for class_name in class_names:
        print(f"{class_name}: {predicted_counts[class_name]}")


if __name__ == "__main__":
    main()
