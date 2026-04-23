from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final summary. Useful when testing many images.",
    )
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


def infer_true_label(image_path: Path, class_names: list[str]) -> int | None:
    name = image_path.stem.lower()
    for index, class_name in enumerate(class_names):
        class_key = class_name.lower()
        if name == class_key or name.startswith(f"{class_key}_") or name.startswith(f"{class_key}-"):
            return index
    return None


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


def print_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> None:
    first_col_width = max(len("true\\pred"), *(len(name) for name in class_names))
    col_width = max(9, *(len(name) for name in class_names))

    header = f"{'true\\pred':<{first_col_width}}"
    for class_name in class_names:
        header += f" {class_name:>{col_width}}"
    print(header)

    for row_index, class_name in enumerate(class_names):
        row = f"{class_name:<{first_col_width}}"
        for col_index in range(len(class_names)):
            row += f" {int(matrix[row_index, col_index]):>{col_width}}"
        print(row)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = collect_bmp_images(args.image, args.image_dir)
    model, checkpoint = load_model(args.checkpoint, device)
    class_names = checkpoint["class_names"]
    predicted_counts = {class_name: 0 for class_name in class_names}
    true_counts = {class_name: 0 for class_name in class_names}
    correct_counts = {class_name: 0 for class_name in class_names}
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    evaluated_count = 0
    correct_count = 0
    unknown_label_count = 0

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
        pred_index = indices[0]
        pred_label = class_names[pred_index]
        true_index = infer_true_label(image_path, class_names)
        predicted_counts[pred_label] += 1

        if true_index is None:
            unknown_label_count += 1
            result_text = "true=unknown"
        else:
            evaluated_count += 1
            true_label = class_names[true_index]
            true_counts[true_label] += 1
            confusion[true_index, pred_index] += 1
            if pred_index == true_index:
                correct_count += 1
                correct_counts[true_label] += 1
            result_text = f"true={true_label} pred={pred_label}"

        if not args.quiet:
            print(f"Image: {image_path.resolve()}")
            print(f"Result: {result_text}")
            for rank, (score, index) in enumerate(zip(scores, indices), start=1):
                print(f"Top{rank}: {class_names[index]} ({score:.4f})")
            print("-" * 60)

    print("Prediction summary:")
    for class_name in class_names:
        print(f"{class_name}: {predicted_counts[class_name]}")

    if evaluated_count == 0:
        print("No ground-truth labels were inferred from filenames.")
        print("Expected filename examples: weapon_001.bmp, supplies_001.bmp, vehicle_001.bmp")
        return

    accuracy = correct_count / evaluated_count
    print(f"Evaluated images: {evaluated_count}")
    print(f"Unknown-label images: {unknown_label_count}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{evaluated_count})")

    print("Per-class accuracy:")
    for class_name in class_names:
        total = true_counts[class_name]
        correct = correct_counts[class_name]
        if total == 0:
            print(f"{class_name}: N/A (0/0)")
        else:
            print(f"{class_name}: {correct / total:.4f} ({correct}/{total})")

    print("Confusion matrix:")
    print_confusion_matrix(confusion, class_names)


if __name__ == "__main__":
    main()
