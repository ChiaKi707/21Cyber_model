from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_INPUT_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch prediction with an ncnn param/bin model.")
    parser.add_argument("--param", type=str, required=True, help="Path to ncnn .param file.")
    parser.add_argument("--bin", type=str, required=True, help="Path to ncnn .bin file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional PyTorch checkpoint. Used only to read labels/input_size/mean/std.",
    )
    parser.add_argument("--labels", type=str, default="labels.txt", help="Labels file used when checkpoint is absent.")
    parser.add_argument("--input-name", type=str, default="input", help="ncnn input blob name.")
    parser.add_argument("--output-name", type=str, default="logits", help="ncnn output blob name.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Input BMP image path.")
    group.add_argument("--image-dir", type=str, help="Directory containing BMP images for batch prediction.")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE, help="Model input size.")
    parser.add_argument("--mean", type=float, nargs=3, default=None, help="BGR mean values in 0-1 range.")
    parser.add_argument("--std", type=float, nargs=3, default=None, help="BGR std values in 0-1 range.")
    parser.add_argument("--topk", type=int, default=3, help="How many classes to print.")
    parser.add_argument("--quiet", action="store_true", help="Only print final summary.")
    parser.add_argument("--print-logits", action="store_true", help="Print raw logits for each image.")
    return parser.parse_args()


def import_ncnn() -> Any:
    try:
        import ncnn  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Python ncnn package is not installed in this environment. "
            "Install/enable ncnn Python bindings first, or run this script in an environment that provides `import ncnn`."
        ) from exc
    return ncnn


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


def load_labels(labels_path: str | Path) -> list[str]:
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"Labels file is empty: {path}")
    return labels


def load_checkpoint_metadata(checkpoint_path: str | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return {}

    try:
        import torch
    except ImportError as exc:
        raise ImportError("--checkpoint requires PyTorch to be available in the current environment.") from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        "class_names": checkpoint.get("class_names"),
        "input_size": checkpoint.get("input_size"),
        "mean": checkpoint.get("mean"),
        "std": checkpoint.get("std"),
    }


def resolve_runtime_metadata(args: argparse.Namespace) -> tuple[list[str], int, list[float], list[float]]:
    metadata = load_checkpoint_metadata(args.checkpoint)

    class_names = metadata.get("class_names")
    if not class_names:
        class_names = load_labels(args.labels)

    input_size = int(metadata.get("input_size") or args.input_size)
    mean = list(args.mean if args.mean is not None else metadata.get("mean") or [0.406, 0.456, 0.485])
    std = list(args.std if args.std is not None else metadata.get("std") or [0.225, 0.224, 0.229])

    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must contain exactly 3 values in BGR order.")

    return list(class_names), input_size, mean, std


def infer_true_label(image_path: Path, class_names: list[str]) -> int | None:
    name = image_path.stem.lower()
    for index, class_name in enumerate(class_names):
        class_key = class_name.lower()
        if name == class_key or name.startswith(f"{class_key}_") or name.startswith(f"{class_key}-"):
            return index
    return None


def get_pixel_bgr_type(ncnn: Any) -> Any:
    if hasattr(ncnn.Mat, "PixelType"):
        return ncnn.Mat.PixelType.PIXEL_BGR
    return ncnn.Mat.PIXEL_BGR


def mat_to_numpy(mat: Any) -> np.ndarray:
    if hasattr(mat, "numpy"):
        return np.asarray(mat.numpy(), dtype=np.float32)
    return np.asarray(mat, dtype=np.float32)


def extract_output(extractor: Any, output_name: str) -> Any:
    result = extractor.extract(output_name)
    if isinstance(result, tuple):
        if len(result) != 2:
            raise RuntimeError(f"Unexpected ncnn extract result: {result}")
        status, output = result
        if status != 0:
            raise RuntimeError(f"ncnn extract failed for output blob: {output_name}")
        return output
    return result


def load_ncnn_net(ncnn: Any, param_path: str, bin_path: str) -> Any:
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False

    if net.load_param(param_path) != 0:
        raise RuntimeError(f"Failed to load ncnn param: {param_path}")
    if net.load_model(bin_path) != 0:
        raise RuntimeError(f"Failed to load ncnn bin: {bin_path}")
    return net


def predict_one_image(
    ncnn: Any,
    net: Any,
    image_path: Path,
    input_name: str,
    output_name: str,
    input_size: int,
    mean: list[float],
    std: list[float],
    topk: int,
) -> tuple[list[float], list[int], list[float]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    mean_vals = [value * 255.0 for value in mean]
    norm_vals = [1.0 / (value * 255.0) for value in std]

    pixel_type = get_pixel_bgr_type(ncnn)
    mat_in = ncnn.Mat.from_pixels_resize(
        image,
        pixel_type,
        image.shape[1],
        image.shape[0],
        input_size,
        input_size,
    )
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    extractor = net.create_extractor()
    input_status = extractor.input(input_name, mat_in)
    if input_status != 0:
        raise RuntimeError(f"ncnn input failed for input blob: {input_name}")

    mat_out = extract_output(extractor, output_name)
    logits = mat_to_numpy(mat_out).reshape(-1).astype(np.float32)
    if logits.size == 0:
        raise RuntimeError(f"ncnn output is empty for image: {image_path}")

    logits_for_softmax = logits - np.max(logits)
    probs = np.exp(logits_for_softmax)
    probs = probs / np.sum(probs)

    k = min(topk, probs.size)
    indices = np.argsort(-probs)[:k]
    scores = probs[indices]
    return scores.tolist(), indices.astype(int).tolist(), logits.tolist()


def print_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> None:
    true_pred_label = "true\\pred"
    first_col_width = max(len(true_pred_label), *(len(name) for name in class_names))
    col_width = max(9, *(len(name) for name in class_names))

    header = f"{true_pred_label:<{first_col_width}}"
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
    ncnn = import_ncnn()
    image_paths = collect_bmp_images(args.image, args.image_dir)
    class_names, input_size, mean, std = resolve_runtime_metadata(args)
    net = load_ncnn_net(ncnn, args.param, args.bin)

    predicted_counts = {class_name: 0 for class_name in class_names}
    true_counts = {class_name: 0 for class_name in class_names}
    correct_counts = {class_name: 0 for class_name in class_names}
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    evaluated_count = 0
    correct_count = 0
    unknown_label_count = 0

    print("Backend: ncnn")
    print(f"Total BMP images: {len(image_paths)}")
    print(f"Classes: {class_names}")
    print(f"Input blob: {args.input_name}")
    print(f"Output blob: {args.output_name}")
    print(f"Input size: {input_size}")

    for image_path in image_paths:
        scores, indices, logits = predict_one_image(
            ncnn=ncnn,
            net=net,
            image_path=image_path,
            input_name=args.input_name,
            output_name=args.output_name,
            input_size=input_size,
            mean=mean,
            std=std,
            topk=args.topk,
        )
        pred_index = indices[0]
        pred_label = class_names[pred_index] if pred_index < len(class_names) else f"class_{pred_index}"
        true_index = infer_true_label(image_path, class_names)
        if pred_label in predicted_counts:
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
            if args.print_logits:
                print("Logits: " + " ".join(f"{value:.6f}" for value in logits))
            for rank, (score, index) in enumerate(zip(scores, indices), start=1):
                label = class_names[index] if index < len(class_names) else f"class_{index}"
                print(f"Top{rank}: {label} ({score:.4f})")
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
