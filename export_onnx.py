from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from models import TinyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained PyTorch model to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--output", type=str, required=True, help="ONNX output path.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument(
        "--ncnn-param",
        type=str,
        default=None,
        help="Optional ncnn param output path. If set, the script will also export .param and .bin.",
    )
    parser.add_argument(
        "--ncnn-bin",
        type=str,
        default=None,
        help="Optional ncnn bin output path. If set, the script will also export .param and .bin.",
    )
    parser.add_argument(
        "--onnx2ncnn-path",
        type=str,
        default="onnx2ncnn",
        help="Path to the onnx2ncnn executable.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run ncnnoptimize after onnx2ncnn and write optimized param/bin outputs.",
    )
    parser.add_argument(
        "--ncnnoptimize-path",
        type=str,
        default="ncnnoptimize",
        help="Path to the ncnnoptimize executable.",
    )
    return parser.parse_args()


def resolve_ncnn_outputs(
    onnx_output: Path,
    ncnn_param: str | None,
    ncnn_bin: str | None,
) -> tuple[Path | None, Path | None]:
    if ncnn_param is None and ncnn_bin is None:
        return None, None

    if (ncnn_param is None) != (ncnn_bin is None):
        raise ValueError("--ncnn-param and --ncnn-bin must be provided together.")

    return Path(ncnn_param), Path(ncnn_bin)


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Command not found: {command[0]}. Please provide the executable path explicitly."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def export_ncnn(
    onnx_path: Path,
    param_path: Path,
    bin_path: Path,
    onnx2ncnn_path: str,
    optimize: bool,
    ncnnoptimize_path: str,
) -> tuple[Path, Path]:
    param_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    onnx2ncnn_exe = shutil.which(onnx2ncnn_path) or onnx2ncnn_path
    run_command([onnx2ncnn_exe, str(onnx_path), str(param_path), str(bin_path)])

    if not optimize:
        return param_path, bin_path

    optimized_param = param_path.with_name(f"{param_path.stem}-opt{param_path.suffix}")
    optimized_bin = bin_path.with_name(f"{bin_path.stem}-opt{bin_path.suffix}")
    ncnnoptimize_exe = shutil.which(ncnnoptimize_path) or ncnnoptimize_path
    run_command(
        [
            ncnnoptimize_exe,
            str(param_path),
            str(bin_path),
            str(optimized_param),
            str(optimized_bin),
            "0",
        ]
    )
    return optimized_param, optimized_bin


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = TinyClassifier(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_size = int(checkpoint["input_size"])
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ncnn_param_path, ncnn_bin_path = resolve_ncnn_outputs(output_path, args.ncnn_param, args.ncnn_bin)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    print(f"Exported ONNX model to: {output_path.resolve()}")
    print("Input shape: [1, 3, 64, 64]")
    print("Output shape: [1, 3]")

    if ncnn_param_path is not None and ncnn_bin_path is not None:
        final_param, final_bin = export_ncnn(
            onnx_path=output_path,
            param_path=ncnn_param_path,
            bin_path=ncnn_bin_path,
            onnx2ncnn_path=args.onnx2ncnn_path,
            optimize=args.optimize,
            ncnnoptimize_path=args.ncnnoptimize_path,
        )
        print(f"Exported ncnn param to: {final_param.resolve()}")
        print(f"Exported ncnn bin to: {final_bin.resolve()}")


if __name__ == "__main__":
    main()
