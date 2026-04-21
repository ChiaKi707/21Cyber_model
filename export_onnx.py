from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from models import TinyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained PyTorch model to ncnn via ONNX + pnnx."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional ONNX output path. If omitted, a temporary ONNX file is used.",
    )
    parser.add_argument("--ncnn-param", type=str, required=True, help="ncnn param output path.")
    parser.add_argument("--ncnn-bin", type=str, required=True, help="ncnn bin output path.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument(
        "--pnnx-path",
        type=str,
        default="pnnx",
        help="Path to the pnnx executable, such as pnnx.exe on Windows.",
    )
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep the exported ONNX file. If --output is omitted, a temp file is still removed automatically.",
    )
    parser.add_argument(
        "--keep-pnnx-artifacts",
        action="store_true",
        help="Keep auxiliary pnnx files such as .pnnx.param, .pnnx.bin and generated python stubs.",
    )
    parser.add_argument(
        "--fp16",
        type=int,
        default=1,
        choices=[0, 1],
        help="Pass fp16 option to pnnx. 1 saves ncnn weights in fp16 when supported.",
    )
    parser.add_argument(
        "--optlevel",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Pass graph optimization level to pnnx.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Command not found: {command[0]}. Please provide the executable path explicitly."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def export_ncnn_with_pnnx(
    onnx_path: Path,
    param_path: Path,
    bin_path: Path,
    pnnx_path: str,
    fp16: int,
    optlevel: int,
) -> tuple[Path, Path, list[Path]]:
    param_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    pnnx_exe = shutil.which(pnnx_path) or pnnx_path
    pnnx_artifact_base = param_path.with_suffix("")
    pnnx_param_path = pnnx_artifact_base.with_suffix(".pnnx.param")
    pnnx_bin_path = pnnx_artifact_base.with_suffix(".pnnx.bin")
    pnnx_py_path = pnnx_artifact_base.with_name(f"{pnnx_artifact_base.name}_pnnx.py")
    pnnx_onnx_path = pnnx_artifact_base.with_suffix(".pnnx.onnx")
    ncnn_py_path = pnnx_artifact_base.with_name(f"{pnnx_artifact_base.name}_ncnn.py")

    command = [
        pnnx_exe,
        str(onnx_path),
        f"ncnnparam={param_path}",
        f"ncnnbin={bin_path}",
        f"pnnxparam={pnnx_param_path}",
        f"pnnxbin={pnnx_bin_path}",
        f"pnnxpy={pnnx_py_path}",
        f"pnnxonnx={pnnx_onnx_path}",
        f"ncnnpy={ncnn_py_path}",
        f"fp16={fp16}",
        f"optlevel={optlevel}",
    ]
    run_command(command)
    aux_files = [
        pnnx_param_path,
        pnnx_bin_path,
        pnnx_py_path,
        pnnx_onnx_path,
        ncnn_py_path,
    ]
    return param_path, bin_path, aux_files


def export_onnx_model(
    checkpoint_path: str,
    output_path: Path,
    opset: int,
) -> tuple[list[str], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = TinyClassifier(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_size = int(checkpoint["input_size"])
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )
    return checkpoint["class_names"], input_size


def main() -> None:
    args = parse_args()
    ncnn_param_path = Path(args.ncnn_param)
    ncnn_bin_path = Path(args.ncnn_bin)

    if args.output is not None:
        onnx_path = Path(args.output)
        class_names, input_size = export_onnx_model(args.checkpoint, onnx_path, args.opset)
        print(f"Exported ONNX model to: {onnx_path.resolve()}")
    else:
        with tempfile.TemporaryDirectory(prefix="tiny_cls_onnx_") as temp_dir:
            onnx_path = Path(temp_dir) / "model.onnx"
            class_names, input_size = export_onnx_model(args.checkpoint, onnx_path, args.opset)
            final_param, final_bin, aux_files = export_ncnn_with_pnnx(
                onnx_path=onnx_path,
                param_path=ncnn_param_path,
                bin_path=ncnn_bin_path,
                pnnx_path=args.pnnx_path,
                fp16=args.fp16,
                optlevel=args.optlevel,
            )
            if not args.keep_pnnx_artifacts:
                for aux_file in aux_files:
                    aux_file.unlink(missing_ok=True)
            print("Exported ONNX model to a temporary file for pnnx conversion.")
            print(f"Input shape: [1, 3, {input_size}, {input_size}]")
            print(f"Output shape: [1, {len(class_names)}]")
            print(f"Exported ncnn param to: {final_param.resolve()}")
            print(f"Exported ncnn bin to: {final_bin.resolve()}")
            return

    print(f"Input shape: [1, 3, {input_size}, {input_size}]")
    print(f"Output shape: [1, {len(class_names)}]")

    final_param, final_bin, aux_files = export_ncnn_with_pnnx(
        onnx_path=onnx_path,
        param_path=ncnn_param_path,
        bin_path=ncnn_bin_path,
        pnnx_path=args.pnnx_path,
        fp16=args.fp16,
        optlevel=args.optlevel,
    )
    print(f"Exported ncnn param to: {final_param.resolve()}")
    print(f"Exported ncnn bin to: {final_bin.resolve()}")

    if not args.keep_pnnx_artifacts:
        for aux_file in aux_files:
            aux_file.unlink(missing_ok=True)

    if not args.keep_onnx:
        onnx_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
