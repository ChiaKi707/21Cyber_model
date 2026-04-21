# Export to ncnn with pnnx

This project now uses:

`PyTorch checkpoint -> ONNX (intermediate) -> pnnx -> ncnn`

The final target files are:

`model.param + model.bin`

## 1. Export ncnn directly

```bash
python export_onnx.py ^
  --checkpoint outputs/tiny_cls/best.pt ^
  --ncnn-param deploy_ncnn/model.param ^
  --ncnn-bin deploy_ncnn/model.bin ^
  --pnnx-path D:\tools\pnnx.exe
```

By default, only the final `ncnn` files are kept.
Intermediate `onnx` and auxiliary `pnnx` files are deleted automatically.

## 2. Keep the intermediate ONNX file if needed

```bash
python export_onnx.py ^
  --checkpoint outputs/tiny_cls/best.pt ^
  --output onnx/tiny_cls.onnx ^
  --ncnn-param deploy_ncnn/model.param ^
  --ncnn-bin deploy_ncnn/model.bin ^
  --pnnx-path D:\tools\pnnx.exe ^
  --keep-onnx
```

If you need the extra `pnnx` debug artifacts too, add:

```bash
--keep-pnnx-artifacts
```

## 3. pnnx options

The export script forwards a few useful options to `pnnx`:

- `--fp16 1` or `0`
- `--optlevel 0`, `1`, or `2`

Defaults are:

- `fp16=1`
- `optlevel=2`

## 4. Quantization suggestion

To reach the `< 8 ms` target on `Loongson 2K0300`, plan for `ncnn INT8` after the FP32 baseline is verified.

## 5. Preprocessing consistency

The Python training/export path uses:

- pixel format: `BGR`
- resize: `64x64`
- normalization:
  - mean: `[0.406, 0.456, 0.485]`
  - std: `[0.225, 0.224, 0.229]`

Equivalent `ncnn` preprocessing in C++:

- `mean_vals = {103.53f, 116.28f, 123.675f}`
- `norm_vals = {1/57.375f, 1/57.12f, 1/58.395f}`

If you change training normalization later, keep the C++ side synchronized.

## 6. Output tensor

The ONNX and `ncnn` models output raw logits of length `3`.
Apply softmax in post-processing to obtain class probabilities.
