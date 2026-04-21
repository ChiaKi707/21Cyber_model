# ONNX to ncnn

This baseline uses only deployment-friendly operators, so the default export path is:

`PyTorch -> ONNX -> ncnn`

## 1. Export ONNX

```bash
python export_onnx.py --checkpoint outputs/tiny_cls/best.pt --output onnx/tiny_cls.onnx
```

You can also export `ONNX + ncnn` in one command:

```bash
python export_onnx.py ^
  --checkpoint outputs/tiny_cls/best.pt ^
  --output onnx/tiny_cls.onnx ^
  --ncnn-param deploy_ncnn/model.param ^
  --ncnn-bin deploy_ncnn/model.bin
```

## 2. Convert to ncnn

If `onnx2ncnn` is already available:

```bash
onnx2ncnn onnx/tiny_cls.onnx deploy_ncnn/tiny_cls.param deploy_ncnn/tiny_cls.bin
```

Optional model optimization:

```bash
ncnnoptimize deploy_ncnn/tiny_cls.param deploy_ncnn/tiny_cls.bin deploy_ncnn/tiny_cls-opt.param deploy_ncnn/tiny_cls-opt.bin 0
```

## 3. Quantization suggestion

To reach the `< 8 ms` target on `Loongson 2K0300`, plan for `ncnn INT8` after the FP32 baseline is verified.

Typical flow:

1. Prepare a calibration image list from the training set.
2. Generate quantization table with the official `ncnn2table` tool.
3. Convert the FP32 model to INT8 using `ncnn2int8`.
4. Benchmark again on the target board.

Exact command names can differ slightly depending on your local `ncnn` build version, so check the tool binaries built with your current `ncnn`.

## 4. Preprocessing consistency

The Python training/export path uses:

- pixel format: `BGR`
- resize: `64x64`
- normalization:
  - mean: `[0.5, 0.5, 0.5]`
  - std: `[0.5, 0.5, 0.5]`

Equivalent `ncnn` preprocessing in C++:

- `mean_vals = {127.5f, 127.5f, 127.5f}`
- `norm_vals = {1/127.5f, 1/127.5f, 1/127.5f}`

If you change training normalization later, keep the C++ side synchronized.

## 5. Output tensor

The ONNX and `ncnn` models output raw logits of length `3`.
Apply softmax in post-processing to obtain class probabilities.
