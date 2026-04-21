# Lightweight 3-Class Image Classification Baseline

This project is a first-version baseline for a lightweight 3-class image classifier designed for:

- Input: `BGR`, 3 channels, `64x64`
- Framework: `PyTorch`
- Export: `ONNX`
- Deployment: `ncnn` with `C++`
- Target board: `Loongson 2K0300 (loongarch64 Linux)`

The baseline model uses depthwise separable convolutions to keep parameter count and compute low enough for later `ncnn INT8` optimization.

## Supported dataset layouts

```text
data/
  weapon/
  supplies/
  vehicle/
```

Recommended usage is to place all images under a single root, grouped only by class. The training script will automatically split the dataset into `train / val / test` with stratified sampling.

It also still supports a pre-split layout:

```text
data/
  train/
    weapon/
    supplies/
    vehicle/
  val/
    weapon/
    supplies/
    vehicle/
  test/
    weapon/
    supplies/
    vehicle/
```

The class directory names become the final labels automatically.

## Environment

Required Python packages:

- `torch`
- `opencv-python`
- `numpy`
- `PyYAML` (optional, only if using `--config`)

## Train

```bash
python train.py --data-root data --output-dir outputs/tiny_cls
```

Optional:

```bash
python train.py --config configs/tiny_cls.yaml
```

To customize automatic split ratio:

```bash
python train.py --data-root data --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

## Predict on a single image

```bash
python predict.py --checkpoint outputs/tiny_cls/best.pt --image demo.bmp
```

## Export ONNX

```bash
python export_onnx.py --checkpoint outputs/tiny_cls/best.pt --output onnx/tiny_cls.onnx
```

Export ONNX and ncnn files in one step:

```bash
python export_onnx.py ^
  --checkpoint outputs/tiny_cls/best.pt ^
  --output onnx/tiny_cls.onnx ^
  --ncnn-param deploy_ncnn/model.param ^
  --ncnn-bin deploy_ncnn/model.bin
```

If you also want optimized ncnn outputs:

```bash
python export_onnx.py ^
  --checkpoint outputs/tiny_cls/best.pt ^
  --output onnx/tiny_cls.onnx ^
  --ncnn-param deploy_ncnn/model.param ^
  --ncnn-bin deploy_ncnn/model.bin ^
  --optimize
```

## ncnn conversion

See [tools/convert_to_ncnn.md](/D:/21cyber/tools/convert_to_ncnn.md).

## Board-side deployment

Reference inference example:

- [deploy_ncnn/infer.cpp](/D:/21cyber/deploy_ncnn/infer.cpp)
- [deploy_ncnn/CMakeLists.txt](/D:/21cyber/deploy_ncnn/CMakeLists.txt)

## Notes

- Training and inference both use `BGR` preprocessing to stay aligned with OpenCV and `ncnn`.
- The default normalization is equivalent to mapping pixel values from `[0, 255]` to `[-1, 1]`.
- If you later change `mean/std`, make the same change in both `predict.py` and `deploy_ncnn/infer.cpp`.
- When using the single-root dataset layout, the generated split manifest is saved to `output_dir/split_manifest.json`.
- The PC-side validation script currently accepts only `.bmp` input images.
