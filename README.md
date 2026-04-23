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

Batch predict all BMP images in a folder:

```bash
python predict.py --checkpoint outputs/tiny_cls/best.pt --image-dir demo_images
```

Batch predict with exported ncnn files on PC:

```bash
python predict_ncnn.py ^
  --param deploy_ncnn/model.param ^
  --bin deploy_ncnn/model.bin ^
  --checkpoint outputs/tiny_cls/best.pt ^
  --image-dir demo_images ^
  --quiet
```

## Export ncnn

```bash
python export_onnx.py ^
  --checkpoint outputs/tiny_cls/best.pt ^
  --ncnn-param deploy_ncnn/model.param ^
  --ncnn-bin deploy_ncnn/model.bin ^
  --pnnx-path D:\tools\pnnx.exe
```

By default, the script keeps only the final `model.param` and `model.bin`.
The intermediate `onnx` file and auxiliary `pnnx` files are cleaned up automatically.

If you also want to keep the intermediate ONNX file:

```bash
python export_onnx.py ^
  --checkpoint outputs/tiny_cls/best.pt ^
  --output onnx/tiny_cls.onnx ^
  --ncnn-param deploy_ncnn/model.param ^
  --ncnn-bin deploy_ncnn/model.bin ^
  --pnnx-path D:\tools\pnnx.exe ^
  --keep-onnx
```

If you also want to inspect the auxiliary `pnnx` outputs:

```bash
python export_onnx.py ^
  --checkpoint outputs/tiny_cls/best.pt ^
  --output onnx/tiny_cls.onnx ^
  --ncnn-param deploy_ncnn/model.param ^
  --ncnn-bin deploy_ncnn/model.bin ^
  --pnnx-path D:\tools\pnnx.exe ^
  --keep-onnx ^
  --keep-pnnx-artifacts
```

## ncnn conversion

See [tools/convert_to_ncnn.md](/D:/21cyber/tools/convert_to_ncnn.md).

## Board-side deployment

Reference inference example:

- [deploy_ncnn/infer.cpp](/D:/21cyber/deploy_ncnn/infer.cpp)
- [deploy_ncnn/CMakeLists.txt](/D:/21cyber/deploy_ncnn/CMakeLists.txt)

## Notes

- Training and inference both use `BGR` preprocessing to stay aligned with OpenCV and `ncnn`.
- The default normalization uses `BGR mean=[0.406, 0.456, 0.485]` and `std=[0.225, 0.224, 0.229]`.
- If you later change `mean/std`, make the same change in both `predict.py` and `deploy_ncnn/infer.cpp`.
- When using the single-root dataset layout, the generated split manifest is saved to `output_dir/split_manifest.json`.
- The PC-side validation script currently accepts only `.bmp` input images, whether using single-image or folder batch prediction.
