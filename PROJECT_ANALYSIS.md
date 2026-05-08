# 工程分析与模型结构说明

## 1. 工程目标
该项目是一个**轻量级 3 类图像分类器**（`TinyClassifier`），基于 PyTorch 训练，支持：
- 训练与验证（`train.py`）
- PyTorch 推理（`predict.py`）
- 导出 ONNX 并转换 NCNN（`export_onnx.py`）
- NCNN 推理（`predict_ncnn.py`）

## 2. 目录与文件职责
- `train.py`：训练入口，加载配置、构建数据集、训练循环、保存 best/last checkpoint。
- `dataset.py`：数据集读取、分层切分、BGR 预处理、数据增强。
- `models/tiny_cls.py`：模型结构定义（深度可分离卷积 + 残差）。
- `predict.py`：PyTorch checkpoint 推理与统计（混淆矩阵、分类准确率）。
- `export_onnx.py`：checkpoint -> ONNX -> pnnx -> ncnn param/bin。
- `predict_ncnn.py`：NCNN Python 绑定推理，逻辑与 `predict.py` 对齐。
- `utils.py`：默认配置、seed、accuracy、配置读写。
- `configs/tiny_cls.yaml`：训练配置样例。
- `tools/convert_to_ncnn.md`：转换部署说明。
- `labels.txt`：默认标签（`supplies / vehicle / weapon`）。

## 3. 训练流程（train.py）
### 3.1 配置合并
- 先读默认配置 + 外部 YAML/JSON，再用 CLI 参数覆盖。

### 3.2 数据组织模式
训练阶段支持两种数据组织：
1) **预切分模式**：存在 `data/train` 与 `data/val`（可选 `data/test`）。
2) **自动切分模式**：直接扫描类目录，按分层比例切 train/val/test，并把结果写入 `split_manifest.json`。

### 3.3 训练细节
- 优化器：`AdamW`
- 学习率调度：`CosineAnnealingLR`
- 损失函数：`CrossEntropyLoss`
- 混合精度：CUDA 时启用 `torch.cuda.amp`。
- 每轮记录 train/val loss 与 acc，持续保存 `last.pt`，验证精度提升时覆盖 `best.pt`。
- 若有 test 集，会在训练结束后用 best 权重评估 test。

## 4. 数据处理与增强（dataset.py）
### 4.1 类别发现与样本收集
- 类别来自子目录名（排序后作为标签顺序）。
- 支持扩展名：`.jpg/.jpeg/.png/.bmp/.webp`。

### 4.2 分层切分
- `train_ratio + val_ratio + test_ratio` 必须等于 1。
- 每个类别单独 shuffle 后按比例切分，尽量保证各 split 都有样本（类别样本>=3时做最小数量修正）。

### 4.3 预处理与增强
- 固定 resize 到 `input_size x input_size`（默认 64）。
- BGR 通道归一化（mean/std 以 0~1 记）。
- 训练增强：随机翻转、随机旋转缩放、亮度对比度扰动、轻微高斯模糊。

## 5. 模型结构分析（TinyClassifier）
`models/tiny_cls.py` 实现了一个面向边缘部署的小模型：

1. **Stem**：`3 -> 16`，3x3, stride=2。
2. **主干 blocks（DepthwiseSeparableBlock）**：
   - 每个 block = 深度卷积（3x3 DW） + 逐点卷积（1x1 PW）。
   - 当 `stride=1` 且通道数不变时，启用残差连接。
3. **通道与下采样路径**：
   - `16 -> 16` (s1)
   - `16 -> 24` (s2)
   - `24 -> 24` (s1)
   - `24 -> 32` (s2)
   - `32 -> 32` (s1)
   - `32 -> 48` (s2)
   - `48 -> 64` (s1)
4. **Head**：`AdaptiveAvgPool2d(1)` + `Dropout(0.1)` + `Linear(64 -> num_classes)`。

### 5.1 设计特点
- 使用深度可分离卷积显著降低参数与 FLOPs，适合 NCNN 端侧部署。
- 残差连接改善可训练性。
- 全局池化 + 小全连接头，使输入尺寸固定后推理开销低。

## 6. 推理与评估工具
### 6.1 PyTorch 推理（predict.py）
- 输入仅支持 BMP。
- 输出 Top-k 概率。
- 根据文件名前缀推断真值（如 `weapon_001.bmp`），统计：
  - 总准确率
  - 每类准确率
  - 混淆矩阵

### 6.2 NCNN 推理（predict_ncnn.py）
- 读取 `.param/.bin` 并执行前向。
- 预处理与训练保持一致（BGR + resize + mean/std）。
- 输出与 `predict.py` 同类统计，支持打印 raw logits。

## 7. 导出与部署链路
`export_onnx.py` 的完整链路：
1. 载入 `best.pt/last.pt`
2. 导出 ONNX（默认 opset=12）
3. 调用 `pnnx` 产出 NCNN `model.param + model.bin`
4. 可选保留 ONNX 与 pnnx 中间文件

`tools/convert_to_ncnn.md` 提示：
- 默认会清理中间文件。
- 推荐先验证 FP32，再考虑 INT8 量化以满足更低时延目标。

## 8. 默认训练/数据超参数
默认参数（`utils.py` + `configs/tiny_cls.yaml`）基本一致：
- epoch=60
- batch_size=128
- lr=1e-3
- weight_decay=1e-4
- input_size=64
- split=0.85/0.10/0.05
- mean=[0.406,0.456,0.485], std=[0.225,0.224,0.229]

## 9. 总结
该工程是一个完整的**轻量化图像分类训练-部署闭环**：
- 训练端强调简洁与可复现；
- 模型端强调低开销与可部署；
- 推理端同时覆盖 PyTorch 与 NCNN；
- 工程端已具备标签、配置、转换文档与批量评估能力。
