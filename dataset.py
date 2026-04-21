from __future__ import annotations

import random
from pathlib import Path
import math
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_classes(root: str | Path) -> list[str]:
    root = Path(root)
    classes = [item.name for item in root.iterdir() if item.is_dir()]
    if not classes:
        raise ValueError(f"No class directories found under: {root}")
    return sorted(classes)


def collect_samples(root: str | Path, class_names: list[str]) -> list[tuple[Path, int]]:
    root = Path(root)
    samples: list[tuple[Path, int]] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMG_EXTENSIONS:
                samples.append((image_path, class_index))
    if not samples:
        raise ValueError(f"No images found under: {root}")
    return samples


def split_samples_stratified(
    samples: list[tuple[Path, int]],
    num_classes: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {ratio_sum:.6f}"
        )

    rng = random.Random(seed)
    class_buckets: list[list[tuple[Path, int]]] = [[] for _ in range(num_classes)]
    for sample in samples:
        class_buckets[sample[1]].append(sample)

    train_split: list[tuple[Path, int]] = []
    val_split: list[tuple[Path, int]] = []
    test_split: list[tuple[Path, int]] = []

    for class_samples in class_buckets:
        if not class_samples:
            continue

        rng.shuffle(class_samples)
        total = len(class_samples)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count

        if total >= 3:
            if train_count == 0:
                train_count = 1
            if val_count == 0:
                val_count = 1
            test_count = total - train_count - val_count
            if test_count <= 0:
                test_count = 1
                if train_count >= val_count and train_count > 1:
                    train_count -= 1
                elif val_count > 1:
                    val_count -= 1

        train_split.extend(class_samples[:train_count])
        val_split.extend(class_samples[train_count : train_count + val_count])
        test_split.extend(class_samples[train_count + val_count : train_count + val_count + test_count])

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    rng.shuffle(test_split)
    return train_split, val_split, test_split


def random_augment_bgr(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    if random.random() < 0.7:
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.92, 1.08)
        center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    if random.random() < 0.7:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-15.0, 15.0)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def preprocess_bgr(
    image: np.ndarray,
    input_size: int,
    mean: list[float],
    std: list[float],
    augment: bool = False,
) -> torch.Tensor:
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    if augment:
        image = random_augment_bgr(image)

    image = image.astype(np.float32) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean_array) / std_array
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image.copy())


class BGRImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        input_size: int,
        mean: list[float],
        std: list[float],
        augment: bool = False,
        class_names: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root}")

        self.class_names = class_names or discover_classes(self.root)
        self.samples = collect_samples(self.root, self.class_names)
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        tensor = preprocess_bgr(
            image=image,
            input_size=self.input_size,
            mean=self.mean,
            std=self.std,
            augment=self.augment,
        )
        return tensor, label


class BGRClassificationDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[tuple[Path, int]],
        class_names: list[str],
        input_size: int,
        mean: list[float],
        std: list[float],
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("Dataset split is empty.")
        self.class_names = class_names
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        tensor = preprocess_bgr(
            image=image,
            input_size=self.input_size,
            mean=self.mean,
            std=self.std,
            augment=self.augment,
        )
        return tensor, label
