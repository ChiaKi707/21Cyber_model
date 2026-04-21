from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_CONFIG = {
    "data_root": "data",
    "output_dir": "outputs/tiny_cls",
    "epochs": 60,
    "batch_size": 128,
    "num_workers": 4,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "input_size": 64,
    "seed": 42,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def load_labels(labels_path: str | Path) -> list[str]:
    lines = Path(labels_path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def save_labels(labels: list[str], labels_path: str | Path) -> None:
    content = "\n".join(labels) + "\n"
    Path(labels_path).write_text(content, encoding="utf-8")


def load_config(config_path: str | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        user_config = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML config files.") from exc
        user_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raise ValueError(f"Unsupported config format: {suffix}")

    config.update(user_config)
    return config


def resolve_runtime_args(args_dict: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for key, value in args_dict.items():
        if value is not None:
            merged[key] = value
    return merged
