from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import (
    BGRClassificationDataset,
    BGRImageFolderDataset,
    collect_samples,
    discover_classes,
    split_samples_stratified,
)
from models import TinyClassifier
from utils import (
    DEFAULT_CONFIG,
    accuracy_from_logits,
    ensure_dir,
    load_config,
    save_labels,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight 3-class image classifier.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config path.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root containing train/val/test.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--mean", type=float, nargs=3, default=None, help="BGR mean values in 0-1 range.")
    parser.add_argument("--std", type=float, nargs=3, default=None, help="BGR std values in 0-1 range.")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    for key in DEFAULT_CONFIG.keys():
        arg_key = key.replace("-", "_")
        value = getattr(args, arg_key)
        if value is not None:
            config[key] = value
    return config


def save_split_manifest(
    output_dir: Path,
    class_names: list[str],
    train_samples: list[tuple[Path, int]],
    val_samples: list[tuple[Path, int]],
    test_samples: list[tuple[Path, int]],
) -> None:
    manifest = {
        "class_names": class_names,
        "splits": {
            "train": [{"path": str(path), "label": class_names[label]} for path, label in train_samples],
            "val": [{"path": str(path), "label": class_names[label]} for path, label in val_samples],
            "test": [{"path": str(path), "label": class_names[label]} for path, label in test_samples],
        },
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader | None, list[str], str]:
    data_root = Path(config["data_root"])
    train_root = data_root / "train"
    val_root = data_root / "val"
    test_root = data_root / "test"

    dataset_mode = "auto_split"
    if train_root.exists() and val_root.exists():
        dataset_mode = "pre_split"
        train_dataset = BGRImageFolderDataset(
            root=train_root,
            input_size=config["input_size"],
            mean=config["mean"],
            std=config["std"],
            augment=True,
        )
        val_dataset = BGRImageFolderDataset(
            root=val_root,
            input_size=config["input_size"],
            mean=config["mean"],
            std=config["std"],
            augment=False,
            class_names=train_dataset.class_names,
        )
        test_dataset = None
        if test_root.exists():
            test_dataset = BGRImageFolderDataset(
                root=test_root,
                input_size=config["input_size"],
                mean=config["mean"],
                std=config["std"],
                augment=False,
                class_names=train_dataset.class_names,
            )
        class_names = train_dataset.class_names
    else:
        class_names = discover_classes(data_root)
        all_samples = collect_samples(data_root, class_names)
        train_samples, val_samples, test_samples = split_samples_stratified(
            samples=all_samples,
            num_classes=len(class_names),
            train_ratio=float(config["train_ratio"]),
            val_ratio=float(config["val_ratio"]),
            test_ratio=float(config["test_ratio"]),
            seed=int(config["seed"]),
        )
        output_dir = ensure_dir(config["output_dir"])
        save_split_manifest(output_dir, class_names, train_samples, val_samples, test_samples)

        train_dataset = BGRClassificationDataset(
            samples=train_samples,
            class_names=class_names,
            input_size=config["input_size"],
            mean=config["mean"],
            std=config["std"],
            augment=True,
        )
        val_dataset = BGRClassificationDataset(
            samples=val_samples,
            class_names=class_names,
            input_size=config["input_size"],
            mean=config["mean"],
            std=config["std"],
            augment=False,
        )
        test_dataset = BGRClassificationDataset(
            samples=test_samples,
            class_names=class_names,
            input_size=config["input_size"],
            mean=config["mean"],
            std=config["std"],
            augment=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader, class_names, dataset_mode


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    autocast_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled and is_train)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits, targets) * batch_size
        total_count += batch_size

    return total_loss / total_count, total_acc / total_count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    return run_one_epoch(model=model, loader=loader, criterion=criterion, device=device, optimizer=None)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    config: dict,
    epoch: int,
    best_val_acc: float,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_size": config["input_size"],
        "mean": config["mean"],
        "std": config["std"],
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_name": "TinyClassifier",
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    set_seed(int(config["seed"]))

    output_dir = ensure_dir(config["output_dir"])
    labels_path = output_dir / "labels.txt"

    train_loader, val_loader, test_loader, class_names, dataset_mode = build_dataloaders(config)
    save_labels(class_names, labels_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyClassifier(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_acc = 0.0
    best_ckpt_path = output_dir / "best.pt"
    last_ckpt_path = output_dir / "last.pt"

    print(f"Device: {device}")
    print(f"Dataset mode: {dataset_mode}")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"Test samples: {len(test_loader.dataset)}")
    if dataset_mode == "auto_split":
        print(f"Split manifest saved to: {output_dir / 'split_manifest.json'}")

    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch:03d}/{config['epochs']:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={elapsed:.1f}s"
        )

        save_checkpoint(last_ckpt_path, model, class_names, config, epoch, best_val_acc)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_ckpt_path, model, class_names, config, epoch, best_val_acc)
            print(f"Saved best checkpoint to: {best_ckpt_path}")

    if test_loader is not None and best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test result: loss={test_loss:.4f} acc={test_acc:.4f}")

    print(f"Training finished. Best val acc: {best_val_acc:.4f}")
    print(f"Labels file saved to: {labels_path}")


if __name__ == "__main__":
    main()
