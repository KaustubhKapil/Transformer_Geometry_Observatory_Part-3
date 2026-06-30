from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import seed_worker


class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, index


def load_class_subset_file(path: Optional[str]) -> Optional[list[str]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes or None


def build_transforms(image_size: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def build_datasets(train_dir: str, val_dir: str, image_size: int, class_subset_file: Optional[str] = None):
    train_tf, eval_tf = build_transforms(image_size)
    allowed = load_class_subset_file(class_subset_file)
    train_ds = IndexedImageFolder(train_dir, transform=train_tf)
    val_ds = IndexedImageFolder(val_dir, transform=eval_tf)

    if allowed:
        allowed_set = set(allowed)
        allowed_train = [c for c in train_ds.classes if c in allowed_set]
        allowed_val = [c for c in val_ds.classes if c in allowed_set]
        class_to_idx_train = {c: i for i, c in enumerate(allowed_train)}
        class_to_idx_val = {c: i for i, c in enumerate(allowed_val)}

        train_samples = [(p, class_to_idx_train[train_ds.classes[y]]) for p, y in train_ds.samples if train_ds.classes[y] in allowed_set]
        val_samples = [(p, class_to_idx_val[val_ds.classes[y]]) for p, y in val_ds.samples if val_ds.classes[y] in allowed_set]

        train_ds.samples = train_samples
        train_ds.targets = [y for _, y in train_samples]
        train_ds.classes = allowed_train
        train_ds.class_to_idx = class_to_idx_train

        val_ds.samples = val_samples
        val_ds.targets = [y for _, y in val_samples]
        val_ds.classes = allowed_val
        val_ds.class_to_idx = class_to_idx_val

    return train_ds, val_ds


def split_indices(num_items: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_items)
    return np.asarray(perm[:size], dtype=np.int64)


def build_loaders(
    train_ds,
    val_ds,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    analysis_indices: np.ndarray,
    trajectory_indices: np.ndarray,
    seed: int,
):
    import torch
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    analysis_loader = DataLoader(
        Subset(val_ds, analysis_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    trajectory_loader = DataLoader(
        Subset(val_ds, trajectory_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, analysis_loader, trajectory_loader
