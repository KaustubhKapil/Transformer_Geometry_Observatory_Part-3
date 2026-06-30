from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .dataset import build_datasets, build_loaders, split_indices
from .hooks import ActivationManager, register_vit_hooks
from .model import ViTProbe
from .trainer import Trainer, LAYER_NAMES
from .utils import ensure_dir, save_json, set_seed, setup_logging


def build_model(cfg: Config):
    model = ViTProbe(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    model.hooks_mgr = register_vit_hooks(model, ActivationManager(LAYER_NAMES))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_dir = ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)
    logger = setup_logging(output_dir)
    save_json(cfg.to_dict(), output_dir / "config.json")

    train_ds, val_ds = build_datasets(cfg.data.train_dir, cfg.data.val_dir, cfg.data.image_size, cfg.data.class_subset_file)

    analysis_path = output_dir / "analysis_indices.npy"
    traj_path = output_dir / "trajectory_indices.npy"
    if analysis_path.exists():
        analysis_indices = np.load(analysis_path)
    else:
        analysis_indices = split_indices(len(val_ds), cfg.data.analysis_set_size, cfg.data.analysis_seed)
        np.save(analysis_path, analysis_indices)
    if traj_path.exists():
        trajectory_indices = np.load(traj_path)
    else:
        trajectory_indices = split_indices(len(val_ds), cfg.data.trajectory_set_size, cfg.data.trajectory_seed)
        np.save(traj_path, trajectory_indices)

    train_loader, val_loader, analysis_loader, _ = build_loaders(
        train_ds,
        val_ds,
        cfg.train.batch_size,
        cfg.data.num_workers,
        cfg.data.pin_memory,
        analysis_indices,
        trajectory_indices,
        cfg.seed,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model.to(device)

    trainer = Trainer(cfg, model, train_loader, val_loader, analysis_loader, device)
    if args.resume:
        ckpt = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume} at epoch {ckpt.get('epoch')}")

    trainer.train()


if __name__ == "__main__":
    main()
