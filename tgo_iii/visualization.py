from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _save_fig(fig, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(mat: np.ndarray, path: str | Path, title: str = "", cmap: str = "viridis", vmin=None, vmax=None, xticklabels=None, yticklabels=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=7)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, path)


def plot_curve(values: Iterable[float], path: str | Path, title: str = "", xlabel: str = "Epoch", ylabel: str = "Value"):
    vals = list(values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(vals) + 1), vals, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def plot_multi_curve(series: Dict[str, Iterable[float]], path: str | Path, title: str = "", xlabel: str = "Epoch", ylabel: str = "Value"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, vals in series.items():
        vals = list(vals)
        ax.plot(range(1, len(vals) + 1), vals, linewidth=1.8, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _save_fig(fig, path)


def plot_bar(labels: List[str], values: Iterable[float], path: str | Path, title: str = "", ylabel: str = ""):
    vals = list(values)
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(labels)), 4.2))
    ax.bar(range(len(labels)), vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, path)


def plot_two_panel_heatmaps(a: np.ndarray, b: np.ndarray, path: str | Path, labels: Optional[List[str]] = None, title_a: str = "A", title_b: str = "B"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(a, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title(title_a)
    im1 = axes[1].imshow(b, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title(title_b)
    if labels is not None:
        for ax in axes:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    _save_fig(fig, path)


def plot_summary_grid(
    probe_acc: float,
    fisher_ratio: float,
    centroid_mean_dist: float,
    local_id_mean: float,
    local_pca_mean: float,
    path: str | Path,
    title: str = "",
):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    vals = [
        ("Linear Probe", probe_acc),
        ("Fisher Ratio", fisher_ratio),
        ("Centroid Mean Dist", centroid_mean_dist),
        ("Local ID Mean", local_id_mean),
        ("Local PCA Rank Mean", local_pca_mean),
    ]
    flat_axes = axes.flatten()
    for ax, (name, val) in zip(flat_axes, vals):
        ax.bar([0], [val])
        ax.set_xticks([0])
        ax.set_xticklabels([name], rotation=15, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    flat_axes[-1].axis("off")
    fig.suptitle(title)
    _save_fig(fig, path)
