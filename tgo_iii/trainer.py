from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .metrics import (
    class_centroid_distances,
    classwise_local_intrinsic_dimension,
    classwise_local_pca_rank,
    fisher_ratio,
    layerwise_fisher_ratio,
    layerwise_probe_accuracy,
    linear_probe_accuracy,
    linear_cka,
    svcca,
    twonn_intrinsic_dimension,
    token_covariance_matrix,
    token_coupling_ratio,
    pairwise_matrix,
    adjacent_series,
)
from .utils import ensure_dir, is_main_process, save_json, setup_logging, timestamp, unwrap_model
from .visualization import (
    plot_bar,
    plot_curve,
    plot_heatmap,
    plot_multi_curve,
    plot_summary_grid,
)


LAYER_NAMES = [
    "Layer_00_PatchEmbed",
    "Layer_00_PosEmbed",
    *[f"Layer_{i:02d}_Block{i:02d}" for i in range(1, 13)],
    "Layer_13_CLS_Final",
]


def _token_tensor(layer_name: str, tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor.ndim != 3:
        return None
    if layer_name == "Layer_13_CLS_Final":
        return None
    return tensor


def _extract_image_vector(layer_name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f"Unexpected activation shape for {layer_name}: {tuple(tensor.shape)}")
    if layer_name == "Layer_00_PatchEmbed":
        return tensor.mean(dim=1)
    return tensor[:, 0, :]


def _nanmean_dict(d: Dict[str, float]) -> float:
    vals = [v for v in d.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, analysis_loader, device):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.analysis_loader = analysis_loader
        self.device = device

        self.output_dir = ensure_dir(cfg.output_dir)
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.results_dir = ensure_dir(self.output_dir / "results")
        self.summaries_dir = ensure_dir(self.output_dir / "summaries")
        self.global_dir = ensure_dir(self.output_dir / "global_analysis")
        self.logger = setup_logging(self.output_dir)

        self.best_acc = -1.0
        self.scaler = GradScaler(enabled=bool(cfg.train.amp))
        self.optim = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.layer_names = LAYER_NAMES

        ds = getattr(self.analysis_loader, "dataset", None)
        base_ds = getattr(ds, "dataset", ds)
        self.class_names = getattr(base_ds, "classes", None)

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.train.opt.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.cfg.train.lr,
                momentum=self.cfg.train.momentum,
                weight_decay=self.cfg.train.weight_decay,
            )
        return torch.optim.AdamW(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

    def _build_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=self.cfg.train.epochs, eta_min=self.cfg.train.min_lr
        )

    def save_checkpoint(self, epoch: int, train_loss: float, val_acc: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_loss": float(train_loss),
            "validation_accuracy": float(val_acc),
            "best_accuracy": float(self.best_acc),
            "random_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
            },
            "config": self.cfg.to_dict(),
        }
        if self.cfg.checkpoint.save_last:
            torch.save(state, self.checkpoint_dir / "last.pth")
        if is_best and self.cfg.checkpoint.save_best:
            torch.save(state, self.checkpoint_dir / "best.pth")

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location="cpu")
        unwrap_model(self.model).load_state_dict(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_acc = float(ckpt.get("best_accuracy", -1.0))
        return ckpt

    def train(self):
        self.model.to(self.device)
        for epoch in range(1, self.cfg.train.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_acc = self.validate(epoch)
            metrics = self.analyze_epoch(epoch)
            self.scheduler.step()

            is_best = val_acc > self.best_acc
            self.best_acc = max(self.best_acc, val_acc)
            self.save_checkpoint(epoch, train_loss, val_acc, is_best=is_best)

            self.logger.info(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | best={self.best_acc:.4f} | "
                f"probe={metrics['linear_probe_accuracy']:.4f} | fisher={metrics['fisher_ratio']:.4f} | "
                f"id={metrics['local_intrinsic_dimension_mean']:.2f} | pca_rank={metrics['local_pca_rank_mean']:.2f}"
            )

        self.final_global_analysis()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total = 0
        pbar = tqdm(self.train_loader, desc=f"Train {epoch:03d}", disable=not is_main_process())
        for images, targets, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optim.zero_grad(set_to_none=True)
            with autocast(enabled=bool(self.cfg.train.amp)):
                logits = self.model(images)
                loss = F.cross_entropy(logits, targets, label_smoothing=self.cfg.train.label_smoothing)
            self.scaler.scale(loss).backward()
            if self.cfg.train.clip_grad and self.cfg.train.clip_grad > 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad)
            self.scaler.step(self.optim)
            self.scaler.update()
            total_loss += float(loss.item()) * images.size(0)
            total += images.size(0)
            pbar.set_postfix(loss=float(loss.item()))
        return total_loss / max(total, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        correct = 0
        total = 0
        pbar = tqdm(self.val_loader, desc=f"Val {epoch:03d}", disable=not is_main_process())
        for images, targets, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
        return correct / max(total, 1)

    @torch.no_grad()
    def analyze_epoch(self, epoch: int) -> Dict[str, object]:
        self.model.eval()
        hooks_mgr = self.model.hooks_mgr

        layer_batches: Dict[str, List[np.ndarray]] = {name: [] for name in self.layer_names}
        labels_batches: List[np.ndarray] = []

        token_cov_sums: Dict[str, np.ndarray] = {}
        token_cov_counts: Dict[str, int] = {}

        for images, targets, _ in tqdm(self.analysis_loader, desc=f"Analysis {epoch:03d}", disable=not is_main_process()):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            _ = self.model(images)
            cache = dict(hooks_mgr.cache)

            labels_batches.append(targets.cpu().numpy())

            for name in self.layer_names:
                t = cache.get(name)
                if t is None:
                    continue

                t = t.detach().float()

                vec = _extract_image_vector(name, t).cpu().numpy()
                layer_batches[name].append(vec)

                tok = _token_tensor(name, t)
                if tok is not None:
                    tok_np = tok.cpu().numpy()
                    cov = token_covariance_matrix(tok_np, center_tokens=bool(self.cfg.analysis.twonn_use_centered))
                    if name not in token_cov_sums:
                        token_cov_sums[name] = np.zeros_like(cov, dtype=np.float64)
                        token_cov_counts[name] = 0
                    token_cov_sums[name] += cov * tok_np.shape[0]
                    token_cov_counts[name] += tok_np.shape[0]

            hooks_mgr.clear()

        labels = np.concatenate(labels_batches, axis=0)

        layer_vectors: Dict[str, np.ndarray] = {}
        for name, batches in layer_batches.items():
            if not batches:
                continue
            layer_vectors[name] = np.concatenate(batches, axis=0).astype(np.float64, copy=False)

        names = [n for n in self.layer_names if n in layer_vectors]
        final_name = names[-1]
        final_vectors = layer_vectors[final_name]

        probe_acc = linear_probe_accuracy(
            final_vectors,
            labels,
            test_fraction=self.cfg.analysis.probe_test_fraction,
            seed=epoch,
            max_iter=self.cfg.analysis.probe_max_iter,
        )
        fisher = fisher_ratio(final_vectors, labels)

        class_labels, centroids, centroid_dists = class_centroid_distances(
            final_vectors, labels, class_names=self.class_names
        )
        centroid_offdiag = centroid_dists[~np.eye(len(centroid_dists), dtype=bool)] if len(centroid_dists) > 1 else np.array([0.0])
        centroid_mean_dist = float(np.mean(centroid_offdiag)) if centroid_offdiag.size else 0.0
        centroid_median_dist = float(np.median(centroid_offdiag)) if centroid_offdiag.size else 0.0
        centroid_min_dist = float(np.min(centroid_offdiag)) if centroid_offdiag.size else 0.0
        centroid_max_dist = float(np.max(centroid_offdiag)) if centroid_offdiag.size else 0.0

        local_id = classwise_local_intrinsic_dimension(
            final_vectors,
            labels,
            centered=bool(self.cfg.analysis.twonn_use_centered),
            min_samples=self.cfg.analysis.min_class_samples,
        )
        local_pca = classwise_local_pca_rank(
            final_vectors,
            labels,
            variance_threshold=self.cfg.analysis.local_pca_variance_threshold,
            min_samples=self.cfg.analysis.min_class_samples,
        )

        layer_probe = layerwise_probe_accuracy(
            layer_vectors,
            labels,
            names,
            test_fraction=self.cfg.analysis.probe_test_fraction,
            seed=epoch,
            max_iter=self.cfg.analysis.probe_max_iter,
        )
        layer_fisher = layerwise_fisher_ratio(layer_vectors, labels, names)

        token_cov_mats: Dict[str, np.ndarray] = {}
        token_coupling_scores: Dict[str, float] = {}
        for name, cov_sum in token_cov_sums.items():
            count = max(token_cov_counts.get(name, 0), 1)
            cov = (cov_sum / float(count)).astype(np.float64, copy=False)
            token_cov_mats[name] = cov
            token_coupling_scores[name] = token_coupling_ratio(cov)

        token_cov_names = [n for n in self.layer_names if n in token_cov_mats]
        token_coupling_mean = float(np.mean([token_coupling_scores[n] for n in token_cov_names])) if token_cov_names else 0.0

        off_diag_mask = ~np.eye(len(names), dtype=bool) if len(names) else np.array([])
        cka_mat = pairwise_matrix(names, layer_vectors, linear_cka)
        svcca_mat = pairwise_matrix(names, layer_vectors, svcca)
        id_scores = {name: twonn_intrinsic_dimension(layer_vectors[name], centered=self.cfg.analysis.twonn_use_centered) for name in names}
        cka_mean = float(cka_mat[off_diag_mask].mean()) if len(names) > 1 else 0.0
        svcca_mean = float(svcca_mat[off_diag_mask].mean()) if len(names) > 1 else 0.0
        id_mean = float(np.mean(list(id_scores.values()))) if id_scores else 0.0
        adjacent_cka = adjacent_series(names, layer_vectors, linear_cka).mean() if len(names) > 1 else 0.0
        adjacent_svcca = adjacent_series(names, layer_vectors, svcca).mean() if len(names) > 1 else 0.0

        local_id_mean = _nanmean_dict(local_id)
        local_pca_mean = _nanmean_dict(local_pca)
        layer_probe_mean = float(np.mean(list(layer_probe.values()))) if layer_probe else 0.0
        layer_fisher_mean = float(np.mean(list(layer_fisher.values()))) if layer_fisher else 0.0

        epoch_dir = ensure_dir(self.results_dir / f"epoch_{epoch:03d}")
        np.save(epoch_dir / "cka_matrix.npy", cka_mat)
        np.save(epoch_dir / "svcca_matrix.npy", svcca_mat)
        np.save(epoch_dir / "twonn_id.npy", np.asarray([id_scores[n] for n in names], dtype=np.float64))
        np.save(epoch_dir / "centroid_distances.npy", centroid_dists.astype(np.float32, copy=False))
        np.save(epoch_dir / "final_layer_vectors.npy", final_vectors.astype(np.float32, copy=False))
        np.save(epoch_dir / "final_labels.npy", labels.astype(np.int64, copy=False))

        token_cov_dir = ensure_dir(epoch_dir / "token_covariance")
        for name, cov in token_cov_mats.items():
            np.save(token_cov_dir / f"{name}.npy", cov.astype(np.float32, copy=False))

        save_json(
            {
                "epoch": epoch,
                "layers": names,
                "class_labels": class_labels,
                "linear_probe_accuracy": float(probe_acc),
                "fisher_ratio": float(fisher),
                "centroid_mean_distance": centroid_mean_dist,
                "centroid_median_distance": centroid_median_dist,
                "centroid_min_distance": centroid_min_dist,
                "centroid_max_distance": centroid_max_dist,
                "local_intrinsic_dimension": local_id,
                "local_pca_rank": local_pca,
                "local_intrinsic_dimension_mean": local_id_mean,
                "local_pca_rank_mean": local_pca_mean,
                "layerwise_probe_accuracy": layer_probe,
                "layerwise_fisher_ratio": layer_fisher,
                "layerwise_probe_mean": layer_probe_mean,
                "layerwise_fisher_mean": layer_fisher_mean,
                "cka_mean": cka_mean,
                "svcca_mean": svcca_mean,
                "id_mean": id_mean,
                "adjacent_cka_mean": float(adjacent_cka),
                "adjacent_svcca_mean": float(adjacent_svcca),
                "token_coupling_mean": token_coupling_mean,
                "layer_ids": id_scores,
                "token_coupling_by_layer": token_coupling_scores,
            },
            epoch_dir / "epoch_metrics.json",
        )

        with open(self.results_dir / "epoch_summaries.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "linear_probe_accuracy": float(probe_acc),
                        "fisher_ratio": float(fisher),
                        "centroid_mean_distance": centroid_mean_dist,
                        "local_intrinsic_dimension_mean": local_id_mean,
                        "local_pca_rank_mean": local_pca_mean,
                        "layerwise_probe_mean": layer_probe_mean,
                        "layerwise_fisher_mean": layer_fisher_mean,
                        "cka_mean": cka_mean,
                        "svcca_mean": svcca_mean,
                        "id_mean": id_mean,
                        "adjacent_cka_mean": float(adjacent_cka),
                        "adjacent_svcca_mean": float(adjacent_svcca),
                        "token_coupling_mean": token_coupling_mean,
                        "layer_ids": id_scores,
                        "layerwise_probe_accuracy": layer_probe,
                        "layerwise_fisher_ratio": layer_fisher,
                        "token_coupling_by_layer": token_coupling_scores,
                    }
                )
                + "\n"
            )

        if epoch in set(self.cfg.analysis.snapshot_epochs) or (epoch % self.cfg.analysis.summary_every == 0) or (epoch == self.cfg.train.epochs):
            snap_dir = ensure_dir(self.summaries_dir / f"epoch_{epoch:03d}")
            plot_heatmap(centroid_dists, snap_dir / "centroid_distance_heatmap.png", title=f"Centroid Distances - Epoch {epoch}", cmap="magma", xticklabels=class_labels, yticklabels=class_labels)
            plot_bar(class_labels, [local_id.get(k, np.nan) for k in class_labels], snap_dir / "local_id_bar.png", title=f"Local ID by Class - Epoch {epoch}", ylabel="TwoNN ID")
            plot_bar(class_labels, [local_pca.get(k, np.nan) for k in class_labels], snap_dir / "local_pca_rank_bar.png", title=f"Local PCA Rank by Class - Epoch {epoch}", ylabel="PCA Rank")
            plot_bar(names, [layer_probe.get(n, 0.0) for n in names], snap_dir / "layerwise_probe_accuracy_bar.png", title=f"Layer-wise Probe Accuracy - Epoch {epoch}", ylabel="Accuracy")
            plot_bar(names, [layer_fisher.get(n, 0.0) for n in names], snap_dir / "layerwise_fisher_ratio_bar.png", title=f"Layer-wise Fisher Ratio - Epoch {epoch}", ylabel="Fisher Ratio")
            plot_heatmap(cka_mat, snap_dir / "cka_heatmap.png", title=f"CKA Heatmap - Epoch {epoch}", cmap="magma", vmin=0.0, vmax=1.0, xticklabels=names, yticklabels=names)
            plot_heatmap(svcca_mat, snap_dir / "svcca_heatmap.png", title=f"SVCCA Heatmap - Epoch {epoch}", cmap="magma", vmin=0.0, vmax=1.0, xticklabels=names, yticklabels=names)
            if token_cov_names:
                rep_name = token_cov_names[-1]
                rep_cov = token_cov_mats[rep_name]
                plot_heatmap(rep_cov, snap_dir / "token_covariance_heatmap.png", title=f"Token Covariance - {rep_name} - Epoch {epoch}", cmap="magma")
            plot_summary_grid(
                probe_acc,
                fisher,
                centroid_mean_dist,
                local_id_mean,
                local_pca_mean,
                snap_dir / "summary_grid.png",
                title=f"TGO-III Summary Epoch {epoch}",
            )

        return {
            "linear_probe_accuracy": float(probe_acc),
            "fisher_ratio": float(fisher),
            "centroid_mean_distance": centroid_mean_dist,
            "centroid_median_distance": centroid_median_dist,
            "centroid_min_distance": centroid_min_dist,
            "centroid_max_distance": centroid_max_dist,
            "local_intrinsic_dimension": local_id,
            "local_pca_rank": local_pca,
            "local_intrinsic_dimension_mean": local_id_mean,
            "local_pca_rank_mean": local_pca_mean,
            "layerwise_probe_accuracy": layer_probe,
            "layerwise_fisher_ratio": layer_fisher,
            "layerwise_probe_mean": layer_probe_mean,
            "layerwise_fisher_mean": layer_fisher_mean,
            "cka_mean": cka_mean,
            "svcca_mean": svcca_mean,
            "id_mean": id_mean,
            "adjacent_cka_mean": float(adjacent_cka),
            "adjacent_svcca_mean": float(adjacent_svcca),
            "token_coupling_mean": token_coupling_mean,
            "layer_ids": id_scores,
            "token_coupling_by_layer": token_coupling_scores,
        }

    def final_global_analysis(self):
        summary_file = self.results_dir / "epoch_summaries.jsonl"
        if not summary_file.exists():
            return
        epochs = []
        probe = []
        fisher = []
        centroid_mean = []
        local_id_mean = []
        local_pca_mean = []
        layer_probe_mean = []
        layer_fisher_mean = []
        adj_cka = []
        adj_svcca = []
        token_coupling_mean = []
        layer_id_series: Dict[str, List[float]] = {}
        token_coupling_series: Dict[str, List[float]] = {}
        layer_probe_series: Dict[str, List[float]] = {}
        layer_fisher_series: Dict[str, List[float]] = {}

        with open(summary_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                epochs.append(int(rec["epoch"]))
                probe.append(float(rec["linear_probe_accuracy"]))
                fisher.append(float(rec["fisher_ratio"]))
                centroid_mean.append(float(rec["centroid_mean_distance"]))
                local_id_mean.append(float(rec["local_intrinsic_dimension_mean"]))
                local_pca_mean.append(float(rec["local_pca_rank_mean"]))
                layer_probe_mean.append(float(rec["layerwise_probe_mean"]))
                layer_fisher_mean.append(float(rec["layerwise_fisher_mean"]))
                adj_cka.append(float(rec["adjacent_cka_mean"]))
                adj_svcca.append(float(rec["adjacent_svcca_mean"]))
                token_coupling_mean.append(float(rec.get("token_coupling_mean", 0.0)))
                for lname, val in rec["layer_ids"].items():
                    layer_id_series.setdefault(lname, []).append(float(val))
                for lname, val in rec.get("token_coupling_by_layer", {}).items():
                    token_coupling_series.setdefault(lname, []).append(float(val))
                for lname, val in rec.get("layerwise_probe_accuracy", {}).items():
                    layer_probe_series.setdefault(lname, []).append(float(val))
                for lname, val in rec.get("layerwise_fisher_ratio", {}).items():
                    layer_fisher_series.setdefault(lname, []).append(float(val))

        plot_curve(probe, self.global_dir / "linear_probe_accuracy_vs_epoch.png", title="Linear Probe Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy")
        plot_curve(fisher, self.global_dir / "fisher_ratio_vs_epoch.png", title="Fisher Ratio vs Epoch", xlabel="Epoch", ylabel="Fisher Ratio")
        plot_curve(centroid_mean, self.global_dir / "centroid_mean_distance_vs_epoch.png", title="Mean Centroid Distance vs Epoch", xlabel="Epoch", ylabel="Distance")
        plot_curve(local_id_mean, self.global_dir / "local_intrinsic_dimension_mean_vs_epoch.png", title="Mean Local Intrinsic Dimension vs Epoch", xlabel="Epoch", ylabel="TwoNN ID")
        plot_curve(local_pca_mean, self.global_dir / "local_pca_rank_mean_vs_epoch.png", title="Mean Local PCA Rank vs Epoch", xlabel="Epoch", ylabel="PCA Rank")
        plot_curve(layer_probe_mean, self.global_dir / "layerwise_probe_mean_vs_epoch.png", title="Mean Layer-wise Probe Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy")
        plot_curve(layer_fisher_mean, self.global_dir / "layerwise_fisher_mean_vs_epoch.png", title="Mean Layer-wise Fisher Ratio vs Epoch", xlabel="Epoch", ylabel="Fisher Ratio")
        plot_curve(adj_cka, self.global_dir / "adjacent_cka_mean_vs_epoch.png", title="Adjacent-layer Mean CKA vs Epoch", xlabel="Epoch", ylabel="CKA")
        plot_curve(adj_svcca, self.global_dir / "adjacent_svcca_mean_vs_epoch.png", title="Adjacent-layer Mean SVCCA vs Epoch", xlabel="Epoch", ylabel="SVCCA")
        plot_curve(token_coupling_mean, self.global_dir / "token_coupling_mean_vs_epoch.png", title="Mean Token Coupling Ratio vs Epoch", xlabel="Epoch", ylabel="Token Coupling Ratio")
        plot_multi_curve(layer_probe_series, self.global_dir / "layerwise_probe_accuracy_by_layer_vs_epoch.png", title="Layer-wise Probe Accuracy by Layer vs Epoch", xlabel="Epoch", ylabel="Accuracy")
        plot_multi_curve(layer_fisher_series, self.global_dir / "layerwise_fisher_ratio_by_layer_vs_epoch.png", title="Layer-wise Fisher Ratio by Layer vs Epoch", xlabel="Epoch", ylabel="Fisher Ratio")
        plot_multi_curve(layer_id_series, self.global_dir / "twonn_id_by_layer_vs_epoch.png", title="TwoNN ID by Layer vs Epoch", xlabel="Epoch", ylabel="ID")
        if token_coupling_series:
            plot_multi_curve(token_coupling_series, self.global_dir / "token_coupling_by_layer_vs_epoch.png", title="Token Coupling Ratio by Layer vs Epoch", xlabel="Epoch", ylabel="Token Coupling Ratio")

        last_epoch = epochs[-1]
        last_snap = self.summaries_dir / f"epoch_{last_epoch:03d}"
        if last_snap.exists():
            for fname in [
                "centroid_distance_heatmap.png",
                "local_id_bar.png",
                "local_pca_rank_bar.png",
                "layerwise_probe_accuracy_bar.png",
                "layerwise_fisher_ratio_bar.png",
                "cka_heatmap.png",
                "svcca_heatmap.png",
                "summary_grid.png",
                "token_covariance_heatmap.png",
            ]:
                src = last_snap / fname
                if src.exists():
                    dst = self.global_dir / f"final_{fname}"
                    dst.write_bytes(src.read_bytes())
