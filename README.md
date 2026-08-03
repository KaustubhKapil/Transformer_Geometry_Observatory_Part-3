# TGO-III: Semantic Geometry Observatory

This repository implements a **Semantic Geometry Observatory** for ViT-Small/16 on ImageNet-100. It is the third part of the **Transformer Geometry Observatory (TGO)** framework.

While TGO-I explored the spectral geometry of Vision Transformers through covariance structure, eigenspectra, rank evolution, and dimensional utilization, and TGO-II focused on representation similarity and intrinsic dimensionality, TGO-III focuses on **semantic structure**.

Here we analyze:

- Linear Probe Accuracy
- Fisher Ratio
- Class Centroid Distances
- Local Intrinsic Dimension
- Local PCA Rank
- Layer-wise Probe Accuracy
- Layer-wise Fisher Ratio

These observables are used to answer:

- Do features become more class-separable during training?
- Do class centroids spread apart in semantic space?
- Does the local manifold of each class expand?
- Do deeper layers become more semantically organized than earlier layers?
- Does a transition appear around the middle blocks of the ViT?

The primary research question of TGO-III is:

> Can semantic geometry become more structured while the representation manifold continues to expand?

---

## Research Motivation

TGO-I showed that spectral structure evolves over training.

TGO-II showed that representational similarity decreases while intrinsic dimensionality increases.

TGO-III asks the next question:

> Does this geometric change actually correspond to semantic separation?

TGO-III is designed to answer this question through direct class-level analysis.

## Files

- `tgo_iii/main.py` — entry point
- `tgo_iii/trainer.py` — training and observatory pipeline
- `tgo_iii/dataset.py` — dataset construction and fixed subset selection
- `tgo_iii/hooks.py` — ViT activation capture
- `tgo_iii/metrics.py` — probe, Fisher, centroid, and local manifold metrics
- `tgo_iii/model.py` — ViT wrapper
- `tgo_iii/visualization.py` — plotting helpers
- `tgo_iii/utils.py` — logging, seeding, JSON helpers

## Run

```bash
python -m tgo_iii.main --config configs/vit_small_imagenet100.yaml
```

Optional resume:

```bash
python -m tgo_iii.main \
    --config configs/vit_small_imagenet100.yaml \
    --resume results_tgo_iii/checkpoints/last.pth
```

---

## Expected Data Layout

```text
/path/to/imagenet100/train/<class_name>/*.JPEG
/path/to/imagenet100/val/<class_name>/*.JPEG
```

---

## Outputs

```text
results_tgo_iii/
├── checkpoints/
│   ├── best.pth
│   └── last.pth
│
├── summaries/
│   ├── epoch_001.json
│   ├── epoch_002.json
│   └── ...
│
├── linear_probe/
├── fisher_ratio/
├── class_centroid_distances/
├── local_intrinsic_dimension/
├── local_pca_rank/
├── layerwise_probe_accuracy/
├── layerwise_fisher_ratio/
│
├── global_analysis/
│   ├── linear_probe_accuracy_vs_epoch.png
│   ├── fisher_ratio_vs_epoch.png
│   ├── centroid_mean_distance_vs_epoch.png
│   ├── local_intrinsic_dimension_mean_vs_epoch.png
│   ├── local_pca_rank_mean_vs_epoch.png
│   ├── layerwise_probe_mean_vs_epoch.png
│   └── layerwise_fisher_mean_vs_epoch.png
│
└── logs/
```

---

## Checkpointing

Only two checkpoints are maintained:

```text
best.pth
last.pth
```

- `last.pth` is overwritten every epoch.
- `best.pth` is updated whenever validation accuracy improves.

This keeps storage overhead low during long training runs.

## Notes

- Analysis metrics are computed on a fixed validation subset.
- Linear probes are trained on frozen features.
- Fisher Ratio measures between-class scatter over within-class scatter.
- Class centroid distances measure how far apart semantic clusters are.
- Local Intrinsic Dimension estimates manifold dimension per class.
- Local PCA Rank estimates how many directions each class needs locally.
- The observatory tracks layer-wise probe accuracy and Fisher Ratio to see where semantic structure appears.

## Relation to TGO-II

TGO-II focused on:

- CKA
- SVCCA
- TwoNN intrinsic dimension
- Token covariance

TGO-III keeps the same observatory style, but shifts the question from representation similarity to semantic organization.

The findings of TGO-III will determine whether semantic structure appears as:

- Better class separability
- Larger between-class distances
- Higher local manifold dimension
- Stronger layer-wise discrimination


## Results
TGO-III extends the Transformer Geometry Observatory by investigating the emergence of semantic organization throughout Vision Transformer training. Unlike TGO-I and TGO-II, which focused on spectral and representation geometry, TGO-III analyzes how class-level structure evolves using Linear Probe Accuracy, Fisher Ratio, Class Centroid Distances, and Local PCA Rank.

Our experiments demonstrate a consistent improvement in semantic organization as training progresses. Linear Probe Accuracy increases steadily across optimization, with the strongest discriminative representations emerging in the final Transformer blocks and the CLS token. Fisher Ratio similarly increases throughout training, indicating progressively better separation between semantic classes. Mean pairwise centroid distances expand before stabilizing, suggesting that class representations occupy increasingly distinct regions of the feature space. Local PCA Rank initially increases before gradually decreasing, indicating an early expansion of local semantic manifolds followed by progressive refinement into compact and discriminative class-specific representations.

Collectively, these observations support the Semantic Expansion Hypothesis, providing empirical evidence that the representational directions discovered throughout training are increasingly allocated toward encoding discriminative semantic concepts rather than merely increasing representational complexity. Together with the findings of TGO-I and TGO-II, TGO-III completes the initial geometric investigation of Transformer learning by establishing a direct connection between spectral evolution, manifold expansion, and semantic organization.

## Pre-Print
Further details have been provided in my pre-print, please go through and share your valuable feedback. I would really appreciate it
`https://www.researchgate.net/publication/411039167_TGO-III_Semantic_Geometry_Observatory`
