from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def center(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = center(X)
    Y = center(Y)
    XTY = X.T @ Y
    XTX = X.T @ X
    YTY = Y.T @ Y
    denom = np.linalg.norm(XTX, ord="fro") * np.linalg.norm(YTY, ord="fro")
    if denom <= 0:
        return 0.0
    return float((np.linalg.norm(XTY, ord="fro") ** 2) / denom)


def _pca_reduce(X: np.ndarray, variance_threshold: float = 0.99, max_components: int = 50) -> np.ndarray:
    X = center(X)
    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features)
    if n_components <= 1:
        return X
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    Z = pca.fit_transform(X)
    evr = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(evr, variance_threshold) + 1)
    k = max(1, min(k, max_components, Z.shape[1]))
    return Z[:, :k]


def svcca(X: np.ndarray, Y: np.ndarray, variance_threshold: float = 0.99, max_components: int = 50) -> float:
    Xr = _pca_reduce(X, variance_threshold=variance_threshold, max_components=max_components)
    Yr = _pca_reduce(Y, variance_threshold=variance_threshold, max_components=max_components)
    n = min(Xr.shape[0], Yr.shape[0], Xr.shape[1], Yr.shape[1])
    if n < 2:
        return 0.0
    Xr = Xr[:, :n]
    Yr = Yr[:, :n]
    cca = CCA(n_components=n, max_iter=1000)
    try:
        cca.fit(Xr, Yr)
        U, V = cca.transform(Xr, Yr)
        corrs = []
        for i in range(U.shape[1]):
            u = U[:, i]
            v = V[:, i]
            if np.std(u) < 1e-12 or np.std(v) < 1e-12:
                continue
            c = np.corrcoef(u, v)[0, 1]
            if np.isfinite(c):
                corrs.append(abs(float(c)))
        return float(np.mean(corrs)) if corrs else 0.0
    except Exception:
        return 0.0


def twonn_intrinsic_dimension(X: np.ndarray, centered: bool = True) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 5:
        return 0.0
    if centered:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mask = (r1 > 1e-12) & (r2 > r1)
    mu = r2[mask] / r1[mask]
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    if mu.size < 5:
        return 0.0
    mu = np.sort(mu)
    n = mu.size
    F = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    x = np.log(mu)
    y = -np.log(np.clip(1.0 - F, 1e-12, 1.0))
    lo = int(0.1 * n)
    hi = max(lo + 5, int(0.9 * n))
    x_fit = x[lo:hi]
    y_fit = y[lo:hi]
    if x_fit.size < 5:
        x_fit, y_fit = x, y
    try:
        slope, _ = np.polyfit(x_fit, y_fit, 1)
        if np.isfinite(slope) and slope > 0:
            return float(slope)
    except Exception:
        pass
    denom = np.dot(x_fit, x_fit)
    if denom <= 0:
        return 0.0
    slope = float(np.dot(x_fit, y_fit) / denom)
    return max(slope, 0.0)


def token_covariance_matrix(X: np.ndarray, center_tokens: bool = True) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        if X.shape[0] < 2 or X.shape[1] < 2:
            return np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
        if center_tokens:
            X = X - X.mean(axis=1, keepdims=True)
        denom = max(X.shape[1] - 1, 1)
        return (X @ X.T) / denom
    if X.ndim == 3:
        if X.shape[1] < 2 or X.shape[2] < 2:
            return np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        if center_tokens:
            X = X - X.mean(axis=-1, keepdims=True)
        denom = max(X.shape[2] - 1, 1)
        cov = np.einsum("btd,bsd->bts", X, X) / denom
        return cov.mean(axis=0)
    raise ValueError(f"Expected a 2D or 3D array, got shape {X.shape}")


def token_coupling_ratio(C: np.ndarray) -> float:
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {C.shape}")
    absC = np.abs(C)
    total = float(absC.sum())
    if total <= 0:
        return 0.0
    diag = float(np.abs(np.diag(C)).sum())
    off = max(total - diag, 0.0)
    return float(off / total)


def pairwise_matrix(
    names: List[str],
    data: Dict[str, np.ndarray],
    fn,
) -> np.ndarray:
    n = len(names)
    out = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(names):
        out[i, i] = 1.0 if fn in {linear_cka, svcca} else 0.0
        for j in range(i + 1, n):
            val = fn(data[a], data[names[j]])
            out[i, j] = out[j, i] = float(val)
    return out


def adjacent_series(names: List[str], data: Dict[str, np.ndarray], fn) -> np.ndarray:
    vals = []
    for i in range(len(names) - 1):
        vals.append(fn(data[names[i]], data[names[i + 1]]))
    return np.asarray(vals, dtype=np.float64)


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return X


def fisher_ratio(X: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    X = _ensure_2d(X)
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size < 2:
        return 0.0
    mu = X.mean(axis=0, keepdims=True)
    sw = 0.0
    sb = 0.0
    for c in classes:
        Xc = X[y == c]
        if Xc.shape[0] == 0:
            continue
        muc = Xc.mean(axis=0, keepdims=True)
        diff = Xc - muc
        sw += float(np.sum(diff * diff))
        dm = muc - mu
        sb += float(Xc.shape[0] * np.sum(dm * dm))
    return float(sb / (sw + eps))


def linear_probe_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 0,
    max_iter: int = 2000,
) -> float:
    X = _ensure_2d(X)
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return 0.0
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    try:
        train_idx, test_idx = next(splitter.split(X, y))
    except Exception:
        return 0.0
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[train_idx])
    Xte = scaler.transform(X[test_idx])
    clf = RidgeClassifier(alpha=1.0)
    try:
        clf.fit(Xtr, y[train_idx])
        pred = clf.predict(Xte)
        return float(accuracy_score(y[test_idx], pred))
    except Exception:
        return 0.0


def class_centroid_distances(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    X = _ensure_2d(X)
    y = np.asarray(y)
    classes = np.unique(y)
    if class_names is not None and len(class_names) >= int(classes.max()) + 1:
        labels = [class_names[int(c)] for c in classes]
    else:
        labels = [str(int(c)) for c in classes]
    centroids = np.stack([X[y == c].mean(axis=0) for c in classes], axis=0)
    diffs = centroids[:, None, :] - centroids[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return labels, centroids, dists


def classwise_local_intrinsic_dimension(
    X: np.ndarray,
    y: np.ndarray,
    centered: bool = True,
    min_samples: int = 5,
) -> Dict[str, float]:
    X = _ensure_2d(X)
    y = np.asarray(y)
    out: Dict[str, float] = {}
    for c in np.unique(y):
        Xc = X[y == c]
        if Xc.shape[0] < min_samples:
            continue
        out[str(int(c))] = twonn_intrinsic_dimension(Xc, centered=centered)
    return out


def classwise_local_pca_rank(
    X: np.ndarray,
    y: np.ndarray,
    variance_threshold: float = 0.95,
    min_samples: int = 3,
) -> Dict[str, float]:
    X = _ensure_2d(X)
    y = np.asarray(y)
    out: Dict[str, float] = {}
    for c in np.unique(y):
        Xc = X[y == c]
        if Xc.shape[0] < min_samples:
            continue
        Xc = center(Xc)
        n_components = min(Xc.shape[0], Xc.shape[1])
        if n_components <= 1:
            out[str(int(c))] = 1.0
            continue
        pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
        pca.fit(Xc)
        evr = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(evr, variance_threshold) + 1)
        out[str(int(c))] = float(max(1, min(k, n_components)))
    return out


def layerwise_probe_accuracy(
    layer_vectors: Dict[str, np.ndarray],
    y: np.ndarray,
    layer_names: List[str],
    test_fraction: float = 0.2,
    seed: int = 0,
    max_iter: int = 2000,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lname in layer_names:
        X = layer_vectors.get(lname)
        if X is None:
            continue
        out[lname] = linear_probe_accuracy(X, y, test_fraction=test_fraction, seed=seed, max_iter=max_iter)
    return out


def layerwise_fisher_ratio(
    layer_vectors: Dict[str, np.ndarray],
    y: np.ndarray,
    layer_names: List[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lname in layer_names:
        X = layer_vectors.get(lname)
        if X is None:
            continue
        out[lname] = fisher_ratio(X, y)
    return out
