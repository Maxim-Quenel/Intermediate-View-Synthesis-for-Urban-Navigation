from typing import Optional

import cv2
import numpy as np


DEFAULT_MIN_VALID_DEPTH = 0.05
EDGE_PERCENTILE = 90.0


def build_valid_mask(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    min_valid_depth: float = DEFAULT_MIN_VALID_DEPTH,
) -> np.ndarray:
    mask = np.isfinite(depth_a) & np.isfinite(depth_b)
    mask &= (depth_a > min_valid_depth) & (depth_b > min_valid_depth)
    return mask


def _edge_map(depth: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    if mask is not None:
        mag = np.where(mask, mag, 0.0)
        valid = mag[mask]
    else:
        valid = mag.ravel()
    if valid.size == 0:
        return None
    thresh = np.percentile(valid, EDGE_PERCENTILE)
    if thresh <= 0:
        return None
    return mag >= thresh


def _f1_score(edge_a: Optional[np.ndarray], edge_b: Optional[np.ndarray]) -> Optional[float]:
    if edge_a is None or edge_b is None:
        return None
    tp = int(np.logical_and(edge_a, edge_b).sum())
    fp = int(np.logical_and(edge_a, ~edge_b).sum())
    fn = int(np.logical_and(~edge_a, edge_b).sum())
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return float((2 * tp) / denom)


def compute_depth_metrics(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    min_valid_depth: float = DEFAULT_MIN_VALID_DEPTH,
) -> dict:
    if depth_a.shape != depth_b.shape:
        raise ValueError("Depth maps must have the same shape for comparison.")

    mask = build_valid_mask(depth_a, depth_b, min_valid_depth=min_valid_depth)
    total = int(mask.size)
    valid_count = int(mask.sum())
    valid_ratio = float(valid_count / total) if total else 0.0

    if valid_count == 0:
        return {
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "edge_f1": None,
            "valid_ratio": valid_ratio,
        }

    a = depth_a[mask].astype(np.float32, copy=False)
    b = depth_b[mask].astype(np.float32, copy=False)
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a < 1e-6 or std_b < 1e-6:
        pearson_r = None
    else:
        pearson_r = float(np.corrcoef(a, b)[0, 1])

    edge_a = _edge_map(depth_a.astype(np.float32, copy=False), mask)
    edge_b = _edge_map(depth_b.astype(np.float32, copy=False), mask)
    edge_f1 = _f1_score(edge_a, edge_b)

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "edge_f1": edge_f1,
        "valid_ratio": valid_ratio,
    }
