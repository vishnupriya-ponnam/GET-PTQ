"""Dice, IoU and HD95, computed exactly as in GET's valid.py."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

try:
    from medpy.metric.binary import hd95 as _medpy_hd95
except Exception:  # medpy not installed: use an identical scipy implementation
    _medpy_hd95 = None


def _surface_distances(result, reference):
    # Same algorithm as medpy.metric.binary.__surface_distances (connectivity 1, unit spacing)
    footprint = ndimage.generate_binary_structure(result.ndim, 1)
    result_border = result ^ ndimage.binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ ndimage.binary_erosion(reference, structure=footprint, iterations=1)
    dt = ndimage.distance_transform_edt(~reference_border)
    return dt[result_border]


def _hd95_scipy(result, reference):
    hd1 = _surface_distances(result, reference)
    hd2 = _surface_distances(reference, result)
    return float(np.percentile(np.hstack((hd1, hd2)), 95))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if (pred.sum() == 0) ^ (gt.sum() == 0):  # GET's convention: image diagonal
        h, w = gt.shape
        return float(np.hypot(h, w))
    return float(_medpy_hd95(pred, gt)) if _medpy_hd95 else _hd95_scipy(pred, gt)


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    d_den, i_den = 2 * tp + fp + fn, tp + fp + fn
    return (2 * tp / d_den if d_den else 0.0), (tp / i_den if i_den else 0.0)


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    d, i = dice_iou(pred, gt)
    return {"dice": float(d), "iou": float(i), "hd95": hd95(pred, gt)}
