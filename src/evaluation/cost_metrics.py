"""Business-cost metrics for the fraud model.

Business cost = (FP × C_LTV) + (FN × C_chargeback)

where:
- FP = false positives (legit users blocked/challenged → lost LTV)
- FN = false negatives (fraudsters allowed → chargeback cost)
"""

import logging
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

logger = logging.getLogger('sentinel.cost_metrics')


def compute_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c_fp: float = 45.0,
    c_fn: float = 900.0,
) -> dict:
    """Compute business cost and classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * c_fp) + (fn * c_fn)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        'precision': prec,
        'recall': rec,
        'fpr': fpr,
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'tn': int(tn),
        'total_business_cost': total_cost,
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    c_fp: float = 45.0,
    c_fn: float = 900.0,
    num_points: int = 200,
) -> dict:
    """Sweep thresholds to find the one minimizing business cost.

    Returns the best threshold and its associated metrics.
    """
    best_cost = float('inf')
    best_threshold = 0.5
    best_metrics = {}

    for t in np.linspace(0.01, 0.99, num_points):
        y_pred = (y_proba >= t).astype(int)
        metrics = compute_cost(y_true, y_pred, c_fp, c_fn)
        if metrics['total_business_cost'] < best_cost:
            best_cost = metrics['total_business_cost']
            best_threshold = float(t)
            best_metrics = metrics

    best_metrics['optimal_threshold'] = best_threshold
    logger.info(
        'Threshold sweep: optimal=%.4f cost=%.2f prec=%.4f rec=%.4f fpr=%.4f',
        best_threshold,
        best_cost,
        best_metrics['precision'],
        best_metrics['recall'],
        best_metrics['fpr'],
    )
    return best_metrics
