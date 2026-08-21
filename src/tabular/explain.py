"""SHAP explainability wrapper.

Provides a standalone function to explain an existing prediction,
useful for batch explanations and the ablation notebook.
"""

import logging
import numpy as np
import xgboost as xgb

from src.tabular.infer import UnifiedRiskModel, FEATURE_NAMES

logger = logging.getLogger('sentinel.explain')


def explain_prediction(
    model: UnifiedRiskModel,
    features: dict,
    top_k: int = 5,
) -> list[dict]:
    """Return the top-k SHAP feature contributions for a single prediction.

    Unlike predict_with_shap, this returns ALL requested features (not just top-3)
    and does not re-run the prediction — it only runs SHAP.
    """
    if not model.is_loaded:
        logger.warning('Model not loaded — returning empty explanation.')
        return []

    arr = model._features_to_array(features)
    dmat = xgb.DMatrix(arr, feature_names=FEATURE_NAMES)
    shap_values = model.explainer.shap_values(dmat)
    contribs = shap_values[0]

    explanations = []
    for i, fname in enumerate(FEATURE_NAMES):
        explanations.append({
            'name': fname,
            'value': float(arr[0, i]),
            'shap_contribution': float(contribs[i]),
        })

    explanations.sort(key=lambda x: abs(x['shap_contribution']), reverse=True)
    return explanations[:top_k]
