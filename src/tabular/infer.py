import os
import logging
import numpy as np
import xgboost as xgb

logger = logging.getLogger('sentinel.model')

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'model.json')

# Canonical feature order — must match training. Defined once, shared everywhere.
FEATURE_NAMES = sorted([
    'amount', 'hour_of_day', 'is_weekend', 'account_age_days', 'historical_chargebacks',
    'device_reuse_count', 'instrument_reuse_count', 'velocity_5min', 'velocity_1hr',
    'ring_risk_score',
])

# Neutral defaults for degraded mode — one per feature, in canonical order
FEATURE_DEFAULTS = {
    'amount': 0.0,
    'hour_of_day': 12,
    'is_weekend': 0,
    'account_age_days': 30.0,
    'historical_chargebacks': 0,
    'device_reuse_count': 1,
    'instrument_reuse_count': 1,
    'velocity_5min': 0,
    'velocity_1hr': 0,
    'ring_risk_score': 0.1,
}


class UnifiedRiskModel:
    def __init__(self):
        self.is_loaded = False
        self.booster = None
        self._explainer = None  # Lazy-loaded SHAP TreeExplainer

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.booster = xgb.Booster()
            self.booster.load_model(MODEL_PATH)
            self.is_loaded = True
            logger.info('XGBoost model loaded from %s', MODEL_PATH)
        else:
            logger.warning('Model file not found at %s — running in mock mode', MODEL_PATH)
        return self

    @property
    def explainer(self):
        if self._explainer is None and self.is_loaded:
            import shap
            self._explainer = shap.TreeExplainer(self.booster)
        return self._explainer

    def _features_to_array(self, features: dict) -> np.ndarray:
        """Convert a feature dict to a 1×N numpy array in canonical order."""
        row = [float(features.get(f, FEATURE_DEFAULTS[f])) for f in FEATURE_NAMES]
        return np.array([row], dtype=np.float32)

    def predict_with_shap(self, features: dict, top_k: int = 3):
        """Return (score, top-k SHAP contributions).

        When the model is loaded, uses real shap.TreeExplainer.
        When the model is NOT loaded (mock mode), returns a neutral score
        with placeholder contributions so the API still responds.
        """
        if not self.is_loaded:
            return self._mock_predict(features, top_k)

        arr = self._features_to_array(features)
        dmat = xgb.DMatrix(arr, feature_names=FEATURE_NAMES)

        # Predict
        score = float(self.booster.predict(dmat)[0])
        score = max(0.0, min(1.0, score))  # clamp to [0,1]

        # Real SHAP
        shap_values = self.explainer.shap_values(dmat)
        contribs = shap_values[0]  # first (only) sample

        contributions = []
        for i, fname in enumerate(FEATURE_NAMES):
            contributions.append({
                'name': fname,
                'value': float(arr[0, i]),
                'shap_contribution': float(contribs[i]),
            })

        contributions.sort(key=lambda x: abs(x['shap_contribution']), reverse=True)
        return score, contributions[:top_k]

    def _mock_predict(self, features: dict, top_k: int):
        """Deterministic fallback when model file is missing."""
        score = 0.1  # neutral — don't use random!
        contributions = [
            {'name': f, 'value': float(features.get(f, 0)), 'shap_contribution': 0.0}
            for f in FEATURE_NAMES[:top_k]
        ]
        return score, contributions
