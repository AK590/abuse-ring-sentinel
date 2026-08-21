"""Feature assembly layer.

Merges tabular request data, real-time Redis counters, and offline
graph-derived scores into the canonical feature dict expected by
the XGBoost model.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from src.tabular.infer import FEATURE_NAMES, FEATURE_DEFAULTS

logger = logging.getLogger('sentinel.features')


def assemble_features(
    amount: float,
    timestamp_iso: str,
    realtime_feats: Dict[str, Any],
    offline_feats: Dict[str, Any],
    account_age_days: float = 30.0,
    historical_chargebacks: int = 0,
) -> Dict[str, Any]:
    """Build the canonical feature dict for model inference.

    Missing keys are filled with neutral defaults from FEATURE_DEFAULTS.
    """
    # Parse timestamp
    try:
        dt = datetime.fromisoformat(timestamp_iso.replace('Z', '+00:00'))
        hour_of_day = dt.hour
        is_weekend = 1 if dt.weekday() >= 5 else 0
    except Exception:
        logger.warning('Failed to parse timestamp %r, using defaults', timestamp_iso)
        hour_of_day = FEATURE_DEFAULTS['hour_of_day']
        is_weekend = FEATURE_DEFAULTS['is_weekend']

    raw = {
        'amount': amount,
        'hour_of_day': hour_of_day,
        'is_weekend': is_weekend,
        'account_age_days': account_age_days,
        'historical_chargebacks': historical_chargebacks,
        **realtime_feats,
        **offline_feats,
    }

    # Ensure every canonical feature is present
    features: Dict[str, Any] = {}
    for f in FEATURE_NAMES:
        features[f] = raw.get(f, FEATURE_DEFAULTS[f])

    return features
