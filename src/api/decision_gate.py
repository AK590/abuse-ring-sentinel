"""Decision gate — the single readable function that evaluates every transaction.

Pipeline:
  1. L0 blocklist check (Redis, <1ms)
  2. Feature fetch (Redis real-time + SQLite/Postgres offline)
  3. XGBoost inference + SHAP
  4. Threshold → ALLOW / CHALLENGE / BLOCK
"""

import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Optional

import yaml

from src.api.schemas import (
    FeatureContribution,
    LatencyMetrics,
    ScoringRequest,
    ScoringResponse,
)
from src.realtime.counters import RealtimeCounters
from src.rules.blocklist import Blocklist
from src.tabular.features import assemble_features
from src.tabular.infer import FEATURE_DEFAULTS, UnifiedRiskModel

logger = logging.getLogger('sentinel.gate')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'cost_matrix.yaml')


def _load_config() -> dict:
    with open(_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


config = _load_config()

# ---------------------------------------------------------------------------
# Shared singletons (in production, wire through FastAPI Depends)
# ---------------------------------------------------------------------------
blocklist = Blocklist()
counters = RealtimeCounters()
model = UnifiedRiskModel().load()

_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'feature_store.db')


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def evaluate_risk_pipeline(req: ScoringRequest) -> ScoringResponse:
    """Run the full L0 → L1 risk pipeline and return a scored decision."""
    t_start = time.perf_counter()
    degraded_mode = False

    # ── 1. L0 blocklist ────────────────────────────────────────────────
    try:
        is_blocked = blocklist.check_transaction(
            req.user_id, req.device_hash, req.instrument_id, req.ip,
        )
    except Exception:
        logger.exception('Blocklist check failed — degrading to not-blocked')
        is_blocked = False
        degraded_mode = True

    t_l0 = time.perf_counter()

    if is_blocked:
        resp = _build_response(
            req, action='BLOCK', score=1.0, top_features=[],
            degraded=degraded_mode,
            t_start=t_start, t_l0=t_l0, t_feat=t_l0, t_model=t_l0,
        )
        _log_decision(req, resp)
        return resp

    # ── 2. Feature fetch ───────────────────────────────────────────────
    realtime_feats = _fetch_realtime(req)
    if realtime_feats is None:
        degraded_mode = True
        realtime_feats = {
            'device_reuse_count': FEATURE_DEFAULTS['device_reuse_count'],
            'instrument_reuse_count': FEATURE_DEFAULTS['instrument_reuse_count'],
            'velocity_5min': FEATURE_DEFAULTS['velocity_5min'],
            'velocity_1hr': FEATURE_DEFAULTS['velocity_1hr'],
        }

    offline_feats = _fetch_offline(req.user_id)
    if offline_feats is None:
        degraded_mode = True
        offline_feats = {'ring_risk_score': FEATURE_DEFAULTS['ring_risk_score']}

    t_feat = time.perf_counter()

    # ── 3. Model inference ─────────────────────────────────────────────
    features = assemble_features(
        amount=req.amount,
        timestamp_iso=req.timestamp,
        realtime_feats=realtime_feats,
        offline_feats=offline_feats,
    )

    score, top_features = model.predict_with_shap(features)
    t_model = time.perf_counter()

    # ── 4. Threshold → action ──────────────────────────────────────────
    action = _score_to_action(score)

    resp = _build_response(
        req, action=action, score=score, top_features=top_features,
        degraded=degraded_mode,
        t_start=t_start, t_l0=t_l0, t_feat=t_feat, t_model=t_model,
    )
    _log_decision(req, resp)
    return resp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_realtime(req: ScoringRequest) -> Optional[dict]:
    try:
        feats = counters.get_features(req.user_id, req.device_hash, req.instrument_id)
        counters.record_transaction(
            req.transaction_id, req.user_id, req.device_hash, req.instrument_id,
        )
        return feats
    except Exception:
        logger.exception('Redis unavailable — using neutral realtime defaults')
        return None


def _fetch_offline(user_id: str) -> Optional[dict]:
    try:
        if not os.path.exists(_DB_PATH):
            return None
        conn = sqlite3.connect(_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT ring_risk_score FROM offline_features WHERE user_id = ?',
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {'ring_risk_score': row[0]}
        return {'ring_risk_score': FEATURE_DEFAULTS['ring_risk_score']}
    except Exception:
        logger.exception('Feature store unavailable — using neutral offline defaults')
        return None


def _score_to_action(score: float) -> str:
    if score >= config['thresholds']['block']:
        return 'BLOCK'
    elif score >= config['thresholds']['challenge']:
        return 'CHALLENGE'
    return 'ALLOW'


def _build_response(
    req: ScoringRequest,
    *,
    action: str,
    score: float,
    top_features: list,
    degraded: bool,
    t_start: float,
    t_l0: float,
    t_feat: float,
    t_model: float,
) -> ScoringResponse:
    return ScoringResponse(
        transaction_id=req.transaction_id,
        action=action,
        final_score=round(max(0.0, min(1.0, score)), 4),
        top_features=[FeatureContribution(**tf) for tf in top_features],
        degraded_mode=degraded,
        latency_ms=LatencyMetrics(
            l0=round((t_l0 - t_start) * 1000, 2),
            feature_fetch=round((t_feat - t_l0) * 1000, 2),
            model=round((t_model - t_feat) * 1000, 2),
            total=round((t_model - t_start) * 1000, 2),
        ),
    )


def _log_decision(req: ScoringRequest, resp: ScoringResponse) -> None:
    logger.info(
        'DECISION tx=%s user=%s action=%s score=%.4f degraded=%s latency_ms=%.2f',
        req.transaction_id,
        req.user_id,
        resp.action,
        resp.final_score,
        resp.degraded_mode,
        resp.latency_ms.total,
    )
