# Abuse-Ring Sentinel v2 — AI Risk Manager

> Razorpay Buildathon — Track 2: AI Risk Manager

Real-time fraud detection system combining **offline graph reasoning** (R-GCN) with **real-time Redis counters** and a **unified XGBoost model** for explainable risk scoring.

## Architecture

```
┌─────────────────────┐
│  Offline batch job   │   (every 4–6 hrs)
│  R-GCN over full     │   Scores every user/device
│  transaction graph   │   → ring_risk_score
└──────────┬──────────┘
           │ writes to
           ▼
┌─────────────────────┐      ┌──────────────────┐
│   Feature Store      │◀────│ Real-time Redis   │
│   (SQLite/Postgres)  │      │ counters          │
└──────────┬──────────┘      └────────┬─────────┘
           │                          │
           ▼                          ▼
   ┌───────────────┐        ┌──────────────────┐
   │ L0: Blocklist  │        │ L1: XGBoost       │
   │ (Redis, <1ms)  │        │ tabular + graph   │
   └───────┬───────┘        │ features (<15ms)  │
           │                 └────────┬─────────┘
           └────────────┬─────────────┘
                        ▼
               ┌──────────────────┐
               │  Decision Gate    │
               │  ALLOW/CHALLENGE/ │
               │  BLOCK            │
               └──────────────────┘
```

## Quick Start

```bash
# 1. Setup
uv venv && .venv/Scripts/activate   # Windows
uv pip install -e .

# 2. Generate synthetic data + train model
python data/synthetic_generator.py
python src/tabular/train.py

# 3. Run R-GCN batch job (offline graph scoring)
python src/graph/batch_score.py

# 4. Start API server
uvicorn src.api.main:app --reload

# 5. Test
python -m pytest tests/ -v
```

## API

### POST /score

```json
{
  "transaction_id": "txn_001",
  "user_id": "usr_42",
  "device_hash": "dev_abc",
  "instrument_id": "instr_xyz",
  "ip": "203.0.113.4",
  "amount": 4999.0,
  "timestamp": "2026-08-21T10:15:00Z",
  "merchant_category_id": "5732"
}
```

### Response

```json
{
  "transaction_id": "txn_001",
  "action": "CHALLENGE",
  "final_score": 0.74,
  "top_features": [
    {"name": "ring_risk_score", "value": 0.91, "shap_contribution": 0.31},
    {"name": "device_reuse_count", "value": 4, "shap_contribution": 0.18}
  ],
  "degraded_mode": false,
  "latency_ms": {"l0": 0.4, "feature_fetch": 3.1, "model": 11.2, "total": 14.7}
}
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Graph scoring offline only | Eliminates live GNN latency risk; batch job can take minutes without affecting p99 |
| Single XGBoost model | One model to explain via SHAP; no arbitrary score blending |
| Fail-open on L0, fail-neutral on features | Redis down → neutral defaults, not a hard block |
| 60-day label lag cutoff | Chargebacks take 45–180 days; recent "legit" labels are unreliable |
| Chronological split only | Prevents temporal leakage that inflates metrics |

## Testing

```bash
python -m pytest tests/ -v
```

Covers: blocklist enforcement, real-time counters, decision gate integration,
degraded mode (Redis/Postgres down), label-lag cutoff, chronological split,
input validation, latency budgets.

## Evaluation

```bash
python src/evaluation/report.py
```

Prints: Precision, Recall, FPR, Total Business Cost on the held-out chronological test set.
