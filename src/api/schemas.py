import re
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal


_SAFE_ID_RE = re.compile(r'^[a-zA-Z0-9_\-.:]+$')
_MAX_ID_LEN = 128


class ScoringRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=_MAX_ID_LEN)
    user_id: str = Field(..., min_length=1, max_length=_MAX_ID_LEN)
    device_hash: str = Field(..., min_length=1, max_length=_MAX_ID_LEN)
    instrument_id: str = Field(..., min_length=1, max_length=_MAX_ID_LEN)
    ip: str = Field(..., min_length=2, max_length=45)  # IPv6 shortest "::1" = 3 chars, max 45
    amount: float = Field(..., gt=0, le=10_000_000)  # defense: reject non-positive and absurd amounts
    timestamp: str = Field(..., min_length=10, max_length=30)
    merchant_category_id: str = Field(..., min_length=1, max_length=10)

    @field_validator('transaction_id', 'user_id', 'device_hash', 'instrument_id', mode='after')
    @classmethod
    def validate_safe_id(cls, v: str) -> str:
        """Reject IDs with characters that could be used for injection attacks."""
        if not _SAFE_ID_RE.match(v):
            raise ValueError(f'ID contains invalid characters: {v!r}')
        return v

    @field_validator('ip', mode='after')
    @classmethod
    def validate_ip(cls, v: str) -> str:
        import ipaddress
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError(f'Invalid IP address: {v!r}')
        return v

    @field_validator('timestamp', mode='after')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        from datetime import datetime
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except Exception:
            raise ValueError(f'Invalid ISO-8601 timestamp: {v!r}')
        return v


class FeatureContribution(BaseModel):
    name: str
    value: float
    shap_contribution: float


class LatencyMetrics(BaseModel):
    l0: float = Field(..., ge=0)
    feature_fetch: float = Field(..., ge=0)
    model: float = Field(..., ge=0)
    total: float = Field(..., ge=0)


class ScoringResponse(BaseModel):
    transaction_id: str
    action: Literal['ALLOW', 'CHALLENGE', 'BLOCK']
    final_score: float = Field(..., ge=0.0, le=1.0)
    top_features: List[FeatureContribution]
    degraded_mode: bool
    latency_ms: LatencyMetrics
