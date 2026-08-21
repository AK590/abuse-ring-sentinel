"""Tests for feature store degraded mode.

Verifies that when Redis or Postgres/SQLite is unreachable,
the system degrades gracefully to neutral defaults instead
of failing hard.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.api.schemas import ScoringRequest
from src.api.decision_gate import evaluate_risk_pipeline
from src.rules.blocklist import get_redis_client


def _make_req(**overrides) -> ScoringRequest:
    base = {
        "transaction_id": "txn_degrade_test",
        "user_id": "usr_test",
        "device_hash": "dev_test",
        "instrument_id": "instr_test",
        "ip": "192.168.1.1",
        "amount": 100.0,
        "timestamp": "2026-08-21T10:15:00Z",
        "merchant_category_id": "5732",
    }
    base.update(overrides)
    return ScoringRequest(**base)


@pytest.fixture
def clean_redis():
    client = get_redis_client()
    client.flushall()
    yield client
    client.flushall()


class TestRedisUnavailable:
    """When Redis is down, system should degrade to neutral defaults."""

    def test_redis_down_returns_valid_decision(self, monkeypatch, clean_redis):
        from src.api import decision_gate

        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        def boom(*a, **kw):
            raise ConnectionError("Redis connection refused")

        monkeypatch.setattr(decision_gate.counters, "get_features", boom)

        resp = evaluate_risk_pipeline(_make_req())
        assert resp.degraded_mode is True
        assert resp.action in ("ALLOW", "CHALLENGE", "BLOCK")
        assert resp.final_score >= 0.0
        assert resp.final_score <= 1.0

    def test_blocklist_redis_down_does_not_hard_fail(self, monkeypatch, clean_redis):
        """If the blocklist Redis call itself fails, treat as not-blocked."""
        from src.api import decision_gate

        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        def boom(*a, **kw):
            raise ConnectionError("Redis connection refused")

        monkeypatch.setattr(decision_gate.blocklist, "check_transaction", boom)

        resp = evaluate_risk_pipeline(_make_req())
        assert resp.degraded_mode is True
        assert resp.action in ("ALLOW", "CHALLENGE", "BLOCK")


class TestFeatureStoreUnavailable:
    """When the SQLite feature store is missing or corrupt."""

    def test_missing_db_uses_neutral_score(self, monkeypatch, clean_redis):
        from src.api import decision_gate

        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)
        monkeypatch.setattr(decision_gate, "_DB_PATH", "/nonexistent/path.db")

        resp = evaluate_risk_pipeline(_make_req())
        # Should still work with neutral defaults
        assert resp.action in ("ALLOW", "CHALLENGE", "BLOCK")

    def test_corrupt_db_degrades(self, monkeypatch, clean_redis, tmp_path):
        from src.api import decision_gate

        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        corrupt_db = tmp_path / "corrupt.db"
        corrupt_db.write_text("this is not a sqlite database")
        monkeypatch.setattr(decision_gate, "_DB_PATH", str(corrupt_db))

        resp = evaluate_risk_pipeline(_make_req())
        assert resp.degraded_mode is True
        assert resp.action in ("ALLOW", "CHALLENGE", "BLOCK")


class TestLatencyBudget:
    """Verify end-to-end latency stays within the 50ms p99 budget."""

    def test_latency_under_50ms(self, monkeypatch, clean_redis):
        from src.api import decision_gate

        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        resp = evaluate_risk_pipeline(_make_req())
        assert resp.latency_ms.total < 50.0, (
            f"Latency {resp.latency_ms.total}ms exceeds 50ms budget"
        )
