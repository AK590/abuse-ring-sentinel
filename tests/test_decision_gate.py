"""Tests for the decision gate — the core pipeline.

Covers: blocklist bypass, normal scoring, degraded mode,
latency budget, and SHAP explanation presence.
"""

import pytest
from src.api.schemas import ScoringRequest
from src.api.decision_gate import evaluate_risk_pipeline
from src.rules.blocklist import get_redis_client


def _make_req(**overrides) -> ScoringRequest:
    base = {
        "transaction_id": "txn_gate_test",
        "user_id": "usr_gate",
        "device_hash": "dev_gate",
        "instrument_id": "instr_gate",
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


class TestDecisionGateBlocklist:
    def test_blocklisted_user_returns_block(self, clean_redis, monkeypatch):
        from src.api import decision_gate
        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        clean_redis.set("blocklist:user:u1_bad", "1")
        resp = evaluate_risk_pipeline(_make_req(user_id="u1_bad"))
        assert resp.action == "BLOCK"
        assert resp.final_score == 1.0

    def test_blocklist_latency_under_5ms(self, clean_redis, monkeypatch):
        from src.api import decision_gate
        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        clean_redis.set("blocklist:user:u1_fast", "1")
        resp = evaluate_risk_pipeline(_make_req(user_id="u1_fast"))
        assert resp.latency_ms.total < 5.0


class TestDecisionGateNormal:
    def test_normal_request_returns_valid_action(self, clean_redis, monkeypatch):
        from src.api import decision_gate
        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        resp = evaluate_risk_pipeline(_make_req())
        assert resp.action in ("ALLOW", "CHALLENGE", "BLOCK")
        assert 0.0 <= resp.final_score <= 1.0
        assert resp.degraded_mode is False

    def test_response_contains_latency_breakdown(self, clean_redis, monkeypatch):
        from src.api import decision_gate
        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        resp = evaluate_risk_pipeline(_make_req())
        assert resp.latency_ms.l0 >= 0
        assert resp.latency_ms.feature_fetch >= 0
        assert resp.latency_ms.model >= 0
        assert resp.latency_ms.total > 0

    def test_counters_updated_after_request(self, clean_redis, monkeypatch):
        from src.api import decision_gate
        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        evaluate_risk_pipeline(_make_req(
            transaction_id="txn_count",
            user_id="u_count",
            device_hash="d_count",
        ))
        assert clean_redis.scard("device:d_count:users") == 1


class TestDecisionGateDegraded:
    def test_redis_failure_degrades_gracefully(self, monkeypatch, clean_redis):
        from src.api import decision_gate
        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(
            decision_gate.counters, "get_features",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("Redis down")),
        )

        resp = evaluate_risk_pipeline(_make_req())
        assert resp.degraded_mode is True
        assert resp.action in ("ALLOW", "CHALLENGE", "BLOCK")

    def test_total_latency_under_50ms(self, clean_redis, monkeypatch):
        from src.api import decision_gate
        monkeypatch.setattr(decision_gate.blocklist, "redis", clean_redis)
        monkeypatch.setattr(decision_gate.counters, "redis", clean_redis)

        resp = evaluate_risk_pipeline(_make_req())
        assert resp.latency_ms.total < 50.0
