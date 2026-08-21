"""Comprehensive test suite for input validation and security.

Tests that the API properly rejects malicious or malformed inputs
including injection attempts, boundary values, and type violations.
"""

import pytest
from src.api.schemas import ScoringRequest


# ---------------------------------------------------------------------------
# Valid baseline request (reusable)
# ---------------------------------------------------------------------------
def _valid_req(**overrides) -> dict:
    base = {
        "transaction_id": "txn_001",
        "user_id": "usr_42",
        "device_hash": "dev_abc",
        "instrument_id": "instr_xyz",
        "ip": "203.0.113.4",
        "amount": 100.0,
        "timestamp": "2026-08-21T10:15:00Z",
        "merchant_category_id": "5732",
    }
    base.update(overrides)
    return base


class TestInputValidation:
    """Test that Pydantic validators reject bad input."""

    def test_valid_request_passes(self):
        req = ScoringRequest(**_valid_req())
        assert req.transaction_id == "txn_001"

    # --- Amount validation ---
    def test_negative_amount_rejected(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(amount=-100.0))

    def test_zero_amount_rejected(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(amount=0.0))

    def test_absurdly_large_amount_rejected(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(amount=999_999_999.0))

    # --- Empty / missing fields ---
    def test_empty_user_id_rejected(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(user_id=""))

    def test_empty_transaction_id_rejected(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(transaction_id=""))

    # --- Injection attacks ---
    def test_sql_injection_in_user_id(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(user_id="'; DROP TABLE users;--"))

    def test_redis_key_injection_in_device_hash(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(device_hash="dev\x00:users"))

    def test_newline_injection(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(user_id="usr_1\nX-Forwarded-For: evil"))

    def test_unicode_smuggling(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(user_id="usr_\u200b42"))  # zero-width space

    # --- IP validation ---
    def test_invalid_ip_rejected(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(ip="not-an-ip"))

    def test_ip_with_port_rejected(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(ip="192.168.1.1:8080"))

    def test_valid_ipv6_accepted(self):
        req = ScoringRequest(**_valid_req(ip="::1"))
        assert req.ip == "::1"

    # --- Timestamp validation ---
    def test_invalid_timestamp_rejected(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(timestamp="not-a-date"))

    def test_valid_timestamp_with_tz(self):
        req = ScoringRequest(**_valid_req(timestamp="2026-08-21T10:15:00+05:30"))
        assert req.timestamp == "2026-08-21T10:15:00+05:30"

    # --- Oversized strings ---
    def test_extremely_long_id_rejected(self):
        with pytest.raises(Exception):
            ScoringRequest(**_valid_req(user_id="x" * 500))
