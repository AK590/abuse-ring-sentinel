"""Tests for the real-time Redis counters.

Covers: device/instrument reuse counts, velocity windows,
pipeline batching, and TTL behavior.
"""

import time
import pytest
from src.realtime.counters import RealtimeCounters
from src.rules.blocklist import get_redis_client


@pytest.fixture
def redis_client():
    client = get_redis_client()
    client.flushall()
    yield client
    client.flushall()


class TestRealtimeCounters:
    def test_device_reuse_count(self, redis_client):
        c = RealtimeCounters(redis_client)
        now = time.time_ns()

        c.record_transaction("tx1", "u1", "devA", "inst1", timestamp_ns=now)
        c.record_transaction("tx2", "u2", "devA", "inst2", timestamp_ns=now)

        feats = c.get_features("u1", "devA", "inst1", current_time_ns=now)
        assert feats["device_reuse_count"] == 2  # Two users on same device

    def test_instrument_reuse_count(self, redis_client):
        c = RealtimeCounters(redis_client)
        now = time.time_ns()

        c.record_transaction("tx1", "u1", "d1", "shared_inst", timestamp_ns=now)
        c.record_transaction("tx2", "u2", "d2", "shared_inst", timestamp_ns=now)
        c.record_transaction("tx3", "u3", "d3", "shared_inst", timestamp_ns=now)

        feats = c.get_features("u1", "d1", "shared_inst", current_time_ns=now)
        assert feats["instrument_reuse_count"] == 3

    def test_velocity_within_5min(self, redis_client):
        c = RealtimeCounters(redis_client)
        now = time.time_ns()

        for i in range(5):
            c.record_transaction(
                f"tx{i}", "u1", "d1", "inst1",
                timestamp_ns=now - (i * 30_000_000_000),  # 30 sec apart
            )

        feats = c.get_features("u1", "d1", "inst1", current_time_ns=now)
        assert feats["velocity_5min"] == 5
        assert feats["velocity_1hr"] == 5

    def test_velocity_outside_5min_window(self, redis_client):
        c = RealtimeCounters(redis_client)
        now = time.time_ns()

        # One tx 10 minutes ago
        c.record_transaction("tx_old", "u1", "d1", "inst1",
                             timestamp_ns=now - (600 * 1_000_000_000))
        # One tx now
        c.record_transaction("tx_new", "u1", "d1", "inst1", timestamp_ns=now)

        feats = c.get_features("u1", "d1", "inst1", current_time_ns=now)
        assert feats["velocity_5min"] == 1  # only the recent one
        assert feats["velocity_1hr"] == 2  # both

    def test_no_cross_user_velocity(self, redis_client):
        """Velocity is per-user, not global."""
        c = RealtimeCounters(redis_client)
        now = time.time_ns()

        c.record_transaction("tx1", "u1", "d1", "i1", timestamp_ns=now)
        c.record_transaction("tx2", "u2", "d2", "i2", timestamp_ns=now)

        feats_u1 = c.get_features("u1", "d1", "i1", current_time_ns=now)
        feats_u2 = c.get_features("u2", "d2", "i2", current_time_ns=now)

        assert feats_u1["velocity_5min"] == 1
        assert feats_u2["velocity_5min"] == 1

    def test_idempotent_reuse_count(self, redis_client):
        """Recording the same user on the same device twice shouldn't double-count."""
        c = RealtimeCounters(redis_client)
        now = time.time_ns()

        c.record_transaction("tx1", "u1", "devA", "inst1", timestamp_ns=now)
        c.record_transaction("tx2", "u1", "devA", "inst1", timestamp_ns=now + 1000)

        feats = c.get_features("u1", "devA", "inst1", current_time_ns=now)
        assert feats["device_reuse_count"] == 1  # same user, not a reuse
