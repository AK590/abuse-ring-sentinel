"""Tests for the L0 Redis blocklist.

Covers: basic add/check, entity type validation, key injection prevention,
remove functionality, and all entity types.
"""

import pytest
from src.rules.blocklist import Blocklist, get_redis_client


@pytest.fixture
def redis_client():
    client = get_redis_client()
    client.flushall()
    yield client
    client.flushall()


class TestBlocklistBasic:
    def test_blocklisted_user_is_blocked(self, redis_client):
        bl = Blocklist(redis_client)
        bl.add("user", "bad_user_1")
        assert bl.check_transaction("bad_user_1", "dev_1", "inst_1", "1.2.3.4") is True

    def test_blocklisted_ip_is_blocked(self, redis_client):
        bl = Blocklist(redis_client)
        bl.add("ip", "10.0.0.1")
        assert bl.check_transaction("good_user", "dev_2", "inst_2", "10.0.0.1") is True

    def test_blocklisted_device_is_blocked(self, redis_client):
        bl = Blocklist(redis_client)
        bl.add("device", "evil_device")
        assert bl.check_transaction("good_user", "evil_device", "inst_3", "1.2.3.4") is True

    def test_blocklisted_instrument_is_blocked(self, redis_client):
        bl = Blocklist(redis_client)
        bl.add("instrument", "stolen_card")
        assert bl.check_transaction("good_user", "dev_3", "stolen_card", "1.2.3.4") is True

    def test_clean_transaction_passes(self, redis_client):
        bl = Blocklist(redis_client)
        assert bl.check_transaction("good_user", "dev_3", "inst_3", "1.2.3.4") is False


class TestBlocklistRemove:
    def test_remove_unblocklists(self, redis_client):
        bl = Blocklist(redis_client)
        bl.add("user", "temp_bad")
        assert bl.is_blocklisted("user", "temp_bad") is True
        bl.remove("user", "temp_bad")
        assert bl.is_blocklisted("user", "temp_bad") is False


class TestBlocklistSecurity:
    def test_invalid_entity_type_rejected(self, redis_client):
        bl = Blocklist(redis_client)
        with pytest.raises(ValueError, match="Invalid entity type"):
            bl.add("admin", "evil")

    def test_key_injection_rejected(self, redis_client):
        bl = Blocklist(redis_client)
        with pytest.raises(ValueError, match="Invalid key component"):
            bl.add("user", "evil\nkey")

    def test_empty_value_rejected(self, redis_client):
        bl = Blocklist(redis_client)
        with pytest.raises(ValueError):
            bl.add("user", "")
