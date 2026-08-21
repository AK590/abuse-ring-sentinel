"""L0 deterministic blocklist backed by Redis.

Design: blocklist checks are the fastest possible path (<1ms).
If Redis is unreachable, callers should catch the exception and
treat it as 'not blocked' (fail-open at L0, the model still runs).
"""

import os
import re
import logging

import redis as _redis_lib

logger = logging.getLogger('sentinel.blocklist')

_SAFE_KEY_RE = re.compile(r'^[a-zA-Z0-9_\-.:]+$')


def get_redis_client():
    """Return a Redis client based on REDIS_URL env var."""
    redis_url = os.environ.get('REDIS_URL')
    if redis_url and redis_url != 'fakeredis':
        return _redis_lib.from_url(redis_url, decode_responses=True)
    # Fallback to fakeredis for hackathon / CI
    import fakeredis
    return fakeredis.FakeRedis(decode_responses=True)


def _validate_key_part(value: str) -> str:
    """Reject values that could cause Redis key injection."""
    if not value or not _SAFE_KEY_RE.match(value):
        raise ValueError(f'Invalid key component: {value!r}')
    return value


class Blocklist:
    PREFIX = 'blocklist:'
    VALID_ENTITY_TYPES = frozenset({'user', 'device', 'instrument', 'ip'})

    def __init__(self, redis_client=None):
        self.redis = redis_client or get_redis_client()

    def _key(self, entity_type: str, entity_value: str) -> str:
        if entity_type not in self.VALID_ENTITY_TYPES:
            raise ValueError(f'Invalid entity type: {entity_type!r}')
        _validate_key_part(entity_value)
        return f'{self.PREFIX}{entity_type}:{entity_value}'

    def add(self, entity_type: str, entity_value: str) -> None:
        self.redis.set(self._key(entity_type, entity_value), '1')

    def remove(self, entity_type: str, entity_value: str) -> None:
        self.redis.delete(self._key(entity_type, entity_value))

    def is_blocklisted(self, entity_type: str, entity_value: str) -> bool:
        return self.redis.exists(self._key(entity_type, entity_value)) > 0

    def check_transaction(
        self, user_id: str, device_hash: str, instrument_id: str, ip: str,
    ) -> bool:
        """Return True if ANY entity is blocklisted."""
        return (
            self.is_blocklisted('user', user_id)
            or self.is_blocklisted('device', device_hash)
            or self.is_blocklisted('instrument', instrument_id)
            or self.is_blocklisted('ip', ip)
        )
