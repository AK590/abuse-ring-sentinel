"""Real-time Redis counters for device reuse, instrument reuse, and velocity.

All operations are O(1) or O(log N) with bounded N (TTL eviction).
"""

import time
import logging
from src.rules.blocklist import get_redis_client

logger = logging.getLogger('sentinel.counters')

# Sliding window sizes in seconds
_WINDOW_5MIN = 300
_WINDOW_1HR = 3600
# Keep velocity sorted sets for at most 2 hours (generous buffer)
_VELOCITY_TTL_SEC = 2 * _WINDOW_1HR
# Reuse sets TTL — 24 hours (matches a reasonable batch window)
_REUSE_TTL_SEC = 24 * 3600


class RealtimeCounters:
    def __init__(self, redis_client=None):
        self.redis = redis_client or get_redis_client()

    def record_transaction(
        self,
        tx_id: str,
        user_id: str,
        device_hash: str,
        instrument_id: str,
        timestamp_ns: int | None = None,
    ) -> None:
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        pipe = self.redis.pipeline(transaction=False)

        # Device reuse
        dk = f'device:{device_hash}:users'
        pipe.sadd(dk, user_id)
        pipe.expire(dk, _REUSE_TTL_SEC)

        # Instrument reuse
        ik = f'instrument:{instrument_id}:users'
        pipe.sadd(ik, user_id)
        pipe.expire(ik, _REUSE_TTL_SEC)

        # Velocity (sorted set with timestamp_ns as score)
        vk = f'user:{user_id}:txs'
        pipe.zadd(vk, {tx_id: timestamp_ns})
        # Trim entries older than the largest window to prevent unbounded growth
        cutoff = timestamp_ns - (_VELOCITY_TTL_SEC * 1_000_000_000)
        pipe.zremrangebyscore(vk, 0, cutoff)
        pipe.expire(vk, _VELOCITY_TTL_SEC)

        pipe.execute()

    def get_features(
        self,
        user_id: str,
        device_hash: str,
        instrument_id: str,
        current_time_ns: int | None = None,
    ) -> dict:
        if current_time_ns is None:
            current_time_ns = time.time_ns()

        pipe = self.redis.pipeline(transaction=False)
        pipe.scard(f'device:{device_hash}:users')
        pipe.scard(f'instrument:{instrument_id}:users')

        min_5m = current_time_ns - (_WINDOW_5MIN * 1_000_000_000)
        min_1h = current_time_ns - (_WINDOW_1HR * 1_000_000_000)
        pipe.zcount(f'user:{user_id}:txs', min_5m, current_time_ns)
        pipe.zcount(f'user:{user_id}:txs', min_1h, current_time_ns)

        device_reuse, instrument_reuse, vel_5, vel_1h = pipe.execute()

        return {
            'device_reuse_count': device_reuse,
            'instrument_reuse_count': instrument_reuse,
            'velocity_5min': vel_5,
            'velocity_1hr': vel_1h,
        }
