"""Label-lag utilities.

The core idea: chargebacks take 45-180 days to materialize, so the
most recent ~60 days of labels are unreliable.  Any transaction in
that window labelled 'legit' might just be a chargeback that hasn't
arrived yet.

This module provides functions to:
1. Apply the label-lag cutoff to a training DataFrame.
2. Verify that a chronological split does not leak future data.
"""

import logging
from datetime import timedelta
import pandas as pd

logger = logging.getLogger('sentinel.label_lag')


def apply_label_lag(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    exclude_last_n_days: int = 60,
) -> pd.DataFrame:
    """Drop rows whose labels are within the unreliable window.

    Returns a copy — does not mutate the input.
    """
    max_date = df[timestamp_col].max()
    cutoff = max_date - timedelta(days=exclude_last_n_days)
    result = df[df[timestamp_col] <= cutoff].copy()
    logger.info(
        'Label-lag cutoff: dropped %d rows after %s (max=%s, lag=%d days)',
        len(df) - len(result),
        cutoff,
        max_date,
        exclude_last_n_days,
    )
    return result


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    timestamp_col: str = 'timestamp',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically (no shuffle). Returns (train, test)."""
    df_sorted = df.sort_values(timestamp_col)
    split_idx = int(len(df_sorted) * train_frac)
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]
    # Sanity check: no temporal leakage
    assert train[timestamp_col].max() <= test[timestamp_col].min(), \
        'Temporal leakage detected: train max > test min'
    logger.info(
        'Chrono split: train %s→%s (%d), test %s→%s (%d)',
        train[timestamp_col].min(), train[timestamp_col].max(), len(train),
        test[timestamp_col].min(), test[timestamp_col].max(), len(test),
    )
    return train, test
