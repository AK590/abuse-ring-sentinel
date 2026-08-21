"""Tests for label-lag cutoff and chronological split integrity.

Ensures that:
1. The 60-day label-lag cutoff properly excludes recent data.
2. The chronological split has no temporal leakage.
3. Train set max timestamp <= test set min timestamp.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.evaluation.label_lag import apply_label_lag, chronological_split


@pytest.fixture
def sample_df():
    """Create a small DataFrame spanning 120 days."""
    np.random.seed(42)
    n = 500
    start = datetime(2026, 1, 1)
    timestamps = [start + timedelta(days=i * 120 / n) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "amount": np.random.uniform(10, 1000, n),
        "label": np.random.choice([0, 1], n, p=[0.95, 0.05]),
    })


class TestLabelLag:
    def test_cutoff_removes_recent_rows(self, sample_df):
        result = apply_label_lag(sample_df, exclude_last_n_days=60)
        max_allowed = sample_df["timestamp"].max() - timedelta(days=60)
        assert result["timestamp"].max() <= max_allowed
        assert len(result) < len(sample_df)

    def test_cutoff_zero_days_keeps_all(self, sample_df):
        result = apply_label_lag(sample_df, exclude_last_n_days=0)
        assert len(result) == len(sample_df)

    def test_cutoff_returns_copy(self, sample_df):
        result = apply_label_lag(sample_df, exclude_last_n_days=30)
        # Modifying result should not affect the original
        result["amount"] = 0
        assert sample_df["amount"].sum() > 0


class TestChronologicalSplit:
    def test_no_temporal_leakage(self, sample_df):
        train, test = chronological_split(sample_df, train_frac=0.8)
        assert train["timestamp"].max() <= test["timestamp"].min()

    def test_split_sizes(self, sample_df):
        train, test = chronological_split(sample_df, train_frac=0.8)
        expected_train = int(len(sample_df) * 0.8)
        assert len(train) == expected_train
        assert len(test) == len(sample_df) - expected_train

    def test_all_rows_preserved(self, sample_df):
        train, test = chronological_split(sample_df, train_frac=0.8)
        assert len(train) + len(test) == len(sample_df)

    def test_combined_with_label_lag(self, sample_df):
        """Integration: label lag + chrono split."""
        lagged = apply_label_lag(sample_df, exclude_last_n_days=30)
        train, test = chronological_split(lagged, train_frac=0.8)
        assert train["timestamp"].max() <= test["timestamp"].min()
        assert len(train) + len(test) == len(lagged)
