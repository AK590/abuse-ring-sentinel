"""Tests for chronological split integrity.

Verifies no temporal leakage in the training pipeline.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.evaluation.label_lag import chronological_split


@pytest.fixture
def ordered_df():
    """DataFrame with perfectly ordered timestamps."""
    n = 1000
    start = datetime(2025, 1, 1)
    return pd.DataFrame({
        "timestamp": [start + timedelta(hours=i) for i in range(n)],
        "value": range(n),
        "label": [0] * 900 + [1] * 100,
    })


class TestChronologicalSplit:
    def test_train_before_test(self, ordered_df):
        train, test = chronological_split(ordered_df, train_frac=0.7)
        assert train["timestamp"].max() <= test["timestamp"].min()

    def test_no_shuffling(self, ordered_df):
        train, test = chronological_split(ordered_df, train_frac=0.8)
        # Values should be monotonically increasing within each split
        assert train["value"].is_monotonic_increasing
        assert test["value"].is_monotonic_increasing

    def test_extreme_split_ratios(self, ordered_df):
        train, test = chronological_split(ordered_df, train_frac=0.99)
        assert len(test) >= 1
        assert train["timestamp"].max() <= test["timestamp"].min()

    def test_50_50_split(self, ordered_df):
        train, test = chronological_split(ordered_df, train_frac=0.5)
        assert abs(len(train) - len(test)) <= 1
        assert train["timestamp"].max() <= test["timestamp"].min()
