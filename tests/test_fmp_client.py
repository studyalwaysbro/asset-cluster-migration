"""Tests for FMP client (unit tests, no API calls)."""
import pytest
from src.data.fmp_client import BandwidthTracker


def test_bandwidth_tracker():
    bt = BandwidthTracker(max_gb=1.0)
    bt.record(500_000_000)
    assert bt.used_gb == pytest.approx(0.5, abs=0.01)


def test_bandwidth_limit():
    bt = BandwidthTracker(max_gb=0.001)
    with pytest.raises(RuntimeError):
        bt.record(2_000_000)
