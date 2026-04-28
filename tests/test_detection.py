"""Tests for detection.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from detection import (
    detect_marked_choice_with_vision,
    _assign_column,
    _cluster_to_5_columns,
)


def test_vision_stub_raises():
    with pytest.raises(NotImplementedError):
        detect_marked_choice_with_vision("any_path.png")


def test_assign_column_first_band():
    cols = _cluster_to_5_columns([100, 300, 500, 700, 900], 1000)
    assert cols is not None
    # x=50 should land in band A (0–200)
    letter, conf = _assign_column(50, cols)
    assert letter == "A"
    assert 0.0 <= conf <= 1.0


def test_assign_column_last_band():
    cols = _cluster_to_5_columns([100, 300, 500, 700, 900], 1000)
    letter, conf = _assign_column(950, cols)
    assert letter == "E"


def test_assign_column_out_of_range():
    cols = _cluster_to_5_columns([100], 1000)
    letter, conf = _assign_column(9999, cols)
    assert letter is None
    assert conf == 0.0


def test_cluster_returns_five_entries():
    cols = _cluster_to_5_columns([50, 250, 450, 650, 850], 1000)
    assert cols is not None
    assert len(cols) == 5
    letters = [c[0] for c in cols]
    assert letters == ["A", "B", "C", "D", "E"]
