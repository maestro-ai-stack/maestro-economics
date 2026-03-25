"""Tests for runtime.transforms — panel-aware data transformations."""

import numpy as np
import pandas as pd
import pytest

from runtime.transforms import (
    winsorize,
    lag,
    lead,
    diff,
    balance_panel,
    dummy,
    standardize,
    merge,
    recode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Simple 2-entity x 4-period balanced panel."""
    return pd.DataFrame({
        "id": ["A"] * 4 + ["B"] * 4,
        "year": [2000, 2001, 2002, 2003] * 2,
        "x": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "y": [100, 200, 300, 400, 500, 600, 700, 800],
    })


@pytest.fixture
def unbalanced_df() -> pd.DataFrame:
    """Entity C is missing year 2002."""
    return pd.DataFrame({
        "id": ["A"] * 3 + ["C"] * 2,
        "year": [2000, 2001, 2002, 2000, 2001],
        "x": [1.0, 2.0, 3.0, 7.0, 8.0],
    })


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------

class TestWinsorize:
    def test_clips_at_quantiles(self):
        df = pd.DataFrame({"v": list(range(1, 101))})
        result = winsorize(df, "v", pct=0.05)
        assert result["v"].min() >= 5  # clipped at 5th percentile
        assert result["v"].max() <= 96  # clipped at 95th percentile

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"v": [1, 50, 100]})
        _ = winsorize(df, "v", pct=0.1)
        assert df["v"].tolist() == [1, 50, 100]


# ---------------------------------------------------------------------------
# lag / lead / diff
# ---------------------------------------------------------------------------

class TestLag:
    def test_one_period_lag(self, panel_df):
        result = lag(panel_df, "x", periods=1, entity="id", time="year")
        # First obs of each entity should be NaN
        a_vals = result.loc[result["id"] == "A", "x_lag1"].tolist()
        assert np.isnan(a_vals[0])
        assert a_vals[1] == 1.0
        assert a_vals[2] == 2.0

    def test_two_period_lag(self, panel_df):
        result = lag(panel_df, "x", periods=2, entity="id", time="year")
        a_vals = result.loc[result["id"] == "A", "x_lag2"].tolist()
        assert np.isnan(a_vals[0])
        assert np.isnan(a_vals[1])
        assert a_vals[2] == 1.0


class TestLead:
    def test_one_period_lead(self, panel_df):
        result = lead(panel_df, "x", periods=1, entity="id", time="year")
        a_vals = result.loc[result["id"] == "A", "x_lead1"].tolist()
        assert a_vals[0] == 2.0
        assert a_vals[1] == 3.0
        assert np.isnan(a_vals[3])  # last obs


class TestDiff:
    def test_first_difference(self, panel_df):
        result = diff(panel_df, "x", periods=1, entity="id", time="year")
        a_vals = result.loc[result["id"] == "A", "x_diff1"].tolist()
        assert np.isnan(a_vals[0])
        assert a_vals[1] == 1.0  # 2 - 1
        assert a_vals[2] == 1.0  # 3 - 2

    def test_does_not_cross_entities(self, panel_df):
        result = diff(panel_df, "x", periods=1, entity="id", time="year")
        b_first = result.loc[result["id"] == "B", "x_diff1"].iloc[0]
        assert np.isnan(b_first)


# ---------------------------------------------------------------------------
# balance_panel
# ---------------------------------------------------------------------------

class TestBalancePanel:
    def test_drops_incomplete_entities(self, unbalanced_df):
        result = balance_panel(unbalanced_df, entity="id", time="year")
        assert set(result["id"].unique()) == {"A"}
        assert len(result) == 3

    def test_already_balanced_unchanged(self, panel_df):
        result = balance_panel(panel_df, entity="id", time="year")
        assert len(result) == len(panel_df)


# ---------------------------------------------------------------------------
# dummy
# ---------------------------------------------------------------------------

class TestDummy:
    def test_creates_dummies(self):
        df = pd.DataFrame({"region": ["east", "west", "east", "north"]})
        result = dummy(df, "region")
        assert "region_east" in result.columns
        assert "region_west" in result.columns
        assert "region_north" in result.columns

    def test_drop_first(self):
        df = pd.DataFrame({"region": ["east", "west", "east", "north"]})
        result = dummy(df, "region", drop_first=True)
        # One fewer dummy column than unique values
        dummy_cols = [c for c in result.columns if c.startswith("region_")]
        assert len(dummy_cols) == 2


# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------

class TestStandardize:
    def test_zero_mean_unit_var(self):
        df = pd.DataFrame({
            "a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = standardize(df, ["a", "b"])
        assert abs(result["a"].mean()) < 1e-10
        assert abs(result["b"].std(ddof=0) - 1.0) < 0.1  # population std ~1

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        original = df["a"].tolist()
        _ = standardize(df, ["a"])
        assert df["a"].tolist() == original


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_inner_merge_with_diagnostics(self):
        left = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        right = pd.DataFrame({"id": [2, 3, 4], "score": [80, 90, 70]})
        result, diag = merge(left, right, on="id", how="inner")
        assert len(result) == 2
        assert "left_only" in diag
        assert "right_only" in diag
        assert "matched" in diag
        assert diag["matched"] == 2
        assert diag["left_only"] == 1
        assert diag["right_only"] == 1

    def test_left_merge(self):
        left = pd.DataFrame({"id": [1, 2], "v": [10, 20]})
        right = pd.DataFrame({"id": [2, 3], "w": [30, 40]})
        result, diag = merge(left, right, on="id", how="left")
        assert len(result) == 2
        assert diag["matched"] == 1


# ---------------------------------------------------------------------------
# recode
# ---------------------------------------------------------------------------

class TestRecode:
    def test_remap_values(self):
        df = pd.DataFrame({"status": [1, 2, 3, 1]})
        result = recode(df, "status", {1: "low", 2: "mid", 3: "high"})
        assert result["status"].tolist() == ["low", "mid", "high", "low"]

    def test_unmapped_values_unchanged(self):
        df = pd.DataFrame({"status": [1, 2, 99]})
        result = recode(df, "status", {1: "low", 2: "mid"})
        assert result["status"].iloc[2] == 99
