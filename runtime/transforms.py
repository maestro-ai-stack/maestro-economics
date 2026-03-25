"""Panel-aware data transformations for economic research."""

from __future__ import annotations

from typing import Any

import pandas as pd


def winsorize(df: pd.DataFrame, col: str, pct: float = 0.01) -> pd.DataFrame:
    """Clip column values at symmetric quantiles.

    Args:
        df: Input DataFrame (not mutated).
        col: Column name to winsorize.
        pct: Fraction to clip from each tail (e.g. 0.05 = 5th and 95th pctile).

    Returns:
        New DataFrame with clipped values.
    """
    result = df.copy()
    lo = result[col].quantile(pct)
    hi = result[col].quantile(1 - pct)
    result[col] = result[col].clip(lower=lo, upper=hi)
    return result


def lag(
    df: pd.DataFrame,
    col: str,
    periods: int = 1,
    entity: str | None = None,
    time: str | None = None,
) -> pd.DataFrame:
    """Create lagged variable within panel groups.

    Args:
        df: Input DataFrame (not mutated).
        col: Column to lag.
        periods: Number of periods to lag.
        entity: Entity identifier column for groupby.
        time: Time column (used for sorting).

    Returns:
        DataFrame with new column ``{col}_lag{periods}``.
    """
    result = df.copy()
    if time is not None:
        result = result.sort_values([entity, time]) if entity else result.sort_values(time)
    new_col = f"{col}_lag{periods}"
    if entity:
        result[new_col] = result.groupby(entity)[col].shift(periods)
    else:
        result[new_col] = result[col].shift(periods)
    return result


def lead(
    df: pd.DataFrame,
    col: str,
    periods: int = 1,
    entity: str | None = None,
    time: str | None = None,
) -> pd.DataFrame:
    """Create lead (forward-shifted) variable within panel groups.

    Args:
        df: Input DataFrame (not mutated).
        col: Column to lead.
        periods: Number of periods to lead.
        entity: Entity identifier column for groupby.
        time: Time column (used for sorting).

    Returns:
        DataFrame with new column ``{col}_lead{periods}``.
    """
    result = df.copy()
    if time is not None:
        result = result.sort_values([entity, time]) if entity else result.sort_values(time)
    new_col = f"{col}_lead{periods}"
    if entity:
        result[new_col] = result.groupby(entity)[col].shift(-periods)
    else:
        result[new_col] = result[col].shift(-periods)
    return result


def diff(
    df: pd.DataFrame,
    col: str,
    periods: int = 1,
    entity: str | None = None,
    time: str | None = None,
) -> pd.DataFrame:
    """First difference within panel groups.

    Args:
        df: Input DataFrame (not mutated).
        col: Column to difference.
        periods: Number of periods for differencing.
        entity: Entity identifier column for groupby.
        time: Time column (used for sorting).

    Returns:
        DataFrame with new column ``{col}_diff{periods}``.
    """
    result = df.copy()
    if time is not None:
        result = result.sort_values([entity, time]) if entity else result.sort_values(time)
    new_col = f"{col}_diff{periods}"
    if entity:
        result[new_col] = result.groupby(entity)[col].diff(periods)
    else:
        result[new_col] = result[col].diff(periods)
    return result


def balance_panel(
    df: pd.DataFrame,
    entity: str,
    time: str,
) -> pd.DataFrame:
    """Keep only entities observed in all time periods.

    Args:
        df: Input DataFrame (not mutated).
        entity: Entity identifier column.
        time: Time column.

    Returns:
        Filtered DataFrame with only complete entities.
    """
    all_times = df[time].unique()
    max_periods = len(all_times)
    counts = df.groupby(entity)[time].nunique()
    complete = counts[counts == max_periods].index
    return df[df[entity].isin(complete)].copy()


def dummy(
    df: pd.DataFrame,
    col: str,
    drop_first: bool = False,
) -> pd.DataFrame:
    """Create dummy (indicator) variables from a categorical column.

    Args:
        df: Input DataFrame (not mutated).
        col: Column to create dummies from.
        drop_first: Whether to drop the first category (for avoiding collinearity).

    Returns:
        DataFrame with dummy columns added (original column removed).
    """
    return pd.get_dummies(df, columns=[col], prefix=col, drop_first=drop_first)


def standardize(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Standardize columns to zero mean and unit variance (population).

    Args:
        df: Input DataFrame (not mutated).
        cols: Columns to standardize.

    Returns:
        DataFrame with standardized columns.
    """
    result = df.copy()
    for col in cols:
        mean = result[col].mean()
        std = result[col].std(ddof=0)
        if std == 0:
            result[col] = 0.0
        else:
            result[col] = (result[col] - mean) / std
    return result


def merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    how: str = "inner",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge two DataFrames with merge diagnostics.

    Args:
        left: Left DataFrame.
        right: Right DataFrame.
        on: Column(s) to join on.
        how: Join type ('inner', 'left', 'right', 'outer').

    Returns:
        Tuple of (merged DataFrame, diagnostics dict).
        Diagnostics include: matched, left_only, right_only counts.
    """
    result = left.merge(right, on=on, how="outer", indicator=True)

    matched = int((result["_merge"] == "both").sum())
    left_only = int((result["_merge"] == "left_only").sum())
    right_only = int((result["_merge"] == "right_only").sum())

    diagnostics = {
        "matched": matched,
        "left_only": left_only,
        "right_only": right_only,
    }

    # Apply the requested join type
    if how == "inner":
        result = result[result["_merge"] == "both"]
    elif how == "left":
        result = result[result["_merge"].isin(["both", "left_only"])]
    elif how == "right":
        result = result[result["_merge"].isin(["both", "right_only"])]
    # 'outer' keeps everything

    result = result.drop(columns=["_merge"])
    return result, diagnostics


def recode(
    df: pd.DataFrame,
    col: str,
    mapping: dict,
) -> pd.DataFrame:
    """Remap values in a column. Unmapped values are left unchanged.

    Args:
        df: Input DataFrame (not mutated).
        col: Column to recode.
        mapping: Dict mapping old values to new values.

    Returns:
        DataFrame with recoded column.
    """
    result = df.copy()
    result[col] = result[col].map(lambda v: mapping.get(v, v))
    return result
