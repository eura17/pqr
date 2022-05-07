"""Implemented classic factor strategies and pretransforms for them."""

from __future__ import annotations

__all__ = [
    "filter",
    "look_back",
    "lag",
    "hold",
    "quantiles",
    "top",
    "bottom",
    "thresholds",
]

from typing import Literal, Callable

import numpy as np
import pandas as pd

from pqr.utils import align


def filter(
        factor: pd.DataFrame,
        *,
        universe: pd.DataFrame,
) -> pd.DataFrame:
    """Filters `factor` values based on given `universe`.

    Actually, replaces factor values with nans, where universe is equal to False.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    universe : pd.DataFrame
        Matrix with True/False, indicating to include factor values or not.

    Returns
    -------
    pd.DataFrame
        Filtered `factor` values.
    """

    universe, factor = align(universe, factor)
    return pd.DataFrame(
        np.where(np.asarray(universe, dtype=bool), np.asarray(factor, bool), np.nan),
        index=factor.index.copy(),
        columns=factor.columns.copy(),
    )


def look_back(
        factor: pd.DataFrame,
        *,
        period: int,
        agg: Literal["pct", "mean", "median", "min", "max"] | Callable[[pd.Series], float],
) -> pd.DataFrame:
    """Aggregates `factor` values column-wise each `period` by `agg`.

    If agg is not predefined can work very slow. In this case the best decision is to write your
    own effective realisation of transformation function.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    period : int
        Period to look back on the data.
    agg : {"pct", "mean", "median", "min", "max"} or callable
        Aggregation func to apply on `factor` values. If callable is given function must be
        appliable on pd.Series and return float.

    Returns
    -------
    pd.DataFrame
        Aggregated `factor` values.
    """

    if agg == "pct":
        factor_array = np.asarray(factor)
        abs_change = (factor_array[period:] - factor_array[:-period])
        base = factor_array[:-period]
        return pd.DataFrame(
            abs_change / base,
            index=factor.index[period:].copy(),
            columns=factor.columns.copy()
        )
    elif agg == "mean":
        return factor.rolling(period, axis=0).mean().iloc[period:]
    elif agg == "median":
        return factor.rolling(period, axis=0).median().iloc[period:]
    elif agg == "min":
        return factor.rolling(period, axis=0).min().iloc[period:]
    elif agg == "max":
        return factor.rolling(period, axis=0).max().iloc[period:]
    else:
        return factor.rolling(period, axis=0).apply(agg).iloc[period:]


def lag(
        factor: pd.DataFrame,
        *,
        period: int,
) -> pd.DataFrame:
    """Lags `factor` values for `period`.

    Can be used both forward and backward.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    period : int
        Period to look back on the data.

    Returns
    -------
    pd.DataFrame
        Lagged `factor` values.
    """

    if period == 0:
        return factor
    elif period > 0:
        return pd.DataFrame(
            factor.to_numpy()[:-period],
            index=factor.index[period:].copy(),
            columns=factor.columns.copy(),
        )
    else:  # period < 0
        return pd.DataFrame(
            factor.to_numpy()[-period:],
            index=factor.index[:period].copy(),
            columns=factor.columns.copy(),
        )


def hold(
        factor: pd.DataFrame,
        *,
        period: int,
) -> pd.DataFrame:
    """Spread `factor` values for `period`.

    Can be used to react on new information every `period` timestamps.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    period : int
        Period to look back on the data.

    Returns
    -------
    pd.DataFrame
        Rolling mean of `factor` values.
    """

    periods = np.zeros(len(factor), dtype=int)
    update_periods = np.arange(len(factor), step=period)
    periods[update_periods] = update_periods
    update_mask = np.maximum.accumulate(periods[:, np.newaxis], axis=0)
    return pd.DataFrame(
        np.take_along_axis(np.asarray(factor, dtype=float), update_mask, axis=0),
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )


def quantiles(
        factor: pd.DataFrame,
        *,
        min_q: float,
        max_q: float,
) -> pd.DataFrame:
    """Picks when `factor` values are between `min_q` and `max_q` quantiles in a period.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    min_q : float
        Quantile to estimate lower boarder of `factor` values to pick.
    max_q : float
        Quantile to estimate upper boarder of `factor` values to pick.

    Returns
    -------
    pd.DataFrame
        Matrix of True/False, indicating whether factor values are between quantile boarders or not.
    """

    factor_array = np.asarray(factor)
    lower, upper = np.nanquantile(factor_array, [min_q, max_q], axis=1, keepdims=True)
    return pd.DataFrame(
        (lower <= factor_array) & (factor_array <= upper),
        index=factor.index.copy(),
        columns=factor.columns.copy(),
    )


def top(
        factor: pd.DataFrame,
        *,
        k: int,
) -> pd.DataFrame:
    """Picks when `factor` values are at the top `k` in a period.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    k : int
        Place to estimate lower boarder of factor values to pick.

    Returns
    -------
    pd.DataFrame
        Matrix of True/False, indicating whether factor values are in the top or not.
    """

    def _top_k_row(arr: np.ndarray) -> float:
        uniq_arr = np.unique(arr[~np.isnan(arr)])
        max_k = len(uniq_arr)

        if max_k > k:
            return np.sort(uniq_arr)[-k]
        elif max_k > 0:
            return np.max(uniq_arr)
        else:
            return np.nan

    factor_array = np.asarray(factor)
    lower = np.apply_along_axis(_top_k_row, axis=1, arr=factor_array)[:, np.newaxis]
    return pd.DataFrame(
        factor_array >= lower,
        index=factor.index.copy(),
        columns=factor.columns.copy(),
    )


def bottom(
        factor: pd.DataFrame,
        *,
        k: int,
) -> pd.DataFrame:
    """Picks when `factor` values are in the bottom `k` in a period.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    k : int
        Place to estimate upper boarder of factor values to pick.

    Returns
    -------
    pd.DataFrame
        Matrix of True/False, indicating whether factor values are in the bottom or not.
    """

    def _bottom_k_row(arr: np.ndarray) -> float:
        uniq_arr = np.unique(arr[~np.isnan(arr)])
        max_k = len(uniq_arr)

        if max_k > k:
            return np.sort(uniq_arr)[k - 1]
        elif max_k > 0:
            return np.min(uniq_arr)
        else:
            return np.nan

    factor_array = np.asarray(factor)
    upper = np.apply_along_axis(_bottom_k_row, axis=1, arr=factor_array)[:, np.newaxis]
    return pd.DataFrame(
        factor_array <= upper,
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )


def thresholds(
        factor: pd.DataFrame,
        *,
        min_t: float,
        max_t: float,
) -> pd.DataFrame:
    """Picks `factor` values between `min_t` and `max_t` thresholds.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    min_t : float
        Lower boarder of `factor` values to pick.
    max_t : float
        Upper boarder of `factor` values to pick.

    Returns
    -------
    pd.DataFrame
        Matrix of True/False, indicating whether factor values are between thresholds or not.
    """

    factor_array = np.asarray(factor)
    return pd.DataFrame(
        (min_t <= factor_array) & (factor_array <= max_t),
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )
