"""Implemented classic factor strategies."""

from __future__ import annotations

__all__ = [
    "quantiles",
    "top",
    "bottom",
    "thresholds",
]

import numpy as np
import pandas as pd


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

    def _top_k_row(arr: np.ndarray) -> np.ndarray:
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

    def _bottom_k_row(arr: np.ndarray) -> np.ndarray:
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
