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
        min_q: float,
        max_q: float,
) -> pd.DataFrame:
    factor_array = np.asarray(factor)
    lower, upper = np.nanquantile(factor_array, [min_q, max_q], axis=1, keepdims=True)
    return pd.DataFrame(
        (lower <= factor_array) & (factor_array <= upper),
        index=factor.index.copy(),
        columns=factor.columns.copy(),
    )


def top(
        factor: pd.DataFrame,
        k: int,
) -> pd.DataFrame:
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
        k: int,
) -> pd.DataFrame:
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
        min_t: float,
        max_t: float,
) -> pd.DataFrame:
    factor_array = np.asarray(factor)
    return pd.DataFrame(
        (min_t <= factor_array) & (factor_array <= max_t),
        index=factor.index.copy(),
        columns=factor.columns.copy()
    )
