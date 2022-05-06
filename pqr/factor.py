"""Basic stuff for transforming and preprocessing raw data into factors."""

from __future__ import annotations

__all__ = [
    "ffilter",
    "fpct",
    "fmean",
    "fmedian",
    "fmin",
    "fmax",
    "flag",
    "fhold",
]

import numpy as np
import pandas as pd

from pqr.utils import align


def ffilter(
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


def fpct(
        factor: pd.DataFrame,
        *,
        period: int,
) -> pd.DataFrame:
    """Calculates `factor` values rate of change for `period`.

    Can be used to make factor from static to dynamic (e.g. momentum from prices).

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    period : int
        Period to look back on the data.

    Returns
    -------
    pd.DataFrame
        Percentage changes of `factor` values.
    """

    factor_array = factor.to_numpy()
    abs_change = (factor_array[period:] - factor_array[:-period])
    base = factor_array[:-period]
    return pd.DataFrame(
        abs_change / base,
        index=factor.index[period:].copy(),
        columns=factor.columns.copy()
    )


def fmean(
        factor: pd.DataFrame,
        *,
        period: int,
) -> pd.DataFrame:
    """Calculates `factor` values rolling mean for `period`.

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

    return factor.rolling(period, axis=0).mean().iloc[period:]


def fmedian(
        factor: pd.DataFrame,
        *,
        period: int,
) -> pd.DataFrame:
    """Calculates `factor` values rolling median for `period`.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    period : int
        Period to look back on the data.

    Returns
    -------
    pd.DataFrame
        Rolling median of `factor` values.
    """

    return factor.rolling(period, axis=0).median().iloc[period:]


def fmin(
        factor: pd.DataFrame,
        *,
        period: int,
) -> pd.DataFrame:
    """Calculates `factor` values rolling min for `period`.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    period : int
        Period to look back on the data.

    Returns
    -------
    pd.DataFrame
        Rolling min of `factor` values.
    """

    return factor.rolling(period, axis=0).min().iloc[period:]


def fmax(
        factor: pd.DataFrame,
        *,
        period: int,
) -> pd.DataFrame:
    """Calculates `factor` values rolling max for `period`.

    Parameters
    ----------
    factor : pd.DataFrame
        Matrix with factor values.
    period : int
        Period to look back on the data.

    Returns
    -------
    pd.DataFrame
        Rolling max of `factor` values.
    """

    return factor.rolling(period, axis=0).max().iloc[period:]


def flag(
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


def fhold(
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
