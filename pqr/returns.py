from __future__ import annotations

__all__ = [
    "calculate",
    "to_returns",
]

import numpy as np
import pandas as pd

from pqr.utils import align


def calculate(
        holdings: pd.DataFrame,
        *,
        universe_returns: pd.DataFrame,
) -> pd.Series:
    holdings, universe_returns = align(holdings, universe_returns)
    returns = np.zeros(len(holdings))
    returns[1:] = np.asarray(holdings)[:-1] * np.asarray(universe_returns)[1:]
    return pd.Series(returns, index=holdings.index.copy())


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices_array = np.asarray(prices)
    universe_returns = np.zeros_like(prices_array, dtype=float)
    universe_returns[1:] = np.diff(prices_array, axis=0) / prices_array[:-1]
    universe_returns = np.nan_to_num(universe_returns, nan=0, neginf=0, posinf=0)
    return pd.DataFrame(
        universe_returns,
        index=prices.index.copy(),
        columns=prices.columns.copy(),
    )
