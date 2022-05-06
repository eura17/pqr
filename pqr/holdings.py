from __future__ import annotations

__all__ = [
    "allocate",
    "ew",
    "scale",
    "limit",
]

import numpy as np
import pandas as pd

from pqr.utils import align


def allocate(
        signals: pd.DataFrame,
        *,
        weights: pd.DataFrame,
) -> pd.DataFrame:
    signals, weights = align(signals, weights)
    signals_array, weights_array = np.asarray(signals), np.asarray(weights)
    signals_array *= weights_array

    norm = np.nansum(signals_array, axis=1, keepdims=True, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame(
            np.nan_to_num(signals_array / norm, nan=0, neginf=0, posinf=0),
            index=signals.index.copy(),
            columns=signals.columns.copy(),
        )


def ew(signals: pd.DataFrame) -> pd.DataFrame:
    return allocate(signals, weights=signals)


def scale(
        holdings: pd.DataFrame,
        *,
        leverage: pd.Series,
) -> pd.DataFrame:
    holdings, leverage = align(holdings, leverage)
    return pd.DataFrame(
        np.asarray(holdings) * np.asarray(leverage)[:, np.newaxis],
        index=holdings.index.copy(),
        columns=holdings.columns.copy(),
    )


def limit(
        holdings: pd.DataFrame,
        *,
        min_leverage: float,
        max_leverage: float,
) -> pd.DataFrame:
    total_leverage = np.nansum(holdings, axis=1, keepdims=True)
    too_low = total_leverage < min_leverage
    too_high = total_leverage > max_leverage

    corrector = np.ones_like(total_leverage, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrector[too_low] *= min_leverage / total_leverage[too_low]
        corrector[too_high] *= max_leverage / total_leverage[too_high]
    corrector = np.nan_to_num(corrector, nan=1, neginf=1, posinf=1)

    corrector = pd.Series(corrector, index=holdings.index.copy())
    return scale(holdings, leverage=corrector)
