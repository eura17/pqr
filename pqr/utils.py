"""Some additional stuff and "sugar" for creating pipelines."""

from __future__ import annotations

__all__ = [
    "align",
    "compose",
    "freeze",
]

from functools import reduce, partial as freeze
from typing import Callable, Any

import pandas as pd


def align(*args: pd.DataFrame | pd.Series) -> tuple[pd.DataFrame | pd.Series, ...]:
    """Aligns dataframes and series to ake them having the same index and columns.

    Parameters
    ----------
    args : sequence of pd.DataFrame or pd.Series
        Dataframes and series to be aligned.

    Returns
    -------
    tuple of pd.DataFrame or pd.Series
      Aligned dataframes and series.
    """

    args = list(args)
    for i in range(len(args) - 1):  # forward-aligning
        args[i], args[i + 1] = _align_two(args[i], args[i + 1])

    for i in range(len(args) - 2, 0, -1):  # backward-aligning
        args[i], args[i - 1] = _align_two(args[i], args[i - 1])
    return tuple(args)


def _are_aligned(
        x: pd.DataFrame | pd.Series,
        y: pd.DataFrame | pd.Series,
        /,
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    index_aligned = ((x.index.shape == y.index.shape) and (x.index == y.index).all())
    if isinstance(x, pd.Series) or isinstance(y, pd.Series):
        return index_aligned

    columns_aligned = ((x.columns.shape == y.columns.shape) and (x.columns == y.columns).all())
    return index_aligned and columns_aligned


def _align_two(
        x: pd.DataFrame | pd.Series,
        y: pd.DataFrame | pd.Series,
        /,
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    if _are_aligned(x, y):
        return x.copy(), y.copy()

    if isinstance(x, pd.Series) or isinstance(y, pd.Series):
        return x.align(y, join="inner", axis=0)

    return x.align(y, join="inner")


def compose(*steps) -> Callable[[Any, ...], Any]:
    """Combines functions to pipeline.

    Parameters
    ----------
    steps : sequence of callable
         Steps to be composed into pipeline.

    Returns
    -------
    callable
        Function, realizing full pipeline.
    """

    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), steps)
