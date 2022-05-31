Quickstart
==========

At first you need some factor data matrix, for which index is date and columns is specific stock.
For example, monthly close prices of USA stocks from 2000. You can use this data to simulate simple
momentum strategy.

.. code:: python

    import pandas as pd

    prices = pd.read_csv("prices.csv", parse_dates=True)

To make prices factor you need to transform and preprocess it. At first, drop all stocks, which
prices is less than 10$. Then calculate percentage changes for 12 months back to get yearly
momentum. Also adjust to look-ahead bias and lag a factor for 1 month. We want to change portfolio
structure every year, so hold values for 12 months. Now we have factor and can base our investment
decisions on it. Use classic quantiles strategy and pick stocks, which have top 30% momentum.

.. code:: python

    import pqr

    momentum_picking = pqr.compose(
        pqr.freeze(pqr.filter, universe=prices > 10),
        pqr.freeze(pqr.look_back, period=12, agg="pct"),
        pqr.freeze(pqr.lag, period=1),
        pqr.freeze(pqr.hold, period=12),
        pqr.freeze(pqr.quantiles, min_q=0.7, max_q=1)
    )

    momentum_signals = momentum_picking(prices)

Then we need to allocate our signals to get relative weights of every stock in a portfolio. Let's
use equally weighted portfolio for example, but you can realise any allocation strategy yourself,
e.g. smart-beta or something more complicated.

.. code:: python

    momentum_holdings = pqr.ew(momentum_signals)

Now we actually have a portfolio, but do not know how it performs. For simplicity assume no
dividend payments, commissions and stocks divisibility (or just very huge balance). So, just
estimate universe returns as percentage change of stocks close prices for 1 period.

.. code:: python

    momentum_returns = pqr.evaluate(momentum_holdings, universe_returns=pqr.to_returns(prices))

We have backtest result of a strategy: signals, holdings and returns. Now you can analyze it in any
way, estimate performance of the strategy and compare it with other strategies, using variety of
other python libraries. Actually, if you want just returns series for a strategy you can combine it
into one pipeline.

.. code:: python

    import pandas as pd
    import pqr

    prices = pd.read_csv("prices.csv", parse_dates=True)

    momentum_strategy = pqr.compose(
        # picking
        pqr.freeze(pqr.filter, universe=prices > 10),
        pqr.freeze(pqr.look_back, period=12, agg="pct"),
        pqr.freeze(pqr.lag, period=1),
        pqr.freeze(pqr.hold, period=12),
        pqr.freeze(pqr.quantiles, min_q=0.7, max_q=1),
        # allocation
        pqr.ew,
        # evaluation
        pqr.freeze(pqr.evaluate, universe_returns=pqr.to_returns(prices)),
    )

    momentum_returns = momentum_strategy(prices)

You are welcome to use other factors and combine them into multi-factor strategies or making even
more complicated pipelines and portfolios. You also can simulate long-short portfolio simply
subtracting signals, holdings and returns. Moreover, leveraging is also supported in pqr, but you
can realise any part of pipeline by your hand.
