# pqr

pqr is a python library for backtesting factor strategies. It is built in top of numpy, so it is 
fast and memory efficient, but provides pandas interface to make usage more convenient and verbose.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pqr.

```bash
pip install pqr
```

## Quickstart

```python
import pandas as pd
import pqr

prices = pd.read_csv("prices.csv", parse_dates=True)

momentum = pqr.compose(
    # picking
    pqr.freeze(pqr.filter, universe=prices > 10),
    pqr.freeze(pqr.look_back, period=12, agg="pct"),
    pqr.freeze(pqr.lag, period=1),
    pqr.freeze(pqr.hold, period=12),
    pqr.freeze(pqr.quantiles, min_q=0.7, max_q=1),
    # allocation
    pqr.ew,
    # evaluation
    pqr.freeze(pqr.calculate, universe_returns=pqr.to_returns(prices)),
)

# returns series of returns of 30% ew momentum 12-1-12 strategy for stocks > 10$
momentum(prices)
```

## Documentation

The official documentation is hosted on readthedocs.org: https://pqr.readthedocs.io/en/latest/

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would 
like to change.

Please make sure to update tests and documentation as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
