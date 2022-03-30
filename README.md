# FinancialProducts

Easy to use implementation of different algorithms to value financial products. Currently in very early stage.

## Vanilla.py
The Vanilla class allows for valuation of European, American and American perpetual Options on Stocks, Futures/Forwards and Currency, as well as calculating the corresponding greeks. 

Calculations are based on Generalized Black Scholes 1973, Merton 1973, Black 1976, Asay 1982, Garman and Kohlhagen 1983, Bjerksund Stensland 1993. Greeks are implemented numerically.

The implied_V function allows to calculate the implied volatility.

## Demonstration.py
Demonstration of the use of the exsisting functions.

## Planned
Full support of using numpy arrays to price multiple products at once.

Programs to price:
Exotic Options, Interest Rate Derivatives, Discrete Dividend Options

## Notes
The tables provided in Espen Gaarder Haugs Complete Guide to Option Pricing Formulas were used to check for errors.

The valuation algorithms are based on theoretical models and are therefore underlying rigid assumptions, which are not met in reality. These functions are a scientific exercise only and should especially not be used for investment decision making.




