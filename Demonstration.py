#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: samuelflegel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Vanilla import implied_V, Vanilla
from BinomialTree import CRR

# Example of Vanilla

Opt1 = Vanilla('European','c', 18, 19, 1, 0.1, 0.05, 0.28)
print(Opt1.price())
print(Opt1.Delta())

Opt2 = Vanilla('European','p', 18, 19, 1, 0.1, 0.05, 0.28)
print(Opt2.price())
print(Opt2.Delta())

Opt3 = Vanilla('American','c', 18, 19, 1, 0.1, 0.05, 0.28)
print(Opt3.price())
print(Opt3.Delta())

Opt4 = Vanilla('American','p', 18, 19, 1, 0.1, 0.05, 0.28)
print(Opt4.price())
print(Opt4.Delta())

Opt5 = Vanilla('American','c', 18, 19, np.inf, 0.1, 0.05, 0.28)
print(Opt5.price())
print(Opt5.Delta())

# Bad example of a volatility smirk 

X = pd.read_csv('TestDataSet.csv', index_col=0)

X['vola'] = implied_V('European', X['Call/Put'], X['UnderlyingPrice'], X['Strike'], 
                 X['TimetoMaturity'], X['InterestRate'], 0, X['Marketprice'], 0.01)


put_data = X.loc[X['Call/Put'] == 'p'].dropna()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(put_data['Strike']/put_data['UnderlyingPrice'], put_data['TimetoMaturity'], 
           put_data['vola'])

ax.set_xlabel('Moneyness')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Volatility')

plt.show()


# Example for a powered call using Cox-Ross-Rubinstein

p2_call = lambda x,y: max(y-x, 0)**2

S = 100
X = 100
T = 0.5
r = 0.1
b = 0.07
v = 0.3
n = 1000

CRR('European', p2_call, S, X, T, r, b, v, n).price()





















