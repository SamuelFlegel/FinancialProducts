#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: samuel
"""

import numpy as np

# Binomial Tree

class BinomialTree:
    """ 
    Wrapper for the binomial tree option pricing model using the parametrization of Cox, Ross and Rubinstein
    Used to price European and American Options with flexible payoff function
    Initialize with 
       
        TypeFlag         - 'American' or 'European' string
        payoff_func      - Python function representing the payoff taking strike as 
                           first and underlying price as second argument
        price            - price of underlying
        strike           - strike price
        time_to_maturity - time to maturity given in years 
                            (set to np.inf for perpetual american options)
        riskfree_rate    - annualized expected risk free interest rate until maturity in percent
        up               - multiplicative factor for an upward movement of underlying price
        down             - multiplicative factor for a downward movement of underlying price
        u_probability    - probability for an upward movement of underlying price
        n_steps          - number of steps in tree (number of computations grows quadratic)
    """
    def __init__(self, 
                 TypeFlag,
                 payoff_func,
                 price, strike, 			
                 time_to_maturity, 
                 riskfree_rate, 
                 up, down, u_probability,
                 n_steps):
        self.S = price
        self.X = strike
        self.t = time_to_maturity/n_steps
        self.r = riskfree_rate
        self.u = up
        self.d = down
        self.p = u_probability
        self.n = n_steps
        self.func = payoff_func
        
        if TypeFlag == 'American':
            self.func_2 = lambda x,y: max(x,y)
        elif TypeFlag == 'European':
            self.func_2 = lambda x,y: x
        else:
            print('bad defined TypeFlag')
        

    def price(self):
        """ Get optionprice"""
        price_table = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n + 1):
            price_table[i,self.n] = self.func(self.X,self.S * self.u**(self.n-i) * self.d**i)
            
        for j in range(self.n):
            k = self.n - j         
            for i in range(k):
                price_table[i,k-1] = self.func_2((self.p * price_table[i,k] + (1-self.p) * price_table[i+1,k]) * np.exp(-self.r*self.t), 
                                                 self.func(self.X,self.S * self.u**(k-1-i) * self.d**i))
        return price_table[0,0]



class CRR:
    """ 
    Wrapper for the binomial tree option pricing model using the parametrization of Cox, Ross and Rubinstein
    Used to price European and American Options with flexible payoff function
    Initialize with 
       
        TypeFlag         - 'American' or 'European' string
        payoff_func      - Python function representing the payoff taking strike as 
                           first and underlying price as second argument
        price            - price of underlying
        strike           - strike price
        time_to_maturity - time to maturity given in years 
                            (set to np.inf for perpetual american options)
        riskfree_rate    - annualized expected risk free interest rate until maturity in percent
        cost_of_carry    - cost of holding the underlying given as annualized rate
                         - stock: cost_of_carry = riskfree_rate - continous dividend yield   
                         - future/forward: cost_of_carry = 0
                         - margined Future: cost_of_carry = 0, riskfree_rate = 0
                         - currency: cost_of_carry = risk free rate of underlying currency
        volatility       - annualized volatility of underlying returns
        n_steps          - number of steps in tree (number of computations grows quadratic)
    """
    def __init__(self, 
                 TypeFlag,
                 payoff_func, 
                 price, strike, 			
                 time_to_maturity, 
                 riskfree_rate, cost_of_carry,
                 volatility, n_steps):
        self.F = TypeFlag
        self.S = price
        self.X = strike
        self.T = time_to_maturity
        self.r = riskfree_rate
        self.b = cost_of_carry
        self.v = volatility
        self.n = n_steps
        self.payoff_func = payoff_func
    
    def price(self):
        up = np.exp(self.v * np.sqrt(self.T/self.n))
        down = np.exp(- self.v * np.sqrt(self.T/self.n))
        prob = (np.exp(self.b * self.T/self.n) - down)/(up - down)
        return BinomialTree(self.F, self.payoff_func, self.S, self.X, self.T, self.r, up, down, prob, self.n).price()


# Example for a powered call

p2_call = lambda x,y: max(y-x, 0)**2

S = 100
X = 100
T = 0.5
r = 0.1
b = 0.07
v = 0.3
n = 1000

CRR('European', p2_call, S, X, T, r, b, v, n).price()











