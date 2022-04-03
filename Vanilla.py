#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: samuel flegel
"""

from scipy.stats import multivariate_normal, norm
import numpy as np

#### Generalized Black Scholes Formula

def GBlackScholes(CallPutFlag, S, X, T, r, b, v):
    d_1 = (np.log(S/X) + (b + v**2/2) * T) / (v * T**(1/2))
    d_2 = d_1 - v * T**(1/2)
    if CallPutFlag == 'c':
        return S * np.e**((b-r)*T) * norm.cdf(d_1) - X * np.e**(-r*T) * norm.cdf(d_2)
    else:
        return X * np.e**(-r*T) * norm.cdf(-d_2) - S * np.e**((b-r)*T) * norm.cdf(-d_1)


#### Bjerksund Stensland 1993 American Option Algorithm

# Help Function
def phi(S, T, gamma, h, i, r, b, v):
    L = (-r + gamma * b + gamma * (gamma - 1) * v**2/2 ) * T
    d = -(np.log(S/h) + (b + (gamma - 1/2) * v**2) * T ) / (v * T**(1/2))
    K = 2 * b / v ** 2 + (2 * gamma - 1)
    return np.e**L * S**gamma * (norm.cdf(d) - (i/S)**K * norm.cdf(d - (2 * np.log(i/S)/(v * T**(1/2)) )))


def BSAmerican_fast(CallPutFlag, S, X, T, r, b, v):
    if T == np.inf: # If perpetual
        if CallPutFlag != 'c':
            gam = 1/2 - b/v**2 - ((b/v**2 - 1/2)**2 + 2*r/v**2 )**(1/2)
            return X/(1-gam) * ((gam-1)/gam * S/X)**gam
        else:
            gam = 1/2 - b/v**2 + ((b/v**2 - 1/2)**2 + 2*r/v**2 )**(1/2)
            return X/(gam - 1) * ((gam-1)/gam * S/X)**gam
    else: # if normal
        if CallPutFlag != 'c':
            r = r - b
            b = -b
            S, X = X, S
        if b >= r:
            return GBlackScholes('c', S, X, T, r, b, v)
        else:
            Beta = (1/2 - b/v**2) + ((b/v**2 - 1/2)**2 + 2 * r/v**2)**(1/2)
            BInfinity = Beta/(Beta - 1) * X
            B0 = max(X, r/(r-b) * X)
            
            ht = -(b * T + 2 * v * T**(1/2)) * B0 / (BInfinity - B0) 
            i = B0 + (BInfinity - B0) * (1 - np.e**(ht))
            alfa = (i - X) * i**(-Beta)
            if S >= i:
                return S - X
            else:
                return (alfa * S ** Beta - alfa * phi(S, T, Beta, i, i, r, b, v)
                        + phi(S, T, 1, i, i, r, b, v) - phi(S, T, 1, X, i, r, b, v)
                        - X * phi(S, T, 0, i, i, r, b, v) + X * phi(S, T, 0, X, i, r, b, v))



class Vanilla:
    """ 
     Vanilla Option Pricing Class

     Used to price European and American Options on Stocks, Futures/Forwards and Currency 
     and calculate corresponding numerical Greeks

     Generalized Black Scholes 1973, Merton 1973, Black 1976, Asay 1982 and
     Garman and Kohlhagen 1983, Bjerksund Stensland 1993


    Initialize with 
       
        TypeFlag         - 'American' or 'European' string
        CallPutFlag      - 'c' string for call, 'p' string for put
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
    """
    def __init__(self, 
                 TypeFlag, CallPutFlag,
                 price, strike, 			
                 time_to_maturity, 
                 riskfree_rate, cost_of_carry,
                 volatility):
        self.F = CallPutFlag
        self.S = price
        self.X = strike
        self.T = time_to_maturity
        self.r = riskfree_rate
        self.b = cost_of_carry
        self.v = volatility
        
        if TypeFlag == 'American':
            self.func = BSAmerican_fast
        elif TypeFlag == 'European':
            self.func = GBlackScholes
        else:
            print('bad defined TypeFlag')
            
        if CallPutFlag not in ['c','p']:
            print('bad defined CallPutFlag')
        

    def price(self):
        """ Get optionprice"""
        return self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v)
    
    def Delta(self, diff = 0.001):
        """ Get delta = Doptionprice / Dprice"""
        return (self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v) 
                - self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v))/(diff * 2)

    def Vanna(self, diff = 0.001):
        """ Get vanna = Ddelta / Dvolatility """
        return (self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v + diff) 
                - self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v - diff)
                - self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v + diff) 
                + self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v - diff))/(4 * diff **2)
    
    def DvannaDvol(self, diff = 0.001):
        """ Get Dvanna / Dvolatility"""
        return (self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v + diff) 
                - 2 * self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v)
                + self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v - diff) 
                - self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v + diff) 
                + 2 * self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v)
                - self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v - diff))/(2 * diff**3)
    
    def Charm(self, diff = 0.001):
        """ Get charm = Ddelta / Dtime"""
        return (self.func(self.F, self.S + diff, self.X, self.T + diff, self.r, self.b, self.v) 
                - self.func(self.F, self.S + diff, self.X, self.T - diff, self.r, self.b, self.v)
                - self.func(self.F, self.S - diff, self.X, self.T + diff, self.r, self.b, self.v) 
                + self.func(self.F, self.S - diff, self.X, self.T - diff, self.r, self.b, self.v))/(-4 * diff **2)           
    
    def Lambda(self, diff = 0.001):
        """ Get lambda = omega = elasticity = delta * price / optionprice"""
        return self.Delta(diff = diff) * self.S / self.price()
    
    def Elasticity(self, diff = 0.001):
        """ Get lambda = omega = elasticity = delta * price / optionprice"""
        return self.Lambda(diff = diff)
    
    def Omega(self, diff = 0.001):
        """ Get lambda = omega = elasticity = delta * price / optionprice"""
        return self.Lambda(diff = diff)
    
    def Gamma(self, diff = 0.001):
        """ Get gamma = convexity = Ddelta / Dprice"""
        return (self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v) 
                + self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v)
                - 2 * self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v))/(diff **2)   
    
    def Convexity(self, diff = 0.001):
        """ Get gamma = convexity = Ddelta / Dprice"""
        return self.Gamma(diff = diff)
    
    def GammaP(self, diff = 0.001):
        """ Get gamma percent = gammaP = gamma * price / 100"""
        return self.S * self.Gamma(diff = diff) / 100
    
    def Zomma(self, diff = 0.001):
        """ Get zomma = Dgamma / Dvol"""
        return (self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v + diff) 
                - 2 * self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v + diff)
                + self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v + diff) 
                - self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v - diff) 
                + 2 * self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v - diff)
                - self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v - diff))/(2 * diff**3)      
    
    def ZommaP(self, diff = 0.001):
        """ Gat zommaP = DgammaP / Dvol"""
        return self.S * self.Zomma(diff = diff) / 100 
    
    def Speed(self, diff = 0.001):
        """ Get speed = Dgamma / Dprice"""
        return (self.func(self.F, self.S + 2 * diff, self.X, self.T, self.r, self.b, self.v) 
                - 3 * self.func(self.F, self.S + diff, self.X, self.T, self.r, self.b, self.v)
                + 3 * self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v)
                - self.func(self.F, self.S - diff, self.X, self.T, self.r, self.b, self.v))/(diff ** 3)   
    
    def SpeedP(self):
        """ Get speed percent = speedP = DgammaP / Dprice"""
        return self.S * self.Speed(diff = diff) / 100 
    
    def Colour(self, diff = 0.001):
        """ Get colour = gamma_bleed = Dgamma / Dtime """
        return (self.func(self.F, self.S + diff, self.X, self.T + diff, self.r, self.b, self.v) 
                - 2 * self.func(self.F, self.S, self.X, self.T + diff, self.r, self.b, self.v)
                + self.func(self.F, self.S - diff, self.X, self.T + diff, self.r, self.b, self.v) 
                - self.func(self.F, self.S + diff, self.X, self.T - diff, self.r, self.b, self.v) 
                + 2 * self.func(self.F, self.S, self.X, self.T - diff, self.r, self.b, self.v)
                - self.func(self.F, self.S - diff, self.X, self.T - diff, self.r, self.b, self.v))/(-2 * diff**3)     
    
    def ColourP(self, diff = 0.001):
        """ Get colourP = gamma_percant_bleed = DgammaP / Dtime """
        return self.S * self.Colour(diff = diff) / 100 
    
    def Vega(self, diff = 0.001):
        """ Get vega = Doptionprice / Dvolatility """
        return (self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v + diff) 
                - self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v - diff))/(diff * 2)

    def VegaP(self):
        """ Get vegaP = volatility / 10 * Doptionprice / Dvolatility """
        return self.v * self.Vega() / 10
    
    def VegaBleed(self, diff = 0.001):
        """ Get vega bleed = Dvega / Dtime """
        return (self.func(self.F, self.S, self.X, self.T + diff, self.r, self.b, self.v + diff) 
                - self.func(self.F, self.S, self.X, self.T - diff, self.r, self.b, self.v + diff)
                - self.func(self.F, self.S, self.X, self.T + diff, self.r, self.b, self.v - diff) 
                + self.func(self.F, self.S, self.X, self.T - diff, self.r, self.b, self.v - diff))/(4 * diff ** 2)  
    
    def Vomma(self, diff = 0.001):
        """ Get vomma = volga = Dvega / Dvolatility """
        return (self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v + diff) 
                + self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v - diff)
                - 2 * self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v))/(diff **2)   
    
    def VommaP(self):
        """ Get vommaP = volatility / 10 * Dvega / Dvolatility """
        return self.v * self.Vomma() / 10 

    def Ultima(self, diff = 0.001):
        """ Get ultima = Dvomma / Dvolatility """
        return (self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v + 2 * diff) 
                - 3 * self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v + diff)
                + 3 * self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v)
                - self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v - diff))/(diff ** 3)         

    def Theta(self, diff = 0.001):
        """ Get theta = expected bleed """
        return (self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v) 
                - self.func(self.F, self.S, self.X, self.T - diff, self.r, self.b, self.v))/-diff

    def Rho(self, diff = 0.001, future = False):
        """ Get rho = Doptionprice / Driskfree_rate, set future=True for Forward/Future as underlying """
        return (self.func(self.F, self.S, self.X, self.T, self.r + diff, self.b + diff * (future != False), self.v) 
                - self.func(self.F, self.S, self.X, self.T, self.r - diff, self.b - diff * (future != False), self.v))/(diff * 2)

    def Phi(self, diff = 0.001):
        """ Get phi = rho2 = Doptionprice / D(riskfree_rate - cost_of_carry) = Doptionprice / Ddividend_yield """
        return (self.func(self.F, self.S, self.X, self.T, self.r, self.b + diff, self.v) 
                - self.func(self.F, self.S, self.X, self.T, self.r, self.b - diff, self.v))/(diff * -2)
        
    def StrikeDelta(self, diff = 0.001):
        """ Get strike delta = discounted probability = Doptionprice / Dstrike """
        return (self.func(self.F, self.S, self.X + diff, self.T, self.r, self.b, self.v) 
                - self.func(self.F, self.S, self.X - diff, self.T, self.r, self.b, self.v))/(diff * 2)

    def StrikeGamma(self, diff = 0.001):
        """ Get strike gamma = RiskNeutralDensity = Dstrike_delta / Dstrike """
        return (self.func(self.F, self.S, self.X + diff, self.T, self.r, self.b, self.v) 
                + self.func(self.F, self.S, self.X - diff, self.T, self.r, self.b, self.v)
                - 2 * self.func(self.F, self.S, self.X, self.T, self.r, self.b, self.v))/(diff **2)  

### Only for European Options
    
#    def Zeta(self): ######
#        """ Get zeta = in the money probability """
#        return norm.cdf(self.F * self.d_2)
#    
#    def DzetaDvol(self): ######
#        """ Get Dzeta/Dvolatility """
#        return - self.F * norm.pdf(self.d_2) * self.d_1 / self.v
#        
#    def ZetaBleed(self): ######
#        """ Get - Dzeta/Dtime """
#        A = self.b / (self.v * self.T**(1/2)) - self.d_1 / (2 * self.T)
#        return norm.pdf(self.d_2) * A * self.F


@np.vectorize
def implied_V(TypeFlag, CallPutFlag, S, X, T, r, b, P, eps):
    "Calculates implied volatility"
    v = np.sqrt(np.abs(np.log(S/X) + r*T) * 2 / T)
    Opt = Vanilla(TypeFlag, CallPutFlag, S, X, T, r, b, v)
    impP = Opt.price()
    while abs(P - impP) > eps:
        v = v - (impP - P)/Opt.Vega()
        Opt = Vanilla(TypeFlag, CallPutFlag, S, X, T, r, b, v)
        impP = Opt.price()
    return v



















    
    
