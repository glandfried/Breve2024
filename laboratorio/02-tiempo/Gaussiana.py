# -*- coding: utf-8 -*-
"""
   TrueskillThroughTime.py
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   :copyright: (c) 2019-2024 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
"""

import math; inf = math.inf; sqrt2 = math.sqrt(2); sqrt2pi = math.sqrt(2 * math.pi)
from scipy.stats import norm
from scipy.stats import truncnorm

__all__ = ['MU', 'SIGMA', 'Gaussian', 'N01', 'N00', 'Ninf', 'Nms', 'cdf', 'pdf', 'ppf', 'trunc', 'approx']

MU = 0.0
SIGMA = 6
PI = SIGMA**-2
TAU = PI * MU


def erfc(x):
    #"""(http://bit.ly/zOLqbc)"""
    z = abs(x)
    t = 1.0 / (1.0 + z / 2.0)
    a = -0.82215223 + t * 0.17087277; b =  1.48851587 + t * a
    c = -1.13520398 + t * b; d =  0.27886807 + t * c; e = -0.18628806 + t * d
    f =  0.09678418 + t * e; g =  0.37409196 + t * f; h =  1.00002368 + t * g
    r = t * math.exp(-z * z - 1.26551223 + t * h)
    return r if not(x<0) else 2.0 - r

def erfcinv(y):
    if y >= 2: return -inf
    if y < 0: raise ValueError('argument must be nonnegative')
    if y == 0: return inf
    if not (y < 1): y = 2 - y
    t = math.sqrt(-2 * math.log(y / 2.0))
    x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
    for _ in [0,1,2]:
        err = erfc(x) - y
        x += err / (1.12837916709551257 * math.exp(-(x**2)) - x * err)
    return x if (y < 1) else -x

def tau_pi(mu,sigma):
    if sigma > 0.0:
        pi_ = sigma ** -2
        tau_ = pi_ * mu
    elif (sigma + 1e-5) < 0.0:
        raise ValueError(" sigma should be greater than 0 ")
    else:
        pi_ = inf
        tau_ = inf
    return tau_, pi_

def mu_sigma(tau_,pi_):
    if pi_ > 0.0:
        sigma = math.sqrt(1/pi_)
        mu = tau_ / pi_
    elif pi_ + 1e-5 < 0.0:
        raise ValueError(" sigma should be greater than 0 ")
    else:
        sigma = inf
        mu = 0.0
    return mu, sigma


def cdf(x, mu=0, sigma=1):
    z = -(x - mu) / (sigma * sqrt2)
    return (0.5 * erfc(z))


def pdf(x, mu, sigma):
    normalizer = (sqrt2pi * sigma)**-1
    functional = math.exp( -((x - mu)**2) / (2*sigma**2) )
    return normalizer * functional


def ppf(p, mu, sigma):
    return mu - sigma * sqrt2  * erfcinv(2 * p)



def v_w(mu, sigma, margin,tie):
    if not tie:
        _alpha = (margin-mu)/sigma
        v = pdf(-_alpha,0,1) / cdf(-_alpha,0,1)
        w = v * (v + (-_alpha))
    else:
        _alpha = (-margin-mu)/sigma
        _beta  = ( margin-mu)/sigma
        v = (pdf(_alpha,0,1)-pdf(_beta,0,1))/(cdf(_beta,0,1)-cdf(_alpha,0,1))
        u = (_alpha*pdf(_alpha,0,1)-_beta*pdf(_beta,0,1))/(cdf(_beta,0,1)-cdf(_alpha,0,1))
        w =  - ( u - v**2 )
    return v, w


def trunc(mu, sigma, margin=0.0, tie=False):
    v, w = v_w(mu, sigma, margin, tie)
    mu_trunc = mu + sigma * v
    sigma_trunc = sigma * math.sqrt(1-w)
    return mu_trunc, sigma_trunc

def approx(N, margin = 0.0, tie= False):
    mu, sigma = trunc(N.mu, N.sigma, margin, tie)
    return Gaussian(mu, sigma)

"""
def tau_pi(mu,sigma):
    # Transforma parámetros mu y sigma en tau y pi.
    if sigma > 0.0:
        pi_ = sigma ** -2
        tau_ = pi_ * mu
    elif (sigma + 1e-5) < 0.0:
        raise ValueError(" sigma should be greater than 0 ")
    else:
        pi_ = inf
        tau_ = inf
    return tau_, pi_

def mu_sigma(tau_,pi_):
    if pi_ > 0.0:
        sigma = math.sqrt(1/pi_)
        mu = tau_ / pi_
    elif pi_ + 1e-5 < 0.0:
        raise ValueError(" sigma should be greater than 0 ")
    else:
        sigma = inf
        mu = 0.0
    return mu, sigma

def cdf(x, mu=0, sigma=1):
    return norm.cdf(x, loc=mu, scale=sigma)

def pdf(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)

def ppf(p, mu, sigma):
    return norm.ppf(x, loc=mu, scale=sigma)

def trunc(mu, sigma, margin, tie):
    a, b = (-margin,margin) if tie else (margin, math.inf)
    trunc = truncnorm(a, b, loc=mu, scale=sigma)
    return trunc.mean(), trunc.std()

def approx(N, margin, tie):# margin = 1; tie = False; N = N01
    mu, sigma = trunc(N.mu, N.sigma, margin, tie)
    return Gaussian(mu, sigma)

"""


class Gaussian(object):
    """
    The `Gaussian` class is used to define the prior beliefs of the agents' skills

    Attributes
    ----------
    mu : float
        the mean of the `Gaussian` distribution.
    sigma :
        the standar deviation of the `Gaussian` distribution.
    """
    def __init__(self,mu=MU, sigma=SIGMA):
        if sigma >= 0.0:
            self.mu, self.sigma = mu, sigma
        else:
            raise ValueError(" sigma should be greater than 0 ")
    @property
    def tau(self):
        if self.sigma > 0.0:
            return self.mu * (self.sigma**-2)
        else:
            return inf
    @property
    def pi(self):
        if self.sigma > 0.0:
            return self.sigma**-2
        else:
            return inf
    def __iter__(self):
        return iter((self.mu, self.sigma))
    def __repr__(self):
        return 'N(mu={:.3f}, sigma={:.3f})'.format(self.mu, self.sigma)
    def __gt__(self, other):
        return approx(Gaussian(self.mu, self.sigma), margin = other, tie = False)
    def __ge__(self, other):
        return self.__gt__(other)
    def __add__(self, M):
        return Gaussian(self.mu + M.mu, math.sqrt(self.sigma**2 + M.sigma**2))
    def __sub__(self, M):
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 + M.sigma**2))
    def __mul__(self, M):
        if type(M) == float:
            if M == inf:
                return Ninf
            else:
                return Gaussian(M*self.mu, abs(M)*self.sigma)
        else:
            if self.sigma == 0.0 or M.sigma == 0.0:
                mu = self.mu/((self.sigma**2/M.sigma**2) + 1) if self.sigma == 0.0 else M.mu/((M.sigma**2/self.sigma**2) + 1)
                sigma = 0.0
            else:
                _tau, _pi = self.tau + M.tau, self.pi + M.pi
                mu, sigma = mu_sigma(_tau, _pi)
            return Gaussian(mu, sigma)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, M):
        _tau = self.tau - M.tau; _pi = self.pi - M.pi
        mu, sigma = mu_sigma(_tau, _pi)
        return Gaussian(mu, sigma)
    def forget(self,gamma,t):
        return Gaussian(self.mu, math.sqrt(self.sigma**2 + t*gamma**2))
    def delta(self, M):
        return abs(self.mu - M.mu) , abs(self.sigma - M.sigma)
    def exclude(self, M):
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 - M.sigma**2) )
    def isapprox(self, M, tol=1e-4):
        return (abs(self.mu - M.mu) < tol) and (abs(self.sigma - M.sigma) < tol)


N01 = Gaussian(0,1)
N00 = Gaussian(0,0)
Ninf = Gaussian(0,inf)
Nms = Gaussian(MU, SIGMA)

