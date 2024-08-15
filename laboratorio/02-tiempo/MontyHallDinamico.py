import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ternary
import numpy as np
from sympy import plot_implicit, symbols, Eq, solve
import math
from operator import mul
from math import gamma
import random
import datetime
from scipy.stats import multinomial
from scipy.optimize import minimize
import pickle
import pandas as pd
from funciones import *

N = 12

regalos = np.array(pd.read_csv("data/regalos.csv"))

def p2(p1, zz, uniforme = 0.25):
    # zz := tasa de olvido
    # Si son 4 casos, lo uniforme es 0.25
    return uniforme * zz + p1 * (1-zz)

def p1(p2, zz, uniforme = 0.25):
    # zz := tasa de olvido
    # Con zz < 1, no puede se olvido total.
    return (p2 - uniforme*zz)/(1-zz)


# D(p | at_1,at_2,at_3) \propto \prod_i pt_i^\at_i
#
#
# fa0 = I(a0 = (1,1,1,1))
#
# [fa0]
#  |      fa1 = I(a1_i = zz + a0_1 (1-zz)  )
#  v
#  a0 -------[fa1]------> a1
#  |                                          m_fa0_a0 = I(a0 = (1,1,1,1) )
#  |
# [fp0] p(p0|a0) = D(p0|a0)
#  |                                          m_fp0_a0 = \int_p0 D(p0|a0)*p_{r*} (Esto integra 1!??, como puede ser que no suba información)
#  v
#  p0
#  |
#  |
# [fr0]    m_fr_p = Cat(r*|p) = p_r*
#  |
#  v
# (r*) Dato
#  |
# ...







