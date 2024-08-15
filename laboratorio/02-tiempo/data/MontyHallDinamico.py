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

####################
### SETUP SESGO
####################

x, y = symbols('x y')
eq = Eq(x**2 + x * (y - 1) + y**2 - y + 0.328333, 0)
solutions = solve(eq)
eqLuna = Eq(x**2 + x * (y - 1) + y**2 - y + 0.332778, 0)
solutions = solve(eqLuna)

sqrt_sol_raices = Eq(-y**2 + 0.666666666666667*y - 0.104444, 0)
limites_sol = solve(sqrt_sol_raices  )
sqrt_luna_raices = Eq(-y**2 + 0.666666666666667*y - 0.110370666666667, 0)
limites_luna = solve(sqrt_luna_raices  )


ysA = np.linspace(start=float(limites_sol[0]),stop=float(limites_sol[1]), num=184)[0:-1]
ysB = np.linspace(start=float(limites_sol[0]),stop=float(limites_sol[1]), num=183)[1:]
ylC = np.linspace(start=float(limites_luna[0]),stop=float(limites_luna[1]),num=15)[0:-1]
ylD = np.linspace(start=float(limites_luna[0]),stop=float(limites_luna[1]),num=15)[1:]

#ys = np.arange(0,1,step=0.000505)
#yl = np.arange(0,1,step=0.004)
#def sol1(ys):
    #return -0.5*ys - 0.866025403784439*np.sqrt(-ys**2 + 0.666666666666667*ys - 0.109) + 0.5
#def sol2(ys):
    #return -0.5*ys + 0.866025403784439*np.sqrt(-ys**2 + 0.666666666666667*ys - 0.109) + 0.5

def sol1(ys):
    return -0.5*ys - 0.866025403784439*np.sqrt(-ys**2 + 0.666666666666667*ys - 0.104444) + 0.5

def sol2(ys):
    return -0.5*ys + 0.866025403784439*np.sqrt(-ys**2 + 0.666666666666667*ys - 0.104444) + 0.5

def luna1(ys):
    return -0.5*ys - 0.866025403784439*np.sqrt(-ys**2 + 0.666666666666667*ys - 0.110370666666667) + 0.5

def luna2(ys):
    return -0.5*ys + 0.866025403784439*np.sqrt(-ys**2 + 0.666666666666667*ys - 0.110370666666667) + 0.5


xA = sol1(ysA)
xB = sol2(ysB)
xC = luna1(ylC)
xD = luna2(ylD)

trayectoriaSol = [ (x, y , 1 - x -y ) for x, y in zip(np.concatenate((xA,np.flip(xB))), np.concatenate((ysA,np.flip(ysB)))) ]
trayectoriaSol = list(reversed(trayectoriaSol))
trayectoriaLuna = [ (x, y , 1 - x -y ) for x, y in zip(np.concatenate((xC,np.flip(xD))), np.concatenate((ylC,np.flip(ylD))) ) ]

Sol4 = [ np.array([t[0],t[1],t[2],0])*0.75 +  np.array([0,0,0,0.25]) for t in trayectoriaSol ]
Luna4 = [ np.array([t[0],t[1],t[2],0])*0.75 +  np.array([0,0,0,0.25]) for t in trayectoriaLuna ]
Luna4 = np.array(list(reversed(Luna4)))

def plot_sol4():
    plt.scatter(range(365),[s[1] for s in Sol4 ])
    plt.scatter(range(365), [s[0] for s in Sol4 ])
    plt.scatter(range(365),[s[2] for s in Sol4 ])
    plt.scatter(range(365),[s[3] for s in Sol4 ])
    plt.show()

def plot_ciclos(byn=True):
    ciclo_lunar = len(trayectoriaLuna)
    ciclo_solar = len(trayectoriaSol)
    color_lunar = np.array([[c/ciclo_lunar, c/ciclo_lunar, c/ciclo_lunar] for c in range(ciclo_lunar)])
    color_solar = np.array([[c/ciclo_solar, c/ciclo_solar, c/ciclo_solar] for c in range(ciclo_solar)])
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    tax.gridlines(multiple=0.1, color="black")
    if byn:
        tax.scatter(trayectoriaSol,  c = color_solar)
        tax.scatter(trayectoriaLuna, c = color_lunar)
    else:
        tax.scatter(trayectoriaSol)
        tax.scatter(trayectoriaLuna)
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f")
    tax.legend()
    tax.show()

def cumple(d=3,m=10,a=1985):
    return (datetime.date(a, m, d)-datetime.date(1492, 10, 12)).days

def reloj_solar(inicio, tiempo):
    return (inicio+tiempo)%365

def reloj_lunar(tiempo):
    return tiempo%28

def crear_regalos(rondas):
    res = []
    for t in range(365):
        # Probabilidad de esconder el regalo
        ps = Sol4[reloj_solar(0, t)]
        res.append(np.random.multinomial(rondas,ps))
    return np.array(res)

def crear_pista(regalo, eleccion):
    opciones = [0,1,2,3]
    opciones.remove(regalo)
    if regalo != eleccion:
        opciones.remove(eleccion)
    return random.choice(opciones)


def crear_secuencia_de_regalos(inicio, rondas=1):
    return [np.argmax(np.random.multinomial(1,Sol4[(t+inicio)%365])) for t in range(rondas*365)]


regalos = crear_regalos(rondas=60)
pd.DataFrame(regalos).to_csv("regalos.csv", index=False)
