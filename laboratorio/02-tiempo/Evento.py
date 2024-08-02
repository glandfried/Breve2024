# -*- coding: utf-8 -*-
import math
import numpy as np
from Gaussiana import *

BETA = 1.0 # El desvío estándar del desempeño: N(desempeño | mu=habilidad, sd=BETA).

#import trueskillthroughtime as ttt
#game = ttt.Game([[ttt.Player(ttt.Gaussian(0,3))], [ttt.Player(ttt.Gaussian(0,3))] ])
#game.likelihoods
#game.posteriors()

class Evento(object):
    #
    # Constructor
    def __init__(self, equipos):
        # Ejemplo:
        # equipos = [ [priorA, priorB], [priorC]  ]
        # donde el el orden indica qué equipo ganó.
        self.equipos = equipos
    #
    # Representación
    def __repr__(self):
        return f'{self.equipos}'
    #
    @property
    def desempeño_individuos(self):
        # Agrega ruido a los desempeños individuales
        # Y suma el desempeño de todos los integrantes del equipo
        return [ [ inegrante + Gaussian(0,BETA) for inegrante in equipo ] for equipo in self.equipos]
    #
    @property
    def desempeño_equipos(self):
        # Desempeño de los equipos.
        return [ np.sum(equipo) for equipo in self.desempeño_individuos]
    #
    @property
    def prior_diferencia(self):
        desempeño_equipos = self.desempeño_equipos
        return desempeño_equipos[0]-desempeño_equipos[1]
    #
    @property
    def likelihood(self):
        #
        evento = self
        # Diferencia
        prior_diferencia = evento.prior_diferencia
        marginal_diferencia = prior_diferencia > 0  # approx(P(diferencia, resultado))
        #  P(resultado | diferencia) = P(diferencia, resultado) / P(diferencia)
        likelihood_diferencia =  marginal_diferencia / evento.prior_diferencia
        #
        # Equipos
        desempeño_equipos = evento.desempeño_equipos
        # (d = ta - tb): diferencia entre los desempeños de los equipos "a" (ta) y "b" (tb)
        likelihood_equipos = [
            desempeño_equipos[1] + likelihood_diferencia, # ta = tb + d  ganadores
            desempeño_equipos[0] - likelihood_diferencia] # tb = ta - d  perdedores
        #
        # Individuos
        desempeño_individuos = evento.desempeño_individuos
        # ta = p1 + p2 + p3; p1 = ta - (p2 + p3)
        likelihood_individuos = [
            [ (likelihood_equipos[e] - desempeño_equipos[e].exclude(desempeño_individuos[e][i]))
               + Gaussian(0,BETA) for i in range(len(evento.equipos[e])) ]  for e in range(len(evento.equipos))]
        return likelihood_individuos
    #
    @property
    def posterior(self):
        likelihood = self.likelihood
        prior = self.equipos
        return [ [ prior[e][i]*likelihood[e][i] for i in range(len(prior[e]))] for e in range(len(prior))]


