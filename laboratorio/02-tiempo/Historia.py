# -*- coding: utf-8 -*-
from Gaussiana import *
from Evento import *
from collections import defaultdict
import math
from copy import copy


eventos = [ [["a"],["b"]],
            [["b"],["a"]] ]
h = Historia(eventos)

h.forward_propagation()
h.posteriors()

class Habilidad(object):
    def __init__(self, forward=Ninf, backward=Ninf, likelihood=Ninf):
        self.forward = forward
        self.backward = backward
        self.likelihood = likelihood
    def __repr__(self):
        return f'Habilidad(forward={self.forward}, backward={self.backward}, likelihood={self.likelihood})'
    def prior(self):
        return self.forward * self.backward
    def posterior(self):
        return self.forward * self.backward * self.likelihood
    def forward_posterior(self):
        return self.forward * self.likelihood
    def backward_posterior(self):
        return self.forward * self.likelihood

class Historia(object):
    def __init__(self, eventos, priors=defaultdict(lambda: Gaussian(0.0, 3.0)) ):
        self.eventos = eventos
        self.priors = priors
        self.habilidades = [ [[Habilidad() for integrante in equipo] for equipo in evento] for evento in eventos]
    def __repr__(self):
        return f'Historia(Eventos={len(eventos)})'
    def forward_propagation(self):
        h = self
        ultimo_mensaje = copy(h.priors)
        for t in range(len(h.eventos)):#t=0
            priors_t = [[ultimo_mensaje[h.eventos[t][e][i]]
                 for i in range(len(h.eventos[t][e]))] for e in range(len(h.eventos[t])) ]
            likelihood = Evento(priors_t).likelihood
            for e in range(len(h.eventos[t])):
                for i in range(len(h.eventos[t][e])):
                    h.habilidades[t][e][i].forward = ultimo_mensaje[h.eventos[t][e][i]]
                    h.habilidades[t][e][i].likelihood = likelihood[e][i]
                    ultimo_mensaje[h.eventos[t][e][i]] = h.habilidades[t][e][i].forward_posterior()
    #
    #def backward_propagation(self):
        #h = self
        #ultimo_mensaje = defaultdict(lambda: Gaussian(0.0, math.inf)) )
        #for t in range(len(h.eventos)):#t=0
            #priors_t = [[ultimo_mensaje[h.eventos[t][e][i]]
                 #for i in range(len(h.eventos[t][e]))] for e in range(len(h.eventos[t])) ]
            #likelihood = Evento(priors_t).likelihood
            #for e in range(len(h.eventos[t])):
                #for i in range(len(h.eventos[t][e])):
                    #h.habilidades[t][e][i].forward = ultimo_mensaje[h.eventos[t][e][i]]
                    #h.habilidades[t][e][i].likelihood = likelihood[e][i]
                    #ultimo_mensaje[h.eventos[t][e][i]] = h.habilidades[t][e][i].forward_posterior()
    def posteriors(self):
        return [ [[habilidad.posterior() for habilidad in equipo] for equipo in evento] for evento in self.habilidades]


