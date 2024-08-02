# -*- coding: utf-8 -*-
from Gaussiana import *

class Habilidad(object):
    def __init__(self, forward=Ninf, backward=Ninf, likelihood=Ninf):
        self.forward = forward
        self.backward = backward
        self.likelihood = likelihood
    def __repr__(self):
        return f'Habilidad(forward={self.forward}, backward={self.backward}, likelihood={self.likelihood})'
    @property
    def prior(self):
        return self.forward * self.backward
    @property
    def posterior(self):
        return self.forward * self.backward * self.likelihood
    @property
    def forward_posterior(self):
        return self.forward * self.likelihood
    @property
    def backward_posterior(self):
        return self.backward * self.likelihood

