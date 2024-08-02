# -*- coding: utf-8 -*-
import trueskillthroughtime as ttt
from Historia import *


eventos = [ [["a"],["b"]],
            [["b"],["a"]] ]

h_ttt = ttt.History(eventos, gamma=0.0, sigma = 3.0, beta=1.0)
h_ttt.convergence()
h_ttt.learning_curves()


h = Historia(eventos)
h.forward_propagation()
h.backward_propagation()
h.posteriors()
h.habilidades


assert h_ttt.learning_curves()['b'][-1][1].isapprox(h.posteriors()[1][0][0])
