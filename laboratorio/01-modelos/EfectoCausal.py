from linear_model import BayesianLinearModel # https://github.com/glandfried/bayesian-linear-model
import random
import numpy as np
def numpy_float(numpy_scalar_in_array):
    return np.squeeze(numpy_scalar_in_array)

from numpy.random import normal as noise
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as normal
from scipy.stats import norm
import statsmodels.api as sm
import time
#phi = polynomial_basis_function

random.seed(2023-7-7)
np.random.seed(2023-7-7)
cmap = plt.get_cmap("tab10")
N = 1000


# Modelo Generativo
#
z1 = np.random.uniform(-3,3, size=N)
w1 = 3*z1 + np.random.normal(size=N,scale=1)
z2 = np.random.uniform(-3,3, size=N)
w2 = 2*z2 + np.random.normal(size=N,scale=1)
z3 = -2*z1 + 2*z2 + np.random.normal(size=N,scale=1)
x = -1*w1 + 2*z3 + np.random.normal(size=N,scale=1)
w3 = 2*x + np.random.normal(size=N,scale=1)
y = 2 - 1*w3 - z3 + w2 + np.random.normal(size=N,scale=1)

# Variable de control
#
PHI = np.concatenate([np.ones(N).reshape(N, 1), x.reshape(N, 1), z3.reshape(N, 1),w2.reshape(N, 1)], axis=1)

# Inferencia
#
blm= BayesianLinearModel(basis=lambda x: x)
blm.update(PHI, y.reshape(N,1) )


# Figura
#
nombre_params=["c0","c1","c2","c3"]
mean = blm.location
cov = blm.dispersion
real = [2,-2,-1,1]
#
plt.xticks(ticks=real)
ax = plt.gca()
a3x.tick_params(axis='both', labelsize=20)
handles = []  # List to store legend handles
labels = []  # List to store legend labels
for i in range(len(mean)):# i = 0
    mean_i = numpy_float(mean[i])
    cov_ii = numpy_float(cov[i,i])
    a = mean_i-10*np.sqrt(cov_ii)
    b = mean_i+10*np.sqrt(cov_ii)
    grilla = np.arange(a,b,step=(b-a)/200)
    line = plt.plot(grilla ,normal(mean=mean_i, cov=cov_ii ).pdf(grilla), linewidth=2,color=cmap(i))
    plt.axvline(x=real[i], color=cmap(i), linewidth=0.3)
    handles.append(line[0])  # Add the line as a handle
    labels.append(f'Label {i+1}')  # Add the label for the legend

plt.legend(handles, nombre_params,  bbox_to_anchor=pos)  # Create the legend with handles and labels
plt.show()

