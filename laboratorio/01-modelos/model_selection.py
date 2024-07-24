import random
import numpy as np
from numpy.random import normal as noise 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as normal
from scipy.stats import norm 
import statsmodels.api as sm
import copy
#phi = polynomial_basis_function
from statsmodels.api import OLS # Para selección de hipótesis
import ModeloLineal as ml       # Para evaluación de hipótesis
import pandas as pd

cmap = plt.get_cmap("tab10")

np.random.seed(1) # Para reproducir los datos

N = 20 # Cantidad de datos
D = 10 # Cantidad de modelos (de 0 al 9)

BETA = (1/0.04)  # Precisión de los datos, el inverso de su varianza
ALPHA = (10e-6) # Precisión de la creencia a prior, el inverso de su varianza

# Realidad causal subyacente
def realidad_causal_subyacente(X, beta =  BETA):
    return np.sin(2 * np.pi * X) + np.random.normal(0,np.sqrt(1/beta),X.shape)

def modelo_causal_deterministico(X, H):
    y = H[0]*X**0
    for d in range(1,len(H)):
        y += H[d]*X**d
    return y

# Data
X = np.random.rand(N,1)-0.5
Y =  realidad_causal_subyacente(X)

# Grilla
X_grilla = np.linspace(0, 1, 100).reshape(-1, 1)-0.5
Y_grilla = realidad_causal_subyacente(X_grilla,np.inf )



# Figura de
# la función objetivo
plt.plot(X_grilla, Y_grilla, '--', color="black")
# y los datos
plt.plot(X,Y,'.', color='black')
plt.ylim(-1.5,1.5)
plt.close()



# Las transformaciones de X que hace el modelo Md de complejidad d
def phi(X, complejidad = D):
    return(pd.DataFrame({f'X{d}': X[:, 0]**d for d in range(complejidad+1)}))

# Itero por modelos Md
modelos_OLS = []
for d in range(D):
    # Ajusto el modelo de compeljidad d
    modelos_OLS.append(OLS(Y, phi(X,d)).fit())

# Figura de los ajustes.
plt.plot(X_grilla, Y_grilla, '--', color="black")
plt.plot(X,y,'.', color='black')
plt.ylim(-1.5,1.5)
for d in range(D):
    plt.plot(X_grilla,
             modelo_causal_deterministico(X_grilla, modelos_OLS[d].params),
             color=cmap(d), label= f'Modelo {d}' )

plt.legend(ncol=2)
plt.close()

from scipy.stats import norm

def prediccion(y,x,H):
    return norm(loc=modelo_causal_deterministico(x, H),
             scale=np.sqrt(1/BETA)).pdf(y)[0]


log_evidencia_OLS = [0 for _ in range(D)]
modelos_OLS = [OLS(Y[0:1], phi(X[0:1],d)).fit() for d in range(D)]
# Itera sobre los datos
for i in range(1,N):
    x = X[i]; y = Y[i] # Siguiente dato observado
    # Itera sobre los modelos
    for d in range(D):
        # Hipótesis de máxima verosimilitud
        H = modelos_OLS[d].params
        log_evidencia_OLS[d] += np.log(prediccion(y,x,H))
        modelos_OLS[d] = OLS(Y[0:i+1], phi(X[0:i+1],d)).fit()


for d in range(D):
    plt.bar(d, np.exp(log_evidencia_OLS[d]), align='center', color=cmap(d), label=f'Modelo {d}')

#plt.show()
plt.close()


import ModeloLineal as ml
from scipy.stats import multivariate_normal

modelos_MAP = []
for d in range(D):
    MU_d, COV_d = ml.posterior(Y,phi(X, complejidad = d), alpha=ALPHA*100)
    modelos_MAP.append({"mean":MU_d.reshape(1,d+1)[0], "cov":COV_d})



# Figura de los ajustes.
plt.plot(X_grilla, Y_grilla, '--', color="black")
plt.plot(X,y,'.', color='black')
plt.ylim(-1.5,1.5)
for d in range(D):
    plt.plot(X_grilla,
             modelo_causal_deterministico(X_grilla, modelos_MAP[d]["mean"]),
             color=cmap(d), label= f'Modelo {d}' )

plt.legend(ncol=2)
#plt.show()



modelos_MAP = []
for d in range(D):
    MU_d, COV_d = ml.posterior(Y[0:1],phi(X[0:1], complejidad = d), alpha=ALPHA*10)
    modelos_MAP.append({"mean":MU_d.reshape(1,d+1)[0], "cov":COV_d})

log_evidencia_MAP = [0 for _ in range(D)]
# Itera sobre los datos
for i in range(1,N):
    x = X[i]; y = Y[i] # Siguiente dato observado
    # Itera sobre los modelos
    for d in range(D):
        # Hipótesis de máxima verosimilitud
        H = modelos_MAP[d]["mean"]
        log_evidencia_MAP[d] += np.log(prediccion(y,x,H))
        MU_d, COV_d = ml.posterior(Y[0:d],phi(X[0:d], complejidad = d), alpha=ALPHA*10)
        modelos_MAP[d] = {"mean":MU_d.reshape(1,d+1)[0], "cov":COV_d}


for d in range(D):
    plt.bar(d, np.exp(log_evidencia_MAP[d]), align='center', color=cmap(d), label=f'Modelo {d}')



log_evidence_d = []
for d in range(D):
    log_evidence_d.append(ml.log_evidence(Y, phi(X,d))[0][0])


for d in range(D):
    plt.bar(d, np.exp(log_evidence_d[d]) /sum(np.exp(log_evidence_d)), align='center', color=cmap(d), label=f'Modelo {d}')


plt.close()


modelos_MAP = []
for d in range(D):
    MU_d, COV_d = ml.posterior(Y,phi(X, complejidad = d), alpha=ALPHA)
    modelos_MAP.append({"mean":MU_d.reshape(1,d+1)[0], "cov":COV_d})

plt.plot(X_grilla, Y_grilla, '--', color="black")
plt.plot(X,y,'.', color='black')
plt.ylim(-1.5,1.5)
for d in range(D):
    plt.plot(X_grilla,
             modelo_causal_deterministico(X_grilla, modelos_MAP[d]["mean"]),
             color=cmap(d), label= f'Modelo {d}' )

plt.axvline(-0.23, linestyle="--", color = "gray", alpha=0.3)
plt.legend(ncol=2)
plt.show()



x_new = np.array([[-0.23]])
y_range = np.arange(-2.5,0.5,0.01).reshape(1,300)

py_xnewDatosMd = []
for d in range(D):
    pred = ml.predictive(t_posteriori=y_range, Phi_posteriori = np.matrix(phi(x_new,d)), alpha=ALPHA, beta=BETA, t_priori= Y[0:4], Phi_priori = np.matrix(phi(X[0:4],d)))
    py_xnewDatosMd.append(pred)



plt.plot(y_range[0,:], py_xnewDatosMd[0], color=cmap(0), label = "Rígido (grado 0)")
plt.plot(y_range[0,:], py_xnewDatosMd[3], color=cmap(3), label = "Simple (grado 3)")
plt.plot(y_range[0,:], py_xnewDatosMd[9], color=cmap(9), label = "Complejo (grado 9)")
plt.xlabel("y|x=-0.23")
plt.ylabel("P(Datos | Modelo)")
plt.legend()
plt.show()


N, D = Phi_posteriori.shape
if t_priori is None: t_priori, Phi_priori = np.zeros((0,1)), np.zeros((0,D))
    m_prior, S_prior = posterior(t_priori, Phi_priori, alpha, beta)
    Phi_posteriori.dot(S_prior.dot(Phi_posteriori.T))
    sigma2 = Phi_posteriori.dot(S_prior.dot(Phi_posteriori.T)) + (1/beta)*np.eye(Phi_posteriori.shape[0])
    mu = Phi_posteriori.dot(m_prior) # m_N.T.dot(Phi)



MU_3, COV_3 = ml.posterior(Y,phi(X, complejidad = ))
MU_3, COV_3 = ml.posterior(Y,phi(X, complejidad = 3))
















prior_predictive_online = np.zeros((10,1))
prior_maxAposteriori_online = np.zeros((10,1))
mean_square_error_MAP = np.zeros((10,1))
prior_predictive_joint = np.zeros((10,1)) 
log_evidence_joint = np.zeros((10,1)) 
w_map = []
maxAposteriori = []
maxApriori = []

def fit(alpha):
    for d in range(10):#d=0
        for i in range(N) :#i=2
            X_priori = X[:i]
            t_priori = t[:i]
            x_posterior = X[i]
            t_posteriori = t[i]
            # Design matrix of training observations
            Phi_priori =  polynomial_basis_function(X_priori, np.array(range(d+1)) )
            Phi_posteriori = polynomial_basis_function(x_posterior , np.array(range(d+1)))
            Phi_posteriori = Phi_posteriori.reshape((1,d+1))
            
            prior_predictive_online[d,0] += np.log(predictive(t_posteriori, Phi_posteriori, beta, alpha, t_priori, Phi_priori ))
            
            ## Otros indicadores.
            w_map_prior = posterior(alpha, beta, t_priori, Phi_priori)[0]
            if N >= 10:
                # Usamos los primeros 10 como entrenamiento
                prior_maxAposteriori_online[d,0] += np.log(likelihood(w_map_prior, t_posteriori, Phi_posteriori , beta))
            mean_square_error_MAP[d,0] += (1/N) * ((t_posteriori - Phi_posteriori.dot(w_map_prior))**2)
            
                    
        Phi =  polynomial_basis_function(X, np.array(range(d+1)) )
        prior_predictive_joint[d,0] = np.log(predictive(t, Phi, beta, alpha ))
        log_evidence_joint[d,0] = log_evidence(t, Phi, beta, alpha)
        w_map.append(posterior(alpha, beta, t, Phi)[0]) 
        maxAposteriori.append(likelihood(w_map[d], t, Phi, beta))
        # Joint max_a_priori
        "Joint max a priori es igual para todas las complejidades (Es independiente de las cantidad de par\'ametros)"
        Phi_priori =  polynomial_basis_function(X[:0], np.array(range(d+1)) )
        t_priori =  t[:0]
        w_maxAprior = posterior(alpha, beta, t_priori, Phi_priori )[0]
        maxApriori.append(likelihood(w_maxAprior , t, Phi, beta)[0])





fit(alpha)
plt.close()
#plt.plot(prior_predictive_joint)
plt.plot(np.exp(log_evidence_joint))
plt.plot(np.exp(prior_predictive_online))
plt.close()
indices = np.arange(10)
total = sum(np.exp(log_evidence_joint))
cmap = plt.get_cmap("tab10")
for i in range(10):
    plt.bar(indices[i], np.exp(log_evidence_joint)[i]/total, align='center', color=cmap(i))


plt.xticks(ticks=indices)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=20)
plt.savefig("pdf/model_selection_evidence.pdf",bbox_inches='tight')
plt.savefig('png/model_selection_evidence.png', bbox_inches='tight',transparent=False)
plt.close()    

plt.close()
plt.xticks(ticks=indices)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=20)
total = sum(np.exp(prior_maxAposteriori_online))
for i in range(10):
    plt.bar(indices[i], np.exp(prior_maxAposteriori_online[i])/total, align='center', color=cmap(i))

#plt.plot(np.exp(prior_maxAposteriori_online))
plt.savefig("pdf/model_selection_maxApriori_online.pdf",bbox_inches='tight')
plt.savefig('png/model_selection_maxApriori_online.png', bbox_inches='tight',transparent=False)
plt.close()

plt.close()
total = sum(maxAposteriori)
plt.xticks(ticks=indices)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=20)
for i in range(10):
    plt.bar(indices[i], maxAposteriori[i]/total, align='center', color=cmap(i))


plt.savefig("pdf/model_selection_maxLikelihood.pdf",bbox_inches='tight')
plt.savefig("png/model_selection_maxLikelihood.png",bbox_inches='tight',transparent=False)
plt.close()


prior_predictive_online = np.zeros((10,1))
prior_maxAposteriori_online = np.zeros((10,1))
mean_square_error_MAP = np.zeros((10,1))
prior_predictive_joint = np.zeros((10,1)) 
log_evidence_joint = np.zeros((10,1)) 
w_map = []
maxAposteriori = []
maxApriori = []

fit(10**(-6))

for d in range(0,10):#d=3
    Phi_grilla = polynomial_basis_function(X_grilla, np.array(range(d+1)) )
    y_map = Phi_grilla.dot(w_map[d])
    plt.plot(X_grilla,y_map  )


plt.ylim(-1.5,1.5)
Phi_grilla = polynomial_basis_function(X_grilla, np.array(range(3+1)) )
y_map = Phi_grilla.dot(w_map[3])
plt.plot(X_grilla,y_map, color="red"  )
plt.plot(X_grilla, y_true, '--', color="black")
plt.plot(X,t,'.', color='black')
plt.savefig("pdf/model_selection_MAP_non-informative.pdf",bbox_inches='tight')
plt.savefig("png/model_selection_MAP_non-informative.png",bbox_inches='tight',transparent=False)
plt.close()    


prior_predictive_online = np.zeros((10,1))
prior_maxAposteriori_online = np.zeros((10,1))
mean_square_error_MAP = np.zeros((10,1))
prior_predictive_joint = np.zeros((10,1)) 
log_evidence_joint = np.zeros((10,1)) 
w_map = []
maxAposteriori = []
maxApriori = []

fit(10**(-1))

for d in range(0,10):#d=3
    Phi_grilla = polynomial_basis_function(X_grilla, np.array(range(d+1)) )
    y_map = Phi_grilla.dot(w_map[d])
    plt.plot(X_grilla,y_map  )


plt.ylim(-1.5,1.5)
Phi_grilla = polynomial_basis_function(X_grilla, np.array(range(3+1)) )
y_map = Phi_grilla.dot(w_map[3])
plt.plot(X_grilla,y_map, color="red"  )
plt.plot(X_grilla, y_true, '--', color="black")
plt.plot(X,t,'.', color='black')
plt.savefig("pdf/model_selection_MAP_informative.pdf",bbox_inches='tight')
plt.savefig("png/model_selection_MAP_informative.png",bbox_inches='tight',transparent=False)
plt.close()    



###
# El prior es regularizador, pero en el sentido contrario: un prior poco informativo favorece a los modelos de menor grado.


prior_predictive_online = np.zeros((10,1))
prior_maxAposteriori_online = np.zeros((10,1))
mean_square_error_MAP = np.zeros((10,1))
prior_predictive_joint = np.zeros((10,1)) 
log_evidence_joint = np.zeros((10,1)) 
w_map = []
maxAposteriori = []
maxApriori = []

model_alpha = np.zeros((10,7))
for i in range(7):
    log_evidence_joint = np.zeros((10,1)) 
    print(i)
    fit(10**(-i))
    model_alpha[:,i] = (np.exp(log_evidence_joint)/sum(np.exp(log_evidence_joint))).reshape((10,))
    
for i in range(10):
    plt.plot(model_alpha[i,:], label=str(i) )


plt.legend(loc="upper left")
plt.xlabel("log10(Incertidumbre a priori)", size=18)
plt.ylabel("P(M | D)",size=18)
plt.savefig("pdf/regularizador.pdf",bbox_inches='tight')
plt.savefig("png/regularizador.png",bbox_inches='tight',transparent=False)
plt.close()    



"""
i=16
belief = np.zeros((len(y_grilla),len(X_grilla),10))
for d in range(10):#d=1
    Phi =  polynomial_basis_function(X[:i], np.array(range(d+1)) )
    for ix in range(len(X_grilla)):
        xi = X_grilla[ix]
        Phi_x = polynomial_basis_function(xi, np.array(range(d+1)))
        Phi_x = Phi_x.reshape((1,d+1))
        belief[:,ix,d] =  predictive(y_grilla, Phi_x, beta, alpha, t[:i], Phi)[::-1] 

max_dens = max(np.max(belief[:,:,3]),np.max(belief[:,:,9]))

plt.imshow(belief[:,:,3],vmin=0, vmax=max_dens)
plt.imshow(belief[:,:,4],vmin=0, vmax=max_dens)

plt.plot(y_grilla,np.flip(belief[:,0,3]))
plt.plot(y_grilla,np.flip(belief[:,0,9]))
plt.savefig("model_selection_evidence_3_9_at_0.pdf",bbox_inches='tight')
plt.close()    
"""
