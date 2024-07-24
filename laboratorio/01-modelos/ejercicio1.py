from scipy.stats import multinomial
import numpy as np
from math import log, sqrt
import pandas as pd
from matplotlib import pyplot as plt

# Probabilidad del regalo
pr = [1/3, 1/3, 1/3] # P(r) = pr[r]
# Probabilidad de cerrar
pc = [1/3, 1/3, 1/3] # P(c) = pc[c]
# Probabilidad de la pista
# P(s|r,M0) = psA[r][s] # No Monty Hall
ps_rM0=[[  0, 1/2, 1/2],   # r=1
        [1/2,   0, 1/2],   # r=2
        [1/2, 1/2,   0]]   # r=3
# P(s|r,c,M1) = psB[c][r][s] # Monty Hall
ps_rcM1=[[[  0, 1/2, 1/2],  # r=1, c=1
          [  0,   0,   1],  # r=2, c=1
          [  0,   1,   0]], # r=3, c=1
         [[  0,   0,   1],  # r=1, c=2
          [1/2,   0, 1/2],  # r=2, c=2
          [  1,   0,   0]], # r=3, c=2
         [[  0,   1,   0],  # r=1, c=3
          [  1,   0,   0],  # r=2, c=3
          [1/2, 1/2,   0]]] # r=3, c=3
# Probabilidad del modelo
# P(m) = pModelo[m]
pModelo = [1/2, 1/2]

# 1.1.2 Simulación de Datos

T = 16 # Cantidad total de episodios
Datos = []
for t in range(T):
    r = np.random.choice(3, p=pr)
    c = np.random.choice(3, p=pc)
    s = np.random.choice(3, p=ps_rcM1[c][r])
    Datos.append((c,s,r))



# Predicciones de los modelos
pDatos_M0 = [1] # Del modelo 0: No Monty Hall
pDatos_M1 = [1] # Del modelo 1: Monty Hall
tiempo = range(16+1)
for _ in tiempo[1:]:
    # Simulaciones
    r = np.argmax(multinomial.rvs(1, pr))
    c = np.argmax(multinomial.rvs(1, pc))
    s = np.argmax(multinomial.rvs(1, ps_rcM1[c][r]))
    # Predicciones
    # P(r,c,s|M) = P(r)P(c)P(s|r,c)
    pDatos_M0.append(      pr[r]
                         * pc[c]
                         * ps_rM0[r][s] )
    pDatos_M1.append(      pr[r]
                         * pc[c]
                         * ps_rcM1[c][r][s] )

pDatos = np.cumprod(pDatos_M0) * pModelo[0] \
       + np.cumprod(pDatos_M1) * pModelo[1]

pM0_Datos = np.cumprod(pDatos_M0) * pModelo[0] / pDatos
pM1_Datos = np.cumprod(pDatos_M1) * pModelo[1] / pDatos

plt.plot(pM0_Datos , label="M0: No Monty Hall")
plt.plot(pM1_Datos , label="M1: Monty Hall")
plt.legend(title="Realidad causal: Monty Hall \n \nProbabilidad de los modelos:")


# Ejercicio 2
import pandas as pd
import numpy as np
import ModeloLineal as ml
import statsmodels.api as sm
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")

Alturas = pd.read_csv("datos/alturas.csv")

plt.scatter(Alturas.altura_madre[Alturas.sexo=="M"], Alturas.altura[Alturas.sexo=="M"], label="Masculinos")
plt.scatter(Alturas.altura_madre[Alturas.sexo=="F"],Alturas.altura[Alturas.sexo=="F"], label="Fememinos")
plt.xlabel("Altura madre")
plt.ylabel("Altura")
plt.legend()
#plt.show()

N, _ = Alturas.shape
y = Alturas.altura
x = pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                  "Altura": Alturas.altura_madre,  # Pendiente
                  "Genero": (Alturas.sexo=="M")+0     # Genero
             })

model = sm.OLS(Alturas.altura,x).fit()

X = np.arange(min(Alturas.altura_madre),max(Alturas.altura_madre), step=0.01)

Predicion_Altura_M = model.params.Base + model.params.Altura*X + model.params.Genero
Predicion_Altura_F = model.params.Base + model.params.Altura*X

plt.scatter(Alturas.altura_madre[Alturas.sexo=="M"],
            Alturas.altura[Alturas.sexo=="M"],
            label="Masculino", color=cmap(0))
plt.plot(X, Predicion_Altura_M , color=cmap(0))
plt.scatter(Alturas.altura_madre[Alturas.sexo=="F"],
            Alturas.altura[Alturas.sexo=="F"],
            label="Femenino", color=cmap(1))
plt.plot(X, Predicion_Altura_F , color=cmap(1))
plt.xlabel("Altura madre")
plt.ylabel("Altura")
plt.legend()
#plt.show()
plt.close()

m_N, S_N = ml.posterior(y, x)
log_pDatos_Modelo2  = ml.log_evidence(y, x)

x1 = pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                  "Altura": Alturas.altura_madre,  # Pendiente
             })

x25 = {"Base": [1 for _ in range(N)],    # Origen
      "Altura": Alturas.altura_madre  # Pendiente
     }
for i in range(25):
    x25[f'id{i}'] = [ ((j % 25) == i)+0 for j in range(N)]

x25 = pd.DataFrame(x25)


log_pDatos_Modelo1  = ml.log_evidence(y, x1)
log_pDatos_Modelo2  = ml.log_evidence(y, x)
log_pDatos_Modelo25  = ml.log_evidence(y, x25)

