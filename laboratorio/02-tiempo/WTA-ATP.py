import pandas as pd
import numpy as np
from datetime import datetime
import trueskillthroughtime as ttt

# Cargar las historias del tennis WTA y ATP
# #
# Si en nuestras bases de datos hubieran personas en común
# podríamos crear una única historia unificada y comparar
# las habilidades entre los géneros.
#
df_wta = pd.read_csv("data/df_wta.csv", dtype=str) # Tennis Femenino
df_wta = df_wta.dropna(subset=['Date']) # 2 datos molestos
df_atp = pd.read_csv("data/df_atp.csv", dtype=str) # Tennis Masculino

# Composición
#
composition_wta = [ [[row["Winner"]], [row["Loser"]]] for index, row in df_wta.iterrows()]
composition_atp = [ [[row["Winner"]], [row["Loser"]]] for index, row in df_atp.iterrows()]

# Tiempos
#
times_wta = [(datetime.strptime(row["Date"], "%Y-%m-%d") - datetime(1900, 1, 1)).days for index, row in df_wta.iterrows()]
times_atp = [(datetime.strptime(row["Date"], "%Y-%m-%d") - datetime(1900, 1, 1)).days for index, row in df_atp.iterrows()]

# Historia e inferencia parcial
# #
# ttt.History() crea la red bayesiana y realiza una
# propagación forward (posterior[t] -> prior[t+1])
#
h_wta = ttt.History(composition=composition_wta, times=times_wta, sigma=2.5)
h_atp = ttt.History(composition=composition_atp, times=times_atp, sigma=2.5)

# Inferencia.
# #
# Iteraciones backward-forward para la propagación de la
# información por toda la red histórica de eventos.
#
# (posterior_forward[t] -> prior[t+1] <- posterior_backward[t+2])
#
h_wta.convergence(iterations=10)
h_atp.convergence(iterations=10)

# Curvas de aprendizaje
#
habilidades_wta = h_wta.learning_curves()
habilidades_atp = h_atp.learning_curves()

# Ranking
#
def estadisticos(habilidades):
    res = []
    for jugadora in habilidades:
        mus = [ h.mu for t, h in habilidades[jugadora]]
        res.append((
            jugadora
            ,len(mus)
            ,np.max(mus)
            #np.percentile(mus, 75),  # Tercer cuartil (75th percentile)
            #np.percentile(mus, 50),  # Segundo cuartil (50th percentile, también conocido como la mediana)
            #np.percentile(mus, 25),  # Primer cuartil (25th percentile)
            #np.min(mus)
        ))
    return res

ranking_wta = sorted(estadisticos(habilidades_wta), key= lambda tupla: tupla[1], reverse=True)
ranking_atp = sorted(estadisticos(habilidades_atp), key= lambda tupla: tupla[1], reverse=True)

import matplotlib.pyplot as plt; cmap = plt.get_cmap("tab10")

# Figura
#
habilidades = habilidades_atp
ranking = ranking_atp

for i in range(10):
    jugadora = ranking[i][0]
    ts = [datetime(1900, 1, 1) + pd.to_timedelta(t, unit='D') for t, h in habilidades[jugadora]]
    mus = [h.mu for t, h in habilidades[jugadora]]
    mus_lower = [h.mu - h.sigma for t, h in habilidades[jugadora]]
    mus_upper = [h.mu + h.sigma for t, h in habilidades[jugadora]]
    plt.plot(ts, mus, color = cmap(i))
    plt.fill_between(ts, mus_lower, mus_upper,
         color = cmap(i), alpha=0.2, label='Intervalo de credibilidad 95%')

plt.ylim((0,7))
plt.show()

