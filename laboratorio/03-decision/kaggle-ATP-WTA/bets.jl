using CSV
using Dates
using DataFrames
using TrueSkillThroughTime
using Plots
global const ttt = TrueSkillThroughTime

# Datos
#     - winner: nombre
#     - loser: nombre
#     - m_winner: estimación media (online)
#     - m_loser: estimación media (online)
#     - s_winner: incertidumbre estimación (online)
#     - s_loser: incertidumbre estimación (online)
#     - b_winner: pago por ganador
#     - b_loser: pago por perdedor
df = CSV.read("predicciones.csv", DataFrame,stringtype=String)

# Criterio Kelly
#
function kelly(odd_w,p_w)
    x = odd_w - 1
    return (x*p_w - (1-p_w))/x
    ## Forma alternativa
    # (pQ-1)/(Q-1)
end

# Laverage Kelly
#
function laverage(odds=[2.0,2.0],p=0.5)
    #
    # Devuelve:
    #     El laverage kelly (positivo cara, negativo sello)
    #
    # Pagos de cara Qc y sello Qs
    Qc, Qs = odds
    lc = kelly(Qc,p)      # laverage para cara
    ls = kelly(Qs,(1-p))  # laverage para sello
    # Si ambos convienen
    if (lc > 0.01) & (ls > 0.01)
        if lc > ls
            return lc
        else
            return -ls
        end
    # Si solo cara conviene
    elseif (lc > 0.0001)
        return lc
    # Si solo sello conviene
    elseif (ls > 0.0001)
        return -ls
    # Si ninguno conviene
    else
        return 0
    end
end

# Simulación de apuestas
#
# Riquezas por criterio
wK = 0 # wK = log_riqueza_con_criterio_kelly
wF = 0 # wF = log_riqueza_con_fractional_kelly
wD = 0 # wD = log_riqueza_con_dynamic_fractional_kelly
wB = 0 # wB = log_riqueza_con_diversified_kelly
wP = 0 # wP = log_riqueza_con_diversificacion
#
# Otras variables
ll = [] # Lista de laverages
f = 0.2 # La fracción que será usando en Fractional Kelly
#
# Itera por partida en la secuencia del tiempo
for r in eachrow(df)#r= df[1,:]
    #
    # Si no hay registrado pagos de la casa de apuestas no hacemos nada.
    if (r.b_winner !== missing)
        #
        # Los pagos que ofrece la casa de apuestas.
        odds = [r.b_winner,r.b_loser]
        #
        # La predicción que hacemos (usando Game de TrueSkillThroughTime)
        p = ttt.Game(
            [[ttt.Player(prior=ttt.Gaussian(r.m_winner, r.s_winner))],[ttt.Player(prior=ttt.Gaussian(r.m_loser, r.s_loser))]]
        ).evidence
        #
        # Laverage Kelly dado pagos y predicción
        l = laverage(odds,p)
        #
        # Actualización de la riqueza
        if l > 0 # Si kelly apuesta al que finalmente gana
            wK = wK + log(  1-l     +    l *r.b_winner )
            wF = wF + log(  1-(l*f) + (f*l)*r.b_winner )
            wD = wD + log(  1-(l*p) + (p*l)*r.b_winner )
        end
        if l < 0 # Si kelly apuesta al que finalmente pierde
            wK = wK + log(1-abs(l))
            wF = wF + log(1-abs(l)*f)
            wD = wD + log(1-abs(l)*(p))
        end
        wB = wB + log(  1-abs(l)     + p*abs(l)*r.b_winner )
        wP = wP + log( p*r.b_winner )
    end
end


wP # -2798
wK # -1979
wB #  40
wF #  367
wD #  2082

#CSV.write("riqueza.csv", w; header=true)
