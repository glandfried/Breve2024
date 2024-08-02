using CSV
using Dates
using DataFrames
using TrueSkillThroughTime
global const ttt = TrueSkillThroughTime


df_atp = CSV.read("data/df_atp.csv", DataFrame,stringtype=String)

composition = [ [[row["Winner"]],[row["Loser"]]]  for row in eachrow(df_atp)]

times = [ Dates.value(row["Date"]-Date("1900-01-01"))  for row in eachrow(df_atp)]

composition = [ [[row["Winner"]],[row["Loser"]]]  for row in eachrow(df_atp)]

h = ttt.History(composition=composition, times = times, online=true, iterations=4, sigma=2.5)

ttt.learning_curves(h)


df = DataFrame()
df.winner = [ ev.teams[1].items[1].agent for b in h.batches for ev in b.events]
df.loser = [ ev.teams[2].items[1].agent for b in h.batches for ev in b.events]
df.m_winner = [ b.skills[ev.teams[1].items[1].agent].online.mu for b in h.batches for ev in b.events]
df.m_loser = [ b.skills[ev.teams[2].items[1].agent].online.mu for b in h.batches for ev in b.events]
df.s_winner = [ b.skills[ev.teams[1].items[1].agent].online.sigma for b in h.batches for ev in b.events]
df.s_loser = [ b.skills[ev.teams[2].items[1].agent].online.sigma for b in h.batches for ev in b.events]
df.b_winner = df_atp.PSW # Pinnacle
df.b_loser = df_atp.PSL # Pinnacle

CSV.write("inferencia/predicciones.csv", df; header=true)
