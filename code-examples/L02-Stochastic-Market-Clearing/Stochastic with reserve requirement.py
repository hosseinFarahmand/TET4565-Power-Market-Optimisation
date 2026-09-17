# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:21:54 2025

@author: farahman
"""

# -*- coding: utf-8 -*-
"""
Stochastic Market Clearing with fixed reserve requirement (per slide)
"""

from pyomo.environ import *

# -----------------------
# Model & sets
# -----------------------
model = ConcreteModel()

model.k = Set(initialize=['k1', 'k2'], doc='Producers')
model.s = Set(initialize=['s1', 's2', 's3', 's4'], doc='Scenarios')

# Generators that cannot provide reserves (kept from your version)
inflexible_generators = ['k1']  # example: k1 has no reserve capability

# -----------------------
# Parameters
# -----------------------
model.C  = Param(model.k, initialize={'k1': 10, 'k2': 30}, doc='Energy cost c_k [€/MWh]')
model.Cu = Param(model.k, initialize={'k1': 0,  'k2': 25}, doc='Up reserve capacity cost c^u_k [€/MW]')
model.Cd = Param(model.k, initialize={'k1': 0,  'k2': 10}, doc='Down reserve capacity cost c^d_k [€/MW]')

model.Pmax = Param(model.k, initialize={'k1': 100, 'k2': 50}, doc='Capacity p̄_k [MW]')

# Wind & load
model.Wscmax = Param(initialize=40, doc='w̄: DA wind cap [MW]')
model.L      = Param(initialize=120, doc='Load l [MW]')
model.w      = Param(model.s, initialize={'s1': 30, 's2': 40, 's3': 80, 's4': 5},
                     doc='w_s^RT: RT wind [MW]')
model.pi     = Param(model.s, initialize={'s1': 0.25, 's2': 0.25, 's3': 0.25, 's4': 0.25},
                     doc='Scenario probability π_s')

# Fixed *system* reserve requirement  Σ_k R^u_k ≥ R   (choose value as needed)
model.Rreq   = Param(initialize=20, doc='System up-reserve requirement R [MW]')

# Cost of rationing p_s^rat (slide notation c^{rat})
model.c_rat  = Param(initialize=200, doc='Rationing cost c^{rat} [€/MWh]')

# -----------------------
# Variables
# -----------------------
# First stage (DA)
model.P  = Var(model.k, domain=NonNegativeReals, doc='p_k: DA energy [MW]')
model.Ru = Var(model.k, domain=NonNegativeReals, doc='R^u_k: up reserve cap [MW]')
model.Rd = Var(model.k, domain=NonNegativeReals, doc='R^d_k: down reserve cap [MW]')
model.Wsc = Var(domain=NonNegativeReals, doc='w^{DA}: DA wind [MW]')

# Second stage (RT)
model.Pu  = Var(model.k, model.s, domain=NonNegativeReals, doc='r^u_{k,s} [MW]')
model.Pd  = Var(model.k, model.s, domain=NonNegativeReals, doc='r^d_{k,s} [MW]')
model.Wsp = Var(model.s, domain=NonNegativeReals, doc='w^{SP}_s [MW]')
model.prat = Var(model.s, domain=NonNegativeReals, doc='p^{rat}_s (rationing) [MW]')

# -----------------------
# Objective: DA + E[RT]
# z = Σ_k (c_k p_k + c^u_k R^u_k + c^d_k R^d_k)
#     + Σ_s π_s ( Σ_k c_k (r^u_{k,s} - r^d_{k,s}) + c^{rat} p^{rat}_s )
# -----------------------
model.FirstStageCost = Expression(
    rule=lambda m: sum(m.C[k]*m.P[k] + m.Cu[k]*m.Ru[k] + m.Cd[k]*m.Rd[k] for k in m.k)
)

model.SecondStageCost = Expression(
    rule=lambda m: sum(
        m.pi[s]*( sum(m.C[k]*(m.Pu[k,s] - m.Pd[k,s]) for k in m.k)
                  + m.c_rat*m.prat[s] )
        for s in m.s
    )
)

model.EC = Objective(rule=lambda m: m.FirstStageCost + m.SecondStageCost,
                     sense=minimize,
                     doc='Expected total cost')

# -----------------------
# First-stage constraints (slide)
# -----------------------
# Σ_k p_k + w^{DA} = l
model.bal0 = Constraint(expr=sum(model.P[k] for k in model.k) + model.Wsc == model.L)

# w^{DA} ≤ w̄
model.Windmax = Constraint(expr=model.Wsc <= model.Wscmax)

# p_k + R^u_k ≤ p̄_k
model.PRu = Constraint(model.k, rule=lambda m,k: m.P[k] + m.Ru[k] <= m.Pmax[k])

# p_k - R^d_k ≥ 0
model.PRl = Constraint(model.k, rule=lambda m,k: m.P[k] - m.Rd[k] >= 0)

# Σ_k R^u_k ≥ R  (fixed reserve requirement)
model.SysReserve = Constraint(expr=sum(model.Ru[k] for k in model.k) >= model.Rreq)

# -----------------------
# Second-stage constraints (per scenario)
# -----------------------
# Σ_k (r^u_{k,s} - r^d_{k,s}) + p^{rat}_s + (w_s^RT - w^{DA} - w^{SP}_s) = 0
model.bal = Constraint(
    model.s,
    rule=lambda m,s: sum(m.Pu[k,s]-m.Pd[k,s] for k in m.k)
                     + m.prat[s] + (m.w[s] - m.Wsc - m.Wsp[s]) == 0
)

# r^u_{k,s} ≤ R^u_k
model.Puu = Constraint(model.k, model.s, rule=lambda m,k,s: m.Pu[k,s] <= m.Ru[k])

# r^d_{k,s} ≤ R^d_k
model.Pdu = Constraint(model.k, model.s, rule=lambda m,k,s: m.Pd[k,s] <= m.Rd[k])

# w^{SP}_s ≤ w^{RT}_s
model.Wspu = Constraint(model.s, rule=lambda m,s: m.Wsp[s] <= m.w[s])

# p^{rat}_s ≤ l
model.prat_u = Constraint(model.s, rule=lambda m,s: m.prat[s] <= m.L)

# -----------------------
# Inflexible generators: fix Ru=Rd=0 (kept)
# -----------------------
for g in inflexible_generators:
    model.Ru[g].fix(0)
    model.Rd[g].fix(0)

# -----------------------
# Solve
# -----------------------
solver = SolverFactory('gurobi')
result = solver.solve(model, tee=True)

# -----------------------
# Report
# -----------------------
print("\n========== Objective ==========")
print(f"First Stage: {value(model.FirstStageCost):.2f}")
print(f"Second Stage: {value(model.SecondStageCost):.2f}")
print(f"Total: {value(model.EC):.2f}")

print("\n********** First Stage **********")
for k in model.k:
    print(f"{k}: P={value(model.P[k]):.2f}, Ru={value(model.Ru[k]):.2f}, Rd={value(model.Rd[k]):.2f}")
print(f"W_DA={value(model.Wsc):.2f}")

print("\n********** Second Stage **********")
for s in model.s:
    print(f"{s}: p^rat={value(model.prat[s]):.2f}, Wsp={value(model.Wsp[s]):.2f}")
    for k in model.k:
        print(f"  {k}: r^u={value(model.Pu[k,s]):.2f}, r^d={value(model.Pd[k,s]):.2f}")
