# -*- coding: utf-8 -*-
"""
Created on Fri June 17 15:19:13 2022

@author: Farahmand

Title: Stochastic Market Clearing

In this code, I have implemented the formulation as discussed during the lecture on 
"Stochastic market clearing (two-stage stochastic programming)." 
For a detailed explanation of the formulation, please refer to slide #24 of the presentation. 
To enhance the results and introduce a greater degree of challenge, consider adding a penalty 
to the objective function for wind spillage. This modification can lead to more dynamic and insightful outcomes.
"""

from pyomo.environ import *

# Define model
model = ConcreteModel()

# Sets
model.k = Set(initialize=['k1', 'k2'], doc='Producers')
model.s = Set(initialize=['s1', 's2', 's3', 's4'], doc='Scenarios')
inflexible_generators = ['k1']  # List of inflexible generators

# Parameters
model.C = Param(model.k, initialize={'k1': 10, 'k2': 30}, doc='Production cost per unit')
model.Cu = Param(model.k, initialize={'k1': 0, 'k2': 25}, doc='Upward reserve cost per unit')
model.Cd = Param(model.k, initialize={'k1': 0, 'k2': 10}, doc='Downward reserve cost per unit')
model.Pmax = Param(model.k, initialize={'k1': 100, 'k2': 50}, doc='Maximum production')
model.Wscmax = 40  # Maximum wind capacity
model.L = 120  # Load
model.pi = Param(model.s, initialize={'s1': 0.25, 's2': 0.25, 's3': 0.25, 's4': 0.25}, doc='Scenario probability')
model.w = Param(model.s, initialize={'s1': 30, 's2': 40, 's3': 80, 's4': 5}, doc='Wind power production scenario')
model.vLOL = 200  # Value of lost load

# Variables
model.z = Var()  # Objective function
model.P = Var(model.k, domain=NonNegativeReals, doc='Energy dispatch')
model.Ru = Var(model.k, domain=NonNegativeReals, doc='Upward reserve dispatch')
model.Rd = Var(model.k, domain=NonNegativeReals, doc='Downward reserve dispatch')
model.Pu = Var(model.k, model.s, domain=NonNegativeReals, doc='Upward energy redispatch')
model.Pd = Var(model.k, model.s, domain=NonNegativeReals, doc='Downward energy redispatch')
model.Wsc = Var(domain=NonNegativeReals, doc='Wind energy dispatch')
model.Lsh = Var(model.s, domain=NonNegativeReals, doc='Load shedding')
model.Wsp = Var(model.s, domain=NonNegativeReals, doc='Wind spillage')

# Objective function
# Separate first and second stage costs
def first_stage_cost_rule(model):
    return sum(model.C[k] * model.P[k] + model.Cu[k] * model.Ru[k] + model.Cd[k] * model.Rd[k] for k in model.k)
model.FirstStageCost = Expression(rule=first_stage_cost_rule, doc='First stage cost')

def second_stage_cost_rule(model):
    return sum(model.pi[s] * (sum(model.C[k] * (model.Pu[k, s] - model.Pd[k, s]) for k in model.k) + model.vLOL * model.Lsh[s]) for s in model.s)
model.SecondStageCost = Expression(rule=second_stage_cost_rule, doc='Second stage cost')

def objective_rule(model):
    return model.FirstStageCost + model.SecondStageCost
model.EC = Objective(rule=objective_rule, sense=minimize, doc='Expected cost (SP only)')

#-------------------------------------------------------------------------------
# First stage constraints

# Power balance equation at the first stage
def power_balance_rule(model):
    return sum(model.P[k] for k in model.k) + model.Wsc == model.L
model.bal0 = Constraint(rule=power_balance_rule, doc='Power balance equation at the forward stage')

# Maximum wind dispatch
def wind_max_rule(model):
    return model.Wsc <= model.Wscmax
model.Windmax = Constraint(rule=wind_max_rule, doc='Maximum wind dispatch')

# Dispatch and reserve upper limit
def reserve_upper_limit_rule(model, k):
    return model.P[k] + model.Ru[k] <= model.Pmax[k]
model.PRu = Constraint(model.k, rule=reserve_upper_limit_rule, doc='Dispatch and reserve upper limit')

# Dispatch and reserve lower limit
def reserve_lower_limit_rule(model, k):
    return model.P[k] - model.Rd[k] >= 0
model.PRl = Constraint(model.k, rule=reserve_lower_limit_rule, doc='Dispatch and reserve lower limit')

#-------------------------------------------------------------------------------
# Second stage constraints

# Power balance equation at the Second stage stage
def balancing_stage_rule(model, s):
    return sum(model.Pu[k, s] - model.Pd[k, s] for k in model.k) + model.Lsh[s] + model.w[s] - model.Wsc - model.Wsp[s] == 0
model.bal = Constraint(model.s, rule=balancing_stage_rule, doc='Power balance equation at the balancing stage')

# Upward energy redispatch upper limit
def upward_redispatch_rule(model, k, s):
    return model.Pu[k, s] <= model.Ru[k]
model.Puu = Constraint(model.k, model.s, rule=upward_redispatch_rule, doc='Upward energy redispatch upper limit')

# Downward energy redispatch upper limit
def downward_redispatch_rule(model, k, s):
    return model.Pd[k, s] <= model.Rd[k]
model.Pdu = Constraint(model.k, model.s, rule=downward_redispatch_rule, doc='Downward energy redispatch upper limit')

# Wind power spillage upper limit
def wind_spillage_rule(model, s):
    return model.Wsp[s] <= model.w[s]
model.Wspu = Constraint(model.s, rule=wind_spillage_rule, doc='Wind power spillage upper limit')

# Load shedding upper limit
def load_shedding_rule(model, s):
    return model.Lsh[s] <= model.L
model.Lshu = Constraint(model.s, rule=load_shedding_rule, doc='Load shedding upper limit')



# Fix Ru and Rd to zero for inflexible generators
for g in inflexible_generators:
    model.Ru[g].fix(0)
    model.Rd[g].fix(0)


#-------------------------------------------------------------------------------
# Solve the model
solver = SolverFactory('gurobi')
result = solver.solve(model, tee=True)
model.z= model.FirstStageCost + model.SecondStageCost

#-------------------------------------------------------------------------------
# Display results
# Displaying the final results
print(f"\n{'='*10} Optimal Solution - Objective Values {'='*10}")
print(f" First Stage Cost: {value(model.FirstStageCost):.2f}, Second Stage Cost: {value(model.SecondStageCost):.2f}, Total Cost: {value(model.EC):.2f}")

# Displaying first stage results in a clearer format
print(f"\n{'*'*10} First Stage Results {'*'*10}")
for k in model.k:
    print(f"Producer {k} -> Energy Dispatch (P): {model.P[k].value:.2f}, Upward Reserve (Ru): {model.Ru[k].value:.2f}, Downward Reserve (Rd): {model.Rd[k].value:.2f}")
print(f"Wind Energy Dispatch (Wsc): {model.Wsc.value:.2f}")

# Displaying second stage results in a clearer format
print(f"\n{'*'*10} Second Stage Results {'*'*10}")
for s in model.s:
    print(f"Scenario {s} -> Load Shedding (Lsh): {model.Lsh[s].value:.2f}, Wind Spillage (Wsp): {model.Wsp[s].value:.2f}")
    for k in model.k:
        print(f"  Producer {k} -> Upward Redispatch (Pu): {model.Pu[k, s].value:.2f}, Downward Redispatch (Pd): {model.Pd[k, s].value:.2f}")