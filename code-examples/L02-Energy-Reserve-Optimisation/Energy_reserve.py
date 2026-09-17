# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:14:48 2024

@author: farahman
"""

from pyomo.environ import *

# Define model
model = ConcreteModel()

# Sets
model.i = Set(initialize=['A', 'B'], doc='Power producers')

# Scalars
DE = 120  # System demand
DR = 20   # Reserve requirement

# Parameters
model.Crat = 200  # Rationing cost

model.CE = Param(
    model.i,
    initialize={'A': 10, 'B': 30},
    doc='Marginal cost of energy production'
)

model.CR = Param(
    model.i,
    initialize={'A': 0, 'B': 5},
    doc='Reserve capacity offer costs'
)

model.Pmax = Param(
    model.i,
    initialize={'A': 100, 'B': 40},
    doc='Capacity limit'
)

# Variables
model.P = Var(model.i, domain=NonNegativeReals,
              doc='Energy dispatch')

model.R = Var(model.i, domain=NonNegativeReals,
              doc='Reserve capacity dispatch')

model.P_rat = Var(domain=NonNegativeReals,
                  doc='Rationing power')


# Objective function
def objective_rule(model):
    return (
        sum(
            model.CE[i] * model.P[i]
            + model.CR[i] * model.R[i]
            for i in model.i
        )
        + model.Crat * model.P_rat
    )

model.Obj = Objective(
    rule=objective_rule,
    sense=minimize
)


# Power balance
def power_balance_rule(model):
    return sum(model.P[i] for i in model.i) + model.P_rat == DE

model.PB = Constraint(rule=power_balance_rule)


# Reserve requirement
def reserve_requirement_rule(model):
    return sum(model.R[i] for i in model.i) >= DR

model.RR = Constraint(rule=reserve_requirement_rule)


# Capacity limits
def capacity_limit_rule(model, i):
    return model.P[i] + model.R[i] <= model.Pmax[i]

model.CL = Constraint(model.i, rule=capacity_limit_rule)


# Rationing constraint
def rationing_constraint_rule(model):
    return model.P_rat <= DE

model.RAT = Constraint(rule=rationing_constraint_rule)


# ---------------------------------------------------------
# IMPORT DUAL VALUES FROM SOLVER
# ---------------------------------------------------------
model.dual = Suffix(direction=Suffix.IMPORT)


# Solve
solver = SolverFactory('gurobi')
result = solver.solve(model, tee=True)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
print('\nOptimal Solution:')
print('-----------------------------')

for i in model.i:
    print(f'Energy Dispatch {i}:  {value(model.P[i]):.2f} MW')
    print(f'Reserve Capacity {i}: {value(model.R[i]):.2f} MW')

print(f'Rationing Power: {value(model.P_rat):.2f} MW')
print(f'Total System Operation Cost: {value(model.Obj):.2f}')


# ---------------------------------------------------------
# DUAL VALUES / SHADOW PRICES
# ---------------------------------------------------------
print('\nDual Values / Shadow Prices:')
print('-----------------------------')

print(f'Power Balance Dual:       {model.dual[model.PB]:.2f} €/MWh')
print(f'Reserve Requirement Dual: {model.dual[model.RR]:.2f} €/MW')