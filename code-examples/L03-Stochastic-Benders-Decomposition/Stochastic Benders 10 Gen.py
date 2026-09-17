# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:32:42 2024

@author: farahman
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:23:47 2024

@author: Farahmand
"""

from pyomo.environ import *
from pyomo.opt import SolverFactory

# Global Variables for complicating variables in this case or first stage decision variables
P_fixed = {}
Ru_fixed = {}
Rd_fixed = {}
Wsc_fixed = 0

#-------------------------------------------------------------------------------
# Master Problem: First Stage Model
# Master problem is the same as the formulation showed in slide 60 of the lecture on
# "Decomposition techniques, Bender’s decomposition (theory and application)-BD
master = ConcreteModel()

# Sets
master.k = Set(initialize=['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10'])  # Producers

# Parameters
C = {'k1': 10, 'k2': 30, 'k3': 15, 'k4': 25, 'k5': 20, 'k6': 18, 'k7': 22, 'k8': 28, 'k9': 17, 'k10': 26}  # Production cost per unit
Cu = {'k1': 0, 'k2': 25, 'k3': 5, 'k4': 20, 'k5': 15, 'k6': 10, 'k7': 18, 'k8': 22, 'k9': 12, 'k10': 30}  # Upward reserve cost per unit
Cd = {'k1': 0, 'k2': 10, 'k3': 8, 'k4': 12, 'k5': 11, 'k6': 9, 'k7': 7, 'k8': 14, 'k9': 13, 'k10': 16}  # Downward reserve cost per unit
Pmax = {'k1': 100, 'k2': 50, 'k3': 80, 'k4': 60, 'k5': 90, 'k6': 70, 'k7': 110, 'k8': 65, 'k9': 85, 'k10': 75}  # Maximum production
Wscmax = 40  # Maximum wind capacity
L = 600  # Load (increased accordingly)
vLOL = 200  # Value of lost load

# Variables
master.P = Var(master.k, domain=NonNegativeReals, initialize=0)  # Energy dispatch
master.Ru = Var(master.k, domain=NonNegativeReals, initialize=0)  # Upward reserve dispatch
master.Rd = Var(master.k, domain=NonNegativeReals, initialize=0)  # Downward reserve dispatch
master.Wsc = Var(domain=NonNegativeReals, initialize=0)  # Wind energy dispatch
master.a = Var(domain=Reals, initialize=0)  # Expected cost
master.Ru['k1'] = 0

# Since k1 is Inflexible generator, we will fix Ru and Rd to zero
master.Ru['k1'].fix(0)
master.Rd['k1'].fix(0)

# Objective function
def master_objective_rule(master):
    return sum(C[k] * master.P[k] + Cu[k] * master.Ru[k] + Cd[k] * master.Rd[k] for k in master.k) + master.a
master.Objective = Objective(rule=master_objective_rule, sense=minimize)

# Constraints
# Power balance equation at the first stage
def master_bal0_rule(master):
    return sum(master.P[k] for k in master.k) + master.Wsc == L
master.bal0 = Constraint(rule=master_bal0_rule)

# Maximum wind dispatch
def master_windmax_rule(master):
    return master.Wsc <= Wscmax
master.Windmax = Constraint(rule=master_windmax_rule)

# Dispatch and reserve upper limit
def master_pru_rule(master, k):
    return master.P[k] + master.Ru[k] <= Pmax[k]
master.PRu = Constraint(master.k, rule=master_pru_rule)

# Dispatch and reserve lower limit
def master_prl_rule(master, k):
    return master.P[k] - master.Rd[k] >= 0
master.PRl = Constraint(master.k, rule=master_prl_rule)

# Lower bound for expected cost
def master_low_a_rule(master):
    return master.a >= -10000
master.Low_a = Constraint(rule=master_low_a_rule)


#-------------------------------------------------------------------------------
# Iteration Loop
# The tolerance value is set at 0.01, which may vary depending on the specific requirements and
# characteristics of different problems
opt = SolverFactory('gurobi')
cut_counter = 1
upper_bound = float('inf')
lower_bound = -float('inf')
tolerance = 0.01
iteration = 1

while abs(upper_bound - lower_bound) > tolerance:
    # Solving the master problem
    master_results = opt.solve(master, tee=True)

    # Update lower bound
    lower_bound =  master.a.value

    # Update global fixed values for subproblem
    P_fixed = {k: master.P[k].value for k in master.k}
    Ru_fixed = {k: master.Ru[k].value for k in master.k}
    Rd_fixed = {k: master.Rd[k].value for k in master.k}
    Wsc_fixed = master.Wsc.value

    # Displaying the lower bound results
    print(f"\nIteration {iteration} - lower bound: {master.a.value}")


 #-------------------------------------------------------------------------------
 # Sub-problem: Second Stage Model
 # Sub-problem is the same as the formulation showed in slide 58 of the lecture on BD

    sub = ConcreteModel()

    # Sets
    sub.k = Set(initialize=['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10'])  # Producers
    sub.s = Set(initialize=['s1', 's2', 's3', 's4'])  # Scenarios

    # Parameters
    pi = {'s1': 0.25, 's2': 0.25, 's3': 0.25, 's4': 0.25}  # Scenario probability
    w = {'s1': 30, 's2': 40, 's3': 80, 's4': 5}  # Wind power production scenario

    # Variables
    sub.Pu = Var(sub.k, sub.s, domain=NonNegativeReals, initialize=0)  # Upward energy redispatch
    sub.Pd = Var(sub.k, sub.s, domain=NonNegativeReals, initialize=0)  # Downward energy redispatch
    sub.Lsh = Var(sub.s, domain=NonNegativeReals, initialize=0)  # Load shedding
    sub.Wsp = Var(sub.s, domain=NonNegativeReals, initialize=0)  # Wind spillage
    sub.Ru = Var(sub.k, domain=NonNegativeReals, initialize=lambda sub, k: Ru_fixed[k])  # Upward reserve
    sub.Rd = Var(sub.k, domain=NonNegativeReals, initialize=lambda sub, k: Rd_fixed[k])  # Downward reserve
    sub.Wsc = Var(domain=NonNegativeReals, initialize=Wsc_fixed)  # Wind energy dispatch
    sub.P = Var(sub.k, domain=NonNegativeReals, initialize=lambda sub, k: P_fixed[k])  # Energy dispatch

    # Suffix to capture dual values
    sub.dual = Suffix(direction=Suffix.IMPORT)

    # Objective function for the subproblem
    def sub_objective_rule(sub):
        return sum(pi[s] * (sum(C[k] * (sub.Pu[k, s] - sub.Pd[k, s]) for k in sub.k) + vLOL * sub.Lsh[s]) for s in sub.s)
    sub.Objective = Objective(rule=sub_objective_rule, sense=minimize)

    # Constraints
    # Power balance equation at the balancing stage
    def sub_bal_rule(sub, s):
        return sum(sub.Pu[k, s] - sub.Pd[k, s] for k in sub.k) + sub.Lsh[s] + w[s] - sub.Wsc - sub.Wsp[s] == 0
    sub.bal = Constraint(sub.s, rule=sub_bal_rule)

    # Upward energy redispatch upper limit
    def sub_puu_rule(sub, k, s):
        return sub.Pu[k, s] <= sub.Ru[k]
    sub.Puu = Constraint(sub.k, sub.s, rule=sub_puu_rule)

    # Downward energy redispatch upper limit
    def sub_pdu_rule(sub, k, s):
        return sub.Pd[k, s] <= sub.Rd[k]
    sub.Pdu = Constraint(sub.k, sub.s, rule=sub_pdu_rule)

    # Wind power spillage upper limit
    def sub_wspu_rule(sub, s):
        return sub.Wsp[s] <= w[s]
    sub.Wspu = Constraint(sub.s, rule=sub_wspu_rule)

    # Load shedding upper limit
    def sub_lshu_rule(sub, s):
        return sub.Lsh[s] <= L
    sub.Lshu = Constraint(sub.s, rule=sub_lshu_rule)

    # Fix subproblem variables to master values to capture duals
    def sub_fix_ru_rule(sub, k):
        return sub.Ru[k] == Ru_fixed[k]
    sub.fix_ru = Constraint(sub.k, rule=sub_fix_ru_rule)

    def sub_fix_rd_rule(sub, k):
        return sub.Rd[k] == Rd_fixed[k]
    sub.fix_rd = Constraint(sub.k, rule=sub_fix_rd_rule)

    def sub_fix_wsc_rule(sub):
        return sub.Wsc == Wsc_fixed
    sub.fix_wsc = Constraint(rule=sub_fix_wsc_rule)

    def sub_fix_p_rule(sub, k):
        return sub.P[k] == P_fixed[k]
    sub.fix_p = Constraint(sub.k, rule=sub_fix_p_rule)

    # Solving the subproblem
    sub_results = opt.solve(sub, suffixes=['dual'], tee=True)

    # Update upper bound
    upper_bound =  sub.Objective()

    # Displaying upper bound results and gap between lower bound and upper bound
    print(f"\nIteration {iteration} - Upper bound: {sub.Objective()}")
    print(f"Gap: {abs(upper_bound - lower_bound)}")


    # Adding Benders cut to the master problem if iteration >= 1
    if iteration >= 1:
        def benders_cut_rule(master):
            return (
                sub.Objective() +
                sum(sub.dual[sub.fix_ru[k]] * (master.Ru[k] - Ru_fixed[k]) for k in sub.k if sub.fix_ru[k] in sub.dual) +
                sum(sub.dual[sub.fix_rd[k]] * (master.Rd[k] - Rd_fixed[k]) for k in sub.k if sub.fix_rd[k] in sub.dual) +
                (sub.dual[sub.fix_wsc] * (master.Wsc - Wsc_fixed) if sub.fix_wsc in sub.dual else 0) +
                sum(sub.dual[sub.fix_p[k]] * (master.P[k] - P_fixed[k]) for k in sub.k if sub.fix_p[k] in sub.dual)
            ) <= master.a
        if f'BendersCut_{cut_counter}' not in master.component_map(Constraint):
            master.add_component(f'BendersCut_{cut_counter}', Constraint(rule=benders_cut_rule))
        cut_counter += 1

    iteration += 1

#-------------------------------------------------------------------------------
# Displaying the final results
print(f"\n{'='*10} Iteration {iteration} - Objective Values{'='*10}") 
print(f" Master problem Objective Value: {master.Objective()}, Sub-problem Objective Value: {lower_bound}, ")

# Displaying master problem results in a clearer format
print(f"\n{'*'*10} Iteration {iteration} - Master Problem Results {'*'*10}")
print(f"Objective Value (z_down): {master.Objective():.2f}")
print(f"Expected Cost (a): {master.a.value:.2f}")
for k in master.k:
    print(f"Producer {k} -> P: {master.P[k].value:.2f}, Ru: {master.Ru[k].value:.2f}, Rd: {master.Rd[k].value:.2f}")
print(f"Wind Energy Dispatch (Wsc): {master.Wsc.value:.2f}")

# Displaying subproblem results in a clearer format
print(f"\n{'*'*10} Iteration {iteration} - Subproblem Results {'*'*10}")
print(f"Objective Value (z_up): {sub.Objective():.2f}")
for s in sub.s:
    print(f"Scenario {s} -> Load Shedding (Lsh): {sub.Lsh[s].value:.2f}, Wind Spillage (Wsp): {sub.Wsp[s].value:.2f}")
    for k in sub.k:
        print(f"  Producer {k} -> Upward Redispatch (Pu): {sub.Pu[k, s].value:.2f}, Downward Redispatch (Pd): {sub.Pd[k, s].value:.2f}")
