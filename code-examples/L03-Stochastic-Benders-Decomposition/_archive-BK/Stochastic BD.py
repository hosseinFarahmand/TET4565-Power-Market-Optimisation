from pyomo.environ import *
from pyomo.opt import SolverFactory

# Master Problem: First Stage Model
master = ConcreteModel()

# Sets
master.k = Set(initialize=['k1', 'k2'])  # Producers

# Parameters
C = {'k1': 10, 'k2': 30}  # Production cost per unit
Cu = {'k1': 0, 'k2': 25}  # Upward reserve cost per unit
Cd = {'k1': 0, 'k2': 10}  # Downward reserve cost per unit
Pmax = {'k1': 100, 'k2': 50}  # Maximum production
Wscmax = 40  # Maximum wind capacity
L = 120  # Load
vLOL = 200  # Value of lost load

# Variables
master.P = Var(master.k, domain=NonNegativeReals, initialize=0)  # Energy dispatch
master.Ru = Var(master.k, domain=NonNegativeReals, initialize=0)  # Upward reserve dispatch
master.Rd = Var(master.k, domain=NonNegativeReals, initialize=0)  # Downward reserve dispatch
master.Wsc = Var(domain=NonNegativeReals, initialize=0)  # Wind energy dispatch
master.a = Var(domain=Reals, initialize=0)  # Expected cost

# Objective function
def master_objective_rule(master):
    return sum(C[k] * master.P[k] + Cu[k] * master.Ru[k] + Cd[k] * master.Rd[k] for k in master.k) + master.a
master.Objective = Objective(rule=master_objective_rule, sense=minimize)

# Constraints
# Power balance equation at the forward stage
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

# Solving the master problem
opt = SolverFactory('gurobi')
master_results = opt.solve(master, tee=True)

# Displaying master problem results
print("Master Problem Results:")
print(f"Objective Value (z_down): {master.Objective()} ")
for k in master.k:
    print(f"Energy Dispatch (P[{k}]): {master.P[k].value}")
    print(f"Upward Reserve Dispatch (Ru[{k}]): {master.Ru[k].value}")
    print(f"Downward Reserve Dispatch (Rd[{k}]): {master.Rd[k].value}")
print(f"Wind Energy Dispatch (Wsc): {master.Wsc.value}")

# Subproblem: Second Stage Model
sub = ConcreteModel()

# Sets
sub.k = Set(initialize=['k1', 'k2'])  # Producers
sub.s = Set(initialize=['s1', 's2', 's3', 's4'])  # Scenarios

# Parameters
pi = {'s1': 0.25, 's2': 0.25, 's3': 0.25, 's4': 0.25}  # Scenario probability
w = {'s1': 5, 's2': 30, 's3': 40, 's4': 80}  # Wind power production scenario

# Internal Variables equal to master variables
sub.Ru_fixed = {k: master.Ru[k].value if master.Ru[k].value is not None else 0 for k in sub.k}  # Fixed upward reserve from master
sub.Rd_fixed = {k: master.Rd[k].value if master.Rd[k].value is not None else 0 for k in sub.k}  # Fixed downward reserve from master
sub.Wsc_fixed = master.Wsc.value if master.Wsc.value is not None else 0  # Fixed wind energy dispatch from master

# Variables
sub.Pu = Var(sub.k, sub.s, domain=NonNegativeReals, initialize=0)  # Upward energy redispatch
sub.Pd = Var(sub.k, sub.s, domain=NonNegativeReals, initialize=0)  # Downward energy redispatch
sub.Lsh = Var(sub.s, domain=NonNegativeReals, initialize=0)  # Load shedding
sub.Wsp = Var(sub.s, domain=NonNegativeReals, initialize=0)  # Wind spillage

# Suffix to capture dual values
sub.dual = Suffix(direction=Suffix.IMPORT)

# Objective function for the subproblem
def sub_objective_rule(sub):
    return sum(pi[s] * (sum(C[k] * (sub.Pu[k, s] - sub.Pd[k, s]) for k in sub.k) + vLOL * sub.Lsh[s]) for s in sub.s)
sub.Objective = Objective(rule=sub_objective_rule, sense=minimize)

# Constraints
# Power balance equation at the balancing stage
def sub_bal_rule(sub, s):
    return sum(sub.Pu[k, s] - sub.Pd[k, s] for k in sub.k) + sub.Lsh[s] + w[s] - sub.Wsc_fixed - sub.Wsp[s] == 0
sub.bal = Constraint(sub.s, rule=sub_bal_rule)

# Upward energy redispatch upper limit
def sub_puu_rule(sub, k, s):
    return sub.Pu[k, s] <= sub.Ru_fixed[k]
sub.Puu = Constraint(sub.k, sub.s, rule=sub_puu_rule)

# Downward energy redispatch upper limit
def sub_pdu_rule(sub, k, s):
    return sub.Pd[k, s] <= sub.Rd_fixed[k]
sub.Pdu = Constraint(sub.k, sub.s, rule=sub_pdu_rule)

# Wind power spillage upper limit
def sub_wspu_rule(sub, s):
    return sub.Wsp[s] <= w[s]
sub.Wspu = Constraint(sub.s, rule=sub_wspu_rule)

# Load shedding upper limit
def sub_lshu_rule(sub, s):
    return sub.Lsh[s] <= L
sub.Lshu = Constraint(sub.s, rule=sub_lshu_rule)

# Solving the subproblem
sub_results = opt.solve(sub, suffixes=['dual'], tee=True)

# Displaying subproblem results
print("\nSubproblem Results:")
print(f"Objective Value (z_up): {sub.Objective()} ")
for s in sub.s:
    print(f"Load Shedding (Lsh[{s}]): {sub.Lsh[s].value}")
    print(f"Wind Spillage (Wsp[{s}]): {sub.Wsp[s].value}")
    for k in sub.k:
        print(f"Upward Energy Redispatch (Pu[{k},{s}]): {sub.Pu[k, s].value}")
        print(f"Downward Energy Redispatch (Pd[{k},{s}]): {sub.Pd[k, s].value}")
        
# Iteration Loop
cut_counter = 1
for m in range(1, 101):
    if m > 1:
        # Adding Benders cut to the master problem with dual values
        def benders_cut_rule(master):
            return (
                sum(sub.dual[sub.Puu[k, s]] * (master.Ru[k] - sub.Ru_fixed[k]) for k in master.k for s in sub.s if sub.Puu[k, s] in sub.dual) +
                sum(sub.dual[sub.Pdu[k, s]] * (master.Rd[k] - sub.Rd_fixed[k]) for k in master.k for s in sub.s if sub.Pdu[k, s] in sub.dual) +
                sum(sub.dual[sub.bal[s]] * (master.Wsc - sub.Wsc_fixed) for s in sub.s if sub.bal[s] in sub.dual)
            ) <= master.a
        master.add_component(f'BendersCut_{cut_counter}', Constraint(rule=benders_cut_rule))
        cut_counter += 1
    
    master_results = opt.solve(master)
    print("\nMaster Problem Results after Iteration:")
    print(f"Objective Value (z_down): {master.Objective.expr()}")
    for k in master.k:
        print(f"Energy Dispatch (P[{k}]): {master.P[k].value}")
        print(f"Upward Reserve Dispatch (Ru[{k}]): {master.Ru[k].value}")
        print(f"Downward Reserve Dispatch (Rd[{k}]): {master.Rd[k].value}")
    print(f"Wind Energy Dispatch (Wsc): {master.Wsc.value}")
    
    Ru_fixed = {k: master.Ru[k].value for k in master.k}
    Rd_fixed = {k: master.Rd[k].value for k in master.k}
    Wsc_fixed = master.Wsc.value
    sub_results = opt.solve(sub, suffixes=['dual'])
    print("\nSubproblem Results after Iteration:")
    print(f"Objective Value (z_up): {sub.Objective.expr()}")
    for s in sub.s:
        print(f"Load Shedding (Lsh[{s}]): {sub.Lsh[s].value}")
        print(f"Wind Spillage (Wsp[{s}]): {sub.Wsp[s].value}")
        for k in sub.k:
            print(f"Upward Energy Redispatch (Pu[{k},{s}]): {sub.Pu[k, s].value}")
            print(f"Downward Energy Redispatch (Pd[{k},{s}]): {sub.Pd[k, s].value}")