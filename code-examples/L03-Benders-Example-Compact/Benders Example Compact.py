# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:42:09 2024

@author: farahmand
Title Benders decomposition-Illustrative Example- A compact form
you can find the formulation for this code on slide #34 in the third lecture

"""

from pyomo.environ import *

# Define master and sub-problems for Benders decomposition

# Master Problem
master = ConcreteModel()

# Sets
master.IP = Set(initialize=range(1, 101), doc='Iteration')

# Parameters
master.SetX = Param(master.IP, initialize=0, mutable=True, doc='X values')
master.SetZ = Param(master.IP, initialize=lambda model, i: 0 if i == 1 else None, mutable=True, doc='Z values')
master.SetDual6 = Param(master.IP, initialize=0, mutable=True, doc='Dual values for constraint s6')
master.SetA = Param(master.IP, initialize=0, mutable=True, doc='A values')

# Variables
master.z_down = Var(doc='Objective function of master problem')
master.a = Var(doc='Auxiliary variable')
master.x = Var(domain=NonNegativeReals, bounds=(0, 16), doc='Decision variable x')

# Objective function
master.of = Objective(expr=-(master.x / 4) + master.a, sense=minimize)

# Constraints
master.m3 = Constraint(expr=master.a >= -100, doc='Lower bound for a')

# Benders cut constraints (initially empty)
master.m4 = ConstraintList()

# Sub-Problem
sub = ConcreteModel()

# Parameters (to be updated by master problem)
sub.Xfixed = Param(initialize=0, mutable=True, doc='Fixed value of x from master problem')

# Variables
sub.z_up = Var(doc='Objective function of sub problem')
sub.y = Var(domain=NonNegativeReals, doc='Decision variable y')
sub.x = Var(doc='Decision variable x (fixed in sub-problem)')

# Objective function
sub.off = Objective(expr=-sub.y, sense=minimize)

# Constraints
sub.s1 = Constraint(expr=sub.y - sub.x <= 5, doc='Constraint s1')
sub.s2 = Constraint(expr=sub.y - (sub.x / 2) <= 7.5, doc='Constraint s2')
sub.s3 = Constraint(expr=sub.y + (sub.x / 2) <= 17.5, doc='Constraint s3')
sub.s4 = Constraint(expr=-sub.y + sub.x <= 10, doc='Constraint s4')
sub.s5 = Constraint(expr=sub.y >= 0, doc='Non-negativity of y')
sub.s6 = Constraint(expr=sub.x == sub.Xfixed, doc='Fixed value of x')

# Iterative Benders Decomposition Loop
solver = SolverFactory('gurobi')
d = 1000

for m in range(1, 101):
    if d <= 0.1:
        break

    # Solve Master Problem
    result = solver.solve(master, tee=True)

    # Update sub-problem parameters from master solution
    sub.Xfixed.set_value(master.x.value)

    # Solve Sub-Problem
    result = solver.solve(sub, tee=True)

    # Check convergence
    d = (sub.z_up.value if sub.z_up.value is not None else 0) - (master.a.value if master.a.value is not None else 0)

    # Add Benders cut to master
    if m > 1:
        master.m4.add(expr=(master.SetDual6[m] * (master.x - master.SetX[m])) + master.SetZ[m] <= master.a)

    # Store iteration results
    master.SetX[m] = master.x.value if master.x.value is not None else 0
    master.SetZ[m] = sub.z_up.value if sub.z_up.value is not None else 0
    master.SetDual6[m] = sub.s6.body() if hasattr(sub.s6, 'body') else 0
    master.SetA[m] = master.a.value

# Display final results
print('Optimal Solution of Master Problem:')
print(f'Objective Value (z_down): {master.z_down.value}')
print(f'Optimal x: {master.x.value}')
print('Optimal Solution of Sub-Problem:')
print(f'Objective Value (z_up): {sub.z_up.value}')
print(f'Optimal y: {sub.y.value}')