from pyomo.environ import *

# Global variable to fix x
Xfixed = 0

# Define the master problem model
master_model = ConcreteModel()

# Sets
master_model.IP = RangeSet(1, 100)


# Scalars
d = 1000

# Master Problem Variables
master_model.x = Var(within=NonNegativeReals, initialize=0)
master_model.a = Var(initialize=0)
master_model.ob= Var(initialize=0)

# Master Problem Objective
def master_objective_rule(master_model):
    return master_model.ob == -(master_model.x / 4) + master_model.a
master_model.master_obj = Objective(rule=lambda model: -(model.x / 4) + model.a, sense=minimize)

# Master Problem Constraints
master_model.m1 = Constraint(expr=master_model.x <= 16)
master_model.m2 = Constraint(expr=master_model.x >= 0)
master_model.m3 = Constraint(expr=master_model.a >= -100)
master_model.m4 = ConstraintList()

# Define the subproblem model
sub_model = ConcreteModel()

# Subproblem Variables
sub_model.y = Var(within=NonNegativeReals, initialize=0)
sub_model.x = Var(initialize=0)
sub_model.ob = Var(initialize=0)

# Subproblem Objective
def sub_objective_rule(sub_model):
    return sub_model.ob == -(sub_model.y)
sub_model.sub_obj = Objective(rule=lambda model: -model.y, sense=minimize)

# Subproblem Constraints
sub_model.s1 = Constraint(expr=sub_model.y - sub_model.x <= 5)
sub_model.s2 = Constraint(expr=sub_model.y - (sub_model.x / 2) <= 7.5)
sub_model.s3 = Constraint(expr=sub_model.y + (sub_model.x / 2) <= 17.5)
sub_model.s4 = Constraint(expr=-sub_model.y + sub_model.x <= 10)
sub_model.s5 = Constraint(expr=sub_model.y >= 0)
sub_model.s6 = Constraint(expr=sub_model.x == Xfixed)

# Solver
solver = SolverFactory('gurobi')
sub_model.dual = Suffix(direction=Suffix.IMPORT)

# Iteration Loop
for m in master_model.IP:
    if d <= 0.1:
        break
    # Solve Master Problem
    solver.solve(master_model, tee=True)

    # Update global variable Xfixed with the result of master_model.x
    Xfixed = master_model.x.value




    

    # Solve Subproblem
    solver.solve(sub_model, tee=True)

    # Adding Benders cut to the master problem if iteration >= 1
    if m > 1:
        def benders_cut_rule(master):
            return (
                value(sub_model.sub_obj()) +
                (sub_model.dual[sub_model.s6] * (master_model.x - Xfixed) if sub_model.s6 in sub_model.dual else 0)
            ) <= master_model.a
        if f'BendersCut_{m}' not in master_model.component_map(Constraint):
            master_model.add_component(f'BendersCut_{m}', Constraint(rule=benders_cut_rule))

   
    # Update d
    d = abs(value(sub_model.sub_obj()) - master_model.a.value)




# Display Results
print("x: ", master_model.x.value)
print("y: ", sub_model.y.value)
print("a: ", master_model.a.value)

