"""
Created on Sun Jun 19 12:23:47 2022

@author: Farahmand

Two equivalent implementations of the subproblem and the cut:

  EXTENSIVE form (slide "Illustrative Example")
      subproblem: x_fixed is a parameter on the right-hand sides, duals pi, mu, sigma, gamma
      cut:  alpha >= -pi[5 + x] - mu[15/2 + x/2] - sigma[32/2 - x/2] - gamma[10 - x]

  COMPACT form (slide "A compact form")
      subproblem: x is a variable fixed by  x = x_fixed, dual rho
      cut:  alpha >= -y(k) + rho(k) * (x - x(k))

Both give the same iterations, the same cuts and the same optimum.

Solver: Gurobi.
Requires:  pip install pyomo matplotlib gurobipy   (and a Gurobi licence; the licence
           bundled with gurobipy is enough for a model of this size)
"""
import math

import matplotlib.pyplot as plt
import pyomo.environ as pyo

# ----------------------------------------------------------------------------
# Data and settings
# ----------------------------------------------------------------------------
RHS3 = 32 / 2            # third constraint; set to 35 / 2 to use the other value
X_MIN, X_MAX = 0.0, 16.0
X_INITIAL = 0.0
ALPHA_DOWN = -1e6        # keeps the master problem bounded while there is no cut
EPS = 1e-6
MAX_ITER = 20
SOLVER = "gurobi"        # Pyomo interface to Gurobi; "gurobi_direct" is used if this one is not available


def get_solver():
    names = list(dict.fromkeys([SOLVER, "gurobi", "gurobi_direct"]))
    for name in names:
        solver = pyo.SolverFactory(name)
        if solver.available(exception_flag=False):
            print(f"Solver: {name}")
            return solver
    raise RuntimeError("Gurobi was not found. Install it with  pip install gurobipy  and check the licence.")


def solve(model, solver):
    result = solver.solve(model)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"{model.name}: {result.solver.termination_condition}")


# ----------------------------------------------------------------------------
# Original (non-decomposed) problem - used only to check the Benders' result
# ----------------------------------------------------------------------------
def solve_original(solver):
    m = pyo.ConcreteModel(name="original")
    m.x = pyo.Var(bounds=(X_MIN, X_MAX))
    m.y = pyo.Var(within=pyo.NonNegativeReals)
    m.obj = pyo.Objective(expr=-m.y - m.x / 4, sense=pyo.minimize)
    m.c1 = pyo.Constraint(expr=m.y - m.x <= 5)
    m.c2 = pyo.Constraint(expr=m.y - m.x / 2 <= 15 / 2)
    m.c3 = pyo.Constraint(expr=m.y + m.x / 2 <= RHS3)
    m.c4 = pyo.Constraint(expr=-m.y + m.x <= 10)
    solve(m, solver)
    return pyo.value(m.x), pyo.value(m.y), pyo.value(m.obj)


# ----------------------------------------------------------------------------
# Subproblem, EXTENSIVE form: x_fixed appears on the right-hand sides
# ----------------------------------------------------------------------------
def build_subproblem_extensive():
    m = pyo.ConcreteModel(name="subproblem (extensive)")
    m.x_fixed = pyo.Param(mutable=True, initialize=X_INITIAL)
    m.y = pyo.Var(within=pyo.NonNegativeReals)
    m.obj = pyo.Objective(expr=-m.y, sense=pyo.minimize)
    m.c_pi = pyo.Constraint(expr=m.y <= 5 + m.x_fixed)
    m.c_mu = pyo.Constraint(expr=m.y <= 15 / 2 + m.x_fixed / 2)
    m.c_sigma = pyo.Constraint(expr=m.y <= RHS3 - m.x_fixed / 2)
    m.c_gamma = pyo.Constraint(expr=-m.y <= 10 - m.x_fixed)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def multipliers_extensive(m):
    """Pyomo reports sensitivities (d objective / d right-hand side), which are <= 0 for
    these '<=' constraints in a minimisation. The slides use multipliers >= 0, so the
    sign is changed here."""
    return {
        "pi": -m.dual[m.c_pi],
        "mu": -m.dual[m.c_mu],
        "sigma": -m.dual[m.c_sigma],
        "gamma": -m.dual[m.c_gamma],
    }


# ----------------------------------------------------------------------------
# Subproblem, COMPACT form: x is a variable, fixed by one extra constraint
# ----------------------------------------------------------------------------
def build_subproblem_compact():
    m = pyo.ConcreteModel(name="subproblem (compact)")
    m.x_fixed = pyo.Param(mutable=True, initialize=X_INITIAL)
    m.x = pyo.Var()
    m.y = pyo.Var(within=pyo.NonNegativeReals)
    m.obj = pyo.Objective(expr=-m.y, sense=pyo.minimize)
    m.c_pi = pyo.Constraint(expr=m.y - m.x <= 5)
    m.c_mu = pyo.Constraint(expr=m.y - m.x / 2 <= 15 / 2)
    m.c_sigma = pyo.Constraint(expr=m.y + m.x / 2 <= RHS3)
    m.c_gamma = pyo.Constraint(expr=-m.y + m.x <= 10)
    m.c_rho = pyo.Constraint(expr=m.x == m.x_fixed)          # dual: rho
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


# ----------------------------------------------------------------------------
# Master problem (the same model for both forms; only the cuts differ)
# ----------------------------------------------------------------------------
def build_master():
    m = pyo.ConcreteModel(name="master")
    m.x = pyo.Var(bounds=(X_MIN, X_MAX))
    m.alpha = pyo.Var(bounds=(ALPHA_DOWN, None))
    m.obj = pyo.Objective(expr=-m.x / 4 + m.alpha, sense=pyo.minimize)
    m.cuts = pyo.ConstraintList()
    return m


def add_cut_extensive(master, mult):
    """alpha >= -pi[5 + x] - mu[15/2 + x/2] - sigma[32/2 - x/2] - gamma[10 - x]"""
    x = master.x
    master.cuts.add(
        master.alpha
        >= -mult["pi"] * (5 + x)
        - mult["mu"] * (15 / 2 + x / 2)
        - mult["sigma"] * (RHS3 - x / 2)
        - mult["gamma"] * (10 - x)
    )
    intercept = -(mult["pi"] * 5 + mult["mu"] * 15 / 2 + mult["sigma"] * RHS3 + mult["gamma"] * 10)
    slope = -mult["pi"] - mult["mu"] / 2 + mult["sigma"] / 2 + mult["gamma"]
    return intercept, slope


def add_cut_compact(master, y_k, rho_k, x_k):
    """alpha >= -y(k) + rho(k) * (x - x(k))"""
    master.cuts.add(master.alpha >= -y_k + rho_k * (master.x - x_k))
    return -y_k - rho_k * x_k, rho_k


# ----------------------------------------------------------------------------
# Benders' loop (steps as on the "Algorithm" slide)
# ----------------------------------------------------------------------------
def benders(form, solver):
    sub = build_subproblem_extensive() if form == "extensive" else build_subproblem_compact()
    master = build_master()

    x_k, lb = X_INITIAL, -math.inf                     # Step 0
    history = []
    for i in range(1, MAX_ITER + 1):
        # Step 1: solve the subproblem with x fixed
        sub.x_fixed.set_value(x_k)
        solve(sub, solver)
        y_k = pyo.value(sub.y)
        ub = -x_k / 4 + pyo.value(sub.obj)             # cost of the fixed x + subproblem objective
        row = dict(i=i, x=x_k, lb=lb, y=y_k, ub=ub, cut=None, duals="")
        history.append(row)

        # Step 2: convergence check
        if abs(ub - lb) <= EPS:
            break

        # Step 3: add a cut and solve the master problem
        if form == "extensive":
            mult = multipliers_extensive(sub)
            row["cut"] = add_cut_extensive(master, mult)
            row["duals"] = ", ".join(f"{k} = {v:g}" for k, v in mult.items() if abs(v) > 1e-9)
        else:
            rho_k = sub.dual[sub.c_rho]                # sensitivity of the subproblem cost to x_fixed
            row["cut"] = add_cut_compact(master, y_k, rho_k, x_k)
            row["duals"] = f"rho = {rho_k:+g}"
        solve(master, solver)
        x_k = pyo.value(master.x)
        lb = pyo.value(master.obj)                     # master objective = lower bound
    else:
        raise RuntimeError("No convergence within MAX_ITER iterations")
    return history


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def fmt(v, digits=3):
    if math.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{round(v, digits):g}"


def print_results(form, history):
    print(f"\n=== {form.upper()} form " + "=" * 44)
    print(f"{'Iteration i':>11} {'x(i)':>8} {'LB':>10} {'y(i)':>8} {'UB':>10}   duals used in the cut")
    for r in history:
        print(f"{r['i']:>11} {fmt(r['x'], 2):>8} {fmt(r['lb']):>10} {fmt(r['y'], 2):>8} {fmt(r['ub']):>10}   {r['duals'] or '-'}")

    print("\nCuts added to the master problem")
    for r in history:
        if r["cut"] is not None:
            intercept, slope = r["cut"]
            sign = "+" if slope >= 0 else "-"
            print(f"  after iteration {r['i']}:  alpha >= {fmt(intercept)} {sign} {fmt(abs(slope))} x")

    last = history[-1]
    print(f"\nIteration {last['i']}: UB = LB = {fmt(last['ub'])}, so we stop.")
    print(f"Optimum: x* = {fmt(last['x'])}, y* = {fmt(last['y'])}, z* = {fmt(last['ub'])}")


def plot_bounds(results, filename="benders_pyomo_bounds.png"):
    """One panel per form, same scale, so the two runs can be compared directly."""
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    lb_colour, ub_colour = "#2a78d6", "#eb6834"

    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 3.9), dpi=150, sharey=True)
    for ax, (form, history) in zip(axes, results.items()):
        its = [r["i"] for r in history]
        ub = [r["ub"] for r in history]
        lb = [r["lb"] if math.isfinite(r["lb"]) else float("nan") for r in history]

        ax.plot(its, lb, color=lb_colour, lw=2, marker="D", ms=7, label="Lower bound")
        ax.plot(its, ub, color=ub_colour, lw=2, marker="s", ms=7, label="Upper bound")

        k = next(n for n, v in enumerate(lb) if not math.isnan(v))
        ax.annotate(f"LB = {fmt(lb[k])}", (its[k], lb[k]), xytext=(8, -4), textcoords="offset points",
                    color=ink, fontsize=9, va="top")
        ax.annotate(f"UB = {fmt(ub[k])}", (its[k], ub[k]), xytext=(8, 6), textcoords="offset points",
                    color=ink, fontsize=9)
        ax.annotate(f"UB = LB = {fmt(ub[-1])}", (its[-1], ub[-1]), xytext=(-6, 10), textcoords="offset points",
                    color=ink, fontsize=9, ha="right")

        ax.set_title(f"{form.capitalize()} form", loc="left", color=ink, fontsize=11)
        ax.set_xlabel("Iteration", color=muted)
        ax.set_xticks(its)
        ax.set_xlim(its[0] - 0.3, its[-1] + 0.3)
        finite = [v for v in ub + lb if not math.isnan(v)]
        ax.set_ylim(min(finite) - 4, max(finite) + 4)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    axes[0].set_ylabel("Objective value", color=muted)
    axes[-1].legend(frameon=False, loc="lower right", labelcolor=ink)
    fig.suptitle("Benders' decomposition: bounds per iteration", x=0.01, ha="left", color=ink, fontsize=12)
    fig.tight_layout()
    fig.savefig(filename)
    print(f"\nChart saved to {filename}")
    return fig


if __name__ == "__main__":
    solver = get_solver()

    x_opt, y_opt, z_opt = solve_original(solver)
    print(f"Original problem solved directly: x* = {fmt(x_opt)}, y* = {fmt(y_opt)}, z* = {fmt(z_opt)}")

    results = {form: benders(form, solver) for form in ("extensive", "compact")}
    for form, history in results.items():
        print_results(form, history)
        assert abs(history[-1]["ub"] - z_opt) <= 1e-6, f"{form} form does not match the direct solution"

    # the two forms must generate identical cuts
    for r_ext, r_com in zip(*results.values()):
        if r_ext["cut"] is not None:
            assert all(abs(a - b) < 1e-8 for a, b in zip(r_ext["cut"], r_com["cut"])), "cuts differ"
    print("\nCheck: both forms give the same cuts, and the same optimum as the original problem.")

    plot_bounds(results)
    plt.show()
