"""
Stochastic dynamic programming (SDP) for a small hydrothermal system, in Pyomo.

System
    one hydro reservoir, three thermal units, load rationing as the last resort
    T stages (months); inflow in each stage is random with three outcomes
    (dry / normal / wet), independent between stages

SDP idea
    State      : reservoir content at the start of a stage, on a grid of points
    Backward   : for stage t = T, ..., 1, for every grid point and every inflow outcome,
                 solve a one-stage LP:   min  thermal cost + rationing cost + future cost
                 The expectation over the inflow outcomes gives the expected future cost
                 F[t][k] and the water value WV[t][k] = -dF/dv  (EUR/MWh).
    Future cost: inside the one-stage LP the future cost of the end reservoir is a linear
                 interpolation between the grid points of F[t+1] (exact for a convex F).
    Forward    : simulate random inflow series with the tables F to get reservoir
                 trajectories and the operating cost.

In each stage the inflow is observed before the decision is taken.
Units: energy in MWh per stage, cost in EUR.

Requires:  pip install pyomo gurobipy numpy matplotlib   (and a Gurobi licence)
"""
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
T = 12                                                    # stages (months)
DEMAND = [50, 50, 45, 40, 35, 30, 30, 35, 40, 45, 50, 55]            # MWh per stage
MEAN_INFLOW = [10, 10, 15, 30, 50, 60, 45, 30, 20, 15, 10, 10]       # MWh per stage
INFLOW_FACTOR = [0.5, 1.0, 1.5]                           # dry, normal, wet
INFLOW_PROB = [0.25, 0.50, 0.25]

V_MAX = 100.0                                             # reservoir capacity [MWh]
V_INITIAL = 50.0                                          # reservoir at the start [MWh]
Q_MAX = 40.0                                              # max hydro production per stage [MWh]

THERMAL_CAP = [15.0, 15.0, 20.0]                          # MWh per stage
THERMAL_COST = [20.0, 40.0, 80.0]                         # EUR/MWh
RATION_COST = 500.0                                       # EUR/MWh not served
END_VALUE = 30.0                                          # value of water left after stage T [EUR/MWh]

N_POINTS = 11                                             # reservoir grid points
V_GRID = np.linspace(0.0, V_MAX, N_POINTS)
N_SIM = 100                                               # simulated inflow series
SOLVER = "appsi_gurobi"                                   # Gurobi through Pyomo's fast (in-memory) interface


# ----------------------------------------------------------------------------
# One-stage problem (built once; the parameters are updated before each solve)
# ----------------------------------------------------------------------------
def build_stage_model():
    m = pyo.ConcreteModel(name="one-stage hydrothermal problem")
    m.G = pyo.RangeSet(0, len(THERMAL_CAP) - 1)           # thermal units
    m.K = pyo.RangeSet(0, N_POINTS - 1)                   # reservoir grid points

    # parameters that change from solve to solve
    m.v_start = pyo.Param(mutable=True, initialize=V_INITIAL)
    m.inflow = pyo.Param(mutable=True, initialize=0.0)
    m.demand = pyo.Param(mutable=True, initialize=0.0)
    m.F_next = pyo.Param(m.K, mutable=True, initialize=0.0)   # future cost at the grid points

    # decisions
    m.q = pyo.Var(bounds=(0, Q_MAX))                      # hydro production
    m.spill = pyo.Var(within=pyo.NonNegativeReals)
    m.v_end = pyo.Var(bounds=(0, V_MAX))                  # reservoir at the end of the stage
    m.p = pyo.Var(m.G, bounds=lambda m, g: (0, THERMAL_CAP[g]))
    m.ration = pyo.Var(within=pyo.NonNegativeReals)
    m.alpha = pyo.Var()                                   # future cost
    m.w = pyo.Var(m.K, within=pyo.NonNegativeReals)       # interpolation weights

    m.cost = pyo.Objective(
        expr=sum(THERMAL_COST[g] * m.p[g] for g in m.G) + RATION_COST * m.ration + m.alpha,
        sense=pyo.minimize,
    )
    # reservoir balance; its dual gives the water value
    m.reservoir = pyo.Constraint(expr=m.v_end + m.q + m.spill == m.v_start + m.inflow)
    # power balance
    m.power_balance = pyo.Constraint(expr=m.q + sum(m.p[g] for g in m.G) + m.ration == m.demand)
    # future cost: interpolation between the grid points
    m.weights = pyo.Constraint(expr=sum(m.w[k] for k in m.K) == 1)
    m.v_interp = pyo.Constraint(expr=m.v_end == sum(m.w[k] * V_GRID[k] for k in m.K))
    m.alpha_interp = pyo.Constraint(expr=m.alpha == sum(m.w[k] * m.F_next[k] for k in m.K))

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def solve_stage(m, solver, v_start, inflow, demand, F_next):
    m.v_start.set_value(float(v_start))
    m.inflow.set_value(float(inflow))
    m.demand.set_value(float(demand))
    for k in m.K:
        m.F_next[k].set_value(float(F_next[k]))
    result = solver.solve(m)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Stage problem not solved: {result.solver.termination_condition}")


# ----------------------------------------------------------------------------
# Backward recursion: expected future cost F and water values WV
# ----------------------------------------------------------------------------
def backward_recursion(m, solver):
    F = np.zeros((T + 1, N_POINTS))            # F[t][k]: expected cost from stage t to the end
    WV = np.zeros((T, N_POINTS))               # WV[t][k]: water value at the start of stage t
    F[T] = -END_VALUE * V_GRID                 # water left at the end has a value

    for t in reversed(range(T)):
        for k, v in enumerate(V_GRID):
            for factor, prob in zip(INFLOW_FACTOR, INFLOW_PROB):
                solve_stage(m, solver, v, factor * MEAN_INFLOW[t], DEMAND[t], F[t + 1])
                F[t, k] += prob * pyo.value(m.cost)
                # dual = d cost / d (start reservoir) <= 0, so the water value is its negative
                WV[t, k] += prob * (-m.dual[m.reservoir])
    return F, WV


# ----------------------------------------------------------------------------
# Forward simulation with the tables from the backward recursion
# ----------------------------------------------------------------------------
def simulate(m, solver, F, seed=1):
    rng = np.random.default_rng(seed)
    volume = np.zeros((N_SIM, T + 1))
    hydro = np.zeros((N_SIM, T))
    thermal = np.zeros((N_SIM, T))
    ration = np.zeros((N_SIM, T))
    cost = np.zeros(N_SIM)

    for n in range(N_SIM):
        volume[n, 0] = V_INITIAL
        outcomes = rng.choice(len(INFLOW_FACTOR), size=T, p=INFLOW_PROB)
        for t in range(T):
            inflow = INFLOW_FACTOR[outcomes[t]] * MEAN_INFLOW[t]
            solve_stage(m, solver, volume[n, t], inflow, DEMAND[t], F[t + 1])
            volume[n, t + 1] = pyo.value(m.v_end)
            hydro[n, t] = pyo.value(m.q)
            thermal[n, t] = sum(pyo.value(m.p[g]) for g in m.G)
            ration[n, t] = pyo.value(m.ration)
            cost[n] += pyo.value(m.cost) - pyo.value(m.alpha)        # stage cost only
        cost[n] -= END_VALUE * volume[n, T]                          # value of the water left
    return dict(volume=volume, hydro=hydro, thermal=thermal, ration=ration, cost=cost)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_results(F, WV, sim):
    print("\nWater values [EUR/MWh] at the start of each stage, by reservoir content")
    cols = range(0, N_POINTS, 2)
    print("stage " + "".join(f"{V_GRID[k]:>8.0f}" for k in cols) + "   MWh")
    for t in range(T):
        print(f"{t + 1:>5} " + "".join(f"{WV[t, k]:>8.1f}" for k in cols))

    expected = float(np.interp(V_INITIAL, V_GRID, F[0]))
    half_width = 1.96 * sim["cost"].std(ddof=1) / np.sqrt(N_SIM)
    print(f"\nExpected total cost from the SDP tables : {expected:>10.1f} EUR")
    print(f"Mean cost of {N_SIM} simulated series    : {sim['cost'].mean():>10.1f} EUR  (+/- {half_width:.1f})")
    print(f"Mean thermal production per stage       : {sim['thermal'].mean():>10.1f} MWh")
    print(f"Mean rationing per stage                : {sim['ration'].mean():>10.2f} MWh")


def plot_results(WV, sim, filename="sdp_hydrothermal.png"):
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    # water values: stage on the x-axis, reservoir content on the y-axis
    mesh = ax1.pcolormesh(np.arange(1, T + 1), V_GRID, WV.T, cmap=blues, shading="nearest", edgecolors="white", linewidth=1)
    bar = fig.colorbar(mesh, ax=ax1, pad=0.02)
    bar.set_label("Water value [EUR/MWh]", color=muted)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=muted, length=0)
    ax1.set_title("Water values from the backward recursion", loc="left", color=ink, fontsize=11)
    ax1.set_xlabel("Stage", color=muted)
    ax1.set_ylabel("Reservoir content at the start of the stage [MWh]", color=muted)
    ax1.set_xticks(range(1, T + 1))

    # simulated reservoir trajectories
    stages = np.arange(0, T + 1)
    for n in range(N_SIM):
        ax2.plot(stages, sim["volume"][n], color="#c9c8c2", lw=0.8, label="Simulated series" if n == 0 else None)
    ax2.plot(stages, sim["volume"].mean(axis=0), color="#2a78d6", lw=2.5, label="Mean")
    ax2.set_title(f"Reservoir content in {N_SIM} simulated inflow series", loc="left", color=ink, fontsize=11)
    ax2.set_xlabel("End of stage", color=muted)
    ax2.set_ylabel("Reservoir content [MWh]", color=muted)
    ax2.set_xticks(stages)
    ax2.set_ylim(-3, V_MAX + 3)
    ax2.grid(axis="y", color=grid, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="upper left", labelcolor=ink)

    for ax in (ax1, ax2):
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.tight_layout()
    fig.savefig(filename)
    print(f"\nChart saved to {filename}")
    return fig


if __name__ == "__main__":
    solver = pyo.SolverFactory(SOLVER)
    if not solver.available(exception_flag=False):
        raise RuntimeError("Gurobi was not found. Install it with  pip install gurobipy  and check the licence.")

    model = build_stage_model()
    F, WV = backward_recursion(model, solver)
    sim = simulate(model, solver, F)
    print_results(F, WV, sim)
    plot_results(WV, sim)
    plt.show()
