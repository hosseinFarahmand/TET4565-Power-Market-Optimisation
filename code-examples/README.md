# Python / Pyomo code examples

This folder contains the teaching examples used in **TET4565**. The examples are arranged by lecture/topic so that they are easy to connect to the course material.

## Contents

| Folder | Main idea | Main files |
|---|---|---|
| `L02-Energy-Reserve-Optimisation` | Joint energy and reserve-capacity optimisation | `Energy_reserve.py` |
| `L02-Stochastic-Market-Clearing` | Two-stage stochastic market clearing and a version with an explicit reserve requirement | `Stochastic Market Clearing.py`, `Stochastic with reserve requirement.py` |
| `L03-Benders-Example` | Introductory Benders decomposition example | `Benders Example.py` |
| `L03-Benders-Example-Compact` | Compact-form Benders example | `Benders Example Compact.py` |
| `L03-Stochastic-Benders-Decomposition` | Benders decomposition applied to stochastic market clearing | `Stochastic Benders Decomposition_Final.py`, `Stochastic Benders 10 Gen.py` |
| `Battery-Optimisation` | Battery participation in day-ahead and reserve/balancing markets with CVaR risk modelling | `Battery_CVaR.py`, `Battery_CVaR_Efficient Frontier.py`, `NO3_prices.xlsx` |

The `_archive-BK` folder contains earlier/backup variants and is not part of the primary teaching sequence.

## Running the examples

Most examples use **Pyomo** and **Gurobi**. A typical workflow is:

```bash
conda env create -f environment.yml
conda activate tet4565
python "code-examples/L02-Energy-Reserve-Optimisation/Energy_reserve.py"
```

You need a working Gurobi installation and licence for scripts that call `SolverFactory('gurobi')`.

For the battery examples, keep `NO3_prices.xlsx` in the same folder as the Python scripts because the code loads the workbook using a relative path.
