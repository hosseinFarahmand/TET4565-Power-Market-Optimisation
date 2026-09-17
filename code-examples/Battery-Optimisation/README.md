# Battery optimisation with CVaR

These scripts model battery participation across power-market revenue streams with risk represented through **Conditional Value at Risk (CVaR)**.

- `Battery_CVaR.py` — expected-profit/CVaR optimisation with result summaries and plots.
- `Battery_CVaR_Efficient Frontier.py` — extends the analysis with a beta sweep and an expected-profit versus CVaR efficient-frontier plot.
- `NO3_prices.xlsx` — input data used by both scripts; keep it in this directory when running the examples.

The models require Pyomo, pandas, NumPy/matplotlib as applicable, `openpyxl` for the Excel workbook, and Gurobi.
