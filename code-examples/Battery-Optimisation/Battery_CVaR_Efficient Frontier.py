# -*- coding: utf-8 -*-
"""
Battery CVaR model with added plots:
  1) SOC versus day-ahead prices
  2) Profit value stacking (waterfall)

Updates:
  - alpha_up and alpha_dn are now user inputs in the main section

Inputs (same folder as script):
  - NO3_prices.xlsx:
      * "Up price" (96)
      * "Down price" (96)
      * "Net activated volume" (96)
      * "reserve capacity up price" (24)
      * "reserve capacity down price" (24)
      * "Dar-ahead" or "Day-ahead" (96)
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np

# -----------------------------
# EXCEL HELPERS
# -----------------------------
def _extract_quarterhour_series(df: pd.DataFrame) -> pd.DataFrame:
    """Detect first row like 'HH:MM - HH:MM', then take 96 rows."""
    first_col = df.columns[0]
    pat = re.compile(r"^\s*\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\s*$")
    idx_candidates = [
        i for i, v in enumerate(df[first_col].tolist())
        if (v is not None) and pat.match(str(v))
    ]
    if not idx_candidates:
        raise ValueError(
            f"Could not find 15-min delivery periods in first column '{first_col}'."
        )
    i0 = idx_candidates[0]
    df_q = df.iloc[i0:i0 + T_TOTAL].copy()
    if len(df_q) < T_TOTAL:
        raise ValueError(
            f"Expected at least {T_TOTAL} quarter-hours after first delivery period row, got {len(df_q)}."
        )
    df_q.reset_index(drop=True, inplace=True)
    return df_q


def _find_sheet(xls_path: str, candidates: List[str]) -> str:
    """Return the first matching sheet name in an Excel file (case-insensitive)."""
    wb = pd.ExcelFile(xls_path)
    sheets_lower = {s.lower(): s for s in wb.sheet_names}
    for cand in candidates:
        s = sheets_lower.get(cand.lower())
        if s is not None:
            return s
    raise ValueError(
        f"None of the expected sheets {candidates} were found in {xls_path}. "
        f"Available sheets: {wb.sheet_names}"
    )


def _load_da_prices_from_prices_file(
    prices_xlsx_path: str,
    scenarios: List[str],
    scenario_to_col: Dict[str, str],
) -> Dict[Tuple[int, str], float]:
    """
    Load day-ahead prices from the same Excel file as the mFRR inputs.
    """
    if not os.path.exists(prices_xlsx_path):
        raise FileNotFoundError(f"Price file not found: {prices_xlsx_path}")

    da_sheet = _find_sheet(
        prices_xlsx_path,
        ["Dar-ahead", "Day-ahead", "Day_ahead", "Day ahead"]
    )
    da_raw = pd.read_excel(prices_xlsx_path, sheet_name=da_sheet)
    da = _extract_quarterhour_series(da_raw)

    da_cols = [c for c in da.columns[1:] if str(c).strip() != ""]
    if not da_cols:
        raise ValueError(f"No scenario/day columns found in DA sheet '{da_sheet}'.")

    pi_DA: Dict[Tuple[int, str], float] = {}
    fallback_col = da_cols[0]

    for w in scenarios:
        col = scenario_to_col.get(w, None)
        if col not in da_cols:
            matched = None
            for c in da_cols:
                try:
                    if pd.to_datetime(c).strftime("%Y-%m-%d") == w:
                        matched = c
                        break
                except Exception:
                    continue
            col = matched if matched is not None else fallback_col

        vals = pd.to_numeric(da[col], errors="coerce").fillna(0.0).to_numpy()
        for t in range(T_TOTAL):
            pi_DA[(t, w)] = float(vals[t])

    return pi_DA


def _load_mfrr_inputs_from_prices_file(prices_xlsx_path: str):
    """
    Reads mFRR + capacity + activation inputs from a single Excel file.
    """
    if not os.path.exists(prices_xlsx_path):
        raise FileNotFoundError(f"Price file not found: {prices_xlsx_path}")

    up = pd.read_excel(prices_xlsx_path, sheet_name="Up price")
    dn = pd.read_excel(prices_xlsx_path, sheet_name="Down price")
    net = pd.read_excel(prices_xlsx_path, sheet_name="Net activated volume")
    cap_up = pd.read_excel(prices_xlsx_path, sheet_name="reserve capacity up price")
    cap_dn = pd.read_excel(prices_xlsx_path, sheet_name="reserve capacity down price")

    up_q = _extract_quarterhour_series(up) if up.shape[0] > T_TOTAL else up.iloc[:T_TOTAL].copy()
    dn_q = _extract_quarterhour_series(dn) if dn.shape[0] > T_TOTAL else dn.iloc[:T_TOTAL].copy()
    net_q = _extract_quarterhour_series(net) if net.shape[0] > T_TOTAL else net.iloc[:T_TOTAL].copy()

    for name, df, need_rows in [
        ("Up price", up_q, T_TOTAL),
        ("Down price", dn_q, T_TOTAL),
        ("Net activated volume", net_q, T_TOTAL),
        ("reserve capacity up price", cap_up, 24),
        ("reserve capacity down price", cap_dn, 24),
    ]:
        if df.shape[0] < need_rows:
            raise ValueError(
                f"Sheet '{name}' has {df.shape[0]} rows; expected at least {need_rows}."
            )

    day_cols_up = [c for c in up_q.columns[1:] if str(c).strip() != ""]
    if not day_cols_up:
        raise ValueError("No day columns found in 'Up price' sheet.")

    day_cols_dn = set([c for c in dn_q.columns[1:] if str(c).strip() != ""])
    day_cols_net = set([c for c in net_q.columns[1:] if str(c).strip() != ""])
    day_cols_cap_up = set([c for c in cap_up.columns[1:] if str(c).strip() != ""])
    day_cols_cap_dn = set([c for c in cap_dn.columns[1:] if str(c).strip() != ""])

    day_cols = [
        c for c in day_cols_up
        if (c in day_cols_dn)
        and (c in day_cols_net)
        and (c in day_cols_cap_up)
        and (c in day_cols_cap_dn)
    ]
    if not day_cols:
        raise ValueError("No common day columns across required sheets in the price file.")

    scenarios: List[str] = []
    scenario_to_col: Dict[str, str] = {}

    for c in day_cols:
        try:
            w = pd.to_datetime(c).strftime("%Y-%m-%d")
        except Exception:
            w = str(c)
        scenarios.append(w)
        scenario_to_col[w] = c

    pi_up: Dict[Tuple[int, str], float] = {}
    pi_dn: Dict[Tuple[int, str], float] = {}
    V_up: Dict[Tuple[int, str], float] = {}
    V_dn: Dict[Tuple[int, str], float] = {}
    pi_cap_up: Dict[Tuple[int, str], float] = {}
    pi_cap_dn: Dict[Tuple[int, str], float] = {}

    for w in scenarios:
        c = scenario_to_col[w]

        up_vals = pd.to_numeric(up_q[c].iloc[:T_TOTAL], errors="coerce").fillna(0.0).to_numpy()
        dn_vals = pd.to_numeric(dn_q[c].iloc[:T_TOTAL], errors="coerce").fillna(0.0).to_numpy()
        net_vals = pd.to_numeric(net_q[c].iloc[:T_TOTAL], errors="coerce").fillna(0.0).to_numpy()

        cap_up_vals = pd.to_numeric(cap_up[c].iloc[:24], errors="coerce").fillna(0.0).to_numpy()
        cap_dn_vals = pd.to_numeric(cap_dn[c].iloc[:24], errors="coerce").fillna(0.0).to_numpy()

        for t in range(T_TOTAL):
            pi_up[(t, w)] = float(up_vals[t])
            pi_dn[(t, w)] = float(dn_vals[t])

            net_v = float(net_vals[t])
            if net_v >= 0:
                V_up[(t, w)] = min(float(net_v), 1e9)
                V_dn[(t, w)] = 0.0
            else:
                V_up[(t, w)] = 0.0
                V_dn[(t, w)] = min(float(-net_v), 1e9)

        for h in range(24):
            pi_cap_up[(h, w)] = float(cap_up_vals[h])
            pi_cap_dn[(h, w)] = float(cap_dn_vals[h])

    return scenarios, scenario_to_col, pi_up, pi_dn, V_up, V_dn, pi_cap_up, pi_cap_dn


def load_scenarios_from_excels(here_folder: str, prices_filename: str = "NO3_prices.xlsx"):
    prices_path = os.path.join(here_folder, prices_filename)

    scenarios, scenario_to_col, pi_up, pi_dn, V_up, V_dn, pi_cap_up, pi_cap_dn = _load_mfrr_inputs_from_prices_file(prices_path)
    prob = {w: 1.0 / len(scenarios) for w in scenarios}
    pi_DA = _load_da_prices_from_prices_file(prices_path, scenarios, scenario_to_col)

    return pi_DA, scenarios, prob, pi_up, pi_dn, V_up, V_dn, pi_cap_up, pi_cap_dn


# -----------------------------
# BUILD MODEL
# -----------------------------
def build_two_stage_extensive_form_cvar_wide(
    T: int,
    batt: BatteryParams,
    risk: CVaRParams,
    pi_DA: Dict[Tuple[int, str], float],
    scenarios: List[str],
    prob: Dict[str, float],
    pi_up: Dict[Tuple[int, str], float],
    pi_dn: Dict[Tuple[int, str], float],
    V_up: Dict[Tuple[int, str], float],
    V_dn: Dict[Tuple[int, str], float],
    pi_cap_up: Dict[Tuple[int, str], float],
    pi_cap_dn: Dict[Tuple[int, str], float],
    alpha_up: float = 1.0,
    alpha_dn: float = 1.0,
) -> pyo.ConcreteModel:

    if not (0.0 < risk.alpha < 1.0):
        raise ValueError("CVaR alpha must be in (0,1).")
    if risk.beta < 0.0:
        raise ValueError("CVaR beta must be >= 0.")
    if alpha_up < 0.0 or alpha_dn < 0.0:
        raise ValueError("alpha_up and alpha_dn must be >= 0.")

    H = math.ceil(T / 4)
    D_MAX = batt.P_ch_max + batt.P_dis_max

    m = pyo.ConcreteModel("BATTERY_TWO_STAGE_EF_CVAR_WIDE")

    # Sets
    m.T = pyo.RangeSet(0, T - 1)
    m.H = pyo.RangeSet(0, H - 1)
    m.O = pyo.Set(initialize=scenarios, ordered=True)
    m.E = pyo.RangeSet(0, T)

    # Params
    m.pi_DA = pyo.Param(
        m.T, m.O,
        initialize=lambda mdl, t, w: float(pi_DA.get((int(t), str(w)), 0.0))
    )
    m.p_omega = pyo.Param(
        m.O,
        initialize=lambda mdl, w: float(prob[str(w)]),
        within=pyo.NonNegativeReals
    )

    m.pi_up = pyo.Param(
        m.T, m.O,
        initialize=lambda mdl, t, w: float(pi_up.get((int(t), str(w)), 0.0))
    )
    m.pi_dn = pyo.Param(
        m.T, m.O,
        initialize=lambda mdl, t, w: float(pi_dn.get((int(t), str(w)), 0.0))
    )

    # Exogenous activations clipped to power limits
    m.V_up = pyo.Param(
        m.T, m.O,
        initialize=lambda mdl, t, w: min(
            batt.P_dis_max,
            alpha_up * float(V_up.get((int(t), str(w)), 0.0))
        )
    )
    m.V_dn = pyo.Param(
        m.T, m.O,
        initialize=lambda mdl, t, w: min(
            batt.P_ch_max,
            alpha_dn * float(V_dn.get((int(t), str(w)), 0.0))
        )
    )

    m.pi_cap_up = pyo.Param(
        m.H, m.O,
        initialize=lambda mdl, h, w: float(pi_cap_up.get((int(h), str(w)), 0.0))
    )
    m.pi_cap_dn = pyo.Param(
        m.H, m.O,
        initialize=lambda mdl, h, w: float(pi_cap_dn.get((int(h), str(w)), 0.0))
    )

    # Stage 1
    m.p_DA = pyo.Var(m.T, domain=pyo.Reals, bounds=(-batt.P_ch_max, batt.P_dis_max))
    m.y_up = pyo.Var(m.H, domain=pyo.Binary)
    m.y_dn = pyo.Var(m.H, domain=pyo.Binary)
    m.r_up = pyo.Var(m.H, domain=pyo.NonNegativeReals, bounds=(0, batt.P_dis_max))
    m.r_dn = pyo.Var(m.H, domain=pyo.NonNegativeReals, bounds=(0, batt.P_ch_max))

    m.MinBidUp = pyo.Constraint(
        m.H, rule=lambda mdl, h: mdl.r_up[h] >= batt.Bmin * mdl.y_up[h]
    )
    m.MaxBidUp = pyo.Constraint(
        m.H, rule=lambda mdl, h: mdl.r_up[h] <= batt.P_dis_max * mdl.y_up[h]
    )
    m.MinBidDn = pyo.Constraint(
        m.H, rule=lambda mdl, h: mdl.r_dn[h] >= batt.Bmin * mdl.y_dn[h]
    )
    m.MaxBidDn = pyo.Constraint(
        m.H, rule=lambda mdl, h: mdl.r_dn[h] <= batt.P_ch_max * mdl.y_dn[h]
    )

    if batt.forbid_simultaneous_up_dn_bids:
        m.NoSimultaneousUpDn = pyo.Constraint(
            m.H, rule=lambda mdl, h: mdl.y_up[h] + mdl.y_dn[h] <= 1
        )

    m.Headroom = pyo.Constraint(
        m.T, rule=lambda mdl, t: mdl.p_DA[t] + mdl.r_up[hour_of_t(int(t))] <= batt.P_dis_max
    )
    m.Footroom = pyo.Constraint(
        m.T, rule=lambda mdl, t: mdl.p_DA[t] - mdl.r_dn[hour_of_t(int(t))] >= -batt.P_ch_max
    )

    # Stage 2
    m.p_ch = pyo.Var(m.T, m.O, domain=pyo.NonNegativeReals, bounds=(0, batt.P_ch_max))
    m.p_dis = pyo.Var(m.T, m.O, domain=pyo.NonNegativeReals, bounds=(0, batt.P_dis_max))
    m.e = pyo.Var(m.E, m.O, domain=pyo.Reals, bounds=(batt.E_min, batt.E_max))

    m.d_plus = pyo.Var(m.T, m.O, domain=pyo.NonNegativeReals, bounds=(0, D_MAX))
    m.d_minus = pyo.Var(m.T, m.O, domain=pyo.NonNegativeReals, bounds=(0, D_MAX))

    m.u = pyo.Var(m.T, m.O, domain=pyo.Binary)
    m.NoSimCh = pyo.Constraint(
        m.T, m.O, rule=lambda mdl, t, w: mdl.p_ch[t, w] <= batt.P_ch_max * (1 - mdl.u[t, w])
    )
    m.NoSimDis = pyo.Constraint(
        m.T, m.O, rule=lambda mdl, t, w: mdl.p_dis[t, w] <= batt.P_dis_max * mdl.u[t, w]
    )

    m.InitSOC = pyo.Constraint(
        m.O, rule=lambda mdl, w: mdl.e[0, w] == batt.e_init
    )
    m.TerminalSOC = pyo.Constraint(
        m.O, rule=lambda mdl, w: mdl.e[T, w] == batt.e_init
    )

    m.SOC = pyo.Constraint(
        m.T, m.O,
        rule=lambda mdl, t, w: mdl.e[t + 1, w]
        == mdl.e[t, w]
        + batt.eta_ch * DELTA_H * mdl.p_ch[t, w]
        - (DELTA_H / batt.eta_dis) * mdl.p_dis[t, w]
    )

    m.SocAdeqUp = pyo.Constraint(
        m.T, m.O,
        rule=lambda mdl, t, w: (mdl.e[t, w] - batt.E_min)
        >= (batt.soc_margin_h / batt.eta_dis) * mdl.r_up[hour_of_t(int(t))]
    )
    m.SocAdeqDn = pyo.Constraint(
        m.T, m.O,
        rule=lambda mdl, t, w: (batt.E_max - mdl.e[t, w])
        >= (batt.eta_ch * batt.soc_margin_h) * mdl.r_dn[hour_of_t(int(t))]
    )

    def link_rule(mdl, t, w):
        p_real = mdl.p_dis[t, w] - mdl.p_ch[t, w]
        p_tar = mdl.p_DA[t] + mdl.V_up[t, w] - mdl.V_dn[t, w]
        return p_real == p_tar + mdl.d_plus[t, w] - mdl.d_minus[t, w]

    m.PowerLink = pyo.Constraint(m.T, m.O, rule=link_rule)

    # NOTE: The activation deliverability constraints
    #   0 ≤ V_up + d_plus - d_minus ≤ r_up
    #   0 ≤ V_dn - d_plus + d_minus ≤ r_dn
    # are omitted here because V_up / V_dn are historical scenario data that are
    # NOT conditioned on the model's own bids. When NoSimultaneousUpDn forces
    # r_up=0 in an hour where the data has V_up > r_dn, both bounds conflict and
    # the problem becomes infeasible. Physical deliverability is already ensured by
    # the SoC adequacy constraints (SocAdeqUp / SocAdeqDn), which guarantee the
    # battery has enough stored energy to honour its reserve bids.

    def ramp_up_rule(mdl, t, w):
        if int(t) == 0:
            return pyo.Constraint.Skip
        p_real_t = mdl.p_dis[t, w] - mdl.p_ch[t, w]
        p_real_tm = mdl.p_dis[int(t) - 1, w] - mdl.p_ch[int(t) - 1, w]
        return p_real_t - p_real_tm <= batt.Ramp_MW_per_h * DELTA_H

    def ramp_dn_rule(mdl, t, w):
        if int(t) == 0:
            return pyo.Constraint.Skip
        p_real_t = mdl.p_dis[t, w] - mdl.p_ch[t, w]
        p_real_tm = mdl.p_dis[int(t) - 1, w] - mdl.p_ch[int(t) - 1, w]
        return p_real_tm - p_real_t <= batt.Ramp_MW_per_h * DELTA_H

    m.RampUp = pyo.Constraint(m.T, m.O, rule=ramp_up_rule)
    m.RampDn = pyo.Constraint(m.T, m.O, rule=ramp_dn_rule)

    # Profit + CVaR
    m.Profit = pyo.Var(m.O, domain=pyo.Reals)

    def profit_def_rule(mdl, w):
        da_rev_w = sum(DELTA_H * mdl.pi_DA[t, w] * mdl.p_DA[t] for t in mdl.T)

        cap_rev_w = sum(
            mdl.pi_cap_up[h, w] * mdl.r_up[h] + mdl.pi_cap_dn[h, w] * mdl.r_dn[h]
            for h in mdl.H
        )

        rec_w = 0.0
        for t in mdl.T:
            act_rev = mdl.pi_up[t, w] * mdl.V_up[t, w] - mdl.pi_dn[t, w] * mdl.V_dn[t, w]
            dev_rev = mdl.pi_up[t, w] * mdl.d_plus[t, w] - mdl.pi_dn[t, w] * mdl.d_minus[t, w]

            # Cost(ω) = Δt·Σ_t [ c^tariff·p_ch + c^deg·(p_ch + p_dis) ]
            # Grid tariff applies only to charging power drawn from the grid (not to activations)
            tariff = batt.grid_tariff * mdl.p_ch[t, w]
            degr   = batt.degr_cost_per_mwh * (mdl.p_ch[t, w] + mdl.p_dis[t, w])

            rec_w += DELTA_H * (act_rev + dev_rev - tariff - degr)

        return mdl.Profit[w] == da_rev_w + cap_rev_w + rec_w

    m.ProfitDef = pyo.Constraint(m.O, rule=profit_def_rule)

    m.eta = pyo.Var(domain=pyo.Reals)
    m.xi = pyo.Var(m.O, domain=pyo.NonNegativeReals)

    m.CVaRShortfall = pyo.Constraint(
        m.O, rule=lambda mdl, w: mdl.xi[w] >= mdl.eta - mdl.Profit[w]
    )

    m.CVaR = pyo.Expression(
        expr=m.eta - (1.0 / (1.0 - risk.alpha)) * sum(m.p_omega[w] * m.xi[w] for w in m.O)
    )
    m.ExpProfit = pyo.Expression(
        expr=sum(m.p_omega[w] * m.Profit[w] for w in m.O)
    )

    m.Obj = pyo.Objective(
        expr=(1 - risk.beta) * m.ExpProfit + risk.beta * m.CVaR,
        sense=pyo.maximize
    )

    return m


# -----------------------------
# SOLVER
# -----------------------------
def solve_gurobi(m: pyo.ConcreteModel, tee: bool = True):
    opt = pyo.SolverFactory("gurobi")
    if not opt.available(exception_flag=False):
        raise RuntimeError("Pyomo cannot find the 'gurobi' solver. Check Gurobi installation and PATH.")

    opt.options["DualReductions"] = 0
    opt.options["InfUnbdInfo"] = 1
    opt.options["MIPGap"] = 1e-6
    opt.options["NumericFocus"] = 1

    res = opt.solve(m, tee=tee)
    print(f"Status: {res.solver.status}, Termination: {res.solver.termination_condition}")
    return res


# -----------------------------
# PRINT HELPERS
# -----------------------------
def _fmt(x, nd=3):
    try:
        if x is None:
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _scenario_components(m, batt: BatteryParams, w):
    """Return (DA_rev, Cap_rev, Act_rev, Dev_rev, Cost_tariff, Cost_degr)."""
    da_rev = sum(DELTA_H * pyo.value(m.pi_DA[t, w]) * pyo.value(m.p_DA[t]) for t in m.T)
    cap_rev = sum(
        pyo.value(m.pi_cap_up[h, w]) * pyo.value(m.r_up[h]) +
        pyo.value(m.pi_cap_dn[h, w]) * pyo.value(m.r_dn[h])
        for h in m.H
    )

    act_rev = dev_rev = tariff = degr = 0.0
    for t in m.T:
        act_rev += DELTA_H * (
            pyo.value(m.pi_up[t, w]) * pyo.value(m.V_up[t, w]) -
            pyo.value(m.pi_dn[t, w]) * pyo.value(m.V_dn[t, w])
        )
        dev_rev += DELTA_H * (
            pyo.value(m.pi_up[t, w]) * pyo.value(m.d_plus[t, w]) -
            pyo.value(m.pi_dn[t, w]) * pyo.value(m.d_minus[t, w])
        )
        tariff += DELTA_H * (batt.grid_tariff * pyo.value(m.p_ch[t, w]))
        degr += DELTA_H * (
            batt.degr_cost_per_mwh * (pyo.value(m.p_ch[t, w]) + pyo.value(m.p_dis[t, w]))
        )

    return da_rev, cap_rev, act_rev, dev_rev, tariff, degr


def print_risk_summary(m: pyo.ConcreteModel, risk: CVaRParams):
    print("\n" + "=" * 100)
    print("RISK SUMMARY (CVaR on PROFIT)")
    print("=" * 100)
    print(f"alpha (confidence) : {risk.alpha:.3f}")
    print(f"beta  (weight)     : {risk.beta:.3f}")
    print("-" * 100)
    print(f"Expected profit    : {pyo.value(m.ExpProfit):14.2f}  EUR")
    print(f"VaR level eta      : {pyo.value(m.eta):14.2f}  EUR")
    print(f"CVaR_alpha(profit) : {pyo.value(m.CVaR):14.2f}  EUR")
    print(f"Objective value    : {pyo.value(m.Obj):14.2f}  EUR")


def print_stage1(m: pyo.ConcreteModel, batt: BatteryParams, risk: CVaRParams, max_rows=96):
    print("\n" + "=" * 100)
    print("FIRST STAGE (Here-and-now): Day-Ahead schedule + Reserve bids")
    print("=" * 100)

    exp_da = sum(
        pyo.value(m.p_omega[w]) *
        sum(DELTA_H * pyo.value(m.pi_DA[t, w]) * pyo.value(m.p_DA[t]) for t in m.T)
        for w in m.O
    )
    exp_cap = sum(
        pyo.value(m.p_omega[w]) *
        sum(
            pyo.value(m.pi_cap_up[h, w]) * pyo.value(m.r_up[h]) +
            pyo.value(m.pi_cap_dn[h, w]) * pyo.value(m.r_dn[h])
            for h in m.H
        )
        for w in m.O
    )

    print(f"Expected DA energy revenue     : {exp_da:14.2f}  EUR")
    print(f"Expected reserve capacity rev. : {exp_cap:14.2f}  EUR")
    print(f"Expected profit (ExpProfit)    : {pyo.value(m.ExpProfit):14.2f}  EUR")
    print(f"Objective (Exp + beta*CVaR)    : {pyo.value(m.Obj):14.2f}  EUR")
    print_risk_summary(m, risk)

    print("\nRESERVE BIDS (hourly)")
    print("  h | y_up  r_up[MW] | y_dn  r_dn[MW]")
    print("----+---------------+---------------")
    for h in m.H:
        print(
            f"{int(h):3d} |"
            f"  {int(round(pyo.value(m.y_up[h]))):1d}   {_fmt(pyo.value(m.r_up[h]), 2):>8} |"
            f"  {int(round(pyo.value(m.y_dn[h]))):1d}   {_fmt(pyo.value(m.r_dn[h]), 2):>8}"
        )

    print("\nDAY-AHEAD DISPATCH p_DA (15-min)")
    print("  t | hr |   p_DA [MW]")
    print("----+----+------------")
    for t in range(min(max_rows, len(list(m.T)))):
        print(f"{t:3d} | {hour_of_t(t):2d} | {_fmt(pyo.value(m.p_DA[t]), 3):>10}")


def print_scenario_profit_table(m: pyo.ConcreteModel, max_rows: int = 50):
    print("\n" + "=" * 100)
    print("SCENARIO PROFITS + CVaR SHORTFALLS")
    print("=" * 100)
    print(" idx | scenario        |  p(w)  |   Profit[EUR] |    xi[EUR]")
    print("-----+-----------------+--------+--------------+-----------")

    O = list(m.O)
    for i, w in enumerate(O[:max_rows]):
        print(
            f"{i:4d} | {str(w):15s} |"
            f" {_fmt(pyo.value(m.p_omega[w]), 4):>6} |"
            f" {_fmt(pyo.value(m.Profit[w]), 2):>12} |"
            f" {_fmt(pyo.value(m.xi[w]), 2):>9}"
        )

    if len(O) > max_rows:
        print(f"... ({len(O) - max_rows} more scenarios not shown)")


def print_value_stacking(m: pyo.ConcreteModel, batt: BatteryParams, w):
    da_rev, cap_rev, act_rev, dev_rev, tariff, degr = _scenario_components(m, batt, w)
    profit = pyo.value(m.Profit[w])

    print("\n" + "=" * 100)
    print(f"VALUE STACKING (scenario ω = {w})")
    print("=" * 100)
    print(f"DA energy revenue          : {da_rev:14.2f}  EUR")
    print(f"Reserve capacity revenue   : {cap_rev:14.2f}  EUR")
    print(f"Activation settlement      : {act_rev:14.2f}  EUR")
    print(f"Deviation settlement       : {dev_rev:14.2f}  EUR")
    print(f"Charging tariff cost       : {-tariff:14.2f}  EUR")
    print(f"Degradation cost           : {-degr:14.2f}  EUR")
    print("-" * 100)
    print(f"Profit[ω]                  : {profit:14.2f}  EUR")


def print_stage2_dispatch(m: pyo.ConcreteModel, w, max_rows=96):
    print("\n" + "=" * 110)
    print(f"SECOND STAGE DISPATCH (scenario ω = {w})")
    print("=" * 110)

    print("  t |hr|  p_DA |  V_up  V_dn | p_real |  p_ch  p_dis |   d+    d-  |  SOC | u")
    print("----+--+------+------------+--------+------------+------------+------+---")
    for t in range(min(max_rows, len(list(m.T)))):
        h = hour_of_t(int(t))
        p_DA_t = pyo.value(m.p_DA[t])
        v_up = pyo.value(m.V_up[t, w])
        v_dn = pyo.value(m.V_dn[t, w])
        p_real = pyo.value(m.p_dis[t, w]) - pyo.value(m.p_ch[t, w])
        soc = pyo.value(m.e[t, w])
        u = int(round(pyo.value(m.u[t, w])))

        print(
            f"{int(t):3d} |{h:2d}|"
            f"{_fmt(p_DA_t, 3):>6} |"
            f"{_fmt(v_up, 3):>6} {_fmt(v_dn, 3):>6} |"
            f"{_fmt(p_real, 3):>6} |"
            f"{_fmt(pyo.value(m.p_ch[t, w]), 3):>5} {_fmt(pyo.value(m.p_dis[t, w]), 3):>6} |"
            f"{_fmt(pyo.value(m.d_plus[t, w]), 3):>6} {_fmt(pyo.value(m.d_minus[t, w]), 3):>6} |"
            f"{_fmt(soc, 2):>5} | {u}"
        )


# -----------------------------
# PLOTTING HELPERS
# -----------------------------
def _series_var_1d(var, T: int):
    return [float(pyo.value(var[t])) for t in range(T)]


def _series_var_2d(var, w, T: int):
    return [float(pyo.value(var[t, w])) for t in range(T)]


def _series_param_2d(par, w, T: int):
    return [float(pyo.value(par[t, w])) for t in range(T)]


def _hourly_var(var, H: int):
    return [float(pyo.value(var[h])) for h in range(H)]


# ── colour palette ─────────────────────────────────────────────────────────────
_C = dict(
    navy   = "#1F4E84",
    blue   = "#2A78D6",
    teal   = "#1D9E75",
    amber  = "#EDA100",
    coral  = "#EB6834",
    red    = "#D03B3B",
    purple = "#7D3C98",
    grey   = "#52514E",
    muted  = "#898781",
    bg     = "#FAFAFA",
)


# ── helpers ────────────────────────────────────────────────────────────────────
def _val(expr):
    return float(pyo.value(expr))


def _series1d(var, keys):
    return [_val(var[k]) for k in keys]


def _series2d(var, keys, w):
    return [_val(var[k, w]) for k in keys]


def _scenario_components_plot(m, batt, w):
    """Return (da_rev, cap_rev, act_rev, dev_rev, tariff, degr) for scenario w."""
    T = list(m.T)
    H = list(m.H)
    da_rev  = sum(DELTA_H * _val(m.pi_DA[t, w]) * _val(m.p_DA[t]) for t in T)
    cap_rev = sum(_val(m.pi_cap_up[h, w]) * _val(m.r_up[h])
                + _val(m.pi_cap_dn[h, w]) * _val(m.r_dn[h]) for h in H)
    act_rev = dev_rev = tariff = degr = 0.0
    for t in T:
        act_rev += DELTA_H * (_val(m.pi_up[t, w]) * _val(m.V_up[t, w])
                             - _val(m.pi_dn[t, w]) * _val(m.V_dn[t, w]))
        dev_rev += DELTA_H * (_val(m.pi_up[t, w]) * _val(m.d_plus[t, w])
                             - _val(m.pi_dn[t, w]) * _val(m.d_minus[t, w]))
        tariff  += DELTA_H * batt.grid_tariff * _val(m.p_ch[t, w])
        degr    += DELTA_H * batt.degr_cost_per_mwh * (_val(m.p_ch[t, w])
                                                       + _val(m.p_dis[t, w]))
    return da_rev, cap_rev, act_rev, dev_rev, tariff, degr


def _style(ax, title="", xlabel="", ylabel="", grid=True, fmteur=False):
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=_C["navy"],
                 loc="left", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_facecolor(_C["bg"])
    if grid:
        ax.grid(True, alpha=0.22, color="#CCCCCC")
    if fmteur:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))


# ── panel A: stage-1 bids ──────────────────────────────────────────────────────
def _panel_stage1(ax, m):
    T = list(m.T)
    H = list(m.H)
    t_h  = [t * DELTA_H for t in T]
    p_DA = _series1d(m.p_DA, T)
    r_up = _series1d(m.r_up, H)
    r_dn = _series1d(m.r_dn, H)

    colours = [_C["coral"] if v > 0 else _C["blue"] for v in p_DA]
    ax.bar(t_h, p_DA, width=DELTA_H * 0.92, color=colours, alpha=0.82, zorder=3)
    ax.axhline(0, color=_C["grey"], lw=0.6)

    ax2 = ax.twinx()
    h_h = [h + 0.5 for h in H]
    ax2.step(h_h, r_up, where="mid", color=_C["teal"],  lw=2.0, label="r_up (MW)")
    ax2.step(h_h, r_dn, where="mid", color=_C["amber"], lw=2.0, ls="--", label="r_dn (MW)")
    ax2.set_ylabel("Reserve bid (MW)", fontsize=9)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.85)

    from matplotlib.patches import Patch
    leg = [Patch(facecolor=_C["coral"], alpha=0.82, label="DA discharge"),
           Patch(facecolor=_C["blue"],  alpha=0.82, label="DA charge")]
    ax.legend(handles=leg, fontsize=8, loc="lower right", framealpha=0.85)
    ax.set_xlim(-0.3, len(H) + 0.3)
    _style(ax, "(A)  Stage-1 bids", "Hour of day", "p_DA (MW)")


# ── panel B: SOC trajectories ──────────────────────────────────────────────────
def _panel_soc(ax, m, scenarios):
    T = list(m.T)
    palette = [_C["blue"], _C["teal"], _C["coral"], _C["amber"],
               _C["purple"], _C["red"], "#4A90D9", "#27AE60", "#E67E22"]
    t_soc = [(t + 1) * DELTA_H for t in T]
    for i, w in enumerate(scenarios):
        soc = [_val(m.e[t + 1, w]) for t in T]
        ax.plot(t_soc, soc, color=palette[i % len(palette)],
                lw=1.6, alpha=0.85, label=str(w))
    e_init = _val(m.e[0, scenarios[0]])
    ax.axhline(e_init, color=_C["grey"], ls=":", lw=1.0, label=f"e_init = {e_init:.0f} MWh")
    e_max = m.e[0, scenarios[0]].ub
    ax.set_ylim(-1, e_max + 4)
    ax.set_xlim(0, len(T) * DELTA_H)
    ax.text(0.01, 0.97, f"E_max = {e_max:.0f} MWh", transform=ax.transAxes,
            fontsize=8, color=_C["muted"], va="top", style="italic")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.85,
              title="Scenario", title_fontsize=8)
    _style(ax, "(B)  Battery SOC trajectories", "Hour of day", "SOC (MWh)")


# ── panel C: profit distribution ───────────────────────────────────────────────
def _panel_profits(ax, m, risk, scenarios):
    profits  = [_val(m.Profit[w]) for w in scenarios]
    exp_val  = _val(m.ExpProfit)
    cvar_val = _val(m.CVaR)
    eta_val  = _val(m.eta)

    short_labels = [str(w)[-10:] for w in scenarios]
    bar_col = [_C["teal"] if p >= 0 else _C["red"] for p in profits]
    ax.bar(short_labels, profits, color=bar_col, alpha=0.85,
           edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(exp_val,  color=_C["navy"], lw=2.0,
               label=f"E[Profit] = {exp_val:,.0f} €")
    ax.axhline(eta_val,  color=_C["amber"], lw=1.6, ls="-.",
               label=f"VaR (η) = {eta_val:,.0f} €")
    ax.axhline(cvar_val, color=_C["red"], lw=1.8, ls="--",
               label=f"CVaR₍{risk.alpha:.2f}₎ = {cvar_val:,.0f} €")
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.legend(fontsize=8, framealpha=0.9)
    _style(ax, "(C)  Scenario profit distribution", "Scenario", "Profit (EUR)", fmteur=True)


# ── panel D: revenue waterfall ─────────────────────────────────────────────────
def _panel_waterfall(ax, m, batt, scenarios):
    n = len(scenarios)
    means = dict(da=0, cap=0, act=0, dev=0, tariff=0, degr=0)
    for w in scenarios:
        da, cap, act, dev, tar, dgr = _scenario_components_plot(m, batt, w)
        means["da"]     += da / n;  means["cap"]    += cap / n
        means["act"]    += act / n; means["dev"]    += dev / n
        means["tariff"] += tar / n; means["degr"]   += dgr / n

    labels = ["DA\nenergy", "Capacity\nreserve", "Activation", "Deviation",
              "Tariff\ncost", "Degr.\ncost"]
    vals   = [means["da"], means["cap"], means["act"], means["dev"],
              -means["tariff"], -means["degr"]]
    cols   = [_C["blue"], _C["teal"], _C["amber"], _C["coral"], _C["red"], _C["red"]]

    running, bottoms = 0.0, []
    for v in vals:
        bottoms.append(running if v >= 0 else running + v)
        running += v
    net = running

    ax.bar(labels, [abs(v) for v in vals], bottom=bottoms,
           color=cols, alpha=0.85, edgecolor="white", lw=0.5, zorder=3)
    ax.bar(["Net\nprofit"], [abs(net)],
           bottom=[0 if net >= 0 else net],
           color=_C["navy"], alpha=0.88, edgecolor="white", lw=0.5, zorder=3)

    run2 = 0.0
    for i, v in enumerate(vals):
        run2 += v
        ax.text(i, run2 + (max(abs(net), 50) * 0.02 * (1 if v >= 0 else -1)),
                f"{v:+,.0f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7.5, fontweight="bold", color=_C["grey"])
    ax.text(len(labels), net + (max(abs(net), 50) * 0.02 * (1 if net >= 0 else -1)),
            f"{net:+,.0f}", ha="center", va="bottom" if net >= 0 else "top",
            fontsize=8, fontweight="bold", color=_C["navy"])

    ax.axhline(0, color=_C["grey"], lw=0.7)
    ax.set_xticks(range(len(labels) + 1))
    ax.set_xticklabels(labels + ["Net\nprofit"])
    _style(ax, "(D)  Revenue stacking (mean, EUR)", "", "EUR", fmteur=True)


# ── panel E: beta sensitivity ──────────────────────────────────────────────────
def _panel_beta(ax, m, risk, scenarios):
    profits  = np.array([_val(m.Profit[w]) for w in scenarios])
    probs    = np.array([_val(m.p_omega[w]) for w in scenarios])
    exp_val  = float(np.dot(probs, profits))
    cvar_val = _val(m.CVaR)
    betas    = np.linspace(0, 1, 21)
    obj_arr  = (1 - betas) * exp_val + betas * cvar_val

    ax.plot(betas, obj_arr, color=_C["navy"], lw=2.2, marker="o", ms=4,
            label="(1−β)·E[P] + β·CVaR")
    ax.axhline(exp_val,  color=_C["blue"], lw=1.3, ls="--",
               label=f"E[Profit] = {exp_val:,.0f} €")
    ax.axhline(cvar_val, color=_C["red"],  lw=1.3, ls="--",
               label=f"CVaR = {cvar_val:,.0f} €")
    ax.fill_between(betas, cvar_val, obj_arr, alpha=0.10,
                    color=_C["red"], label="Risk cost")
    cur_obj = (1 - risk.beta) * exp_val + risk.beta * cvar_val
    ax.axvline(risk.beta, color=_C["amber"], lw=1.2, ls=":",
               label=f"β = {risk.beta} (current)")
    ax.scatter([risk.beta], [cur_obj], color=_C["amber"], s=70, zorder=5)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    _style(ax, "(E)  β sensitivity (mean–CVaR trade-off)",
           "β (risk-aversion weight)", "Objective value (EUR)", fmteur=True)


# ── panel F: summary card ──────────────────────────────────────────────────────
def _panel_summary(ax, m, batt, risk, scenarios):
    ax.axis("off")
    profits   = [_val(m.Profit[w]) for w in scenarios]
    exp_val   = _val(m.ExpProfit)
    cvar_val  = _val(m.CVaR)
    eta_val   = _val(m.eta)
    obj_val   = _val(m.Obj)
    r_up_vals = [_val(m.r_up[h]) for h in m.H]
    r_dn_vals = [_val(m.r_dn[h]) for h in m.H]

    rows = [
        ("Battery",               f"{batt.P_dis_max:.0f} MW  /  {batt.E_max:.0f} MWh",                        _C["navy"]),
        ("Scenarios",             f"{len(scenarios)}  (πω = {1/len(scenarios):.3f})",                _C["navy"]),
        ("α (confidence)",   f"{risk.alpha:.2f}   →  worst {(1-risk.alpha)*100:.0f}%",               _C["navy"]),
        ("β (risk weight)",  f"{risk.beta:.2f}   →  {'risk-neutral' if risk.beta==0 else 'risk-averse'}", _C["navy"]),
        ("", "", "white"),
        ("E[Profit]",             f"{exp_val:+,.1f}  EUR",    _C["blue"]),
        ("VaR (η)",          f"{eta_val:+,.1f}  EUR",    _C["amber"]),
        (f"CVaR₍{risk.alpha:.2f}₎", f"{cvar_val:+,.1f}  EUR", _C["red"]),
        ("Objective",             f"{obj_val:+,.1f}  EUR",    _C["navy"]),
        ("", "", "white"),
        ("Best scenario",         f"{max(profits):+,.1f}  EUR",  _C["teal"]),
        ("Worst scenario",        f"{min(profits):+,.1f}  EUR",  _C["red"]),
        ("", "", "white"),
        ("Max r_up bid",          f"{max(r_up_vals):.2f} MW  (h={r_up_vals.index(max(r_up_vals))})", _C["teal"]),
        ("Max r_dn bid",          f"{max(r_dn_vals):.2f} MW  (h={r_dn_vals.index(max(r_dn_vals))})", _C["amber"]),
    ]
    ROW_H = 0.062
    for i, (lbl, val, col) in enumerate(rows):
        y = 0.98 - i * ROW_H
        if lbl:
            ax.text(0.01, y, lbl, transform=ax.transAxes, fontsize=9.5,
                    fontweight="bold", color=_C["grey"], va="top")
            ax.text(0.55, y, val, transform=ax.transAxes, fontsize=9.5,
                    color=col, va="top", fontweight="bold")
    ax.set_title("(F)  Model summary & risk measures", fontsize=10.5,
                 fontweight="bold", color=_C["navy"], loc="left", pad=6)


# ── main dashboard function ────────────────────────────────────────────────────
def plot_cvar_dashboard(
    m: pyo.ConcreteModel,
    batt: "BatteryParams",
    risk: "CVaRParams",
    scenarios,
    save_path: str = "battery_cvar_results.png",
    dpi: int = 180,
    show: bool = True,
):
    """
    Generate a 6-panel publication-quality figure from a solved Battery_CVaR model.

    Panels:
      A – Stage-1 bids (DA schedule + reserve bids)
      B – SOC trajectories across all scenarios
      C – Scenario profit distribution + VaR / CVaR lines
      D – Revenue stacking waterfall (mean across scenarios)
      E – β sensitivity: objective vs risk-aversion weight
      F – Key-numbers summary card
    """
    fig = plt.figure(figsize=(15, 9.8))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.46, wspace=0.38,
        left=0.06, right=0.97, top=0.88, bottom=0.07,
    )

    _panel_stage1(  fig.add_subplot(gs[0, 0]), m)
    _panel_soc(     fig.add_subplot(gs[0, 1]), m, list(scenarios))
    _panel_profits( fig.add_subplot(gs[0, 2]), m, risk, list(scenarios))
    _panel_waterfall(fig.add_subplot(gs[1, 0]), m, batt, list(scenarios))
    _panel_beta(    fig.add_subplot(gs[1, 1]), m, risk, list(scenarios))
    _panel_summary( fig.add_subplot(gs[1, 2]), m, batt, risk, list(scenarios))

    fig.text(0.5, 0.955,
             "Battery_CVaR.py  —  Two-stage stochastic MILP results",
             ha="center", fontsize=15, fontweight="bold", color=_C["navy"])
    fig.text(
        0.5, 0.934,
        (f"DA + mFRR multi-market  ·  {batt.P_dis_max:.0f} MW / {batt.E_max:.0f} MWh  ·  "
         f"{len(list(scenarios))} scenarios  ·  α = {risk.alpha}"
         f"  ·  β = {risk.beta}"),
        ha="center", fontsize=10, color=_C["muted"],
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Figure saved → {save_path}")

    if show:
        plt.show()

    return fig


# ── standalone figure functions (one figure per panel) ─────────────────────────
def _make_fig(title_main, title_sub, figsize=(9, 5.8)):
    """Create a white figure with a two-line suptitle in the lecture deck style."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.suptitle(title_main, fontsize=13, fontweight="bold",
                 color=_C["navy"], y=0.97)
    fig.text(0.5, 0.91, title_sub, ha="center", fontsize=9, color=_C["muted"])
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.11, right=0.93)
    return fig, ax


def plot_fig_stage1(m, batt, risk, scenarios,
                    save_path="fig1_stage1_bids.png", dpi=180, show=True):
    """Figure 1 — Stage-1 DA schedule and reserve bids."""
    fig, ax = _make_fig(
        "Stage-1 bids: DA schedule and mFRR reserve capacity",
        f"{batt.P_dis_max:.0f} MW / {batt.E_max:.0f} MWh  ·  "
        f"{len(list(scenarios))} scenarios  ·  β = {risk.beta}",
    )
    _panel_stage1(ax, m)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {save_path}")
    if show: plt.show()
    return fig


def plot_fig_soc(m, batt, risk, scenarios,
                 save_path="fig2_soc_trajectories.png", dpi=180, show=True):
    """Figure 2 — Battery SOC trajectories across all scenarios."""
    fig, ax = _make_fig(
        "Battery state of charge — scenario trajectories",
        f"{batt.P_dis_max:.0f} MW / {batt.E_max:.0f} MWh  ·  "
        f"η_ch = {batt.eta_ch}  ·  η_dis = {batt.eta_dis}  ·  e_init = {batt.e_init:.0f} MWh",
        figsize=(9, 5.8),
    )
    _panel_soc(ax, m, list(scenarios))
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {save_path}")
    if show: plt.show()
    return fig


def plot_fig_profits(m, batt, risk, scenarios,
                     save_path="fig3_profit_distribution.png", dpi=180, show=True):
    """Figure 3 — Scenario profit distribution with VaR and CVaR."""
    fig, ax = _make_fig(
        "Scenario profit distribution",
        f"α = {risk.alpha}  ·  β = {risk.beta}  ·  "
        f"CVaR_{risk.alpha} = avg profit of worst {(1-risk.alpha)*100:.0f}% scenarios",
    )
    _panel_profits(ax, m, risk, list(scenarios))
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {save_path}")
    if show: plt.show()
    return fig


def plot_fig_waterfall(m, batt, risk, scenarios,
                       save_path="fig4_revenue_waterfall.png", dpi=180, show=True):
    """Figure 4 — Revenue stacking waterfall (mean across scenarios)."""
    fig, ax = _make_fig(
        "Revenue stacking — mean across scenarios (EUR)",
        f"DA energy + capacity reserve + activation + deviation settlement − costs",
        figsize=(10, 5.8),
    )
    ax.set_position([0.08, 0.13, 0.88, 0.75])   # give more horizontal room
    _panel_waterfall(ax, m, batt, list(scenarios))
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {save_path}")
    if show: plt.show()
    return fig


def plot_fig_beta(m, batt, risk, scenarios,
                  save_path="fig5_beta_sensitivity.png", dpi=180, show=True):
    """Figure 5 — β sensitivity: mean–CVaR trade-off."""
    fig, ax = _make_fig(
        "Risk-aversion sweep: β sensitivity",
        f"Objective = (1−β)·E[Profit] + β·CVaR_{risk.alpha}  ·  "
        f"β = {risk.beta} (current run highlighted)",
    )
    _panel_beta(ax, m, risk, list(scenarios))
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {save_path}")
    if show: plt.show()
    return fig


def plot_fig_summary(m, batt, risk, scenarios,
                     save_path="fig6_summary.png", dpi=180, show=True):
    """Figure 6 — Model summary and risk measures card."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    _panel_summary(ax, m, batt, risk, list(scenarios))
    fig.suptitle("Battery_CVaR.py — Key results", fontsize=13,
                 fontweight="bold", color=_C["navy"], y=0.97)
    fig.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {save_path}")
    if show: plt.show()
    return fig


def plot_all_figures(m, batt, risk, scenarios,
                     save_dir=".", prefix="battery", dpi=180, show=False):
    """
    Generate all six figures individually and save them to save_dir.

    Parameters
    ----------
    m         : solved Pyomo model
    batt      : BatteryParams
    risk      : CVaRParams
    scenarios : list of scenario keys
    save_dir  : folder for output files (created if missing)
    prefix    : filename prefix  (e.g. 'battery' → 'battery_fig1_stage1_bids.png')
    dpi       : output resolution
    show      : call plt.show() after each figure (False = silent batch mode)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    def _path(name):
        return os.path.join(save_dir, f"{prefix}_{name}") if save_dir != "." \
               else f"{prefix}_{name}"

    figs = []
    figs.append(plot_fig_stage1(  m, batt, risk, scenarios, _path("fig1_stage1_bids.png"),       dpi, show))
    figs.append(plot_fig_soc(     m, batt, risk, scenarios, _path("fig2_soc_trajectories.png"),   dpi, show))
    figs.append(plot_fig_profits( m, batt, risk, scenarios, _path("fig3_profit_distribution.png"),dpi, show))
    figs.append(plot_fig_waterfall(m, batt, risk, scenarios,_path("fig4_revenue_waterfall.png"),  dpi, show))
    figs.append(plot_fig_beta(    m, batt, risk, scenarios, _path("fig5_beta_sensitivity.png"),   dpi, show))
    figs.append(plot_fig_summary( m, batt, risk, scenarios, _path("fig6_summary.png"),            dpi, show))
    print(f"\nAll 6 figures saved to: {os.path.abspath(save_dir)}/")
    return figs




# -------------------------------------------------------------
# TERMINAL SETTINGS WILL BE UPDATED BY USER
# ----------------------------------------------------------------
T_TOTAL = 96
DELTA_H = 0.25  # 15 min in hours


@dataclass
class BatteryParams:
    # Power limits (MW)
    P_ch_max: float = 10.0
    P_dis_max: float = 10.0

    # Energy limits (MWh)
    E_min: float = 0.0
    E_max: float = 50.0
    e_init: float = 30.0

    # Efficiencies
    eta_ch: float = 0.98
    eta_dis: float = 0.98

    # Reserve bid minimum (MW)
    Bmin: float = 1.0

    # Deliverability margin (hours)
    soc_margin_h: float = 0.25

    # Ramp limit for realized power (MW/h)
    Ramp_MW_per_h: float = 999.0

    # Costs
    grid_tariff: float = 43.5
    degr_cost_per_mwh: float = 10.0
    # Example note:
    # battery is 10 MW / 50 MWh, usable 80% => E_use = 40 MWh.
    # Assume N = 6000 cycles. Lifetime throughput = 2 * 40 * 6000 = 480000 MWh.
    # If replacement/cell wear cost allocated to cycling is €5,000,000:
    # c_deg = 5000000 / 480000 ≈ 10.4 €/MWh

    # Switch
    forbid_simultaneous_up_dn_bids: bool = True


@dataclass
class CVaRParams:
    alpha: float
    beta: float 


def hour_of_t(t: int) -> int:
    return t // 4


# -------------------------------------------------------------------------------------------
# BETA SWEEP — efficient frontier
# -------------------------------------------------------------------------------------------
def run_beta_sweep(
    batt,
    alpha: float,
    betas,
    pi_DA, scenarios, prob,
    pi_up, pi_dn, V_up, V_dn,
    pi_cap_up, pi_cap_dn,
    alpha_up: float = 0.01,
    alpha_dn: float = 0.01,
    tee: bool = False,
):
    """
    Solve the CVaR model for each value in `betas` and return a list of
    result dicts with keys: beta, exp_profit, cvar, var_eta, obj, solved.

    Usage
    -----
        results = run_beta_sweep(batt, alpha=0.95,
                                 betas=[0, 0.1, 0.25, 0.5, 0.75, 1.0],
                                 pi_DA=..., scenarios=..., ...)
        plot_efficient_frontier(results, alpha=0.95,
                                save_path="efficient_frontier.png")
    """
    records = []
    for b in betas:
        print(f"\n{'='*60}")
        print(f"  Beta sweep:  β = {b:.3f}  (α = {alpha})")
        print(f"{'='*60}")
        risk_b = CVaRParams(alpha=alpha, beta=b)
        m_b = build_two_stage_extensive_form_cvar_wide(
            T=T_TOTAL, batt=batt, risk=risk_b,
            pi_DA=pi_DA, scenarios=scenarios, prob=prob,
            pi_up=pi_up, pi_dn=pi_dn, V_up=V_up, V_dn=V_dn,
            pi_cap_up=pi_cap_up, pi_cap_dn=pi_cap_dn,
            alpha_up=alpha_up, alpha_dn=alpha_dn,
        )
        res_b = solve_gurobi(m_b, tee=tee)
        tc = str(res_b.solver.termination_condition).lower()
        solved = not ("infeasible" in tc or "unbounded" in tc or "unknown" in tc)
        if solved:
            ep   = float(pyo.value(m_b.ExpProfit))
            cvar = float(pyo.value(m_b.CVaR))
            eta  = float(pyo.value(m_b.eta))
            obj  = float(pyo.value(m_b.Obj))
            print(f"  → E[Profit]={ep:.1f}  CVaR={cvar:.1f}  η={eta:.1f}  Obj={obj:.1f}")
        else:
            ep = cvar = eta = obj = float("nan")
            print(f"  → NOT SOLVED ({tc})")
        records.append(dict(beta=b, exp_profit=ep, cvar=cvar,
                            var_eta=eta, obj=obj, solved=solved))
    return records


def plot_efficient_frontier(
    results,
    alpha: float = 0.95,
    save_path: str = "fig7_efficient_frontier.png",
    dpi: int = 180,
    show: bool = True,
):
    """
    Plot the mean–CVaR efficient frontier from the output of run_beta_sweep().

    Each solved (CVaR, E[Profit]) point is plotted and labelled with its β value.
    The curve connects them in order of increasing β.

    Parameters
    ----------
    results   : list of dicts returned by run_beta_sweep()
    alpha     : CVaR confidence level (for axis label)
    save_path : output file path  ('' to skip saving)
    dpi       : output resolution
    show      : call plt.show()
    """
    # Filter to solved points
    pts = [(r["cvar"], r["exp_profit"], r["beta"])
           for r in results if r["solved"] and not math.isnan(r["cvar"])]
    if not pts:
        print("No solved points to plot.")
        return None

    cvar_arr = [p[0] for p in pts]
    ep_arr   = [p[1] for p in pts]
    beta_arr = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    fig.patch.set_facecolor("white")

    # ── frontier curve ────────────────────────────────────────────────────────
    ax.plot(cvar_arr, ep_arr,
            color=_C["navy"], lw=2.2, zorder=2,
            marker="o", ms=8, mfc="white", mec=_C["navy"], mew=2.0)

    # ── colour-coded dots by beta ─────────────────────────────────────────────
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = plt.colormaps["RdYlGn"].resampled(len(pts))
    for i, (cx, ey, b) in enumerate(pts):
        ax.scatter(cx, ey, color=cmap(i), s=110, zorder=4,
                   edgecolors=_C["navy"], linewidths=1.2)

    # ── β labels ──────────────────────────────────────────────────────────────
    # compute offset direction per point to avoid label overlap
    for i, (cx, ey, b) in enumerate(pts):
        # alternate label above/below based on index
        va  = "bottom" if i % 2 == 0 else "top"
        dy  = 0.02 * (max(ep_arr) - min(ep_arr)) * (1 if va == "bottom" else -1)
        ax.annotate(
            f"β = {b:.2f}",
            xy=(cx, ey),
            xytext=(cx, ey + dy),
            ha="center", va=va,
            fontsize=9, fontweight="bold", color=_C["navy"],
            arrowprops=dict(arrowstyle="-", color=_C["muted"], lw=0.8),
        )

    # ── directional arrows annotation ─────────────────────────────────────────
    if len(pts) >= 2:
        # Arrow near β=0 label
        ax.annotate("", xy=(cvar_arr[1], ep_arr[1]),
                    xytext=(cvar_arr[0], ep_arr[0]),
                    arrowprops=dict(arrowstyle="-|>", color=_C["muted"],
                                   lw=1.2, mutation_scale=14))

    # ── shaded desirable region ───────────────────────────────────────────────
    ax.fill_betweenx(
        [min(ep_arr), max(ep_arr)],
        max(cvar_arr), max(cvar_arr) * 1.12 if max(cvar_arr) > 0 else max(cvar_arr) * 0.88,
        alpha=0.06, color=_C["teal"], label="Higher CVaR → better tail protection",
    )

    # ── axes & decoration ────────────────────────────────────────────────────
    pad_x = abs(max(cvar_arr) - min(cvar_arr)) * 0.12 or abs(max(cvar_arr)) * 0.05 or 50
    pad_y = abs(max(ep_arr)   - min(ep_arr))   * 0.18 or abs(max(ep_arr))   * 0.05 or 50
    ax.set_xlim(min(cvar_arr) - pad_x, max(cvar_arr) + pad_x * 1.5)
    ax.set_ylim(min(ep_arr)   - pad_y, max(ep_arr)   + pad_y)

    ax.set_xlabel(f"CVaR$_{{α={alpha}}}$  (EUR)  →  higher is better (less negative tail)",
                  fontsize=10)
    ax.set_ylabel("E[Profit]  (EUR)  →  higher is better", fontsize=10)
    ax.set_title("Efficient frontier: Expected profit vs CVaR",
                 fontsize=13, fontweight="bold", color=_C["navy"], pad=10)
    ax.set_facecolor(_C["bg"])
    ax.grid(True, alpha=0.22, color="#CCCCCC")
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ── colour bar (β scale) ──────────────────────────────────────────────────
    import matplotlib.colors as mcolors
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps["RdYlGn"].resampled(len(pts)),
                               norm=mcolors.Normalize(vmin=min(beta_arr),
                                                      vmax=max(beta_arr)))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("β  (risk-aversion weight)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ── text notes ────────────────────────────────────────────────────────────
    ax.text(0.01, 0.99,
            f"α = {alpha}  ·  {len(pts)} solved β values  ·  "
            f"Battery: 10 MW / 50 MWh  ·  NO3 market",
            transform=ax.transAxes, fontsize=8, color=_C["muted"],
            va="top", style="italic")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    batt = BatteryParams(
        soc_margin_h=0.25,                 # 15-min endurance
        Ramp_MW_per_h=999.0,               # set e.g. 60 for 1 MW/min
        forbid_simultaneous_up_dn_bids=True,
    )

    # CVaR controls — adjust these two values to change the risk profile
    alpha = 0.95   # confidence level: CVaR_alpha = avg profit of worst (1-alpha) fraction
    beta  = 0.5    # risk-aversion weight: 0 = risk-neutral, 1 = fully risk-averse
    risk  = CVaRParams(alpha=alpha, beta=beta)

    # Activation scaling factors
    alpha_up = 0.01  # scaling factor for up reserve
    alpha_dn = 0.01  # scaling factor for down reserve

    SAVE_PLOTS = True   # set False to skip saving the figure to disk
    PLOT_WORST_SCENARIO = False
    # ── Efficient frontier sweep ───────────────────────────────────────────────
    # Set BETA_SWEEP = True to solve the model for every β in BETA_VALUES and
    # generate the efficient frontier plot (fig7_efficient_frontier.png).
    # This runs one full MILP per β value — expect several minutes for 10+ values.
    BETA_SWEEP       = True
    BETA_VALUES      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    here = os.path.dirname(os.path.abspath(__file__))

    (pi_DA, scenarios, prob,
     pi_up, pi_dn, V_up, V_dn,
     pi_cap_up, pi_cap_dn) = load_scenarios_from_excels(here)

    m = build_two_stage_extensive_form_cvar_wide(
        T=T_TOTAL,
        batt=batt,
        risk=risk,
        pi_DA=pi_DA,
        scenarios=scenarios,
        prob=prob,
        pi_up=pi_up,
        pi_dn=pi_dn,
        V_up=V_up,
        V_dn=V_dn,
        pi_cap_up=pi_cap_up,
        pi_cap_dn=pi_cap_dn,
        alpha_up=alpha_up,
        alpha_dn=alpha_dn,
    )

    res = solve_gurobi(m, tee=True)

    tc = str(res.solver.termination_condition).lower()
    if "infeasible" in tc or "unbounded" in tc or "unknown" in tc:
        print("Model not solved to optimality; skipping result printing/plotting.")
    else:
        print_stage1(m, batt=batt, risk=risk, max_rows=96)
        print_scenario_profit_table(m, max_rows=50)

        if PLOT_WORST_SCENARIO:
            w0 = min(list(m.O), key=lambda ww: pyo.value(m.Profit[ww]))
        else:
            w0 = list(m.O)[0]

        print(f"\nUsing activation scaling factors: alpha_up = {alpha_up}, alpha_dn = {alpha_dn}")

        print_value_stacking(m, batt=batt, w=w0)
        print_stage2_dispatch(m, w=w0, max_rows=96)

        # ── individual figures (one file per panel) ────────────────────────
        out_dir = os.path.dirname(os.path.abspath(__file__))   # same folder as script
        plot_all_figures(
            m, batt, risk, scenarios,
            save_dir=out_dir,
            prefix="battery",
            dpi=180,
            show=True,   # set False for batch/headless runs
        )

    # ── efficient frontier (independent of single solve above) ──────────────
    if BETA_SWEEP:
        here = os.path.dirname(os.path.abspath(__file__))
        print(f"\n{'#'*60}")
        print(f"  Starting efficient frontier sweep")
        print(f"  α = {alpha}   β values = {BETA_VALUES}")
        print(f"{'#'*60}")
        ef_results = run_beta_sweep(
            batt=batt, alpha=alpha, betas=BETA_VALUES,
            pi_DA=pi_DA, scenarios=scenarios, prob=prob,
            pi_up=pi_up, pi_dn=pi_dn, V_up=V_up, V_dn=V_dn,
            pi_cap_up=pi_cap_up, pi_cap_dn=pi_cap_dn,
            alpha_up=alpha_up, alpha_dn=alpha_dn,
            tee=False,   # set True to see Gurobi log for each solve
        )
        ef_path = os.path.join(here, "battery_fig7_efficient_frontier.png") \
                  if SAVE_PLOTS else ""
        plot_efficient_frontier(
            ef_results, alpha=alpha,
            save_path=ef_path,
            dpi=180, show=True,
        )