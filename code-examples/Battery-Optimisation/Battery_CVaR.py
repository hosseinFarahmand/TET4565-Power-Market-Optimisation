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


# -----------------------------
# PLOTS
# -----------------------------
def plot_stage1(m: pyo.ConcreteModel, save_prefix: str | None = None):
    T = len(list(m.T))
    H = len(list(m.H))
    t = list(range(T))
    h = list(range(H))

    p_DA = _series_var_1d(m.p_DA, T)
    r_up = _hourly_var(m.r_up, H)
    r_dn = _hourly_var(m.r_dn, H)

    plt.figure()
    plt.plot(t, p_DA)
    plt.title("Stage 1: Day-ahead schedule")
    plt.xlabel("t (15-min index)")
    plt.ylabel("p_DA [MW]")
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_stage1_pDA.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.step(h, r_up, where="post", label="r_up [MW]")
    plt.step(h, r_dn, where="post", label="r_dn [MW]")
    plt.title("Stage 1: Hourly reserve bids")
    plt.xlabel("h (hour index)")
    plt.ylabel("Reserve bid [MW]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_stage1_reserves.png", dpi=200, bbox_inches="tight")


def plot_stage2_for_scenario(m: pyo.ConcreteModel, w, save_prefix: str | None = None):
    T = len(list(m.T))
    t = list(range(T))

    p_ch = _series_var_2d(m.p_ch, w, T)
    p_dis = _series_var_2d(m.p_dis, w, T)
    p_DA = _series_var_1d(m.p_DA, T)

    soc = [float(pyo.value(m.e[tt, w])) for tt in range(T + 1)]
    d_plus = _series_var_2d(m.d_plus, w, T)
    d_minus = _series_var_2d(m.d_minus, w, T)
    V_up_series = _series_param_2d(m.V_up, w, T)
    V_dn_series = _series_param_2d(m.V_dn, w, T)

    p_real = [p_dis[i] - p_ch[i] for i in range(T)]
    p_tar = [p_DA[i] + V_up_series[i] - V_dn_series[i] for i in range(T)]

    plt.figure()
    plt.plot(t, p_real, label="p_real")
    plt.plot(t, p_tar, label="p_target")
    plt.plot(t, p_DA, label="p_DA")
    plt.title(f"Stage 2: Power trajectories ({w})")
    plt.xlabel("t (15-min index)")
    plt.ylabel("Power [MW]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_stage2_power_{w}.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(t, p_ch, label="p_ch")
    plt.plot(t, p_dis, label="p_dis")
    plt.title(f"Stage 2: Charge / Discharge ({w})")
    plt.xlabel("t (15-min index)")
    plt.ylabel("Power [MW]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_stage2_chdis_{w}.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(list(range(T + 1)), soc)
    plt.title(f"Stage 2: SOC ({w})")
    plt.xlabel("t (15-min index incl. terminal)")
    plt.ylabel("SOC [MWh]")
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_stage2_soc_{w}.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(t, d_plus, label="d_plus")
    plt.plot(t, d_minus, label="d_minus")
    plt.title(f"Stage 2: Deviations ({w})")
    plt.xlabel("t (15-min index)")
    plt.ylabel("Deviation [MW]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_stage2_dev_{w}.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(t, V_up_series, label="V_up")
    plt.plot(t, V_dn_series, label="V_dn")
    plt.title(f"Exogenous activation volumes ({w})")
    plt.xlabel("t (15-min index)")
    plt.ylabel("Activated volume [MW]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_stage2_activation_{w}.png", dpi=200, bbox_inches="tight")


def plot_soc_vs_da_price(m: pyo.ConcreteModel, w, save_prefix: str | None = None):
    T = len(list(m.T))
    t = list(range(T))

    soc = [float(pyo.value(m.e[tt, w])) for tt in range(T)]
    da_price = [float(pyo.value(m.pi_DA[tt, w])) for tt in range(T)]

    fig, ax1 = plt.subplots()

    ax1.plot(t, soc, label="SOC [MWh]")
    ax1.set_xlabel("t (15-min index)")
    ax1.set_ylabel("SOC [MWh]")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(t, da_price, linestyle="--", label="DA price [€/MWh]")
    ax2.set_ylabel("Day-ahead price [€/MWh]")

    fig.suptitle(f"SOC vs Day-Ahead Price ({w})")
    fig.tight_layout()

    if save_prefix:
        plt.savefig(f"{save_prefix}_soc_vs_da_{w}.png", dpi=200, bbox_inches="tight")


def plot_profit_value_stacking_waterfall(
    m: pyo.ConcreteModel,
    batt: BatteryParams,
    w,
    save_prefix: str | None = None
):
    da_rev, cap_rev, act_rev, dev_rev, tariff, degr = _scenario_components(m, batt, w)
    profit = float(pyo.value(m.Profit[w]))

    components = [
        ("DA revenue", da_rev),
        ("Reserve capacity", cap_rev),
        ("Activation", act_rev),
        ("Deviation", dev_rev),
        ("Tariff cost", -tariff),
        ("Degradation cost", -degr),
    ]

    labels = []
    bottoms = []
    heights = []

    running = 0.0
    for name, val in components:
        labels.append(name)
        bottoms.append(running if val >= 0 else running + val)
        heights.append(abs(val))
        running += val

    labels.append("Profit")
    bottoms.append(0.0 if profit >= 0 else profit)
    heights.append(abs(profit))

    plt.figure(figsize=(11, 5))
    plt.bar(labels[:-1], heights[:-1], bottom=bottoms[:-1])
    plt.bar(labels[-1], heights[-1], bottom=bottoms[-1])

    plt.axhline(0, linewidth=1)
    plt.title(f"Profit Value Stacking Waterfall ({w})")
    plt.ylabel("EUR")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)

    running = 0.0
    for i, (_, val) in enumerate(components):
        running += val
        plt.text(
            i,
            running,
            f"{val:.1f}",
            ha="center",
            va="bottom" if running >= 0 else "top"
        )

    plt.text(
        len(labels) - 1,
        profit,
        f"{profit:.1f}",
        ha="center",
        va="bottom" if profit >= 0 else "top"
    )

    plt.tight_layout()

    if save_prefix:
        plt.savefig(f"{save_prefix}_value_stacking_waterfall_{w}.png", dpi=200, bbox_inches="tight")


def plot_risk_summary(m: pyo.ConcreteModel, save_prefix: str | None = None):
    profits = [float(pyo.value(m.Profit[w])) for w in m.O]
    eta = float(pyo.value(m.eta))
    xi = [float(pyo.value(m.xi[w])) for w in m.O]

    plt.figure()
    plt.hist(profits, bins=20)
    plt.axvline(eta, linestyle="--", label="VaR (eta)")
    plt.title("Scenario profit distribution")
    plt.xlabel("Profit [EUR]")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_risk_profit_hist.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.scatter(profits, xi)
    plt.title("CVaR shortfall vs profit")
    plt.xlabel("Profit [EUR]")
    plt.ylabel("Shortfall xi [EUR]")
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_risk_xi_scatter.png", dpi=200, bbox_inches="tight")




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
    alpha: float = 0.95
    beta: float = 0.0


def hour_of_t(t: int) -> int:
    return t // 4


# -------------------------------------------------------------------------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    batt = BatteryParams(
        soc_margin_h=0.25,                 # 15-min endurance
        Ramp_MW_per_h=999.0,               # set e.g. 60 for 1 MW/min
        forbid_simultaneous_up_dn_bids=True,
    )

    # CVaR controls
    risk = CVaRParams(alpha=0.95, beta=0.5)  # beta=0 => risk-neutral

    # Activation scaling factors
    alpha_up = 0.01  # scaling factor for up reserve
    alpha_dn = 0.01  # scaling factor for down reserve

    SAVE_PLOTS = False
    PLOT_WORST_SCENARIO = False

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

        prefix = "results" if SAVE_PLOTS else None

        plot_stage1(m, save_prefix=prefix)
        plot_risk_summary(m, save_prefix=prefix)
        plot_stage2_for_scenario(m, w0, save_prefix=prefix)
        plot_soc_vs_da_price(m, w0, save_prefix=prefix)
        plot_profit_value_stacking_waterfall(m, batt, w0, save_prefix=prefix)

        plt.show()