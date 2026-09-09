"""app.py(메인 프로세스)와 worker.py(계산 프로세스)가 공유하는 결과 가공 로직.

pem_model.py의 main() 출력(profile/results/lcohs/stack)을 프론트엔드용
JSON-safe dict로 바꾼다. 로직 자체는 v1의 app.py에 있던 run_model()과 동일하고,
실행되는 위치만 "메인 프로세스"에서 "격리된 워커 프로세스"로 옮겼을 뿐이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _monthly_sum(values, months) -> list[float]:
    s = pd.Series(np.asarray(values, dtype=float), index=months)
    grouped = s.groupby(level=0).sum()
    return [float(grouped.get(mo, 0.0)) for mo in range(1, 13)]


LCOH_ITEM_LABELS = {
    "annualized_capex": "설비비(연환산)",
    "fixed_OM": "고정 O&M",
    "stack_replacement": "스택 교체",
    "battery_replacement": "배터리 교체",
    "electricity": "전기요금",
    "curtailment_penalty": "출력제한 위약금",
    "variable_OM": "변동 O&M",
    "water": "용수",
}


def build_payload(m, out: dict, region: str, kind: str) -> dict:
    profile = out["profile"]
    results = out["results"]
    lcohs = out["lcohs"]
    stack = out["stack"]
    months = profile.month

    cases = {}
    for name, r in results.items():
        lc = lcohs[name]
        items = {LCOH_ITEM_LABELS.get(k, k): round(float(v), 4)
                 for k, v in lc["items_usd_per_kg"].items()}
        ledger = r.ledger
        e_paid = max(r.E_paid, 1e-9)
        hrs = m._production_hours(r, getattr(r, "dt", 1.0) or 1.0)
        cases[name] = {
            "annual_h2_t": round(r.H2_total / 1000.0, 3),
            "lcoh": round(float(lc["LCOH"]), 3),
            "sec_eff": round(r.SEC_eff(), 2),
            "op_hours": round(r.op_hours, 0),
            "curtail_pct": round(r.E_curtail / e_paid * 100, 2),
            "ess_kwh": round(r.E_rated, 1),
            "ess_capex_usd": round(float(lc["capex_ess"]), 0),
            "lcoh_items": items,
            "ledger_pct": {
                "curtail": round(ledger["curtail"] / e_paid * 100, 2),
                "idle": round(ledger["idle"] / e_paid * 100, 2),
                "rte": round(ledger["rte"] / e_paid * 100, 2),
                "used": round((ledger["bop"] + ledger["stack"]) / e_paid * 100, 2),
            },
            "monthly_h2_kg": [round(v, 1) for v in _monthly_sum(r.H2_series, months)],
            "hour_ledger": {
                "producing": round(hrs["producing"], 0),
                "idle_with_gen": round(hrs["idle_with_gen"], 0),
                "no_gen": round(hrs["no_gen"], 0),
            },
        }

    best_lcoh = min(cases.items(), key=lambda kv: kv[1]["lcoh"])
    best_h2 = max(cases.items(), key=lambda kv: kv[1]["annual_h2_t"])

    return {
        "region": region,
        "kind": kind,
        "year": profile.year,
        "capacity_factor_pct": round(profile.capacity_factor * 100, 2),
        "peak_kw": round(profile.peak, 1),
        "mean_kw": round(profile.mean, 1),
        "e_paid_mwh": round(profile.E_paid / 1000.0, 1),
        "monthly_gen_mwh": [round(v / 1000.0, 1) for v in _monthly_sum(profile.P_in, months)],
        "stack": {
            "a_tot_cm2": round(stack.A_tot, 0),
            "n_cells": round(stack.N_cells, 1),
            "j_rated": stack.j_rated,
            "capex_stack_usd": round(stack.capex_stack, 0),
            "capex_bop_usd": round(stack.capex_bop, 0),
        },
        "cases": cases,
        "best_lcoh_case": best_lcoh[0],
        "best_h2_case": best_h2[0],
    }
