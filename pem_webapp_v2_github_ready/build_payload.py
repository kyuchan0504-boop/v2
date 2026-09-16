"""app.py(메인 프로세스)와 worker.py(계산 프로세스)가 공유하는 결과 가공 로직.

pem_model.py의 main() 출력(profile/results/lcohs/stack)을 프론트엔드용
JSON-safe dict로 바꾼다. 로직 자체는 v1의 app.py에 있던 run_model()과 동일하고,
실행되는 위치만 "메인 프로세스"에서 "격리된 워커 프로세스"로 옮겼을 뿐이다.

2026-09 업데이트 — 다년(multi-year) 프로필 대응:
  새 pem_model.py는 profile/DispatchResult가 이제 "1개년"이 아니라 지정한
  RE_START_YEAR~RE_END_YEAR 전체 기간(예: 태양광 2018~2025년, 8개년)을 하나로
  이어붙인 시계열이다. 그래서 H2_total/op_hours처럼 "그 기간 전체 합"인 값은
  그대로 쓰면 연간 값이 아니라 "n개년치 합"이 되어 8배씩 부풀려진다.
  m._annual_mean(r, value) = value / n_years 로 나눠야 진짜 "연평균"이 나온다
  (pem_model.py 내부의 _size_ess 등도 결과 표시에 똑같이 이 함수를 쓴다).
  월별 합계도 마찬가지로, 여러 해의 같은 달을 그냥 더하면 안 되고
  m.monthly_statistics()가 계산해주는 "그 달의 연도별 평균"을 써야 한다.
  반면 LCOH/SEC_eff([$/kg], [kWh/kg])나 curtail_pct/ledger_pct(비율),
  ESS 용량·CAPEX(설비 사이징 값), peak/mean/capacity_factor(시간당 통계)는
  원래부터 기간 길이와 무관한 값이라 그대로 써도 된다.
"""
from __future__ import annotations


def _monthly_avg(m, series, timestamps) -> list[float]:
    """여러 해가 섞인 시계열에서 "그 달의 연도별 평균"을 1~12월 순서로 반환."""
    stats = m.monthly_statistics(series, timestamps)
    return [float(stats["mean"].get(mo, 0.0)) for mo in range(1, 13)]


LCOH_ITEM_LABELS = {
    "annualized_capex": "설비비(연환산)",  # 구 모델 키 (호환용으로 남겨둠)
    "capex": "설비비(현재가치환산)",       # 신모델(2026-09, 열화·교체 반영 lifecycle DCF) 키
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

    cases = {}
    for name, r in results.items():
        lc = lcohs[name]
        items = {LCOH_ITEM_LABELS.get(k, k): round(float(v), 4)
                 for k, v in lc["items_usd_per_kg"].items()}
        ledger = r.ledger
        e_paid = max(r.E_paid, 1e-9)
        n_yr = max(m._annual_n(r), 1e-12)
        hrs = m._production_hours(r, getattr(r, "dt", 1.0) or 1.0)
        h2_timestamps = r.timestamps if r.timestamps is not None else profile.timestamps
        cases[name] = {
            "annual_h2_t": round(m._annual_mean(r, r.H2_total) / 1000.0, 3),
            "lcoh": round(float(lc["LCOH"]), 3),
            "sec_eff": round(r.SEC_eff(), 2),
            "op_hours": round(m._annual_mean(r, r.op_hours), 0),
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
            "monthly_h2_kg": [round(v, 1) for v in _monthly_avg(m, r.H2_series, h2_timestamps)],
            "hour_ledger": {
                "producing": round(hrs["producing"] / n_yr, 0),
                "idle_with_gen": round(hrs["idle_with_gen"] / n_yr, 0),
                "no_gen": round(hrs["no_gen"] / n_yr, 0),
            },
        }

    best_lcoh = min(cases.items(), key=lambda kv: kv[1]["lcoh"])
    best_h2 = max(cases.items(), key=lambda kv: kv[1]["annual_h2_t"])

    n_years = profile.n_years
    years = profile.years

    return {
        "region": region,
        "kind": kind,
        "year": years[-1],                 # 하위호환용(단일 연도 표시가 필요하면 마지막 해)
        "year_start": years[0],
        "year_end": years[-1],
        "n_years": n_years,
        "capacity_factor_pct": round(profile.capacity_factor * 100, 2),
        "peak_kw": round(profile.peak, 1),
        "mean_kw": round(profile.mean, 1),
        "e_paid_mwh": round(profile.E_paid / n_years / 1000.0, 1),
        "monthly_gen_mwh": [round(v / 1000.0, 1)
                            for v in _monthly_avg(m, profile.P_in, profile.timestamps)],
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
