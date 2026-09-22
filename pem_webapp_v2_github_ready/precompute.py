#!/usr/bin/env python3
"""사전계산(precompute) 배치 스크립트 — 결과를 미리 다 뽑아서 results_cache/에 저장.

왜 필요한가:
  이전 버전(v2)은 사용자가 웹에서 버튼을 누를 때마다 그 자리에서 20~40초씩
  실제로 계산했다. 그러다 보니 동시접속·유료 플랜·타임아웃 문제가 계속
  따라왔다. 대회에서 실제로 필요한 조합(지역 × 발전원 × 연도범위 × 촉매)은
  어차피 정해진 유한한 집합이니, 그걸 미리 한 번씩만 계산해서 JSON으로
  저장해두고 웹은 그 파일을 읽어서 즉시 보여주기만 하면 된다 — 몇 명이
  몰려도 무료 플랜으로 충분하다.

무엇을 계산하는가 (2026-09 데이터 점검 결과 기준):
  태양광: 17개 지역 전부, 2018~2025년(8개년) — 전 지역·전 연도 이용률/결측
          문제 없음을 확인했다.
  풍력:   부산은 전체 기간 이용률이 0%에 가까워 아예 제외한다.
          울산은 2022년부터 무너지므로(2022년 1.6%, 2023년 0%) 2018~2021년만
          쓴다. 나머지 지역(강원·경기·경남·경북·인천·전남·전북·제주·충남)은
          2023년에 12월 데이터가 통째로 빠져있어서(모든 지역 공통 현상) 그 해를
          피해 2018~2022년만 쓴다.

2026-09-18 업데이트 — 촉매(catalyst) 차원 추가:
  0918 모델부터 촉매를 IrO2(기본) 외에 MnO2, NMO-0.05~0.4 중에서 고를 수
  있다. 지역/발전원 27개 조합 × 촉매 7종 = 최대 189건까지 계산 대상이 늘어난다.
  이미 계산해둔 IrO2 결과(27건)는 재사용하고, 나머지 6개 촉매만 새로 돌리면
  된다 — cache_path()가 이제 (kind, region, catalyst) 3개를 받아서 파일 이름에
  촉매 이름도 넣으므로("태양광_강원_IrO2.json"), 기존 파일은 먼저
  migrate_legacy_cache()로 한 번 이름을 바꿔줘야 "이미 있음"으로 인식된다.

사용법:
  python3 precompute.py --data-root "<원본 xlsx가 있는 폴더>"
  python3 precompute.py --data-root "..." --only 강원:태양광 세종:태양광  (일부만, 전체 촉매)
  python3 precompute.py --data-root "..." --only 강원:태양광:MnO2  (일부만, 특정 촉매)
  python3 precompute.py --data-root "..." --catalysts IrO2 MnO2  (일부 촉매만 전체 지역에)
  python3 precompute.py --data-root "..." --force  (이미 있는 캐시도 덮어쓰기)

주의: --data-root는 배포 저장소에 들어가지 않는다(용량이 크고, 한 번 계산해
  두면 웹 서비스에는 더 이상 원본 xlsx가 필요 없기 때문). results_cache/의
  JSON 파일만 커밋해서 배포하면 된다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKER_PATH = BASE_DIR / "worker.py"
CACHE_DIR = BASE_DIR / "results_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
BUNDLE_PATH = CACHE_DIR / "bundle.json"

SOLAR_REGIONS = ["강원", "경기", "경남", "경북", "광주", "대구", "대전", "부산",
                 "서울", "세종", "울산", "인천", "전남", "전북", "제주", "충남", "충북"]
SOLAR_YEAR_RANGE = (2018, 2025)

# 부산은 제외. 울산은 2018~2021만, 나머지는 2018~2022만 (2026-09 데이터 점검 근거는
# README의 "2026-09 데이터 점검" 절 참고).
WIND_YEAR_RANGE_DEFAULT = (2018, 2022)
WIND_YEAR_RANGE_OVERRIDE = {"울산": (2018, 2021)}
WIND_REGIONS = ["강원", "경기", "경남", "경북", "울산", "인천", "전남", "전북", "제주", "충남"]
WIND_EXCLUDED = ["부산"]

# pem_model.py의 PARAM_TABLE에 등록된 7종. 이리듐(IrO2)이 기본값이고 나머지
# 6종이 새로 추가된 촉매다. 이름은 전부 ASCII라 파일명에 그대로 써도 안전하다
# (한글 파일명이 GitHub 업로드에서 통째로 빠진 사고 이후로 파일 "이름"에는
# 한글을 아예 안 쓴다는 원칙을 그대로 지킨다).
CATALYSTS = ["IrO2", "MnO2", "NMO-0.05", "NMO-0.1", "NMO-0.2", "NMO-0.3", "NMO-0.4"]
DEFAULT_CATALYST = "IrO2"

WORKER_TIMEOUT_S = 300  # 실측 8개년 태양광 1건 ~85초, 여유를 넉넉히 둠


def build_job_list(catalysts: list[str] | None = None) -> list[dict]:
    cats = catalysts if catalysts is not None else CATALYSTS
    jobs = []
    for catalyst in cats:
        for region in SOLAR_REGIONS:
            jobs.append({"region": region, "kind": "태양광", "catalyst": catalyst,
                         "start_year": SOLAR_YEAR_RANGE[0], "end_year": SOLAR_YEAR_RANGE[1]})
        for region in WIND_REGIONS:
            start, end = WIND_YEAR_RANGE_OVERRIDE.get(region, WIND_YEAR_RANGE_DEFAULT)
            jobs.append({"region": region, "kind": "풍력", "catalyst": catalyst,
                         "start_year": start, "end_year": end})
    return jobs


def cache_path(kind: str, region: str, catalyst: str = DEFAULT_CATALYST) -> Path:
    return CACHE_DIR / f"{kind}_{region}_{catalyst}.json"


def _legacy_cache_path(kind: str, region: str) -> Path:
    """촉매 차원이 생기기 전(항상 IrO2였던 시절)의 파일 이름."""
    return CACHE_DIR / f"{kind}_{region}.json"


def migrate_legacy_cache() -> int:
    """예전(촉매 구분 없던 시절) 캐시 파일을 _IrO2.json 이름으로 옮긴다.

    이미 새 이름으로 존재하면 건드리지 않는다(중복 계산 스킵 판단에 방해되지
    않도록). 실행할 때마다 안전하게 다시 불러도 되는 멱등 함수다.
    """
    migrated = 0
    for region in SOLAR_REGIONS:
        legacy = _legacy_cache_path("태양광", region)
        target = cache_path("태양광", region, DEFAULT_CATALYST)
        if legacy.exists() and not target.exists():
            legacy.rename(target)
            migrated += 1
    for region in WIND_REGIONS:
        legacy = _legacy_cache_path("풍력", region)
        target = cache_path("풍력", region, DEFAULT_CATALYST)
        if legacy.exists() and not target.exists():
            legacy.rename(target)
            migrated += 1
    return migrated


def run_job(job: dict, data_root: str) -> tuple[bool, str, float]:
    cmd = [sys.executable, str(WORKER_PATH),
           "--region", job["region"], "--kind", job["kind"],
           "--data-root", data_root,
           "--start-year", str(job["start_year"]), "--end-year", str(job["end_year"]),
           "--catalyst", job["catalyst"]]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=WORKER_TIMEOUT_S, cwd=BASE_DIR)
    except subprocess.TimeoutExpired:
        return False, f"타임아웃({WORKER_TIMEOUT_S}s 초과)", time.time() - t0
    elapsed = time.time() - t0
    if proc.stderr.strip():
        # 워커 로그(진행상황)는 참고용으로만 흘려보냄. 너무 길면 마지막 일부만.
        tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
        print(f"    [worker log tail] {tail}")
    if not proc.stdout.strip():
        return False, f"출력 없음 (exit={proc.returncode})", elapsed
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return False, f"JSON 파싱 실패: {e}", elapsed
    if proc.returncode != 0 or "error" in payload:
        return False, payload.get("error", f"워커 실패 (exit={proc.returncode})"), elapsed
    cache_path(job["kind"], job["region"], job["catalyst"]).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return True, "ok", elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="원본 xlsx 데이터 폴더")
    parser.add_argument("--force", action="store_true", help="이미 있는 캐시도 다시 계산")
    parser.add_argument("--only", nargs="*", default=None,
                        help="'지역:발전원' 또는 '지역:발전원:촉매' 형식으로 일부만 재실행 "
                             "(예: 세종:태양광 또는 세종:태양광:MnO2)")
    parser.add_argument("--catalysts", nargs="*", default=None, choices=CATALYSTS,
                        help="지정하면 이 촉매들만 계산 (기본: 7종 전부)")
    parser.add_argument("--workers", type=int, default=1,
                        help="동시에 돌릴 워커 프로세스 수 (기본 1 = 순차 실행). "
                             "CPU 코어 수만큼 늘리면 전체 배치 시간이 그만큼 줄어든다.")
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    migrated = migrate_legacy_cache()
    if migrated:
        print(f"기존(촉매 구분 없던 시절) 캐시 {migrated}건을 IrO2 이름으로 이전했습니다.")

    # 2026-09-18: --catalysts로 일부 촉매만 골라 이 배치를 돌려도, 이전에 이미
    # 계산해둔 다른 촉매/지역 조합(예: IrO2 27건)이 manifest에서 사라지면 안
    # 된다. 그래서 이번 실행분만으로 manifest를 새로 만드는 대신, 기존
    # manifest.json에 있던 항목을 먼저 불러와 베이스라인으로 삼고, 이번에
    # 처리한(또는 스킵한) 항목만 그 위에 덮어써서 항상 "지금까지 계산된 전체
    # 목록"을 유지한다.
    all_entries: dict[tuple[str, str, str], dict] = {}
    if MANIFEST_PATH.exists():
        try:
            old_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            for e in old_manifest.get("entries", []):
                cat = e.get("catalyst", DEFAULT_CATALYST)
                all_entries[(e["kind"], e["region"], cat)] = {**e, "catalyst": cat}
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[경고] 기존 manifest.json을 읽지 못해 새로 시작합니다: {exc}")

    jobs = build_job_list(args.catalysts)
    if args.only:
        wanted = set(args.only)

        def _match(j: dict) -> bool:
            rk = f"{j['region']}:{j['kind']}"
            rkc = f"{j['region']}:{j['kind']}:{j['catalyst']}"
            return rk in wanted or rkc in wanted

        jobs = [j for j in jobs if _match(j)]
        if not jobs:
            print(f"--only에 맞는 조합이 없습니다: {args.only}")
            return 1

    n_solar = sum(1 for j in jobs if j["kind"] == "태양광")
    n_wind = sum(1 for j in jobs if j["kind"] == "풍력")
    print(f"총 {len(jobs)}건 예정 (태양광 {n_solar}건, 풍력 {n_wind}건) · workers={args.workers}")

    lock = threading.Lock()
    failed: list[dict] = []

    def save_manifest_locked():
        # 배치 전체가 끝나야만 저장하면, 중간에 죽었을 때(샌드박스가 한번씩
        # 끊겨서 실제로 여러 번 겪었다) 이미 다 계산해둔 파일이 있어도 웹앱이
        # 하나도 못 읽는다. 그래서 매 건이 끝날 때마다 그 시점까지의 상태를
        # 바로바로 통째로 다시 써서, 언제 중단되든 그 순간까지 계산된 결과는
        # 웹에서 바로 쓸 수 있게 한다.
        entries = list(all_entries.values())
        MANIFEST_PATH.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entries": entries,
            "failed": failed,
            "excluded": [{"region": r, "kind": "풍력", "reason": "이용률 0%에 가까워 제외"}
                         for r in WIND_EXCLUDED],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        # bundle.json: 실제로 배포/커밋되는 건 이 파일 하나뿐이다. 예전엔
        # "풍력_강원.json"처럼 지역마다 한글 파일명으로 따로 저장했는데, 그
        # 방식이 GitHub 업로드 과정에서 한글 파일명 27개가 통째로 커밋에서
        # 빠지는 사고를 냈다(manifest.json만 올라가고 실제 결과는 하나도
        # 안 올라감). 파일 "이름"에는 한글을 아예 안 쓰도록, 지역/발전원/촉매별
        # 결과를 이 파일 하나(영문 이름)의 JSON 값(키)으로만 합쳐서 저장한다
        # — 한글은 파일 내용에만 있으니 이런 문제가 재발할 수 없다.
        # 키는 f"{kind}|{region}|{catalyst}" 형식(촉매 차원 추가, 2026-09-18).
        bundle_results = {}
        for e in entries:
            cpath = cache_path(e["kind"], e["region"], e["catalyst"])
            if cpath.exists():
                bundle_results[f"{e['kind']}|{e['region']}|{e['catalyst']}"] = json.loads(
                    cpath.read_text(encoding="utf-8"))
        BUNDLE_PATH.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "catalysts": CATALYSTS,
            "entries": entries,
            "results": bundle_results,
        }, ensure_ascii=True), encoding="utf-8")

    def process_one(i: int, job: dict) -> None:
        cpath = cache_path(job["kind"], job["region"], job["catalyst"])
        label = (f"[{i}/{len(jobs)}] {job['kind']} {job['region']} [{job['catalyst']}] "
                 f"({job['start_year']}-{job['end_year']})")
        if cpath.exists() and not args.force:
            print(f"{label} -> 이미 있음, 스킵")
            with lock:
                all_entries[(job["kind"], job["region"], job["catalyst"])] = {
                    **job, "cached_at": None, "skipped_existing": True}
                save_manifest_locked()
            return
        print(f"{label} -> 계산 중...", flush=True)
        ok, msg, elapsed = run_job(job, args.data_root)
        with lock:
            if ok:
                print(f"{label} -> 완료 ({elapsed:.1f}s)")
                all_entries[(job["kind"], job["region"], job["catalyst"])] = {
                    **job, "cached_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": round(elapsed, 1)}
            else:
                print(f"{label} -> 실패: {msg} ({elapsed:.1f}s)")
                failed.append({**job, "error": msg})
            save_manifest_locked()  # 매 건마다 즉시 갱신

    t_start = time.time()
    if args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            process_one(i, job)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_one, i, job) for i, job in enumerate(jobs, 1)]
            for fut in as_completed(futures):
                fut.result()  # 워커 스레드 안에서 난 예외를 여기서 다시 던져서 눈에 띄게 함

    total_elapsed = time.time() - t_start
    n_ok_this_run = sum(1 for j in jobs if (j["kind"], j["region"], j["catalyst"]) in all_entries)
    print(f"\n총 {total_elapsed/60:.1f}분 소요. 이번 실행 대상 {len(jobs)}건 중 반영 {n_ok_this_run}건, "
          f"실패 {len(failed)}건. 전체 누적 캐시 {len(all_entries)}건.")
    if failed:
        print("실패 목록:")
        for f in failed:
            print(f"  - {f['kind']} {f['region']} [{f['catalyst']}]: {f['error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
