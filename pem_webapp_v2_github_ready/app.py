"""
PEM 수전해 + 재생에너지 모델 — 대회 현장 시연용 웹앱 (v2)

2026-09 대개편 — "그때그때 계산" -> "사전계산 캐시 조회"로 전환:
  이전까지의 v2는 매 요청마다 pem_model.py를 별도 프로세스(worker.py)에서
  실행해서 그 자리에서 20~40초씩 계산했다(대기열 + 워커스레드 풀 + 폴링
  구조). 실제로 필요한 조합(지역 × 발전원)은 어차피 유한하게 정해져 있어서
  precompute.py가 오프라인으로 그 조합들을 전부 미리 계산해 results_cache/에
  JSON으로 저장해두고, 이 웹앱은 그 파일을 읽어서 즉시 돌려주기만 한다.

  그래서 이제 여기엔 작업 큐, 워커 스레드, subprocess 실행, 타임아웃, 동시
  실행 개수 제한 같은 게 전혀 없다 — 몇 명이 동시에 눌러도 그냥 딸린 파일을
  메모리에서 읽어서 보내주는 것뿐이라 무료 플랜으로도 충분하다. 새로운
  지역/연도 조합을 추가하려면 precompute.py를 다시 돌려서 results_cache/에
  파일을 채우고 다시 배포하면 된다.

  gunicorn worker 개수 제한(--workers 1) 같은 예전 주의사항도 이제 의미가
  없다 — 상태를 메모리에 들고 있는 큐가 없어졌기 때문에 여러 worker를 써도
  안전하다.
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "results_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"


def _load_cache() -> tuple[dict[tuple[str, str], dict], dict[str, list[str]], dict]:
    """results_cache/manifest.json + 각 결과 JSON을 메모리로 올린다.

    반환: (results, regions, year_info)
      results: {(kind, region): payload}
      regions: {"태양광": [...지역들...], "풍력": [...]}  (실제로 캐시가 있는 것만)
      year_info: {kind: {region: {"start":.., "end":.., "n_years":..}}}
                 — 프론트엔드가 "이 지역은 몇 년치 데이터 기준"을 보여줄 때 씀
    """
    results: dict[tuple[str, str], dict] = {}
    regions: dict[str, list[str]] = {"태양광": [], "풍력": []}
    year_info: dict[str, dict[str, dict]] = {"태양광": {}, "풍력": {}}

    if not MANIFEST_PATH.exists():
        print(f"[경고] {MANIFEST_PATH} 없음 — precompute.py를 먼저 돌려야 합니다.")
        return results, regions, year_info

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in manifest.get("entries", []):
        region, kind = entry["region"], entry["kind"]
        cpath = CACHE_DIR / f"{kind}_{region}.json"
        if not cpath.exists():
            print(f"[경고] manifest엔 있는데 파일이 없음: {cpath}")
            continue
        payload = json.loads(cpath.read_text(encoding="utf-8"))
        results[(kind, region)] = payload
        regions.setdefault(kind, []).append(region)
        year_info.setdefault(kind, {})[region] = {
            "start": entry["start_year"], "end": entry["end_year"],
            "n_years": entry["end_year"] - entry["start_year"] + 1,
        }

    for kind in regions:
        regions[kind] = sorted(set(regions[kind]))
    return results, regions, year_info


RESULTS, REGIONS, YEAR_INFO = _load_cache()


@app.route("/")
def index():
    return render_template("index.html", regions=REGIONS, year_info=YEAR_INFO)


@app.route("/api/regions")
def api_regions():
    return jsonify({**REGIONS, "_year_info": YEAR_INFO})


@app.route("/api/run")
def api_run():
    region = request.args.get("region", "")
    kind = request.args.get("kind", "태양광")
    if not region:
        return jsonify({"error": "region 파라미터가 필요합니다."}), 400
    payload = RESULTS.get((kind, region))
    if payload is None:
        return jsonify({
            "error": f"'{kind}'에 '{region}' 사전계산 결과가 없습니다. "
                     f"사용 가능한 지역: {', '.join(REGIONS.get(kind, [])) or '(없음)'}"
        }), 400
    return jsonify(payload)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "cached_combinations": len(RESULTS)})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
