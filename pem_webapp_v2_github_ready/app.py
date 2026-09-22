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
BUNDLE_PATH = CACHE_DIR / "bundle.json"
MANIFEST_PATH = CACHE_DIR / "manifest.json"  # 이제 안 씀 — 참고용으로만 남겨둠

# pem_model.py의 PARAM_TABLE과 동일한 7종. 지역/발전원과 달리 촉매는 "데이터가
# 있고 없고"가 아니라 물리 모델 파라미터 선택이라, 사전계산만 다 돼있으면
# 모든 지역·발전원에 항상 7개 다 제공된다(precompute.py의 CATALYSTS와 맞춰
# 둔다 — 하나 추가되면 여기도 같이 고쳐야 함).
CATALYSTS = ["IrO2", "MnO2", "NMO-0.05", "NMO-0.1", "NMO-0.2", "NMO-0.3", "NMO-0.4"]
DEFAULT_CATALYST = "IrO2"


def _load_cache() -> tuple[dict[tuple[str, str, str], dict], dict[str, list[str]], dict, list[str]]:
    """results_cache/bundle.json 하나를 메모리로 올린다.

    2026-09-16 변경: 예전엔 지역마다 "풍력_강원.json"처럼 한글이 들어간
    파일명으로 27개를 따로 저장했는데, 이 구조가 GitHub 업로드 과정에서
    한글 파일명 27개가 통째로 커밋에서 빠지는 문제를 냈다(manifest.json만
    올라가고 실제 결과 파일들은 하나도 안 올라감 — 파일시스템/깃 계층에서
    비-ASCII 파일명이 또 말썽을 부린 것). 파일명 자체에는 아예 한글을 쓰지
    않도록, 결과를 bundle.json 파일 하나(영문 이름)에 다 합쳐서 저장하고
    지역/발전원/촉매 이름은 그 안의 JSON 값(키)으로만 넣는 방식으로 바꿨다 —
    이러면 한글은 파일 "내용"에만 있고 파일 "이름"에는 전혀 없어서 이런 종류의
    문제가 재발할 수 없다.

    2026-09-18 변경: 촉매 차원 추가. bundle.json의 entries/results 키에
    "catalyst"가 없는(=촉매 기능 추가 전에 만들어진 옛 bundle) 항목은
    DEFAULT_CATALYST("IrO2")로 간주해서 하위호환한다.

    반환: (results, regions, year_info, catalysts)
      results: {(kind, region, catalyst): payload}
      regions: {"태양광": [...지역들...], "풍력": [...]}  (실제로 캐시가 있는 것만)
      year_info: {kind: {region: {"start":.., "end":.., "n_years":..}}}
                 — 프론트엔드가 "이 지역은 몇 년치 데이터 기준"을 보여줄 때 씀
      catalysts: 실제로 캐시에 결과가 하나라도 있는 촉매 이름 목록(정렬됨)
    """
    results: dict[tuple[str, str, str], dict] = {}
    regions: dict[str, list[str]] = {"태양광": [], "풍력": []}
    year_info: dict[str, dict[str, dict]] = {"태양광": {}, "풍력": {}}
    catalysts_seen: set[str] = set()

    if not BUNDLE_PATH.exists():
        print(f"[경고] {BUNDLE_PATH} 없음 — precompute.py를 먼저 돌려야 합니다.")
        return results, regions, year_info, []

    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    bundle_results = bundle.get("results", {})
    for entry in bundle.get("entries", []):
        region, kind = entry["region"], entry["kind"]
        catalyst = entry.get("catalyst", DEFAULT_CATALYST)
        key = f"{kind}|{region}|{catalyst}"
        payload = bundle_results.get(key)
        if payload is None:
            # 하위호환: 촉매 차원 추가 전(2026-09-18 이전)에 만들어진 bundle은
            # "kind|region" 2단 키만 있을 수 있다.
            payload = bundle_results.get(f"{kind}|{region}")
        if payload is None:
            print(f"[경고] entries엔 있는데 results엔 없음: {key}")
            continue
        results[(kind, region, catalyst)] = payload
        regions.setdefault(kind, []).append(region)
        catalysts_seen.add(catalyst)
        year_info.setdefault(kind, {})[region] = {
            "start": entry["start_year"], "end": entry["end_year"],
            "n_years": entry["end_year"] - entry["start_year"] + 1,
        }

    for kind in regions:
        regions[kind] = sorted(set(regions[kind]))
    return results, regions, year_info, sorted(catalysts_seen)


RESULTS, REGIONS, YEAR_INFO, CATALYSTS_AVAILABLE = _load_cache()
# 캐시에 아직 하나도 없으면(막 배포 직후 등) 정의된 전체 목록을 보여주고,
# 있으면 실제로 계산 완료된 것만 보여준다.
CATALYST_LIST = CATALYSTS_AVAILABLE or CATALYSTS


@app.route("/")
def index():
    return render_template("index.html", regions=REGIONS, year_info=YEAR_INFO,
                          catalysts=CATALYST_LIST)


@app.route("/api/regions")
def api_regions():
    return jsonify({**REGIONS, "_year_info": YEAR_INFO, "_catalysts": CATALYST_LIST})


@app.route("/api/run")
def api_run():
    region = request.args.get("region", "")
    kind = request.args.get("kind", "태양광")
    catalyst = request.args.get("catalyst", DEFAULT_CATALYST)
    if not region:
        return jsonify({"error": "region 파라미터가 필요합니다."}), 400
    payload = RESULTS.get((kind, region, catalyst))
    if payload is None:
        return jsonify({
            "error": f"'{kind}'/'{region}'/'{catalyst}' 사전계산 결과가 없습니다. "
                     f"사용 가능한 지역: {', '.join(REGIONS.get(kind, [])) or '(없음)'}, "
                     f"사용 가능한 촉매: {', '.join(CATALYST_LIST) or '(없음)'}"
        }), 400
    return jsonify(payload)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "cached_combinations": len(RESULTS),
                    "catalysts": CATALYST_LIST})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
