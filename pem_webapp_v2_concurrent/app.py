"""
PEM 수전해 + 재생에너지 모델 — 대회 현장 시연용 웹앱 (v2: 동시 접속 테스트용)

v1과 다른 점 (왜 이렇게 바꿨는지):
  v1은 pem_model.py를 이 프로세스 안에서 직접 import해서 돌렸다. pem_model.py가
  RE_REGION_NAME 같은 값을 "모듈 전역변수"로 두고 여러 함수가 직접 참조하는
  구조라서, 요청 두 개가 동시에 오면 전역변수가 서로 덮어써질 위험이 있었고,
  그래서 Lock으로 계산 전체를 직렬화했다(한 번에 한 명씩만 계산).

  v2는 매 요청마다 pem_model.py를 별도의 완전히 새 파이썬 프로세스(worker.py)에서
  돌린다. 프로세스가 다르면 메모리도 완전히 분리되니 전역변수가 절대 섞이지
  않는다 -> Lock 없이도 여러 요청을 동시에 안전하게 처리할 수 있다.
  대신 매 요청마다 새 프로세스를 띄우는 비용(numpy/pandas 재로딩 등 1~2초)이
  추가로 든다 — 실제 계산(로컬 7~9초, Render 무료 플랜은 최대 1분 가까이)에
  비하면 크지 않다고 판단했다.

  Render 무료 플랜은 메모리가 512MB로 넉넉하지 않다. 동시에 너무 많은 계산
  프로세스가 뜨면 메모리 부족으로 서비스 전체가 죽을 수 있어서, 아래
  MAX_CONCURRENT_JOBS로 "동시에 실제로 계산 중인" 프로세스 개수를 제한해둔다.
  그 자리가 다 찼을 때 들어온 요청은 에러가 아니라 그냥 자리가 날 때까지
  대기했다가 순서대로 실행된다. 현장에서 여러 대로 동시에 테스트해보고
  Render 로그에 메모리 부족(OOM) 흔적이 없으면 이 값을 조금씩 올려봐도 된다.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import threading
import unicodedata

from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "지역별 태양광, 풍력")
WORKER_PATH = os.path.join(BASE_DIR, "worker.py")

# 풍력 데이터 품질 점검 결과 (v1과 동일 — README 참고)
WIND_OK_REGIONS = ["강원", "경남", "경북", "전남", "전북"]
WIND_YEAR = 2023
SOLAR_YEAR = 2025

# 동시에 뜰 수 있는 "실제 계산 중" 프로세스 개수 상한 (Render 무료 플랜
# 512MB 메모리 보호용). 환경변수 MAX_CONCURRENT_JOBS로 조정 가능 (기본 2).
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
_JOB_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_JOBS)

# 워커 프로세스 타임아웃(초). Render 무료 플랜에서 계산이 1분 가까이 걸리는 걸
# 확인했으므로 여유를 두고 150초로 설정.
WORKER_TIMEOUT_S = 150


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s))


def scan_regions() -> dict:
    """data 폴더를 읽어서 {"태양광": [...지역들...], "풍력": [...]} 형태로 반환."""
    out: dict[str, list[str]] = {"태양광": [], "풍력": []}
    for kind in out:
        folder = os.path.join(DATA_ROOT, kind)
        if not os.path.isdir(folder):
            continue
        regions = []
        for p in glob.glob(os.path.join(folder, "*.xlsx")):
            base = _nfc(os.path.basename(p))
            region = base.split("_")[0]
            regions.append(region)
        out[kind] = sorted(set(regions))
    out["풍력"] = sorted(set(out["풍력"]) & set(WIND_OK_REGIONS))
    return out


REGIONS = scan_regions()


def run_model_in_subprocess(region: str, kind: str) -> dict:
    year = WIND_YEAR if kind == "풍력" else SOLAR_YEAR
    cmd = [
        sys.executable, WORKER_PATH,
        "--region", region, "--kind", kind,
        "--data-root", DATA_ROOT, "--year", str(year),
    ]
    # 동시 실행 개수 제한 — 자리가 날 때까지 여기서 기다린다 (요청 실패가 아니라
    # 그냥 대기, v1의 COMPUTE_LOCK과 비슷하지만 "여러 자리"라는 점이 다르다)
    with _JOB_SEMAPHORE:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S,
            cwd=BASE_DIR,
        )

    # pem_model.py의 원본 로그는 워커가 stderr로 돌려놨으니 여기서 서버 로그로 남긴다
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if not proc.stdout.strip():
        raise RuntimeError(f"워커가 아무 출력도 내지 않았습니다 (exit={proc.returncode})")

    payload = json.loads(proc.stdout)
    if proc.returncode != 0 or "error" in payload:
        raise RuntimeError(payload.get("error", f"워커 실패 (exit={proc.returncode})"))
    return payload


@app.route("/")
def index():
    return render_template("index.html", regions=REGIONS)


@app.route("/api/regions")
def api_regions():
    return jsonify(REGIONS)


@app.route("/api/run")
def api_run():
    region = _nfc(request.args.get("region", ""))
    kind = _nfc(request.args.get("kind", "태양광"))
    if not region:
        return jsonify({"error": "region 파라미터가 필요합니다."}), 400
    if region not in REGIONS.get(kind, []):
        return jsonify({"error": f"'{kind}'에 '{region}' 데이터가 없습니다."}), 400

    try:
        payload = run_model_in_subprocess(region, kind)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "계산이 너무 오래 걸려 중단됐습니다. 다시 시도해주세요."}), 504
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"계산 중 오류: {exc}"}), 500

    return jsonify(payload)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
