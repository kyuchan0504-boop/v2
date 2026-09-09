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

두 번째 개선 (몇 명이 몰려도 절대 에러/타임아웃이 안 나게):
  처음 버전은 브라우저가 /api/run 요청을 보내고 계산이 끝날 때까지 그 요청 하나를
  계속 붙들고 있는 방식이었다. 이러면 대기줄이 길어졌을 때(예: 10명이 동시에
  몰리면 뒷사람은 몇 분을 기다려야 함) 그 긴 시간 동안 브라우저 요청이 열려있게
  되고, 결국 gunicorn/Render의 타임아웃에 걸려 "오류"처럼 보이는 실패가 날 수
  있었다.

  그래서 지금은 "작업 큐" 방식으로 바꿨다:
    1) /api/run 요청 → 계산을 큐에 등록만 하고 즉시 job_id를 돌려준다 (거의 0초)
    2) 브라우저는 /api/status/<job_id>를 1~2초마다 확인(polling)해서
       "대기 중(앞에 N명)" → "계산 중" → "완료" 순서로 화면에 보여준다
    3) 실제 계산은 백그라운드 워커 스레드들이 큐에서 하나씩 꺼내서 처리한다.
       동시에 실제로 계산 중인 개수는 MAX_CONCURRENT_JOBS로 제한한다(메모리 보호,
       Render 무료 플랜은 512MB). 그 이상 몰린 사람은 에러가 아니라 큐에서 순서를
       기다린다 — 몇 명이 들어와도 "터지지" 않고, 대기 시간만 늘어난다.

  주의: 이 설계는 gunicorn worker 프로세스가 정확히 1개(--workers 1)일 때만
  올바르게 동작한다. 작업 큐(JOB_QUEUE)와 작업 상태(JOBS)가 파이썬 프로세스 메모리
  안에만 있기 때문에, worker를 2개 이상으로 늘리면 요청이 서로 다른 프로세스로
  가면서 큐/상태가 공유되지 않아 오작동한다. 절대 --workers 값을 늘리지 말 것
  (동시 처리량은 MAX_CONCURRENT_JOBS와 --threads로 조절한다).
"""
from __future__ import annotations

import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
import unicodedata
import uuid

from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "지역별 태양광, 풍력")
WORKER_PATH = os.path.join(BASE_DIR, "worker.py")

# 풍력 데이터 품질 점검 결과 (v1과 동일 — README 참고)
WIND_OK_REGIONS = ["강원", "경남", "경북", "전남", "전북"]
WIND_YEAR = 2023
SOLAR_YEAR = 2025

# 동시에 "실제로 계산 중"일 수 있는 워커 스레드 개수 (Render 무료 플랜 512MB
# 메모리 보호용). 환경변수 MAX_CONCURRENT_JOBS로 조정 가능 (기본 2). 이 개수를
# 넘는 요청은 에러가 아니라 큐에서 대기한다 — 그래서 몇 명이 들어와도 안전하다.
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))

# 워커 프로세스 타임아웃(초). Render 무료 플랜에서 계산이 1분 가까이 걸리는 걸
# 확인했으므로 여유를 두고 150초로 설정.
WORKER_TIMEOUT_S = 150

# 완료/실패한 작업 기록을 얼마나 오래 들고 있을지(초). 오래된 기록은 주기적으로
# 청소해서 하루 종일 켜둬도 메모리가 계속 쌓이지 않게 한다.
JOB_RETENTION_S = 60 * 60  # 1시간

# ---------------------------------------------------------------------------
# 작업 큐 상태
# ---------------------------------------------------------------------------
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
_SEQ_LOCK = threading.Lock()
_seq_counter = 0


def _next_seq() -> int:
    global _seq_counter
    with _SEQ_LOCK:
        _seq_counter += 1
        return _seq_counter


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


def _prune_old_jobs() -> None:
    cutoff = time.time() - JOB_RETENTION_S
    with JOBS_LOCK:
        stale = [
            jid for jid, j in JOBS.items()
            if j["status"] in ("done", "error") and j.get("finished_at", 0) < cutoff
        ]
        for jid in stale:
            del JOBS[jid]


def _worker_loop() -> None:
    """백그라운드 워커 스레드. 큐에서 job_id를 하나씩 꺼내 계산하고 JOBS에 결과를 채운다."""
    while True:
        job_id = JOB_QUEUE.get()
        try:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    continue
                job["status"] = "running"
                job["started_at"] = time.time()
                region, kind = job["region"], job["kind"]

            try:
                payload = run_model_in_subprocess(region, kind)
                with JOBS_LOCK:
                    job["status"] = "done"
                    job["result"] = payload
                    job["finished_at"] = time.time()
            except subprocess.TimeoutExpired:
                with JOBS_LOCK:
                    job["status"] = "error"
                    job["error"] = "계산이 너무 오래 걸려 중단됐습니다. 다시 시도해주세요."
                    job["finished_at"] = time.time()
            except Exception as exc:  # noqa: BLE001
                with JOBS_LOCK:
                    job["status"] = "error"
                    job["error"] = f"계산 중 오류: {exc}"
                    job["finished_at"] = time.time()
        finally:
            JOB_QUEUE.task_done()
            _prune_old_jobs()


# 워커 스레드 풀을 앱 로딩 시 한 번만 띄운다 (MAX_CONCURRENT_JOBS개).
# gunicorn --workers 1 이어야 이 풀이 정확히 하나만 존재한다 (모듈 상단 주석 참고).
for _i in range(MAX_CONCURRENT_JOBS):
    threading.Thread(target=_worker_loop, name=f"pem-worker-{_i}", daemon=True).start()


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

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "region": region,
            "kind": kind,
            "seq": _next_seq(),
            "queued_at": time.time(),
        }
    JOB_QUEUE.put(job_id)
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "존재하지 않거나 만료된 작업입니다."}), 404

        status = job["status"]
        if status == "queued":
            ahead = sum(
                1 for j in JOBS.values()
                if j["status"] == "queued" and j["seq"] < job["seq"]
            )
            return jsonify({"status": "queued", "ahead": ahead})
        if status == "running":
            return jsonify({"status": "running"})
        if status == "done":
            return jsonify({"status": "done", "result": job["result"]})
        # error
        return jsonify({"status": "error", "error": job.get("error", "알 수 없는 오류")})


@app.route("/healthz")
@app.route("/api/debug")
def api_debug():
    """일꾼 스레드가 실제로 살아있는지 눈으로 확인하기 위한 임시 진단용 엔드포인트.
    문제가 해결되면 지워도 되지만, 지금은 원인 파악용으로 남겨둔다."""
    worker_threads = [
        {"name": t.name, "alive": t.is_alive()}
        for t in threading.enumerate()
        if t.name.startswith("pem-worker-")
    ]
    with JOBS_LOCK:
        job_summary = [
            {"job_id": jid[:8], "status": j["status"], "seq": j["seq"]}
            for jid, j in JOBS.items()
        ]
    return jsonify({
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "worker_threads_expected": MAX_CONCURRENT_JOBS,
        "worker_threads_found": worker_threads,
        "all_thread_names": [t.name for t in threading.enumerate()],
        "queue_size": JOB_QUEUE.qsize(),
        "jobs_in_memory": job_summary,
        "process_id": os.getpid(),
    })
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
