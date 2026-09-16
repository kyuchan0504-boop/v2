#!/usr/bin/env python3
"""
독립 실행 워커 — 한 번의 지역/종류 계산을 완전히 격리된 프로세스에서 수행한다.

왜 이런 구조인가 (v1과의 차이):
  pem_model.py는 RE_REGION_NAME, J_MAX 같은 값을 "모듈 전역변수"로 두고
  여러 함수가 직접 참조한다. v1은 이걸 한 프로세스 안에서 Lock으로 직렬화해서
  안전하게 만들었다(한 번에 한 명씩만 계산). 이 워커는 매 요청마다 완전히
  새로운 파이썬 프로세스로 실행되므로, 전역변수가 프로세스별로 독립된 메모리에
  있어 여러 요청이 동시에 들어와도 절대 섞이지 않는다 — Lock 자체가 필요 없어진다.
  대신 매번 numpy/pandas/pem_model을 새로 import하는 비용(1~2초)이 드는데,
  실제 계산 시간(로컬 7~9초, Render 무료 플랜에서는 최대 1분 가까이)에 비하면
  크지 않다고 판단했다.

호출 규약:
  python3 worker.py --region 강원 --kind 태양광 --data-root <path> \
      --start-year 2018 --end-year 2025
  성공: stdout에 JSON 결과 딱 한 줄, exit code 0
  실패: stdout에 {"error": "..."} JSON 한 줄, exit code 1
  pem_model.py가 계산 중 찍는 수많은 print() 로그는 전부 stderr로 돌려서
  stdout을 순수 JSON 전용 채널로 지킨다 (부모 프로세스가 json.loads()로 그대로 파싱).

2026-09 업데이트 — pem_model.py 교체 (열화 반영 lifecycle LCOH 모델):
  main()의 시그니처가 main(make_plots, run_case1_sweep)로 바뀌어서
  run_hysteresis 인자가 사라졌다(구 모델 전용 인자였음) — 그래서 아래 호출에서
  뺐다. 새 모델은 Case 3(하이브리드 ESS) 최적점 탐색 중 ESS 용량이 병리적으로
  작은 후보를 만나면 시간별 이벤트 루프가 과도하게 느려지는 문제가 있어
  pem_model.py의 _size_ess에 최소 용량 하한을 추가해뒀다(정확도 손실 없이
  계산시간 약 7배 단축, 118초 -> 22초 로컬 실측).

2026-09 업데이트 #2 — 사전계산(precompute) 전용으로 전환:
  새 모델은 단일 연도(RE_YEAR) 대신 RE_START_YEAR/RE_END_YEAR 범위를 쓴다
  (지정하지 않으면 그 발전원의 기본 범위 전체를 계산해버려서 훨씬 느려짐 —
  반드시 명시할 것). 이제 이 워커는 실시간 웹 요청이 아니라 precompute.py가
  오프라인으로 결과를 미리 뽑아둘 때만 사용한다(app.py는 더 이상 이 워커를
  요청마다 부르지 않고, precompute.py가 만들어둔 results_cache/*.json을 그냥
  읽어서 즉시 응답한다). 그래서 엑셀 요약 저장(SAVE_FINAL_SUMMARY_XLSX)과 CSV
  저장(SAVE_RESULT_CSV)은 꺼서 필요없는 부수 파일이 생기지 않게 한다.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    args = parser.parse_args()

    try:
        import pem_model as m  # 이 프로세스에만 사는 전역변수
        from build_payload import build_payload

        m.RE_DATA_ROOT = args.data_root
        m.RE_REGION_NAME = args.region
        m.RE_KIND = args.kind
        m.RE_START_YEAR = args.start_year
        m.RE_END_YEAR = args.end_year
        m.FIG_DIR = None
        m.SHOW_FIGURES = False
        m.FAST_MODE = True
        m.SAVE_FINAL_SUMMARY_XLSX = False
        m.SAVE_RESULT_CSV = False

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = m.main(make_plots=False, run_case1_sweep=False)
        sys.stderr.write(buf.getvalue())  # 원본 로그는 서버 로그(stderr)로만 흘러가게

        payload = build_payload(m, out, args.region, args.kind)
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stdout.write(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
