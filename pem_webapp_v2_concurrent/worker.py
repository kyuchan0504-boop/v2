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
  python3 worker.py --region 강원 --kind 태양광 --data-root <path> --year 2025
  성공: stdout에 JSON 결과 딱 한 줄, exit code 0
  실패: stdout에 {"error": "..."} JSON 한 줄, exit code 1
  pem_model.py가 계산 중 찍는 수많은 print() 로그는 전부 stderr로 돌려서
  stdout을 순수 JSON 전용 채널로 지킨다 (부모 프로세스가 json.loads()로 그대로 파싱).
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
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    try:
        import pem_model as m  # 이 프로세스에만 사는 전역변수
        from build_payload import build_payload

        m.RE_DATA_ROOT = args.data_root
        m.RE_REGION_NAME = args.region
        m.RE_KIND = args.kind
        m.RE_YEAR = args.year
        m.FIG_DIR = None
        m.SHOW_FIGURES = False
        m.PLOT_DETAIL_FIGURES = False
        m.PLOT_BATTERY_STRESS = False
        m.FAST_MODE = True

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = m.main(make_plots=False, run_case1_sweep=False, run_hysteresis=False)
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
