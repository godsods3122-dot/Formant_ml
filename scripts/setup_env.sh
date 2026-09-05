#!/usr/bin/env bash
# 개발 환경 구축. 프로젝트 루트에서 실행한다.
#
#   bash scripts/setup_env.sh
#   source .venv/bin/activate
#
# 주의: 환경에 따라 pytorch.org 가 막혀 있을 수 있다. 아래는 PyPI 만 쓴다.
# editable 설치를 하므로 PYTHONPATH=src 를 따로 걸 필요가 없다.
set -euo pipefail

PY=${PYTHON:-python3}
VENV=${VENV:-.venv}

"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e ".[dev]"

echo
echo "완료. 다음으로:"
echo "  source $VENV/bin/activate"
echo "  python -m pytest tests -q                      # 92종 검증"
echo "  python -m formant_ml.demo --out out            # CPU 약 2분"
echo "  python scripts/analyze.py out/*.wav"
