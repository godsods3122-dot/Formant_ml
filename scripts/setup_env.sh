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
echo "  export OMP_NUM_THREADS=2                       # 컨테이너에서 torch 4 스레드는 매우 느리다"
echo "  python -m pytest tests/engine -q               # v2 성질 테스트"
echo "  python scripts/v2_listen.py --out out/v2        # v2 청취 세트 + 측정표"
echo "  python -m pytest tests -q                      # v1 포함 전체"
echo "  python -m formant_ml.demo --out out            # CPU 약 2분"
echo "  python scripts/analyze.py out/*.wav"
