# 직렬 실행 — 한 작업이 RSS 3.8 GB 를 쓴다. 두 개를 같이 돌리면 OOM 으로 조용히 죽는다
# (로그에 아무 흔적도 안 남는다. out/lad 1 차 시도가 그렇게 사라졌다).
set -u
cd /home/user/Formant_ml
TAG="$1"; shift
mkdir -p "out/$TAG"
for n in "$@"; do
  OMP_NUM_THREADS=2 .venv/bin/python scripts/copyfit.py data/voices/yang_00000${n}.wav \
    --profile profiles/yang_female.json --out "out/$TAG/s${n}" ${EXTRA:-} \
    > "scratchpad/${TAG}_${n}.log" 2>&1
done
echo DONE > "scratchpad/${TAG}.done"
