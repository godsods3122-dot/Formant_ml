# 직렬 대기열. 한 건이 RSS 3.8 GB 라 동시에 못 돌린다.
set -u
cd /home/user/Formant_ml
while [ ! -f scratchpad/lad.done ]; do sleep 30; done
run() {  # run <태그> <파일번호> [추가 인자...]
  tag=$1; n=$2; shift 2
  mkdir -p "out/$tag"
  OMP_NUM_THREADS=2 .venv/bin/python scripts/copyfit.py "data/voices/yang_00000${n}.wav" \
    --profile profiles/yang_female.json --out "out/$tag/s${n}" "$@" \
    > "scratchpad/${tag}_${n}.log" 2>&1
  echo "$(date +%H:%M) $tag/$n 끝" >> scratchpad/queue.log
}
run lad 101
echo Q1 > scratchpad/q1.done
run sub1 040 --subf0 1.0
run sub1 101 --subf0 1.0
echo Q2 > scratchpad/q2.done
run ph800 040 --phase-iters 800
run ph800 101 --phase-iters 800
echo DONE > scratchpad/queue.done
