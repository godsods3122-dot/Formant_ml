set -u
cd /home/user/Formant_ml
run() { tag=$1; n=$2; shift 2
  mkdir -p "out/$tag"
  OMP_NUM_THREADS=2 .venv/bin/python scripts/copyfit.py "data/voices/yang_00000${n}.wav" \
    --profile profiles/yang_female.json --out "out/$tag/s${n}" "$@" \
    > "scratchpad/${tag}_${n}.log" 2>&1
  echo "$(date +%H:%M) $tag/$n 끝" >> scratchpad/queue.log
}
while pgrep -f "out/lad/s101" > /dev/null; do sleep 30; done
echo "$(date +%H:%M) lad/101 끝" >> scratchpad/queue.log
run ph800 040 --phase-iters 800
echo Q1 > scratchpad/q1.done
run ph800 101 --phase-iters 800
echo Q2 > scratchpad/q2.done
run sub1 040 --subf0 1.0 --phase-iters 800
run sub1 101 --subf0 1.0 --phase-iters 800
echo DONE > scratchpad/queue.done
