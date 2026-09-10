set -u
cd /home/user/Formant_ml
run() { tag=$1; n=$2; shift 2
  mkdir -p "out/$tag"
  OMP_NUM_THREADS=2 .venv/bin/python scripts/copyfit.py "data/voices/yang_00000${n}.wav" \
    --profile profiles/yang_female.json --out "out/$tag/s${n}" "$@" \
    > "scratchpad/${tag}_${n}.log" 2>&1
  echo "$(date +%H:%M) $tag/$n 끝" >> scratchpad/queue.log
}
run hc 040 --corr 2.0 --hnr 2.0
echo Q1 > scratchpad/q1.done
run hc 101 --corr 2.0 --hnr 2.0
echo DONE > scratchpad/q4.done
