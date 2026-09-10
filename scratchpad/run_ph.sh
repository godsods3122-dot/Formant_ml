set -u
cd /home/user/Formant_ml
mkdir -p out/ph800
for n in 040 101; do
  OMP_NUM_THREADS=2 .venv/bin/python scripts/copyfit.py data/voices/yang_00000${n}.wav \
    --profile profiles/yang_female.json --out out/ph800/s${n} --phase-iters 800 \
    > scratchpad/ph_${n}.log 2>&1 &
done
wait
echo DONE > scratchpad/ph.done
