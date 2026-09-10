set -u
cd /home/user/Formant_ml
OMP_NUM_THREADS=2 .venv/bin/python scripts/copyfit.py data/voices/yang_00000040.wav \
  --profile profiles/yang_female.json --out out/lad/s040 > scratchpad/lad_040.log 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  echo "$(date +%H:%M:%S) rss=$(ps -o rss= -p $P 2>/dev/null | tr -d ' ')KB free=$(free -m | awk '/Mem:/{print $7}')MB" >> scratchpad/mem.log
  sleep 30
done
wait $P; echo "exit=$?" >> scratchpad/mem.log
echo DONE > scratchpad/lad.done
