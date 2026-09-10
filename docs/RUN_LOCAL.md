# 내 컴퓨터에서 돌리기

원격 컨테이너는 격리돼 있어서 **거기서 내 컴퓨터로 붙는 통로는 없다.** 세션끼리
파일시스템을 공유하지 않는다. 대신 **내 컴퓨터에서 Claude Code 세션을 하나 띄우면**
그 세션은 내 파일시스템에 직접 접근하므로, 원격에서 하던 일을 내 메모리·내 CPU 로
그대로 이어서 한다.

## 0. 로컬 Claude Code 세션으로 이어받기

1. Claude Code 를 설치한다. 터미널이 편하면 CLI, 아니면 데스크톱 앱(Mac/Windows)
   이나 IDE 확장(VS Code, JetBrains)도 같은 일을 한다.
   설치 방법은 https://code.claude.com/docs 를 본다 (CLI 는 npm 패키지
   `@anthropic-ai/claude-code`).
2. 저장소를 받고 브랜치를 잡는다.

   ```bash
   git clone https://github.com/godsods3122-dot/Formant_ml
   cd Formant_ml
   git checkout claude/voice-parametrization-engine-fit-jscklg
   python3.11 -m venv .venv
   .venv/bin/pip install -e ".[dev]"
   ```

3. 그 디렉터리에서 `claude` 를 실행하고, 첫 메시지로 이렇게 준다:

   > `docs/HANDOFF.md` 와 `docs/MEASUREMENTS.md` 의 §35~§42 를 읽고 이어서 해줘.
   > 지금 할 일은 `docs/RUN_LOCAL.md` 의 "지금 남은 A/B" 다. 말은 되도록 한국어로.

   두 문서에 지금까지의 측정과 되돌린 시도가 전부 적혀 있어서, 새 세션이 같은
   실수를 반복하지 않는다.

4. 원격 세션과 겹치지 않게 **결과 태그(`out/<태그>`)를 다르게** 쓴다. 커밋은 같은
   브랜치에 해도 되지만, 밀기 전에 `git pull --rebase` 를 한 번 한다.

## 지금 남은 A/B

| 태그 | 명령에 더할 것 | 무엇을 보는가 |
|---|---|---|
| `lad` | (없음) | 혼합 사다리가 `out/fix` 보다 나은가 |
| `ph800` | `--phase-iters 800` | 위상 단계의 예산 (§39) |
| `ph800lr` | `--phase-iters 800 --lr-phase 0.12` | 탐침 편향까지 우회 (§39.1) |
| `sub1` | `--subf0 1.0` | F0 아래 초과 (+11.7 dB, 셋 중 가장 큼) |
| `sharp1` | `--sharp 1.0` | 유성 첨예도 (0.867) |
| `cont1` | `--cont 1.0` | 창별 악화 |

기준선은 `out/fix` (포락 92.44 / 90.87 %) 다. 파일은 `yang_00000040`,
`yang_00000101` 두 개를 같이 봐야 한다 — 한 파일만 보면 §38 처럼 결론이 갈린다.

## 준비

원격 컨테이너의 제약은 메모리다 — 적합 한 건이 **RSS 3.8 GB** 를 쓰는데 컨테이너의
가용 메모리가 9 GB 라 **두 건을 같이 돌리면 OOM 으로 조용히 죽는다** (로그에 아무
흔적도 안 남는다. `out/lad` 1 차 시도가 그렇게 사라졌다). 그래서 여기서는 직렬로만
돌린다. 메모리가 넉넉한 기계에서는 병렬로 돌려 훨씬 빨리 끝난다.

## 준비

위 0 절의 clone·venv 까지 하면 끝이다. `data/voices/` 의 음원과
`profiles/yang_female.json` 은 저장소에 들어 있다.

## 한 건 돌리기

```bash
OMP_NUM_THREADS=2 .venv/bin/python scripts/copyfit.py \
    data/voices/yang_00000040.wav \
    --profile profiles/yang_female.json \
    --out out/lad/s040
```

끝나면 `out/lad/s040_target.wav`, `_fit.wav`, `_track.npz`, `_report.json` 이 생긴다.

## 여러 건 병렬로

메모리 / 3.8 GB 만큼 동시에 돌린다. 16 GB 면 3 개, 32 GB 면 7 개.

```bash
for n in 040 101 034 094; do
  OMP_NUM_THREADS=2 .venv/bin/python scripts/copyfit.py \
      data/voices/yang_00000${n}.wav --profile profiles/yang_female.json \
      --out out/lad/s${n} > log_${n}.txt 2>&1 &
done
wait
```

## GPU

`--device cuda` (또는 애플 실리콘이면 `--device mps`). 엔진은 전부 torch 연산이라
그대로 올라간다. **다만 병렬 개수는 GPU 메모리가 아니라 CPU 메모리로 정하라** —
목표 신호 분석(Praat)과 STFT 준비가 CPU 쪽에서 그만큼 쓴다.

```bash
.venv/bin/python scripts/copyfit.py data/voices/yang_00000040.wav \
    --profile profiles/yang_female.json --out out/lad/s040 --device cuda
```

## 결과를 읽는 자

```bash
# 대역 오차 · 구멍 · 골 깊이
.venv/bin/python scripts/bandcmp.py out/fix/s040 out/lad/s040
.venv/bin/python scripts/bandcmp.py --valleys out/lad/s040

# 프레임별 대역 편향 (장기 평균이 못 보는 것)
.venv/bin/python scripts/probe_band_bias.py out/lad/s040

# 공진 대비 — "울림이 부족하다" 를 재는 자
.venv/bin/python scripts/probe_contrast.py out/fix/s040 out/lad/s040

# 네 합격 조건
.venv/bin/python scripts/acceptance.py out/lad/s040 --seeds 5
```

## 요청받은 세 벌점 켜기

전부 기본값 0 이다. 세기는 A/B 로 정해야 한다.

```bash
--cont 1.0     # 창별 손실이 직전 창보다 나빠진 만큼
--sharp 1.0    # 유성 구간이 목표보다 뭉툭한 만큼
--subf0 1.0    # F0 아래가 목표보다 시끄러운 만큼
```

## 결과를 다시 여기로

```bash
git add out/<태그>
git commit -m "로컬 적합 결과"
git push
```

`out/` 이 `.gitignore` 에 있으면 `git add -f` 를 쓰거나, `_report.json` 과
`_track.npz` 만 올려도 된다 — 그 둘이면 여기서 다시 렌더해 분석할 수 있다.
