"""역모형 학습 자료 — 참값을 아는 합성 음성 (MEASUREMENTS §52.426).

분석기는 기관 매개(내전·협착·연구개)의 움직임을 못 잰다(§52.425 자 검증). 그래서 **이 엔진으로 참값을 아는 소리**를 만들고
소리 → 기관 매개 역모형을 학습해 코퍼스에 적용한다.

바탕은 코퍼스 분석 궤적(`out/corpus_an`)이다 — F0·포먼트·유성 타이밍은 이 화자의 실제 발화에서 온다. 기관 매개만 제스처 과정에서
무작위로 다시 뽑는다:

* 내전: 유성 구간 목표 U(0.30, 0.85), 무성 틈 U(0.00, 0.25). 전이는 S 자(시그모이드) — 성대 궤적 모양 (JSLHR 2019).
  전이 시간은 로그정규(중앙 40 ms). 하한은 문헌 최고 각속도(441 °/s, 51°)에서 나오는 ≈ 20 ms 의 절반 — 가끔 **의도적으로 빠른** 표본도 둔다.
* 협착 a_c: 분석 궤적(자음 구조)에 로그 영역의 느린 요동 + 드물게 탄음형 폐쇄(20~50 ms)를 더한다.
* 폐압: 5~15 cmH2O 의 느린 요동(150 ms). 연구개: 분석 값 + 드문 비음화 사건. Rd 오프셋·기울기·기식: 느린 요동.

    python scripts/inverse_data.py --out out/inv --variants 6 --jobs 10
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TARGETS = ("adduction", "a_c", "p_sub", "velum", "rd_offset")


def smooth_noise(rng, n, ms, sd):
    """σ = ms 가우시안으로 편 백색 잡음, 표준편차 sd."""
    from scipy.ndimage import gaussian_filter1d
    x = gaussian_filter1d(rng.standard_normal(n + 400), ms)[200:200 + n]
    return x / (x.std() + 1e-12) * sd


def sigmoid_path(rng, n, bounds, levels, t_ms):
    """경계마다 S 자로 이어지는 계단 — levels[i] 는 구간 i 의 목표, t_ms[i] 는 경계 i 의 전이 시간(10–90 %)."""
    out = np.full(n, levels[0], float)
    t = np.arange(n, dtype=float)
    for k, b in enumerate(bounds):
        w = max(t_ms[k], 1.0) / 4.39                     # 10–90 % 폭 = 4.39 × 로지스틱 척도
        out += (levels[k + 1] - levels[k]) / (1.0 + np.exp(np.clip(-(t - b) / w, -60.0, 60.0)))
    return out


def make(args):
    path, out, k, seed = args
    dst = os.path.join(out, f"{os.path.basename(path)[:-4]}_{k}.npz")
    if os.path.exists(dst):
        return dst, "skip"
    rng = np.random.default_rng(seed)
    from formant_ml.engine.control import ControlTrack, INDEX
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.voice import EngineConfig, VoiceEngine
    d = np.load(path, allow_pickle=True)
    V = d["values"].astype(float).copy()
    v = d["voiced"][:len(V)].astype(bool)
    n = len(V)
    # 구간을 3 s 로 자른다 (학습 창)
    L = min(n, 3000)
    s0 = int(rng.integers(0, max(1, n - L + 1)))
    V, v = V[s0:s0 + L], v[s0:s0 + L]
    n = L
    # 내전 — 유성/무성 경계마다 S 자
    edges = np.flatnonzero(np.diff(v.astype(int))) + 1
    lev = [rng.uniform(0.30, 0.85) if v[0] else rng.uniform(0.0, 0.25)]
    tms = []
    for e in edges:
        lev.append(rng.uniform(0.30, 0.85) if v[e] else rng.uniform(0.0, 0.25))
        fast = rng.random() < 0.1                        # 의도적으로 빠른 표본
        tms.append(rng.uniform(4.0, 12.0) if fast else float(np.exp(rng.normal(np.log(40.0), 0.45))))
    add = sigmoid_path(rng, n, edges, lev, tms) + smooth_noise(rng, n, 30, 0.03)
    V[:, INDEX["adduction"]] = np.clip(add, 0.0, 1.0)
    # 협착 — 분석 궤적 + 느린 로그 요동 + 드문 탄음형 폐쇄
    la = np.log(np.clip(V[:, INDEX["a_c"]], 0.02, 4.0)) + smooth_noise(rng, n, 40, 0.25)
    for _ in range(rng.poisson(n / 1500.0)):
        c = int(rng.integers(50, n - 50)); w = rng.uniform(20.0, 50.0)
        la += (np.log(0.05) - la[c]) * np.exp(-0.5 * ((np.arange(n) - c) / (w / 2.35)) ** 2)
    V[:, INDEX["a_c"]] = np.clip(np.exp(la), 0.02, 4.0)
    V[:, INDEX["p_sub"]] = np.clip(rng.uniform(6.0, 12.0) + smooth_noise(rng, n, 150, 1.5), 3.0, 18.0)
    vel = np.clip(V[:, INDEX["velum"]], 0, 1)
    for _ in range(rng.poisson(n / 2000.0)):
        c = int(rng.integers(50, n - 50)); w = rng.uniform(40.0, 120.0)
        vel = np.maximum(vel, rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((np.arange(n) - c) / (w / 2.35)) ** 2))
    V[:, INDEX["velum"]] = vel
    V[:, INDEX["rd_offset"]] = np.clip(smooth_noise(rng, n, 60, 0.4), -1.5, 1.5)
    V[:, INDEX["tilt"]] = np.clip(2.0 + smooth_noise(rng, n, 80, 1.0), -2.0, 8.0)
    V[:, INDEX["aspiration"]] = np.clip(np.exp(smooth_noise(rng, n, 60, 0.6)) * 0.2, 0.0, 2.0)
    prof = SpeakerProfile.load("profiles/yang_female.json")
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False, speaker="female"), prof)
    y = eng.render(ControlTrack(V, 1.0, voiced=v))
    y = y / (np.abs(y).max() + 1e-9) * rng.uniform(0.2, 0.9)
    y = y + rng.standard_normal(y.size) * 10 ** (rng.uniform(-75, -55) / 20)     # 녹음 바닥
    tg = np.stack([V[:, INDEX[t]] for t in TARGETS], 1).astype(np.float32)
    np.savez_compressed(dst, audio=y.astype(np.float32), targets=tg, voiced=v)
    return dst, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/corpus_an/*.npz")
    ap.add_argument("--out", default="out/inv")
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--start", type=int, default=0, help="변형 번호 시작 (자료를 늘릴 때)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(a.src))
    jobs = [(f, a.out, k, 1000 * i + k) for i, f in enumerate(files) for k in range(a.start, a.start + a.variants)]
    n = 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for dst, st in ex.map(make, jobs):
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    print("끝", n)


if __name__ == "__main__":
    main()
