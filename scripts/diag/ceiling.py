"""**이 지표의 천장** — 결정론적 부분이 완벽해도 잡음의 실현이 다르면 몇 점인가.

env = 100·(1 − mel SC) 다. 음성의 마찰·기식은 **난수**라서 어떤 모형도 목표의 그 실현을
못 맞춘다. 그러면 완벽한 모형도 100 이 아니다. 그 상한을 목표 녹음만으로 잰다.

대리 신호 세 가지:
 (가) 무성 우세 칸의 **위상만** 새로 뽑는다 (크기 스펙트럼은 그대로) — 가장 관대한 상한
 (나) 무성 프레임을 **같은 단시간 스펙트럼의 독립 잡음**으로 갈아 끼운다 — 현실적인 상한
 (다) 전 구간의 위상을 새로 뽑는다 — 배음까지 실현이 달라진 경우 (하한 쪽 참고)
"""
import sys, numpy as np, torch, soundfile as sf
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from formant_ml.engine.fit import mel_bank, MEL_FFT
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
from formant_ml.engine.analyze import analyze
from scipy.signal import stft, istft
prof = SpeakerProfile.load("profiles/yang_female.json")
FS = 48000
MEL = mel_bank(MEL_FFT, FS, 80).numpy()

def env_score(a, b):
    """fit.py 와 같은 식: 멜 대역 크기의 스펙트럼 수렴도."""
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    f, _t, A = stft(a, FS, nperseg=MEL_FFT, noverlap=MEL_FFT - MEL_FFT // 4)
    _, _, B = stft(b, FS, nperseg=MEL_FFT, noverlap=MEL_FFT - MEL_FFT // 4)
    A, B = np.abs(A), np.abs(B)
    k = min(A.shape[0], MEL.shape[1])
    Ma, Mb = MEL[:, :k] @ A[:k], MEL[:, :k] @ B[:k]
    m = min(Ma.shape[-1], Mb.shape[-1]); Ma, Mb = Ma[..., :m], Mb[..., :m]
    d = Ma - Mb
    return 100.0 * (1.0 - np.sqrt((d * d).sum()) / (np.sqrt((Ma * Ma).sum()) + 1e-9))

def rephase(x, mask_tf=None, seed=0):
    rng = np.random.default_rng(seed)
    f, t_, Z = stft(x, FS, nperseg=1024, noverlap=1024 - 256)
    M, P = np.abs(Z), np.angle(Z)
    newP = rng.uniform(-np.pi, np.pi, P.shape)
    if mask_tf is None:
        P2 = newP
    else:
        w = np.asarray(mask_tf, float).ravel()
        if len(w) < P.shape[1]:
            w = np.pad(w, (0, P.shape[1] - len(w)), mode="edge")
        w = np.broadcast_to(w[None, :P.shape[1]], P.shape)
        P2 = np.where(w > 0.5, newP, P)
    _, y = istft(M * np.exp(1j * P2), FS, nperseg=1024, noverlap=1024 - 256)
    return y[:len(x)]

for stem in sys.argv[1:]:
    x, fs = sf.read(stem + "_target.wav"); x = np.asarray(x, float)
    if fs != FS:
        print(f"  ! {stem}: {fs} Hz"); continue
    fm = np.asarray(fricative_mask(x, fs, prof, 48), bool)
    tr = analyze(x, fs, prof, 48, t0=0.0, full=x)
    voi = np.asarray(tr.voiced, bool)
    m = min(len(fm), len(voi))
    unv = (~voi[:m]) | fm[:m]                       # 잡음 우세 프레임 (1 ms 격자)
    # 1024/256 STFT 프레임 격자로 옮긴다
    nfr = 1 + len(x) // 256
    idx = np.clip((np.arange(nfr) * 256) // 48, 0, m - 1)
    unv_fr = unv[idx].astype(float)
    print(f"\n== {stem}   무성·마찰 프레임 {100*unv.mean():.0f} %")
    print(f"   (가) 잡음 칸 위상만 새로   env = {env_score(x, rephase(x, unv_fr, 1)):6.2f} %")
    print(f"   (나) 잡음 칸 위상 새로 ×3 벌 평균 "
          f"{np.mean([env_score(x, rephase(x, unv_fr, s)) for s in (2,3,4)]):6.2f} %")
    print(f"   (다) 전 구간 위상 새로      env = {env_score(x, rephase(x, None, 5)):6.2f} %")
    zero = np.zeros_like(unv_fr)
    print(f"   **대조: 위상을 하나도 안 바꾼 왕복** env = {env_score(x, rephase(x, zero, 9)):6.2f} %")
    half = unv_fr * (np.arange(len(unv_fr)) % 2 == 0)
    print(f"   참고: 잡음 칸의 절반만 새로  env = {env_score(x, rephase(x, half, 11)):6.2f} %")
    print(f"   참고: 같은 신호끼리        env = {env_score(x, x):6.2f} %")
