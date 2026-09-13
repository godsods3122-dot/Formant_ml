"""남은 오차가 **어디** 있는가 — 구간(유성/마찰/무음)과 대역으로 쪼갠다.

env 는 전체 놈의 비라 한 덩어리로 보이지만, 분자 ‖목표−합성‖² 는 칸마다 더한 것이므로
그 기여를 그대로 쪼갤 수 있다.
"""
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from formant_ml.engine.fit import mel_bank, MEL_FFT
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
from formant_ml.engine.analyze import analyze
from scipy.signal import stft
prof = SpeakerProfile.load("profiles/yang_female.json")
FS = 48000
MEL = mel_bank(MEL_FFT, FS, 80).numpy()
fmel = np.array([np.argmax(MEL[i]) * (FS / MEL_FFT) for i in range(80)])

for stem in sys.argv[1:]:
    a, fs = sf.read(stem + "_target.wav"); b, _ = sf.read(stem + "_fit.wav")
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    f, _t, A = stft(a, FS, nperseg=MEL_FFT, noverlap=MEL_FFT - MEL_FFT // 4)
    _, _, B = stft(b, FS, nperseg=MEL_FFT, noverlap=MEL_FFT - MEL_FFT // 4)
    k = min(A.shape[0], MEL.shape[1])
    Ma, Mb = MEL[:, :k] @ np.abs(A[:k]), MEL[:, :k] @ np.abs(B[:k])
    m = min(Ma.shape[-1], Mb.shape[-1]); Ma, Mb = Ma[..., :m], Mb[..., :m]
    D2 = (Ma - Mb) ** 2
    tot = D2.sum(); den = np.sqrt((Ma * Ma).sum())
    env = 100 * (1 - np.sqrt(tot) / den)
    fm = np.asarray(fricative_mask(a, fs, prof, 48), bool)
    tr = analyze(a, fs, prof, 48, t0=0.0, full=a)
    voi = np.asarray(tr.voiced, bool)
    mm = min(len(fm), len(voi))
    hop = MEL_FFT // 4
    idx = np.clip((np.arange(m) * hop) // 48, 0, mm - 1)
    lvl = 20*np.log10(Ma.sum(0) + 1e-12); quiet = lvl < lvl.max() - 40
    seg = np.where(quiet, "무음", np.where(fm[idx], "마찰", np.where(voi[idx], "유성", "기타")))
    print(f"\n== {stem}   env {env:.2f} %   (천장 참고: §천장 측정)")
    print("   구간별 오차 기여 (분자 제곱합의 몫)")
    for s in ("유성", "마찰", "무음", "기타"):
        w = seg == s
        if not w.any(): continue
        print(f"     {s}  프레임 {100*w.mean():5.1f} %   오차 몫 {100*D2[:, w].sum()/tot:5.1f} %   "
              f"에너지 몫 {100*(Ma[:, w]**2).sum()/(Ma**2).sum():5.1f} %")
    print("   대역별 오차 기여")
    for lo, hi in ((0,1000),(1000,3000),(3000,6000),(6000,10000),(10000,16000),(16000,24000)):
        w = (fmel >= lo) & (fmel < hi)
        if not w.any(): continue
        print(f"     {lo//1000:2d}-{hi//1000:2d}k  멜 {w.sum():2d} 개   오차 몫 {100*D2[w].sum()/tot:5.1f} %   "
              f"에너지 몫 {100*(Ma[w]**2).sum()/(Ma**2).sum():5.1f} %")
