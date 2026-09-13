"""포락 점수(1 − 멜 SC)의 오차가 **어디서** 나오는가 — 대역 × 구간 종류로 나눈 SC 분자의 몫.

    python scripts/error_map.py out/L25/s040 out/L46/s040

적합기와 같은 멜(256 점)·기대 스펙트럼 평활로 목표 Mt, 합성 Mp 를 만들고 Σ(Mt−Mp)² 를 칸마다 나눈다.
구간 종류: 모음(유성·비마찰), 마찰, 전이(유성 경계 ±15 ms), 조용함(정점 −40 dB 아래).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, soundfile as sf, torch
from scipy.ndimage import binary_dilation
from formant_ml.engine import fit as F
from formant_ml.engine.analyze import analyze
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.segment import fricative_mask
from formant_ml.engine.voice import VoiceEngine, EngineConfig
torch.set_num_threads(2)
prof = SpeakerProfile.load("profiles/yang_female.json")
BANDS = ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 24000))
for stem in sys.argv[1:]:
    t, fs = sf.read(stem + "_target.wav"); t = np.asarray(t, float)
    y = np.asarray(sf.read(stem + "_fit.wav")[0], float)[:len(t)]; y = np.pad(y, (0, len(t) - len(y)))
    tr = analyze(t, fs, prof, 48, t0=0.0, full=t)
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, speaker="female", residual=False), prof)
    fit = F.CopySynthFitter(eng, t, fs, tr, phase_weight=0.0)
    with torch.no_grad():
        yt = torch.as_tensor(y[None, :], dtype=fit.target.dtype)
        raw = fit._cabs(F._stft(yt, F.MEL_FFT, fit.wins[F.MEL_FFT]))[:, :fit.bin_max[F.MEL_FFT]]
        Mp = (fit.mel[:, :raw.shape[1]] @ fit._expect(raw, F.MEL_FFT))[0].numpy()
        Mt = fit.tgt_M[0].numpy()
    k = min(Mp.shape[1], Mt.shape[1]); Mp, Mt = Mp[:, :k], Mt[:, :k]
    E = (Mt - Mp) ** 2; tot = E.sum(); sc = np.sqrt(tot) / np.sqrt((Mt ** 2).sum())
    cf = fit.mel.numpy().argmax(1) * fs / F.MEL_FFT
    hop = F.MEL_FFT // 4
    fr_t = (np.arange(k) * hop) / fs * 1000
    idx = np.clip(fr_t.astype(int), 0, len(tr.voiced) - 1)
    voi = np.asarray(tr.voiced, bool); fm = fricative_mask(t, fs, prof, 48)
    fmi = fm[np.clip(idx, 0, len(fm) - 1)]; vi = voi[idx]
    edge = binary_dilation(voi, iterations=15) & ~binary_dilation(~voi, iterations=0) ^ binary_dilation(voi, iterations=15)
    trans = (binary_dilation(voi, iterations=15) & ~voi) | (voi & binary_dilation(~voi, iterations=15))
    tri = trans[idx]
    lv = 10 * np.log10((Mt ** 2).sum(0) + 1e-30); quiet = lv < lv.max() - 40
    cls = np.where(quiet, "조용함", np.where(fmi, "마찰", np.where(tri, "전이", np.where(vi, "모음", "기타"))))
    print(f"\n== {stem}: 포락 {100 * (1 - sc):.2f} %  (SC {sc:.4f}) — SC 분자 Σ(Mt−Mp)² 의 몫 [%]")
    names = ["모음", "전이", "마찰", "조용함", "기타"]
    print(f"  {'대역 Hz':<12}" + "".join(f"{n:>7}" for n in names) + "    합")
    for a, b in BANDS:
        sel = (cf >= a) & (cf < b)
        row = [100 * E[sel][:, cls == n].sum() / tot for n in names]
        print(f"  {a:>5}-{b:<6}" + "".join(f"{v:7.1f}" for v in row) + f"  {sum(row):6.1f}")
    row = [100 * E[:, cls == n].sum() / tot for n in names]
    print(f"  {'합':<12}" + "".join(f"{v:7.1f}" for v in row))
    # 모음 저역에서 오차의 성격: 수준(dB 평균 차) 대 모양
    sel = (cf < 2000)
    for n in ("모음", "전이"):
        m = cls == n
        if m.any():
            d = 20 * np.log10(Mp[sel][:, m].sum(0) + 1e-12) - 20 * np.log10(Mt[sel][:, m].sum(0) + 1e-12)
            print(f"  {n} 0~2 kHz 프레임 수준 차: 중앙 {np.median(d):+.2f} dB, |차| 평균 {np.abs(d).mean():.2f} dB")
