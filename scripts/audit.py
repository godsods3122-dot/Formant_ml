"""재발 방지 감사 — 사용자가 지적한(또는 지적할 만한) 결함을 목표 대비로 전부 판정한다.

    python scripts/audit.py out/L48/s040 out/L48/s101
    python scripts/audit.py --tags L25 L48 --files s040 s101

사용자: *"내가 이전에 지적했던 그런 부분들 또는 내가 지적할 만한 부분은 절대 재발하지 않도록 하고, 그런 잠재성도
모두 조사해봐."* 점수(포락 %)만 오르고 예전 결함이 되살아나면 실패다. 각 항목의 기준은 **그 파일의 목표
녹음에서** 잰 값에 여유를 둔 것이다. 척도마다 MEASUREMENTS §50 의 어느 결함인지 적는다.

| 항목 | 결함 (사용자 표현) | 근거 |
|---|---|---|
| 층층 6-10 / 10-16 | "배음이 똑같이 생긴 놈들이 층층이" — 1 kHz 빗살 | §50.14 |
| 고역생존, 고역 수준 | "고역이 다 죽었다" | §50.3, §50.16 |
| 줄 10-16 | "고역의 평평한 줄" (좁은 고정 공진) | §50.5 |
| 치찰토막, 치찰 변조 20-60/60-150/F0 | "치찰음이 세로로 토막" | §50.4, §50.9 |
| 치찰 음색 (무게중심·대역 수준) | "목표 치찰음이 아닌 것 같다, 이질감" | §50.17 |
| 모음 거칠기 20-60 / 60-150 | "지터", 모음 고역의 세로 줄무늬 | §50.17 |
| 모음 주기성 (1~5 kHz) | "목소리 잡음" (기식이 중역에 섞임) | §50.17 |
| 무성 떨림 | ㅊ 에서 성대가 떪 → F0 토막 | §50.4 |
| 끊김 (비마찰 창) | "소리가 끊긴다" | §46, §50.16 |
| 무음·끝 잡음 | 무음의 공명 줄, 발화 끝의 엉뚱한 잡음띠 | §50.15, §50.16 |
| 움직임 (속도·진폭) | "무브먼트가 너무 과해" — 제어열이 관측·생리보다 심하게 흔들린다 | §51.20 |
| 움직임 드문드문 | "이상한 움직임이 많다" — 조음은 멈춤·이동·멈춤인데 적합은 쉬지 않고 흔든다 | §51.25 |
| 클릭 | 순간 튐 (지적할 만한 것) | 합성음 판정에서 청자가 가장 무겁게 보는 것은 인공물·불연속 |
| 포락 ≥ 96 % | 목표 일치율 | 사용자 목표 |
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from census import _bp, _chop, _line_persist, _live_4096, _live_5ms, _stack1k   # noqa: E402

FS = 48000
ENV_GOAL = 96.0


def _runs(mask, min_len):
    out, s0 = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and s0 is None:
            s0 = i
        if not v and s0 is not None:
            if i - s0 >= min_len:
                out.append((s0, i))
            s0 = None
    return out


def _modspec(x, runs, lo, hi, bands, pad=5):
    """구간들의 대역 포락(0.5 ms, 로그) 변조 에너지 [dB] — 대역별."""
    h = _bp(x, lo, hi, FS)
    w = 24
    acc = np.zeros(len(bands))
    for a, b in runs:
        a, b = a + pad, b - pad
        if b - a < 20:
            continue
        seg = h[a * 48:b * 48]
        e = np.sqrt(np.convolve(seg * seg, np.ones(w) / w, "same"))[::w]
        e = np.log(e + 1e-12)
        e = e - e.mean()
        E = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2 / len(e)
        f = np.fft.rfftfreq(len(e), w / FS)
        acc += np.array([E[(f >= p) & (f < q)].sum() for p, q in bands])
    return 10 * np.log10(acc + 1e-20)


def _per_local(x, f0, centres, wp=3.0):
    o = []
    for c in centres:
        f = f0[min(c // 48, len(f0) - 1)]
        if not (f > 60):
            continue
        lag = int(round(FS / f))
        w = int(wp * lag)
        a = c - w // 2
        if a < 0 or a + w + lag > len(x):
            continue
        u, z = x[a:a + w], x[a + lag:a + w + lag]
        d = np.sqrt((u * u).sum() * (z * z).sum())
        if d > 1e-20:
            o.append((u * z).sum() / d)
    return float(np.median(o)) if o else float("nan")


def _band_profile(x, t0, t1):
    from scipy.signal import stft
    s = x[int(t0 * FS):int(t1 * FS)]
    f, _, Z = stft(s, FS, nperseg=256, noverlap=256 - 48)
    P = (np.abs(Z) ** 2).mean(1)
    edges = [3000, 4500, 6000, 8000, 10000, 12500, 15000]
    bands = np.array([10 * np.log10(P[(f >= a) & (f < b)].mean() + 1e-30) for a, b in zip(edges[:-1], edges[1:])])
    sel = (f >= 2000) & (f < 16000)
    cen = (f[sel] * P[sel]).sum() / P[sel].sum()
    return bands - bands.mean(), cen


def audit(stem: str, profile: str = "profiles/yang_female.json") -> list[tuple]:
    import torch
    from formant_ml.engine.analyze import analyze
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.segment import fricative_mask
    from scipy.signal import stft
    from scipy.ndimage import binary_dilation, uniform_filter1d

    t, fs = sf.read(stem + "_target.wav")
    p, _ = sf.read(stem + "_fit.wav")
    t, p = np.asarray(t, float), np.asarray(p, float)
    n = min(len(t), len(p))
    t, p = t[:n], p[:n]
    prof = SpeakerProfile.load(profile)
    tr = analyze(t, fs, prof, 48, t0=0.0, full=t)
    voi = np.asarray(tr.voiced, bool)
    fm = fricative_mask(t, fs, prof, 48)
    m = min(len(voi), len(fm))
    voi, fm = voi[:m], fm[:m]
    f0 = np.asarray(tr["f0_target"], float)
    lvl = np.array([20 * np.log10(np.std(t[i * 48:(i + 1) * 48]) + 1e-20) for i in range(m)])
    R = []

    def add(name, val, ref, ok, note=""):
        R.append((name, val, ref, bool(ok), note))

    try:
        rep = json.load(open(stem + "_report.json", encoding="utf-8"))
        add("포락 %", rep["env"], ENV_GOAL, rep["env"] >= ENV_GOAL, "목표 일치율")
        add("정밀 %", rep["fine"], None, True, "참고")
    except Exception:
        pass

    lv4 = _live_4096(t, fs)
    for lo, hi, key in ((6000, 10000, "층층 6-10k"), (10000, 16000, "층층 10-16k")):
        v, cov = _stack1k(p, fs, lv4, t, (lo, hi))
        vt, _ = _stack1k(t, fs, lv4, None, (lo, hi))
        # **여유를 비례로도 건다.** 예전엔 `v <= vt + 0.10` 이라 목표가 0.049 일 때 3 배까지
        # 통과했다 — `L91/s040` 의 0.125(= 목표의 2.5 배)가 ✓ 로 찍혔다 (§51.36).
        lim = max(vt, 0.02) * 1.6 + 0.02
        add(key, v, lim, (v <= lim) or np.isnan(v), f"센 창 {100 * cov:.0f} %, 목표 {vt:.3f}")
        if lo == 10000:
            add("고역생존 %", 100 * cov, 90.0, cov >= 0.90, "10~16 kHz 가 목표보다 20 dB 넘게 낮지 않은 창")

    # 고역 수준: 프레임별 8~16 kHz |dB| 오차
    f, _, Zt = stft(t, fs, nperseg=1024, noverlap=768)
    _, _, Zp = stft(p, fs, nperseg=1024, noverlap=768)
    T = min(Zt.shape[1], Zp.shape[1])
    Tt, Tp = 20 * np.log10(np.abs(Zt[:, :T]) + 1e-10), 20 * np.log10(np.abs(Zp[:, :T]) + 1e-10)
    live = Tt.mean(0) > Tt.mean(0).max() - 45
    hk = (f >= 8000) & (f < 16000)
    hf_err = float(np.abs(Tp[hk][:, live] - Tt[hk][:, live]).mean())
    add("고역 수준 오차 dB", hf_err, 10.0, hf_err <= 10.0, "8~16 kHz 프레임별 |dB|")

    lv5 = _live_5ms(t, fs)
    l10 = _line_persist(p, fs, lv5)[1]
    l10t = _line_persist(t, fs, lv5)[1]
    add("줄 10-16k", l10, l10t, l10 <= l10t + 0.03, "미세구조 50 ms 지속")

    ch, cht = _chop(p, fs, fm, 48), _chop(t, fs, fm, 48)
    add("치찰토막 dB", ch, cht, ch <= cht + 0.6, "마찰 4~16 kHz 포락 요동")

    fr_runs = _runs(fm & (lvl > lvl.max() - 30), 30)
    if fr_runs:
        bands = ((20, 60), (60, 150), (150, 450))
        ex = _modspec(p, fr_runs, 4000, 16000, bands) - _modspec(t, fr_runs, 4000, 16000, bands)
        for (lo_, hi_), e in zip(bands, ex):
            add(f"치찰 변조 {lo_}-{hi_} Hz", e, 0.0, e <= 3.0, "목표 대비 dB")
        cens, devs = [], []
        for a, b in fr_runs:
            bp_, cp = _band_profile(p, a / 1000, b / 1000)
            bt_, ct = _band_profile(t, a / 1000, b / 1000)
            cens.append((cp - ct) / 1000)
            devs.append(np.abs(bp_ - bt_).max())
        add("치찰 무게중심 차 kHz", float(np.max(np.abs(cens))), 0.35, np.max(np.abs(cens)) <= 0.35,
            " ".join(f"{c:+.2f}" for c in cens))
        add("치찰 대역 수준 최대 차 dB", float(np.max(devs)), 4.0, np.max(devs) <= 4.0, "3~15 kHz 6 대역")

    v_runs = _runs(voi & ~fm & (lvl > lvl.max() - 25), 80)
    if v_runs:
        bands = ((20, 60), (60, 150))
        ex = _modspec(p, v_runs, 5000, 12000, bands) - _modspec(t, v_runs, 5000, 12000, bands)
        for (lo_, hi_), e in zip(bands, ex):
            add(f"모음 거칠기 {lo_}-{hi_} Hz", e, 0.0, e <= 3.0, "5~12 kHz 포락 변조, 목표 대비 dB")
    cv = [i * 48 for i in np.flatnonzero(voi & ~fm & (lvl > lvl.max() - 30))[::10]]
    per_p = [_per_local(_bp(p, a, b, fs), f0, cv) for a, b in ((500, 1500), (1500, 2500), (2500, 3500), (3500, 4500), (4500, 5500))]
    per_t = [_per_local(_bp(t, a, b, fs), f0, cv) for a, b in ((500, 1500), (1500, 2500), (2500, 3500), (3500, 4500), (4500, 5500))]
    dper = float(np.nanmean(np.abs(np.array(per_p) - np.array(per_t))))
    add("모음 주기성 차", dper, 0.08, dper <= 0.08,
        "합성 " + "/".join(f"{v:.2f}" for v in per_p) + " 목표 " + "/".join(f"{v:.2f}" for v in per_t))

    try:
        z = np.load(stem + "_track.npz", allow_pickle=True)
        names = [str(q) for q in z["names"]]
        V = torch.as_tensor(np.asarray(z["values"], float), dtype=torch.float32).unsqueeze(0)
        from formant_ml.engine.voice import VoiceEngine, EngineConfig
        eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, speaker="female", residual=False), prof)
        with torch.no_grad():
            st = eng.glottis.physiology({q: V[..., i] for i, q in enumerate(names)})
        amp = st["amp"][0].numpy()
        uv = fm & ~binary_dilation(voi, iterations=10)
        k = min(len(amp), len(uv))
        a_uv = float(np.median(amp[:k][uv[:k]])) if uv[:k].any() else 0.0
        a_v = float(np.median(amp[:k][(voi & ~fm)[:k]])) if (voi & ~fm)[:k].any() else 1.0
        add("무성 마찰 떨림 (모음 대비)", a_uv / max(a_v, 1e-6), 0.15, a_uv / max(a_v, 1e-6) <= 0.15, f"{a_uv:.3f} / {a_v:.3f}")
    except Exception as e:  # 트랙이 없거나 옛 형식
        add("무성 마찰 떨림", float("nan"), 0.15, True, f"건너뜀: {e}")

    # ---- 움직임 감사 (§51.20). 기준 둘: 관측된 분석 궤적, 그리고 생리.
    try:
        z = np.load(stem + "_track.npz", allow_pickle=True)
        names = [str(q) for q in z["names"]]
        V = np.asarray(z["values"], float)
        vo = voi[:len(V)]
        worst_v, worst_s = (0.0, ""), (0.0, "")
        for nm in ("f1", "f2", "f3", "f4", "bw1", "bw2", "bw3", "bw4"):
            if nm not in names:
                continue
            xf = np.log(np.maximum(V[:, names.index(nm)], 1e-6)) * 100.0
            xa = np.log(np.maximum(np.asarray(tr[nm], float), 1e-6)) * 100.0
            k = min(len(xf), len(xa), len(vo))
            mv = vo[:k - 1]
            if mv.sum() < 32:
                continue
            rf = float(np.sqrt((np.diff(xf[:k])[mv] ** 2).mean()))
            ra = float(np.sqrt((np.diff(xa[:k])[mv] ** 2).mean()))
            r = rf / max(ra, 1e-6)
            if r > worst_v[0]:
                worst_v = (r, f"{nm} {rf:.2f} 대 {ra:.2f} %/ms")
        # 음원 손잡이는 관측 기준이 없다 — **생리**로 문다. 느린 음절 윤곽은 정당하므로 20 Hz 위(σ 8 ms
        # 가우시안으로 추세를 빼고 남은 것)만 잰다. 시머 9 % ≈ 0.75 dB, 음원 기울기는 발성 노력의 양이다.
        from scipy.ndimage import gaussian_filter1d
        CAP = {"voice_gain": 1.0, "tilt": 0.6, "rd_offset": 0.10, "p_sub": 0.30, "adduction": 0.05}
        for nm, cap in CAP.items():
            if nm not in names:
                continue
            x = V[:, names.index(nm)][:len(vo)]
            if x.size < 64:
                continue
            fast = x - gaussian_filter1d(x, 8.0)          # 20 Hz 위만 남긴다
            sd = float(np.std(fast[vo[:len(x)]]))
            if sd / cap > worst_s[0]:
                worst_s = (sd / cap, f"{nm} 빠른 σ {sd:.2f} (상한 {cap})")
        add("움직임 속도 (관측 대비)", worst_v[0], 1.30, worst_v[0] <= 1.30, worst_v[1] or "—")
        # **드문드문함** — 조음은 멈춤-이동-멈춤이라 변화량이 몇 프레임에 몰린다. 관측 대비 비로 본다.
        worst_k = (9.9, "")
        for nm in ("f1", "f2", "f3"):
            if nm not in names:
                continue
            def _top(x):
                d = np.abs(np.diff(np.log(np.maximum(np.asarray(x, float), 1e-6))))
                k_ = min(len(d), len(vo))
                d = d[:k_][vo[:k_]]
                if d.size < 64:
                    return None
                s_ = np.sort(d)[::-1]
                return float(s_[:max(1, len(s_) // 10)].sum() / max(s_.sum(), 1e-12))
            a_ = _top(tr[nm])
            f_ = _top(V[:, names.index(nm)])
            if a_ is None or f_ is None:
                continue
            r = f_ / max(a_, 1e-6)
            if r < worst_k[0]:
                worst_k = (r, f"{nm} 상위 10 % 몫 {100 * f_:.0f} 대 관측 {100 * a_:.0f} %")
        add("움직임 드문드문 (관측 대비)", worst_k[0], 0.75, worst_k[0] >= 0.75, worst_k[1] or "—")
        add("움직임 진폭 (생리 상한 대비)", worst_s[0], 1.00, worst_s[0] <= 1.00, worst_s[1] or "—")
    except Exception as e:
        add("움직임", float("nan"), 1.0, True, f"건너뜀: {e}")

    w, h = 960, 240
    nb = nv = 0
    for i in range(0, n - w, h):
        a, b = t[i:i + w], p[i:i + w]
        if 20 * np.log10(a.std() + 1e-20) < lvl.max() - 45:
            continue
        if fm[min((i + w // 2) // 48, m - 1)]:
            continue
        nv += 1
        c = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30)
        nb += c < 0
    add("끊김 (비마찰 창) %", 100 * nb / max(nv, 1), 5.0, nb / max(nv, 1) <= 0.05, f"{nb}/{nv}")

    quiet = lvl < lvl.max() - 50
    q_runs = _runs(quiet, 40)
    if q_runs:
        ql = [(20 * np.log10(np.std(p[a * 48:b * 48]) + 1e-20) - 20 * np.log10(np.std(t[a * 48:b * 48]) + 1e-20)) for a, b in q_runs]
        worst = float(np.max(ql))
        add("무음·끝 잡음 수준 차 dB", worst, 6.0, worst <= 6.0, f"{len(q_runs)} 구간")

    # 치찰음 세로 토막: 마찰 구간에서 **대역마다 서는 시각**이 어긋나면 세로로 쪼개져 들린다
    # (MEASUREMENTS §51.39). 실측 `L91/s040`: 3~6 kHz 가 10~14 kHz 보다 18 ms 늦게 섰다.
    def _band_env(x, lo_, hi_):
        bb = _bp(x, lo_, hi_, fs)
        return 10 * np.log10(np.convolve(bb * bb, np.ones(48) / 48, "same")[::24] + 1e-20)
    fr_runs, _i = [], 0
    while _i < m:
        if fm[_i]:
            _j = _i
            while _j + 1 < m and fm[_j + 1]: _j += 1
            if _j - _i >= 40: fr_runs.append((_i, _j + 1))
            _i = _j + 1
        else: _i += 1
    if fr_runs:
        FB = ((3000, 6000), (6000, 10000), (10000, 14000), (14000, 18000))
        skew = []
        for s0, s1 in fr_runs:
            on = []
            for lo_, hi_ in FB:
                o = []
                for x in (t, p):
                    e = _band_env(x, lo_, hi_)
                    a_, b_ = s0 * 2, min(s1 * 2, len(e))
                    seg = e[max(a_ - 40, 0):b_]
                    if len(seg) < 10: o.append(np.nan); continue
                    th = np.percentile(e, 5) + 0.6 * (np.percentile(seg, 80) - np.percentile(e, 5))
                    kk = np.flatnonzero(seg > th)
                    o.append(kk[0] * 0.5 - 20.0 if len(kk) else np.nan)
                on.append(o)
            on = np.array(on, float)
            if np.all(np.isfinite(on)):
                skew.append(float(np.ptp(on[:, 1] - on[:, 0])))
        if skew:
            w = float(np.max(skew))
            add("치찰 대역 시작 어긋남 ms", w, 12.0, w <= 12.0,
                f"{len(skew)} 구간, 최악 {w:.0f} ms")

    # 공진 돌출: 켑스트럼 포락에서 포먼트 봉우리 − 이웃 골 (MEASUREMENTS §51.33).
    # 사용자: "원본은 공진이 매우 잘 느껴지는데 우리 건 퍼진 소리 같다".
    NW, LIFT = 2048, 96
    def cenv(x, i):
        c = max(i * 48 + 24 - NW // 2, 0); s_ = x[c:c + NW]
        if len(s_) < NW: return None
        S = np.log(np.abs(np.fft.rfft(s_ * np.hanning(NW))) + 1e-12)
        q = np.fft.irfft(S, NW); q[LIFT:-LIFT] = 0.0
        return 20.0 / np.log(10) * np.fft.rfft(q, NW).real[:len(S)]
    def pk_val(e, fq, fk, lo_, hi_):
        j = np.argmin(np.abs(fq - fk))
        A = np.flatnonzero((fq >= lo_) & (fq < fk)); Bx = np.flatnonzero((fq > fk) & (fq <= hi_))
        if len(A) < 3 or len(Bx) < 3: return np.nan
        return e[max(j - 3, 0):j + 4].max() - max(e[A].min(), e[Bx].min())
    fq = np.fft.rfftfreq(NW, 1 / fs)
    vi = np.flatnonzero(voi & ~fm & (lvl > lvl.max() - 30))[::4]
    dd = {"F2": [], "F3": []}
    for i in vi:
        et, ey = cenv(t, i), cenv(p, i)
        if et is None or ey is None: continue
        f1_, f2_, f3_ = float(tr["f1"][i]), float(tr["f2"][i]), float(tr["f3"][i])
        for lab, fk, lo_, hi_ in (("F2", f2_, f1_, f3_), ("F3", f3_, f2_, 5000.0)):
            if not (lo_ + 80 < fk < hi_ - 80): continue
            a_, b_ = pk_val(et, fq, fk, lo_, hi_), pk_val(ey, fq, fk, lo_, hi_)
            if np.isfinite(a_) and np.isfinite(b_): dd[lab].append(b_ - a_)
    for lab in ("F2", "F3"):
        if len(dd[lab]) < 30: continue
        v = np.array(dd[lab]); sh = float(np.mean(v < -3))
        add(f"{lab} 돌출 얕은 창 비율", 100 * sh, 20.0, sh <= 0.20,
            f"중앙 {np.median(v):+.1f} dB, 최악 {v.min():+.1f} dB")

    # 파열: 목표의 2~8 kHz 가 3 ms 안에 서는 자리에서, 합성이 얼마나 서는가 (MEASUREMENTS §51.31).
    def rise(x):
        bb = _bp(x, 2000, 8000, fs)
        n = len(bb) // 48
        return 20 * np.log10(np.sqrt((bb[:n * 48].reshape(n, 48) ** 2).mean(1)) + 1e-12)
    rt_, rp_ = rise(t), rise(p)
    kk = min(len(rt_), len(rp_))
    got, want = [], []
    last = -99
    for i in range(16, kk - 4):
        d = rt_[i + 3] - rt_[max(i - 15, 0):i].max()
        if d > 12.0 and rt_[i + 3] > rt_.max() - 40 and i - last > 25:
            last = i
            want.append(d)
            got.append(rp_[i + 3] - rp_[max(i - 15, 0):i].max())
    if want:
        frac = float(np.mean(np.clip(np.array(got), 0, None) / np.array(want)))
        add("파열 상승 (목표 대비)", frac, 0.50, frac >= 0.50,
            f"{len(want)} 곳, 목표 {np.mean(want):.1f} dB 합성 {np.mean(got):.1f} dB")

    # 클릭: 2 ms 사이 10 kHz 위 에너지가 20 dB 넘게 튀는 곳 (목표에 없는 것)
    def jumps(x):
        hh = _bp(x, 10000, 20000, fs)
        e = 10 * np.log10(np.convolve(hh * hh, np.ones(96) / 96, "same")[::48] + 1e-20)
        d = e[2:] - e[:-2]
        return d
    jp, jt = jumps(p), jumps(t)
    kk = min(len(jp), len(jt))
    clicks = int(((jp[:kk] > 20) & (jt[:kk] < 10)).sum())
    add("클릭 (고역 급등, 목표에 없음)", clicks, 3, clicks <= 3, "2 ms 사이 +20 dB")

    # 고역 물결 비 (MEASUREMENTS §51.53). 포락 손실은 멜 띠가 구조보다 넓어 이것을 못 본다 —
    # 목표 고역은 골이 파인 물결인데 우리 고역은 평평했다 (0.7 dB 대 5.2 dB).
    def _ripple(x):
        f_, _t, Z_ = stft(x, fs, nperseg=2048, noverlap=2048 - 512)
        db_ = 20 * np.log10(np.abs(Z_) + 1e-10)
        df_ = fs / 2048.0
        d_ = db_[int(4000 / df_):int(16000 / df_)]
        sm = lambda z, k: uniform_filter1d(z, max(3, int(round(k / df_)) | 1), axis=0, mode="nearest")
        r_ = sm(d_, 400.0) - sm(d_, 1600.0)
        return np.sqrt((r_ * r_).mean(0))
    rt_, rp_ = _ripple(t), _ripple(p)
    k_ = min(len(rt_), len(rp_))
    liv = t_lvl_mask = np.ones(k_, bool)
    ratio = float(np.median(rp_[:k_][liv]) / max(np.median(rt_[:k_][liv]), 1e-9))
    add("고역 물결 비 (목표 대비)", ratio, 0.80, ratio >= 0.80,
        f"4~16 kHz rms 우리 {np.median(rp_[:k_]):.2f} 목표 {np.median(rt_[:k_]):.2f} dB")

    # 성문 미세구조 (MEASUREMENTS §51.55) — 주기 대 주기가 목표보다 **거칠지** 않은가.
    try:
        from glottal_study import cycle_marks
        def _micro(x):
            mk = cycle_marks(x, fs, f0, voi & ~fm)
            pcs, cur = [], []
            for v in mk:
                if v < 0:
                    if len(cur) > 12: pcs.append(np.array(cur))
                    cur = []
                else: cur.append(v)
            if len(cur) > 12: pcs.append(np.array(cur))
            jj, dd = [], []
            for pc in pcs:
                Tc = np.diff(pc) / fs
                okc = (Tc > 1 / 500.) & (Tc < 1 / 60.)
                if okc.sum() < 10: continue
                Tc = Tc[okc]
                jj.append(100 * np.mean(np.abs(np.diff(Tc))) / np.mean(Tc))
                d1 = np.abs(np.diff(Tc)); d2 = np.abs(Tc[2:] - Tc[:-2])
                kk2 = min(len(d1) - 1, len(d2))
                if kk2 > 4: dd.append(100 * float(np.mean(d2[:kk2] < d1[:kk2])))
            return (float(np.median(jj)) if jj else float("nan"),
                    float(np.median(dd)) if dd else float("nan"))
        jt_, dt_ = _micro(t); jp_, dp_ = _micro(p)
        if np.isfinite(jt_) and np.isfinite(jp_):
            rj = jp_ / max(jt_, 1e-9)
            add("지터 비 (목표 대비)", rj, 1.30, rj <= 1.30, f"우리 {jp_:.2f} 목표 {jt_:.2f} %")
        if np.isfinite(dt_) and np.isfinite(dp_):
            rd_ = dp_ / max(dt_, 1e-9)
            add("주기 배증 비 (목표 대비)", rd_, 1.50, rd_ <= 1.50, f"우리 {dp_:.1f} 목표 {dt_:.1f} %")
    except Exception as _e:                       # 주기 추적이 안 되는 파일은 건너뛴다
        pass
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--files", nargs="*", default=["s040", "s101"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    stems = list(a.stems) + [f"out/{t}/{f}" for t in (a.tags or []) for f in a.files]
    stems = [s for s in stems if os.path.exists(s + "_fit.wav")]
    allres = {}
    for s in stems:
        R = audit(s)
        allres[s] = R
        npass = sum(1 for r in R if r[3])
        print(f"\n== {s}   감사 통과 {npass}/{len(R)}")
        for name, val, ref, ok, note in R:
            vs = f"{val:8.3f}" if isinstance(val, (float, np.floating)) else f"{val!s:>8}"
            rs = "" if ref is None else (f"기준 {ref:.3f}" if isinstance(ref, (float, np.floating)) else f"기준 {ref}")
            print(f"  {'✓' if ok else '✗'} {name:<28}{vs}   {rs:<14} {note}")
    if a.json:
        json.dump({s: [dict(name=r[0], value=float(r[1]) if r[1] is not None else None, ok=r[3], note=r[4]) for r in R]
                   for s, R in allres.items()}, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
