"""원본 녹음의 짧은 과도음(튐) — 찾기와 자동 클릭 규칙 (MEASUREMENTS §52.464).

`scripts/diag/corpus_transients.py` 에 있던 것을 엔진으로 옮겼다(§52.470) — 엔진(`pipeline.load_clean`)이 `scripts/` 를 거꾸로 불러오지 않게.
코퍼스 목록·군집 같은 진단은 그 스크립트에 남고 여기서 `scan` 을 가져다 쓴다.

**검출**: 대역(1.5–6 / 6–12 / 12–20 kHz)마다 0.5 ms 포락의 ±10 ms 중앙값 대비 솟음의 최댓값이 `RISE_DB` 넘는 봉우리 (1.5 kHz 위 수준이 파일 최대 −60 dB 안).
**특징** (`COLS`): 지속, 대역 솟음, 고립도(뒤 5~15 ms − 앞 −15~−5 ms), 유성 문맥, 반복도(주기 T0 앞뒤 대비), 3 ms 파형 조각.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

RISE_DB = 10.0
HOP_MS, WIN_MS, BASE_MS = 0.125, 0.5, 10.0
SNIP = (-0.5, 2.5)          # 조각 [ms]
CORR_TIGHT = 0.85
REP_DB = 6.0               # 반복도 문턱 — 주기 앞뒤보다 이만큼 커야 '되풀이 안 함'
BANDS = ((100.0, 1500.0), (1500.0, 6000.0), (6000.0, 12000.0), (12000.0, 20000.0))


def env_db(y, sr, hop, w):
    win = np.hanning(w + 2)[1:-1]
    return 10.0 * np.log10(np.convolve(y * y, win / win.sum(), mode="same")[::hop] + 1e-20)


def scan(path, sr=None) -> dict:
    """path: 파일 경로 또는 (신호 배열, sr 를 함께 줌)."""
    if isinstance(path, str):
        x, sr = sf.read(path, dtype="float64")
    else:
        x = np.asarray(path, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    hop = max(1, int(round(HOP_MS * 1e-3 * sr)))
    w = max(2, int(round(WIN_MS * 1e-3 * sr)))
    xh = sosfiltfilt(butter(4, 1500.0, "hp", fs=sr, output="sos"), x)
    e = env_db(xh, sr, hop, w)
    k = int(round(2 * BASE_MS * 1e-3 * sr / hop)) | 1
    p = e - median_filter(e, size=k, mode="nearest")
    bandE = [env_db(sosfiltfilt(butter(4, (lo, min(hi, 0.45 * sr)), "bp", fs=sr, output="sos"), x), sr, hop, w) for lo, hi in BANDS]
    bandP = [b - median_filter(b, size=k, mode="nearest") for b in bandE]
    lo_sig = sosfiltfilt(butter(4, (80.0, 1500.0), "bp", fs=sr, output="sos"), x)
    top = e.max()
    ms = lambda v: int(round(v * 1e-3 * sr / hop))
    # **대역별 솟음의 최댓값**으로 찾는다 — 1.5 kHz 위를 통째로 보면 모음 속 클릭(079 1.2085 s, 8–16 kHz +16 dB)이 모음 고역에 묻힌다.
    p = np.max(np.stack(bandP[1:]), axis=0)
    pk = np.flatnonzero((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] >= RISE_DB) & (e[1:-1] > top - 60.0)) + 1
    keep, last = [], -10 ** 9
    for i in pk[np.argsort(-p[pk])]:
        if all(abs(i - j) > ms(3.0) for j in keep):
            keep.append(i)
    rows, snips = [], []
    a0, a1 = int(SNIP[0] * 1e-3 * sr), int(SNIP[1] * 1e-3 * sr)
    for i in sorted(keep):
        c = i * hop
        if c + a0 < 0 or c + a1 > len(x):
            continue
        # 지속: 솟음이 봉우리 −6 dB 위에 머문 폭
        j0 = i
        while j0 > 0 and p[j0 - 1] > p[i] - 6.0:
            j0 -= 1
        j1 = i
        while j1 + 1 < len(p) and p[j1 + 1] > p[i] - 6.0:
            j1 += 1
        before = np.mean(e[max(0, i - ms(15)):max(1, i - ms(5))]) if i > ms(5) else e[i] - 30
        after = np.mean(e[i + ms(5):i + ms(15)]) if i + ms(15) < len(e) else e[-1]
        # 유성 문맥: 앞 −25~−5 ms 와 뒤 5~25 ms 의 저역 자기상관 최고 (2.5~12.5 ms 주기)
        def per(s0, s1):
            s0, s1 = max(0, s0), min(len(lo_sig), s1)
            u = lo_sig[s0:s1] - lo_sig[s0:s1].mean() if s1 - s0 > 100 else None
            if u is None or np.dot(u, u) < 1e-16:
                return 0.0
            r = np.correlate(u, u, "full")[len(u) - 1:]
            l0, l1 = int(sr / 400), min(int(sr / 80), len(r) - 1)
            return float(r[l0:l1].max() / r[0]) if l1 > l0 else 0.0
        pv_b, pv_a = per(c - int(0.025 * sr), c - int(0.005 * sr)), per(c + int(0.005 * sr), c + int(0.025 * sr))
        # **반복도**: 유성 문맥이면 주기 T0 (저역 ±20 ms 자기상관) 앞뒤의 같은 대역 솟음과 견준다 — 성문 폐쇄의 고역 솟음은 주기마다 되풀이되고
        # (0 dB 근처) 클릭은 안 되풀이된다 (declick 의 성문 터짐 판정과 같은 생각). 무성 문맥이면 반복도는 솟음 그대로.
        bi = 1 + int(np.argmax([bp[i] for bp in bandP[1:]]))
        rep = p[i]
        if max(pv_b, pv_a) > 0.5:
            s0, s1 = max(0, c - int(0.02 * sr)), min(len(lo_sig), c + int(0.02 * sr))
            u = lo_sig[s0:s1] - lo_sig[s0:s1].mean()
            r = np.correlate(u, u, "full")[len(u) - 1:]
            l0, l1 = int(sr / 400), min(int(sr / 80), len(r) - 1)
            if l1 > l0 and r[0] > 0:
                T0 = (l0 + int(np.argmax(r[l0:l1]))) / sr
                d = max(1, int(round(T0 * sr / hop)))
                win_ = ms(0.4)
                nb = [bandE[bi][max(0, i + sg * d - win_):i + sg * d + win_ + 1].max() for sg in (-1, 1)
                      if 0 <= i + sg * d < len(bandE[bi])]
                if nb:
                    rep = bandE[bi][i] - max(nb)
        sn = xh[c + a0:c + a1].copy()
        sn /= np.linalg.norm(sn) + 1e-20
        rows.append([c / sr, p[i], e[i] - top, (j1 - j0 + 1) * hop / sr * 1e3, after - before,
                     pv_b, pv_a] + [bp[i] for bp in bandP] + [rep])
        snips.append(sn)
    return dict(rows=np.array(rows, float).reshape(-1, 12), snips=np.array(snips, float).reshape(len(snips), a1 - a0))



#: **자동 클릭 규칙** (§52.464). 정답(079 의 1.2085·1.668 s 마우스 클릭) 둘을 잡고, 음성 대조군(079 ㅌ·ㄲ 개방, ㄱ 약한 개시, 121 ㅋ 개방, 055 유성 파열
#: 0.178 s, 121 ㅅ, 055 C2 creak)에서 하나도 안 잡도록 맞췄다. 되풀이 안 함(성문 폐쇄 솟음이 아님), 앞뒤 수준이 같음(파열·개시가 아님), 짧음, 또렷함,
#: ±15 ms 안에 비슷한 세기의 다른 튐이 없음(creak 는 불규칙 펄스가 줄지어 온다).
AUTO_REP_DB, AUTO_ISO_DB, AUTO_WIDTH_MS, AUTO_RISE_DB, AUTO_COMP_MS = 8.0, 4.0, 1.5, 12.0, 15.0


def auto_click_times(x, sr) -> list[float]:
    """원본 신호에서 자동 규칙에 걸리는 고립 클릭 시각 [s]."""
    R = scan(x, sr)["rows"]
    if not len(R):
        return []
    out = []
    for i in np.flatnonzero((R[:, 11] >= AUTO_REP_DB) & (np.abs(R[:, 4]) <= AUTO_ISO_DB) & (R[:, 3] <= AUTO_WIDTH_MS) & (R[:, 1] >= AUTO_RISE_DB)):
        dt = np.abs(R[:, 0] - R[i, 0])
        if not np.any((dt > 1e-3) & (dt <= AUTO_COMP_MS * 1e-3) & (R[:, 1] >= R[i, 1] - 6.0)):
            out.append(float(R[i, 0]))
    return out


COLS = ["t", "솟음", "수준(최대대비)", "지속ms", "고립도", "앞주기성", "뒤주기성", "p0.1-1.5k", "p1.5-6k", "p6-12k", "p12-20k", "반복도"]
