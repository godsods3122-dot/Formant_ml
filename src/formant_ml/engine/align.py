"""음소 강제 정렬 — 다국어 음소 인식기(wav2vec2 XLSR-53 espeak, CTC) 의 틀별 확률에 정답 음소열을 비터비로 맞춘다 (§52.487).

틀 간격 20 ms. 음소 i 의 구간 = [첫 틀 시작, 다음 음소 첫 틀 시작). 앞·뒤의 빈 틀은 쉼(`sil`)이다. 모델은 처음 부를 때 받아 캐시한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
FRAME_S = 0.02
_CACHE = {}


def _model():
    if "m" not in _CACHE:
        import torch  # noqa: F401
        from huggingface_hub import hf_hub_download
        from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC
        _CACHE["fe"] = AutoFeatureExtractor.from_pretrained(MODEL)
        _CACHE["m"] = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
        _CACHE["vocab"] = json.load(open(hf_hub_download(MODEL, "vocab.json"), encoding="utf-8"))
    return _CACHE["fe"], _CACHE["m"], _CACHE["vocab"]


def log_probs(x: np.ndarray, sr: int) -> np.ndarray:
    import torch
    from scipy.signal import resample_poly
    fe, m, _ = _model()
    y = resample_poly(np.asarray(x, np.float64), 16000, sr).astype(np.float32)
    with torch.no_grad():
        lg = m(**fe(y, sampling_rate=16000, return_tensors="pt")).logits[0]
    return torch.log_softmax(lg, -1).numpy().astype(np.float64)


def ctc_align(lp: np.ndarray, tokens: list[int], blank: int) -> list[tuple[int, int]]:
    """CTC 비터비 강제 정렬 → 토큰마다 (첫 틀, 끝 틀) (끝 포함)."""
    T = lp.shape[0]
    ext = [blank]
    for t in tokens:
        ext += [t, blank]
    S = len(ext)
    NEG = -1e18
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int64)
    dp[0, 0] = lp[0, ext[0]]
    if S > 1:
        dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(dp[t - 1, s], s)]
            if s >= 1:
                cands.append((dp[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                cands.append((dp[t - 1, s - 2], s - 2))
            v, k = max(cands)
            dp[t, s] = v + lp[t, ext[s]]
            bp[t, s] = k
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = [s]
    for t in range(T - 1, 0, -1):
        s = bp[t, s]
        path.append(s)
    path = path[::-1]
    spans = []
    for i in range(len(tokens)):
        fr = [t for t, s in enumerate(path) if s == 2 * i + 1]
        spans.append((fr[0], fr[-1]) if fr else (None, None))
    # 빈 자리(같은 토큰 연속 등으로 못 잡은 것)는 이웃 사이로 메운다
    for i, (a, b) in enumerate(spans):
        if a is None:
            prev_end = spans[i - 1][1] if i else 0
            nxt = next((spans[j][0] for j in range(i + 1, len(spans)) if spans[j][0] is not None), T - 1)
            spans[i] = (prev_end, max(prev_end, nxt - 1))
    return spans


@dataclass
class Seg:
    label: str
    t0: float
    t1: float


def align(x: np.ndarray, sr: int, ipa: list[str]) -> list[Seg]:
    """음소열 → 구간 [s]. 앞뒤 쉼은 `sil`."""
    _, _, vocab = _model()
    lp = log_probs(x, sr)
    blank = vocab["<pad>"]
    toks = [vocab.get(p, vocab.get(p.rstrip("ːh"), vocab["<unk>"])) for p in ipa]
    spans = ctc_align(lp, toks, blank)
    T = lp.shape[0]
    starts = [a for a, _ in spans]
    segs = []
    if starts[0] > 0:
        segs.append(Seg("sil", 0.0, starts[0] * FRAME_S))
    for i, p in enumerate(ipa):
        t0 = starts[i] * FRAME_S
        t1 = starts[i + 1] * FRAME_S if i + 1 < len(ipa) else (spans[-1][1] + 1) * FRAME_S
        segs.append(Seg(p, t0, t1))
    dur = len(x) / sr
    if segs[-1].t1 < dur - 1e-3:
        segs.append(Seg("sil", segs[-1].t1, dur))
    return segs


STOPS = {"k", "kː", "kʰ", "ɡ", "t", "tː", "tʰ", "d", "p", "pː", "pʰ", "b", "tɕ", "tɕh", "dʑ"}
FRICS = {"s", "ɕ", "h"}


def _runs(mask: np.ndarray):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def refine(segs: list[Seg], x: np.ndarray, sr: int, prof, search_s: float = 0.15) -> list[Seg]:
    """CTC 경계를 음향 표지에 맞춘다 (§52.487). CTC 는 음소의 시작 봉우리만 뾰족하게 잡고 길이를 모르며, 봉우리가 100 ms 넘게 앞서기도
    한다 (038 의 ㅊ: CTC 0.44 s, 실제 0.60 s). 표지 (5 ms, `segment.frame_features`):
      발화 = 정점 −25 dB 위 또는 유성 (시작) · −28 dB 위 (끝), 마찰 = 무성 · 고역 우세 (≥ 20 ms),
      폐쇄 = 국소 골 (±60 ms 중앙값보다 10 dB 이상 낮고 정점 −15 dB 밑, ≥ 15 ms), 개방 = 고역/저역 비가 15 ms 안에 +8 dB 뛰는 틀.
    마찰음 → 가장 가까운 마찰 구간, 파열·파찰음 → 가장 가까운 폐쇄 (+ 뒤따르는 무성 기식), 폐쇄가 없으면 가장 가까운 개방 표지 (10 ms 짜리).
    고정되지 않은 경계는 이웃 고정 경계 사이에서 CTC 시각의 비율을 지킨다."""
    from .segment import frame_features, FRICATIVE_HL_DB
    db, hl, voi, hop = frame_features(np.asarray(x, np.float64), sr, prof)
    dt = hop / sr
    n = len(db)
    t = (np.arange(n) + 0.5) * dt
    on = np.flatnonzero((db > -25.0) | voi)
    off = np.flatnonzero(db > -28.0)
    t_on, t_off = t[on[0]] - dt / 2, t[off[-1]] + dt / 2
    act = (t >= t_on) & (t <= t_off)
    fric = [(t[a] - dt / 2, t[b - 1] + dt / 2) for a, b in _runs(act & ~voi & (hl > FRICATIVE_HL_DB)) if b - a >= 4]
    w = max(1, int(0.06 / dt))
    pad = np.pad(db, w, mode="edge")
    dloc = db - np.array([np.median(pad[i:i + 2 * w + 1]) for i in range(n)])
    clos = [(t[a] - dt / 2, t[b - 1] + dt / 2) for a, b in _runs(act & (dloc < -10.0) & (db < -15.0)) if b - a >= 3]
    k3 = max(1, int(0.015 / dt))
    rel = [t[i] for i in range(k3, n) if act[i] and hl[i] - hl[i - k3] > 8.0 and (i == k3 or hl[i - 1] - hl[i - 1 - k3] <= 8.0)]
    ph = [s for s in segs if s.label != "sil"]
    ctc = [s.t0 for s in ph] + [ph[-1].t1]
    b = list(ctc)
    b[0], b[-1] = t_on, t_off
    fixed = {0, len(b) - 1}
    used = set()
    for k, s in enumerate(ph):
        c = s.t0
        if s.label in FRICS and fric:
            r = min(fric, key=lambda r: abs(r[0] - c))
            if abs(r[0] - c) < search_s and r not in used:
                b[k], b[k + 1] = r[0], r[1]; fixed |= {k, k + 1}; used.add(r)
        elif s.label in STOPS:
            cand = [r for r in clos if r not in used and abs(r[0] - c) < search_s]
            if cand:
                r = min(cand, key=lambda r: abs(r[0] - c))
                j = int(r[1] / dt)
                while j < n and not voi[j] and act[j]:
                    j += 1
                b[k], b[k + 1] = r[0], max(r[1], (j + 0.5) * dt); fixed |= {k, k + 1}; used.add(r)
            elif rel:
                r = min(rel, key=lambda r: abs(r - c))
                if abs(r - c) < search_s:
                    b[k], b[k + 1] = r - 0.005, r + 0.010; fixed |= {k, k + 1}
    fx = sorted(fixed)
    for lo, hi in zip(fx[:-1], fx[1:]):
        if hi - lo < 2:
            continue
        a0, a1, c0, c1 = b[lo], b[hi], ctc[lo], ctc[hi]
        for k in range(lo + 1, hi):
            u = (ctc[k] - c0) / max(c1 - c0, 1e-6)
            b[k] = a0 + u * (a1 - a0)
    for k in range(1, len(b)):
        b[k] = max(b[k], b[k - 1] + 0.01)
    out = [Seg("sil", 0.0, b[0])] if b[0] > 0 else []
    out += [Seg(s.label, b[k], b[k + 1]) for k, s in enumerate(ph)]
    dur = len(x) / sr
    if b[-1] < dur:
        out.append(Seg("sil", b[-1], dur))
    return out


KIND_OF = {**{p: "stop" for p in STOPS}, "s": "fric", "ɕ": "fric", "h": "fric", "n": "nasal", "m": "nasal", "ŋ": "nasal",
           "ɾ": "liquid", "l": "liquid", "j": "glide", "w": "glide", "sil": "sil"}
MIN_MS = {"vowel": 15.0, "glide": 5.0, "stop": 10.0, "fric": 20.0, "nasal": 10.0, "liquid": 5.0, "sil": 5.0}


def _emissions(db, hl, voi, dloc):
    """부류별 틀 점수 (로그 가능도의 대리) — 유성 V, 음량 db (정점 대비), 고역/저역 비 hl, 국소 골 깊이 dloc (±60 ms 중앙값 대비)."""
    V = voi.astype(float)
    c = lambda x: np.clip(x, -1.0, 1.0)
    e = {
        "vowel": 1.2 * V + c((db + 15.0) / 8.0) + c((-hl - 12.0) / 8.0),
        "glide": 1.2 * V + c((db + 15.0) / 8.0) + 0.5 * c((-hl - 12.0) / 8.0),
        "nasal": 1.2 * V + c((-dloc - 2.0) / 4.0) - 2.0 * np.clip((-dloc - 12.0) / 4.0, 0.0, 1.0) + c((-hl - 18.0) / 6.0),
        "liquid": 1.2 * V + 0.5 * c((-dloc - 1.0) / 4.0),
        "fric": 1.2 * (1.0 - V) + c(hl / 6.0) + c((db + 30.0) / 10.0),
        "stop": 1.5 * c((-dloc - 8.0) / 5.0) + 0.8 * c((hl + 10.0) / 10.0) + 0.5 * (1.0 - V),
        "sil": c((-db - 28.0) / 6.0) * 2.0,
    }
    return e


def hybrid_align(x: np.ndarray, sr: int, phones, prof, ctc_w: float = 0.4) -> list[Seg]:
    """음소 부류별 음향 점수 + CTC 음소 확률의 분절 비터비 (§52.487). phones: `korean.Phone` 목록. 앞뒤에 쉼을 둔다.

    CTC 봉우리만으로는 경계가 100 ms 넘게 앞선다 (038 의 ㅊ: CTC 0.44 s, 실제 0.58 s). 틀 5 ms, 순서 고정, 부류별 최소 길이."""
    from .segment import frame_features
    db, hl, voi, hop = frame_features(np.asarray(x, np.float64), sr, prof)
    dt = hop / sr
    T = len(db)
    w = max(1, int(0.06 / dt))
    pad = np.pad(db, w, mode="edge")
    med = np.array([np.median(pad[i:i + 2 * w + 1]) for i in range(T)])
    dloc = db - med
    E = _emissions(db, hl, voi, dloc)
    _, _, vocab = _model()
    lp = log_probs(x, sr)                                           # (T20, V) 20 ms
    t20 = np.minimum((np.arange(T) * dt / FRAME_S).astype(int), lp.shape[0] - 1)
    labels = ["sil"] + [p.ipa for p in phones] + ["sil"]
    kinds = ["sil"] + [("vowel" if p.kind == "vowel" else KIND_OF.get(p.ipa, "vowel")) for p in phones] + ["sil"]
    P = len(labels)
    S = np.zeros((P, T))
    for k, (lab, kd) in enumerate(zip(labels, kinds)):
        S[k] = E[kd]
        if lab != "sil":
            tok = vocab.get(lab, vocab.get(lab.rstrip("ːh"), vocab["<unk>"]))
            S[k] += ctc_w * np.maximum(lp[t20, tok], -20.0) / 5.0
    mins = [max(1, int(MIN_MS[kd] / 1000.0 / dt)) for kd in kinds]
    cum = np.concatenate([np.zeros((P, 1)), np.cumsum(S, 1)], 1)
    NEG = -1e18
    D = np.full((P, T + 1), NEG)                                    # D[k, t] = 음소 0..k 가 틀 0..t−1 을 덮는 최선 (k 가 t 에서 끝남)
    B = np.zeros((P, T + 1), dtype=np.int64)
    for t in range(mins[0], T + 1):
        D[0, t] = cum[0, t]
    for k in range(1, P):
        best, arg = NEG, 0
        for t in range(1, T + 1):
            s = t - mins[k]                                        # 음소 k 가 [s', t) — s' ≤ t − min
            if s >= 1 and D[k - 1, s] - cum[k, s] > best:
                best, arg = D[k - 1, s] - cum[k, s], s
            if best > NEG / 2:
                D[k, t] = best + cum[k, t]
                B[k, t] = arg
    bounds = [T]
    t = T
    for k in range(P - 1, 0, -1):
        t = B[k, t]
        bounds.append(t)
    bounds = bounds[::-1]                                           # 음소 1..P−1 의 시작
    starts = [0] + bounds[:-1]
    ends = bounds[:-1] + [T]
    # 위 역추적: bounds[i] 는 음소 i+1 의 시작 — 다시 짠다
    st = [0] * P
    t = T
    for k in range(P - 1, 0, -1):
        st[k] = B[k, t]
        t = st[k]
    en = st[1:] + [T]
    dur = len(x) / sr
    out = []
    for k in range(P):
        t0, t1 = st[k] * dt, min(en[k] * dt, dur)
        if k == P - 1:
            t1 = dur
        if t1 > t0:
            out.append(Seg(labels[k], t0, t1))
    return out
