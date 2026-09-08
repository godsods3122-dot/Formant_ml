"""녹음을 조음 유형별 구간으로 가른다 — 코퍼스를 적합에 먹이기 위한 앞단.

왜 필요한가
-----------
복사합성은 한 번에 100~300 ms 를 다룬다. 45 분짜리 코퍼스를 그 단위로 쪼개려면
"여기는 마찰, 여기는 모음" 을 자동으로 말해 주는 것이 있어야 한다. 손으로 고른 구간
넷으로 내린 결론이 일반적인지 확인하는 것이 지금 단계의 과제이므로
(docs/HANDOFF.md §5 "가장 큰 약점: 표본이 파일 하나다"), 이 분할기가 그 관문이다.

판정 규칙은 `analyze.py` 와 **같은 것**을 쓴다 — 유성은 피치가 잡혔는가로, 마찰은
고역/저역비로. 두 곳이 다르면 적합의 초기값과 구간 라벨이 어긋난다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analyze import FRICATIVE_HL_DB
from .profile import SpeakerProfile

# 활성(=무음이 아님) 판정. 파일 정점 대비 dB.
ACTIVE_DB = -40.0


@dataclass
class Segment:
    t0: float
    t1: float
    kind: str            # "vowel" | "fricative" | "mixed" | "nasal"
    file: str = ""

    @property
    def dur(self) -> float:
        return self.t1 - self.t0

    def __str__(self) -> str:
        return f"{self.kind:9s} {self.t0:7.3f}~{self.t1:7.3f} ({1000*self.dur:5.1f} ms)"


def frame_features(y: np.ndarray, sr: int, prof: SpeakerProfile,
                   hop_ms: float = 5.0, win_ms: float = 25.0):
    """(rms_db, 고역저역비 dB, 유성 마스크, hop 샘플수).

    F0 는 Praat 으로 **파일 전체**에서 잰다. 조각에 걸면 NaN 이 돌아온다
    (analyze.py 와 같은 이유).
    """
    import parselmouth
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = int(sr * win_ms / 1000.0) // 2 * 2
    n = max(0, (len(y) - win) // hop)
    if n == 0:
        return (np.zeros(0),) * 3 + (hop,)
    w = np.hanning(win)
    fr = np.fft.rfftfreq(win, 1.0 / sr)
    lo_m = (fr >= 100) & (fr < 1000)
    hi_m = (fr >= 3000) & (fr < 16000)
    seg = np.stack([y[i * hop:i * hop + win] for i in range(n)])
    rms = np.sqrt((seg ** 2).mean(1) + 1e-20)
    P = np.abs(np.fft.rfft(seg * w, axis=-1)) ** 2 + 1e-20
    hl = 10.0 * np.log10(P[:, hi_m].sum(1) / P[:, lo_m].sum(1))
    snd = parselmouth.Sound(y.astype(np.float64), sr)
    pt = snd.to_pitch(time_step=hop / sr, pitch_floor=max(60.0, prof.f0_lo * 0.6),
                      pitch_ceiling=prof.f0_hi * 1.3)
    t = (np.arange(n) * hop + win / 2) / sr
    voi = np.array([(lambda v: v is not None and v == v)(pt.get_value_at_time(x))
                    for x in t], dtype=bool)
    return 20.0 * np.log10(rms / (rms.max() + 1e-20) + 1e-12), hl, voi, hop


def _runs(mask: np.ndarray, hop: int, sr: int, min_ms: float):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) * hop / sr * 1000.0 >= min_ms:
                out.append((i * hop / sr, j * hop / sr))
            i = j
        else:
            i += 1
    return out


def segments(y: np.ndarray, sr: int, prof: SpeakerProfile,
             min_fric_ms: float = 45.0, min_vowel_ms: float = 90.0,
             pad_ms: float = 30.0, max_ms: float = 260.0,
             file: str = "") -> list[Segment]:
    """녹음 -> 적합에 먹일 구간 목록.

    * `fricative` — 마찰 구간에 앞뒤로 `pad_ms` 를 붙인다. 마찰만 잘라 내면 성문
      제스처의 선행/후행(프로파일 §sibilant_voicing: 앞 35 ms, 뒤 55 ms)이 창 밖으로
      나가고, 그러면 적합이 그 전이를 못 본다.
    * `vowel` — 유성 정상부. 길면 앞에서 `max_ms` 만 쓴다.
    """
    db, hl, voi, hop = frame_features(y, sr, prof)
    if len(db) == 0:
        return []
    active = db > ACTIVE_DB
    fric = active & (~voi) & (hl > FRICATIVE_HL_DB)
    vowel = active & voi
    out: list[Segment] = []
    pad = pad_ms / 1000.0
    dur = len(y) / sr
    for a, b in _runs(fric, hop, sr, min_fric_ms):
        t0, t1 = max(0.0, a - pad), min(dur, b + pad)
        out.append(Segment(t0, min(t1, t0 + max_ms / 1000.0), "fricative", file))
    for a, b in _runs(vowel, hop, sr, min_vowel_ms):
        out.append(Segment(a, min(b, a + max_ms / 1000.0), "vowel", file))
    out.sort(key=lambda s: s.t0)
    return out


def fricative_mask(y: np.ndarray, sr: int, prof: SpeakerProfile,
                   hop: int) -> np.ndarray:
    """적합 프레임 격자 위의 마찰 마스크. `analyze()` 와 같은 규칙."""
    db, hl, voi, h = frame_features(y, sr, prof, hop_ms=1000.0 * hop / sr)
    n = len(db)
    if n == 0:
        return np.zeros(0, dtype=bool)
    return (db > ACTIVE_DB) & (~voi) & (hl > FRICATIVE_HL_DB)
