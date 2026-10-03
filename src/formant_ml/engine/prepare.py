"""적합 준비 단계 — 녹음 구간에서 관측 궤적을 만든다 (MEASUREMENTS §52.434).

`scripts/copyfit.py` 안에 인라인으로 있던 "분석 → 앞공동 관측 → (복원) → 파열 궤적 잇기 → 폐쇄 관측" 을 여기로 옮겼다.
옮긴 까닭은 두 가지다.

1. **시간 원점을 한 곳에서 정한다.** 제어열·파열·펄스는 구간(`t0`) 원점이다. 예전에는 호출하는 쪽이 `y`(파일)와 `seg`(구간)를
   골라 넘겼고, 두 관측(`close_stops`, `observe_front_len`)에 파일 신호가 들어가 `--from` 만큼 어긋났다(§52.430·431).
   `Segment` 가 구간 신호와 원점을 함께 들고 다니므로 원점이 다른 신호를 넘길 길이 없다.
2. **시험이 적합과 같은 경로를 탄다.** `tests/engine/test_prepare_contract.py` 가 이 함수들을 알려진 합성 신호로 부른다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """적합 구간. `audio` 는 **구간 신호**(원점 = `t0`), `full` 은 파일 신호. 관측 함수에는 `audio` 만 넘긴다."""
    full: np.ndarray
    sr: int
    t0: float
    t1: float

    @property
    def audio(self) -> np.ndarray:
        return self.full[int(round(self.t0 * self.sr)):int(round(self.t1 * self.sr))]

    @classmethod
    def cut(cls, full: np.ndarray, sr: int, t0: float, t1: float | None) -> "Segment":
        full = np.asarray(full, dtype=np.float64)
        return cls(full, int(sr), float(t0), float(t1) if t1 is not None else len(full) / sr)


def observe(segment: Segment, prof, hop: int, front_obs: bool = False, log=print):
    """분석 + (선택) 앞공동 관측 → 구간 원점의 `ControlTrack`."""
    from . import analyze as an
    seg = segment.audio
    track = an.analyze(seg, segment.sr, prof, hop, t0=segment.t0, full=segment.full)
    if front_obs:
        nf = an.observe_front_len(track, seg, segment.sr, hop)
        fl = track.values[:, an.INDEX["front_len"]]
        log(f"  앞공동 길이 관측: {nf} 프레임에서 관측, 궤적 {fl.min():.2f}~{fl.max():.2f} cm (중앙 {np.median(fl):.2f})")
    return track


def burst_observations(track, segment: Segment, hop: int, burst_smooth_ms: float | None = None,
                       stop_close: bool = False, log=print) -> dict:
    """파열 기반 관측 — `--init` 복원 **뒤에** 부른다 (§52.397). 바꾼 프레임 수를 돌려준다."""
    from . import analyze as an
    out = {"smoothed": 0, "closed": 0}
    if burst_smooth_ms:
        out["smoothed"] = an.smooth_over_bursts(track, float(burst_smooth_ms))
        log(f"  파열 궤적 이어 붙이기: ±{float(burst_smooth_ms):g} ms, {out['smoothed']} 프레임 "
            f"(파열 {len(getattr(track, 'bursts', ()))} 곳)")
    if stop_close:
        out["closed"] = an.close_stops(track, segment.audio, segment.sr, hop)
        log(f"  파열 폐쇄 관측: {out['closed']} 프레임을 a_c {an.STOP_CLOSE_AC:g} cm² 로 닫았다")
    return out
