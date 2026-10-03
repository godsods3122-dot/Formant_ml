"""파열 검출은 **전체 파일 맥락**에서 한다 (MEASUREMENTS §52.290).

사용자가 목표 파일 맨 앞의 구강 클릭을 짚었다 — *"이 때 쓰라고 이벤트 파라미터를 준 건데"*.
그 클릭은 우리가 오늘 내내 적합해 온 구간(0.451~1.293 s) 안에 있는데(0.463 s) **검출이 0 개**였다.
검출기는 `BURST_BACK_MS`(15 ms) 만큼 되돌아봐 기준선을 잡는데, 구간 앞이 잘려 있으면 기준선이 올라가
상승량이 줄어든다(실측 20 dB → 11.3 dB, 문턱 12 아래).
"""
import numpy as np
import pytest

from formant_ml.engine import analyze as A

SR, HOP = 48000, 48


def _click_signal(dur=1.0, at=0.5, sr=SR):
    """조용한 바닥 위에 2~8 kHz 로 넓은 짧은 파열 하나."""
    rng = np.random.default_rng(0)
    n = int(dur * sr)
    x = rng.standard_normal(n) * 1e-4
    i = int(at * sr)
    w = int(0.003 * sr)
    burst = rng.standard_normal(w) * np.hanning(w) * 0.2
    x[i:i + w] += burst
    return x


def test_full_context_finds_a_burst_near_the_segment_start():
    """구간 시작에서 12 ms 떨어진 파열 — 잘린 구간만 보면 놓치고, 전체 맥락이면 찾는다."""
    x = _click_signal(dur=1.0, at=0.5)
    cut = 0.488                                  # 파열(0.5 s)이 시작에서 12 ms
    seg = x[int(cut * SR):int(0.9 * SR)]
    n_seg = len(seg) // HOP

    only_seg = A.find_bursts(seg, SR, HOP)
    b_all = A.find_bursts(x, SR, HOP)
    off = int(round(cut * SR / HOP))
    with_full = b_all - off
    with_full = with_full[(with_full >= 0) & (with_full < n_seg)]

    assert len(b_all) >= 1, "전체 파일에서는 파열이 보여야 한다"
    assert len(with_full) >= 1, "전체 맥락으로 옮기면 구간 안에서 찾아야 한다"
    # 자른 구간만 보면 놓치거나 시각이 어긋난다 — 그것이 이 고침의 이유다
    assert len(only_seg) <= len(with_full)


def test_analyze_uses_full_when_given(monkeypatch):
    """`analyze(..., full=...)` 이면 파열 검출도 그 맥락을 써야 한다."""
    from pathlib import Path
    src = Path("src/formant_ml/engine/analyze.py").read_text(encoding="utf-8")
    body = src[src.index("    if BURST_TRACK:"):]
    body = body[:body.index("if pulses is not None")]
    assert "find_bursts(full, sr, hop)" in body, "전체 파일로 찾아야 한다"
    assert "t0" in body and "off" in body, "구간 좌표로 되돌려야 한다"
    assert "find_bursts(y, sr, hop)" in body, "full 이 없으면 예전처럼 구간으로 (되돌림 경로)"


def test_offset_maps_into_segment_coordinates():
    """전체 좌표에서 찾은 파열이 **구간 프레임 번호**로 옮겨져야 한다."""
    x = _click_signal(dur=1.0, at=0.5)
    cut = 0.3
    b_all = A.find_bursts(x, SR, HOP)
    assert len(b_all) >= 1
    off = int(round(cut * SR / HOP))
    mapped = b_all - off
    seg_n = len(x[int(cut * SR):]) // HOP
    inside = mapped[(mapped >= 0) & (mapped < seg_n)]
    assert len(inside) >= 1
    got = float(inside[0]) * HOP / SR + cut
    assert abs(got - 0.5) < 0.01, got
