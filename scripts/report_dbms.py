#!/usr/bin/env python3
"""적합 트랙의 **dB/ms** — 소스 세기가 프레임마다 얼마나 페이드하는가.

    python scripts/report_dbms.py out/bench3/s34 out/bench3/s34_d0.03 ...

생리 기준은 **1.5 dB/ms** 다 (docs/MEASUREMENTS.md §21.3 / `fit.DB_RATE_KNEE_DB_MS`):
음소률 상단 16 Hz 에서 진폭 15 dB 짜리 제스처의 최대 기울기가 15·2π·0.016 이다.
그 위는 조음이 못 내는 속도다.

이 자는 **제어열만 본다** — 렌더를 안 하므로 시드에 안 흔들린다. 첨두 밀도(시드 의존
있음)와 첨도(크게 있음)와 달리 조건 비교에 바로 쓸 수 있다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from formant_ml.engine import noise as N                        # noqa: E402
from formant_ml.engine.control import INDEX                     # noqa: E402

KNEE = 1.5


def db_rate(x: np.ndarray) -> tuple[float, float]:
    d = np.abs(np.diff(20.0 * np.log10(np.maximum(x, 1e-12))))
    return float(np.median(d)), float(np.percentile(d, 95))


def env_proxy(v: np.ndarray) -> np.ndarray:
    """마찰 소스 포락 `env = drive x fric_gain`.

    성문 면적은 `physiology` 가 내는 값이라 트랙에 없다. 무성 마찰음은 성문이 크게
    벌어져 있어(§14.2) 여기서는 0.2 cm² 로 고정한다 — drive 의 **시간 변화**를
    보는 것이라 상수 배는 dB 차분에서 지워진다.
    """
    g = lambda k: torch.tensor(v[:, INDEX[k]], dtype=torch.float64)[None]
    ps, ac, fg = g("p_sub"), g("a_c"), g("fric_gain")
    u, _ = N.series_flow(ps, torch.full_like(ps, 0.2), ac)
    re, _, _ = N.reynolds(u, ac)
    drive = (((re ** 2 - N.RE_CRIT ** 2).clamp_min(0.0) / N.RE_REF ** 2) ** 1.5
             * (0.1 / ac.clamp_min(0.02)))
    return (drive * fg)[0].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    a = ap.parse_args()
    print(f"dB/ms — 생리 기준 {KNEE} dB/ms (16 Hz · 15 dB 제스처의 최대 기울기)")
    print(f"{'트랙':<20}{'env 중앙':>10}{'95분위':>9}{'>3dB/ms':>9}{'>6dB/ms':>9}"
          f"{'fric_gain':>11}{'drive':>8}{'tract_gain':>12}")
    for stem in a.stems:
        d = np.load(stem + "_track.npz", allow_pickle=True)
        v = d["values"]
        env = env_proxy(v)
        dd = np.abs(np.diff(20.0 * np.log10(np.maximum(env, 1e-12))))
        m, p95 = db_rate(env)
        print(f"{os.path.basename(stem):<20}{m:10.2f}{p95:9.2f}"
              f"{100 * (dd > 3).mean():8.0f}%{100 * (dd > 6).mean():8.0f}%"
              f"{db_rate(v[:, INDEX['fric_gain']])[0]:11.2f}"
              f"{db_rate(env / np.maximum(v[:, INDEX['fric_gain']], 1e-12))[0]:8.2f}"
              f"{db_rate(v[:, INDEX['tract_gain']])[0]:12.2f}")


if __name__ == "__main__":
    main()
