"""합성음에 **실측한 방**을 걸어 청취 비교를 공정하게 만든다.

    python scripts/add_room.py out/sib/s101_fit.wav --out out/sib/s101_fit_room.wav

이건 엔진이 아니다 — 목소리의 물리가 아니라 **녹음 경로**다
------------------------------------------------------------
학습 데이터로 뽑는 파라미터는 마른(anechoic) 소리여야 하므로 `engine/` 에는 절대
넣지 않는다. 이 도구는 **귀로 비교할 때만** 쓴다.

왜 필요한가 (docs/MEASUREMENTS.md §19)
--------------------------------------
목표 녹음은 방에서 잡은 것이고 합성은 마르다. 방은 마찰음의 포락선을 **매끄럽게**
만든다. 그래서 같은 소스라도 마른 합성이 더 거칠게 들리고, 변조 지표로 재면
목표 대비 3~4 배로 나온다. 실제로 §12.2 에서 **목표가 순수 가우시안 대조군보다
매끄러웠다**(60~400 Hz 변조 13.6 % 대 17.8 %) — 마른 난류로는 불가능한 값이고,
그게 방이 있다는 증거다.

측정한 방 (`yang_00000101`, 오프셋 뒤 대역 에너지의 하강 기울기 중앙값, n=16~32):

    500-1k Hz  RT60 0.299 s      2-4k  0.294 s      8-12k  0.246 s
    1-2k       0.381 s           4-8k  0.299 s

고역이 짧은 것(공기 흡수)까지 보통의 작은 방 모양이다. 이 값들은 말소리 자체의
오프셋 감쇠도 섞여 있으므로 **상한**으로 읽어야 한다.

이 방을 걸면 마찰 구간의 변조 지수가 목표 대비 4.12 / 2.33 / 3.54 배에서
1.5 / 1.3 / 1.4 배로 내려간다 — 남아 있던 지글거림의 대부분이 방의 부재였다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

#: 실측 RT60 [s] 를 대역별로. 위 docstring 의 표 그대로.
RT60_BANDS = ((0.0, 700.0, 0.30), (700.0, 1500.0, 0.38), (1500.0, 3000.0, 0.29),
              (3000.0, 6000.0, 0.30), (6000.0, 24000.0, 0.25))
#: 직접음 대비 잔향의 섞임. 변조 지수를 1.0 근처로 맞추는 값 (§19).
DEFAULT_MIX = 0.45
#: 초기 지연 [ms]. 직접음과 첫 반사 사이. 이게 0 이면 잔향이 직접음을 흐린다.
PREDELAY_MS = 8.0


def impulse_response(fs: float, mix_seed: int = 7) -> np.ndarray:
    """대역별 RT60 을 갖는 지수 감쇠 잡음 IR. 에너지는 1 로 정규화한다.

    대역마다 감쇠가 다르므로 **대역별로 만들어 더한다** — 하나의 지수에 기울기만
    씌우면 고역이 길게 남아 "쉬" 하는 꼬리가 생긴다.
    """
    from scipy.signal import butter, sosfiltfilt
    rng = np.random.default_rng(mix_seed)
    n = int(1.2 * max(r for *_, r in RT60_BANDS) * fs)
    t = np.arange(n) / fs
    w = rng.standard_normal(n)
    ir = np.zeros(n)
    for lo, hi, rt in RT60_BANDS:
        hi = min(hi, fs / 2 * 0.99)
        if lo <= 0:
            sos = butter(4, hi / (fs / 2), btype="low", output="sos")
        else:
            sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
        ir += sosfiltfilt(sos, w) * np.exp(-6.907755 * t / rt)
    ir[:int(PREDELAY_MS * 1e-3 * fs)] = 0.0
    return ir / (np.sqrt((ir ** 2).sum()) + 1e-12)


def add_room(x: np.ndarray, fs: float, mix: float = DEFAULT_MIX) -> np.ndarray:
    ir = impulse_response(fs)
    wet = np.convolve(x, ir)[:len(x)]
    wet *= np.sqrt((x ** 2).mean()) / (np.sqrt((wet ** 2).mean()) + 1e-12)
    return (1.0 - mix) * x + mix * wet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mix", type=float, default=DEFAULT_MIX)
    a = ap.parse_args()
    y, sr = sf.read(a.wav)
    mono = y.ndim == 1
    x = y if mono else y.mean(1)
    z = add_room(np.asarray(x, float), float(sr), a.mix)
    out = a.out or (os.path.splitext(a.wav)[0] + "_room.wav")
    sf.write(out, z, sr)
    print(f"{out}  (RT60 대역별 0.25~0.38 s, mix {a.mix:.2f})")


if __name__ == "__main__":
    main()
