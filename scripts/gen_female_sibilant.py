"""여성 화자의 치찰음 페이드 인/아웃 음원 생성.

치찰음 음량은 협착부 유량(부피속도)을 따라간다. 이 스크립트는 유량 포락선을
fade_in/fade_out (초) 으로 지정해 마찰음이 부드럽게 들고 나는 소리를 만든다.
같은 소리를 '게이트(상수)' 로도 내서 차이를 A/B 로 비교할 수 있게 한다.

    PYTHONPATH=src python scripts/gen_female_sibilant.py --out out
"""
from __future__ import annotations

import argparse
import os

from formant_ml.config import Config
from formant_ml.score import render
from formant_ml.utils import save_wav
from formant_ml.voice import VoiceProfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Config()
    sr = cfg.audio.sample_rate
    here = os.path.dirname(os.path.abspath(__file__))
    prof = VoiceProfile.load(os.path.join(here, "..", "profiles", "female_ko.json"))

    # 파일 간 상대 음량을 살리기 위해 RMS 정규화(피크 정규화는 페이드를 지운다).
    def W(name: str, score: dict, rms: float = 0.06) -> None:
        y = render({"seed": 5, **score}, prof, cfg)
        save_wav(os.path.join(args.out, name), y, sr, target_rms=rms)
        print(f"  {name}  ({y.shape[-1] / sr:.2f}s)")

    print(f"여성 치찰음 페이드 음원 -> {args.out}/")

    # 1) '스' 게이트(상수) vs 느린 페이드 — 같은 소리, 포락선만 다르다.
    W("f01_s_gate.wav",
      {"timeline": [{"type": "fricative", "phone": "s", "dur": 0.9,
                     "fade_in": 0.0, "fade_out": 0.0}]})
    W("f02_s_fade_slow.wav",
      {"timeline": [{"type": "fricative", "phone": "s", "dur": 0.9,
                     "fade_in": 0.30, "fade_out": 0.32}]})

    # 2) 유량이 두 번 부풀었다 꺼지는 마찰음 (flow 곡선 직접 지정).
    W("f03_s_flow_pulses.wav",
      {"timeline": [{"type": "fricative", "phone": "s", "dur": 1.2,
                     "flow": [[0, 0.0], [0.15, 1.0], [0.42, 0.15],
                              [0.62, 1.0], [0.9, 0.1], [1.0, 0.0]]}]})

    # 3) 된소리 'ㅆ' — 기류가 확 붙고(빠른 fade in) 여운을 남기며 빠진다.
    W("f04_ss_hard_onset.wav",
      {"timeline": [{"type": "fricative", "phone": "ss", "dur": 0.7,
                     "fade_in": 0.012, "fade_out": 0.28}]})

    # 4) 후치경 'ㅅ+ㅣ' 마찰음의 부드러운 페이드.
    W("f05_sh_fade.wav",
      {"timeline": [{"type": "fricative", "phone": "sh", "dur": 0.8,
                     "fade_in": 0.22, "fade_out": 0.24}]})

    # 5) 음절 '사' — 마찰음이 유량과 함께 부드럽게 들어와 모음으로 넘어간다.
    W("f06_syllable_sa_soft.wav",
      {"timeline": [{"type": "syllable", "onset": "s", "vowel": "a",
                     "dur": 0.6, "f0": [232, 198], "fade_in": 0.06}]})

    # 6) 짧은 문장풍: 스…사…시 (페이드가 발화 흐름에서 어떻게 들리는지).
    W("f07_line_seu_sa_si.wav",
      {"timeline": [
          {"type": "fricative", "phone": "s", "dur": 0.5,
           "fade_in": 0.14, "fade_out": 0.16},
          {"type": "silence", "dur": 0.12},
          {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.5,
           "f0": [230, 200], "fade_in": 0.05},
          {"type": "silence", "dur": 0.1},
          {"type": "syllable", "onset": "sh", "vowel": "i", "dur": 0.5,
           "f0": [236, 205], "fade_in": 0.04},
      ]})

    print("완료.")


if __name__ == "__main__":
    main()
