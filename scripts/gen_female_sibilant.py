"""여성 화자의 치찰음(실측 지문) + 유량 페이드 + '으아' 압력 릴리스 음원 생성.

치찰음 지문은 실제 녹음("사" 2회 + 길게 끈 치찰음 2회)에서 극-영점 모델을
경사하강으로 적합해(analysis/sibilant.fit_sibilant, rmse 1.05 dB) profiles/
female_ko.json 에 넣은 값이다. 앞니 공명 ~9.9 kHz 가 10 kHz 피크를, 낮은
floor(-27.6 dB)가 저역을 비워 실측 /s/ 의 날카로운 고역을 재현한다.

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

    def W(name: str, score: dict, rms: float = 0.06) -> None:
        y = render({"seed": 5, "smooth_frames": 3, **score}, prof, cfg)
        save_wav(os.path.join(args.out, name), y, sr, target_rms=rms)
        print(f"  {name}  ({y.shape[-1] / sr:.2f}s)")

    print(f"여성 치찰음/으아 음원 -> {args.out}/")

    # ---- 재구성 본편: 길게 끈 치찰음 -> 압력 실린 '으' -> '아' -----------------
    reconstruct = [
        {"type": "fricative", "phone": "s", "dur": 0.9,
         "fade_in": 0.24, "fade_out": 0.03, "level_db": 4},
        {"type": "glide", "vowels": ["eu", "a"], "dur": 0.62,
         "pressure": [[0, 1.5], [0.2, 1.25], [1, 0.82]], "adduction": 0.95,
         "f0": [[0, 203], [0.28, 195], [1, 182]], "tilt": [[0, 3.5], [1, 0.0]]},
    ]
    W("g01_s_to_eu_a.wav", {"timeline": reconstruct})

    # 압력을 더 세게(더 눌린 '으') / 더 여리게(기식 섞인 '으') A/B
    W("g02_s_to_eu_a_hard.wav", {"timeline": [
        {"type": "fricative", "phone": "s", "dur": 0.8,
         "fade_in": 0.18, "fade_out": 0.03},
        {"type": "glide", "vowels": ["eu", "a"], "dur": 0.6,
         "pressure": [[0, 2.1], [0.18, 1.7], [1, 0.9]], "adduction": 0.98,
         "f0": [[0, 214], [0.3, 205], [1, 188]], "tilt": [[0, 5], [1, 0.5]]}]})
    W("g03_s_to_eu_a_soft.wav", {"timeline": [
        {"type": "fricative", "phone": "s", "dur": 0.8,
         "fade_in": 0.24, "fade_out": 0.05},
        {"type": "glide", "vowels": ["eu", "a"], "dur": 0.66,
         "pressure": [[0, 1.35], [0.25, 1.2], [1, 0.8]], "adduction": 0.85,
         "f0": [[0, 198], [0.3, 192], [1, 180]], "tilt": [[0, 1.5], [1, -1]]}]})

    # ---- 실측 녹음 흐름 재현: 사, 사, 길게 s, 길게 s (직접 비교용) --------------
    W("g04_like_recording.wav", {"timeline": [
        {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.55,
         "f0": [210, 185], "fade_in": 0.04},
        {"type": "silence", "dur": 0.35},
        {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.55,
         "f0": [210, 185], "fade_in": 0.04},
        {"type": "silence", "dur": 0.5},
        {"type": "fricative", "phone": "s", "dur": 0.9,
         "fade_in": 0.12, "fade_out": 0.12},
        {"type": "silence", "dur": 0.5},
        {"type": "fricative", "phone": "s", "dur": 1.1,
         "fade_in": 0.15, "fade_out": 0.15},
    ]})

    # ---- 페이드 구조 A/B (실측 지문이 적용된 치찰음으로) -----------------------
    W("g05_s_gate.wav", {"timeline": [
        {"type": "fricative", "phone": "s", "dur": 0.9,
         "fade_in": 0.0, "fade_out": 0.0}]})
    W("g06_s_fade_slow.wav", {"timeline": [
        {"type": "fricative", "phone": "s", "dur": 0.9,
         "fade_in": 0.3, "fade_out": 0.32}]})
    W("g07_s_flow_pulses.wav", {"timeline": [
        {"type": "fricative", "phone": "s", "dur": 1.2,
         "flow": [[0, 0.0], [0.15, 1.0], [0.42, 0.15],
                  [0.62, 1.0], [0.9, 0.1], [1.0, 0.0]]}]})

    print("완료.")


if __name__ == "__main__":
    main()
