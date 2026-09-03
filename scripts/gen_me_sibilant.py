"""업로드 녹음에서 추출한 화자 목소리(profiles/me.json)로 치찰음을 재구성.

반영한 것: 실측 목소리(F0/포먼트/Rd/tilt/치찰음), 시간가변 치찰음 파라미터
(무게중심이 협착->모음 이동에 따라 하강), 구강 공명(noise_back_leak), 실측에
맞춘 유량 페이드/지속시간, 그리고 치찰음을 풀 때 남은 성문하압으로 눌려 나오는
'으'(pressed, 과하지 않게).

    PYTHONPATH=src python scripts/gen_me_sibilant.py --out out
"""
from __future__ import annotations

import argparse
import os

from formant_ml.config import Config
from formant_ml.score import render
from formant_ml.utils import save_wav
from formant_ml.voice import VoiceProfile

# 협착->모음 이동에 따른 치찰음 무게중심 하강(실측 6700->3900). 앞니 공명·앞공동
# 극을 함께 내린다. 마지막 값은 모음 구간이라 청감엔 영향 없다(노이즈가 이미 꺼짐).
SIB_GLIDE = {
    "sib_teeth_f": [[0, 10000], [0.18, 9200], [0.30, 6800], [1, 6200]],
    "sib_pole_f":  [[0, 5300], [0.30, 3600], [1, 3400]],
    # 모음으로 갈수록 구강(성도 전체) 결합을 키워 vowel 포먼트가 얹히게 한다
    "noise_back_leak": [[0, 0.18], [0.30, 0.45], [1, 0.5]],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = Config()
    sr = cfg.audio.sample_rate
    here = os.path.dirname(os.path.abspath(__file__))
    prof = VoiceProfile.load(os.path.join(here, "..", "profiles", "me.json"))

    def W(name, score, rms=0.05):
        y = render({"seed": 5, "smooth_frames": 2, **score}, prof, cfg)
        save_wav(os.path.join(args.out, name), y, sr, target_rms=rms)
        print(f"  {name}  ({y.shape[-1] / sr:.2f}s)")

    print(f"내 목소리 치찰음 재구성 -> {args.out}/")

    # 1) '사' — 실측 지속시간(마찰 0.14 + 모음 0.44), 시간가변 치찰음(무게중심 하강)
    sa = {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.58,
          "onset_s": 0.14, "fade_in": 0.03,
          "f0": [[0, 129], [1, 123]], **SIB_GLIDE}
    W("m01_sa.wav", {"timeline": [sa]})
    W("m02_sa_sa.wav", {"timeline": [
        sa, {"type": "silence", "dur": 0.3}, dict(sa)]})

    # 2) 재구성 본편: 길게 끈 치찰음 -> 눌린 '으' -> '아'
    release = [
        {"type": "fricative", "phone": "s", "dur": 0.55,
         "fade_in": 0.16, "fade_out": 0.05, "level_db": -8,
         "sib_teeth_f": [[0, 8600], [0.4, 10000], [0.8, 9600], [1, 6800]],
         "sib_pole_f": [[0, 4600], [0.5, 5300], [1, 3800]],
         "noise_back_leak": [[0, 0.15], [1, 0.4]]},
        {"type": "glide", "vowels": ["eu", "a"], "dur": 0.5,
         "pressure": [[0, 1.25], [0.25, 1.06], [1, 0.8]], "adduction": 0.9,
         "f0": [[0, 138], [0.3, 131], [1, 123]],
         "tilt": [[0, prof.tilt + 1], [1, prof.tilt]]},
    ]
    W("m03_s_to_eu_a.wav", {"timeline": release})

    # 3) 녹음 흐름 재현: 사, 사, 길게 s, 길게 s (실측 지속시간/궤적)
    long_s = {"type": "fricative", "phone": "s", "dur": 0.7,
              "fade_in": 0.28, "fade_out": 0.2, "level_db": -6,
              "sib_teeth_f": [[0, 8200], [0.3, 9500], [0.6, 9800], [1, 8600]],
              "sib_pole_f": [[0, 4400], [0.5, 5300], [1, 4600]],
              "noise_back_leak": [[0, 0.15], [1, 0.35]]}
    W("m04_like_recording.wav", {"timeline": [
        sa, {"type": "silence", "dur": 0.32}, dict(sa),
        {"type": "silence", "dur": 0.5}, long_s,
        {"type": "silence", "dur": 0.5},
        {**long_s, "dur": 1.1}]})

    print("완료.")


if __name__ == "__main__":
    main()
