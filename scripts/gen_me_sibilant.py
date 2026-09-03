"""업로드 녹음에서 추출한 목소리(profiles/me.json) + 공기음향 난류로 재구성.

물리(문헌): 난류는 임계 레이놀즈수에서 켜지고(Stevens 1971), 소스 세기는 협착
압력강하 ½ρ(U/A)² 에 비례하며, 무게중심은 입자속도와 함께 오른다. 폐압이 성문·
구강 협착을 직렬로 지나 하나의 유량을 만들어 발성·기식·마찰음을 함께 구동한다.
전부 협착 면적 궤적 하나에서 유도된다(임의 페이드/치찰음 곡선이 아니라).

    PYTHONPATH=src python scripts/gen_me_sibilant.py --out out
"""
from __future__ import annotations

import argparse
import os

from formant_ml.config import Config
from formant_ml.score import render
from formant_ml.utils import save_wav
from formant_ml.voice import VoiceProfile

# 실측 /사/: 협착이 프리케이션 동안 서서히 열려 무게중심이 6500->3900 으로 내려간다.
# 협착 해제를 발성 개시와 **겹치게** 맞춘다. 예전엔 마찰음이 꺼진 뒤 발성까지
# 120 ms 무음이 생겨 '무음 + 급개시' = 폐쇄음으로 들렸다(/사/ 가 "스트라").
SA_AREA = [[0, 0.20], [0.06, 0.13], [0.24, 0.18], [0.30, 0.26],
           [0.34, 1.0], [0.44, 3.0], [1, 3.0]]


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

    print(f"내 목소리 · 공기음향 재구성 -> {args.out}/")

    # 1) '사' — 협착 면적에서 진폭·게이트·무게중심 하강이 전부 유도된다
    sa = {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.58,
          "onset_s": 0.14, "aero": True, "transition_s": 0.05,
          "constriction_area": SA_AREA, "voice_onset_s": 0.03,
          "aspiration": 1.0, "f0": [[0, 129], [1, 123]]}
    W("m01_sa.wav", {"timeline": [sa]})
    W("m02_sa_sa.wav", {"timeline": [sa, {"type": "silence", "dur": 0.3}, dict(sa)]})

    # 2) 재구성 본편: 길게 끈 치찰음 -> 눌린 '으' -> '아'
    W("m03_s_to_eu_a.wav", {"timeline": [
        {"type": "fricative", "phone": "s", "dur": 0.55, "aero": True,
         "constriction_area": [[0, 0.5], [0.12, 0.11], [0.5, 0.10], [0.8, 0.16],
                               [0.92, 0.6], [1, 2.5]],
         "glottal_area": 0.12, "level_db": -6},
        {"type": "glide", "vowels": ["eu", "a"], "dur": 0.5,
         "pressure": [[0, 1.25], [0.25, 1.06], [1, 0.8]], "adduction": 0.9,
         "f0": [[0, 138], [0.3, 131], [1, 123]],
         "tilt": [[0, prof.tilt + 1], [1, prof.tilt]]}]})

    # 3) 유성 마찰음 /z/ — 성대가 떨며 마찰음을 성문주기로 변조한다(소스-치찰음 결합).
    #    성문이 좁게 떨어 유량이 맥동 -> 마찰음이 그 주기로 변조(Jackson&Shadle 2000).
    W("m05_voiced_fricative_z.wav", {"timeline": [
        {"type": "fricative", "phone": "z", "dur": 0.6, "aero": True,
         "constriction_area": [[0, 0.4], [0.1, 0.12], [0.9, 0.12], [1, 0.5]],
         "glottal_area": 0.04, "harmonic_amp": 0.9, "noise_am": 0.9,
         "f0": [[0, 126], [1, 122]], "level_db": -8}]})

    # 4) 녹음 흐름 재현: 사, 사, 길게 s, 길게 s
    long_s = {"type": "fricative", "phone": "s", "dur": 0.7, "aero": True,
              "constriction_area": [[0, 0.6], [0.2, 0.11], [0.6, 0.10],
                                    [0.85, 0.14], [0.95, 0.5], [1, 2.0]],
              "glottal_area": 0.12, "level_db": -5}
    W("m04_like_recording.wav", {"timeline": [
        sa, {"type": "silence", "dur": 0.32}, dict(sa),
        {"type": "silence", "dur": 0.5}, long_s,
        {"type": "silence", "dur": 0.5}, {**long_s, "dur": 1.1}]})

    print("완료.")


if __name__ == "__main__":
    main()
