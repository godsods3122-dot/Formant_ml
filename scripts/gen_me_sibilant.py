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

# 협착 궤적을 손으로 그리지 않는다. 예전엔 7 개 꺾은점으로 페이드를 만들었는데,
# 그건 물리가 아니라 곡선 맞추기였고 호흡 압력과 서로 싸워서 정점이 가청 구간의
# 24 % 로 앞당겨졌다(실측 52~63 %). 이제 협착은 그냥 '좁게 유지하다 해제' 이고,
# 페이드 인/아웃은 전부 **호흡 구동압**에서 나온다(aeroacoustic.breath_drive).

#: 독립 마찰음에서 가청 /s/ 는 세그먼트의 약 48 % 다 — 나머지는 압력이 임계
#: 유속에 못 미쳐 소리가 안 나는 앞뒤 램프다(그게 페이드 인/아웃의 실체다).
#: 그래서 1.1 s 짜리 /s/ 를 들으려면 세그먼트를 2.3 s 로 잡는다.
AUDIBLE_FRAC = 0.48


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

    # 1) '사' — 호흡 압력이 페이드를, 협착 해제가 마찰음 종료를 만든다.
    #    onset_s 0.11 -> 가청 /s/ 160 ms, 정점 60 %, 상승 96 / 하강 64 ms
    #    (실측 134~139 ms, 52~63 %, 70~87 / 52~64 ms).
    sa = {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.58,
          "onset_s": 0.11, "aero": True, "transition_s": 0.12,
          "f0": [[0, 129], [1, 123]]}
    W("m01_sa.wav", {"timeline": [sa]})
    W("m02_sa_sa.wav", {"timeline": [sa, {"type": "silence", "dur": 0.3}, dict(sa)]})

    # 2) 재구성 본편: 길게 끈 치찰음 -> 짧은 '으' 전이 -> '아'
    #    '으' 는 별도 모음이 아니라 /s/ 자세(혀 높음)의 잔상이다. 로커스에서
    #    출발해 60 ms 안에 '아' 로 간다 — 길게 끌면 그게 활음(/j/)이라 "야" 가 된다.
    #    압력은 /s/ 내내 서서히 오르므로 페이드 인이 마찰음 전체에 걸친다.
    W("m03_s_to_eu_a.wav", {"timeline": [
        {"type": "syllable", "onset": "s", "vowel": "a", "dur": 1.05,
         "onset_s": 0.55, "aero": True, "transition_s": 0.12,
         "f0": [[0, 131], [1, 122]]}]})

    # 3) 유성 마찰음 /z/ — 성대가 떨며 마찰음을 성문주기로 변조한다(소스-치찰음 결합).
    #    성문이 좁게 떨어 유량이 맥동 -> 마찰음이 그 주기로 변조(Jackson&Shadle 2000).
    W("m05_voiced_fricative_z.wav", {"timeline": [
        {"type": "fricative", "phone": "z", "dur": 0.6, "aero": True,
         "constriction_area": [[0, 0.4], [0.1, 0.12], [0.9, 0.12], [1, 0.5]],
         "glottal_area": 0.04, "harmonic_amp": 0.9, "noise_am": 0.9,
         "f0": [[0, 126], [1, 122]], "level_db": -8}]})

    # 4) 녹음 흐름 재현: 사, 사, 길게 s, 길게 s
    #    협착은 **고정**(0.10 cm²)이다. 페이드 인/아웃이 전부 압력에서 나오므로
    #    무게중심도 함께 오르내린다(Stevens 1971) — 실측 7350->9008->7194 Hz.
    def long_s(audible):
        return {"type": "fricative", "phone": "s", "dur": audible / AUDIBLE_FRAC,
                "aero": True, "constriction_area": [[0, 0.10], [1, 0.10]],
                "glottal_area": 0.12, "level_db": -5}
    W("m04_like_recording.wav", {"timeline": [
        sa, {"type": "silence", "dur": 0.32}, dict(sa),
        {"type": "silence", "dur": 0.5}, long_s(1.10),
        {"type": "silence", "dur": 0.5}, long_s(1.65)]})

    print("완료.")


if __name__ == "__main__":
    main()
