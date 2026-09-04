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
from dataclasses import replace

from formant_ml import gestures as G
from formant_ml import score as SC
from formant_ml.config import Config
from formant_ml.score import render
from formant_ml.utils import save_wav
from formant_ml import voice as VP
from formant_ml.voice import VoiceProfile

#: **샘플레이트에 의존하는 잡음 보정 상수들** (44.1 kHz 용).
#:
#: 이 상수들은 `band_shelf(n_bands, ...)` 가 0~나이퀴스트를 n_bands 로 나누는 데서
#: 샘플레이트에 딸려 온다. 24 kHz 에서는 한 밴드가 300 Hz 이고 3500 Hz 셸프 위로
#: 28 밴드가 남는데, 44.1 kHz 에서는 551 Hz 에 34 밴드다. 같은 게인을 쓰면 잡음
#: 파워가 22 kHz 까지 퍼져서 모음 4~6 kHz 비중이 2.1 % -> 0.27 % 로 8 배 줄어든다.
#:
#: 그래서 24 kHz 기본값을 그대로 쓰면 44.1 kHz 출력이 이렇게 어긋난다:
#:   모음-마찰음 +9.9 dB (목표 +11~13), 전이 최저 0.070 (목표 >=0.13),
#:   모음 4~6 kHz 0.27 % (실측 2.1 %)
#: 아래 값으로 다시 맞추면: +12.0 dB / 0.159 / 2.20 % / 0-1k 92.9 % (실측 92.6).
#:
#: **모듈 기본값을 안 건드리는 이유**: 테스트 79 개가 24 kHz 를 전제로 잡혀 있다
#: (hop 240, 밴드 인덱스, 프레임 크기). 기본값을 44.1 kHz 로 바꿨더니 17 개가
#: 깨졌다. 제대로 된 해법은 이 상수들을 **샘플레이트 불변**으로 만드는 것이다
#: (셸프를 활성 밴드 수로 정규화). HANDOFF §5g 참조.
#: ASPIRATION_GAIN 10.0 -> 2.5: 마찰음에서 목소리로 넘어가는 창의 기식이 너무
#: 컸다. 실측은 그 구간 최대가 모음 정상부의 27 % 인데 83 % 가 나왔고, 포락선
#: 으로 보면 **모음보다 큰 스파이크 버스트**였다(전이/모음 1.08, 실측 0.36).
#: 그게 경성 개시로 들린다. 2.5 에서 25~27 % 로 맞는다.
CAL_44K = {"BREATH_NOISE_GAIN": 16.0, "SYLLABLE_FRICATIVE_CAL_DB": 11.5}

# 협착 궤적을 손으로 그리지 않는다. 예전엔 7 개 꺾은점으로 페이드를 만들었는데,
# 그건 물리가 아니라 곡선 맞추기였고 호흡 압력과 서로 싸워서 정점이 가청 구간의
# 24 % 로 앞당겨졌다(실측 52~63 %). 이제 협착은 그냥 '좁게 유지하다 해제' 이고,
# 페이드 인/아웃은 전부 **호흡 구동압**에서 나온다(aeroacoustic.breath_drive).

#: 독립 마찰음에서 가청 /s/ 는 세그먼트의 약 68 % 다 — 나머지는 협착이 아직
#: 덜 좁아 임계 유속에 못 미치는 앞뒤 구간이다(그게 페이드 인/아웃의 실체다).
#: 그래서 1.1 s 짜리 /s/ 를 들으려면 세그먼트를 1.6 s 로 잡는다.
#:
#: 0.53 -> 0.68 로 올렸다. 구동을 **혀 제스처**로 바꾸면서(HANDOFF §5h) 포락선이
#: 협착에서 나오게 됐고, 협착은 호흡압보다 가청 구간을 넓게 만든다. 물리가
#: 바뀐 게 아니라 같은 소리의 가청 구간이 달라진 것이라 여기서 되받는다
#: (재측정: 1.62 s 세그먼트 -> 가청 1103 ms, 실측 1097 ms).
AUDIBLE_FRAC = 0.68


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # **44.1 kHz 로 낸다.** 기본값 24 kHz 는 나이퀴스트가 12 kHz 인데, 이 화자의
    # 앞니 공명이 10015 Hz 다 — 봉우리 바로 위 1.2 kHz 에서 스펙트럼이 벽에 부딪힌다.
    # 실제 /s/ 는 16 kHz 위까지 이어지고 기준 녹음도 44.1 kHz 다. 그래서 24 kHz 로
    # 내면 /s/ 가 통째로 둔탁해진다(사용자 지적).
    # 측정: 치찰음 봉우리가 24 kHz 에서 10266~10289 Hz, 44.1 kHz 에서 9905~10099 Hz
    # (실측 9905~9991). 나이퀴스트가 봉우리를 위로 밀고 있었다.
    here = os.path.dirname(os.path.abspath(__file__))
    prof = VoiceProfile.load(os.path.join(here, "..", "profiles", "me.json"))
    cfg = Config()
    # 극의 개수도 **나이퀴스트에 딸려 온다**. 균일관은 c/(2L)=1 kHz 마다 극이
    # 하나이므로(voice.POLE_SPACING_HZ), 12 kHz 까지면 12 개, 22 kHz 까지면
    # 22 개다. 12 개로 두면 8.8 kHz 위에 극이 없어서 캐스케이드가 절벽이 되고
    # 모음의 6 kHz 위가 통째로 죽는다(실측 대비 9~13 kHz -37.8 dB).
    # 상한이 있다. 캐스케이드가 **DC 정규화**(Klatt 관례)라 나이퀴스트 근처의
    # 극은 이득이 1 을 크게 넘고, 그런 극을 몇 개 얹으면 응답 전체가 거기서
    # 지배돼 저역이 상대적으로 사라진다. 측정(모음 대역 비중, 실측 대비):
    #   극 15 개(최고 12951 Hz): 50~300 Hz 18.7 dB (실측 18.3) — 정상
    #   극 18 개(최고 15951 Hz): 50~300 Hz -12.0 dB — 저역이 30 dB 무너짐
    #   극 22 개(최고 21951 Hz): 50~300 Hz -87.5 dB — 완전 붕괴
    # 그래서 측정된 극 위로 두 개까지만 연장한다.
    n_pol = len(prof.formants) + 2
    cfg = replace(cfg, audio=replace(cfg.audio, sample_rate=44100,
                                     hop_size=441, fmax=22050.0),
                  filt=replace(cfg.filt, n_formants=n_pol))
    sr = cfg.audio.sample_rate
    # 샘플레이트를 올렸으면 잡음 보정도 같이 올려야 한다(위 CAL_44K 주석 참조).
    G.BREATH_NOISE_GAIN = CAL_44K["BREATH_NOISE_GAIN"]
    SC.SYLLABLE_FRICATIVE_CAL_DB = CAL_44K["SYLLABLE_FRICATIVE_CAL_DB"]

    def W(name, score, rms=0.05):
        y = render({"seed": 5, "smooth_frames": 2, **score}, prof, cfg)
        save_wav(os.path.join(args.out, name), y, sr, target_rms=rms)
        print(f"  {name}  ({y.shape[-1] / sr:.2f}s)")

    print(f"내 목소리 · 공기음향 재구성 -> {args.out}/")

    # 1) '사' — 호흡 압력이 페이드를, 협착 해제가 마찰음 종료를 만든다.
    #    onset_s 0.11 -> 가청 /s/ 160 ms, 정점 60 %, 상승 96 / 하강 64 ms
    #    (실측 134~139 ms, 52~63 %, 70~87 / 52~64 ms).
    # **혀 제스처 구동**(HANDOFF §6.1 완료). 독립 마찰음과 같은 논문 구조를
    # 음절에도 적용했다 — 폐압은 일정하고 포락선은 혀가 만든다.
    # 재측정(44.1 kHz): 마찰 127.7 ms(실측 122~139), 정점 59 %(57~68),
    # 모음 세기상승 75.5 ms(81~87), 모음 개시 H1-H2 +10.2 dB(+11.8~12.6).
    # 혀 제스처는 **닫고 - 유지하고 - 연다**. 고원(hold_s)이 곧 마찰음의 몸통
    # 이다(Kim et al. 2022: "높은 Pio 고원 = 음향 마찰음 길이"). 고원 없이
    # 삼각형으로 두면 상승만 있고 몸통이 없어 짧고 가파른 소리가 된다.
    # 측정: onset_s 0.09 + hold 0.07 -> 가청 136.4 ms, 정점 64 %, 10->90 상승
    # 43.5 ms (실측 130 ms, 42~57 %, 48 ms).
    sa = {"type": "syllable", "onset": "s", "vowel": "a", "dur": 0.72,
          "onset_s": 0.09, "hold_s": 0.07, "aero": True, "drive": "tongue",
          "f0": [[0, 129], [1, 123]]}
    # 발화는 **무음에서 시작한다**. 첫 세그먼트가 곧바로 시작하면 필터 상태가
    # 0 에서 출발해 첫 1 ms 에 진폭 0.34 짜리 트랜지언트가 난다(실측 0.0054).
    lead = {"type": "silence", "dur": 0.08}
    W("m01_sa.wav", {"timeline": [lead, sa]})
    W("m02_sa_sa.wav", {"timeline": [lead, sa, {"type": "silence", "dur": 0.3}, dict(sa)]})

    # 2) 재구성 본편: 길게 끈 치찰음 -> 짧은 '으' 전이 -> '아'
    #    '으' 는 별도 모음이 아니라 /s/ 자세(혀 높음)의 잔상이다. 로커스에서
    #    출발해 60 ms 안에 '아' 로 간다 — 길게 끌면 그게 활음(/j/)이라 "야" 가 된다.
    #    압력은 /s/ 내내 서서히 오르므로 페이드 인이 마찰음 전체에 걸친다.
    W("m03_s_to_eu_a.wav", {"timeline": [
        lead,
        {"type": "syllable", "onset": "s", "vowel": "a", "dur": 1.05,
         "onset_s": 0.09, "hold_s": 0.50, "aero": True, "drive": "tongue",
         "f0": [[0, 131], [1, 122]]}]})

    # 3) 유성 마찰음 /z/ — 성대가 떨며 마찰음을 성문주기로 변조한다(소스-치찰음 결합).
    #    성문이 좁게 떨어 유량이 맥동 -> 마찰음이 그 주기로 변조(Jackson&Shadle 2000).
    W("m05_voiced_fricative_z.wav", {"timeline": [
        {"type": "fricative", "phone": "z", "dur": 0.6, "aero": True,
         "constriction_area": [[0, 0.4], [0.1, 0.12], [0.9, 0.12], [1, 0.5]],
         "glottal_area": 0.04, "harmonic_amp": 0.9, "noise_am": 0.9,
         # 호흡을 유지한다. 기본 호흡 제스처는 구간의 40 % 만 누르고 힘을 놓는데,
         # 여기선 협착을 90 % 까지 잡고 있다. 그래서 **혀가 아직 안 열었는데
         # 압력이 먼저 빠져** 마찰음이 65 % 에서 죽고 뒤쪽 3 분의 1 이 마찰 없는
         # 맹숭한 유성음으로 남았다(측정). 사람은 /z/ 를 끄는 동안 숨을 계속
         # 밀고, 끝내는 건 혀다(§2). sustain 을 주면 96 % 까지 산다.
         "breath_sustain": 0.95,
         "f0": [[0, 126], [1, 122]], "level_db": -8}]})

    # 4) 녹음 흐름 재현: 사, 사, 길게 s, 길게 s
    #    협착은 **고정**(0.10 cm²)이다. 페이드 인/아웃이 전부 압력에서 나오므로
    #    무게중심도 함께 오르내린다(Stevens 1971) — 실측 7350->9008->7194 Hz.
    def long_s(audible):
        # **혀 제스처 구동**(논문 구조). 폐압은 일정하고 포락선은 전부 협착에서
        # 나온다 — Signorello et al.(2018) 이 기관 천자로 잰 Ps 는 마찰음 내내
        # 거의 일정하고, 변하는 건 혀가 만드는 구강내압이다. 예전의 "협착 고정 +
        # 호흡 아치" 는 Po/Ps 를 0.59 에 고정시키고(실측 0.85) 유량을 가운데서
        # 최대로 만들었다(실측은 가운데가 최소인 U 자).
        # 재측정: 정점 57 %(실측 56~57), 상승/하강 1.3, 봉우리 9991 Hz,
        # 정점-3 dB 이내 20 %(호흡 구동은 30 % — 그만큼 덜 평평하다).
        return {"type": "fricative", "phone": "s", "dur": audible / AUDIBLE_FRAC,
                "aero": True, "drive": "tongue",
                "glottal_area": 0.12, "level_db": -5}
    W("m04_like_recording.wav", {"timeline": [
        lead, sa, {"type": "silence", "dur": 0.32}, dict(sa),
        {"type": "silence", "dur": 0.5}, long_s(1.10),
        {"type": "silence", "dur": 0.5}, long_s(1.65)]})

    print("완료.")


if __name__ == "__main__":
    main()
