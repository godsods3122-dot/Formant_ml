"""비언어 발성(제스처): 웃음, 숨, 한숨, 프라이, 속삭임, 흐느낌, 헛기침.

이것들은 특수 케이스가 아니다. 전부 같은 물리 손잡이를 다르게 흔든 것뿐이고,
그래서 전부 스크립트에서 연속적으로 섞을 수 있다("반쯤 웃으면서 말하기").

예: 웃음 '하하하' 는
    * 호기 압력의 펄스열 (5~7 Hz)  -> harmonic_amp / noise 세기의 주기적 포락선
    * 펄스마다 F0 가 튀어올랐다 떨어짐
    * 성문 마찰(/ㅎ/)이 앞에 붙음    -> noise_entry 를 성문쪽으로
    * 펄스가 짧아 Rd 가 pressed 쪽
    * 후반부로 갈수록 공기가 빠져 기식성(rd, noise_am) 증가
이 다섯 줄이 곧 아래 `laugh()` 다. 파라미터를 바꾸면 킥킥/껄껄/헛웃음이 된다.

모든 함수는 (n_frames, VoiceProfile, **옵션) -> 제어 dict 를 돌려준다.
값은 (1, T, ·) 텐서이며 그대로 `Controls` 에 넣을 수 있다.
"""
from __future__ import annotations

import math

import torch

from .presets import FRICATIVES, VOWELS
from .utils import band_bump, band_shelf, ramp
from .voice import VoiceProfile, extend_formants


def _c(t: int, v: float) -> torch.Tensor:
    return torch.full((1, t, 1), float(v))


def _phase_ramp(t: int, rate_hz: float, frame_rate: float, phase0: float = 0.0):
    n = torch.arange(t, dtype=torch.float32) / frame_rate
    return 2 * math.pi * rate_hz * n + phase0


#: 유성 구간에 깔리는 성문 기식 잡음. (게인, 셸프 하한 [Hz], 옥타브당 기울기)
#:
#: 실측 이 화자 '아' 는 4~6 kHz 에 에너지의 1.88 % 가 있는데, 잡음 없이 합성하면
#: 0.00 % 다 — **어떤 소스 기울기로도 못 만든다**(하모닉이 그 위로 안 올라간다.
#: tilt 를 -8~+6 까지 훑어도 4~6 kHz 는 최대 0.28 %). 기식성은 성문 난류가
#: 성도를 지나 깔리는 **잡음 바닥**이고, 그게 빠지면 모음이 둔탁해져 밝은 /s/ 와
#: 이어지지 않는다 — "목소리랑 연결이 안 된다" 의 스펙트럼 쪽 원인이다.
#:
#: 셸프를 1200 Hz 에 두면 잡음이 F1/F2 공진에 얹혀 1~2 kHz 부터 채운다(게인을
#: 올려도 4~6 kHz 가 차기 전에 1~2 kHz 가 69 % 로 넘친다). 3500 Hz 로 올리면
#: 하모닉이 이미 떨어진 자리에만 깔려서 실측 분포와 맞는다.
#: 3.6 -> 2.7 재적합(2026-09-04). 위 목표(모음 4~6 kHz 1.88 %)에 맞춰 잡은
#: 값인데 실제로는 3.7 % 를 내고 있었다 — 그동안 기식 경로가 여러 번 바뀌면서
#: 어긋났다. 모음 내내 안 꺼지는 히스로 들리던 원인이고, HANDOFF §5(c) 의
#: "모음 4~6 kHz 가 조금 세다" 도 이것이다.
#: 재측정: 0-1k 92.8 / 1-2k 4.1 / 4-6k 2.1 % (실측 92.6 / 5.0 / 2.1).
BREATH_NOISE_GAIN = 2.7
BREATH_NOISE_HZ = 4800.0
BREATH_NOISE_SLOPE = 3.5
#: 셸프 **아래쪽 바닥**. 0 이면 안 된다.
#:
#: 성문 난류는 광대역이다. 셸프 모양은 성도·복사가 고역을 들어 올려서 생기는
#: 것이지, 소스가 3.5 kHz 아래에서 침묵하기 때문이 아니다. 0 으로 두면 합성
#: 모음의 **하모닉 사이가 완전히 비어** 선 스펙트럼이 된다 — 실측 /아/ 는
#: 200~2000 Hz 에서 하모닉 사이 바닥이 H1 대비 -40 dB 언저리로 이어지는데
#: 합성은 -60 ~ -80 dB 였다. 그 차이가 "사각파처럼 들린다" 는 지적이다.
BREATH_NOISE_FLOOR = 0.30

#: 협착 뒤 공동으로 새어 성도 전체를 지나는 난류의 비율.
#:
#: 0.25 -> 0.70. 실측 /s/ 는 저역 치마가 두껍다. 6~13 kHz 봉우리를 기준으로
#: 80 Hz -9.3 dB, 300 Hz -2.9 dB, 800~2000 Hz +1.1 dB 인데, 0.25 에서는 각각
#: -17.5 / -8.9 / -5.7 dB 로 8 dB 씩 얇았다.
#: 긴 /s/ 로도 재확인했다(6~11 kHz 기준 0.5~2.5 kHz, 마찰음 중앙):
#:   실측 -15.5 dB / 누출 0.70 -16.5 / 0.30 -23.4 — 0.70 이 맞는다.
#: (누출 경로는 치찰음 필터를 우회하므로 0.30 으로 낮추면 저역 치마가 통째로
#:  사라진다. 0.30 을 시도했다가 되돌린 이유다.)
#: 치찰음 필터(앞공동)만 지난 소리는 봉우리만 남은 '입 밖에 얹힌 히스' 다 —
#: 실제 /s/ 는 협착 뒤 공동도 함께 울린다.
NOISE_BACK_LEAK = 0.70


def base(t: int, prof: VoiceProfile, n_formants: int = 8, n_bands: int = 40,
         vowel: str = "a", sample_rate: int = 24000) -> dict:
    """프로파일의 기본 상태(중립 모음, 말하는 F0)."""
    # 화자 포먼트로 스케일: 모음의 상대 형태는 유지하고 전체 규모만 화자에 맞춘다
    scale = (prof.formants[0] / VOWELS["a"][0]) if prof.formants else 1.0
    f = [v * scale for v in VOWELS.get(vowel, VOWELS["a"])]
    ff = extend_formants(f + list(prof.formants[len(f):]), n_formants)
    return {
        "f0": _c(t, prof.f0_median),
        "harmonic_amp": _c(t, 1.0),
        "rd": _c(t, prof.rd_median),
        "tilt": _c(t, prof.tilt),
        "jitter": _c(t, prof.jitter),
        "shimmer": _c(t, prof.shimmer),
        "formant_freq": torch.tensor(ff).reshape(1, 1, -1).expand(1, t, n_formants
                                                                 ).contiguous(),
        "formant_bw": prof.bandwidth_tensor(1, t, n_formants),
        "formant_gain": prof.gain_tensor(1, t, n_formants),
        "noise_bands": torch.full((1, t, n_bands), 1e-4),
        # 성문 기식(두 번째 노이즈 경로). 모든 세그먼트가 키를 가져야 이어붙일 때
        # 깨지지 않는다.
        #
        # 예전엔 여기가 0 이었고 `breathiness` 는 noise_am(변조 깊이)에만 걸려
        # 있었다 — **있지도 않은 잡음의 변조 깊이**만 정하고 있었던 셈이다.
        # 그래서 화자의 기식성 0.16 이 소리에 아무 기여도 안 했다.
        # **발성 중에만** 난다. 기식성은 성대가 떨면서 동시에 새는 기류의 난류라,
        # 유성 세기에 비례한다. 상수로 깔면 무성 마찰음에도 잡음 바닥이 생겨
        # /s/ 의 페이드 인/아웃 모양까지 바뀐다(측정: 페이드인 비율 58 % -> 45 %).
        "aspiration_bands": (band_shelf(n_bands, BREATH_NOISE_HZ,
                                        BREATH_NOISE_GAIN * prof.breathiness,
                                        sample_rate, slope_oct=BREATH_NOISE_SLOPE,
                                        floor=BREATH_NOISE_FLOOR * prof.breathiness)
                             .reshape(1, 1, -1).expand(1, t, n_bands).contiguous()),
        "noise_entry": _c(t, 0.0),
        "noise_am": _c(t, prof.breathiness),
        "noise_rough": _c(t, prof.roughness),
        # 난류가 성도를 울릴 때는 감쇠가 크다 (Controls.noise_bw_scale 주석 참고)
        "noise_bw_scale": _c(t, 3.0),
        "noise_back_leak": _c(t, NOISE_BACK_LEAK),

    }


# --------------------------------------------------------------------- 웃음
def laugh(t: int, prof: VoiceProfile, frame_rate: float = 100.0, rate_hz: float = 5.5,
          voiced: float = 0.85, breathiness: float = 0.45, pitch_lift: float = 1.45,
          decay: float = 0.55, vowel: str = "a", n_formants: int = 8,
          n_bands: int = 40, sample_rate: int = 24000) -> dict:
    """웃음 '하하하'. rate_hz 가 펄스 속도, voiced 가 유성/기식 비율.

    voiced=0.2, rate=8 이면 '킥킥', voiced=0.95, rate=4 이면 '껄껄',
    voiced=0.0 이면 소리 없이 숨만 터지는 웃음이 된다.
    """
    c = base(t, prof, n_formants, n_bands, vowel, sample_rate)
    ph = _phase_ramp(t, rate_hz, frame_rate)
    # 각 호기 펄스: 빠르게 열리고 천천히 닫히는 비대칭 포락선
    frac = torch.frac(torch.as_tensor(ph) / (2 * math.pi))
    pulse = torch.exp(-frac / 0.35) * (frac < 0.85).float()
    pulse = (pulse / pulse.max().clamp_min(1e-6)).reshape(1, t, 1)
    # 전체적으로 공기가 빠지며 잦아든다
    fade = ramp(t, [(0.0, 1.0), (1.0, decay)])

    c["harmonic_amp"] = pulse * fade * voiced
    # 펄스마다 F0 가 위로 튄다 (웃음의 특징적인 억양)
    c["f0"] = _c(t, prof.f0_median) * (1.0 + (pitch_lift - 1.0) * pulse) \
        * ramp(t, [(0.0, 1.06), (1.0, 0.88)])
    # 짧고 눌린 성문 펄스 -> 뒤로 갈수록 기식적으로
    c["rd"] = ramp(t, [(0.0, max(prof.rd_low, 0.45)), (1.0, prof.rd_high)])
    c["noise_am"] = _c(t, 0.8)
    c["noise_rough"] = _c(t, min(1.0, prof.roughness + 0.10))
    # 성문 마찰(/ㅎ/): 노이즈가 성도 전체를 통과한다
    c["noise_entry"] = _c(t, 0.0)
    nb = band_bump(n_bands, 1400.0, 3500.0, breathiness, sample_rate)
    c["noise_bands"] = (nb.reshape(1, 1, -1) * (0.35 + pulse)).contiguous()
    c["jitter"] = _c(t, min(0.03, prof.jitter * 4))
    c["shimmer"] = _c(t, min(0.3, prof.shimmer * 3))
    return c


#: 숨소리 난류 소스의 저역 모서리 [Hz]. **아래로 떨어진다.**
#:
#: 예전에는 가우시안 범프(중심 1680 Hz, sigma 1250)였다. 그건 난류가 아니라
#: 좁은 혹이라, 성도 전체를 지나면 F1/F2 만 남고 고역은 통째로 사라진다.
#: 측정(삽입된 들숨): 에너지의 **87.2 %가 500~1500 Hz**, 8 kHz -64 dB,
#: 12 kHz -88 dB. 저역 덩어리다.
#:
#: 실제 숨소리는 성문·인두의 난류라 **광대역**이고, 성도가 저역에서 공진해도
#: 저역이 두드러지지 않는다 — 소스에 저역이 없기 때문이다. 그래서 셸프로
#: 바꾼다(모서리 위는 넓게 평탄, 아래로는 떨어진다). 고역은 성도 캐스케이드와
#: 소스 사전(TurbulenceSource.spectral_prior)이 알아서 떨어뜨린다.
#:
#: **모서리는 F1/F2 보다 위여야 한다.** 이 잡음은 성문에서 주입돼 성도 전체를
#: 지나므로(noise_entry=0), 소스가 879/1292 Hz 근처에서 평탄하면 그 두 극이
#: 그대로 도드라진다. 모서리를 쓸어 본 결과(100-500 / 500-1.5k / 1.5-4k / 4-8k):
#:    900 Hz   0.0 / 82.8 / 15.5 /  1.7 %   <- 저역 덩어리
#:   1800 Hz   0.0 / 13.4 / 70.5 / 16.1 %
#:   2200 Hz   (채택)
#:   3500 Hz   0.0 /  0.0 / 37.1 / 62.9 %   <- 너무 얇다
#: 8 kHz 위가 0 % 인 건 성도 캐스케이드의 고역 절벽 때문이다(HANDOFF §6.8).
BREATH_SOURCE_HZ = 2200.0
BREATH_SOURCE_SLOPE = 1.6


def breath(t: int, prof: VoiceProfile, inhale: bool = False, strength: float = 0.5,
           n_formants: int = 8, n_bands: int = 40, sample_rate: int = 24000) -> dict:
    """들숨/날숨. 성대 진동 없이 성문 마찰만."""
    c = base(t, prof, n_formants, n_bands, sample_rate=sample_rate)
    c["harmonic_amp"] = torch.zeros(1, t, 1)
    # 발성이 없으면 기식성 바닥도 없다(base 가 깔아 둔 것을 끈다).
    c["aspiration_bands"] = torch.zeros(1, t, n_bands)
    env = ramp(t, [(0.0, 0.05), (0.35, 1.0), (1.0, 0.1)]) if inhale else \
        ramp(t, [(0.0, 1.0), (1.0, 0.05)])
    c["noise_bands"] = (band_shelf(n_bands, BREATH_SOURCE_HZ, strength,
                                   sample_rate, slope_oct=BREATH_SOURCE_SLOPE,
                                   floor=0.0).reshape(1, 1, -1)
                        * env).contiguous()
    c["noise_entry"] = _c(t, 0.0)
    c["noise_rough"] = _c(t, 0.20)
    return c


def sigh(t: int, prof: VoiceProfile, n_formants: int = 8, n_bands: int = 40,
         sample_rate: int = 24000) -> dict:
    """한숨: 유성으로 시작해 기식으로 풀리며 F0 가 내려간다."""
    c = base(t, prof, n_formants, n_bands, sample_rate=sample_rate)
    c["harmonic_amp"] = ramp(t, [(0.0, 0.9), (0.6, 0.55), (1.0, 0.0)])
    c["f0"] = _c(t, prof.f0_median) * ramp(t, [(0.0, 1.12), (1.0, 0.8)])
    c["rd"] = ramp(t, [(0.0, prof.rd_median), (1.0, min(2.7, prof.rd_high + 0.5))])
    c["noise_am"] = ramp(t, [(0.0, 0.3), (1.0, 1.0)])
    c["noise_bands"] = (band_bump(n_bands, 1800.0, 4000.0, 0.35, sample_rate
                                  ).reshape(1, 1, -1)
                        * ramp(t, [(0.0, 0.3), (1.0, 1.0)])).contiguous()
    return c


def creak(t: int, prof: VoiceProfile, rate_hz: float = 45.0, n_formants: int = 8,
          n_bands: int = 40, sample_rate: int = 24000) -> dict:
    """성대 프라이(보컬 프라이). 아주 낮은 F0 + 큰 지터 + 압착된 성문파.

    (2질량 ODE 로 하면 주기 배가 분기로 '진짜' 프라이가 나온다.
     LF 경로에서는 낮은 F0 + jitter + pressed Rd 로 근사한다.)
    """
    c = base(t, prof, n_formants, n_bands, sample_rate=sample_rate)
    c["f0"] = _c(t, max(rate_hz, 35.0))
    c["rd"] = _c(t, max(0.3, prof.rd_low - 0.2))
    c["jitter"] = _c(t, 0.03)
    c["shimmer"] = _c(t, 0.25)
    c["harmonic_amp"] = _c(t, 0.6)
    c["tilt"] = _c(t, prof.tilt - 2.0)
    return c


def whisper(t: int, prof: VoiceProfile, vowel: str = "a", strength: float = 0.6,
            n_formants: int = 8, n_bands: int = 40, sample_rate: int = 24000) -> dict:
    """속삭임: 성대 진동 0, 성문 협착 난류가 성도 전체를 통과."""
    c = base(t, prof, n_formants, n_bands, vowel, sample_rate)
    c["harmonic_amp"] = torch.zeros(1, t, 1)
    c["noise_bands"] = band_bump(n_bands, 1600.0, 5000.0, strength, sample_rate
                                 ).reshape(1, 1, -1).expand(1, t, -1).contiguous()
    c["noise_entry"] = _c(t, 0.0)
    c["noise_rough"] = _c(t, 0.15)
    return c


def sob(t: int, prof: VoiceProfile, frame_rate: float = 100.0, rate_hz: float = 2.6,
        n_formants: int = 8, n_bands: int = 40) -> dict:
    """흐느낌: 웃음과 같은 펄스 구조지만 느리고, F0 가 내려꽂히며 기식적."""
    c = laugh(t, prof, frame_rate, rate_hz, voiced=0.7, breathiness=0.6,
              pitch_lift=1.25, decay=0.4, n_formants=n_formants, n_bands=n_bands)
    c["f0"] = c["f0"] * ramp(t, [(0.0, 1.15), (1.0, 0.75)])
    c["rd"] = ramp(t, [(0.0, prof.rd_high), (1.0, min(2.7, prof.rd_high + 0.6))])
    return c


def throat_clear(t: int, prof: VoiceProfile, n_formants: int = 8, n_bands: int = 40,
                 sample_rate: int = 24000) -> dict:
    """헛기침: 짧은 압착 유성 + 넓은 대역 난류."""
    c = base(t, prof, n_formants, n_bands, sample_rate=sample_rate)
    c["harmonic_amp"] = ramp(t, [(0.0, 0.0), (0.15, 1.0), (0.5, 0.3), (1.0, 0.0)])
    c["f0"] = _c(t, prof.f0_low * 0.85)
    c["rd"] = _c(t, 0.4)
    c["jitter"] = _c(t, 0.02)
    c["noise_bands"] = (band_bump(n_bands, 1200.0, 3000.0, 0.6, sample_rate
                                  ).reshape(1, 1, -1)
                        * ramp(t, [(0.0, 0.2), (0.2, 1.0), (1.0, 0.15)])).contiguous()
    c["noise_entry"] = _c(t, 0.0)
    c["noise_rough"] = _c(t, 0.25)
    return c


GESTURES = {
    "laugh": laugh, "breath": breath, "sigh": sigh, "creak": creak,
    "whisper": whisper, "sob": sob, "throat_clear": throat_clear,
}
