"""스크립트(YAML/JSON) -> 제어 파라미터 -> 음성.

설계 목표는 하나다: **물리모델의 모든 손잡이를 텍스트로 지정할 수 있게 한다.**
음소든 웃음이든 특별 취급이 없고, 전부 같은 파라미터를 다르게 흔든 것이다.

    name: 데모
    voice: profiles/me.json        # 없으면 기본 프로파일
    timeline:
      - {type: syllable, onset: s, vowel: a, dur: 0.45, f0: [150, 120]}
      - {type: laugh, dur: 1.1, rate_hz: 5.5, voiced: 0.85, tilt: 3}
      - {type: vowel, vowel: i, dur: 0.4, rd: [[0, 0.5], [1, 2.2]]}

값 쓰는 법 (모든 파라미터 공통)
-------------------------------
    rd: 1.2                     상수
    rd: [0.5, 2.2]              시작 -> 끝 선형
    rd: [[0, 0.5], [0.7, 0.5], [1, 2.2]]   구간별 브레이크포인트 (위치는 0~1)

주소가 있는 파라미터는 `PARAM_HELP` 에 전부 있다 (`python -m formant_ml.render
--list-params`). 세그먼트 안에서 쓰면 그 구간에만, `params:` 아래에 쓰면
전체 발화에 적용된다.
"""
from __future__ import annotations

import inspect
import json
import os

import torch

from .config import Config
from dataclasses import fields as dc_fields

from .dsp.sibilant import PRESETS as SIB_PRESETS
from .dsp.sibilant import SibilantParams
from . import aerodynamics as aero
from . import aeroacoustic as aac
from .gestures import GESTURES, base
from .models.synth import Controls, PhysicalVoiceSynth
from .presets import FRICATIVES, LOCUS, VOWELS
from .utils import band_bump, band_shelf, ramp
from .voice import VoiceProfile, extend_formants

# 프레임률 스칼라 제어 (전부 (1, T, 1))
SCALAR_PARAMS = {
    "f0": "기본주파수 [Hz]",
    "harmonic_amp": "유성 성분 세기 (0 = 무성)",
    "rd": "성문파 형상 0.3 pressed ~ 2.7 breathy (개방지수)",
    "tilt": "소스 스펙트럼 기울기 [dB/oct @1kHz] — 고역을 살리거나 죽인다",
    "jitter": "주기 요동 비율 (0.004 = 0.4%)",
    "shimmer": "진폭 요동 비율",
    "noise_entry": "난류 주입 위치 0=성문 … K=입술 … K+6=성도 완전 우회",
    "noise_am": "성문동기 노이즈 변조 깊이 (기식성)",
    "noise_rough": "난류의 시간 변조 (0 = 정상 히스, 1 = 거친 난류)",
    "noise_bw_scale": "노이즈 경로 포먼트 대역폭 배율 (1~6). 낮으면 잡음이 "
                      "공진에서 울려 음조가 들린다",
    "level_db": "이 구간 전체 레벨 [dB]. 유성·무성 성분을 함께 키우고 줄인다",
    "noise_back_leak": "구강 결합 0~1. 협착 뒤 공동까지 새어 통과하는 비율. "
                       "0 이면 마찰음이 입 밖에 얹힌 히스처럼 들린다",
    "pressure": "성문하압 (1.0 = 보통 발화). 세기·F0·성문파 형상이 함께 따라온다",
    "adduction": "성대 내전 0~1. 낮으면 기식, 높으면 모달. 발성 역치압을 정한다",
    "noise_gain": "난류 전체 세기",
    "noise_center": "난류 대역 중심 [Hz]",
    "noise_bw": "난류 대역폭 [Hz]",
    "sib_pole_f": "치찰음 앞공동 공진 [Hz]  (/s/ 5~8k, /ʃ/ 2.5~4k)",
    "sib_pole_bw": "그 대역폭 [Hz] (좁을수록 쨍하다)",
    "sib_zero_f": "치찰음 반공진 [Hz]",
    "sib_zero_bw": "반공진 대역폭 [Hz]",
    "sib_tilt": "치찰음 기울기 [dB/oct]",
    "sib_slope_lo": "치찰음 봉우리 아래 상승 기울기 [dB/oct]. 크면 삼각형, "
                    "작으면 둥근 돔이 된다 (사람 /s/ 는 20~40)",
    "sib_slope_hi": "치찰음 봉우리 위 하강 기울기 [dB/oct, 음수]",
    "sib_teeth_f": "앞니 공명 [Hz]. 혀끝-앞니 틈으로 빠지는 휘파람 성분 "
                   "(실측: 7.2 kHz)",
    "sib_teeth_bw": "앞니 공명 대역폭 [Hz]",
    "sib_floor_db": "직접 방사 바닥 [dB, 피크 대비]. 높을수록 스펙트럼이 "
                    "전역적으로 깔린다 (실측: -11)",
    "sib_mix": "치찰음 필터 적용량 0~1",
}
FORMANT_PARAMS = {f"f{k + 1}": f"제{k + 1} 포먼트 [Hz]" for k in range(12)}
FORMANT_BW_PARAMS = {f"bw{k + 1}": f"제{k + 1} 포먼트 대역폭 [Hz]" for k in range(12)}
VECTOR_PARAMS = {
    "formant_freq": "포먼트 전체 [Hz] 리스트(상수)",
    "formant_bw": "포먼트 대역폭 전체 리스트(상수)",
    "formant_gain": "포먼트 게인 전체 리스트(상수)",
    "disp_freq": "위상차 올패스 중심주파수 리스트 [Hz]",
    "disp_radius": "위상차 올패스 극반지름 리스트 (0~0.95)",
    "area": "도파관 단면적 함수 [cm^2] (tract: waveguide 일 때)",
}
PARAM_HELP = {**SCALAR_PARAMS, **FORMANT_PARAMS, **FORMANT_BW_PARAMS,
              **VECTOR_PARAMS}
SEGMENT_TYPES = ["vowel", "glide", "fricative", "syllable", "silence",
                 *sorted(GESTURES)]


# ------------------------------------------------------------------ 값 해석
def curve(spec, t: int) -> torch.Tensor:
    """스칼라 / [시작, 끝] / [[위치, 값], ...] -> (1, T, 1)."""
    if isinstance(spec, (int, float)):
        return torch.full((1, t, 1), float(spec))
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)):
        return ramp(t, [(float(p), float(v)) for p, v in spec])
    vals = [float(v) for v in spec]
    if len(vals) == 1:
        return torch.full((1, t, 1), vals[0])
    pos = [i / (len(vals) - 1) for i in range(len(vals))]
    return ramp(t, list(zip(pos, vals)))


#: 마찰음 노이즈 게인 1.0 일 때 측정된 "모음 - 마찰음" RMS 차 [dB].
#: 합성 경로(소스 스펙트럼 사전, 치찰음 필터, 성도)가 바뀌면 이 값도 다시 재야 한다.
#: tests/test_voice.py::test_fricative_level_matches_profile 가 드리프트를 잡는다.
FRICATIVE_CAL_DB = -18.5   # 앞니 다이폴 도입으로 재측정(고역이 크게 살아남)


def fricative_gain(prof: VoiceProfile) -> float:
    """프로파일의 `fricative_level_db` 를 실제 노이즈 게인으로 바꾼다."""
    want = -float(prof.fricative_level_db)          # 모음이 이만큼 커야 한다
    return float(10.0 ** ((FRICATIVE_CAL_DB - want) / 20.0))


#: 공기음향 마찰음을 요청하는 키들. 하나라도 있으면 임의 페이드 대신 협착 면적
#: 궤적에서 유량·레이놀즈 게이트·무게중심을 **유도**한다(aeroacoustic.py).
_AERO_KEYS = ("constriction_area", "a_open", "a_closed", "release_s", "hold_ratio",
              "glottal_area", "aero")
_PS_CGS = 8000.0    # 보통 발화 성문하압 [dyn/cm^2] ≈ 8 cmH2O (pressure=1.0 기준)

#: 앞니라는 **장애물**이 있는 음소(치찰음). 제트가 앞니에 부딪혀 다이폴 소스를
#: 만든다(Shadle 1985/1990). /f/ /θ/ /h/ 는 장애물이 없어 훨씬 약하고 평평하다.
OBSTACLE_SIBILANTS = {"s", "ss", "z", "sh"}


def wants_aero(seg: dict) -> bool:
    return any(k in seg for k in _AERO_KEYS)


def aero_frication(seg: dict, t: int, frame_rate: float,
                   glottal_area=None):
    """협착 면적 궤적 -> (진폭 포락선, 무게중심 배율). 둘 다 (1, T, 1).

    물리(aeroacoustic.py): 폐압이 성문·구강 협착을 직렬로 지나며 유량 U 를 만들고,
    협착부 레이놀즈수가 임계값을 넘을 때만 난류(마찰음)가 난다. 협착이 열리거나
    (모음) 성문이 닫히면(발성) 유량이 줄어 저절로 꺼지고, 입자속도가 떨어지면
    무게중심도 내려간다. 손으로 그린 페이드/치찰음 곡선이 아니라 **면적에서 전부
    유도**된다.

    `glottal_area` 로 성문 면적 궤적(스칼라/곡선)을 줄 수 있다. 안 주면 seg 에서
    읽고, 그것도 없으면 무성 마찰음 기본값(0.12, 열림)을 쓴다.
    """
    ac_area = aac.constriction_area(
        t, frame_rate,
        a_closed=float(seg.get("a_closed", 0.10)),
        a_open=float(seg.get("a_open", 3.0)),
        hold=float(seg.get("hold_ratio", 0.5)),
        release=float(seg.get("release_s", 0.06)),
        shape=seg.get("constriction_area"))
    # 성문 면적: 무성 마찰음은 열려 있고(기류 셈), 발성으로 가며 내전해 닫힌다.
    if glottal_area is None:
        glottal_area = seg.get("glottal_area", 0.12)
    ag_t = curve(glottal_area, t) if not isinstance(glottal_area, (int, float)) \
        else torch.full((1, t, 1), float(glottal_area))
    ps = _PS_CGS * float(seg.get("pressure_scale", 1.0))
    ps_t = torch.full((1, t, 1), ps)
    u = aac.series_flow(ps_t, ag_t, ac_area)
    amp = aac.frication_source_amp(u, ac_area)
    env = amp / amp.amax().clamp_min(1e-9)          # 피크 1 로 정규화(레벨은 nb 가)
    cent = aac.velocity_centroid_scale(u, ac_area)
    # 성문에서의 난류(기식). 성문이 내전하며 좁아지는 **도중**에 유속이 올라 최대가
    # 된다 — 그래서 구강 마찰음이 꺼지고 발성이 아직 안 붙은 전이 구간을 이 기식이
    # 메운다. 이게 없으면 그 자리에 무음이 생겨 '무음 + 급개시' = 폐쇄음(/t/)으로
    # 들린다(실측: /사/ 에서 120 ms 무음 -> "스트라" 처럼 들림).
    asp = aac.aspiration_source_amp(u, ag_t)
    asp_env = asp / asp.amax().clamp_min(1e-9)
    return env, cent, asp_env


def _scale_sib_center(sib: SibilantParams, cent: torch.Tensor) -> None:
    """무게중심 배율(1,T,1)을 치찰음 공진 주파수들에 곱한다(제자리).

    입자속도가 떨어지면(협착이 열리면) 앞공동 공진·앞니 공명·반공진이 함께
    내려간다 = 스펙트럼 전체가 아래로 미끄러진다(Stevens 1971).
    """
    for k in ("pole_f", "zero_f", "teeth_f"):
        v = getattr(sib, k)
        if v is not None:
            setattr(sib, k, v * cent)


def _vowel_formants(name: str, prof: VoiceProfile, n: int) -> list[float]:
    """이 화자의 해당 모음 포먼트.

    측정값이 있으면 그대로 쓰고, 없으면 프리셋을 화자 성도 규모로 스케일한다.
    (균일 스케일은 근사다 — F1 과 F2 를 같은 비율로 옮기는데 실제 화자는
     그렇지 않다. 그래서 측정값이 있으면 항상 그쪽이 우선이다.)
    """
    measured = prof.vowel_formants.get(name)
    if measured:
        return extend_formants(list(measured), n)
    scale = (prof.formants[0] / VOWELS["a"][0]) if prof.formants else 1.0
    f = [v * scale for v in VOWELS.get(name, VOWELS["a"])]
    return extend_formants(f + list(prof.formants[len(f):]), n)


# ------------------------------------------------------------- 세그먼트 만들기
def build_segment(seg: dict, prof: VoiceProfile, cfg: Config) -> dict:
    """세그먼트 하나 -> 프레임률 제어 dict (+ 'sib' 하위 dict)."""
    a = cfg.audio
    K, NB = cfg.filt.n_formants, cfg.noise.n_bands
    dur = float(seg.get("dur", 0.3))
    t = max(1, int(round(dur * a.frame_rate)))
    kind = seg.get("type", "vowel")

    if kind in GESTURES:
        fn = GESTURES[kind]
        # 제스처 함수가 실제로 받는 인자만 넘긴다. 예전에는 TypeError 를 잡아
        # **인자 없이 다시 호출**했는데, 오타 하나가 세그먼트의 모든 옵션을
        # 조용히 삼켜 버렸다(무엇이 무시됐는지 알 수도 없었다).
        accepted = set(inspect.signature(fn).parameters)
        opts = {k: v for k, v in seg.items()
                if k not in PARAM_HELP and k not in ("type", "dur", "sib")}
        unknown = set(opts) - accepted
        if unknown:
            raise ValueError(
                f"'{kind}' 세그먼트가 모르는 옵션: {sorted(unknown)}. "
                f"가능한 옵션: {sorted(accepted - {'t', 'prof', 'n_formants', 'n_bands'})} "
                f"또는 파라미터 이름: {sorted(PARAM_HELP)[:6]} …")
        c = fn(t, prof, n_formants=K, n_bands=NB, **opts)
    elif kind == "silence":
        c = base(t, prof, K, NB)
        c["harmonic_amp"] = torch.zeros(1, t, 1)
        c["noise_bands"] = torch.full((1, t, NB), 1e-6)
    elif kind == "vowel":
        c = base(t, prof, K, NB, seg.get("vowel", "a"))
        c["formant_freq"] = torch.tensor(
            _vowel_formants(seg.get("vowel", "a"), prof, K)
        ).reshape(1, 1, -1).expand(1, t, K).contiguous()
    elif kind == "glide":
        names = seg.get("vowels", ["a", "i"])
        tracks = []
        for k in range(K):
            pts = [(i / max(len(names) - 1, 1), _vowel_formants(v, prof, K)[k])
                   for i, v in enumerate(names)]
            tracks.append(ramp(t, pts))
        c = base(t, prof, K, NB, names[0])
        c["formant_freq"] = torch.cat(tracks, dim=-1)
    elif kind in ("fricative", "syllable"):
        phone = seg.get("onset", seg.get("phone", "s"))
        vowel = seg.get("vowel", "a")
        c = base(t, prof, K, NB, vowel)
        cf, bw, pos, g = FRICATIVES.get(phone, FRICATIVES["s"])
        # 소스는 광대역, 모양은 치찰음 필터가 만든다(둘이 겹치면 손잡이가 죽는다).
        # 레벨: 마찰음이 유성음보다 얼마나 조용한지는 화자 프로파일이 정한다.
        # 안 맞추면 치찰음만 튀어나와 들린다(실측 한국어 "스": 모음이 5.4 dB 큼).
        nb = band_shelf(NB, 500.0, g * fricative_gain(prof),
                        a.sample_rate).reshape(1, 1, -1)
        # 앞니 다이폴: 치찰음의 지배적 소스다. 이게 없으면 난류 소스의 -6 dB/oct
        # 롤오프가 그대로 남아 9~12 kHz 가 반토막 나고 5~6 kHz 에 없는 혹이 생겨
        # '스' 로 안 들린다(실측 대조: 고역 79% vs 합성 38%).
        if phone in OBSTACLE_SIBILANTS and seg.get("obstacle_dipole", True):
            nb = nb * aac.obstacle_dipole_bands(
                NB, a.sample_rate,
                float(seg.get("dipole_db_oct", 10.0))).reshape(1, 1, -1)
        aero_env = aero_cent = None
        if kind == "fricative":
            c["harmonic_amp"] = torch.zeros(1, t, 1)
            if wants_aero(seg):
                # 공기음향 경로: 협착 면적에서 진폭·무게중심을 유도(권장).
                aero_env, aero_cent, _asp = aero_frication(seg, t, a.frame_rate)
                env = aero_env
            else:
                # 단순 경로: 유량 포락선을 직접 준다(페이드 인/아웃, 초).
                flow = aero.frication_flow(
                    t, a.frame_rate,
                    fade_in=float(seg.get("fade_in", 0.03)),
                    fade_out=float(seg.get("fade_out", 0.04)),
                    shape=seg.get("flow"))
                env = aero.flow_to_noise_amp(
                    flow, float(seg.get("flow_exp", aero.FRICATION_FLOW_EXPONENT)))
            c["noise_bands"] = (nb.expand(1, t, NB) * env).contiguous()
        else:
            # 실측("사"): /s/ 120 ms, 전이 60 ms, 모음 360 ms.
            # 비율이 아니라 절대 시간이 음성학적으로 맞다 — 자음 길이는 음절
            # 길이에 비례해 늘어나지 않는다.
            onset_s = float(seg.get("onset_s", 0.12))
            split = float(seg.get("onset_ratio", onset_s / max(dur, 1e-3)))
            split = min(max(split, 0.05), 0.8)
            # 발성 시작을 **공기역학**으로 만든다. 세기만 램프로 올리면 녹음을
            # 페이드인한 것처럼 들린다 — 실제로는 성문하압이 오르면서 세기·F0·
            # 성문파 형상·기식 소음이 한꺼번에 따라 움직인다 (aerodynamics.py).
            # 발성 개시 시간(VOT)은 **초** 로 잡는다. 예전에는 세그먼트 길이의
            # 12~14% 로 두어서, 0.58 s 음절이면 내전에 81 ms 가 걸렸다 — 마찰음이
            # 꺼진 뒤 발성이 붙기까지 무음이 100 ms 넘게 생겨 '무음 + 급개시' 가
            # 폐쇄음으로 들렸다(/사/ 가 "스트라"). 한국어 평음 ㅅ 의 VOT 는 대략
            # 25~40 ms 다.
            vot = float(seg.get("voice_onset_s", 0.03)) * a.frame_rate / max(t, 1)
            ps = ramp(t, [(0.0, 0.55), (split, 0.7), (min(split + vot, 1.0), 1.0),
                          (1.0, 0.95)])
            add = ramp(t, [(0.0, 0.04), (split, 0.12),
                           (min(split + vot * 1.2, 1.0), 1.0), (1.0, 1.0)])
            asp = aero.apply(c, ps, add, rd_modal=prof.rd_median,
                             rd_breathy=min(2.6, prof.rd_high + 0.6),
                             route_noise=False)
            # 마찰음 레벨은 **같은 음절의 유성 세기**를 기준으로 맞춘다.
            # 전역 상수로 맞추면, 공기역학 모형이 내는 모음 세기(압력·Rd 에 따라
            # 달라진다)와 어긋나 음절마다 비율이 흔들린다(실측: 목표 +9.7 dB 인데
            # -6.2 dB 가 나왔다).
            nb = nb * float(c["harmonic_amp"].max().clamp_min(1e-3))
            if wants_aero(seg):
                # 공기음향 경로: 협착 면적이 열리며(release) 레이놀즈수가 임계
                # 아래로 떨어져 마찰음이 저절로 꺼진다 = 물리적 fade-out. 무게중심도
                # 함께 내려간다. 손으로 그린 게이트/치찰음 곡선이 필요 없다.
                seg2 = dict(seg)
                seg2.setdefault("a_open", 3.0)
                seg2.setdefault("hold_ratio", split)
                seg2.setdefault("release_s", float(seg.get("transition_s", 0.05)))
                trans = float(seg.get("transition_s", 0.05)) * a.frame_rate / max(t, 1)
                # 성문은 /s/ 동안 열려 있다가(무성) 발성이 시작되며 내전해 닫힌다.
                # 성문이 닫히면 유량이 성문에서 막혀 구강 마찰음도 함께 꺼진다 —
                # 그래서 마찰음 offset 이 발성 onset 과 물리적으로 맞물린다.
                ag_curve = seg.get("glottal_area", [
                    [0.0, 0.12], [split, 0.12], [min(split + trans, 1.0), 0.03],
                    [1.0, 0.03]])
                env, aero_cent, asp_env = aero_frication(
                    seg2, t, a.frame_rate, glottal_area=ag_curve)
                # 기식은 **기류는 있는데 아직 발성이 안 붙은** 동안에만 난다.
                # 성대가 제대로 떨기 시작하면 성문이 주기적으로 닫혀 난류가 사라진다.
                # 이 게이트가 있어야 기식이 정확히 '마찰음 꺼짐 ~ 발성 시작' 창을
                # 메운다. 안 그러면 그 자리가 무음이 되어 무음+급개시 = /t/ 로
                # 들린다(실측 /사/ 가 "스트라" 로 들리던 원인).
                # 게이트가 둘이다. (1-v_frac): 발성이 붙으면 성문이 주기적으로 닫혀
                # 난류가 사라진다. (1-env): 구강 협착이 좁을 때는 압력강하가 입에
                # 몰려 성문 유속이 낮다 — 협착이 열려야 압력강하가 성문으로 옮겨와
                # 기식이 난다. 둘을 곱하면 기식이 정확히 '마찰음 꺼짐 ~ 발성 시작'
                # 창에만 남아, 그 자리의 무음(=/t/ 지각)을 메운다.
                vamp = c["harmonic_amp"]
                v_frac = (vamp / vamp.amax().clamp_min(1e-6)).clamp(0.0, 1.0)
                asp = (asp_env * (1.0 - v_frac) * (1.0 - env).clamp(0.0, 1.0)
                       * float(seg.get("aspiration", 1.0)))
            else:
                # 마찰음 게이트: 모음으로 넘어갈 때 빠르게 꺼진다(자연스러운 fade-out).
                env = ramp(t, [(0.0, 1.0), (split, 1.0), (split + 0.1, 0.02),
                               (1.0, 0.01)])
                # 협착 형성 구간의 유량 fade-in 을 곱한다(fade-out 은 위 게이트가 담당).
                fin = aero.frication_flow(
                    t, a.frame_rate, fade_in=float(seg.get("fade_in", 0.025)),
                    fade_out=0.0, shape=seg.get("flow"))
                env = env * aero.flow_to_noise_amp(
                    fin, float(seg.get("flow_exp", aero.FRICATION_FLOW_EXPONENT)))
            # 노이즈 경로가 하나뿐이라, 구강 협착(마찰음)에서 성문(기식)으로
            # 주입 위치를 옮기며 섞는다.
            asp_n = band_shelf(NB, 900.0, fricative_gain(prof), a.sample_rate
                               ).reshape(1, 1, -1) * float(
                                   c["harmonic_amp"].max().clamp_min(1e-3))
            # 주입 위치는 **부드럽게 미끄러뜨리면 안 된다**. 중간값은 캐스케이드의
            # 앞부분만 우회한 '반쪽 필터' 라 어떤 성도 형상에도 대응하지 않고,
            # 봉우리가 나이퀴스트 쪽으로 튄다(실측: /스/ 의 마찰음 피크가
            # 6.9 kHz 대신 11.9 kHz 로 나왔다). 지배적인 소스 쪽으로 빠르게 넘긴다.
            w_asp = torch.sigmoid((asp - env) * 8.0)
            c["noise_bands"] = (nb * env + asp_n * asp * 0.5).contiguous()
            c["noise_entry"] = (1.0 - w_asp) * (float(K) + 6.0)
            # 포먼트 궤적은 **로커스 이론**대로. 유성 구간을 /이/ 에서 시작해
            # 모음으로 미끄러뜨리면 그건 /j/ 활음이라 "사" 가 "야" 로 들린다.
            # 치경 로커스에서 출발해 짧은 전이(기본 50 ms)로 모음에 도달한다.
            tgt = _vowel_formants(vowel, prof, K)
            loc = LOCUS.get(phone, LOCUS["s"])
            trans = float(seg.get("transition_s", 0.05)) * a.frame_rate / max(t, 1)
            # 전이는 **실제 유성이 시작하는 순간**부터다. 그 전에 시작하면 포먼트가
            # 이미 모음 쪽으로 움직인 뒤에 소리가 나서 활음처럼 들린다.
            amp = c["harmonic_amp"][0, :, 0]
            voiced = (amp > 0.05 * float(amp.max().clamp_min(1e-6))).nonzero()
            onset = (float(voiced[0]) / max(t - 1, 1)) if len(voiced) else split
            tracks = []
            for k in range(K):
                start = tgt[k] if k >= 3 or loc[k] is None else float(loc[k])
                tracks.append(ramp(t, [(0.0, start), (onset, start),
                                       (min(onset + trans, 1.0), tgt[k]),
                                       (1.0, tgt[k])]))
            c["formant_freq"] = torch.cat(tracks, dim=-1)
        # 치찰음 필터를 쓰는 동안에는 노이즈를 포먼트 캐스케이드에 통과시키지
        # 않는다. 앞공동 공진을 '캐스케이드의 마지막 포먼트' 와 '치찰음 극' 두
        # 군데서 모델링하면 이중 계산이 되어, 치찰음 파라미터를 돌려도 소리가
        # 안 바뀐다(캐스케이드의 고정된 극이 이긴다).
        if kind == "fricative":
            c["noise_entry"] = torch.full((1, t, 1), float(K) + 6.0)
        # 음소가 *범주*(s 냐 ʃ 냐)를 정하고, 화자 프로파일이 그 안에서 *개인차*를
        # 준다. 프로파일만 쓰면 /s/ 와 /ʃ/ 가 똑같은 소리가 나고, 프리셋만 쓰면
        # 화자 지문이 사라진다. 프로파일의 표준 /s/ 대비 비율을 프리셋에 곱한다.
        base_p = dict(SIB_PRESETS.get(phone, SIB_PRESETS["s"]))
        ref = SIB_PRESETS["s"]
        if seg.get("use_profile_sibilant", True):
            for key in ("pole_f", "pole_bw", "zero_f", "zero_bw", "teeth_f",
                        "teeth_bw"):
                if key in prof.sibilant and ref.get(key):
                    base_p[key] = base_p[key] * (prof.sibilant[key] / ref[key])
            for key in ("tilt", "slope_lo", "slope_hi", "floor_db"):
                if key in prof.sibilant:
                    base_p[key] = base_p[key] + (prof.sibilant[key] - ref[key])
        sib = SibilantParams.constant((1, t, 1), mix=1.0,
                                      roughness=prof.roughness, **base_p)
        # 공기음향 경로: 입자속도에서 유도한 무게중심 배율로 치찰음 공진을 시변화.
        # 협착이 열리며 속도가 떨어지면 봉우리가 내려간다(Stevens 1971). 실측 /사/
        # 의 무게중심 하강(6700->3900)이 손 곡선 없이 여기서 나온다.
        if aero_cent is not None:
            _scale_sib_center(sib, aero_cent)
        c["sib"] = sib
    else:
        raise ValueError(f"모르는 세그먼트 type: {kind!r}. "
                         f"가능한 값: {', '.join(SEGMENT_TYPES)}")

    c.setdefault("sib", prof.sibilant_params((1, t, 1), mix=0.0))
    apply_overrides(c, seg, t, cfg)
    return c


def apply_overrides(c: dict, spec: dict, t: int, cfg: Config) -> None:
    """스크립트에 적힌 파라미터로 제어 dict 를 덮어쓴다."""
    K, NB = cfg.filt.n_formants, cfg.noise.n_bands
    noise_shape = {}
    for key, val in spec.items():
        if key in ("type", "dur"):
            continue
        # pressure/adduction 은 *파생* 손잡이다 — 아래 aero.apply 가 spec 에서 직접
        # 읽어 세기·F0·Rd·기식을 만든다. 제어 dict 에 곡선으로 저장하면 안 된다:
        # Controls 필드도 아니고, 일부 세그먼트에만 있으면 이어붙이기(torch.cat)가
        # 깨진다(치찰음+압력모음 타임라인에서 실제로 터졌다).
        if key in ("pressure", "adduction"):
            continue
        if key in ("noise_gain", "noise_center", "noise_bw"):
            noise_shape[key] = val
        elif key.startswith("sib_"):
            field = key[4:]
            setattr(c["sib"], field, curve(val, t))
        elif key == "level_db":
            g = 10.0 ** (curve(val, t) / 20.0)
            c["harmonic_amp"] = c["harmonic_amp"] * g
            c["noise_bands"] = c["noise_bands"] * g
        elif key in SCALAR_PARAMS:
            c[key] = curve(val, t)
        elif key in FORMANT_PARAMS:
            k = int(key[1:]) - 1
            if k < K:
                c["formant_freq"] = c["formant_freq"].clone()
                c["formant_freq"][..., k: k + 1] = curve(val, t)
        elif key in FORMANT_BW_PARAMS:
            k = int(key[2:]) - 1
            if k < K:
                c["formant_bw"] = c["formant_bw"].clone()
                c["formant_bw"][..., k: k + 1] = curve(val, t)
        elif key in ("formant_freq", "formant_bw", "formant_gain", "area"):
            v = torch.tensor([float(x) for x in val], dtype=torch.float32)
            c[key] = v.reshape(1, 1, -1).expand(1, t, len(v)).contiguous()
        elif key in ("disp_freq", "disp_radius"):
            v = torch.tensor([float(x) for x in val], dtype=torch.float32)
            c[key] = v.reshape(1, 1, -1).expand(1, t, len(v)).contiguous()
    if "pressure" in spec or "adduction" in spec:
        # 압력/내전이 주어지면 세기·F0·Rd·기식을 거기서 일관되게 끌어낸다
        aero.apply(c, curve(spec.get("pressure", 1.0), t),
                   curve(spec.get("adduction", 1.0), t))
    if noise_shape:
        sr = cfg.audio.sample_rate
        g = noise_shape.get("noise_gain", 1.0)
        ctr = noise_shape.get("noise_center", 5000.0)
        bw = noise_shape.get("noise_bw", 4000.0)
        base_nb = band_bump(NB, float(ctr) if not isinstance(ctr, list) else
                            float(ctr[0]),
                            float(bw) if not isinstance(bw, list) else float(bw[0]),
                            1.0, sr).reshape(1, 1, -1)
        c["noise_bands"] = (base_nb * curve(g, t)).contiguous()


# ---------------------------------------------------------------- 이어붙이기
def enforce_formant_spacing(freq: torch.Tensor, min_gap: float = 60.0
                            ) -> torch.Tensor:
    """F1 < F2 < … 와 최소 간격을 구조적으로 보장 (인코더와 같은 규약).

    인코더는 cumsum(softplus) 로 이걸 보장하지만 스크립트는 아무 값이나 쓸 수
    있다. 극이 겹치면 캐스케이드 이득이 폭발하므로 여기서 한 번 막는다.
    """
    out = [freq[..., :1]]
    for k in range(1, freq.shape[-1]):
        out.append(torch.maximum(freq[..., k:k + 1], out[-1] + min_gap))
    return torch.cat(out, dim=-1)


def _smooth(x: torch.Tensor, n: int) -> torch.Tensor:
    """프레임축 이동평균 (세그먼트 경계의 계단을 없앤다)."""
    if n < 2:
        return x
    k = torch.ones(x.shape[-1], 1, n, dtype=x.dtype) / n
    pad = torch.nn.functional.pad(x.transpose(1, 2), (n // 2, n - 1 - n // 2),
                                  mode="replicate")
    return torch.nn.functional.conv1d(pad, k, groups=x.shape[-1]).transpose(1, 2)


def build_controls(score: dict, prof: VoiceProfile, cfg: Config) -> Controls:
    """스크립트 전체 -> 하나의 Controls (위상은 발화 전체에서 연속이다).

    `score["prosody"]` 가 있으면 운율 계획을 적용한다 (prosody.py).
    """
    from .prosody import ProsodyPlan, apply_to_controls, warp_timeline
    plan = ProsodyPlan.from_dict(score.get("prosody"))
    timeline = warp_timeline(score.get("timeline", []), plan)
    segs = [build_segment(s, prof, cfg) for s in timeline]
    if not segs:
        raise ValueError("timeline 이 비어 있습니다")
    keys = set().union(*[set(s) for s in segs]) - {"sib"}
    merged = {k: torch.cat([s[k] for s in segs], dim=1) for k in keys}
    # 필드 목록을 **하드코딩하지 않는다**. 예전에는 7 개를 적어 두어서, 나중에
    # 추가한 스커트 기울기가 이어붙이는 순간 조용히 사라졌다(합성에는 None 이
    # 전달되고 아무도 알려주지 않았다).
    sib_fields = {}
    for f in dc_fields(SibilantParams):
        vals = [getattr(s["sib"], f.name) for s in segs]
        sib_fields[f.name] = (None if any(v is None for v in vals)
                              else torch.cat(vals, dim=1))
    t = merged["f0"].shape[1]

    # 전역 오버라이드
    if score.get("params"):
        apply_overrides(merged | {"sib": SibilantParams(**sib_fields)},
                        score["params"], t, cfg)

    # 경계 평활. `noise_entry` 는 **일부러 빼 둔다**: 협착 위치는 연속적으로
    # 미끄러지는 양이 아니라 조음 상태다. 무음(성문 주입)에서 /s/(입술쪽 주입)로
    # 보간하면 중간 프레임이 '캐스케이드의 앞부분만 우회한 반쪽 필터' 가 되는데,
    # 그건 어떤 성도 형상에도 대응하지 않는 응답이고 국소 이득이 100 배까지
    # 솟아 전이부에 클릭을 만든다. 들리는 전이는 noise_bands 포락선이 만든다.
    n = int(score.get("smooth_frames", 2))
    for k in ("f0", "rd", "tilt", "formant_freq", "formant_bw", "harmonic_amp",
              "noise_bands", "noise_am", "noise_rough"):
        if k in merged:
            merged[k] = _smooth(merged[k], n)

    merged["formant_freq"] = enforce_formant_spacing(merged["formant_freq"])

    df, dr = prof.dispersion_tensors(1, t)
    merged.setdefault("disp_freq", df)
    merged.setdefault("disp_radius", dr)
    if merged.get("disp_freq") is None or merged.get("disp_radius") is None:
        merged["disp_freq"] = merged["disp_radius"] = None

    merged["sib"] = SibilantParams(**sib_fields)
    merged["noise_rough"] = merged.get("noise_rough")
    valid = {f for f in Controls.__dataclass_fields__}
    ctrl = Controls(**{k: v for k, v in merged.items() if k in valid})
    return apply_to_controls(ctrl, plan, cfg.audio.frame_rate,
                             cfg.source.f0_min, cfg.source.f0_max)


def render(score: dict, prof: VoiceProfile | None = None, cfg: Config | None = None,
           seed: int | None = None) -> torch.Tensor:
    """스크립트 -> 파형 (1, N)."""
    cfg = cfg or Config()
    prof = prof or VoiceProfile()
    ctrl = build_controls(score, prof, cfg)
    synth = PhysicalVoiceSynth(cfg, tract_mode=score.get("tract", "formant"))
    gen = None
    if seed is not None or "seed" in score:
        gen = torch.Generator().manual_seed(int(score.get("seed", seed or 0)))
    with torch.no_grad():
        return synth(ctrl, generator=gen)["audio"]


def load_score(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:                               # pragma: no cover
            raise ImportError("YAML 스크립트를 읽으려면 pyyaml 이 필요합니다 "
                              "(pip install pyyaml). JSON 으로 써도 됩니다.") from e
        return yaml.safe_load(text)
    return json.loads(text)


def load_profile(score: dict, base_dir: str = ".") -> VoiceProfile:
    p = score.get("voice")
    if not p:
        return VoiceProfile()
    path = p if os.path.isabs(p) else os.path.join(base_dir, p)
    return VoiceProfile.load(path if os.path.exists(path) else p)
