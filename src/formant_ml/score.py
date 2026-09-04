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


#: 마찰음 노이즈 게인 1.0 일 때 측정된 "모음 - 마찰음" 레벨차 [dB].
#: 합성 경로(소스 스펙트럼 사전, 치찰음 필터, 성도)가 바뀌면 이 값도 다시 재야 한다.
#: tests/test_voice.py::test_fricative_level_matches_profile 가 드리프트를 잡는다.
#:
#: **고원부**(포락선이 1.0 인 구간)에서 잰다. 전체 RMS 로 재면 페이드 길이가
#: 레벨에 섞여 들어간다 - 페이드를 절반씩으로 늘리자 같은 게인인데 RMS 가
#: 3.7 dB 떨어졌다. 그러면 페이드를 만질 때마다 음량이 따라 움직이고, 음량을
#: 맞추려고 게인을 올리면 페이드가 도로 얕아진다. 둘은 분리되어야 한다.
FRICATIVE_CAL_DB = -19.2

#: 음절 안의 마찰음은 위 상수 대신 **같은 음절의 유성 최대 진폭**으로 기준화한다
#: (아래 참조). 그 기준이 독립 마찰음 경로와 어긋난 만큼을 여기서 되돌린다.
#: 측정: 프로파일이 -11.0 dB 를 요청했을 때 음절은 +7.8 dB 를 냈다 -> 3.2 dB 크다.
SYLLABLE_FRICATIVE_CAL_DB = 3.2


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

#: 성문 기식의 게인 보정. fricative_gain 은 앞니 다이폴(+약 16 dB)을 상쇄하려고
#: 낮춰 잡혀 있는데, 성문에는 장애물이 없어 그 부스트를 안 받는다. 되돌리는 값.
ASPIRATION_GAIN = 6.3


def wants_aero(seg: dict) -> bool:
    return any(k in seg for k in _AERO_KEYS)


def aero_drive(seg: dict, t: int, frame_rate: float, glottal_area=None) -> dict:
    """협착 면적 궤적 -> 공기역학 상태 전부. 전부 (1, T, 1).

    반환 키: env(마찰음 진폭), cent(무게중심 배율), asp(성문 기식),
             ps_norm(성대 구동압, 1.0=보통 발화), add(내전), pm_frac(구강내압 비율).

    핵심은 **구강내압**이다. /s/ 는 협착이 좁아 협착 뒤에 압력이 쌓이고(Ps 의
    60~70%), 그만큼 성대를 구동할 압력이 남지 않아 목소리가 눌린다. 협착을 풀면
    그 압력이 시상수 τ=V/(c·Ac) 로 빠지는데, Ac 가 커질수록 τ 가 21 ms -> 0.6 ms
    로 급락해서 **빠르고 부드럽게** 사라진다. 그 순간 (Ps-Pm) 이 발성 역치를 넘어
    목소리가 붙는다 — VOT 를 손으로 박지 않아도 여기서 유도된다.
    """
    ac_area = aac.constriction_area(
        t, frame_rate,
        a_closed=float(seg.get("a_closed", 0.10)),
        a_open=float(seg.get("a_open", 3.0)),
        hold=float(seg.get("hold_ratio", 0.5)),
        release=float(seg.get("release_s", 0.06)),
        shape=seg.get("constriction_area"))
    if glottal_area is None:
        glottal_area = seg.get("glottal_area", 0.12)
    ag_t = curve(glottal_area, t) if not isinstance(glottal_area, (int, float)) \
        else torch.full((1, t, 1), float(glottal_area))
    # 호흡 구동압은 **곡선**이다. /s/ 를 내려면 상당한 압력이 필요한데, 그 압력을
    # 툭 던지는 게 아니라 서서히 올렸다가 서서히 내린다. 상수로 두면 마찰음이
    # 첫 프레임부터 최대라 그 급개시가 파열음(/k/)으로 들린다.
    # 게다가 레이놀즈 게이트 때문에 압력이 낮은 동안은 난류가 아예 안 켜져서,
    # 압력 램프가 그대로 부드러운 페이드 인/아웃이 된다.
    ps_scale = seg.get("pressure_scale")
    if ps_scale is None:
        # 기본값은 **호흡 제스처**다(실측에서 적합). 신경 구동 -> 근육 2단 지연
        # -> 압력. 올릴 때가 내릴 때보다 느려서(능동 동원 vs 힘 놓기) 가청
        # 포락선의 절반 이상이 페이드 인이 된다 — 실측 56~68 %.
        ps_t = _PS_CGS * aac.breath_drive(
            t, hold=float(seg.get("breath_hold", aac.BREATH_HOLD)),
            rise=float(seg.get("breath_rise", aac.BREATH_RISE)),
            fall=float(seg.get("breath_fall", aac.BREATH_FALL)),
            sustain=seg.get("breath_sustain"))
    else:
        ps_t = _PS_CGS * (curve(ps_scale, t) if not isinstance(ps_scale, (int, float))
                          else torch.full((1, t, 1), float(ps_scale)))

    # 구강 공동을 상태변수로 적분한다: 되밀린 압력 Pm 이 유입을 깎고, 협착을
    # 지나는 유량 u 가 곧 마찰음을 만든다. 정상상태는 series_flow 와 같고,
    # 다른 건 과도구간 — 그게 /s/ 의 개시와 해제(VOT)다.
    pm, u = aac.oral_cavity(ps_t, ag_t, ac_area, frame_rate)
    drive = aac.transglottal_pressure(ps_t, pm)
    ps_norm = drive / _PS_CGS                       # 1.0 = 보통 발화 구동압

    # 내전: 성문 면적이 좁아질수록 1 에 가깝다(성문이 닫힌다).
    ag_ref = float(seg.get("glottal_open_area", 0.12))
    add = (1.0 - (ag_t / max(ag_ref, 1e-3)).clamp(0.0, 1.0)).clamp(0.02, 1.0)

    amp = aac.frication_source_amp(u, ac_area)
    env = amp / amp.amax().clamp_min(1e-9)
    asp = aac.aspiration_source_amp(u, ag_t)
    return {"env": env, "cent": aac.velocity_centroid_scale(u, ac_area),
            "asp": asp / asp.amax().clamp_min(1e-9),
            "ps_norm": ps_norm, "add": add,
            "pm_frac": pm / _PS_CGS}


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
                # /s/ 를 내려면 상당한 압력이 필요하고, 그 압력은 서서히 올랐다
                # 서서히 내린다. 그래서 독립 마찰음은 거의 절반이 페이드 인,
                # 절반이 페이드 아웃이다. 압력이 낮은 동안은 레이놀즈 게이트가
                # 난류를 아예 안 켜므로, 이 램프가 그대로 부드러운 페이드가 된다.
                seg_f = dict(seg)
                # 독립 마찰음에는 **해제가 없다**. constriction_area 의 기본값은
                # "좁게 유지하다 60 ms 만에 모음 면적으로 확 연다" 인데, 그건
                # 자음이 모음으로 넘어가는 해제 동작이다. 뒤에 모음이 없으면
                # 혀는 그 자세를 유지할 뿐이다. 그대로 두면 압력을 아무리 만져도
                # 같은 자리에서 끊긴다: 면적이 0.1 -> 3.0 으로 30 배 열리면
                # 유속이 30 배 떨어져 30 ms 만에 -44 dB 절벽이 된다(측정).
                #
                # 그래서 협착을 **고정**한다. 페이드 인/아웃은 전부 호흡 구동압이
                # 만든다(aero_drive 의 breath_drive). 그래야 무게중심도 진폭과
                # 함께 오르내린다(Stevens 1971) — 실측 7350->9008->7194 Hz 를
                # 협착을 손대지 않고 재현한다. 면적으로 페이드를 만들면 무게중심이
                # 반대로 움직여(협착이 열리면 유속이 떨어진다) 실측과 어긋난다.
                seg_f.setdefault("constriction_area", [[0.0, 0.10], [1.0, 0.10]])
                _d = aero_drive(seg_f, t, a.frame_rate)
                aero_env, aero_cent = _d["env"], _d["cent"]
                env = aero_env
            else:
                # 단순 경로: 유량 포락선을 직접 준다.
                # 기본값을 **초로 박지 않는다**. 30/40 ms 로 두었더니 400 ms 짜리
                # /s/ 가 92% 평탄한 고원이 되어, 페이드 파라미터를 아무리 만져도
                # 소리에 반영이 안 됐다. 안 주면 길이의 45% 씩(=거의 절반이
                # 페이드 인, 절반이 페이드 아웃) 잡는다 - aero.FRICATION_FADE_FRAC.
                fi, fo = seg.get("fade_in"), seg.get("fade_out")
                flow = aero.frication_flow(
                    t, a.frame_rate,
                    fade_in=None if fi is None else float(fi),
                    fade_out=None if fo is None else float(fo),
                    shape=seg.get("flow"))
                env = aero.flow_to_noise_amp(
                    flow, float(seg.get("flow_exp", aero.FRICATION_FLOW_EXPONENT)))
            c["noise_bands"] = (nb.expand(1, t, NB) * env).contiguous()
        else:
            # 실측("사"): /s/ 120 ms, 전이 60 ms, 모음 360 ms.
            # 비율이 아니라 절대 시간이 음성학적으로 맞다 — 자음 길이는 음절
            # 길이에 비례해 늘어나지 않는다.
            # 협착 유지시간. 가청 /s/ 는 이보다 길다 — 앞에 호흡 램프가, 뒤에
            # 해제+VOT 가 붙기 때문이다(0.11 -> 가청 160 ms, 실측 134~139 ms).
            onset_s = float(seg.get("onset_s", 0.11))
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
            drive = None
            if wants_aero(seg):
                # 공기역학 경로: 발성 구동압을 **구강내압에서 유도**한다.
                # /s/ 동안은 협착 뒤 압력이 Ps 의 60~70% 를 잡아먹어 성대를 구동할
                # 압력이 없다(목소리가 눌린다). 협착을 풀면 그 압력이 τ=V/(c·Ac) 로
                # 빠지며 구동압이 살아나 발성이 붙는다 -> VOT 가 유도된다.
                # 조음기관은 **셋이 따로** 움직인다. 예전엔 transition_s 하나가
                # 셋을 다 끌어서, 포먼트 전이를 길게 잡으면 후두 내전까지 같이
                # 늘어져 /s/ 가 224 ms 로 끌렸다(실측 134~139 ms).
                #   * 혀끝(협착 해제) release_s   ~50 ms — 빠르다
                #   * 후두(성문 내전) adduction_s ~40 ms — 빠르다, VOT 를 정한다
                #   * 혀몸(포먼트 전이) transition_s ~45~120 ms — 느리다
                # 한국어 평음 ㅅ 의 VOT 25~40 ms 는 후두 제스처가 정하는 값이지
                # 혀몸이 모음 목표에 도달하는 시간이 아니다.
                # 협착은 onset_s 동안 유지되다 열린다. 정점을 앞당기려고 이 값을
                # 줄이면 안 된다 — 호흡 램프가 이미 가청 개시를 뒤로 밀어놔서,
                # 협착을 그대로 두면 정점이 저절로 가청 구간의 59 % 에 온다
                # (실측 52~63 %). 줄이면 36 % 로 앞당겨져 페이드 인이 사라진다.
                hold_r = float(seg.get("hold_ratio", split))
                # 후두는 혀끝과 **거의 같이** 움직인다. 예전엔 성문이 split(명목상
                # /s/ 끝)에서야 닫히기 시작해 혀보다 한참 늦었고, 그 지연이 그대로
                # 마찰음 꼬리로 남아 하강이 85 ms 로 늘어졌다(실측 52~64 ms).
                # 해제 하나가 두 제스처를 함께 촉발해야 연결이 유기적이다.
                addt = float(seg.get("adduction_s", 0.04)) * a.frame_rate / max(t, 1)
                lag = float(seg.get("adduction_lag_s", 0.01)) * a.frame_rate / max(t, 1)
                ag_curve = seg.get("glottal_area", [
                    [0.0, 0.12], [min(hold_r + lag, 1.0), 0.12],
                    [min(hold_r + lag + addt, 1.0), 0.03], [1.0, 0.03]])
                seg2 = dict(seg)
                seg2.setdefault("a_open", 3.0)
                seg2.setdefault("hold_ratio", hold_r)
                seg2.setdefault("release_s", 0.05)      # 혀끝은 빨리 떨어진다
                # 호흡 제스처를 **마찰음 구간에 맞춰 시간축을 축소**한다(비율이
                # 불변량이므로 그냥 split 을 곱하면 된다). 모음에는 압력이 계속
                # 필요하므로 0 으로 떨어뜨리지 않고 sustain 에서 유지한다 —
                # 음절에서 마찰음이 꺼지는 건 압력이 아니라 협착이 열려서다.
                seg2.setdefault("breath_hold", aac.BREATH_HOLD * split)
                seg2.setdefault("breath_rise", aac.BREATH_RISE * split)
                seg2.setdefault("breath_fall", aac.BREATH_FALL * split)
                seg2.setdefault("breath_sustain", 0.95)
                drive = aero_drive(seg2, t, a.frame_rate, glottal_area=ag_curve)
                ps, add = drive["ps_norm"], drive["add"]
            else:
                vot = float(seg.get("voice_onset_s", 0.03)) * a.frame_rate / max(t, 1)
                ps = ramp(t, [(0.0, 0.55), (split, 0.7),
                              (min(split + vot, 1.0), 1.0), (1.0, 0.95)])
                add = ramp(t, [(0.0, 0.04), (split, 0.12),
                               (min(split + vot * 1.2, 1.0), 1.0), (1.0, 1.0)])
            asp = aero.apply(c, ps, add, rd_modal=prof.rd_median,
                             rd_breathy=min(2.6, prof.rd_high + 0.6),
                             route_noise=False)
            # 마찰음 레벨은 **같은 음절의 유성 세기**를 기준으로 맞춘다.
            # 전역 상수로 맞추면, 공기역학 모형이 내는 모음 세기(압력·Rd 에 따라
            # 달라진다)와 어긋나 음절마다 비율이 흔들린다(실측: 목표 +9.7 dB 인데
            # -6.2 dB 가 나왔다).
            # 그런데 그 기준화가 **독립 마찰음 경로와 어긋나 있었다**. 유성 최대
            # 진폭으로 재기준화하면 FRICATIVE_CAL_DB(독립 마찰음에서 잰 값)가 그대로
            # 안 맞는다 - 프로파일이 -11.0 dB 를 요청해도 음절에서는 +7.8 dB 만
            # 나왔다(3.2 dB 크다). 기울기는 1:1 로 맞으니 상수 오프셋이다.
            nb = nb * float(c["harmonic_amp"].max().clamp_min(1e-3)) \
                * (10.0 ** (-SYLLABLE_FRICATIVE_CAL_DB / 20.0))
            if drive is not None:
                # 위에서 이미 같은 면적 궤적으로 계산했다(유량·압력·난류가 한
                # 구동에서 나온다). 협착이 열리며 레이놀즈수가 임계 아래로 떨어져
                # 마찰음이 저절로 꺼지고, 같은 순간 구강내압이 빠져 발성이 붙는다.
                env, aero_cent, asp_env = drive["env"], drive["cent"], drive["asp"]
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
                       * float(seg.get("aspiration", 4.0)))
            else:
                # 마찰음 게이트: 모음으로 넘어갈 때 빠르게 꺼진다(자연스러운 fade-out).
                env = ramp(t, [(0.0, 1.0), (split, 1.0), (split + 0.1, 0.02),
                               (1.0, 0.01)])
                # 협착 형성 구간의 유량 fade-in 을 곱한다(fade-out 은 위 게이트가 담당).
                # 페이드 인은 **/s/ 구간 길이에 비례**한다. 25 ms 로 박아 두면
                # /s/ 가 길어질수록 상대적으로 짧아져 페이드가 안 들린다.
                # 음절의 /s/ 는 0 ~ split 구간이므로 그 길이를 기준으로 잡는다.
                s_dur = split * (t - 1) / max(a.frame_rate, 1e-6)     # /s/ 길이 [s]
                fin = aero.frication_flow(
                    t, a.frame_rate,
                    fade_in=float(seg.get(
                        "fade_in", aero.FRICATION_FADE_FRAC * s_dur)),
                    fade_out=0.0, shape=seg.get("flow"))
                env = env * aero.flow_to_noise_amp(
                    fin, float(seg.get("flow_exp", aero.FRICATION_FLOW_EXPONENT)))
            # 노이즈 경로가 하나뿐이라, 구강 협착(마찰음)에서 성문(기식)으로
            # 주입 위치를 옮기며 섞는다.
            # 기식은 성문에서 나므로 **앞니 다이폴이 없다**. fricative_gain 은
            # 다이폴 부스트를 상쇄하느라 -16 dB 쯤 내려가 있어서, 그걸 그대로 쓰면
            # 기식이 30 배 작아져 전이 구간을 못 메운다. 그만큼 되돌린다.
            asp_n = band_shelf(NB, 900.0,
                               fricative_gain(prof) * ASPIRATION_GAIN,
                               a.sample_rate).reshape(1, 1, -1) * float(
                                   c["harmonic_amp"].max().clamp_min(1e-3))
            # 주입 위치는 **부드럽게 미끄러뜨리면 안 된다**. 중간값은 캐스케이드의
            # 앞부분만 우회한 '반쪽 필터' 라 어떤 성도 형상에도 대응하지 않고,
            # 봉우리가 나이퀴스트 쪽으로 튄다(실측: /스/ 의 마찰음 피크가
            # 6.9 kHz 대신 11.9 kHz 로 나왔다). 지배적인 소스 쪽으로 빠르게 넘긴다.
            if drive is not None:
                # 두 소스를 **각자의 경로**로 보낸다(합성기가 더한다).
                #   구강 마찰음 -> 협착 하류 + 치찰음 필터 (noise_entry = K+6)
                #   성문 기식   -> 성도 전체            (aspiration_bands)
                # 예전에는 경로가 하나뿐이라 noise_entry 를 둘 사이에서 튕겼는데,
                # 그 전환이 전이에 레벨 점프를 만들어 연결이 끊겼다.
                c["noise_bands"] = (nb * env).contiguous()
                c["noise_entry"] = torch.full((1, t, 1), float(K) + 6.0)
                c["aspiration_bands"] = (asp_n * asp).contiguous()
            else:
                w_asp = torch.sigmoid((asp - env) * 8.0)
                c["noise_bands"] = (nb * env + asp_n * asp * 0.5).contiguous()
                c["noise_entry"] = (1.0 - w_asp) * (float(K) + 6.0)
            # 포먼트 궤적은 **로커스 이론**대로. 유성 구간을 /이/ 에서 시작해
            # 모음으로 미끄러뜨리면 그건 /j/ 활음이라 "사" 가 "야" 로 들린다.
            # 치경 로커스에서 출발해 짧은 전이(기본 50 ms)로 모음에 도달한다.
            tgt = _vowel_formants(vowel, prof, K)
            loc = LOCUS.get(phone, LOCUS["s"])
            trans = float(seg.get("transition_s", 0.045)) * a.frame_rate / max(t, 1)
            amp = c["harmonic_amp"][0, :, 0]
            voiced = (amp > 0.05 * float(amp.max().clamp_min(1e-6))).nonzero()
            onset = (float(voiced[0]) / max(t - 1, 1)) if len(voiced) else split
            # 혀는 **협착을 푸는 순간**부터 움직인다. 발성은 그보다 늦게 붙는다
            # (구강내압이 빠져야 성대가 떨 수 있으므로). 그래서 F1 도 F2/F3 도
            # 전이를 **해제 시점**에서 출발시킨다.
            #
            # F2 만 발성 시작에 맞춰 두었더니 "사" 가 "샤" 로 들렸다. 그러면 F2 가
            # 로커스에 머문 채로 유성이 시작되고, 그 뒤 100 ms 동안 1292 까지
            # 내려가는 활강이 통째로 들린다 - 고 F2 에서 저 F2 로의 유성 활강은
            # 정의상 /j/ 다. 실제로는 해제와 함께 혀가 이미 움직이고 있어서,
            # 소리가 붙을 때는 F2 전이가 거의 끝나 있다.
            onset_f1 = onset_f2 = min(onset, split)
            # F1 과 F2/F3 는 **속도가 다르다**. 다만 방향이 예상과 반대다.
            # 실측(본 화자 "사" 녹음, 30 ms 창 LPC, 유성 시작 기준):
            #   F2 : -20 ms 1317 -> +0 ms 1277 -> +30 ms 1129  (이미 도착해 있다)
            #   F1 : +0 ms 599 -> +25 ms 670 -> +85 ms 875     (천천히 열린다)
            # 혀 몸통(F2)은 협착을 푸는 순간 튕기듯 제자리로 가고, 턱(F1)은
            # 관성이 커서 뒤늦게 열린다. 그래서 F1 쪽 전이가 오히려 더 길다.
            trans_f1 = float(seg.get("f1_transition_s", 0.055)) \
                * a.frame_rate / max(t, 1)
            tracks = []
            for k in range(K):
                start = tgt[k] if k >= 3 or loc[k] is None else float(loc[k])
                on_k, tr = (onset_f1, trans_f1) if k == 0 else (onset_f2, trans)
                tracks.append(ramp(t, [(0.0, start), (on_k, start),
                                       (min(on_k + tr, 1.0), tgt[k]),
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
