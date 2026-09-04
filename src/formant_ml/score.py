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
    "noise_bw_scale": "구강 마찰음 경로 포먼트 대역폭 배율 (1~6). 낮으면 잡음이 "
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
#:
#: 3.2 -> 25.0 으로 재측정했다. 무게중심 배율의 앵커를 구간 정점으로 옮기면서
#: 치찰음 봉우리가 8.4 -> 10.0 kHz 로 올라갔는데, 하필 거기가 앞니 다이폴 부스트가
#: 가장 큰 자리(f_max_boost 11 kHz)라 /s/ 가 통째로 세졌다. 그 결과 음절에서
#: **마찰음이 모음보다 9.5 dB 컸다** — 실측은 모음이 11.4~12.7 dB 크다(21 dB 오차).
#: /s/ 만 튀어나오고 모음이 묻히니 "목소리랑 연결이 안 된다" 로 들린다.
#:
#: 위의 FRICATIVE_CAL_DB 테스트는 이걸 못 잡는다 — 그건 **비공기역학** 경로만
#: 재는데 무게중심 배율은 공기역학 경로에만 걸리기 때문이다. 그래서 음절 경로용
#: 검사를 따로 뒀다(test_voice.py::test_syllable_fricative_level_matches_profile).
#:
#: 25.0 -> 17.0 재측정: 앞니 다이폴 기울기를 제트에 묶으면서(_dipole_jet_correction)
#: 음절의 /s/ 스펙트럼이 재배분됐다. 모양만 바꾸도록 RMS 정규화를 했는데도
#: 대역별 재배분이 성도·치찰음 필터를 거치며 레벨을 8 dB 옮긴다.
SYLLABLE_FRICATIVE_CAL_DB = 17.0


def fricative_gain(prof: VoiceProfile) -> float:
    """프로파일의 `fricative_level_db` 를 실제 노이즈 게인으로 바꾼다."""
    want = -float(prof.fricative_level_db)          # 모음이 이만큼 커야 한다
    return float(10.0 ** ((FRICATIVE_CAL_DB - want) / 20.0))


#: 공기음향 마찰음을 요청하는 키들. 하나라도 있으면 임의 페이드 대신 협착 면적
#: 궤적에서 유량·레이놀즈 게이트·무게중심을 **유도**한다(aeroacoustic.py).
_AERO_KEYS = ("drive", "a_min", "a_rest", "close_frac",
              "constriction_area", "a_open", "a_closed", "release_s", "hold_ratio",
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
    # `drive="tongue"` 이면 협착 궤적을 **혀 제스처**로 만든다(논문 구조).
    # 그때 포락선은 전부 여기서 나오고 폐압은 일정하다 — aeroacoustic.
    # tongue_constriction 의 주석과 HANDOFF §5h 참조.
    if seg.get("drive") == "tongue" and seg.get("constriction_area") is None:
        ac_area = aac.tongue_constriction(
            t, frame_rate,
            a_min=float(seg.get("a_min", aac.TONGUE_A_MIN)),
            a_rest=float(seg.get("a_rest", aac.TONGUE_A_REST)),
            close_frac=float(seg.get("close_frac", aac.TONGUE_CLOSE_FRAC)))
    elif torch.is_tensor(seg.get("constriction_area")):
        # 이미 만들어진 궤적(음절의 CV 혀 제스처)을 그대로 쓴다.
        ac_area = seg["constriction_area"]
    else:
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
    if ps_scale is None and seg.get("drive") == "tongue":
        # **폐압을 일정하게 둔다.** Signorello et al.(2018) 이 기관 천자로 직접
        # 잰 Ps 는 마찰음 내내 8.0 -> 8.9 -> 8.4 hPa 로 거의 안 변한다. 변하는
        # 건 구강내압 Po 이고, 그건 혀가 만든다. 예전에는 여기에 호흡 아치를
        # 걸어 놓고 협착을 고정했는데(정반대), 그러면 Po/Ps 가 0.59 에 **고정**
        # 되고(실측은 0.08 -> 0.85 -> 0.08 로 크게 변한다) 유량이 가운데서
        # 최대가 된다(실측은 가운데가 최소인 U 자다). 부호까지 반대였다.
        ps_t = torch.full((1, t, 1), _PS_CGS * float(seg.get("pressure", 1.0)))
        # 발화 개시에서는 폐압이 서는 데 시간이 걸린다. 이게 없으면 성도가
        # 열려 있는 첫 프레임에 유량이 즉시 최대가 되어 성문 기식이 계단으로
        # 켜지고, 그 계단이 성도를 때려 파열음 버스트가 된다(§breath_onset).
        ps_t = ps_t * aac.breath_onset(
            t, frame_rate, float(seg.get("breath_onset_s", aac.BREATH_ONSET_S)),
            float(seg.get("breath_delay_s", 0.0)))
    elif ps_scale is None:
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
    # 기준은 **그 구간에서 실제로 벌어진 최대 성문 면적**이다. 0.12 로 박아
    # 두면 개대를 넓힐 때(마찰음은 0.25 cm² 안팎으로 크게 벌어진다) ag/0.12 가
    # 1 을 넘어 clamp 에 걸리고, 내전이 0.02 에 붙어 있다가 성문이 0.12 아래로
    # 내려온 뒤에야 움직인다 — 내전 램프가 뒤쪽으로 압축돼 기식 구간이 안 생긴다.
    ag_ref = float(seg.get("glottal_open_area", float(ag_t.amax())))
    add = (1.0 - (ag_t / max(ag_ref, 1e-3)).clamp(0.0, 1.0)).clamp(0.02, 1.0)

    amp = aac.frication_source_amp(u, ac_area)
    env = amp / amp.amax().clamp_min(1e-9)
    asp = aac.aspiration_source_amp(u, ag_t)
    # 성문/구강 난류의 **세기 배분**. asp 는 아래에서 제 최대값으로 정규화되는데
    # 그러면 직렬 모형이 계산해 둔 이 배분이 지워진다. 따로 꺼내 둔다.
    # 성문 난류가 (1) 얼마나 **생기고** (2) 얼마나 협착을 지나 **나오는가**.
    # 서로 다른 물리라 둘 다 곱한다.
    asp_share = (aac.glottal_drop_fraction(ag_t, ac_area)
                 * aac.constriction_transmission(
                     ac_area, float(seg.get("a_open", aac.NEUTRAL_TRACT_AREA))))
    # 소스와 필터를 **나눈다**. 예전에는 입자속도에서 낸 배율 하나를 극·영점·앞니
    # 공명에 전부 곱했는데, 그건 Stevens 의 소스 관계를 성도 공진에 잘못 건 것이다.
    # /s/ 개시에는 압력이 낮아 속도가 느리므로 배율이 0.62 까지 내려가고, 앞니
    # 공명 10.0 kHz 가 6.2 kHz(= /ʃ/ 영역)에 놓였다 — **/s/ 가 /ʃ/ 로 시작했다.**
    # 게다가 페이드 인 구간은 협착 면적이 **고정**이다(우리가 그렇게 만들었다).
    # 기하가 안 변하는데 공진만 움직이는 건 자기모순이다.
    #
    #   cent      : 협착 **면적**(기하) -> 앞공동 극·반공진. 협착이 고정인 동안
    #               정확히 1.0 이라 지문이 첫 프레임부터 그대로 나온다.
    #   src_tilt  : 입자속도 -> 난류 **소스** 기울기 [dB/oct]. 개시가 어두운 건
    #               맞지만, 그건 봉우리가 옮겨가서가 아니라 소스가 약해서다.
    #
    # 앞니 공명(teeth_f)은 혀끝-앞니 간극이 정하는 순수 기하라 **둘 다 안 곱한다**.
    cent = aac.area_centroid_scale(ac_area)
    src_tilt = aac.source_tilt_shift(u, ac_area)
    # 앞니 다이폴의 **세기**는 제트가 정한다(주파수가 아니라). 협착이 목표
    # 자세에 못 미치면 제트가 느려 다이폴이 약해지고 앞공동 극이 드러난다.
    teeth = aac.obstacle_strength(ac_area, glottal_area=ag_t)
    return {"env": env, "cent": cent, "src_tilt": src_tilt, "teeth": teeth,
            "asp_share": asp_share,
            "asp": asp / asp.amax().clamp_min(1e-9),
            "ps_norm": ps_norm, "add": add,
            "pm_frac": pm / _PS_CGS}


#: `cent` 를 곱하는 대상. **`teeth_f` 는 여기 없다** — 앞니 공명은 혀끝과 앞니
#: 사이 간극이 정하는 순수 기하이고, 협착이 열려도 앞니는 그 자리에 있다.
#: 극(앞공동 1/4 파장)과 영점(뒤공동 반공진)만 최협착점이 물러나며 내려간다.
_GEOMETRIC_SIB_KEYS = ("pole_f", "zero_f")


def _dipole_jet_correction(n_bands: int, sample_rate: float, db_per_oct: float,
                           strength: torch.Tensor) -> torch.Tensor:
    """제트 세기 `strength`(0~1)에 맞춘 앞니 다이폴 기울기 보정 (1,T,NB).

    다이폴 게인은 10^(d·oct/20) 이라 기울기 d 가 **지수에 선형**이다. 따라서
    기울기를 g 배로 줄이는 건 원래 게인을 g 제곱하는 것과 같고, 이미 곱해 둔
    전체 다이폴에 대한 보정은 `dip^(g-1)` — 즉 기울기 d·(g-1) 짜리 다이폴을
    한 번 더 곱하면 된다. 그래서 같은 함수를 시변 기울기로 재사용한다.

    왜 필요한가: 8~13 kHz 를 지배하는 건 앞니 공진기(좁은 봉우리)가 아니라 이
    광대역 상승이다. 협착이 목표 자세에 못 미쳐 제트가 느리면 장애물 다이폴도
    약해져야 앞공동 극(5.3 kHz)이 드러난다 — 실측에서 짧은 '사' 가 4.7~5.6 kHz
    에서 봉우리를 세우는 이유다.
    """
    corr = aac.obstacle_dipole_bands(n_bands, sample_rate,
                                     db_per_oct * (strength - 1.0))
    # **모양만** 바꾸고 전체 세기는 그대로 둔다(밴드 RMS 로 정규화).
    # 안 그러면 기울기를 낮추는 게 곧 /s/ 를 조용하게 만드는 일이 된다 —
    # /s/ 에너지가 거의 다 고역에 있어서 기울기를 깎으면 13 dB 가 통째로
    # 빠지고, 그러면 모음이 고역 정점을 가져간다(측정). 세기는 프로파일의
    # fricative_level_db 와 보정 상수가 정하고, 이 함수는 배분만 한다.
    return corr / corr.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


def _scale_sib_center(sib: SibilantParams, cent: torch.Tensor,
                      src_tilt: torch.Tensor | None = None,
                      teeth: torch.Tensor | None = None) -> None:
    """협착 기하에서 나온 배율을 치찰음 공진에 곱한다(제자리).

    `cent` 는 **면적**에서 나온다(`aeroacoustic.area_centroid_scale`). 협착이
    좁게 유지되는 동안 1.0 이고, 열릴 때만 앞공동 극과 반공진이 내려간다.

    `src_tilt` 는 입자속도에서 나온 난류 **소스** 기울기 변화 [dB/oct] 다
    (Stevens 1971). 공진 주파수가 아니라 `tilt` 에 더한다 — 그래야 압력이 낮은
    개시 구간에서 봉우리 **위치**는 지문(앞니 10.0 kHz) 그대로이고 스펙트럼
    **색깔**만 어두워진다. 예전처럼 공진에 곱하면 /s/ 가 /ʃ/ 로 시작한다.
    """
    for k in _GEOMETRIC_SIB_KEYS:
        v = getattr(sib, k)
        if v is not None:
            setattr(sib, k, v * cent)
    if src_tilt is not None and sib.tilt is not None:
        sib.tilt = sib.tilt + src_tilt
    if teeth is not None:
        # 앞니 공진의 **세기**. 제트가 느리면 다이폴이 물러나고 앞공동 극이
        # 드러난다(aeroacoustic.obstacle_strength). 주파수는 안 건드린다.
        sib.teeth_gain = teeth if sib.teeth_gain is None else sib.teeth_gain * teeth


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
        # 침묵은 **기류가 없다는 뜻**이다. 잡음은 전부 기류에서 나오므로 세 경로를
        # 다 꺼야 한다. 예전엔 `aspiration_bands` 를 안 껐다 — base() 가 그걸
        # 상수 셸프로 깔아 두기 때문에, "침묵" 구간이 사실은 **최대 -3 dB 의
        # 기식 잡음**이었다(m02 의 두 '사' 사이, m04 의 토큰 사이 -19~-21 dB).
        # 발성이 끝나도 잡음이 안 꺼지고 계속 나던 게 이것이다.
        c = base(t, prof, K, NB, sample_rate=a.sample_rate)
        c["harmonic_amp"] = torch.zeros(1, t, 1)
        c["noise_bands"] = torch.full((1, t, NB), 1e-6)
        c["aspiration_bands"] = torch.zeros(1, t, NB)
    elif kind == "vowel":
        c = base(t, prof, K, NB, seg.get("vowel", "a"), a.sample_rate)
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
        c = base(t, prof, K, NB, names[0], a.sample_rate)
        c["formant_freq"] = torch.cat(tracks, dim=-1)
    elif kind in ("fricative", "syllable"):
        phone = seg.get("onset", seg.get("phone", "s"))
        vowel = seg.get("vowel", "a")
        c = base(t, prof, K, NB, vowel, a.sample_rate)
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
            # 다이폴 기울기는 **제트에 묶인다**. 8~13 kHz 를 지배하는 건 앞니
            # 공진기(좁은 봉우리)가 아니라 이 광대역 상승이라, 협착이 목표에
            # 못 미쳐 제트가 느릴 때 이것도 같이 약해져야 앞공동 극이 드러난다.
            # 시변 텐서로 주면 (1,T,NB) 가 되고, 상수면 예전대로 (NB,) 다.
            _dpo = float(seg.get("dipole_db_oct", 10.0))
            dip = aac.obstacle_dipole_bands(NB, a.sample_rate, _dpo)
            nb = nb * (dip if dip.dim() == 3 else dip.reshape(1, 1, -1))
        aero_env = aero_cent = aero_tilt = aero_teeth = None
        if kind == "fricative":
            c["harmonic_amp"] = torch.zeros(1, t, 1)
            # 무성 마찰음에는 기식성 잡음 바닥이 없다 — 그건 **발성 중에** 성대가
            # 덜 닫혀 새는 잡음이라 발성이 없으면 존재하지 않는다. 남겨두면
            # /s/ 위에 상수 잡음이 깔려 페이드 모양이 뭉개진다(58 % -> 45 %).
            c["aspiration_bands"] = torch.zeros(1, t, NB)
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
                # `drive="tongue"` 이 아닐 때만 협착을 고정한다. 혀 제스처
                # 모드에서는 협착이 **바로 그 포락선**이므로 고정하면 안 된다.
                if seg_f.get("drive") != "tongue":
                    seg_f.setdefault("constriction_area",
                                     [[0.0, 0.10], [1.0, 0.10]])
                _d = aero_drive(seg_f, t, a.frame_rate)
                aero_env, aero_cent = _d["env"], _d["cent"]
                aero_tilt = _d["src_tilt"]
                aero_teeth = _d["teeth"]
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
                # **혀 제스처 구동**(논문 구조, HANDOFF §6.1). 폐압을 일정하게
                # 두고 포락선을 협착에서 낸다 — Signorello et al.(2018) 이 기관
                # 천자로 잰 Ps 는 마찰음 내내 거의 일정하고(8.0->8.9->8.4 hPa),
                # 변하는 건 혀가 만드는 구강내압이다(2.5->7.6->5.6).
                #
                # 최소 면적 순간이 곧 마찰음 정점이자 해제 시작이고, 그 하나가
                # 포먼트 전이·내전·구강내압 방전을 **함께** 촉발한다.
                if seg.get("drive") == "tongue" and "constriction_area" not in seg:
                    # 폐쇄/해제 시간은 "가청 길이 onset_s, 정점 위치 peak" 에서
                    # 유도한다(aeroacoustic.cv_gesture_times). CV 는 해제가 모음
                    # 면적까지 열려 로그 거리가 한쪽만 기므로, 독립 마찰음의
                    # close_frac 을 그대로 쓰면 정점이 75 % 로 밀린다.
                    # **혀는 소리가 안 나는 자세에서 출발해야 한다.** 제스처의
                    # 시작 면적이 가청 경계보다 좁으면 첫 프레임부터 마찰음이
                    # 켜져 있어서, 그 계단이 파열음처럼 들린다("ksa").
                    # 측정: 개대를 0.25 로 넓히자 가청 경계가 0.392 -> 0.766 cm²
                    # 로 올라갔는데 a_rest 는 0.60 그대로여서, 합성 /s/ 의 첫
                    # 프레임 값이 0.10 이었다(실측은 0.017 = 무음).
                    # 그래서 기본 출발 자세를 가청 경계에서 **유도**한다.
                    _abd = float(seg.get("abduction_area", 0.25))
                    _amin = float(seg.get("a_min", aac.TONGUE_CV_A_MIN))
                    _rest = float(seg.get("a_rest", max(
                        aac.TONGUE_A_REST,
                        aac.REST_AUDIBLE_MARGIN * aac.audible_area(_amin, _abd))))
                    _cls, _rel = aac.cv_gesture_times(
                        onset_s, float(seg.get("frication_peak", aac.TONGUE_CV_PEAK)),
                        a_min=_amin, a_rest=_rest,
                        a_open=float(seg.get("a_open", 3.0)),
                        glottal_area=_abd)
                    cls = float(seg.get("close_s", _cls))
                    rel = float(seg.get("release_s", _rel))
                    # 폐압은 **협착을 만들면서 같이** 오른다. 성도가 아직 열려
                    # 있는 동안 압력이 다 걸리면 그 구간이 /h/ 가 된다 —
                    # 측정: 압력 램프 45 ms, 협착이 가청 경계를 지나는 시점
                    # 30 ms 였을 때 앞쪽 기식 혹이 0.037(마찰음은 0.0005).
                    # 램프를 폐쇄 시간에 맞추면 0.014 로 줄고 마찰음은 그대로다.
                    # 램프 길이는 생리적 상수(~45 ms)로 두고 **시작을 늦춘다**.
                    # 협착이 다 만들어지는 순간(cls)에 압력이 서도록 맞춘다.
                    seg.setdefault("breath_delay_s", max(cls - aac.BREATH_ONSET_S, 0.0))
                    hld = float(seg.get("hold_s", onset_s))
                    ac_cv, hold_r = aac.tongue_constriction_cv(
                        t, a.frame_rate, cls, rel, hld,
                        a_min=_amin, a_rest=_rest,
                        a_open=float(seg.get("a_open", 3.0)))
                    # 혀 제스처에 맞는 후두 타이밍 기본값. 마찰음의 성문 개대는
                    # 크고(Löfqvist & Yoshioka 1984) 되모으는 데 시간이 걸린다.
                    seg = {**seg, "constriction_area": None}
                    seg.setdefault("adduction_s", 0.05)
                    seg.setdefault("firming_s", 0.06)
                    _cv_area = ac_cv
                else:
                    _cv_area = None
                    hold_r = float(seg.get("hold_ratio", split))
                # 후두는 혀끝과 **거의 같이** 움직인다. 예전엔 성문이 split(명목상
                # /s/ 끝)에서야 닫히기 시작해 혀보다 한참 늦었고, 그 지연이 그대로
                # 마찰음 꼬리로 남아 하강이 85 ms 로 늘어졌다(실측 52~64 ms).
                # 해제 하나가 두 제스처를 함께 촉발해야 연결이 유기적이다.
                addt = float(seg.get("adduction_s", 0.04)) * a.frame_rate / max(t, 1)
                # 내전은 마찰음이 **끝날 때** 시작한다(해제 시점). 더 일찍
                # 당기면 안 된다 — 성문이 좁아지는 순간 직렬 저항의 무게중심이
                # 성문으로 옮겨가서, 협착(Ac)은 그대로인데 그 **뒤**의 구강내압이
                # 무너진다. 측정(내전을 0.40 지점에서 시작했을 때):
                #   60 ms  Ag 0.080  Pm 0.313  마찰음 0.79
                #   80 ms  Ag 0.034  Pm 0.091  마찰음 0.18   <- Ac 는 아직 0.10
                # 즉 혀가 협착을 풀기도 전에 /s/ 가 죽고 발성이 붙어버린다.
                # 실제 /s/ 는 성문을 벌린 채로 내는 소리이고, 마찰음을 끝내는 건
                # 후두가 아니라 **혀**다.
                #
                # 이렇게 두면 순서가 맞는다: 마찰음 정점 100 ms -> 해제 110 ms
                # -> 발성 140 ms (VOT 30 ms, 한국어 평음 ㅅ 의 25~40 ms).
                # 발성 개시 시점은 여전히 구강내압이 정한다 — 협착이 열려 Pm 이
                # 빠져야 (Ps-Pm) 이 역치를 넘는다.
                #
                # 마찰음과 목소리를 잇는 건 마찰음의 꼬리가 아니라 **기식**이다.
                # 실측에서도 고역이 마찰음 끝에서 0.14 까지 떨어졌다가 발성과
                # 함께 0.33 으로 **다시 오른다** — 그건 마찰음 잔향이 아니라
                # 성대가 덜 모인 상태의 기식성 발성이다.
                # 내전 시작은 **해제 시점** 기준이다(혀 제스처면 hold_r 이 곧
                # 최소 면적 순간). split 에 걸면 혀 제스처에서 어긋난다.
                lag = (float(seg["adduction_start_ratio"]) * split - hold_r
                       if "adduction_start_ratio" in seg else 0.0)
                # 내전은 **두 단계**다. /s/ 는 성문을 벌린 자세라, 모음으로 갈 때
                # 성대가 한 번에 모달 위치로 가지 않는다. 먼저 발성이 가능한 정도만
                # 빠르게 모이고(VOT 를 정한다), 그 뒤로 100 ms 남짓에 걸쳐 마저
                # 모이고 조여진다. 그 사이 성문이 아직 벌어져 있어 모음 첫머리가
                # **기식성**으로 나온다 — 마찰음과 목소리가 섞이는 구간이 이것이다.
                # 한 단계로 두면 목소리가 이미 다 닫힌 성문에서 시작해 잡음 없이
                # 툭 튀어나온다(성문파열음).
                # 끝값이 중요하다. 0.03 cm²(=3 mm²)는 **기식성** 성문이지 모달이
                # 아니다. 거기서 멈추면 성문 제트속도가 압력만으로 정해져 있어
                # (v=sqrt(2Ps/ρ), 면적과 무관) 레이놀즈수가 계속 임계 위에 남고,
                # 기식이 모음 내내 최대로 켜져 있다(측정: 전 구간 0.99). 모달은
                # 성대가 주기마다 완전히 닫혀 후두 틈만 남는 상태라 0.004 cm² 쯤이고,
                # 그제서야 Re 가 임계 아래로 내려가 기식이 꺼진다.
                firm = float(seg.get("firming_s", 0.12)) * a.frame_rate / max(t, 1)
                # **마찰음의 성문 개대는 크다.** Löfqvist & Yoshioka(1984): 성문
                # 개대 진폭은 마찰음이 파열음보다 크고, 시점도 이르다. 개대가
                # 좁으면 마찰음 길이와 기식 구간이 서로 묶인다 — 성문을 오래
                # 열어 두어 발성을 늦추려 하면 유량이 커져 마찰음까지 길어진다.
                # 넓게 열면 직렬 저항이 협착에 몰려서 마찰음은 협착이, 발성
                # 시점은 후두가 따로 정한다.
                abd = float(seg.get("abduction_area", 0.25))
                ag_curve = seg.get("glottal_area", [
                    [0.0, abd], [min(hold_r + lag, 1.0), abd],
                    [min(hold_r + lag + addt, 1.0), 0.03],      # 발성 시작: 기식성
                    [min(hold_r + lag + addt + firm, 1.0), 0.004],   # 모달로 조여짐
                    [1.0, 0.004]])
                seg2 = dict(seg)
                seg2.setdefault("a_open", 3.0)
                seg2.setdefault("hold_ratio", hold_r)
                if _cv_area is not None:
                    # 혀 제스처 궤적을 그대로 넘긴다. 폐압은 aero_drive 가
                    # drive="tongue" 를 보고 일정하게 잡는다.
                    seg2["constriction_area"] = _cv_area
                else:
                    seg2.setdefault("release_s", 0.05)  # 혀끝은 빨리 떨어진다
                    # 호흡 제스처를 마찰음 구간에 맞춰 시간축을 축소한다(비율이
                    # 불변량이라 split 을 곱하면 된다). 모음에는 압력이 계속
                    # 필요하므로 0 으로 안 떨어뜨리고 sustain 에서 유지한다.
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
                             route_noise=False, frame_rate=a.frame_rate)
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
                aero_tilt = drive["src_tilt"]
                aero_teeth = drive["teeth"]
                asp_share = drive["asp_share"]
                # 기식을 손으로 게이팅하지 않는다. 예전엔 (1-v_frac)·(1-env) 를
                # 곱해서 **발성이 붙는 순간 잡음을 껐다**. 그래서 마찰음과 목소리가
                # 한 프레임도 겹치지 않고(실측 겹침 180~260 ms, 그때 합성 0 ms),
                # 목소리가 무음에서 툭 시작해 성문파열음처럼 들렸다.
                #
                # 실제로는 발성이 난류를 없애지 않는다. 성문이 주기마다 열리는 동안
                # 공기는 계속 지나가므로, 성대가 아직 덜 모인 모음 첫머리는
                # **유성 + 난류가 동시에** 난다(기식성 발성). 그걸 줄이는 건 발성이
                # 아니라 **내전**이다. 성문 난류는 이미 성문 면적에 대한 레이놀즈
                # 게이트로 계산돼 있으니(aspiration_source_amp), 물리에 맡긴다.
                # (1-env) 도 이중계상이었다: 협착이 좁을 때 성문 유속이 낮다는 건
                # 직렬 유량 u 에 이미 들어 있고, asp 는 그 u 로 계산된다.
                # 4.0 -> 0.8. 예전 값은 기식이 (1-v_frac)·(1-env) 두 게이트로
                # 좁은 창에만 남던 시절에 맞춘 것이다. 게이트를 물리로 바꾼 뒤
                # 그대로 두었더니 /s/ 구간에서 기식이 치찰음을 덮어(기식 진폭
                # 0.365 > 마찰음 0.174), 마찰음 게인을 아무리 낮춰도 음절의
                # 모음-마찰음 비가 +7.8 dB 에서 안 올라갔다.
                asp = asp_env * float(seg.get("aspiration", 0.8))
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
                if phone in OBSTACLE_SIBILANTS and seg.get("obstacle_dipole", True):
                    nb = nb * _dipole_jet_correction(
                        NB, a.sample_rate, float(seg.get("dipole_db_oct", 10.0)),
                        aero_teeth)
                c["noise_bands"] = (nb * env).contiguous()
                c["noise_entry"] = torch.full((1, t, 1), float(K) + 6.0)
                # 성문 잡음은 **두 가지**이고 발성에 대한 의존이 서로 반대다.
                #
                #  (1) 전이 기식 (asp_n, 900 Hz 셸프): 성문이 벌어진 채 기류만
                #      지나가는 동안의 난류. 발성이 자리잡으면 성대가 주기마다
                #      완전히 닫혀 직류 기류가 끊기므로 사라진다.
                #  (2) 기식성 바닥 (3500 Hz 셸프): 발성 **중에** 성대가 덜 닫혀
                #      새는 잡음. 발성에 비례해 커진다.
                #
                # 정적 성문 면적으로는 (1) 이 저절로 안 꺼진다. 성문 제트속도는
                # v=sqrt(2Ps/ρ) 로 압력만이 정하고 면적과 무관해서, 성문을
                # 0.004 cm² 까지 조여도 진폭이 /s/ 때(3103)와 거의 같다(3169).
                # 주기적 완전폐쇄를 정적 면적이 표현 못 하기 때문이다. 그 몫을
                # 발성 비율로 대신 준다.
                #
                # (1) 만 있으면 모음이 900 Hz 셸프 잡음에 잠겨 1~2 kHz 가 42 %
                # 가 된다(실측 4.2 %). (2) 만 있으면 마찰음과 발성 사이가 빈다.
                # 바닥은 **유성 진폭 자체**에 비례한다(정규화된 비율이 아니라).
                # 새는 잡음은 기류에서 나오므로 소스가 세면 같이 세야 비율이
                # 유지된다. 비율로 스케일하면 음절처럼 유성 진폭이 1.0 보다
                # 작을 때 잡음만 상대적으로 커진다(측정: 4~6 kHz 가 7.2 %,
                # 목표 1.9 %).
                vfrac = (c["harmonic_amp"]
                         / c["harmonic_amp"].amax().clamp_min(1e-6)).clamp(0.0, 1.0)
                # (1) 전이 기식에는 **직렬 저항 배분**을 곱한다. 성문 난류는
                # 성문에 걸리는 압력강하에서 나오고, /s/ 처럼 구강 협착이 성문
                # 보다 좁으면 그 몫이 41 % 뿐이다. 이걸 빼먹으면(= asp 를 제
                # 최대값으로만 정규화하면) 마찰음 구간 내내 기식이 제 세기로
                # 나오고, 그게 성도 캐스케이드를 지나 F1/F2 로 나와 /s/ 정점의
                # 36 % 가 1~2 kHz 에 쌓인다 — 봉우리가 10 kHz 가 아니라 1.7 kHz
                # 로 잡혔다. 협착이 풀리면 배분이 1 로 가므로 전이에서는 기식이
                # 제 세기를 되찾는다(전이를 메우는 건 여전히 기식이다).
                c["aspiration_bands"] = (
                    asp_n * asp * asp_share * (1.0 - vfrac)
                    + c["aspiration_bands"] * c["harmonic_amp"]).contiguous()
            else:
                w_asp = torch.sigmoid((asp - env) * 8.0)
                c["noise_bands"] = (nb * env + asp_n * asp * 0.5).contiguous()
                c["noise_entry"] = (1.0 - w_asp) * (float(K) + 6.0)
            # 포먼트 궤적은 **로커스 이론**대로. 유성 구간을 /이/ 에서 시작해
            # 모음으로 미끄러뜨리면 그건 /j/ 활음이라 "사" 가 "야" 로 들린다.
            # 치경 로커스에서 출발해 짧은 전이(기본 50 ms)로 모음에 도달한다.
            tgt = _vowel_formants(vowel, prof, K)
            loc = LOCUS.get(phone, LOCUS["s"])
            # 45 ms 로 두면 발성이 붙을 때 F2 전이가 72 % 밖에 안 끝나서,
            # 남은 128 Hz 활강이 유성 구간 안에서 들린다 = /j/ = "야".
            # 32 ms 면 97 % 가 끝나 남는 활강이 30 Hz 남짓이라 안 들린다
            # (실측: 발성 개시 +2 ms 에 F2 가 이미 목표에 있다).
            trans = float(seg.get("transition_s", 0.032)) * a.frame_rate / max(t, 1)
            # 전이 시작은 **해제 시점**이다. 발성 개시에 묶으면 안 된다 —
            # 후두 타이밍을 건드릴 때마다 포먼트 궤적이 같이 끌려다닌다.
            # (실제로 내전을 앞당겼더니 전이가 30 ms 일찍 시작됐고, 그 활강이
            #  유성 구간 안으로 들어와 "사" 가 "야" 로 들렸다.)
            # 혀는 협착을 푸는 순간 움직이기 시작하고, 발성은 구강내압이 빠진
            # 뒤에 붙는다. 그래서 소리가 날 때쯤 전이는 이미 끝나 있어야 한다.
            # 혀는 **협착을 푸는 순간**부터 움직인다. 발성은 그보다 늦게 붙는다
            # (구강내압이 빠져야 성대가 떨 수 있으므로). 그래서 F1 도 F2/F3 도
            # 전이를 **해제 시점**에서 출발시킨다.
            #
            # F2 만 발성 시작에 맞춰 두었더니 "사" 가 "샤" 로 들렸다. 그러면 F2 가
            # 로커스에 머문 채로 유성이 시작되고, 그 뒤 100 ms 동안 1292 까지
            # 내려가는 활강이 통째로 들린다 - 고 F2 에서 저 F2 로의 유성 활강은
            # 정의상 /j/ 다. 실제로는 해제와 함께 혀가 이미 움직이고 있어서,
            # 소리가 붙을 때는 F2 전이가 거의 끝나 있다.
            onset_f1 = onset_f2 = hold_r if wants_aero(seg) else split
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
            _scale_sib_center(sib, aero_cent, aero_tilt, aero_teeth)
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
                   curve(spec.get("adduction", 1.0), t),
                   frame_rate=cfg.audio.frame_rate)
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
    # **여기(무음 구간)에서는 noise_entry 를 뒤 세그먼트 값으로 채운다.**
    # 위 주석대로 보간은 안 되지만, 여기 있는 건 보간이 아니라 **점프 제거**다.
    # 기류가 없는 구간에서는 주입 위치가 소리에 아무 영향이 없으므로, 뒤에서
    # 쓸 값을 미리 넣어 두면 경계에서 필터가 안 튄다.
    #
    # 안 하면 무음(0) -> /s/(K+6=18) 로 한 프레임에 뛰면서 성도 응답이 통째로
    # 바뀌고, 그 불연속이 클릭이 된다 — 사용자가 "ksa" 로 듣던 그 /k/ 다.
    # 측정(무음 + '사'): 경계 직후 1 ms 최대진폭 0.648 -> 0.544 (이것만으로),
    # sib.mix 램프까지 같이 하면 0.278. 실측 녹음의 개시는 0.0054 다.
    if "noise_entry" in merged and "noise_bands" in merged:
        ne = merged["noise_entry"]
        amp = merged["noise_bands"].amax(dim=-1, keepdim=True)
        if "harmonic_amp" in merged:
            amp = amp + merged["harmonic_amp"]
        quiet = (amp < 1e-3)[0, :, 0]
        idx = torch.arange(ne.shape[1])
        loud = (~quiet).nonzero().reshape(-1)
        if len(loud):
            # 각 무음 프레임을 **뒤쪽 첫 유음 프레임**의 값으로 채운다.
            nxt = torch.searchsorted(loud, idx.clamp_max(int(loud[-1])))
            src = loud[nxt.clamp_max(len(loud) - 1)]
            ne = torch.where(quiet.reshape(1, -1, 1), ne[:, src, :], ne)
            merged["noise_entry"] = ne

    n = int(score.get("smooth_frames", 2))
    for k in ("f0", "rd", "tilt", "formant_freq", "formant_bw", "harmonic_amp",
              "noise_bands", "noise_am", "noise_rough"):
        if k in merged:
            merged[k] = _smooth(merged[k], n)
    # 치찰음 필터는 **켤 때 램프**한다. mix 는 항등응답과의 크로스페이드라
    # (sibilant_response: (1-m)+m·H) 중간값이 물리적으로 정의된다 — noise_entry
    # 와 달리 미끄러뜨려도 되는 양이다. 한 프레임에 0->1 로 뛰면 성도 응답이
    # 급변해 경계에서 클릭이 난다.
    if sib_fields.get("mix") is not None and n > 0:
        sib_fields["mix"] = _smooth(sib_fields["mix"], max(n, 4))

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
