"""시간 영역 경로 — 성문 면적 · 성도 면적 → `tube_td` (MEASUREMENTS §52.476).

물리 사슬 순서 (사용자: *"모듈 순서 잘 고려해보고"*):

    제어열 (프레임률) ─ 성문 생리 (`GlottalSource.physiology`: f0·진폭·Rd·정적 면적) ─ 성문 면적 A_g(t) ─┐
                     └ 조음 (art1..6·tract_len → 면적 함수, a_c·c_place 협착, oral_open 입술, velum 포트) ─┴─ tube_td (96 kHz)
                                                                                                         ─ 저역통과·솎기 (48 kHz) ─ [방·녹음은 적합기]

성문은 **배음 표를 더하지 않는다**(예전 LF 조화 근사). 성대 변위(매끈한 주기 함수)와 쉼 틈만 정하고, 닫힘은 **접촉**(면적이 0 에서
잘림)이, 유량은 관과 함께 비선형으로 풀린다 — 열림 몫·닫힘의 꺾임·유량의 기울임·주기별 모양 변화·음원-성도 결합이 거기서 나온다. 난류는 관 안의 레이놀즈 조건이 낸다 — 따로 더하는 잡음원이 없다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import tube_td as td
from .articulation import PAR_NAMES as ART_NAMES, MaedaLAM, SpeakerScale, uniform_tube
from .control import frames_to_samples
from .tube import N_ART, cos_area

#: **관 모양을 내는 방식** (§52.485). "cos": 틀마다 자유 코사인 곡선 + 가우스 협착 + 입술 배율 (옛 곡선 맞춤, `area_frames`).
#: "maeda": 조음기 일곱(`control` 의 jaw..larynx) → Maeda 조음 모형 → 면적 함수와 **성도 길이** (`tract_len`·`a_c`·`c_place`·`oral_open`
#: 은 쓰지 않는다 — 길이는 후두 높이·입술 돌출이, 협착·폐쇄는 혀·입술이 낸다). 원 화자(남성) → 이 화자는 화자 상수 배율 셋.
ARTIC = "cos"
#: "w02": VocalTractLab 여성 화자 W02 의 해부학 변수 (`control` 의 vtl_*) → 미분 가능한 대리 모형 (`articulation.W02Surrogate`,
#: `profiles/w02_surrogate.pt`) → 면적 함수·성도 길이 (§52.486). 대리 모형의 무게는 얼린다 (적합 대상이 아니다).
_W02 = None
#: **해부 적응** (§52.501, `copyfit --w02-anat`): W02 대리 모형 대신 해부 13 개(VTL `AnatomyParams`)까지 받는 대리 모형
#: (`articulation.W02AnatSurrogate`) — 이 화자의 해부(입술 폭·하악·구개·인두·후두 길이·폭 …)를 화자 상수로 적합한다.
W02_ANAT = False
N_W02_ANAT = 13
#: **나이 축 해부** (§52.523, `copyfit --w02-age`, `W02_ANAT` 위에서). 해부 13 개를 따로 풀면 적합이 생리적으로 어긋난 조합으로 한계에 붙었다
#: (038 KWF_q1: 후두 길이 2.00 cm = VTL 성장식 10 세, 하악 0.90 cm = 1 세, 구강–인두 각 −104°). 대신 W02(독일 성인 여성 MRI) 해부를
#: VTL `AnatomyParams::calcFromAge` 의 **여성 성장식**(Goldstein 1980)을 따라 나이로 옮긴다: 길이 a_i = W02_i · G_i(나이)/G_i(W02_AGE_REF),
#: 각 a = W02 + G(나이) − G(기준). 나이는 [W02_AGE_LO, W02_AGE_HI] 의 화자 상수 하나 — 성인 여성 성도 길이 14.0–17.5 cm(Story et al. 2018 CT,
#: 18–25 세 여성 10 명, 15.6 ± 1.14 cm)의 아래 끝이 여성 성장 곡선의 12–13 세에 해당한다. 동아시아 여성은 인두가 작다(Xue & Hao 2006).
#: 개인차는 길이 ±W02_DEV(성인 여성 성도 길이 변동계수 7.3 % 의 두 배), 각 ±W02_DEV_DEG 의 tanh 좌표 (`w02d_u*`, 0 = 성장 곡선 위).
W02_AGE = False
W02_AGE_LO, W02_AGE_HI, W02_AGE_REF = 12.0, 25.0, 22.0
W02_DEV, W02_DEV_DEG = 0.15, 6.0


def _os_env_get(k: str) -> str:
    import os
    return os.environ.get(k, "")


def goldstein_female(t: torch.Tensor) -> torch.Tensor:
    """VTL `AnatomyParams::calcFromAge` 의 여성 식 (Goldstein 1980) — 나이 t [년, ≥ 7] → 해부 13 개 (VTL 차례) [cm·도]. 미분 가능."""
    t = t.double()
    E = torch.exp
    angfhj = 26.856 - 0.170 * t
    angfhv = 4.383 + 0.203 * t                                   # 30 개월 이상
    angpj = 27.85 - 0.401 * t
    anspns = (44.414 / (1 + E(-1.210 - 0.699 ** 2 * t)) + 7.807 / (1 + E(5.218 - 0.727 ** 2 * t))) / 10
    atlptm = (27.727 + 0.094 * t) / 10
    hyglott = (11.143 + 0.298 * t - E(1.4 - 1.235 * t)) / 10
    lenvc = (5.997 + 0.39 * t - E(0.804 - 2.435 * t)) / 10
    lips = (22.07 - 13.41 * E(-0.041 * t)) / 10
    palhit = (17.813 - 8.859 * E(-0.077 * t)) / 10
    palwi = (32.709 - 7.378 * E(-0.174 * t)) / 10
    snhy = (64.002 / (1 + E(-0.621 - 1.143 ** 2 * t)) + 38.254 / (1 + E(3.06 - 0.597 ** 2 * t))) / 10
    snpns = (22.865 / (1 + E(-0.507 - 1.326 ** 2 * t)) + 21.141 / (1 + E(1.35 - 0.519 ** 2 * t))) / 10
    symph = (23.411 / (1 + E(-1.310 - 0.626 ** 2 * t)) + 4.481 / (1 + E(23.061 - 1.369 ** 2 * t))) / 10
    thihy = (5.633 + 0.185 * t) / 10
    ang = torch.clamp(-(angfhv + angfhj - angpj + 90.0) + 8.0, max=-90.0)
    s70, s120 = math.sin(70 * math.pi / 180), math.sin(120 * math.pi / 180)
    half = torch.full_like(t, 0.5)
    return torch.stack([lips - 0.5, symph * s70 - 1.0, half, half, palhit, palwi, anspns - 1.0, atlptm - 0.2,
                        snhy - snpns - 0.5 * thihy - 0.2, hyglott * s120 + thihy + 0.1, atlptm - 0.2, lenvc, ang], -1)
#: W02 조음기의 생리 시상수 [ms] (목표 근사 모형, Birkholz et al. 2011 — `copyfit` 의 움직임 사전과 같은 표). 조음기는 질량·근육 동역학이 있어 1 ms
#: 에 뛸 수 없다: 조음 변수를 성도 모형에 넣기 전 σ = `W02_SMOOTH` × 시상수 의 가우시안으로 고른다 (§52.502). 없으면 1 ms 격자 적합의 조음이
#: 틀마다 떨어 면적 log rms 0.02–0.17 (곡선 조음 0.003–0.01) — 부피 변화(dV/dt)가 거친 소리가 되어 갈라짐 1.02–1.15 (원본 0.44–0.63). 0 이면 끈다.
W02_TAU = {"HX": 40.0, "HY": 40.0, "JX": 25.0, "JA": 25.0, "LP": 20.0, "LD": 12.0, "TCX": 15.0, "TCY": 15.0, "TTX": 10.0,
           "TTY": 10.0, "TBX": 12.0, "TBY": 12.0, "TRX": 20.0, "TRY": 20.0, "TS1": 15.0, "TS2": 15.0, "TS3": 15.0}
W02_SMOOTH = 0.5
#: `W02_SMOOTH` 의 생리 값 (§52.530, `copyfit --physio-dyn`). 위 시상수는 Birkholz 2011 목표 근사 모형(임계 감쇠 3 차, 극 셋이 −1/τ)의 것이다 —
#: 그 모형의 −3 dB 대역은 (1 + (ωτ)²)^{3/2} = √2 에서 f = 0.081/τ (입술 벌림 τ 12 ms → 6.8 Hz), 같은 대역의 가우시안은 σ = 1.6 τ (10–90 % 오름
#: 2.56 σ ≈ 4.2 τ 도 같다). 0.5 τ 는 대역이 3 배 넓어 입술 칸이 모음 안에서 ~20 Hz 로 8 % 흔들렸다 (KWF_m13, `out/_tmp/vf/mod_audit.py`).
W02_SMOOTH_TAM = 1.6
#: **음소별 입술 범위를 관에 단단히** (§52.527, `copyfit --lip-bounds`). 사용자: *"입술 관련 근육은 잘 반영됐나?"* — 벌점(`fit.ART_VOWEL_LP` ·
#: 입술 열림 하한)은 ~900 틀의 평균이라 몇 틀의 위반이 묻혔고, 무성화된 '시' 의 iː 는 모음 명세 대신 마찰 명세를 받아 입술이 0.15 cm² 로 닫혔다 (KWF_m10).
#: 생리 한계는 매개변수로 막는다 ([[fit-within-physiology-hard-limits]]): 생리 평활 뒤 값을 음소별 범위로 자른다 (기울기는 범위 안에서만).
#: 범위 (VTL 단위, W02 MRI 자세와 VTL 입술 면적에서): 모음 가운데 60 % — 평순 내밂 [−0.3, 0.3] · 위아래 거리 LD ≥ 0.2 (입술 끝 ~0.5 cm²), 원순 내밂
#: [0.4, 1.0] · LD ≥ 0.1 (모음 가운데 80 %); 양순음이 아닌 자음 — LD ≥ 0.15. 무성화 모음도 모음. 경계는 4 ms 가우시안으로 고르게 (생리 평활 뒤라
#: 입술 궤적은 이미 조음기 시상수로 고르다).
W02_LIP_BOUNDS = False
#: **입 장애음에서 연인두를 닫는다 — 단단히** (§52.530, `copyfit --velum-bounds`, `--phoneme-targets` 와 함께). m13 의 ㄲ 폐쇄(0.69–0.73 s)에서 적합이
#: 연인두 통로를 0.09–0.12 cm² 로 열어 150–340 cm³/s 가 코로 빠졌다 — 입 안 압력이 안 올라 성대가 폐쇄 내내 떨고 코로 소리가 났다 (1.5–10 kHz 원본보다
#: +5 ~ +15 dB). 0.74 s 에 통로가 닫히자 입 안 압력이 곧 폐압에 닿아 흐름이 멎었다. 입 장애음(파열 · 마찰 · 파찰, 된소리 · 거센소리 · 유성 변이음)은
#: 연인두가 닫혀야 입 안 압력이 선다 (Warren et al. 1989 — 통로 < 0.05 cm² 가 적절한 닫힘, 정상 화자는 거의 0). 그 구간의 통로 상한
#: `VELUM_OBS_MAX` cm² (0.005 — 7 cmH2O 에서 새는 유량 ≈ 17 cm³/s), 장애음 앞 `VELUM_RAMP_MS` 동안 열림 → 닫힘으로 비스듬히 (연구개는 10–15 cm/s,
#: 닫는 데 수십 ms — 장애음보다 먼저 닫는다), 뒤도 같은 폭으로 풀린다. ㅎ 은 뺀다 (입 안에 협착이 없다).
VELUM_BOUNDS = False
VELUM_OBS_MAX = 0.005
VELUM_RAMP_MS = 20.0
VELUM_NAS_FRAC = 0.5       # 비음 뒤 장애음: 폐쇄의 앞 이 몫 안에 연구개를 닫는다 (§52.533; 1 이면 예전 — 개방에 닫힘)
_VELB_CACHE: dict = {}


def _velum_hi(T: int, hop_s: float):
    """연인두 제어 `velum` 의 상한 (T,) [0–1] — `VELUM_BOUNDS`. 음소 정렬이 없으면 None."""
    import numpy as np
    from . import fit as _fit
    if _fit.ART_SEGMENTS is None or _fit.ART_LABELS is None:
        return None
    key = (T, round(hop_s, 9), len(_fit.ART_SEGMENTS), VELUM_OBS_MAX, VELUM_RAMP_MS)
    if key in _VELB_CACHE:
        return _VELB_CACHE[key]
    n_al = int(round(max(t1 for _, t1, _ in _fit.ART_SEGMENTS) / hop_s)) + 1
    hi = np.ones(n_al)
    vmax = VELUM_OBS_MAX / td.A_VMAX
    nr = max(1, int(round(VELUM_RAMP_MS * 1e-3 / hop_s)))
    labs = [str(l) for l in _fit.ART_LABELS]
    for sk, ((t0, t1, shp), lab) in enumerate(zip(_fit.ART_SEGMENTS, labs)):
        if not lab or lab in ("sil", "sp"):
            continue
        obs = lab[0] in "pbtdkgɡsɕʃzʑcɟ" or lab.startswith(("tɕ", "dʑ", "ts", "dz"))
        if not obs:
            continue
        i0, i1 = int(round(t0 / hop_s)), int(round(t1 / hop_s))
        r = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, nr + 1) / (nr + 1)))          # 0 → 1
        if sk > 0 and labs[sk - 1].replace("ː", "") in ("m", "n", "ŋ", "ɲ"):
            # **비음 바로 뒤** (§52.531): 연구개는 비음에서 열려 있다가 장애음 폐쇄 동안 닫혀 간다 (닫는 데 수십 ms) — 038 의 ɲ → dʑ 는 폐쇄 중에도
            # 원본 저역이 모음과 같고 (−27 ~ −36 dB) 0.5–2 kHz 만 15 dB 꺼진 비음 웅얼이었다. 장애음 시작에 닫으면 폐쇄가 통째로 무음이 됐다
            # (m20 저역 −54 ~ −63 dB). 폐쇄 구간 동안 열림 → 닫힘으로 비스듬히.
            # **닫는 곳은 폐쇄의 앞 `VELUM_NAS_FRAC`** (§52.533): 개방(구간 끝)에 닫았더니 m25 의 dʑ 폐쇄 동안 코 유량이 개방 바로 전까지 100–341 cm³/s 로 새어
            # 입 안 압력이 안 차고 개방의 마찰 터짐이 없었다 (10–16 kHz 600–610 ms 원본 −58 ~ −62 dB, 합성 −88: −27 ~ −31 dB). 연구개는 앞 비음 동안 오르기
            # 시작해 구강 폐쇄의 앞부분에서 닫힌다 (Moll & Daniloff 1971 — 예기 동작); 닫힌 관은 2 ms 에 압력 6.9 cmH2O (§52.531 closed_tube).
            m_ = max(1, i1 - i0)
            mc = max(1, int(round(VELUM_NAS_FRAC * m_)))
            rr = np.ones(m_)
            rr[:mc] = 0.5 * (1.0 - np.cos(np.pi * np.arange(mc) / mc))             # 0 → 1 (앞 몫 안에서), 그 뒤 닫힘
            hi[i0:i1] = np.minimum(hi[i0:i1], 1.0 - (1.0 - vmax) * rr)
        else:
            hi[i0:i1] = np.minimum(hi[i0:i1], vmax)
            a0 = max(0, i0 - nr)
            seg = vmax + (1.0 - vmax) * r[::-1][nr - (i0 - a0):]                      # 장애음 앞: 1 → vmax
            hi[a0:i0] = np.minimum(hi[a0:i0], seg)
        b1 = min(n_al, i1 + nr)
        seg = vmax + (1.0 - vmax) * r[:b1 - i1]                                       # 뒤: vmax → 1
        hi[i1:b1] = np.minimum(hi[i1:b1], seg)
    if T > n_al:
        hi = np.concatenate([np.full(T - n_al, hi[0]), hi])
    else:
        hi = hi[:T]
    _VELB_CACHE[key] = hi
    return hi


#: **공명음의 가성대 틈 하한** (§52.532, `copyfit --ff-bounds`). 여성의 모달 · 가성 발성에서 가성대 틈은 2.0–7.0 mm (남성 2.3–8.3; Agarwal, Scherer & Hollien
#: 2003 J Voice 17 — 단층 촬영 정면 상). 틈을 타원 (너비 × 성대 길이) 으로 셈해 면적 하한 π/4 · `FF_GAP_MIN_MM` · L. 3 mm 로 좁혀야 성문 음원이 눈에 띄게
#: 바뀐다 (3 차원 발성 모형, Zhang 2025 PMC11968131). m13 · m20 · m21 · n8 은 모음 틀의 45 · 45 · 58 · 78 % 가 이 하한 밖이었다 (n8 중앙 0.095 cm², 하위 10 %
#: 0.032 — 거의 닫힘): 적합이 가성대를 성량 · 밝기 보상에 썼고, m21 의 ㄲ 뒤 ʌ 에서 가성대 0.10–0.13 cm² 가 참성대 최대 열림 (0.13) 보다 좁아 성문 위 압력이
#: 7.8 cmH2O 까지 차 떨림을 깎았다. 모음 · 활음 · 비음 · 유음 구간에 단단히 건다 (가성대 면적 = 하한 + δ·softplus((a − 하한)/δ), δ = 하한의 5 % — 하한 아래로는
#: 못 가고 위에서는 그대로, 매끈). 장애음 (된소리의 후두 긴장 등) 과 무음은 풀어 두고, 구간 앞뒤 `FF_RAMP_MS` 는 구간 안쪽으로 비스듬히.
FF_BOUNDS = False
FF_GAP_MIN_MM = 2.0
FF_RAMP_MS = 15.0
_FFB_CACHE: dict = {}


def _ff_lo_w(T: int, hop_s: float):
    """가성대 틈 하한의 가중 (T,) [0–1] — `FF_BOUNDS`. 음소 정렬이 없으면 None."""
    import numpy as np
    from . import fit as _fit
    if _fit.ART_SEGMENTS is None or _fit.ART_LABELS is None:
        return None
    key = (T, round(hop_s, 9), len(_fit.ART_SEGMENTS), FF_RAMP_MS)
    if key in _FFB_CACHE:
        return _FFB_CACHE[key]
    n_al = int(round(max(t1 for _, t1, _ in _fit.ART_SEGMENTS) / hop_s)) + 1
    w = np.zeros(n_al)
    nr = max(1, int(round(FF_RAMP_MS * 1e-3 / hop_s)))
    son = set("aeiouɨʌɛəɯøyœwjɐmnŋɲlɾr")
    for (t0, t1, shp), lab in zip(_fit.ART_SEGMENTS, [str(l) for l in _fit.ART_LABELS]):
        if not lab or lab in ("sil", "sp") or lab[0] not in son:
            continue
        i0, i1 = int(round(t0 / hop_s)), int(round(t1 / hop_s))
        m_ = max(1, i1 - i0)
        k = min(nr, m_ // 2)
        r = np.ones(m_)
        if k > 0:
            q = 0.5 * (1.0 - np.cos(np.pi * (np.arange(k) + 0.5) / k))
            r[:k] = np.minimum(r[:k], q)
            r[m_ - k:] = np.minimum(r[m_ - k:], q[::-1])
        w[i0:i0 + m_] = np.maximum(w[i0:i0 + m_], r[:max(0, min(m_, n_al - i0))])
    # 이웃한 두 공명음 사이 (모음 → 비음 등) 는 비스듬히 내려가지 않게 잇는다
    for (ta, tb, _), (tc, td_, _), la, lb in zip(_fit.ART_SEGMENTS[:-1], _fit.ART_SEGMENTS[1:], _fit.ART_LABELS[:-1], _fit.ART_LABELS[1:]):
        la, lb = str(la), str(lb)
        if la and lb and la[0] in son and lb[0] in son and la not in ("sil", "sp") and lb not in ("sil", "sp"):
            j0, j1 = max(0, int(round(tb / hop_s)) - nr), min(n_al, int(round(tb / hop_s)) + nr)
            w[j0:j1] = 1.0
    if T > n_al:
        w = np.concatenate([np.zeros(T - n_al), w])
    else:
        w = w[:T]
    _FFB_CACHE[key] = w
    return w
#: **과제별 사용 범위** (§52.527) — 사용자: *"말할 때 입술을 그리 크게 벌리진 않잖아? … 노래한다든지 하는 경우엔 입을 크게 벌릴 수도 있긴 해."*
#: "speech": W02 MRI 말소리 자세(여성, 모음·자음 52 개)의 범위 × 1.2 — 입술 위아래 거리 LD ≤ 1.5 (MRI 최대 /a/ 1.27, 입술 끝 3.5 cm²),
#: 턱 각 JA ≥ −5.6 (MRI 가장 벌린 −5.07). "sing": VTL 전체 범위. KWF_m10 은 ʌ · ɲ · dʑ 에서 입술 끝 5.3–6.2 cm² 로 노래만큼 벌렸다.
#: **더 크게 벌린다** (§52.534). 사용자 (KWF_m25vn 을 듣고): *"입술을 더 크게 벌리도록 하고, 턱도 더 크게 벌릴 수 있도록 제약을 넣어"*. 적합도 같은 쪽을 원했다 —
#: 조음 변수 원값이 모음에서 입술 거리 2.0–2.6 cm · 턱 −6.5 ~ −7.0° 로 말소리 상한 (1.5 cm, −5.6°) 에 붙어 있었다. MRI 자세는 누워서 지속 발음한 모음이라 이어 말의 또렷한
#: 발음보다 작다. 입술 상한 = MRI /a/ 1.27 cm 의 2 배 2.6 cm, 턱은 조음 모형 (VTL) 범위 끝 −7° (그 밖은 대리 모형의 학습 범위 밖). 모음별 입술 하한
#: `VOWEL_LD_MIN` = MRI 모음 자세의 60 % (빠른 말의 덜 도달).
W02_TASK = "speech"
SPEECH_LD_MAX = 2.6
SPEECH_JA_MIN = -7.0
VOWEL_LD_MIN = {"a": 0.75, "ɐ": 0.75, "i": 0.5, "e": 0.4, "ɛ": 0.4, "ʌ": 0.35, "ɨ": 0.35, "ə": 0.35}
_LIPB_CACHE = {}
VOWELS_KO = ("a", "ɐ", "ʌ", "ɔ", "o", "u", "ɯ", "ɨ", "i", "e", "ɛ", "ø", "y", "ʊ", "ə", "æ", "ɘ", "ɤ")
ROUNDED_KO = ("o", "u", "ɔ", "ø", "y", "ʊ", "w")
LABIAL_KO = ("p", "b", "m", "ɸ", "f", "v", "w", "β")


def _lip_bounds(T: int, hop_s: float, name: str):
    """(lo, hi) (T,) — 음소 정렬(`fit.ART_SEGMENTS` · `ART_LABELS`)에서. 앞쪽 틀 수가 정렬보다 많으면(앞 여유 틀) 앞을 첫 값으로 메운다."""
    import numpy as np
    from . import fit as _fit
    if _fit.ART_SEGMENTS is None or _fit.ART_LABELS is None:
        return None, None
    key = (T, round(hop_s, 9), name, len(_fit.ART_SEGMENTS))
    if key in _LIPB_CACHE:
        return _LIPB_CACHE[key]
    n_al = int(round(max(t1 for _, t1, _ in _fit.ART_SEGMENTS) / hop_s)) + 1
    glo, ghi = {"LP": (-1.0, 1.0), "LD": (-1.85, 3.7), "JA": (-7.0, 0.0)}[name]
    if W02_TASK == "speech":                          # 말소리 사용 범위 (위 설명)
        if name == "LD":
            ghi = SPEECH_LD_MAX
        elif name == "JA":
            glo = SPEECH_JA_MIN
    lo = np.full(n_al, glo)
    hi = np.full(n_al, ghi)
    if name == "JA":
        _LIPB_CACHE[None] = None
        lo2, hi2 = lo, hi
        if T > n_al:
            lo2 = np.concatenate([np.full(T - n_al, lo[0]), lo]); hi2 = np.concatenate([np.full(T - n_al, hi[0]), hi])
        else:
            lo2, hi2 = lo[:T], hi[:T]
        _LIPB_CACHE[key] = (lo2, hi2)
        return lo2, hi2
    for (t0, t1, shp), lab in zip(_fit.ART_SEGMENTS, _fit.ART_LABELS):
        lab = str(lab).replace("ː", "")
        if lab in ("", "sil", "sp"):
            continue
        base = lab[0]
        d = t1 - t0
        if base in VOWELS_KO:
            i0, i1 = int(round((t0 + 0.1 * d) / hop_s)), int(round((t1 - 0.1 * d) / hop_s))
            rnd = base in ROUNDED_KO
            if name == "LP":
                lo[i0:i1], hi[i0:i1] = (0.4, 1.0) if rnd else (-0.3, 0.3)
            else:
                lo[i0:i1] = 0.1 if rnd else max(0.2, VOWEL_LD_MIN.get(base, 0.2))
        elif name == "LD" and not any(ch in lab for ch in LABIAL_KO):
            i0, i1 = int(round(t0 / hop_s)), int(round(t1 / hop_s))
            lo[i0:i1] = np.maximum(lo[i0:i1], 0.15)
    from scipy.ndimage import gaussian_filter1d
    sg = 0.004 / hop_s          # 4 ms — 15 ms 는 50–60 ms 모음의 범위를 씻어 냈다 (ɨ 내밂 ±0.46, iː LD ≥ −0.26)
    lo = gaussian_filter1d(lo, sg, mode="nearest")
    hi = gaussian_filter1d(hi, sg, mode="nearest")
    if T > n_al:
        lo = np.concatenate([np.full(T - n_al, lo[0]), lo])
        hi = np.concatenate([np.full(T - n_al, hi[0]), hi])
    else:
        lo, hi = lo[:T], hi[:T]
    _LIPB_CACHE[key] = (lo, hi)
    return lo, hi
#: **조음기 최대 속도** [cm/s, JA °/s, TS 1/s] (§52.506) — 사용자: *"인간의 생리적 범위로만 작동하게 만들어야지. 자꾸 비정상적인 것들까지
#: 다 하도록 하면 어떡해."* 생리 평활(`W02_SMOOTH`)만으로는 관에 들어가는 궤적이 혀뿌리 170–237·혀날 최대 287·입술 벌림 최대 234 cm/s
#: 로 한계의 수~십 배였고 틀의 40–88 % 가 한계를 넘었다 (`out/_tmp/vf/physio_audit.py`). 빠른 말의 최대 조음 속도 (X-ray microbeam·EMA:
#: Kuehn & Moll 1976, Westbury 1994, Tasko & McClean 2004 — 혀끝 20–30, 혀 몸통 10–20, 아래턱 8–15, 입술 15–25 cm/s, 설골·후두 ~5) 를
#: **미분 가능한 속도 제한기**로 강제한다: y_t = y_{t−1} + v·tanh((x_t − y_{t−1})/v), v = 최대 속도 × 틀 간격. 0 이면 끈다.
W02_VMAX = {"HX": 6.0, "HY": 6.0, "JX": 12.0, "JA": 150.0, "LP": 20.0, "LD": 25.0, "TCX": 15.0, "TCY": 15.0, "TTX": 25.0, "TTY": 25.0,
            "TBX": 20.0, "TBY": 20.0, "TRX": 12.0, "TRY": 12.0, "TS1": 10.0, "TS2": 10.0, "TS3": 10.0}
#: 성문·호흡 제어의 최대 변화 속도 [단위/s] — 폐압 (0 → 7 cmH2O 에 ~70 ms), 내전 (외전·내전 ~50 ms), 수렴, 갑상피열근 활성.
VF_VMAX = {"p_sub": 100.0, "adduction": 20.0, "rd_offset": 20.0, "vf_ta": 10.0}
SLEW_ON = True
_W02A = None


def _slew(x: torch.Tensor, v: float) -> torch.Tensor:
    """(B, T) 궤적의 미분 가능한 속도 제한 — 한 틀에 v 보다 크게 못 움직인다 (tanh 포화)."""
    if v <= 0.0 or x.shape[-1] < 2:
        return x
    y = x[..., 0]
    out = [y]
    for t in range(1, x.shape[-1]):
        y = y + v * torch.tanh((x[..., t] - y) / v)
        out.append(y)
    return torch.stack(out, -1)


def _w02a():
    global _W02A
    if _W02A is None:
        import os
        from pathlib import Path
        from .articulation import W02AnatSurrogate
        # 모형 파일을 고정한다 (환경 변수 W02A_PATH) — 학습 중 덮어써지는 파일을 패스마다 다시 읽으면 성도 모형이 판 사이에 바뀐다
        _W02A = W02AnatSurrogate.load(Path(os.environ["W02A_PATH"])) if os.environ.get("W02A_PATH") else W02AnatSurrogate.load()
        for q in _W02A.parameters():
            q.requires_grad_(False)
    return _W02A


#: 협착 난류의 음원 자리를 **틀마다의 앞니 자리**로 (§52.519, `tube_td.NOISE_PLACE` 일 때). W02 해부 적응 조음에서 입술–앞니 거리를 예측기
#: (`articulation.W02TeethNet`, `profiles/w02_teeth.pt`)로 내어 관에 넘긴다 — 예전에는 입술 끝에서 1 cm · 명목 길이 15 cm 고정이었다.
NOISE_TEETH = True
_W02T = None


def _w02t():
    global _W02T
    if _W02T is None:
        from .articulation import W02TeethNet
        _W02T = W02TeethNet.load()
        for q in _W02T.parameters():
            q.requires_grad_(False)
    return _W02T


def _w02():
    global _W02
    if _W02 is None:
        from .articulation import W02Surrogate
        _W02 = W02Surrogate.load()
        for q in _W02.parameters():
            q.requires_grad_(False)
    return _W02

OS = 2                     # 내부 표본률 = 출력 × OS (tube_td.FS_SIM = 96 kHz)
#: **관 해상도 배율** (§52.508, `copyfit --tube-refine R`). 28 칸 · 96 kHz 관은 12–16 kHz 를 깎는다 — 이산 파동의 분산(쿠랑 수 0.68 에서 위상속도
#: 12 kHz −3.5 %, 16 kHz −7 %)과 0.54 cm 계단. 같은 자세를 56 칸 · 192 kHz 로 풀면 1–3 kHz 대비 14–16 kHz 가 +5–6 dB (a · ㅅ 자세), 112 칸 ·
#: 384 kHz 와는 0.5 dB 안 — 56 칸에서 수렴한다. 칸과 표본률을 함께 R 배로 늘려 쿠랑 수를 지킨다. 조음 대리 모형은 28 칸을 내므로 `_refine_area`
#: (극값 보존 보간)로 늘린다. 1 이면 예전과 같다.
TUBE_REFINE = 1
#: 성대 긴장의 주기 규모 무작위 요동 (§52.515, `VoiceTD._vf_jitter`): 로그 긴장의 표준편차 σ 와 요동의 대역 [Hz]. 0 이면 끈다.
VF_JIT_SIGMA = 0.0
#: **신경 작용 — 후두근 운동 단위 요동** (§52.523, `physics/neural.py`, Titze 1991). 켜면 윤상갑상근(CT) 긴장의 운동 단위 합(평균 1)을 표본률로 덮개 긴장
#: Q 에, 갑상피열근(TA) 긴장을 프레임률로 갑상피열근 활성 `vf_ta` 에 곱한다. 단위 수 `VF_NEURAL_NMU` (200 이면 긴장 σ CT 1.6 · TA 2.6 %, 그 36–44 % 가
#: 4–12 Hz 떨림). 실현은 판의 씨앗으로 고정. 모드 3 (보–막) 에서만.
VF_NEURAL = False
#: **성문·협착 난류의 세기 계수를 하나로** (§52.524). 둘 다 같은 Stevens 꼴의 제트 난류다 (Stevens 1998 — 기식과 마찰은 같은 음원 식, 장애물 기하만 다르다).
#: 따로 풀면 적합이 성문 쪽만 0 으로 밀었다 (출발 0.003 → 0.0004, 협착 0.03) — 무작위 잡음은 크기 손실을 늘리기만 해 줄이는 쪽으로 편향되고, 그러면 고역이
#: 닫힘 충격만 남아 펄스형이 된다. 켜면 성문 난류에 협착 계수를 쓰고 `td_log_noise_g` 는 적합하지 않는다.
TIE_NOISE = False
#: **성문 진동 면적의 생리 상한** [cm²] (§52.524). 운동학 판(k5)이 성량 손잡이 `voice_gain` 을 +23.5 dB 까지 올려 진동 면적이 성문 최대 면적의 열 배를 넘었고
#: 성문 유량 정점이 700–1000 cm³/s 가 됐다 (여성 말소리 교류 ~200 cm³/s, Holmberg et al. 1988). 큰 말소리 여성의 최대 성문 면적 수준에서 tanh 로 포화시킨다
#: — 점막 모드의 몸체 진폭 상한 1 mm (면적 ≈ 2 · 성대 길이 · 진폭 ≈ 0.2 cm²) 와 같은 크기. 0 이면 끈다.
AVIB_MAX = 0.2
VF_NEURAL_NMU = 200
VF_NEURAL_TREMOR = (6.0, 0.03)          # 생리적 떨림 [Hz], 공통 구동 깊이
VF_JIT_HZ = 100.0


def set_tube_refine(r: int) -> None:
    """관 칸 수·내부 표본률·과표본 배율을 함께 R 배로 (쿠랑 수 유지). 엔진을 만들기 전에 부른다."""
    global OS, TUBE_REFINE
    r = max(1, int(r))
    TUBE_REFINE = r
    td.FS_SIM = 96000.0 * r
    td.N_SECT = 28 * r
    OS = 2 * r
    _LP.clear()


def _refine_area(A: torch.Tensor, n: int) -> torch.Tensor:
    """(…, M) 칸 면적 → (…, n) 칸. 칸 가운데 사이 면적 직선 보간 — 단 이웃 둘보다 좁거나 넓은 칸(협착·폐쇄·공동)은 제 값을 칸 길이
    내내 지킨다. 그냥 보간하면 한 칸짜리 폐쇄(1e-4 cm²)가 반 칸 어긋난 세분 칸에서 열린다; 그냥 되풀이하면 0.54 cm 계단이 남는다.
    log 면적 보간은 폐쇄 옆 칸을 0.5 → 0.06 cm² 로 좁혀 협착을 길게 만들어서(면적 직선이면 0.13) 쓰지 않는다."""
    M = A.shape[-1]
    if n == M:
        return A
    p = (torch.arange(n, dtype=A.dtype, device=A.device) + 0.5) * (M / n) - 0.5
    i0 = p.floor().clamp(0, M - 2)
    f = (p - i0).clamp(0.0, 1.0)
    i0 = i0.long()
    li = A[..., i0] * (1.0 - f) + A[..., i0 + 1] * f
    ap = torch.cat([A[..., :1], A, A[..., -1:]], -1)
    ext = (A <= torch.minimum(ap[..., :-2], ap[..., 2:])) | (A >= torch.maximum(ap[..., :-2], ap[..., 2:]))
    own = (torch.arange(n, device=A.device) * M) // n
    return torch.where(ext[..., own], A[..., own], li)
CONS_SIGMA = 0.03          # 협착의 공간 폭 (관 길이 대비, σ ≈ 0.45 cm → 폭 ~1 cm). 0.08(σ 1.2 cm)이면 치찰음의 앞공동(~1 cm)이 뭉개져
                           # ㅅ 의 10–12 kHz 봉우리가 서지 않았다 (§52.481)
SOFTMIN_TAU = 0.05         # 협착 겹침의 부드러운 최소 [cm²]
LIP_CELLS = 2              # 입술 구간 (oral_open 이 좁힌다)
OUT_SCALE = 1e-7           # 방사 음압 단위 → 디지털 척도 (전체 이득은 적합기가 맞춘다)
VOL_FLOW = 1.0             # 부피 변화 유량(조음 운동이 미는 공기)의 배율 — 진단용 스위치
GAP_PER_RD = 0.08          # rd_offset 1 당 막부 쉼 틈 [cm²] — −: 안쪽 압착(닫힌 몫 ↑), +: 기식. ±1.5 가 정상 떨림 반폭(0.1~0.16)을 덮는다
CONTACT_EPS = 0.002        # 접촉의 부드러움 [cm²] — 점막이 눌리는 폭. 0 이면 꺾임이 수학적 모서리가 된다
FOLD_SKEW = 0.6            # 성대 변위의 열림 몫 (한 주기의 60 % 동안 벌어지고 40 % 동안 모인다)
PHASE_AT_CLOSURE = False   # 펄스 위상 0 = 닫힘 순간 (§52.478)
PHC_EPS = 0.01             # 닫힘 자리 계산의 떨림 폭 바닥 [cm²] — 첫머리에서 r 이 발산하지 않게
PHC_RMAX = 0.98            # 접촉 문턱 비 r 의 부드러운 상한 (arccos 기울기 ≤ 5)
PHC_SMOOTH_MS = 10.0       # 닫힘 자리의 시간 평활 — 위상이 이보다 빨리 옮겨 가지 않는다
EPI_LEN_CM = 2.0           # 후두개관(epilarynx) 길이 — 성대 위 ~2 cm 의 좁은 관 (Titze & Story 1997; 여성은 짧다)
EPS_FIT = False            # 켜면 접촉의 부드러움을 화자 상수 `log_contact_eps` 로 적합한다 (§52.479) — 닫힘의 날카로움 = 고역 기울기
EPI_ON = False             # 켜면 성문 쪽 EPI_LEN_CM 의 면적을 적합 상수 `log_epi_area` 로 좁힌다 (§52.479)
EPI_INIT_AREA = 1.0        # 후두개관 좁힘의 초기 면적 [cm²] (`--throat` 는 0.5)
PRE_MS = 30.0              # 앞 굴림 — 첫 프레임 상태로 관을 먼저 돌려 정상 흐름에 앉힌 뒤 버린다 (§52.477)
#: **성문 모형** (§52.488). "kin": 운동학 — 펄스 위상으로 정한 매끈한 변위 + 접촉 (`glottal_area`). "vf": **자기 진동 성대** —
#: Birkholz 삼각 성문 두 질량 (`tube_td.VF_STATIC`) 이 폐압과 성문 위 압력에 밀려 스스로 떤다. 면적·진폭·닫힌 몫·개시·멈춤을 정하지 않는다
#: — 제어는 VocalTractLab 의 것과 같은 넷뿐이다:
#:   f0 → 긴장 Q = 1 + (f0 − 고유 f0)/(dF0/dQ) (`tube_td.VF_NATURAL_F0`·`VF_DF0_DQ`, W02) × 음높이 맞춤 `vf_qcorr`,
#:   내전 → 성대돌기의 쉼 변위 (아래 `VF_R_*`; 0 = 모달, + 외전, − 압착),
#:   rd_offset → 아래·위 쉼 변위의 차 (수렴) `VF_CONV`·exp(`VF_CONV_PER_RD`·rd_offset),
#:   폐압 → P_s. 뒤쪽(연골부) 틈은 화자 상수 `log_vf_chink`. voice_gain·fold_skew·pulse_shift·펄스 위상은 쓰지 않는다.
GLOTTIS = "kin"
#: **내전 → 쉼 변위** (위 질량, 성대돌기 자리) — 부드러운 경사 u = softplus(k(m − 내전))/k (≈ max(0, m − 내전)) 에
#: r = S·u + S2·u² − r_floor. 내전 `VF_ADD_MODAL` 에서 0 (VTL 모달), 압착 쪽은 −r_floor 에서 멈추고, 벌림 쪽은 이차로 넓어진다.
#: 발성 영역 실측 (/a/ 관, 뒤쪽 틈 0.005 cm², f0 200–420 Hz · 폐압 4–10 cmH2O): 위 쉼 변위 −0.05 mm 부터 떨고 — 0.1 mm 만 눌러도 멎고,
#: 멎은 자리는 기울기가 0 이라 적합이 되살리지 못한다 — 벌림 쪽은 f0 300 Hz 이상 1 mm, 200 Hz · 10 cmH2O 에서 2 mm 에서 멎는다.
#: 무성 마찰의 성문 면적 0.2–0.4 cm² (성대 1.5 cm 삼각형이면 1.3–2.7 mm). 옛 분석의 내전 초기값(유성 틀 0.35–0.62)이 선형 대응에서 0.4 mm
#: 압착이 되어 B 구간 유성 틀 절반이 무음이었다 (§52.488). 성대 모드에서 내전은 막부의 쉼 변위만 뜻한다 (뒤쪽 틈은 `log_vf_chink`).
#: 내전 0 → 1.9 mm, 0.1 → 0.9, 0.2 → 0.39, 0.35 → 0.06, 0.44 → 0, 0.6 → −0.025, 1 → −0.03 mm.
VF_ADD_MODAL = 0.44        # 코퍼스 유성 틀 내전의 중앙 (§52.426, 내전 사전의 중앙)
VF_R_SLOPE = 0.40          # cm / 내전 1 (모달 쪽 기울기)
VF_R_QUAD = 1.5            # cm / 내전² (벌림 쪽)
VF_R_K = 12.0
VF_R_FLOOR = 0.003         # cm — 압착의 끝 (−0.03 mm)
#: **모달 내전의 쉼 반틈새** [cm, 성대돌기 · 위 날] (§52.532, `copyfit --vf-rest-modal`). 0 (VTL 모달) 이면 모음에서 막부가 칼날로 맞붙어 길이 전체가 한꺼번에
#: 닿고 떨어졌다 — 라인 스캔 카이모그램 (`kymo.py`): 닿음 앞→뒤 +1.2 % · 떨어짐 뒤→앞 +0.9 % 주기, 건강한 여성의 편한 발성은 지퍼 꼴 (닿음 앞 → 뒤,
#: 떨어짐 뒤 → 앞; HSV + EGG, PMC3508318). 날카로운 동시 닫힘이 고역 배음을 줄로 세웠다: m23 을 쉼 틈 +0.1 · +0.2 · +0.3 mm 로 다시 렌더하면 고역 줄
#: (2–5 · 5–8 · 8–12 · 12–16 kHz, 코덱 없는 출력) 0.63 · 0.38 · 0.33 · 0.37 → 0.38 · 0.17 · 0.16 · 0.12 → 0.26 · 0.15 · 0.09 · 0.06 → 0.20 · 0.12 · 0.06 · 0.05
#: (원본 0.26 · 0.06 · 0.09 · 0.02), 지퍼 +2.7/+2.5 → +4.9/+3.6 → +7.3/+4.2 %, Praat HNR 17.6 → 14.8 · 15.3 · 16.2 (원본 15.1). 2 층 성대 모형의 기준 기하가
#: 쉼 최소 반틈새 0.09 mm, 문턱압 최소 0.0–0.1 mm (Zhang 2009 JASA 125). 더하는 몫 `VF_R_MODAL` · w(내전), w = 매끈한 min(1, (1 − 내전)/(1 − `VF_ADD_MODAL`))
#: — 모달 (0.44) 에서 0.94 · 그 아래 (벌림 쪽) 1 · 압착 끝 (내전 1) 0: 압착 끝 −0.03 mm 와 벌림 쪽 (무성 마찰 1.1 mm → 1.2) 은 거의 그대로. 0 이면 예전 대응.
VF_R_MODAL = 0.0


def _vf_r_modal_w(add):
    """`VF_R_MODAL` 의 가중 w(내전) — torch · float 모두."""
    v = (1.0 - add) / (1.0 - VF_ADD_MODAL)
    if isinstance(v, torch.Tensor):
        return v - torch.nn.functional.softplus(12.0 * (v - 1.0)) / 12.0
    return v - math.log1p(math.exp(12.0 * (v - 1.0))) / 12.0


def _vf_r_mid() -> float:
    """r(VF_ADD_MODAL) = 0 이 되는 경사의 무릎 m (이분법)."""
    lo, hi = VF_ADD_MODAL - 0.5, VF_ADD_MODAL
    for _ in range(80):
        m = 0.5 * (lo + hi)
        u = math.log1p(math.exp(VF_R_K * (m - VF_ADD_MODAL))) / VF_R_K
        lo, hi = (m, hi) if VF_R_SLOPE * u + VF_R_QUAD * u * u < VF_R_FLOOR else (lo, m)
    return 0.5 * (lo + hi)


VF_R_MID = _vf_r_mid()
VF_CONV = 0.0045           # 아래 − 위 쉼 변위 [cm] (VTL W02 모달 0.00495 − 0.0004) — 수렴이 클수록 압착 쪽 발성 경계가 넓다
VF_CONV_PER_RD = 0.5       # 수렴 = VF_CONV · exp(이 값 · rd_offset) — 늘 수렴형 (발산형 쉼 자세는 떨림을 막는다)
#: **f0 → 긴장 Q 대응** (§52.488). "vtl": VTL 의 직선 f0 = 고유 f0 + dF0/dQ·(Q − 1). "table": **이 모형이 실제로 내는** f0(Q) — /a/ 관
#: (F1 이 높아 음원-성도 결합이 약하다), 폐압 7 cmH2O, 모달 쉼 변위, 뒤쪽 틈 0.005 cm² 에서 잰 표를 log–log 로 보간해 뒤집는다. VTL 식보다
#: 4–15 % 낮고 Q 2 위에서 느려진다 (VTL 은 두 질량 압력을 관 모의에서 얻고, 여기서는 준정상 베르누이 — 결합이 다르다). 폐압(4–10)·내전
#: 에 따른 차이는 1–2 %. 그러면 음높이 맞춤(`vf_qcorr`)은 성도 결합·폐압의 잔차만 맡는다 — "vtl" 에서는 400 Hz 대가 배율 한계(×1.65)에 걸렸다.
VF_F0_MAP = "vtl"
VF_Q_TAB = (0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
VF_F0_TAB = (74.5, 99.8, 125.7, 160.7, 195.5, 246.2, 299.1, 340.4, 381.7, 457.1, 564.7)


def vf_tension(f0: torch.Tensor) -> torch.Tensor:
    """f0 [Hz] → 긴장 Q (`VF_F0_MAP`)."""
    if VF_F0_MAP != "table":
        return (1.0 + (f0 - td.VF_NATURAL_F0) / td.VF_DF0_DQ).clamp_min(0.3)
    lf = torch.log(torch.tensor(VF_F0_TAB, dtype=f0.dtype, device=f0.device))
    lq = torch.log(torch.tensor(VF_Q_TAB, dtype=f0.dtype, device=f0.device))
    x = torch.log(f0.clamp(50.0, 800.0))
    i = torch.searchsorted(lf, x.detach().contiguous()).clamp(1, lf.numel() - 1)
    x0, x1, y0, y1 = lf[i - 1], lf[i], lq[i - 1], lq[i]
    return torch.exp(y0 + (x - x0) * (y1 - y0) / (x1 - x0))        # 양 끝 밖은 끝 조각의 직선으로 늘인다


#: **성대 정적 변수의 화자 배율** (§52.489, 화자 상수 `td_log_vf_len`·`_thick`·`_mk`·`_damp` — 적합한다). W02 화자 파일의 삼각 성문 정적 변수는
#: VTL 기본값(남성 값에 가깝다)이다. 여성 막부 길이 ~1.0 cm (남성 1.5; Titze 1994) 인데 VTL 은 1.3·√Q — 이 화자의 Q 1.6–3.5 에서 1.6–2.4 cm.
#: 초기값: 길이 ×0.65, 질량·강성(조직 부피 ∝ 길이) ×0.65 — 고유 진동수 √(k/m) 는 그대로. 두께·감쇠 ×1.
VF_SCALE_INIT = dict(len=0.65, thick=1.0, mk=0.65, damp=1.0)


#: **몸체–덮개 성대** (§52.500, `copyfit --vf-body`): 성대 모드 2 — 두 덮개 질량이 몸체 질량에 붙는다 (`tube_td.vf_static_bc`). 몸체 강성은
#: 제어 `vf_ta`(갑상피열근 활성의 대리)가 틀마다 35^(값 − 0.5) 배로 움직인다. 끄면 모드 1 (VTL 두 질량).
VF_BODY = False
VF_TA_RANGE = 35.0
#: `vf_ta` 를 풀이기에 넣기 전 가우시안 σ [ms] — 근육 활성은 느리다 (폐압과 같은 까닭, §52.497). 1 ms 격자에서 몸체 강성이 틀마다 몇 배씩 뛰어
#: 적합이 무너졌다 (VBA 2.3 단계 포락 → −97 %, 13 dB).
VF_TA_SMOOTH_MS = 10.0


#: **N 줄 보–막 성대** (§52.504, `copyfit --vf-ns`): 성대 모드 3. 보–막 성대(Serry, Zañartu & Peterson 2026 — Alzamendi 2021 의 후속)를
#: 성대 길이의 첫 정재파(막 밴드의 k = π/L 가지)로 투영하고 상하를 `tube_td.VF_NSTRIP` 줄로 푼다 (`physics/beam_membrane.py`). 목표 f0 는
#: **늘어남 ε** 로 옮긴다 (모드 3 이 실제로 내는 f0(ε) 표, /a/ 관·폐압 7 cmH2O·결합 ×3·덮개 전단에서 잰 것 — ε 0.35 → 0.4 에서 374 → 480 Hz 로 뛴다 — 음높이 맞춤·고리는 긴장 Q 로 잔차만), 제어
#: `vf_ta` 는 갑상피열근 활성 (능동 응력 → 보 강성·축력·굽힘 모멘트). 층 응력·기하·질량·감쇠가 모두 조직 법칙에서 나온다.
VF_NS = False
VF_NS_EPS_TAB = (-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
VF_NS_F0_TAB = (177.4, 185.3, 195.9, 207.8, 219.7, 239.4, 263.7, 291.8, 322.1, 356.9, 373.5, 480.0, 551.7, 640.0)
#: 화자 상수 초기값 — 길이·두께 배율 (Titze 1989 여성 척도 대비), 층간 결합 배율 (Serry K_c 대비 — ×1 은 유량 마루 1.2 L/s·진폭 11 mm 로
#: 비생리, ×3 에서 150–220 cm³/s), 세로 조직 감쇠비, 갑상피열근 굽힘 모멘트의 정적 하중 이득 (1 이면 불룩함 3–4 mm — 내전 제어가
#: 정적 평형 기준의 쉼 틈이라 모양만 남기려 작게 둔다).
VF_NS_INIT = dict(len=1.0, thick=1.0, kc=3.0, zeta=0.1, bulge=0.1, shear=1.0)
#: **좌우 성대 비대칭의 생리 범위** (§52.533, `copyfit --vf-lr`). m23 에 긴장 비대칭 ±2 · ±5 · ±10 % 를 넣어 다시 렌더: 좌우 위상차 (위 줄 변위의 교차상관)
#: 중앙 2.2 · 5.8 · 7.9 % (최대 6.2 · 11.8 · **50**), 반배음 90 % 값 −9.6 · −13.2 · **−6.3 dB** (원본 −14.9), Praat 지터 1.15 · 1.42 · 2.15 %. ±10 % 는 두 성대의
#: 잠금이 자리에 따라 풀려 이중 음성이 선다 (Steinecke & Herzel 1995 의 비대칭 분기와 같은 꼴); 정상 습관 발성의 54 % 가 좌우 위상차 ≤ 6 % (PMC7587608) 라
#: 긴장 비대칭 상한 5 %, 출발 2 %. 질량 비대칭도 같은 상한.
VF_LR_DQ_MAX = 0.05
VF_LR_DQ_INIT = 0.02
VF_LR_DM_MAX = 0.05
#: **성대 화자 상수의 생리 범위 — 단단히** (§52.529, `copyfit --vf-ns-hard`). 사용자: *"그런 정체는 대부분 손실함수의 문제 아니야?"* — 약한 사전
#: (`fit.VF_NS_ANAT`, 덮개 강성 1–10 배)에 맡겼더니 적합이 덮개 강성을 ×4.05, 감쇠비를 0.057 (하한) 로 옮겼고, 그 성대의 발성 문턱은 6.5–13 cmH2O
#: 이거나 15 에서도 떨지 않았다 (`out/_tmp/vf/pth_sweep.py`) — 원본 유성 틀의 절반이 무성, 폐압은 상한에 붙음. 사람의 발성 문턱은 2–5 cmH2O
#: (높은 음 6–8; Titze 1992). 같은 지도에서 덮개 강성 ×1 이면 2.5–4.5, ×2 면 3.5–8.5 → 덮개 강성 [0.8, 1.5]. 감쇠비는 문턱에 거의 영향이 없어
#: 조직 감쇠 문헌값 0.1 이상 [0.1, 0.3] (Ishizaka & Flanagan 1972 덮개 0.1). 길이·두께는 개인차 ±15 % · ±30 %.
VF_NS_HARD = None
#: **정정 (§52.530)**: 위 덮개 [0.8, 1.5] 의 근거였던 지도는 내전 0.5 (아랫날 쉼 −0.015 mm — 눌림) 에서 잰 것이었다. 쉼 틈이 생리적인 내전 0.35
#: (아래 0.056 · 위 0.10 mm) 에서는 m13 상수(덮개 ×3.99, 두께 0.9)가 ε −0.15 … 0.4 (179–413 Hz) 내내 문턱 3.3–4.8 cmH2O (사람 2–4) 였고, 덮개 ×1.5 로
#: 자른 n1 상수(두께 1.18)는 383–430 Hz 에서 7.8–9.9 로 오히려 높았다 — 문턱은 덮개 하나가 아니라 두께 · 전단 · 내전이 함께 정한다. 덮개의 상한은
#: 같은 지도에서 어떤 두께로도 문턱이 10 cmH2O 아래로 안 오는 ×8 의 아래 ×6, 하한은 그대로. 문턱은 문턱 항(`fit.PTH_W`)과 폐압 상한이 직접 묶는다.
#: 감쇠비 하한 정정 (§52.531): 0.1 은 두 질량 모형의 매개변수(Ishizaka & Flanagan 1972)였다 — 조직 측정으로는 성대 덮개의 손실 탄젠트 tan δ = G''/G'
#: 가 0.1–0.5 (Chan & Titze 1999, 진동수 0.01–15 Hz; 발성 진동수 쪽으로 줄어든다) 라 감쇠비 tan δ / 2 ≈ 0.05–0.25. n6 상수에서 감쇠비 0.1 → 0.055 면
#: 문턱이 15–25 % 낮다 (419 Hz 10.2 → 7.8 cmH2O).
VF_NS_HARD_DEFAULT = dict(kc=(0.8, 6.0), zeta=(0.05, 0.30), len=(0.85, 1.15), thick=(0.7, 1.3))
#: **몸체 구동 점막 성대** (§52.523, `copyfit --vf-body-drive`, 모드 3 위에서). 사용자: *"성대를 강체처럼 생각하니까 저런 펄스가 나오지. 성대는
#: 점막인데"*. 운동학 성문(정한 면적 모양 + 접촉 꺾임)은 주기마다 같은 닫힘 충격으로 고역 가로줄을 세웠고(038 줄 0.19 · 0.26 · 0.11, 원본
#: 0.05 · 0.10 · 0.01), 자유 진동 보–막 성대는 손실이 매끈하지 않아 적합이 안 됐다(§52.522). 몸체–덮개 이론(Hirano 1974; Story & Titze 1995)
#: 그대로: 무겁고 굳은 **몸체**(갑상피열근 + 인대 보)는 원본 닫힘 시각에 묶인 펄스 위상을 따르게 굳은 용수철로 구동점에 매고(암시적 풀이라 안정,
#: 커널의 몸체 강성 열 · 표본별 몸체 하중 열을 쓴다 — 커널·수반은 그대로), **덮개(점막 막 줄)** 는 층간 결합·장력·전단·접촉·성문 압력으로 제
#: 동역학을 푼다 — 점막파, 깊이·길이로 번지는 닫힘, 무른 접촉은 덮개가 만든다. 구동점 x_d = x_정적 − X_b·cos φ (x_정적 = 갑상피열근 굽힘 하중의
#: 정적 위치 fB/(cB·Q), X_b = ½ · 진동 면적 / 성대 길이 — 운동학 판의 진동 폭과 같은 뜻, 생리 상한 `VF_BODY_DRIVE_XMAX` 를 tanh 로).
#: 매는 강성 = `VF_BODY_DRIVE_K` × (몸체 강성 + 층간 결합). 켜면 위상 고정 고리는 쓰지 않는다 (시각은 몸체가 쥔다).
VF_BODY_DRIVE = False
VF_BODY_DRIVE_K = 100.0
VF_BODY_DRIVE_XMAX = 0.1     # cm — 몸체 진폭 상한 (덮개 진폭 ~1 mm, Titze 1994)
#: **구동이 약할 때는 몸체를 풀어 둔다** (§52.533). 매는 강성이 늘 몸체 강성의 `VF_BODY_DRIVE_K` 배라, 구동 폭 (운동학 로지스틱, 씨앗 0.02 에서 ~30 ms 에 자란다) 이
#: 0 인 동안 몸체가 정적 자리에 굳게 묶여 덮개도 스스로 못 떤다 — m21 의 ㄲ 개방 뒤 775–805 ms: 성대를 미는 압력차 ~10.5 cmH2O 에서도 떨림 0, 닫힘 폭 50 μm 로 낮춰도
#: 40 ms (떨림 시작 원본보다 +44 · +52 ms). 매는 강성 = K·(몸체 + 층간)·g(구동 폭), g = a²/(a² + `VF_BODY_DRIVE_FREE`²) — 구동 폭이 이 값보다 작으면 점막 성대가
#: 제 물리로 떨기 시작하고 (개방의 압력 계단이 들뜬다), 자라면 원본 박자에 묶인다. 0 이면 예전처럼 늘 묶는다.
VF_BODY_DRIVE_FREE = 0.0
VF_DRIVE_VG = False          # 몸체 진폭에 성량 손잡이를 쓸지 (§52.524 — 끈다: 진폭은 생리 진폭만)


def _pchip_slopes(x, y):
    """Fritsch–Carlson 단조 3 차 보간의 마디 기울기 (numpy)."""
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = np.diff(x)
    dl = np.diff(y) / h
    m = np.zeros_like(y)
    m[0], m[-1] = dl[0], dl[-1]
    for k in range(1, len(y) - 1):
        if dl[k - 1] * dl[k] <= 0:
            m[k] = 0.0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / dl[k - 1] + w2 / dl[k])
    return m


def vf_ns_eps(f0: torch.Tensor) -> torch.Tensor:
    """목표 f0 [Hz] → 늘어남 ε — 모드 3 의 f0(ε) 표를 log f0 에 대해 **단조 3 차(PCHIP)** 로 뒤집는다 (C¹, §52.524; 예전 조각 직선은 마디에서
    꺾였다). 양 끝 밖은 끝 기울기로 곧게 늘인다."""
    import math as _m
    lfl = [_m.log(v) for v in VF_NS_F0_TAB]
    mk = _pchip_slopes(lfl, VF_NS_EPS_TAB)
    lf = torch.tensor(lfl, dtype=f0.dtype, device=f0.device)
    ee = torch.tensor(VF_NS_EPS_TAB, dtype=f0.dtype, device=f0.device)
    mm = torch.tensor(mk, dtype=f0.dtype, device=f0.device)
    x = torch.log(f0.clamp(60.0, 900.0))
    i = torch.searchsorted(lf, x.detach().contiguous()).clamp(1, lf.numel() - 1)
    x0, x1, y0, y1, m0, m1 = lf[i - 1], lf[i], ee[i - 1], ee[i], mm[i - 1], mm[i]
    hh = x1 - x0
    t = (x - x0) / hh
    inside = (t >= 0) & (t <= 1)
    tc = t.clamp(0.0, 1.0)
    h00, h10, h01, h11 = 2 * tc ** 3 - 3 * tc ** 2 + 1, tc ** 3 - 2 * tc ** 2 + tc, -2 * tc ** 3 + 3 * tc ** 2, tc ** 3 - tc ** 2
    yin = h00 * y0 + h10 * hh * m0 + h01 * y1 + h11 * hh * m1
    ylo = ee[0] + mm[0] * (x - lf[0])
    yhi = ee[-1] + mm[-1] * (x - lf[-1])
    y = torch.where(inside, yin, torch.where(x < lf[0], ylo, yhi))
    return y.clamp(-0.3, 0.7)


def vf_static(len_s, thick_s, mk_s, damp_s):
    """화자 배율 넷 → 정적 변수 텐서 (`tube_td.VF_STATIC` 차례: 길이, 두께 둘, 질량 둘, 감쇠비 둘, 용수철 둘, 접촉 둘, 결합, 입구, 출구;
    `VF_BODY` 면 + 몸체 질량·강성·감쇠비·비선형 둘 — 질량·강성은 질량강성 배율, 감쇠비는 감쇠 배율을 같이 받는다)."""
    one = torch.ones((), dtype=len_s.dtype, device=len_s.device)
    if VF_BODY:
        base = torch.as_tensor(td.vf_static_bc(), dtype=len_s.dtype, device=len_s.device)
        sc = torch.stack([len_s, thick_s, thick_s, mk_s, mk_s, damp_s, damp_s, mk_s, mk_s, mk_s, mk_s, mk_s, one, one,
                          mk_s, mk_s, damp_s, one, one])
    else:
        base = torch.as_tensor(td.VF_STATIC, dtype=len_s.dtype, device=len_s.device)
        sc = torch.stack([len_s, thick_s, thick_s, mk_s, mk_s, damp_s, damp_s, mk_s, mk_s, mk_s, mk_s, mk_s, one, one])
    return base * sc


#: **성대 모드의 폐압 초기값 규칙** (§52.489). 분석의 폐압은 무음에서도 평평하고(5 cmH2O) 목소리가 잦아드는 끝에서도 내려가지 않아, 자기 진동
#: 성대는 무음 구간에 벌어진 성문으로 숨소리를 내고 끝에서 떨림 문턱을 오갔다. 음압은 폐압이 두 배일 때 ~9 dB 오른다 (Titze 1994 의 SPL–Ps
#: 관계, 문턱 위): 말소리 틀 P = `VF_PS_REF`·2^((L − L_중앙)/`VF_PS_DB2`) (L 은 목표 틀 음압 dB, 중앙은 유성 틀), [`VF_PS_LO`, `VF_PS_HI`]
#: 로 가둔다; 무음 틀(유성도 마찰도 아니고 L < 최대 − `VF_SIL_DB`)은 `VF_PS_SIL` — 단 **구간 앞뒤 끝에 붙은 무음만** (발화 앞·뒤의 숨 멈춤):
#: 안쪽 무음은 파열음 폐쇄라 폐압이 남아 있어야 개방 파열이 난다. 10 ms 로 고른다.
VF_PS_REF = 7.0
#: 성대 모드에서 폐압 제어를 풀이기에 넣기 전 가우시안 σ [ms] (§52.497). 폐압은 호흡 근육이 정하는 느린 양인데, 1 ms 격자의 적합이 주기 크기를
#: 다듬는 손잡이로 써서 틀마다 0.5 cmH2O 씩 떨었다 (VMC_a 5 ms 평활 이탈 rms 0.22 cmH2O, 운동학판 0.03) — 자기 진동 성대에서는 폐압이 곧 떨림 폭이라
#: 주기마다 파형이 바뀌었다. 0 이면 끈다.
VF_ADD_SMOOTH_MS = 0.0     # 내전·수렴(rd_offset) 제어의 가우시안 σ [ms] (성대 모드) — 0 이면 끈다
VF_PS_SMOOTH_MS = 10.0     # VMA_a · VMC_a 제어에서 포락 61.2 → 65.8 · 41.6 → 48.7 %, 갈라짐 0.72 → 0.51 · 0.65 → 0.51
VF_PS_DB2 = 9.0
VF_PS_LO, VF_PS_HI = 2.5, 14.0
VF_PS_SIL = 0.3
VF_SIL_DB = 30.0


def vf_psub_rule(level_db, voiced, fricative, frame_ms: float):
    """틀별 목표 음압 [dB] · 유성 · 마찰 표지 → 폐압 초기값 [cmH2O] (`VF_PS_*`)."""
    import numpy as np
    L = np.asarray(level_db, float)
    v = np.asarray(voiced, bool)[:L.size]
    f = np.asarray(fricative, bool)[:L.size]
    v = np.pad(v, (0, L.size - v.size))
    f = np.pad(f, (0, L.size - f.size))
    ref = np.median(L[v]) if v.any() else np.median(L)
    P = np.clip(VF_PS_REF * 2.0 ** ((L - ref) / VF_PS_DB2), VF_PS_LO, VF_PS_HI)
    sp = v | f
    if sp.any():                                   # 말소리가 아닌 안쪽 틀(파열 폐쇄 등)은 양옆 말소리 틀의 폐압을 잇는다
        P = np.interp(np.arange(L.size), np.flatnonzero(sp), P[sp])
    q = ~v & ~f & (L < L.max() - VF_SIL_DB)
    sil = np.zeros_like(q)
    i = 0
    while i < q.size and q[i]:
        sil[i] = True
        i += 1
    i = q.size - 1
    while i >= 0 and q[i]:
        sil[i] = True
        i -= 1
    P[sil] = VF_PS_SIL
    k = max(1, int(round(10.0 / frame_ms)))
    Pp = np.pad(P, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(Pp, np.ones(k) / k, "valid")[:L.size], int(sil.sum())


#: **후두 구조** (§52.489, `copyfit --larynx`). 사용자: *"성도 말고도 가성대 등등 방해물도 고려해야 해"* / *"이걸로 성도 구조를 추론하라는
#: 뜻이야"* / *"좀 이상한 걸로 적합을 한 듯한데, W2a으로 해"*. 포락 95 % 판 **W2a** (079 발화, 포먼트 엔진 + 보조 극·영점·고역 영점/마루)가 원본을
#: 맞추려고 둔 보조 구조를 옛 식으로 틀마다 풀어, **조음(F1·F2)과 상관없이 늘 같은 자리**에 서는 것만 구조로 읽었다: 반공명 **6.43 kHz**
#: [6.0–7.1] (F1·F2 상관 −0.01·+0.07, 옆 마루 6.24) — 이상와; 반공명 **11.8 kHz** [11.0–12.0] — 성문 바로 위 곁공동 **후두실**(모르가니 굴).
#: 038 판(W3a·W18e)의 F3–F4 틈 4.7–5.1 kHz 영점은 W2a 에 없다 (0 %) — 화자 구조가 아니다. 곁관 자리는 결정적 임펄스 응답의 골로 맞췄다
#: (`out/_tmp/vf/calib_branch2.py`·`calib_neck3.py`): 좌우 이상와 각 0.4 cm² × 1.024 cm → 6.42 kHz; 후두실은 헬름홀츠 — 목 0.15 cm² × 0.2 cm
#: (실제 입구 틈 ~0.2 cm², 0.2–0.3 cm) + 공동 0.07 cm² × 0.8 cm (0.056 cm³) → 11.49 kHz. 곁관 칸은 안정 조건상 0.365 cm 보다 길어야 해 1/4 파장만
#: 으로는 ~9.7 kHz 위를 못 내고, 목이 0.06 cm 로 짧거나 공동이 0.06 cm² 아래면 첫 칸의 엇갈린 풀이가 발산한다 — 11.5 kHz 가 이 격자의 한계.
#: **가성대**는 후두실 위 칸(성문 위 ~0.8 cm)의 좁힘 min(관, A_ff·(1 − 0.9·ff_add)) — 쉼 틈 A_ff 는 화자 상수 `td_log_ff_area`, 내전 `ff_add`
#: 는 제어 (압착·된소리). 가성대 뒤로 넓어지는 이음매의 분출 손실·레이놀즈 난류는 관 물리가 낸다.
LARYNX = False
LARYNX_BRANCHES = [(td.PIR_FRAC, 0.4, 1.024), (td.PIR_FRAC, 0.4, 1.024), (0.0, 0.07, 0.8, 0.15, 0.2)]
FF_CELL = 1
LARYNX_LUMEN = 0.8         # 후두실 높이의 관 면적 상한 [cm²] (해부학 0.3–1; 안정)
#: **가성대 · 연구개 자세의 생리 동역학** (§52.530, `copyfit --physio-dyn`). 사용자: *"변조가 낀 느낌이라고 해야 하나? 뭔가 방해물이 성도를
#: 닫았다 열었다 하는 거 아닌지"*. 가성대 내전(`ff_add`)에는 평활도 속도 제한도 없어 적합이 40–60 Hz 로 0 ↔ 0.9 를 흔들었다 (KWF_m7–m13 · n2
#: 빠른 흔들림 RMS 0.13–0.16) — 관 칸 1 의 가성대 틈이 0.08 ↔ 0.39 cm², 모음에서 성도의 가장 좁은 곳이 1 초에 50 번 여닫혔고 출력 1–3 kHz 포락의
#: 4–60 Hz 변조가 6.6 dB (원본 4.3). 가성대는 갑상피열근 바깥 갈래 · 피열후두개 괄약이 움직이는 후두 자세라 근육 활성의 σ 는 성대 긴장
#: (`VF_TA_SMOOTH_MS`)과 같은 10 ms, 최대 속도는 내전과 같은 20 /s (전 범위 50 ms — 성문 여닫음 몸짓 50–100 ms, Löfqvist & Yoshioka 1980).
#: 연구개(`velum`, 구개범거근) 도 σ 10 ms. 0 이면 끈다.
FF_SMOOTH_MS = 0.0
FF_VMAX = 20.0
VELUM_SMOOTH_MS = 0.0
FF_AREA_INIT = 0.35


#: **비강** (§52.490). W02 (MRI) 는 구강·인두 기하뿐이고 비강은 VTL 의 일반 모형(Dang & Honda 1994 일본 남성 피험자 4 + 부비동 넷)이다 —
#: 이 화자의 것이 아니다. 예전 비강(손으로 만든 16 칸 11.5 cm, 부비동 없음)도 마찬가지. W2a 가 비음 구간에 둔 비강 극 393·1362·3196 Hz ·
#: 영점 2631 Hz 에 비음(ㄴ, 입 막음, 연구개 0.4 cm²) 전달함수를 맞춰 훑었다 (`out/_tmp/vf/nasal_tf.py`): **VTL 비강 + 부비동, 길이 ×0.9 ·
#: 단면 ×0.7** 이 가장 가깝다 (봉우리 434·1289·3143, 골 2643; 오차 0.176) — 부비동 없이 길이만 ×0.75 면 극은 맞지만 2.6 kHz 영점이 없다
#: (0.319). 여성 비강이 더 작다는 것과 맞는다. 부비동의 400–800 Hz 극·영점 쌍은 W2a 의 비강 모형(극 셋·영점 하나)이 나타낼 수 없어 증거가 없다.
NOSE = "default"
NOSE_SCALE = (0.9, 0.7)
#: **목 조임** (§52.490, `copyfit --throat`). 사용자: *"스트리머 특성상 목을 살짝 조아서 말하는 경우가 많을 테니"*. 습관적 후두 조임 —
#: 피열후두개 괄약(후두개관 좁힘, 'twang' 의 2–4 kHz 강조) + 가성대 내전 (`ff_add`, 제어) — 을 화자 설정으로: 성문에서 `EPI_LEN_CM` 까지
#: 면적을 화자 상수 `td_log_epi_area` 로 부드럽게 좁힌다 (모든 조음 모드).
THROAT = False
THROAT_EPI_INIT = 0.5


def configure_structures() -> None:
    """`LARYNX`·`NOSE` 에 맞게 관의 곁관 목록과 비강 기하를 조립한다."""
    br = list(LARYNX_BRANCHES) if LARYNX else []
    if NOSE == "vtl":
        td.set_nose("vtl", *NOSE_SCALE)
        br += list(td.SINUS_VTL)
    else:
        td.set_nose("default")
    td.SIDE_BRANCHES = br if (LARYNX or NOSE == "vtl") else None


def set_larynx(on: bool) -> None:
    global LARYNX
    LARYNX = bool(on)
    configure_structures()


def set_nose(kind: str) -> None:
    global NOSE
    NOSE = kind
    configure_structures()


VF_CHINK_FIX = None        # 진단용: 뒤쪽 틈을 이 값 [cm²] 으로 고정 (적합 상수 대신)
VF_PRE_MS = 40.0           # 성대 모드의 앞 굴림 — 떨림은 폐압이 선 뒤 ~20 ms 안에 정상 폭에 선다
PRE_RAMP_MS = 15.0         # 앞 굴림 안에서 폐압을 0 → 첫 값으로 올리는 시간 (올림 코사인)


def _softmin(a, b, tau=SOFTMIN_TAU):
    return -tau * torch.logaddexp(-a / tau, -b / tau)


def _softmin_all(a, tau):
    """마지막 축의 부드러운 최소 (위로 치우치지 않게 칸 수의 log 를 뺀다)."""
    return -tau * (torch.logsumexp(-a / tau, -1) - math.log(a.shape[-1]))


def fold_shape(phi: torch.Tensor, skew=None) -> torch.Tensor:
    """한 주기 안 위치 φ ∈ [0,1) → 성대 변위 −1~1 (매끈한 주기 함수, 벌어짐 `skew`·모임 1−skew). 모서리가 없다 — 꺾임은 접촉이 만든다.
    skew 는 스칼라 또는 φ 와 같은 모양의 텐서(표본마다), 없으면 `FOLD_SKEW`."""
    skew = FOLD_SKEW if skew is None else skew
    up = torch.cos(math.pi * (phi / skew).clamp(0.0, 1.0))
    dn = torch.cos(math.pi * (1.0 + ((phi - skew) / (1.0 - skew)).clamp(0.0, 1.0)))
    return -torch.where(phi < skew, up, dn)


def contact(x: torch.Tensor, eps=None) -> torch.Tensor:
    """막부 면적 → 0 에서 잘린 면적 (성대가 닿으면 더 닫히지 않는다). 부드러운 최대 ½(x + √(x²+ε²))."""
    eps = CONTACT_EPS if eps is None else eps
    return 0.5 * (x + torch.sqrt(x * x + eps * eps))


def _box(y: torch.Tensor, n: int) -> torch.Tensor:
    """(B, N, C) 시간축 이동 평균 (길이 n, 끝은 가장자리 값) — 누적합으로 O(N)."""
    B, N, C = y.shape
    # 누적합은 float64 로 (§52.524) — 표본률 신호(16 만 표본)를 float32 로 누적하면 합이 10⁴–10⁵ 이 되어 1e-3 단위밖에 못 가려, 표본률로 올린 모든
    # 제어에 1e-3 수준 계단 잡음이 얹혔다: 성문 면적이 성문 최대 면적에 대해 거칠었고 (2 차 차분 비 h 1e-4 에서 1.24, float64 는 9e-4) 적합 손실의
    # 방향 차분이 보폭마다 뒤집혔다.
    yd = y.double()
    yp = torch.cat([yd[:, :1].expand(B, n // 2, C), yd, yd[:, -1:].expand(B, n - n // 2, C)], 1)
    S = torch.cat([torch.zeros_like(yp[:, :1]), yp.cumsum(1)], 1)
    return ((S[:, n:n + N] - S[:, :N]) / n).to(y.dtype)


def spline_up(x: torch.Tensor, hop: int) -> torch.Tensor:
    """프레임률 (B, T, C) → 표본률, **3차 B-스플라인** (직선 보간에 길이 hop 이동 평균 두 번).

    직선 보간은 프레임마다 기울기가 꺾인다. 관은 입술 유량의 **미분**을 방사하므로 그 꺾임이 1 ms 마다 계단이 되어
    1 kHz 버즈와 그 배음으로 나온다 — 진동 없이 정적 성문 틈만 넣어도 첫머리 6–12 kHz 가 목표보다 +27 dB, 스플라인으로 −21 dB (§52.477).
    기관의 속도에는 계단이 없다. 이음은 C² 이고 대칭이라 시간이 밀리지 않는다.
    """
    return _box(_box(frames_to_samples(x, hop), hop), hop)


class TDPath(nn.Module):
    """시간 영역 경로의 화자·발화 상수 (`calibration.FIELDS` 에 등록 — 적합기가 전역 스칼라로 맞춘다)."""

    def __init__(self, fs: float, hop: int):
        super().__init__()
        self.fs, self.hop = float(fs), int(hop)
        self.log_ag_max = nn.Parameter(torch.tensor(math.log(0.18)))     # 성대 진동 최대 면적 [cm²] (문헌 0.1~0.3)
        self.log_noise_g = nn.Parameter(torch.tensor(math.log(0.005)))   # 성문 난류 세기 (적합한다)
        self.log_noise_c = nn.Parameter(torch.tensor(math.log(0.01)))    # 협착 난류 세기 (적합한다)
        self.log_epi_area = nn.Parameter(torch.tensor(math.log(EPI_INIT_AREA)))  # 후두개관 면적 [cm²] (문헌 0.2~1.0, 화자) — `EPI_ON`
        self.log_contact_eps = nn.Parameter(torch.tensor(math.log(CONTACT_EPS)))  # 접촉의 부드러움 [cm²] (화자, 점막) — `EPS_FIT`
        # Maeda 조음 모형 (`ARTIC = "maeda"`) — 원 화자 → 이 화자 배율 (화자 상수): 인두·구강 길이, 단면
        self.lam = MaedaLAM()
        _s = SpeakerScale()
        self.log_sc_ph = nn.Parameter(torch.tensor(math.log(_s.pharynx)))
        self.log_sc_or = nn.Parameter(torch.tensor(math.log(_s.oral)))
        self.log_sc_area = nn.Parameter(torch.tensor(math.log(_s.area)))
        # W02 (VocalTractLab 여성, `ARTIC = "w02"`) — 화자 적응 배율 (길이·단면, 화자 상수)
        self.log_w02_len = nn.Parameter(torch.tensor(0.0))
        self.log_w02_area = nn.Parameter(torch.tensor(0.0))
        # W02 해부 적응 (`W02_ANAT`) — 해부 13 개의 시그모이드 좌표 (0 = W02 해부, 화자 상수)
        for i in range(N_W02_ANAT):
            setattr(self, f"w02a_u{i}", nn.Parameter(torch.tensor(0.0)))
        # 나이 축 해부 (`W02_AGE`) — 나이 좌표 하나 (0 = W02_AGE_REF) + 개인차 13 개의 tanh 좌표 (0 = 성장 곡선 위)
        self.w02_age_u = nn.Parameter(torch.tensor(0.0))
        for i in range(N_W02_ANAT):
            setattr(self, f"w02d_u{i}", nn.Parameter(torch.tensor(0.0)))
        # 자기 진동 성대 (`GLOTTIS = "vf"`) — 뒤쪽(연골부) 틈 [cm²] (화자 상수)
        self.log_vf_chink = nn.Parameter(torch.tensor(math.log(0.005)))
        self.log_vf_len = nn.Parameter(torch.tensor(math.log(VF_SCALE_INIT["len"])))
        self.log_vf_thick = nn.Parameter(torch.tensor(math.log(VF_SCALE_INIT["thick"])))
        self.log_vf_mk = nn.Parameter(torch.tensor(math.log(VF_SCALE_INIT["mk"])))
        self.log_vf_damp = nn.Parameter(torch.tensor(math.log(VF_SCALE_INIT["damp"])))
        self.log_ff_area = nn.Parameter(torch.tensor(math.log(FF_AREA_INIT)))         # 가성대 쉼 틈 [cm²] (화자) — `LARYNX`
        # N 줄 보–막 성대 (`VF_NS`, 성대 모드 3) — 화자 상수
        for _k, _v in VF_NS_INIT.items():
            setattr(self, f"log_ns_{_k}", nn.Parameter(torch.tensor(math.log(_v))))
        # 좌우 성대 비대칭 (§52.533, `beam_membrane.LR_FOLDS`) — 긴장 δQ = VF_LR_DQ_MAX·σ(u) ∈ (0, 최대) (좌우를 맞바꾸면 같은 소리라 δ = 0 에서 기울기가
        # 0 — 0 이 아닌 곳에서 출발), 질량 δM = VF_LR_DM_MAX·tanh(u) (부호는 긴장 비대칭에 대해)
        self.ns_lr_u_q = nn.Parameter(torch.tensor(math.log(VF_LR_DQ_INIT / (VF_LR_DQ_MAX - VF_LR_DQ_INIT))))
        self.ns_lr_u_m = nn.Parameter(torch.tensor(0.0))
        #: 음높이 맞춤 (프레임률 긴장 배율, (B, T) 또는 None) — 적합 변수가 아니다. 적합기가 합성 f0 를 재어 고친다 (`pitch_lock`).
        self.vf_qcorr = None
        #: 음량 맞춤의 프레임률 폐압 배율 ((B, T) 또는 None) — 적합 변수가 아니다 (`fit.vf_level_lock`, §52.504).
        self.vf_pscorr = None
        #: 주기 시각 맞춤의 **표본률** 긴장 배율 (출력 구간 96 kHz, numpy (n,) 또는 None) — 적합 변수가 아니다 (`fit.vf_cycle_lock`).
        self.vf_qcyc = None
        #: 위상 고정 고리의 목표 닫힘 시각표 [출력 구간 96 kHz 표본] (numpy 또는 None, §52.493) — 적합 변수가 아니다 (`fit.vf_pll_calibrate`).
        self.vf_pll_t = None
        self.last_ev = None
        self.seed = 0
        self.last = None
        self.last_rec = None

    def tube_frames(self, c: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """프레임률 제어 → (구강 면적 (B, T, N_SECT) [cm²], 성도 길이 (B, T) [cm]). `ARTIC` 이 방식을 고르고, `LARYNX` 면 가성대를 좁힌다."""
        A, L = self._tube_frames(c)
        if A.shape[-1] != td.N_SECT:                   # 관 해상도 배율 (§52.508) — 대리 모형의 28 칸을 관의 칸 수로
            A = _refine_area(A, td.N_SECT)
        if THROAT and ARTIC != "cos":                  # cos 는 area_frames 안에서 (EPI_ON) 이미 좁혔다
            xc = (torch.arange(A.shape[-1], dtype=A.dtype, device=A.device) + 0.5) / A.shape[-1] * L.unsqueeze(-1).to(A.dtype)
            wep = torch.sigmoid((EPI_LEN_CM - xc) / (0.5 * L.unsqueeze(-1).to(A.dtype) / A.shape[-1]))
            A = A * (1.0 - wep) + wep * _softmin(A, torch.exp(self.log_epi_area).to(A.dtype))
        if LARYNX:
            ffa = c["ff_add"].clamp(0.0, 1.0)
            if FF_SMOOTH_MS > 0.0:                     # 후두 자세의 생리 동역학 (§52.530)
                ffa = _gauss_t(ffa, FF_SMOOTH_MS / (1000.0 * self.hop / self.fs))
                if SLEW_ON and FF_VMAX > 0.0:
                    ffa = _slew(ffa, FF_VMAX * self.hop / self.fs)
            aff = torch.exp(self.log_ff_area).to(A.dtype) * (1.0 - 0.9 * ffa).to(A.dtype)
            if FF_BOUNDS:                              # 공명음의 가성대 틈 하한 (§52.532)
                wl = _ff_lo_w(aff.shape[-1], self.hop / self.fs)
                if wl is not None:
                    from ..physics import beam_membrane as _BM
                    Lf = _BM.SERRY_MALE["L0"] * 100.0 * _BM.FEMALE_LEN * (float(torch.exp(self.log_ns_len)) if VF_NS else 1.0)
                    lo = torch.as_tensor(wl, dtype=A.dtype, device=A.device) * (math.pi / 4.0 * FF_GAP_MIN_MM * 0.1 * Lf)
                    dl = 0.05 * lo + 1e-6
                    aff = lo + dl * torch.nn.functional.softplus((aff - lo) / dl)
            k = FF_CELL
            # 후두실 높이(참성대–가성대 사이)의 관 안쪽 면적은 0.3–1 cm² — 곡선 조음의 첫 칸이 1.9 cm² 로 넓으면 후두실 목으로 드는 관성이
            # 작아져 작은 공동 칸의 국소 진동이 안정 한계(ωdt ≈ 2.01)를 넘어 터졌다 (038 C 출발 이득 −230 dB, §52.491).
            # 칸 번호는 28 칸 기준 — 관 해상도 배율 r 이면 그 칸의 세분 r 개 전부 (같은 길이의 후두실·가성대)
            r_ = max(1, A.shape[-1] // 28)
            k0, k1 = k * r_, (k + 1) * r_
            a0 = _softmin(A[..., 0:r_], torch.full_like(A[..., 0:r_], LARYNX_LUMEN), 0.02)
            A = torch.cat([a0, A[..., r_:k0], _softmin(A[..., k0:k1], aff.unsqueeze(-1), 0.02), A[..., k1:]], -1)
        return A, L

    def _tube_frames(self, c: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if ARTIC == "maeda":
            p = torch.stack([c[n] for n in ART_NAMES], -1).double()
            A, x = self.lam(p)
            sc = tuple(torch.exp(q).double() for q in (self.log_sc_ph, self.log_sc_or, self.log_sc_area))
            Au, L = uniform_tube(A, x, td.N_SECT, self.lam.n_ph, sc)
            dt = c["p_sub"].dtype
            return Au.to(dt), L.to(dt)
        if ARTIC == "w02" and W02_ANAT:
            sur = _w02a()
            p = torch.stack([self._w02_smooth(c, n) for n in sur.names], -1)
            a = self.w02_anatomy(sur)
            A, L, _ = sur(p, a.expand(*p.shape[:-1], -1).to(p.dtype))
            dt = c["p_sub"].dtype
            return A.double().to(dt), L.double().to(dt)
        if ARTIC == "w02":
            sur = _w02()
            p = torch.stack([self._w02_smooth(c, n) for n in sur.names], -1)
            A, L, _ = sur(p)
            dt = c["p_sub"].dtype
            return (A.double() * torch.exp(self.log_w02_area).double()).to(dt), (L.double() * torch.exp(self.log_w02_len).double()).to(dt)
        return self.area_frames(c), c["tract_len"].clamp(10.0, 20.0)

    def tube_frames_soft(self, c: dict, beta: float = 50.0) -> torch.Tensor:
        """명세 손실용 면적 (B, T, 28) [cm²] — W02 해부 대리 모형의 **부호 있는 면적**(바닥 자르기 전, 음수 = 조음기가 파고든 깊이)을 완만한
        softplus(β) 로 (§52.522). 관에 들어가는 면적은 softplus(β 2000) + 1e-4 로 바닥에 붙어, 깊은 폐쇄에서는 log 면적의 기울기가 0 이었다 —
        ㄲ 이 녹음보다 30 ms 늦게 열렸는데 "개방 뒤 ≥ 0.02 cm²" 명세가 열 방향을 몰랐다. 생리 평활·속도 한계는 `_tube_frames` 와 같다."""
        from .articulation import W02_N_CELLS, AREA_EPS, AREA_BIAS
        sur = _w02a()
        p = torch.stack([self._w02_smooth(c, n) for n in sur.names], -1)
        a = self.w02_anatomy(sur)
        y = sur.raw(p, a.expand(*p.shape[:-1], -1).to(p.dtype))
        s = torch.exp(y[..., :W02_N_CELLS]).double() - AREA_EPS - AREA_BIAS
        return torch.nn.functional.softplus(s, beta=beta) + 1e-4

    def teeth_frames(self, c: dict):
        """입술–앞니 거리 (B, T) [cm] — W02 해부 적응 조음일 때만 (아니면 None). 음원 자리 분류에만 쓰여 기울기를 내지 않는다 (§52.519)."""
        if not (ARTIC == "w02" and W02_ANAT and NOISE_TEETH and td.NOISE_PLACE):
            return None
        sur, net = _w02a(), _w02t()
        with torch.no_grad():
            p = torch.stack([self._w02_smooth(c, n) for n in net.names], -1)
            a = self.w02_anatomy(sur)
            return net(p, a.expand(*p.shape[:-1], -1).to(p.dtype)).double()

    def _w02_smooth(self, c: dict, n: str) -> torch.Tensor:
        """조음 변수 한 열 (B, T) — 조음기 시상수의 `W02_SMOOTH` 배 σ 가우시안 (§52.502). 음소별 입술 범위(`W02_LIP_BOUNDS`)가 있으면
        관에 들어가는 값에 단단히 건다 (§52.527)."""
        x = c[f"vtl_{n}"]
        tau = W02_TAU.get(n, 0.0)
        if W02_SMOOTH > 0.0 and tau > 0.0:
            x = _gauss_t(x, W02_SMOOTH * tau / (1000.0 * self.hop / self.fs))
        if SLEW_ON and W02_VMAX.get(n, 0.0) > 0.0:
            x = _slew(x, W02_VMAX[n] * self.hop / self.fs)
        if W02_LIP_BOUNDS and n in ("LP", "LD", "JA"):
            lo, hi = _lip_bounds(x.shape[-1], self.hop / self.fs, n)
            if lo is not None:
                lo = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
                hi = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
                x = torch.minimum(torch.maximum(x, lo), hi)
        return x

    def w02_anatomy(self, sur=None) -> torch.Tensor:
        """해부 13 개 [cm·도] — 학습 범위 [alo, ahi] 안 시그모이드, 좌표 0 이 W02 해부 (§52.501)."""
        sur = _w02a() if sur is None else sur
        alo, ahi, ax0 = sur.alo.double(), sur.ahi.double(), sur.ax0.double()
        if W02_AGE:
            return self.w02_age_anatomy(sur).clamp(alo, ahi)       # 대리 모형의 학습 범위가 모형의 유효 범위다
        u0 = torch.logit(((ax0 - alo) / (ahi - alo)).clamp(1e-3, 1 - 1e-3))
        u = torch.stack([getattr(self, f"w02a_u{i}").double() for i in range(N_W02_ANAT)])
        return alo + (ahi - alo) * torch.sigmoid(u0 + u)

    def w02_age(self) -> torch.Tensor:
        """나이 [년] — [W02_AGE_LO, W02_AGE_HI] 안 시그모이드, 좌표 0 이 W02_AGE_REF (`W02_AGE`)."""
        lo, hi, ref = W02_AGE_LO, W02_AGE_HI, W02_AGE_REF
        u0 = math.log((ref - lo) / (hi - ref))
        return lo + (hi - lo) * torch.sigmoid(self.w02_age_u.double() + u0)

    def w02_age_anatomy(self, sur=None) -> torch.Tensor:
        """나이 축 해부 13 개 [cm·도] (`W02_AGE`) — W02 해부를 여성 성장식의 나이 비율로 옮기고 개인차(길이 ±W02_DEV, 각 ±W02_DEV_DEG)를 더한다."""
        sur = _w02a() if sur is None else sur
        ax0 = sur.ax0.double()
        g = goldstein_female(self.w02_age())
        g0 = goldstein_female(torch.tensor(W02_AGE_REF, dtype=torch.float64))
        d = torch.tanh(torch.stack([getattr(self, f"w02d_u{i}").double() for i in range(N_W02_ANAT)]))
        a = ax0 * (g / g0) * (1.0 + W02_DEV * d)
        ang = ax0[-1] + (g[-1] - g0[-1]) + W02_DEV_DEG * d[-1]
        return torch.cat([a[:-1], ang.reshape(1)])

    def oral_min_area(self, c: dict, frac: float = 0.3) -> torch.Tensor:
        """관 모양의 구강 최소 면적 (B, T) [cm²] — 성문에서 `frac` 부터 입술까지의 부드러운 최소 (`voice.TUBE_AC`, §52.487).
        조음 모형(W02·Maeda)에서는 `a_c` 가 관 모양에 안 쓰이므로, 구강압·치찰 사전은 이 값을 봐야 한다."""
        A, _ = self.tube_frames(c)
        k = int(frac * A.shape[-1])
        return _softmin_all(A[..., k:], 0.01)

    def area_frames(self, c: dict) -> torch.Tensor:
        """프레임률 조음 → 프레임률 구강 면적 (B, T, N_SECT)."""
        coef = torch.stack([c[f"art{k}"] for k in range(1, N_ART + 1)], -1)          # (B, T, K)
        A = cos_area(coef, n=td.N_SECT)                                               # (B, T, N)
        x = (torch.arange(td.N_SECT, dtype=A.dtype, device=A.device) + 0.5) / td.N_SECT
        w = torch.exp(-0.5 * ((x - c["c_place"].unsqueeze(-1)) / CONS_SIGMA) ** 2)   # 협착 자리
        ac = c["a_c"].clamp_min(td.A_FLOOR).unsqueeze(-1)
        A = A * (1.0 - w) + w * _softmin(A, ac)
        if EPI_ON:
            # 후두개관: 성문에서 EPI_LEN_CM 까지 면적을 화자 상수 A_epi 로 부드럽게 좁힌다 (경계는 반 칸 폭의 로지스틱).
            # 3–5.5 kHz 몸통(원본보다 7 dB 모자람)과 3 kHz 부근 후두강 공진(Takemoto et al. 2006)의 자리다.
            L = c["tract_len"].clamp(10.0, 20.0).unsqueeze(-1)
            xc = x * L                                                               # 칸 가운데의 성문에서 거리 [cm]
            wep = torch.sigmoid((EPI_LEN_CM - xc) / (0.5 * L / td.N_SECT))
            A = A * (1.0 - wep) + wep * _softmin(A, torch.exp(self.log_epi_area))
        lip = (c["oral_open"].clamp(0.0, 1.0) ** 2).unsqueeze(-1)
        return torch.cat([A[..., :-LIP_CELLS], A[..., -LIP_CELLS:] * lip.clamp_min(1e-4)], -1)

    def volume_rate(self, A: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """칸마다 부피 변화 유량 dV/dt [cm³/s] (B, T, N) — **프레임률 중앙 차분**을 이어 연속 함수로 만든다.

        면적을 프레임 사이에서 직선으로 이으면 그 시간 미분이 1 ms 마다 계단이 되고, 풀이기 안에서 그 계단을 부피 유량으로 넣었더니
        조음값이 1 ms 에 0.2 % 만 떨려도 6~16 kHz 가 30 dB 넘쳤다(면적을 고정하면 목표와 맞았다). 조음기의 속도에는 계단이 없다.
        """
        V = A * (L / A.shape[-1]).unsqueeze(-1)
        dt_f = self.hop / self.fs
        Vp = torch.cat([V[:, :1], V, V[:, -1:]], 1)
        return (Vp[:, 2:] - Vp[:, :-2]) / (2.0 * dt_f)

    def glottal_area(self, c: dict, st: dict, pulse_phase: torch.Tensor, n_sim: int) -> torch.Tensor:
        """성문 면적 (B, n_sim) = 후부 틈 + 접촉(막부 쉼 틈 + ½ 떨림 폭 × 변위(φ)).

        * 후부 틈 = `ag_dc` (내전이 정한다) — 연골부라 떨지 않고 닫히지 않는다. 내전 수준 사전(§52.442)이 이 뜻으로 잰 값이다:
          옛 엔진에서 `ag_dc` 는 떨림과 따로 늘 열린 면적이었다. 막부 쉼 틈으로 읽었더니 사전이 붙든 내전 0.44 가 막부를 0.12 cm²
          벌려 놓아 떨림이 접촉 문턱에 걸렸고, 주기마다 닿았다 말았다 했다 (닫힌 몫 0.00↔0.28, 저역 σ 0.236 → 0.285, §52.477).
        * 막부 쉼 틈 = `GAP_PER_RD` × `rd_offset` — 음질 손잡이가 안쪽 압착(−)·기식(+)으로 옮긴다.
        * 떨림 폭 = `amp` × 최대 면적. 쉼 틈의 두 배를 넘으면 성대가 닿아 면적이 잘리고(닫힘의 꺾임 → 고역, 닫힌 몫), 못 넘으면
          매끈한 변조만 남는다. 열림 몫·닫힘의 날카로움을 표(Rd → 열림 몫)로 정하지 않는다 — 폭과 틈의 비에서 나온다
          (예전 모양은 진폭 규칙으로 꺾임을 섞어 첫머리 6–12 kHz 가 목표보다 +15 dB 밝았다).
        """
        up = lambda v: spline_up(v.double().unsqueeze(-1), self.hop * OS)[..., 0][:, :n_sim]      # 표본률은 float64 (§52.524)
        # 펄스 위상(48 kHz, 풀린 값)을 96 kHz 로 선형 보간한다 — 감긴 위상을 보간하면 경계에서 튄다.
        pp = pulse_phase.double()
        idx = torch.arange(n_sim, device=pp.device, dtype=pp.dtype) / OS
        i0 = idx.floor().long().clamp(max=pp.shape[-1] - 1)
        i1 = (i0 + 1).clamp(max=pp.shape[-1] - 1)
        fr = (idx - i0.to(idx.dtype))
        ph = pp[:, i0] * (1.0 - fr) + pp[:, i1] * fr
        if "pulse_shift" in c:
            # 연속 위상 보정 — 프레임률 제어를 스플라인으로 이어 위상에 더한다 (§52.480). 이산 지연 교정의 계단이 없다.
            ph = ph - 2.0 * math.pi * up(c["pulse_shift"]).to(ph.dtype)
        vg = 10.0 ** (c["voice_gain"] / 20.0)
        a_vib = up(st["amp"] * vg) * torch.exp(self.log_ag_max)
        if AVIB_MAX > 0.0:                               # 진동 면적의 생리 상한 (§52.524) — 성량 손잡이(±24 dB)로 0.7 cm² 넘게 벌어졌다
            a_vib = AVIB_MAX * torch.tanh(a_vib / AVIB_MAX)
        a_rest = up(GAP_PER_RD * c["rd_offset"])
        # **위상 원점 = 닫힘(접촉이 시작되는 순간)** (§52.478). 펄스 잠금은 원본의 성문 닫힘 시각을 위상 0 에 둔다. 변위 모양의 원점(가장
        # 닫힌 자리)에 두면 열림 몫(쉼 틈/떨림 폭)이 프레임마다 바뀔 때 닫힘 순간이 주기 안에서 떠다녀 원본 닫힘과 어긋난다. 닫히는 가지에서
        # 변위 = cos(πx) 이므로 접촉 문턱 r = −2·쉼틈/떨림폭 에서 x_c = arccos(r)/π, φ_c = skew + (1−skew)·x_c 만큼 밀어 닫힘을 원점에 붙인다.
        if PHASE_AT_CLOSURE:
            # 프레임률에서 재고 PHC_SMOOTH_MS 로 고른 뒤 스플라인으로 잇는다 — 표본마다 재면 첫머리(떨림 폭 → 0)에서 r 이 ±∞ 로
            # 뒤집혀 φ_c 가 0.4 주기씩 튀었고 적합이 발산했다(S_td12). 떨림 폭에 PHC_EPS 를 더하고 tanh 로 부드럽게 가둔다.
            avf = st["amp"] * vg * torch.exp(self.log_ag_max)
            r = -2.0 * GAP_PER_RD * c["rd_offset"] / (avf + PHC_EPS)
            rc = PHC_RMAX * torch.tanh(r / PHC_RMAX)
            skf = c["fold_skew"].clamp(0.3, 0.9) if "fold_skew" in c else FOLD_SKEW
            phc = skf + (1.0 - skf) * torch.arccos(rc) / math.pi
            k = max(1, int(round(PHC_SMOOTH_MS / (1000.0 * self.hop / self.fs))))
            phc = _box(phc.unsqueeze(-1), k)[..., 0]
            phi_c = up(phc)
            phi = torch.remainder(ph / (2.0 * math.pi) + phi_c.to(ph.dtype), 1.0)
        else:
            phi = torch.remainder(ph / (2.0 * math.pi), 1.0)
        eps = torch.exp(self.log_contact_eps).double() if EPS_FIT else None
        sk = up(c["fold_skew"].clamp(0.3, 0.9)).to(phi.dtype) if "fold_skew" in c else None
        return up(st["ag_dc"]) + contact(a_rest + 0.5 * a_vib * fold_shape(phi, sk), eps)

    def forward(self, c: dict, st: dict, pulse_phase: torch.Tensor, n_out: int) -> dict:
        """c: 프레임률 제어 (B, T) dict, st: `GlottalSource.physiology` 상태, pulse_phase: (B, ≥n_out) 풀린 위상 [rad].

        **앞 굴림** (`PRE_MS`): 구간은 이미 흐르고 있는 상태에서 시작한다. 정지한 관에 첫 폐압을 계단으로 걸면 유량이 튀어 딸깍 소리가 나고,
        방 잔향(RT 0.35 s)이 그것을 수백 ms 고역 꼬리로 번진다 — 038 첫머리 6–12 kHz 가 무음에서 목표 −86 dB 대신 −36 → −60 dB 로
        깔렸다(마른 관 출력 0–10 ms 가 −36 dB, 난류를 꺼도 같다). 그래서 첫 프레임 상태를 `PRE_MS` 만큼 앞에 붙이고(폐압은 앞 `PRE_RAMP_MS`
        동안 0 에서 올린다, 성문 위상은 첫 기울기로 뒤로 잇는다) 풀이기를 돌린 뒤 그 앞부분을 버린다.
        """
        T = next(v.shape[1] for v in c.values() if torch.is_tensor(v) and v.dim() >= 2)
        K = max(1, int(round((VF_PRE_MS if GLOTTIS == "vf" else PRE_MS) / 1000.0 * self.fs / self.hop)))
        self._n_pre_frames = K
        n_pre = K * self.hop

        def pad(d):
            return {k: (torch.cat([v[:, :1].expand(v.shape[0], K, *v.shape[2:]), v], 1)
                        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == T else v) for k, v in d.items()}
        pp = pulse_phase
        slope = (pp[:, self.hop] - pp[:, 0]) / self.hop
        pre = pp[:, :1] + slope[:, None] * (torch.arange(n_pre, dtype=pp.dtype, device=pp.device) - n_pre)
        st = dict(st)
        if GLOTTIS == "vf":
            qc = self.vf_qcorr
            st["vf_qcorr"] = (torch.ones_like(c["p_sub"]) if qc is None or qc.shape[-1] != T
                              else qc.to(c["p_sub"].dtype).to(c["p_sub"].device))
            pc_ = self.vf_pscorr
            st["vf_pscorr"] = (torch.ones_like(c["p_sub"]) if pc_ is None or pc_.shape[-1] != T
                               else pc_.to(c["p_sub"].dtype).to(c["p_sub"].device))
        r = self._run(pad(c), pad(st), torch.cat([pre, pp], 1), n_out + n_pre)
        if self.last_rec is not None:
            self.last_rec = [x[n_pre * OS:(n_pre + n_out) * OS] for x in self.last_rec]
        return {k: v[:, n_pre:n_pre + n_out] for k, v in r.items()}

    def vf_inputs(self, c: dict, st: dict, n_sim: int, pulse_phase: torch.Tensor | None = None):
        """자기 진동 성대의 표본률 입력 — (긴장 Q (B, n), 쉼 변위 (B, n, 2) [cm], 뒤쪽 틈 (B, n) [cm²]).
        `VF_BODY_DRIVE` 면 pulse_phase (B, ≥ n/OS) [rad, 풀린 값, 48 kHz] 로 몸체 구동점을 만든다."""
        H = self.hop * OS
        f0 = st["f0"].clamp(50.0, 800.0)
        Q = vf_tension(f0) * st["vf_qcorr"]
        add, rdo = c["adduction"].clamp(0.0, 1.0), c["rd_offset"]
        if VF_ADD_SMOOTH_MS > 0.0:
            sg = VF_ADD_SMOOTH_MS / (1000.0 * self.hop / self.fs)
            add, rdo = _gauss_t(add, sg), _gauss_t(rdo, sg)
        if SLEW_ON:
            fr_s = self.hop / self.fs
            add, rdo = _slew(add, VF_VMAX["adduction"] * fr_s), _slew(rdo, VF_VMAX["rd_offset"] * fr_s)
        u = torch.nn.functional.softplus(VF_R_K * (VF_R_MID - add)) / VF_R_K
        r2 = VF_R_SLOPE * u + VF_R_QUAD * u * u - VF_R_FLOOR
        if VF_R_MODAL > 0.0:                               # 모달 쉼 반틈새 (§52.532)
            r2 = r2 + VF_R_MODAL * _vf_r_modal_w(add)
        r1 = r2 + VF_CONV * torch.exp(VF_CONV_PER_RD * rdo)
        if VF_NS:
            # 모드 3: 목표 f0 → 늘어남 ε, 긴장 Q 는 음높이 맞춤의 잔차만. VR 열 = [r1, r2, 0, 계수 13 개]
            from ..physics import beam_membrane as BM
            ta = c["vf_ta"].clamp(0.0, 1.0) if "vf_ta" in c else torch.full_like(r1, 0.3)
            if VF_TA_SMOOTH_MS > 0.0:
                ta = _gauss_t(ta, VF_TA_SMOOTH_MS / (1000.0 * self.hop / self.fs))
            if SLEW_ON:
                ta = _slew(ta, VF_VMAX["vf_ta"] * self.hop / self.fs)
            if VF_NEURAL:                                   # 갑상피열근 긴장의 운동 단위 요동 (프레임률, 연축 ~15 ms)
                ta = (ta * self._neural("ta", ta.shape[-1], self.fs / self.hop, ta.dtype, ta.device)).clamp(0.0, 1.0)
            eps = vf_ns_eps(f0)
            e = {k: torch.exp(getattr(self, f"log_ns_{k}")).to(r1.dtype) for k in VF_NS_INIT}
            if VF_NS_HARD:
                # 화자 상수의 생리 범위를 단단히 (§52.529) — 범위 밖에서는 기울기 0
                e = {k: (v.clamp(VF_NS_HARD[k][0], VF_NS_HARD[k][1]) if k in VF_NS_HARD else v) for k, v in e.items()}
            _bulge = BM.STATIC_BULGE
            BM.STATIC_BULGE = True
            try:
                co = BM.ns_coefs_torch(eps.to(r1.dtype), ta.to(r1.dtype), td.VF_NSTRIP, scale_len=e["len"], scale_thick=e["thick"],
                                       k_scale=e["kc"], zeta=e["zeta"], shear_scale=e["shear"])
            finally:
                BM.STATIC_BULGE = _bulge
            co = torch.cat([co[..., :6], co[..., 6:7] * e["bulge"], co[..., 7:]], -1)
            Q3 = st["vf_qcorr"] * torch.ones_like(r1)
            u = spline_up(torch.cat([torch.stack([Q3, r1, r2, torch.zeros_like(r1)], -1), co], -1), H)[:, :n_sim]
            ch = (torch.exp(self.log_vf_chink) if VF_CHINK_FIX is None else torch.tensor(VF_CHINK_FIX, dtype=u.dtype)).expand(u.shape[0], n_sim)
            q = u[..., 0] * self._vf_jitter(n_sim, u.dtype, u.device)
            if VF_NEURAL:                                   # 윤상갑상근 긴장의 운동 단위 요동 (표본률) → 덮개 긴장
                q = q * self._neural("ct", n_sim, td.FS_SIM, q.dtype, q.device)
            if VF_BODY_DRIVE and pulse_phase is not None:
                u = self._body_drive(u, q, c, st, pulse_phase, n_sim)
            return q, u[..., 1:], ch
        cols = [Q, r1, r2]
        if VF_BODY:
            ta = c["vf_ta"].clamp(0.0, 1.0) if "vf_ta" in c else torch.full_like(r1, 0.5)
            if VF_TA_SMOOTH_MS > 0.0:
                ta = _gauss_t(ta, VF_TA_SMOOTH_MS / (1000.0 * self.hop / self.fs))
            if SLEW_ON:
                ta = _slew(ta, VF_VMAX["vf_ta"] * self.hop / self.fs)
            cols.append(torch.exp(math.log(VF_TA_RANGE) * (ta - 0.5)))       # 몸체 강성 배율 (갑상피열근의 대리, §52.500)
        u = spline_up(torch.stack(cols, -1), H)[:, :n_sim]
        ch = (torch.exp(self.log_vf_chink) if VF_CHINK_FIX is None else torch.tensor(VF_CHINK_FIX, dtype=u.dtype)).expand(u.shape[0], n_sim)
        return u[..., 0] * self._vf_jitter(n_sim, u.dtype, u.device), u[..., 1:], ch

    def _body_drive(self, u, q, c: dict, st: dict, pulse_phase: torch.Tensor, n_sim: int):
        """몸체를 펄스 위상의 구동점에 굳은 용수철로 맨다 (`VF_BODY_DRIVE`). u: 모드 3 의 표본률 열 [Q, r1, r2, 0, 계수 13 개]
        (계수 차례 `beam_membrane.NS_COEF_NAMES` — cK 2 · cB 3 · fB 6 · Lc 7 이 u 의 6 · 7 · 10 · 11 열)."""
        up = lambda v: spline_up(v.unsqueeze(-1), self.hop * OS)[..., 0][:, :n_sim]
        pp = pulse_phase.double()
        idx = torch.arange(n_sim, device=pp.device, dtype=pp.dtype) / OS
        i0 = idx.floor().long().clamp(max=pp.shape[-1] - 1)
        i1 = (i0 + 1).clamp(max=pp.shape[-1] - 1)
        fr = idx - i0.to(idx.dtype)
        ph = (pp[:, i0] * (1.0 - fr) + pp[:, i1] * fr).to(u.dtype)
        # 성량 손잡이(`voice_gain`, ±24 dB)는 옛 LF 음원의 물리 아닌 손잡이다 — 몸체 구동에서는 쓰지 않는다 (§52.524). m7 의 ㄲ 폐쇄 중 성량 손잡이가 앞 모음보다
        # 높은 +10 dB 로 몸체를 흔들어 떨림이 남았다 (원본 0.33 · 합성 0.55). 진폭은 생리 진폭(폐압 − 구강압 − 문턱압의 로지스틱)만 따른다.
        vg = 10.0 ** (c["voice_gain"] / 20.0) if VF_DRIVE_VG else 1.0
        a_vib = up(st["amp"] * vg).to(u.dtype) * torch.exp(self.log_ag_max).to(u.dtype)       # 운동학 판과 같은 진동 면적 [cm²]
        cK, cB, fB, Lc = u[..., 6], u[..., 7], u[..., 10], u[..., 11]
        xb = VF_BODY_DRIVE_XMAX * torch.tanh(0.5 * a_vib / Lc.clamp_min(1e-3) / VF_BODY_DRIVE_XMAX)
        x_st = fB / (cB * q).clamp_min(1e-9)
        xd = x_st - xb * torch.cos(ph)
        kd = VF_BODY_DRIVE_K * (cB + cK)
        if VF_BODY_DRIVE_FREE > 0.0:                    # 구동이 약할 때는 몸체를 풀어 둔다 (§52.533)
            am = up(st["amp"]).to(u.dtype)
            kd = kd * am * am / (am * am + VF_BODY_DRIVE_FREE ** 2)
        return torch.cat([u[..., :7], (cB + kd).unsqueeze(-1), u[..., 8:10], (fB + kd * q * xd).unsqueeze(-1), u[..., 11:]], -1)

    def _neural(self, muscle: str, n: int, fs: float, dtype, device) -> torch.Tensor:
        """후두근 상대 긴장 (n,) — 운동 단위 모형 (`VF_NEURAL`), 씨앗 고정, 같은 길이면 다시 쓰지 않는다."""
        key = (muscle, n, fs, VF_NEURAL_NMU, VF_NEURAL_TREMOR, self.seed)
        cache = self.__dict__.setdefault("_neural_cache", {})
        if key not in cache:
            from ..physics import neural as _nr
            cache[key] = _nr.tension(n, fs, muscle, n_mu=VF_NEURAL_NMU, tremor_hz=VF_NEURAL_TREMOR[0],
                                     tremor_depth=VF_NEURAL_TREMOR[1], seed=int(self.seed))
        return torch.as_tensor(cache[key], dtype=dtype, device=device)

    def _vf_jitter(self, n: int, dtype, device) -> torch.Tensor | float:
        """성대 긴장의 **주기 규모 무작위 요동** (§52.515) — exp(σ ξ), ξ = 단위 표준편차 가우스 잡음을 `VF_JIT_HZ` 2 차 저역통과 (상관 시간 ≈ 한 주기).
        신경근 운동 단위 발화의 긴장 요동(Titze 1991)과 난류 힘이 만드는 주기별 떨림 — 완전 주기 성대는 5–16 kHz 배음을 끝까지 고른 빗살(가로줄)로
        세운다; 원본은 고역 배음이 흩어져 있다 (앞 세대 §52.3: 지터 2.5 % 만으로 8–12 kHz 배음성 0.71 → 0.06). 실현은 시드로 고정 (재현), 0 이면 끔."""
        if VF_JIT_SIGMA <= 0.0:
            return 1.0
        key = (n, str(device), dtype, VF_JIT_SIGMA, VF_JIT_HZ, self.seed)
        if getattr(self, "_vfj_key", None) != key:
            import numpy as np
            from scipy.signal import butter, sosfilt
            rng = np.random.default_rng(1_000_003 + int(self.seed))
            xi = sosfilt(butter(2, VF_JIT_HZ, "low", fs=td.FS_SIM, output="sos"), rng.standard_normal(n + 4096))[4096:]
            xi = xi / max(float(xi.std()), 1e-12)
            self._vfj_key, self._vfj = key, torch.as_tensor(np.exp(VF_JIT_SIGMA * xi), dtype=dtype, device=device)
        return self._vfj

    def vf_statics(self) -> torch.Tensor:
        if VF_NS:
            from ..physics import beam_membrane as BM
            vs = torch.as_tensor(BM.vf_static_ns(), dtype=torch.float64)
            if getattr(BM, "LR_FOLDS", False) and vs.shape[0] > 22:
                dq = VF_LR_DQ_MAX * torch.sigmoid(self.ns_lr_u_q.double())
                dm = VF_LR_DM_MAX * torch.tanh(self.ns_lr_u_m.double())
                vs = torch.cat([vs[:21], dq.reshape(1), dm.reshape(1)])
            return vs
        return vf_static(*(torch.exp(q).double() for q in (self.log_vf_len, self.log_vf_thick, self.log_vf_mk, self.log_vf_damp)))

    @staticmethod
    def measure_f0(rec, fs: float = td.FS_SIM, amin: float = 0.01):
        """풀이 진단 (n, 9) → (주기 경계 표본 (K,), 경계마다 주기 [s] (K,)) — 아래 성문 면적에서 10 ms 이동 평균을 뺀 값이
        위로 0 을 지나는 곳 (닫히든 안 닫히든 잡힌다). 면적 흔들림(마루 − 골)이 `amin` 이 안 되는 주기는 떨림이 아니다."""
        import numpy as np
        a = rec[:, 7]
        w = max(1, int(0.010 * fs))
        k = np.ones(w) / w
        z = a - np.convolve(a, k, mode="same")
        up = np.where((z[:-1] < 0.0) & (z[1:] >= 0.0))[0]
        if up.size < 2:
            return np.zeros(0, int), np.zeros(0)
        per = np.diff(up) / fs
        ok = np.array([a[i:j].max() - a[i:j].min() >= amin for i, j in zip(up[:-1], up[1:])])
        ok &= (per > 1.0 / 900.0) & (per < 1.0 / 50.0)
        return up[1:][ok], per[ok]

    def _run(self, c: dict, st: dict, pulse_phase: torch.Tensor, n_out: int) -> dict:
        n_sim = n_out * OS
        Af, Lf = self.tube_frames(c)
        H = self.hop * OS
        Q = spline_up(self.volume_rate(Af, Lf), H)[:, :n_sim] * VOL_FLOW
        A = spline_up(Af, H)[:, :n_sim]
        vf = GLOTTIS == "vf"
        if vf:
            VQ, VR, Ag = self.vf_inputs(c, st, n_sim, pulse_phase if VF_BODY_DRIVE else None)
            if self.vf_qcyc is not None and not VF_BODY_DRIVE:    # 앞 굴림 몫은 1, 출력 몫은 주기 시각 맞춤의 배율 (몸체 구동이면 시각은 몸체가)
                import numpy as np
                npre = getattr(self, "_n_pre_frames", 0) * self.hop * OS
                qq = np.ones(n_sim)
                k = min(n_sim - npre, self.vf_qcyc.shape[0])
                if k > 0:
                    qq[npre:npre + k] = self.vf_qcyc[:k]
                VQ = VQ * torch.as_tensor(qq, dtype=VQ.dtype, device=VQ.device).unsqueeze(0)
        else:
            Ag = self.glottal_area(c, st, pulse_phase, n_sim)
        ps_f = c["p_sub"].clamp_min(0.0)
        if vf and "vf_pscorr" in st:
            ps_f = ps_f * st["vf_pscorr"]
        if vf and VF_PS_SMOOTH_MS > 0.0:
            ps_f = _gauss_t(ps_f, VF_PS_SMOOTH_MS / (1000.0 * self.hop / self.fs))
        if vf and SLEW_ON:
            ps_f = _slew(ps_f, VF_VMAX["p_sub"] * self.hop / self.fs)
        vel = c["velum"].clamp(0.0, 1.0)
        if VELUM_SMOOTH_MS > 0.0:                      # 연구개의 생리 동역학 (§52.530)
            vel = _gauss_t(vel, VELUM_SMOOTH_MS / (1000.0 * self.hop / self.fs))
        if VELUM_BOUNDS:                               # 입 장애음의 연인두 닫힘 (§52.530)
            vh = _velum_hi(vel.shape[-1], self.hop / self.fs)
            if vh is not None:
                vel = torch.minimum(vel, torch.as_tensor(vh, dtype=vel.dtype, device=vel.device))
        slow = spline_up(torch.stack([Lf, ps_f, vel], -1), H)[:, :n_sim]
        L, Ps, Av = slow[..., 0], slow[..., 1] * td.CMH2O, slow[..., 2] * td.A_VMAX
        Df = self.teeth_frames(c)
        Dsim = None if Df is None else spline_up(Df.to(Lf.dtype).unsqueeze(-1), H)[..., 0][:, :n_sim].detach().double().cpu().numpy()
        nr = int(PRE_RAMP_MS / 1000.0 * self.fs * OS)
        ramp = 0.5 * (1.0 - torch.cos(math.pi * (torch.arange(n_sim, dtype=Ps.dtype, device=Ps.device) / nr).clamp(max=1.0)))
        Ps = Ps * ramp
        outs = []
        gas = []
        recs = []
        evs = []
        pll = None
        if vf and self.vf_pll_t is not None and not VF_BODY_DRIVE:        # 몸체 구동이면 시각은 몸체가 쥔다 (§52.523)
            import numpy as np
            npre96 = getattr(self, "_n_pre_frames", 0) * self.hop * OS
            pt = self.vf_pll_t
            if pt.size >= 2 and pt[0] < 0.015 * self.fs * OS and npre96 > 0:
                # 구간이 떨림 가운데서 시작하면 첫 주기 길이로 앞 굴림 속으로 시각표를 잇는다 — 고리가 출력 전에 잠기도록 (폐압이 오르는
                # 앞 30 % 는 비운다). 없으면 출력 첫 몇 주기가 풀린 채 나온다 (VKC 오차의 29 % 가 첫 50 ms, §52.493).
                p0 = float(pt[1] - pt[0])
                k = int((pt[0] + 0.7 * npre96) // p0)
                if k > 0:
                    pt = np.r_[pt[0] - p0 * np.arange(k, 0, -1), pt]
            pll = pt + npre96
        for b in range(A.shape[0]):
            if _os_env_get("TD_DUMP_IN") and not getattr(self, "_dumped_in", False):    # 진단: 관 입력 그대로 (§52.524)
                import numpy as _np
                self._dumped_in = True
                _np.savez(_os_env_get("TD_DUMP_IN"), A=A[b].double().detach().numpy(), L=L[b].double().detach().numpy(),
                          Ag=Ag[b].double().detach().numpy(), Ps=Ps[b].double().detach().numpy(), Av=Av[b].double().detach().numpy(),
                          ng=float(torch.exp(self.log_noise_g)), nc=float(torch.exp(self.log_noise_c)), Q=Q[b].double().detach().numpy(),
                          VQ=VQ[b].double().detach().numpy() if vf else _np.zeros(1), VR=VR[b].double().detach().numpy() if vf else _np.zeros(1),
                          VS=self.vf_statics().detach().numpy() if vf else _np.zeros(1), seed=self.seed, PP=pulse_phase.double().detach().numpy(),
                          teeth=_np.zeros(1) if Dsim is None else _np.asarray(Dsim[b], dtype=float))
            y, gab = td.tube_torch(A[b].double(), L[b].double(), Ag[b].double(), Ps[b].double(), Av[b].double(),
                                   torch.exp(self.log_noise_c if TIE_NOISE else self.log_noise_g).double(), torch.exp(self.log_noise_c).double(), seed=self.seed,
                                   Q=Q[b].double(), vf=(VQ[b].double(), VR[b].double(), self.vf_statics()) if vf else None,
                                   return_area=True, pll=pll, vf_mode=3 if VF_NS else (2 if VF_BODY else 1), fs=td.FS_SIM,
                                   teeth=None if Dsim is None else Dsim[b])
            outs.append(y)
            gas.append(gab)
            recs.append(td.TubeFn.last_rec)
            evs.append(getattr(td.TubeFn, "last_ev", None))
            if vf and pll is not None:                                     # 고리가 건 log 긴장 배율 (진단)
                import numpy as np
                self.last_pll_u = np.log(td.TubeFn.last_q / np.maximum(VQ[b].detach().double().cpu().numpy(), 1e-9))
        self.last_rec = recs
        import os as _os
        if _os.environ.get("TD_DUMP"):
            # 진단 (§52.505): 성문·관 기록을 1 ms 로 — 유량·성문 면적·관의 가장 좁은 면적과 그 칸·구강 첫 칸 압력
            import numpy as _np
            _k = int(td.FS_SIM / 1000)
            _A = A[0].detach().double().cpu().numpy()
            _npre = getattr(self, "_n_pre_frames", 0) * self.hop * OS
            _np.savez(_os.environ["TD_DUMP"], rec=recs[0][_npre::_k], amin=_A[_npre::_k].min(-1), imin=_A[_npre::_k].argmin(-1),
                      ps=Ps[0].detach().double().cpu().numpy()[_npre::_k], A=_A[_npre::_k], L=L[0].detach().double().cpu().numpy()[_npre::_k])
        if _os.environ.get("TD_DUMP_FULL"):
            # 진단 (§52.515): 관 출력(방사 음압)과 성문 유량을 내부 표본률 그대로 — 가로줄이 어느 단계에서 생기는지 가른다
            import numpy as _np
            _npre = getattr(self, "_n_pre_frames", 0) * self.hop * OS
            _np.savez(_os.environ["TD_DUMP_FULL"], out=outs[0].detach().double().cpu().numpy()[_npre:],
                      ug=recs[0][_npre:, 0], ga=recs[0][_npre:, 7], fs=td.FS_SIM)
        # 잡은 닫힘 시각 — 출력 구간 96 kHz 표본으로
        npre96 = getattr(self, "_n_pre_frames", 0) * self.hop * OS
        self.last_ev = [None if e is None else e - npre96 for e in evs] if vf else None
        if vf:
            import numpy as np
            Ag = torch.stack(gas, 0).to(A.dtype)                  # 미분 가능한 성문 면적 (성문 역산 항, §52.492)
        y96 = torch.stack(outs, 0) * OUT_SCALE
        # 96 → 48 kHz: 창 씌운 싱크 저역통과 (21 kHz) 뒤 솎기
        k = _lp_kernel(y96.dtype, y96.device)
        y = torch.nn.functional.conv1d(y96.unsqueeze(1), k, padding=k.shape[-1] // 2)[:, 0, ::OS][:, :n_out]
        return dict(audio=y.to(st["amp"].dtype), glottal_area=Ag[:, ::OS][:, :n_out])


def _gauss_t(x: torch.Tensor, sig: float) -> torch.Tensor:
    """(B, T) 을 시간축 가우시안 σ [틀] 로 (끝은 복제) — 미분 가능."""
    r = max(1, int(3.0 * sig))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / sig) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    xp = torch.nn.functional.pad(x.unsqueeze(1), (r, r), mode="replicate")
    return torch.nn.functional.conv1d(xp, k)[:, 0]


_LP = {}


def _lp_kernel(dtype, device, taps: int | None = None, fc: float = 21000.0):
    # 탭 수는 과표본 배율에 비례 — 전이 폭을 kHz 로 같게 (192 kHz 에서 63 탭이면 전이가 두 배로 넓어져 13 kHz 위를 깎는다)
    taps = 32 * OS - 1 if taps is None else taps
    key = (dtype, str(device), taps, td.FS_SIM)
    if key not in _LP:
        n = torch.arange(taps, dtype=torch.float64) - (taps - 1) / 2
        h = torch.where(n == 0, torch.tensor(2 * fc / td.FS_SIM, dtype=torch.float64),
                        torch.sin(2 * math.pi * fc / td.FS_SIM * n) / (math.pi * n + (n == 0).double()))
        h = h * torch.kaiser_window(taps, periodic=False, beta=8.0, dtype=torch.float64)
        h = h / h.sum()
        _LP[key] = h.to(dtype=dtype, device=device).view(1, 1, -1)
    return _LP[key]
