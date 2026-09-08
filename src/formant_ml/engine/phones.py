"""한국어 음소 제스처 라이브러리 — "텍스트 -> ms 단위 물리 factor 스크립트" 의 규칙판.

모든 목표값은 `SpeakerProfile` 에서 온다 (녹음 계측, docs/MEASUREMENTS.md). 여기에는
**구조**(어떤 파라미터가 언제 어떤 순서로 움직이는가)만 있다.

* 파라미터마다 시간축이 독립이다. F3 를 F1 보다 먼저 보낼 수 있다.
* 조음 방식은 스위치가 아니라 `a_c`(협착 면적) 궤적이다: 탄음은 짧은 골, 마찰음은 좁은
  유지, 파찰음은 폐쇄 → 방전(구강압 물리가 버스트를 낸다) → 마찰.
* 과도음 템플릿은 프로파일이 켤 때만 붙는다. 계측된 탄음에는 고역 버스트가 없었다.
"""
from __future__ import annotations

from .control import ControlTrack, track_from_keyframes
from .profile import DEFAULT_PROFILE, SpeakerProfile


# 마찰 세기의 기준점: `fric_gain = 1` 일 때 마찰음이 뒤따르는 모음보다 몇 dB 인가.
# 소스 스펙트럼을 실측에 맞춰 바꿀 때마다 이 값이 움직이므로 **한 곳에** 둔다.
# 재보는 법: profiles 의 level_db 를 이 값으로 두고 scripts/ab_male.py 의 "자음 레벨" 을 읽는다.
SIB_REF_DB = -25.2


class Builder:
    """키프레임을 시간순으로 쌓는다. 파라미터별 독립 시간축."""

    def __init__(self, profile: SpeakerProfile | None = None, f0_hz: float | None = None,
                 p_sub: float | None = None):
        self.p = profile or DEFAULT_PROFILE
        self.kf: list[dict] = []
        self.t = 0.0
        self.f0 = float(f0_hz or self.p.f0_nominal)
        self.ps = float(p_sub or self.p.p_sub)
        self.add(0.0, p_sub=0.0, adduction=0.6, tension=0.5, f0_target=self.f0,
                 a_c=3.0, c_place=0.9, velum=0.0, oral_open=1.0, tract_gain=1.0,
                 obstacle=0.0, back_leak=0.3)

    # ---------------------------------------------------------- 저수준
    def add(self, t, **kw):
        self.kf.append(dict(t=round(t, 5), **kw))

    def event(self, t, name, **kw):
        self.kf.append(dict(t=round(t, 5), event=dict(name=name, **kw)))

    def V(self, name):
        return self.p.vowels[name]

    def ms(self, x):
        return float(x) / 1000.0

    # ---------------------------------------------------------- 분절
    def silence(self, dur=None):
        dur = self.ms(self.p.timing["silence_ms"]) if dur is None else dur
        self.add(self.t, p_sub=0.0); self.t += dur; self.add(self.t, p_sub=0.0)
        return self

    def vowel(self, name, dur=None, f0_end=None, final=False):
        f1, f2, f3 = self.V(name)
        dur = self.ms(self.p.timing["final_vowel_ms" if final else "vowel_ms"]) if dur is None else dur
        self.add(self.t, p_sub=self.ps, adduction=0.6, a_c=3.0, f1=f1, f2=f2, f3=f3,
                 tract_gain=1.0, f0_target=self.f0, back_leak=0.3)
        self.t += dur
        f0e = f0_end or self.f0
        self.add(self.t, f1=f1, f2=f2, f3=f3, f0_target=f0e, p_sub=self.ps)
        self.f0 = f0e
        return self

    def lateral(self, next_vowel="a", hold=None, release=None, onset=True):
        """설측음 [l] (어두 또는 종성→다음 분절). 측지 노치 + 자세 유지 + F3 선행 전이."""
        L = self.p.lateral
        l1, l2, l3 = L["f"]
        v1, v2, v3 = self.V(next_vowel)
        hold = self.ms(L["hold_ms"]) if hold is None else hold
        release = self.ms(L["release_ms"]) if release is None else release
        z1, z2 = L["zeros"]
        t0 = self.t
        self.add(t0 - 0.03, lat_z1=z1, lat_z2=z2, lat_mix=0.0)        # 영점 자리는 미리 잡아 둔다
        self.add(t0, p_sub=self.ps, adduction=0.6, a_c=0.6, c_place=0.85,
                 f1=l1, f2=l2, f3=l3, bw1=60, bw2=140, bw3=240,
                 lat_z1=z1, lat_z2=z2, lat_bw=L["zero_bw"], lat_mix=1.0, tract_gain=L["gain"])
        if L.get("transient_amp", 0.0) > 0 and onset:
            self.event(t0 + 0.004, "tongue_contact", amp=L["transient_amp"])
        t1 = t0 + hold
        self.add(t1, f1=l1, f2=l2, f3=l3, lat_z1=z1, lat_z2=z2, lat_mix=1.0,
                 tract_gain=L["gain"], a_c=0.6, bw1=60, bw2=140, bw3=240)
        self.add(t1 - 0.012, f3=l3)                                   # F3 먼저 출발
        self.add(t1 + release * 0.6, f3=v3, lat_mix=0.0)              # 측지가 연속으로 닫힌다
        self.add(t1 + release, f1=v1, f2=v2, tract_gain=1.0, a_c=3.0,
                 bw1=0.0, bw2=0.0, bw3=0.0, c_place=0.9)
        self.t = t1 + release
        return self

    def tap(self, prev_vowel="a", next_vowel="a"):
        """모음 사이 [ɾ]: 짧은 골. 발성은 이어진다. 깊이·길이는 화자 프로파일."""
        T = self.p.tap
        a1, a2, a3 = self.V(prev_vowel); b1, b2, b3 = self.V(next_vowel)
        l1, l2, l3 = T["f"]
        # F2 로커스는 문맥 의존: 앞뒤 모음 F2 평균 쪽으로 (치경 로커스 ~1700~2200)
        l2 = 0.5 * l2 + 0.25 * (a2 + b2)
        ap, cl, rl = self.ms(T["approach_ms"]), self.ms(T["closure_ms"]), self.ms(T["release_ms"])
        t0 = self.t
        self.add(t0, f1=a1, f2=a2, f3=a3, tract_gain=1.0, a_c=3.0, p_sub=self.ps)
        tc = t0 + ap
        self.add(tc, f1=l1, f2=l2, f3=l3, tract_gain=T["gain"], a_c=0.25, c_place=0.87,
                 bw1=70, bw2=150, bw3=240)
        amp = float(T.get("transient_amp", 0.0))
        if amp > 0:
            self.event(tc + 0.002, "tongue_contact", amp=amp, rate=1.1)
        tr = tc + cl
        self.add(tr, f1=l1, f2=l2, f3=l3, tract_gain=T["gain"], a_c=0.25, bw1=70, bw2=150, bw3=240)
        if amp > 0:
            self.event(tr + 0.003, "tongue_release", amp=amp * 0.7)
            self.event(tr + 0.022, "tongue_floor", amp=amp * 0.5, rate=0.9)
        self.add(tr + rl * 0.7, f3=b3)
        self.add(tr + rl, f1=b1, f2=b2, tract_gain=1.0, a_c=3.0,
                 bw1=0.0, bw2=0.0, bw3=0.0, c_place=0.9)
        self.t = tr + rl
        return self

    def sibilant(self, next_vowel="a", tense=False, dur=None):
        """/ㅅ, ㅆ/ — **성문 제스처와 구강 제스처가 서로 다른 시간축을 갖는다.**

        v2 초판은 성문 개대를 마찰 구간에 맞춰 열고 닫았다. 그래서 마찰과 발성이 서로
        모르는 두 사건이 되고("따로 논다"), 마찰이 끝나는 순간 소리가 갈아 끼워졌다.

        실측(같은 화자, 5 ms 프레임):
          * 마찰로 들어갈 때 유성도(HNR)가 **35 ms 에 걸쳐** 9 → 0 dB 로 꺼진다.
          * 마찰이 끝나면 발성은 15~20 ms 만에 돌아오는데,
            **고역 잡음은 그 뒤로 50~70 ms 더 이어지며 감쇠한다**(기식 꼬리).
          * 그동안 중역(0.8~2.5 kHz)은 55 ms 에 걸쳐 올라온다 (혀가 협착을 푸는 시간).

        그래서 성문은 협착보다 **먼저 열리고 늦게 닫힌다**(Löfqvist & Yoshioka 가 무성
        마찰음에서 관찰한 순서). 기식 꼬리는 우리가 그리지 않는다 — 성문이 아직 열려 있고
        협착이 풀려 ΔPg 가 커지므로 `glottis` 의 기식 항이 저절로 낸다.
        """
        S = self.p.sibilant
        v1, v2, v3 = self.V(next_vowel)
        dur = self.ms(S["tense_dur_ms" if tense else "dur_ms"]) if dur is None else dur
        lvl = S.get("tense_level_db", S["level_db"]) if tense else S.get("lax_level_db", S["level_db"])
        gain = 10 ** ((lvl - SIB_REF_DB) / 20.0)
        l1, l2, l3 = S.get("locus", [470.0, 1800.0, 2700.0])
        l2 = 0.5 * l2 + 0.5 * v2                       # 로커스는 뒤따르는 모음 쪽으로 당겨진다
        # 협착 전이 시간. **길이에 비례해야 한다.**
        #
        # v1 은 혀 제스처를 `tongue_constriction` 으로 그렸다: 고원이 62 %
        # (`TONGUE_SUSTAIN_HOLD`), 정점이 57 % (`TONGUE_CLOSE_FRAC`) 이므로 폐쇄가
        # 21.7 %, 해제가 16.3 % 다. 그 비대칭이 곧 실측 상승/하강비 1.28~1.35 이고,
        # "가청 포락선의 절반 이상이 페이드 인" 이 거기서 나온다.
        #
        # v2 는 40 / 18 ms 를 **절대값**으로 박아 두었다. 130 ms CV 에서는 31 / 14 %
        # 라 대충 맞지만, 600 ms 로 끄는 /s/ 에서는 6.7 / 3 % 가 되어 혀가 순식간에
        # 자세를 잡고 나머지를 고원으로 버틴다 — 페이드 인이 사라진다.
        # 측정 (고역 11~16 kHz 가 중역 2~4 kHz 보다 늦게 서는 폭, 긴 /s/):
        #   rise 40 ms +11 ms / 200 ms +53 ms / 500 ms +235 ms  (사람 127~296 ms)
        #
        # 그래서 비례로 두되 예전 값을 **바닥**으로 남긴다 — 짧은 CV 는 그대로다.
        rise = max(0.040, 0.217 * dur)
        fall = max(0.018, 0.163 * dur)
        # 경음은 성문을 덜 벌리고 빨리 닫는다 (Cho·Jun·Ladefoged 2002: /s'/ 의 성문 개대가 작다)
        adduct = 0.10 if tense else 0.06
        lead = self.ms(S.get("abduct_lead_ms", 35))
        lag = self.ms(S.get("abduct_lag_ms", 25 if tense else 55))
        t0 = self.t
        self.add(t0 - lead, adduction=0.6)                        # 성문이 먼저 열리기 시작
        self.add(t0, p_sub=self.ps, adduction=adduct, a_c=0.35, c_place=0.91,
                 back_leak=S["back_leak"], front_len=self.p.sib_front_len_cm,
                 obstacle=S["obstacle"], fric_gain=gain, oral_open=1.0,
                 f1=l1, f2=l2, f3=l3, tract_gain=0.9)
        # **짧은 음절은 목표까지 못 간다** (undershoot). 뒤따르는 모음을 예기해
        # 혀가 협착을 덜 만들고, 그만큼 제트가 굵어 앞니 다이폴이 약해지며 앞공동
        # 극이 드러난다. v1 실측: 긴 /s/ 의 목표가 0.050 cm² 인데 CV 는 0.11 —
        # 2.2 배 넓다 (`TONGUE_CV_A_MIN`). 그 값에서 봉우리 5276 Hz / 4~6 kHz
        # 50.2 % 로 실측(4673~5556 Hz, 43~48 %) 안에 들어왔고, 0.050 을 쓰면
        # 10218 Hz / 17 % 로 지속음과 구분이 안 됐다.
        #
        # 130 ms 급 음절에서 2.2 배, 400 ms 이상이면 목표 그대로. 그 사이는 선형.
        under = 1.0 + 1.2 * min(max((0.40 - dur) / (0.40 - 0.13), 0.0), 1.0)
        a_min = S["a_min"] * under
        self.add(t0 + rise, a_c=a_min)
        self.add(t0 + dur - fall, a_c=a_min, adduction=adduct)
        self.add(t0 + dur, a_c=3.0, obstacle=0.0, front_len=0.0, back_leak=0.3, fric_gain=gain)
        self.add(t0 + dur + lag, adduction=0.6, fric_gain=1.0)    # 성문은 늦게 닫힌다 → 기식 꼬리
        self.add(t0 + dur + 0.055, f1=v1, f2=v2, f3=v3, tract_gain=1.0)
        self.t = t0 + dur
        return self

    def affricate(self, next_vowel="a", aspirated=False, tense=False):
        """/ㅈ, ㅊ, ㅉ/ — 폐쇄(구강압 축적) → 방전 버스트 → 마찰 → 기식 → 모음.

        실측: 개시가 10 ms 안에 −70 → −30 dB 로 서고, 그 뒤 100 ms 에 걸쳐 감쇠한다.
        ㅊ 의 해제 뒤 스펙트럼은 100 Hz~5 kHz 가 거의 **평탄**했다 — 마찰이 아니라 성문
        기식이 지배한다는 뜻이다. 그래서 격음은 성문을 협착보다 훨씬 늦게 닫는다.
        """
        S = self.p.sibilant
        v1, v2, v3 = self.V(next_vowel)
        closure = 0.060
        fric = 0.110 if aspirated else 0.080
        front = self.p.sib_front_len_cm * 1.25
        adduct = 0.10 if tense else (0.03 if aspirated else 0.07)
        lag = self.ms(20 if tense else (110 if aspirated else 55))
        lvl = S.get("aspirated_level_db" if aspirated else "affricate_level_db", -7.0)
        gain = 10 ** ((lvl - SIB_REF_DB) / 20.0)
        l1, l2, l3 = S.get("locus", [470.0, 1800.0, 2700.0])
        t0 = self.t
        self.add(t0 - 0.035, adduction=0.6, oral_open=1.0)
        self.add(t0, p_sub=self.ps, adduction=adduct, a_c=0.0, c_place=0.88,
                 back_leak=S["back_leak"], front_len=front, obstacle=0.5,
                 fric_gain=gain, oral_open=0.02,
                 f1=l1, f2=0.5 * l2 + 0.5 * v2, f3=l3, tract_gain=0.85)
        t1 = t0 + closure
        self.add(t1, a_c=0.0, adduction=adduct, oral_open=0.02)
        self.add(t1 + 0.004, a_c=0.10, oral_open=1.0)                    # 급한 해제
        self.add(t1 + fric, a_c=0.35 if aspirated else 0.6)
        self.add(t1 + fric + 0.02, a_c=3.0, obstacle=0.0, front_len=0.0, back_leak=0.3, fric_gain=gain)
        self.add(t1 + fric + lag, adduction=0.6, fric_gain=1.0)
        self.add(t1 + fric + 0.05, f1=v1, f2=v2, f3=v3, tract_gain=1.0)
        self.t = t1 + fric
        return self

    def nasal(self, place="n", next_vowel="a", dur=None, coda=False, prev_vowel="a"):
        """비음 — 두 제스처가 **서로 다른 시간축**을 갖는다 (실측이 시킨 구조).

          (1) 연구개 개방 `velum`  : 폐쇄보다 **먼저 열리고**(선행 비음화) 해제 뒤에도
                                     한참 닫힌다(후행 비음화). 60 / 90 ms.
          (2) 구강 폐쇄 `oral_open`: 35 ms 에 닫히고 18 ms 에 열린다 (실측: 중역이
                                     내려가는 데 45 ms, 되돌아오는 데 15 ms).

        두 경로가 병렬이라 저역은 끊기지 않는다. 폐쇄 위치는 **측지 영점**으로만 들어간다
        (ㅁ 1250 / ㄴ 1700 / ㅇ 2400 Hz) — 머머의 포먼트를 place 마다 손으로 그리지 않는다.
        """
        N = self.p.nasal
        v1, v2, v3 = self.V(next_vowel)
        a1, a2, a3 = self.V(prev_vowel)
        dur = self.ms(N["dur_ms"]) if dur is None else dur
        lead, lag = self.ms(N["velum_lead_ms"]), self.ms(N["velum_lag_ms"])
        clo, rel = self.ms(N["closure_ms"]), self.ms(N["release_ms"])
        nz = N["zero_hz"][place]
        m1, m2, m3 = N["f_murmur"]
        # 구강 측지의 공명(닫힌 구강)이 머머의 구강 성분을 만든다. 폐쇄 위치가 뒤일수록 짧다.
        oral_f2 = {"m": 950.0, "n": 1350.0, "ng": 1800.0}[place]
        # 폐쇄 위치별 세기 (실측 A/B: ㅁ 머머가 ㄴ 보다 3 dB 크다). 비강 분기에만 곱한다.
        ngain = N["gain"] * N.get("place_gain", {"m": 2.0, "n": 1.0, "ng": 1.2})[place]
        t0 = self.t                                        # 구강 폐쇄가 완성되는 시각
        self.add(t0 - clo - lead, velum=0.0)               # 연구개는 폐쇄보다 먼저 열린다
        self.add(t0 - clo, velum=0.9, nasal_f=N["poles"][0], nasal_f2=N["poles"][1],
                 nasal_f3=N["poles"][2], nasal_z=nz, nasal_damp=N["damp"], nasal_gain=ngain,
                 f1=a1 if not coda else a1, f2=a2, f3=a3, oral_open=1.0)
        self.add(t0, velum=1.0, oral_open=0.02, p_sub=self.ps, adduction=0.6,
                 f1=m1, f2=oral_f2, f3=m3, bw1=250, bw2=350, bw3=450, tract_gain=1.0)
        t1 = t0 + dur
        self.add(t1, velum=1.0, oral_open=0.02, f1=m1, f2=oral_f2, f3=m3,
                 bw1=250, bw2=350, bw3=450)
        self.add(t1 + rel, oral_open=1.0, f1=v1, f2=v2, f3=v3, bw1=0, bw2=0, bw3=0)
        self.add(t1 + lag, velum=0.0)                      # 후행 비음화: 천천히 닫힌다
        self.t = t1 + rel
        return self

    def build(self, tail=None) -> ControlTrack:
        self.t += self.ms(self.p.timing["silence_ms"]) if tail is None else tail
        self.add(self.t, p_sub=0.0)
        return track_from_keyframes(self.kf, seconds=self.t, frame_ms=1.0)


# ------------------------------------------------------------------ 데모 단어
def _b(profile, f0, p_sub):
    return Builder(profile, f0, p_sub).silence()


def ra(profile=None, f0=None, p_sub=None) -> ControlTrack:
    b = _b(profile, f0, p_sub); f = b.f0
    b.lateral("a"); b.vowel("a", f0_end=f * 0.85, final=True); b.silence()
    return b.build()


def ara(profile=None, f0=None, p_sub=None) -> ControlTrack:
    b = _b(profile, f0, p_sub); f = b.f0
    b.vowel("a"); b.tap("a", "a"); b.vowel("a", f0_end=f * 0.85, final=True); b.silence()
    return b.build()


def sa(profile=None, f0=None, p_sub=None, tense=False) -> ControlTrack:
    b = _b(profile, f0, p_sub); f = b.f0
    b.sibilant("a", tense=tense); b.vowel("a", f0_end=f * 0.85, final=True); b.silence()
    return b.build()


def na(profile=None, f0=None, p_sub=None) -> ControlTrack:
    b = _b(profile, f0, p_sub); f = b.f0
    b.nasal("n", "a"); b.vowel("a", f0_end=f * 0.85, final=True); b.silence()
    return b.build()


def ma(profile=None, f0=None, p_sub=None) -> ControlTrack:
    b = _b(profile, f0, p_sub); f = b.f0
    b.nasal("m", "a"); b.vowel("a", f0_end=f * 0.85, final=True); b.silence()
    return b.build()


def ja(profile=None, f0=None, p_sub=None) -> ControlTrack:
    b = _b(profile, f0, p_sub); f = b.f0
    b.affricate("a"); b.vowel("a", f0_end=f * 0.85, final=True); b.silence()
    return b.build()


def cha(profile=None, f0=None, p_sub=None) -> ControlTrack:
    b = _b(profile, f0, p_sub); f = b.f0
    b.affricate("a", aspirated=True); b.vowel("a", f0_end=f * 0.85, final=True); b.silence()
    return b.build()


def irinilssirirae(profile=None, f0=None, p_sub=None) -> ControlTrack:
    """'일인일실이래' [이리닐씨리래] — 여성 녹음 0.69~1.52 s 의 음절 길이·F0 를 따른다."""
    b = _b(profile, f0, p_sub); f = b.f0
    b.f0 = f * 1.25                                                 # 고조로 시작 (계측 400 대)
    b.vowel("i", 0.080, f0_end=f * 1.45)
    b.tap("i", "i")
    b.vowel("i", 0.085, f0_end=f * 1.75)
    b.nasal("n", "i", dur=0.03)
    b.vowel("i", 0.070, f0_end=f * 1.70)
    b.lateral("i", hold=0.055, release=0.02)
    b.sibilant("i", tense=True)
    b.vowel("i", 0.065, f0_end=f * 1.65)
    b.tap("i", "ae")
    b.vowel("ae", 0.220, f0_end=f * 0.85, final=True)
    b.silence()
    return b.build()
