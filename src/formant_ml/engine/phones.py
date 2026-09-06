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
                 a_c=3.0, c_place=0.9, velum=0.0, tract_gain=1.0, obstacle=0.0, back_leak=0.3)

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
        self.add(t0, p_sub=self.ps, adduction=0.6, a_c=0.6, c_place=0.85,
                 f1=l1, f2=l2, f3=l3, bw1=60, bw2=140, bw3=240,
                 lat_z1=z1, lat_z2=z2, lat_bw=L["zero_bw"], tract_gain=L["gain"])
        if L.get("transient_amp", 0.0) > 0 and onset:
            self.event(t0 + 0.004, "tongue_contact", amp=L["transient_amp"])
        t1 = t0 + hold
        self.add(t1, f1=l1, f2=l2, f3=l3, lat_z1=z1, lat_z2=z2, tract_gain=L["gain"],
                 a_c=0.6, bw1=60, bw2=140, bw3=240)
        self.add(t1 - 0.012, f3=l3, lat_bw=L["zero_bw"])              # F3 먼저 출발
        self.add(t1 + release * 0.6, f3=v3, lat_bw=4000.0)
        self.add(t1 + release * 0.6 + 0.001, lat_z1=0.0, lat_z2=0.0)
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
        """/ㅅ, ㅆ/: 성문을 벌리고 혀가 협착을 만든다. 마찰 세기는 레이놀즈가 정한다.

        포락선(계측): 상승 ~30 ms, 고원, 하강 ~15 ms. 정점 주파수는 앞공동 길이(프로파일).
        경음은 짧고 크다(성문 개대 작음 → adduction 을 덜 내린다).
        """
        S = self.p.sibilant
        v1, v2, v3 = self.V(next_vowel)
        dur = self.ms(S["tense_dur_ms" if tense else "dur_ms"]) if dur is None else dur
        lvl = S.get("tense_level_db", S["level_db"]) if tense else S.get("lax_level_db", S["level_db"])
        gain = 10 ** ((lvl - (-11.0)) / 20.0)             # fric_gain=1 ⇒ 모음 대비 −11 dB (보정 기준)
        l1, l2, l3 = 350 * 1.35, 0.5 * (1750 * 1.15) + 0.5 * v2, v3     # 치경 로커스
        t0 = self.t
        rise, fall = 0.040, 0.018
        adduct = 0.10 if tense else 0.06        # 발성 게이트(0.10) 아래: 무성
        # 혀는 침묵 동안 이미 다가와 있다(a_c 0.35). 레이놀즈 구동이 (Re²−Re_c²)^1.5 라 마지막
        # 접근에서 급히 서므로, 넓은 데서 출발하면 개시가 10 ms 짜리 계단이 된다(측정).
        self.add(t0, p_sub=self.ps, adduction=adduct, a_c=0.35, c_place=0.91, back_leak=S["back_leak"],
                 front_len=self.p.sib_front_len_cm, obstacle=S["obstacle"], fric_gain=gain,
                 f1=l1, f2=l2, f3=l3, tract_gain=0.9)
        self.add(t0 + rise, a_c=S["a_min"])
        self.add(t0 + dur - fall, a_c=S["a_min"], adduction=adduct)
        self.add(t0 + dur, a_c=3.0, adduction=0.5, obstacle=0.0, front_len=0.0, back_leak=0.3)
        self.add(t0 + dur + 0.045, adduction=0.6, f1=v1, f2=v2, f3=v3, tract_gain=1.0, fric_gain=1.0)
        self.t = t0 + dur
        return self

    def affricate(self, next_vowel="a", aspirated=False, tense=False):
        """/ㅈ, ㅊ, ㅉ/: 폐쇄(구강압 축적) → 방전 버스트(10 ms 안에 선다) → 감쇠 마찰 → 모음.

        정점 주파수는 치경구개라 /s/ 보다 앞공동이 길다(계측 ~5 kHz, 남성) → 길이 ×1.5.
        ㅊ 은 해제 뒤 성문이 늦게 닫혀 기식이 길다.
        """
        S = self.p.sibilant
        v1, v2, v3 = self.V(next_vowel)
        closure = 0.060
        fric = 0.110 if aspirated else 0.080
        t0 = self.t
        front = self.p.sib_front_len_cm * 1.6
        adduct = 0.10 if tense else (0.03 if aspirated else 0.07)
        self.add(t0, p_sub=self.ps, adduction=adduct, a_c=0.0, c_place=0.88, back_leak=S["back_leak"],
                 front_len=front, obstacle=0.5, fric_gain=4.0 if aspirated else 3.2,   # 계측: ㅈ 모음 −4..−9, ㅊ −5 dB
                 f1=350 * 1.35, f2=v2 * 1.1, f3=v3, tract_gain=0.85)
        t1 = t0 + closure
        self.add(t1, a_c=0.0, adduction=adduct)
        self.add(t1 + 0.004, a_c=0.10)                                   # 급한 해제
        self.add(t1 + fric * 0.7, adduction=adduct)                      # 마찰 동안 성문 벌림 유지
        self.add(t1 + fric, a_c=0.35 if aspirated else 0.6)
        self.add(t1 + fric + 0.02, a_c=3.0, obstacle=0.0, front_len=0.0, back_leak=0.3,
                 adduction=0.6 if not aspirated else 0.2, fric_gain=1.0)
        if aspirated:
            self.add(t1 + fric + 0.08, adduction=0.6)                    # 기식 뒤에 성문이 닫힌다
        self.add(t1 + fric + 0.05, f1=v1, f2=v2, f3=v3, tract_gain=1.0)
        self.t = t1 + fric
        return self

    def nasal(self, place="n", next_vowel="a", dur=None, coda=False, prev_vowel="a"):
        """비음: 연구개 개방 + 구강 폐쇄. 머머 레벨/전이는 프로파일(계측: −6 dB, 70 ms)."""
        N = self.p.nasal
        v1, v2, v3 = self.V(next_vowel)
        dur = self.ms(N["dur_ms"]) if dur is None else dur
        tr = self.ms(N["transition_ms"])
        nz = N["zero_hz"][place]
        f2 = {"m": 1100.0, "n": 1400.0, "ng": 1900.0}[place]
        t0 = self.t
        if coda:                                                          # 모음 → 비음
            a1, a2, a3 = self.V(prev_vowel)
            self.add(t0 - tr, f1=a1, f2=a2, f3=a3, velum=0.0, tract_gain=1.0)
            self.add(t0, velum=1.0, f1=300.0, f2=f2, f3=2600.0, tract_gain=N["gain"], a_c=0.02,
                     nasal_f=N["pole_hz"], nasal_z=nz)
            self.add(t0 + dur, velum=1.0, f1=300.0, tract_gain=N["gain"], a_c=0.02)
            self.t = t0 + dur
            return self
        self.add(t0, p_sub=self.ps, adduction=0.6, velum=1.0, nasal_f=N["pole_hz"], nasal_z=nz,
                 f1=300.0, f2=f2, f3=2600.0, tract_gain=N["gain"], a_c=0.02)
        self.add(t0 + dur, velum=1.0, f1=300.0, tract_gain=N["gain"], a_c=0.02)
        self.add(t0 + dur + tr, velum=0.0, f1=v1, f2=v2, f3=v3, tract_gain=1.0, a_c=3.0)
        self.t = t0 + dur
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
