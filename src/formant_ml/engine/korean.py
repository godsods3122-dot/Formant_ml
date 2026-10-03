"""한국어 전사 → 음소열 (IPA) 과 음소별 조음 목표 (W02 MRI 자세) — 음소 목표 조음의 입력 (MEASUREMENTS §52.487).

규칙 (서울말, 발화 안은 쉼 없이 이어진다고 본다):
* 음절을 초성·중성·종성으로 가른다. 초성 ㅇ 은 소리가 없다. 종성은 7 대표음 (ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅇ).
* 연음: 종성 뒤 초성이 ㅇ 이면 종성이 다음 음절 초성으로 넘어간다.
* 평음 ㄱ ㄷ ㅂ ㅈ 은 유성음(모음·ㄴ ㅁ ㅇ ㄹ) 사이에서 유성 [ɡ d b dʑ].
* 경음화: 장애음 종성(ㄱ ㄷ ㅂ) 뒤의 평음은 된소리. 비음화: 장애음 종성 + 비음 초성 → 같은 자리 비음.
* ㅅ·ㅆ 은 /i/·/j/ 앞에서 [ɕ].

정렬용 기호는 음소 인식기(wav2vec2 XLSR-53 espeak) 어휘에 있는 것만 쓴다 — 된소리는 길이 기호(kː tː pː), ㅊ 은 tɕh.
"""
from __future__ import annotations

from dataclasses import dataclass

ONSETS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
NUCLEI = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
CODAS = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

VOWEL = {"ㅏ": ["a"], "ㅐ": ["e"], "ㅑ": ["j", "a"], "ㅒ": ["j", "e"], "ㅓ": ["ʌ"], "ㅔ": ["e"], "ㅕ": ["j", "ʌ"], "ㅖ": ["j", "e"],
         "ㅗ": ["o"], "ㅘ": ["w", "a"], "ㅙ": ["w", "e"], "ㅚ": ["w", "e"], "ㅛ": ["j", "o"], "ㅜ": ["u"], "ㅝ": ["w", "ʌ"],
         "ㅞ": ["w", "e"], "ㅟ": ["w", "i"], "ㅠ": ["j", "u"], "ㅡ": ["ɯ"], "ㅢ": ["ɯ", "i"], "ㅣ": ["i"]}
# 초성: (무성 기본형, 유성 이음, 부류)
ONSET = {"ㄱ": ("k", "ɡ", "velar"), "ㄲ": ("kː", "kː", "velar"), "ㅋ": ("kʰ", "kʰ", "velar"),
         "ㄷ": ("t", "d", "alveolar"), "ㄸ": ("tː", "tː", "alveolar"), "ㅌ": ("tʰ", "tʰ", "alveolar"),
         "ㅂ": ("p", "b", "labial"), "ㅃ": ("pː", "pː", "labial"), "ㅍ": ("pʰ", "pʰ", "labial"),
         "ㅈ": ("tɕ", "dʑ", "affricate"), "ㅉ": ("tɕ", "tɕ", "affricate"), "ㅊ": ("tɕh", "tɕh", "affricate"),
         "ㅅ": ("s", "s", "fricative"), "ㅆ": ("s", "s", "fricative"), "ㅎ": ("h", "h", "glottal"),
         "ㄴ": ("n", "n", "alveolar-nasal"), "ㅁ": ("m", "m", "labial-nasal"), "ㄹ": ("ɾ", "ɾ", "liquid")}
LENIS = {"ㄱ", "ㄷ", "ㅂ", "ㅈ"}
TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅈ": "ㅉ", "ㅅ": "ㅆ"}
# 종성 대표음
CODA7 = {"ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㄳ": "ㄱ", "ㅋ": "ㄱ", "ㄺ": "ㄱ", "ㄴ": "ㄴ", "ㄵ": "ㄴ", "ㄶ": "ㄴ", "ㄷ": "ㄷ", "ㅅ": "ㄷ",
         "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ", "ㄹ": "ㄹ", "ㄼ": "ㄹ", "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㅀ": "ㄹ",
         "ㅁ": "ㅁ", "ㄻ": "ㅁ", "ㅂ": "ㅂ", "ㅄ": "ㅂ", "ㅍ": "ㅂ", "ㄿ": "ㅂ", "ㅇ": "ㅇ"}
CODA_IPA = {"ㄱ": ("k", "velar"), "ㄴ": ("n", "alveolar-nasal"), "ㄷ": ("t", "alveolar"), "ㄹ": ("l", "liquid"),
            "ㅁ": ("m", "labial-nasal"), "ㅂ": ("p", "labial"), "ㅇ": ("ŋ", "velar-nasal")}
NASAL_OF = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}
SONORANT_CODA = {"ㄴ", "ㄹ", "ㅁ", "ㅇ"}

# 모음 → W02 모음 자세 (여성 한국어 모음 포먼트에 가장 가까운 W02 독일어 모음, 적합의 출발점일 뿐이다)
VOWEL_SHAPE = {"a": "a", "ʌ": "O", "o": "o", "u": "U", "ɯ": "@", "i": "i", "e": "e", "j": "i", "w": "u"}
PLACE_SHAPE = {"velar": "tb-velar-closure", "velar-nasal": "tb-velar-closure", "alveolar": "tt-alveolar-closure",
               "alveolar-nasal": "tt-alveolar-closure", "labial": "ll-labial-closure", "labial-nasal": "ll-labial-closure",
               "liquid": "tt-alveolar-lateral", "affricate": "tt-alveolopalatal-closure", "fricative": "tt-alveolar-fricative",
               # 'i' 앞 ㅅ·ㅆ 은 치경구개 [ɕ] — 혀날이 치조 뒤를 좁히고 입술은 편다 (§52.519). W02 의 후치조 마찰은 독일어 [ʃ] 라 입술을 내밀어
               # (입술–앞니 1.12 cm) 앞 공동이 길고, 정적 시험의 봉우리가 2 kHz 였다 (원본 ㅅ 10 kHz). 치조 마찰 자세(입술 폄)를 자리 창 칸 21–26 으로.
               "fricative-pal": "tt-alveolopalatal-fricative",
               # 구개 비음 ɲ (MFA 의 구개음화된 ㄴ · 038 '청자' 의 받침) — 혀 몸통·혀날이 경구개에 닿는다 (§52.519). 치조로 두었더니 입 안 곁가지가 길어
               # 반공명이 2–2.75 kHz 로 내려가고 3–4.25 kHz 가 원본보다 +20–34 dB 넘쳤다 (원본 골 3–4.7 kHz ≈ 곁가지 2.5 cm). W02 화자 파일에 구개 폐쇄
               # 자세가 없어 구개 마찰 자세에서 만든다 (`profiles/w02_shapes.npz`, 혀 옆면은 연구개 폐쇄 값, 혀 몸통·혀날을 닿을 때까지 올림).
               "palatal-nasal": "tb-palatal-closure"}


@dataclass
class Phone:
    ipa: str          # 정렬용 기호
    kind: str         # vowel / glide / velar / alveolar / labial / … / glottal
    syl: int          # 음절 번호
    shape: str = ""   # W02 목표 자세 이름 (glottal 은 다음 모음 자세)
    nasal: bool = False


def _jamo(ch: str):
    k = ord(ch) - 0xAC00
    if not 0 <= k < 11172:
        return None
    return ONSETS[k // 588], NUCLEI[(k % 588) // 28], CODAS[k % 28]


def g2p(text: str) -> list[Phone]:
    syl = [_jamo(c) for c in text if _jamo(c) is not None]
    out: list[Phone] = []
    for n, (on, nu, co) in enumerate(syl):
        prev_coda = CODA7.get(syl[n - 1][2], "") if n else ""
        if n and syl[n - 1][2] and on == "ㅇ":             # 연음 — 앞 종성이 이 음절 초성으로
            on = syl[n - 1][2] if syl[n - 1][2] in ONSET else CODA7.get(syl[n - 1][2], "ㅇ")
            prev_coda = ""
        if on != "ㅇ" and on in ONSET:
            o = on
            if prev_coda in ("ㄱ", "ㄷ", "ㅂ") and o in TENSE:
                o = TENSE[o]
            voiced_ctx = n > 0 and (prev_coda == "" or prev_coda in SONORANT_CODA)
            base, voiced, kind = ONSET[o]
            ipa = voiced if (o in LENIS and voiced_ctx) else base
            if o in ("ㅅ", "ㅆ") and VOWEL[nu][0] in ("i", "j"):
                ipa, kind = "ɕ", "fricative-pal"
            out.append(Phone(ipa, kind, n))
        for v in VOWEL[nu]:
            out.append(Phone(v, "glide" if v in ("j", "w") else "vowel", n))
        if co and not (n + 1 < len(syl) and syl[n + 1][0] == "ㅇ" and co in ONSET):
            c7 = CODA7[co]
            if n + 1 < len(syl) and syl[n + 1][0] in ("ㄴ", "ㅁ") and c7 in NASAL_OF:
                c7 = NASAL_OF[c7]
            ipa, kind = CODA_IPA[c7]
            out.append(Phone(ipa, kind, n))
    # 목표 자세 — 자음은 뒤(없으면 앞) 모음 맥락 a / i / u
    for k, p in enumerate(out):
        if p.kind in ("vowel", "glide"):
            p.shape = VOWEL_SHAPE[p.ipa]
            continue
        nxt = next((q.ipa for q in out[k + 1:] if q.kind in ("vowel", "glide")), None)
        prv = next((q.ipa for q in reversed(out[:k]) if q.kind in ("vowel", "glide")), None)
        v = nxt or prv or "a"
        ctx = "i" if v in ("i", "e", "j") else ("u" if v in ("o", "u", "w") else "a")
        if p.kind == "glottal":
            p.shape = VOWEL_SHAPE.get(nxt or "a", "a")
        else:
            p.shape = f"{PLACE_SHAPE[p.kind]}({ctx})"
        p.nasal = p.kind.endswith("nasal")
    return out


# MFA korean_mfa 음소 → (부류, 기호) — 정렬 결과를 W02 목표 자세로 옮길 때 (§52.487)
_MFA_V = {"a": "a", "ɐ": "a", "aː": "a", "ʌ": "ʌ", "ʌː": "ʌ", "o": "o", "oː": "o", "u": "u", "uː": "u", "ɨ": "ɯ", "ɯ": "ɯ", "ɨː": "ɯ",
          "i": "i", "iː": "i", "e": "e", "eː": "e", "ɛ": "e", "ɛː": "e", "ø": "e", "y": "i", "j": "j", "w": "w", "ɰ": "ɯ", "ɥ": "w"}
_MFA_C = {"k": "velar", "kʰ": "velar", "k͈": "velar", "ɡ": "velar", "ŋ": "velar-nasal", "t": "alveolar", "tʰ": "alveolar", "t͈": "alveolar",
          "d": "alveolar", "n": "alveolar-nasal", "ɲ": "palatal-nasal", "p": "labial", "pʰ": "labial", "p͈": "labial", "b": "labial",
          "m": "labial-nasal", "s": "fricative", "sʰ": "fricative", "s͈": "fricative", "ɕ": "fricative-pal", "ɕʰ": "fricative-pal",
          "ɕ͈": "fricative-pal", "tɕ": "affricate", "tɕʰ": "affricate", "t͈ɕ": "affricate", "dʑ": "affricate", "l": "liquid", "ɾ": "liquid",
          "ʎ": "liquid", "h": "glottal", "ç": "glottal", "x": "glottal", "ɦ": "glottal", "ʔ": "glottal", "k̚": "velar", "t̚": "alveolar",
          "p̚": "labial"}


_MFA_TRUE_V = {"a", "ɐ", "aː", "ʌ", "ʌː", "o", "oː", "u", "uː", "ɨ", "ɯ", "ɨː", "i", "iː", "e", "eː", "ɛ", "ɛː", "ø", "y"}
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def _coda_of(d: dict, ph: list) -> dict:
    """MFA 자음 차례 → 그 자음이 받침이면 그 음절의 받침 자모 (낱말 층의 한글에서, §52.519). 낱말 안에서 자음 앞의 홀소리 수로 음절을 세고,
    뒤가 홀소리·활음이 아니면(자음·끝) 받침으로 본다. 활음(j · w · ɰ · ɥ)은 홀소리로 세지 않는다."""
    words = d["tiers"].get("words", {}).get("entries", [])
    out = {}
    for k, (a, b, p) in enumerate(ph):
        if p in _MFA_V:
            continue
        nxt = ph[k + 1][2] if k + 1 < len(ph) else ""
        if nxt in _MFA_TRUE_V or nxt in ("j", "w", "ɰ", "ɥ"):
            continue
        wd = next((w for w in words if w[0] - 1e-6 <= a and b <= w[1] + 1e-6), None)
        if wd is None:
            continue
        syl = [c for c in wd[2] if 0xAC00 <= ord(c) < 0xAC00 + 11172]
        nv = sum(1 for (a2, b2, p2) in ph if wd[0] - 1e-6 <= a2 and b2 <= a + 1e-6 and p2 in _MFA_TRUE_V)
        if 1 <= nv <= len(syl):
            out[k] = _JONG[(ord(syl[nv - 1]) - 0xAC00) % 28]
    return out


def mfa_segments(path: str) -> list[dict]:
    """MFA json (phones 층) → [{label, t0, t1, shape}] — 앞뒤 쉼은 `sil`, 자음 자세는 뒤(없으면 앞) 모음 맥락."""
    import json
    d = json.load(open(path, encoding="utf-8"))
    ph = [(e[0], e[1], e[2]) for e in d["tiers"]["phones"]["entries"] if e[2] not in ("", "sil", "sp", "spn")]
    out, T = [], float(d["end"])
    vow = [_MFA_V.get(p) for _, _, p in ph]
    coda = _coda_of(d, ph)
    for k, (a, b, p) in enumerate(ph):
        if p in _MFA_V:
            shape = VOWEL_SHAPE[_MFA_V[p]]
        else:
            kind = _MFA_C.get(p, "alveolar")
            if p == "ɲ" and coda.get(k) == "ㅇ":
                # MFA 는 파찰음 앞 받침 비음을 ɲ 로 적는다 (코퍼스 98 번) — 받침이 ㅇ 이면 연구개 [ŋ] (§52.519). 038 '청자' 의 원본 웅얼거림은 반공명이
                # 3188 · 4078 Hz — 연구개 포트에서 연구개 폐쇄까지 입 안 곁가지 2.1–2.7 cm 의 자리다 (구개 폐쇄면 1.6–2.4 kHz, 치조면 1.4 kHz).
                kind = "velar-nasal"
            nxt = next((v for v in vow[k + 1:] if v), None)
            prv = next((v for v in reversed(vow[:k]) if v), None)
            v = nxt or prv or "a"
            ctx = "i" if v in ("i", "e", "j") else ("u" if v in ("o", "u", "w") else "a")
            shape = VOWEL_SHAPE.get(nxt or "a", "a") if kind == "glottal" else f"{PLACE_SHAPE[kind]}({ctx})"
        out.append({"label": p, "t0": a, "t1": b, "shape": shape})
    segs = []
    if out and out[0]["t0"] > 0:
        segs.append({"label": "sil", "t0": 0.0, "t1": out[0]["t0"], "shape": None})
    for k, s in enumerate(out):                    # 사이 쉼(짧은 멈춤)은 앞 음소에 붙인다
        if k and s["t0"] > segs[-1]["t1"] + 1e-6:
            segs[-1]["t1"] = s["t0"]
        segs.append(s)
    if segs and segs[-1]["t1"] < T:
        segs.append({"label": "sil", "t0": segs[-1]["t1"], "t1": T, "shape": None})
    return segs



#: 곡선 조음(`voice_td.ARTIC = "cos"`)의 협착 자리 [성문 0 → 입술 1] — MFA 정렬로 초기값을 놓을 때 (`phoneme_init`, §52.491).
COS_PLACE = {"alveolar": 0.88, "postalveolar": 0.82, "palatal": 0.78, "velar": 0.68, "uvular": 0.62}


def phoneme_init(values, index: dict, segments: list, frame_s: float, voiced=None):
    """MFA 정렬(구간 기준 초, `mfa_segments` 형식) → 곡선 조음 제어열 초기값을 제자리에 고친다 (§52.491). 돌려주는 값은 마찰 틀 표지.

    폐쇄음: 폐쇄 몫(앞 60 %)에서 제자리 `a_c` 0 (양순은 `oral_open` 0.02), 파찰음(tɕ 계열)은 앞 40 % 폐쇄 · 뒤 마찰, 마찰음은 가운데 20–80 % 에서
    `a_c` 0.08 cm², 비음은 폐쇄 + `velum` 0.6. 모음·쉼은 건드리지 않는다. 적합은 이 위에서 다듬는다 — 곡선 조음에는 음운 정보가 없어 분석
    초기값이 놓친 자음 협착(038 B 의 ㅈ, `a_c` 2 cm²)을 적합기가 스스로 찾지 못했다."""
    import numpy as np
    V = values
    T = V.shape[0]
    fric = np.zeros(T, bool)
    ia, ic, io_, iv = index["a_c"], index["c_place"], index["oral_open"], index["velum"]
    for sg in segments:
        shp, lab = sg.get("shape"), str(sg.get("label", "")).replace("ː", "")
        if not shp or "(" not in shp:
            continue
        t0, t1 = float(sg["t0"]), float(sg["t1"])
        d = t1 - t0
        fr = lambda a, b: slice(max(0, int(round(a / frame_s))), min(T, int(round(b / frame_s))))
        place = next((k for k in ("postalveolar", "alveolar", "palatal", "velar", "uvular", "labial") if k in shp), None)
        nasal = lab in ("m", "n", "ŋ", "ɲ")
        affr = lab.startswith(("tɕ", "t͈ɕ", "dʑ"))
        if "closure" in shp:
            cl = fr(t0, t0 + (0.4 if affr else 0.6) * d)
            if place == "labial":
                V[cl, io_] = 0.02
            elif place is not None:
                V[cl, ia] = 0.0
                V[cl, ic] = COS_PLACE[place]
            if nasal:
                V[fr(t0, t1), iv] = 0.6
            if affr:
                fz = fr(t0 + 0.4 * d, t1)
                V[fz, ia] = 0.08
                V[fz, ic] = COS_PLACE.get(place, 0.82)
                fric[fz] = True
        elif "fricative" in shp:
            fz = fr(t0 + 0.2 * d, t1 - 0.2 * d)
            V[fz, ia] = 0.08
            V[fz, ic] = COS_PLACE.get(place, 0.85)
            fric[fz] = True
    return fric
