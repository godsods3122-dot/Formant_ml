"""복사합성 — 녹음을 분석해 그 파라미터로 물리모델을 구동한다.

    PYTHONPATH=src python3 scripts/copysynth.py IN.wav -o out/recon.wav [--from S --to S]

왜 이걸 먼저 하는가
-------------------
음절을 손으로 작곡하면 "스펙트럼 지표는 맞는데 사람 소리가 아닌" 상태에서
빠져나올 방법이 없다. 무엇이 틀렸는지 비교할 정답이 없기 때문이다.
복사합성은 **정답이 있다** — 원본 파형. 되합성이 원본처럼 안 들리면 엔진이
틀린 것이고, 어느 파라미터가 부족한지 바로 좁혀진다.
PLAN.md 의 Phase 1 이고, 이걸 건너뛰고 Phase 3(작곡)부터 한 것이 실패의 원인이다.

경로는 **포먼트 캐스케이드**를 쓴다. 도파관은 이득 정규화가 아직 안 풀렸고
(방사 임피던스 부재), 캐스케이드는 이 레포가 목표치 ±3 % 로 검증해 두었다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from formant_ml.analysis.acoustic import load as load16
from formant_ml.analysis.acoustic import band_db
from formant_ml.analysis.acoustic import voicing as periodicity
from formant_ml.analysis.track import track_formants
from formant_ml.config import Config
from formant_ml.data.features import yin_f0
from formant_ml.models.synth import Controls, PhysicalVoiceSynth
from formant_ml.utils import save_wav


def frame_rms(y: np.ndarray, hop: int, win: int) -> np.ndarray:
    t = max(0, 1 + (len(y) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(t)[:, None]
    return np.sqrt((y[idx] ** 2).mean(1))


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """중앙값 + 이동평균. 한 프레임짜리 검출 실패를 지운다."""
    if len(x) < k or k < 2:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    med = np.median(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0)
    mp = np.pad(med, pad, mode="edge")
    return np.mean(np.stack([mp[i:i + len(x)] for i in range(k)]), axis=0)


def analyse(y: np.ndarray, cfg: Config):
    """파형 -> 복사합성에 필요한 프레임별 파라미터."""
    sr, hop = cfg.audio.sample_rate, cfg.audio.hop_size
    win = 1024
    F, B = track_formants(y, sr, hop, win, n=cfg.filt.n_formants)
    t = len(F)
    yt = torch.from_numpy(y).float()[None]
    f0, voi = yin_f0(yt, sr, hop)
    f0, voi = f0[0, :t].numpy(), voi[0, :t].numpy()
    rms = frame_rms(y, hop, win)[:t]

    # **이진 유성 판정을 게이트로 쓰면 안 된다.** 유음처럼 진폭이 잠깐 꺼지고
    # 포먼트가 빨리 움직이는 구간에서 검출기가 자신을 잃고 0 을 내는데, 그
    # 순간 (a) 하모닉 진폭이 0 이 되어 소리가 파이고 (b) 무성으로 보고 마찰
    # 노이즈가 쏟아진다. 스펙트로그램에서 4~7 kHz 세로 뭉치로 나타난다.
    # 실제로 "라" 는 전 구간 유성이다 — 검출기가 틀린 것이지 소리가 무성인
    # 것이 아니다. 그래서 **연속 주기성**을 쓰고 시간축으로 평활한다.
    per, _ = periodicity(y, sr, hop_ms=hop / sr * 1000.0,
                         win_ms=win / sr * 1000.0)
    per = _smooth(per[:t], 7)
    per = np.clip((per - 0.25) / 0.45, 0.0, 1.0)

    # 비주기성만으로 노이즈를 켜면 안 된다. 모음 꼬리처럼 **조용한 유성 구간**
    # 에서는 SNR 이 떨어져 주기성 측정이 무너지는데, 그걸 무성으로 보고 노이즈를
    # 부으면 발화 끝에 쉬익 소리가 붙는다(스펙트로그램에서 확인).
    # 실제 마찰/기식은 **고역 에너지를 동반한다.** 그것을 조건으로 건다.
    hb = band_db(y, sr, 2500.0, 8000.0, hop / sr * 1000.0, win / sr * 1000.0)
    lb = band_db(y, sr, 100.0, 2500.0, hop / sr * 1000.0, win / sr * 1000.0)
    hf = _smooth(np.clip((hb - lb + 25.0) / 20.0, 0.0, 1.0)[:t], 7)

    # F0 는 신뢰 구간만 쓰고 나머지는 이어 붙인다 (0 이 들어가면 위상이 끊긴다)
    med = float(np.median(f0[voi > 0.5])) if (voi > 0.5).any() else 120.0
    fv = np.where(voi > 0.5, f0, np.nan)
    ok = ~np.isnan(fv)
    f0 = (np.interp(np.arange(t), np.arange(t)[ok], fv[ok]) if ok.any()
          else np.full(t, med))
    f0 = _smooth(np.clip(f0, cfg.source.f0_min, cfg.source.f0_max), 3)
    return dict(freq=F, bw=B, f0=f0, periodicity=per, hf=hf, rms=rms, t=t)


#: 이 레벨(발화 최대 대비 dB) 아래에서는 성문 소스를 끈다. build_controls 주석 참고.
SILENCE_GATE_DB = -30.0

#: 유성 구간의 기식 난류 바닥(진폭 대비). build_controls 주석 참고.
ASPIRATION_FLOOR = 0.10

#: 성문 소스의 기본 기울기와 LF 형상. 고른 근거는 main() 의 --tilt 주석에 있다.
#: 다른 스크립트(make_listening_set 등)가 여기를 읽으므로 기본값이 안 어긋난다.
DEFAULT_TILT, DEFAULT_RD = 1.0, 0.6


def build_controls(a: dict, cfg: Config, jitter: float, shimmer: float,
                   tilt: float, rd: float):
    t, K = a["t"], cfg.filt.n_formants
    nb = cfg.noise.n_bands

    def T(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    freq = T(a["freq"]).clamp(cfg.filt.f_min, cfg.filt.f_max)[None]
    # 대역폭 상한은 **슬롯마다 다르다.**
    #  - 실제로 잡힌 극: `bw_max`(800). 이걸 풀면 안 된다 — `_tame_bandwidths`
    #    의 상한(Fant x2.5)이 10 kHz 에서 5375 Hz 라, 풀어 두면 고역 극이
    #    프레임마다 있다 없다 하고 짧은 토큰에서 대역오차가 커진다.
    #  - 추적기가 못 찾은 슬롯: `bw_neutral` 그대로 둔다. bw_max 로 자르면
    #    '극 없음' 이 10.8 kHz 짜리 Q=13.5 공명으로 되살아난다
    #    (config.bw_neutral 과 track._fill 주석).
    bwv = np.asarray(a["bw"], dtype=np.float32)
    bw = T(np.where(bwv >= cfg.filt.bw_neutral * 0.5, cfg.filt.bw_neutral,
                    np.clip(bwv, cfg.filt.bw_min, cfg.filt.bw_max)))[None]
    rms = a["rms"] / max(a["rms"].max(), 1e-9)
    per = a["periodicity"]
    hf = a["hf"]
    # 난류 소스의 스펙트럼. **두 항은 물리가 달라서 모양도 다르다.**
    #
    # (1) 구강 마찰/파열의 난류: 협착부 제트에서 나고, 평탄한 백색으로 두면
    #     캐스케이드의 고역 극들을 그대로 통과해 +13.4 dB/oct 로 치솟는다.
    #     유성 구간에 섞인 2 % 만으로도 "쨍한" 소리가 됐다. 저역 셸프를 건다.
    #
    # (2) **유성 기식**: 발성 중에도 성대가 완전히 닫히지 않아 새는 난류다.
    #     이게 빠져 있었다(바닥 0.002). 그래서 합성의 고역이 원본의 **안개**가
    #     아니라 성긴 **하모닉 빗살**이었다 — 사람 소리가 아니라 합성음으로
    #     들리는 이유다. 대역 파워는 맞는데 질감이 달라서, 파워만 보는 지표로는
    #     안 잡혔다. 맞는 지표는 **빗살 깊이**(대역 안 파워평균 - 로그평균):
    #
    #     | (dB) | 1~2.5k | 2.5~4k | 4~7k | 7~9.5k |
    #     |---|---|---|---|---|
    #     | 원본 | 12.5 | 12.3 | 6.5 | 7.1 |
    #     | 기식 0.002 | 22.1 | 16.1 | 10.1 | 8.1 |
    #     | **기식 0.10** | 19.0 | 12.6 | 7.3 | 6.4 |
    #
    #     성문에서 나는 난류이므로 **평탄**하다(협착 제트가 아니다). 그리고
    #     `noise_entry=0` 으로 성도 전체를 지난다.
    #     세기 0.10 은 대역 레벨 오차와 빗살 오차의 합을 최소화하는 값이고,
    #     0.05~0.15 / tilt 0.5~1.5 에서 평평하다(운 좋은 한 점이 아니다).
    fb = np.linspace(0.0, cfg.audio.sample_rate / 2, nb)
    oral = (1.0 / (1.0 + (fb / 700.0) ** 2)) ** 0.9
    oral = oral / oral.max()

    # **발성이 아예 없는 프레임에서는 성문 소스를 끈다.**
    # 진폭은 원칙적으로 측정된 세기를 그대로 따른다(유성도로 곱해서 끄면 검출
    # 실패가 그대로 진폭 구멍이 된다 — 유음에서 겪었다). 다만 발화 사이의
    # **무음**은 다르다: 거기엔 발성이 없고 남은 것은 녹음실 잡음인데, 무음
    # 프레임의 포먼트 추적 결과는 신호가 아니라 잡음 바닥에 맞춘 것이라
    # 아무 뜻이 없고 극이 8 kHz 부근에 몰린다. 그 필터로 성문 소스를 통과시키니
    # 발화 사이에 8 kHz 삐 소리가 났다.
    #
    # **유성도가 아니라 절대 레벨**로 끈다. 유음의 세기 골은 뒤 모음보다
    # 5~8 dB 낮을 뿐이라(§3.2) -30 dB 문턱과는 한참 떨어져 있어 안전하다.
    # 문턱은 -24 ~ -36 dB 에서 결과가 같다.
    lvl = 20.0 * np.log10(rms + 1e-9)
    u = np.clip((lvl - SILENCE_GATE_DB) / 10.0, 0.0, 1.0)
    gate = (u * u * (3.0 - 2.0 * u)).astype(np.float32)      # smoothstep

    return Controls(
        f0=T(a["f0"]).reshape(1, t, 1),
        harmonic_amp=T(rms * gate).reshape(1, t, 1),
        rd=torch.full((1, t, 1), rd),
        formant_freq=freq, formant_bw=bw,
        formant_gain=torch.ones(1, t, K),
        # 무성 구간의 마찰 성분. 유성 구간에는 소량의 기식만 남긴다.
        # 노이즈는 **연속 비주기성**에 비례한다. 이진 판정에 묶으면 검출이
        # 흔들릴 때마다 마찰음 버스트가 터진다.
        noise_bands=(T(rms * 0.12 * (1.0 - per) ** 2 * hf).reshape(1, t, 1)
                     * T(oral).reshape(1, 1, nb)
                     # 기식은 발성의 부산물이다 -> 같은 게이트를 건다
                     + T(rms * gate * ASPIRATION_FLOOR).reshape(1, t, 1)
                     ).contiguous(),
        # 기식은 **성문**에서 난다 -> 성도 전체를 지난다. 여기를 K*0.7 로 두면
        # 노이즈가 상위 포먼트(5~11 kHz)만 통과해 +33 dB/oct 로 치솟고,
        # 유성 구간에 섞인 2 % 만으로도 고역을 덮어 "쨍한" 소리가 된다.
        noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.full((1, t, 1), 0.5),
        tilt=torch.full((1, t, 1), tilt),
        jitter=torch.full((1, t, 1), jitter),
        shimmer=torch.full((1, t, 1), shimmer),
    )


def match_envelope(y: np.ndarray, target_rms: np.ndarray, hop: int,
                   floor: float = 1e-5) -> np.ndarray:
    """합성 파형의 프레임 세기를 원본의 세기 곡선에 맞춘다.

    성도 캐스케이드의 전체 이득은 포먼트 배치에 따라 달라지므로, 제어에 넣은
    진폭이 그대로 출력 세기가 되지 않는다. 무음 구간에서 특히 나빴다 —
    원본이 -39 dB 인데 합성이 -25 dB 로 나와 배경이 쉬익 소리로 들렸다.
    복사합성에서 세기 곡선은 **분석된 파라미터**이므로 맞추는 것이 옳다.
    """
    t = len(target_rms)
    n = t * hop
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    fr = y[:n].reshape(t, hop)
    cur = np.sqrt((fr ** 2).mean(1))
    tgt = target_rms / max(target_rms.max(), 1e-12)
    cur = cur / max(cur.max(), 1e-12)
    g = np.where(cur > floor, tgt / np.maximum(cur, floor), 0.0)
    g = np.clip(g, 0.0, 3.0)
    # 프레임 경계 클릭을 피해 샘플 단위로 선형보간
    gs = np.interp(np.arange(n), np.arange(t) * hop + hop / 2, g)
    return y[:n] * gs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("-o", "--out", default="out/recon.wav")
    ap.add_argument("--from", dest="t0", type=float, default=None)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    # 실측 정상 음성 범위(지터 0.2~0.5 %, 시머 2~4 %). 이전 0.006/0.04 는
    # 그보다 높았다. 고역 하모닉을 뭉갠다고 의심했으나 지표로는 기각됐다
    # (원본의 고역도 하모닉이 안 잡힌다) — 생리적 값으로만 낮춘 것이다.
    ap.add_argument("--jitter", type=float, default=0.002)
    ap.add_argument("--shimmer", type=float, default=0.02)
    # **두 번 다시 골랐다. 값을 바꿀 때는 아래 조건을 그대로 쓰라.**
    #
    # 1 차(tilt=-12/Rd=0.6): `ltv_filter` 의 직사각 블록이 만든 가짜 에너지를
    #    통해서 잰 값이라 무효였다(§2.3). 부호 규약도 반대로 알고 있었다 —
    #    §4.3 의 "값이 클수록 어두워진다" 는 `filters.one_pole_tilt` 이야기고,
    #    여기 `GlottalSource` 의 tilt 는 **값이 클수록 밝아진다**.
    # 2 차(tilt=+4/Rd=0.9): `track._fill` 의 가짜 극 2 개가 7~11 kHz 를
    #    +25.6 dB 올려 둔 상태에서 잰 값이라 역시 무효였다.
    #
    # 3 차(현재): 가짜 극을 고친 뒤, **무음을 포함한 전체 발화 5.8 초**로 골랐다.
    # 짧은 토큰만 보면 안 된다 — 무음 구간의 결함이 안 보이고, 추적기가 전역
    # 씨 프레임을 쓰기 때문에 발췌와 전체의 결과가 서로 다르다.
    # tilt 와 Rd 는 사실상 같은 손잡이다(둘 다 고역 기울기). 제대로 된 해법은
    # 상수가 아니라 프레임별 Rd 를 H1-H2 로 추정하는 것이다.
    #
    # 4 차(현재): `track._space_out` 으로 고역 극 몰림을 막은 뒤 다시 골랐다.
    # 그 전에는 7~9.5 kHz 의 혹이 어느 tilt 에서도 안 없어져서, tilt 가 그 혹을
    # 상대적으로 눌러 보려고 자기를 어둡게 맞추고 있었다.
    #
    # **판정은 '라우드니스를 맞춘 뒤 절대 대역 레벨' 하나로 한다.**
    # 총에너지로 정규화한 대역비를 쓰면 한 대역이 뜰 때 나머지가 전부 낮아
    # 보이고(2.5~4 kHz '구멍' 이 그 착시였다), 프레임 평균을 뺀 포락선 모양을
    # 쓰면 한 대역이 통째로 뜨는 것을 못 본다. 셋을 섞어 쓰다 제자리를 돌았다.
    #
    # 결과(전체 발화, 원본 대비 dB):
    #   0.1~1k +0.2 / 1~2.5k -1.6 / 2.5~4k -7.3 / 4~7k -3.1 / 7~9.5k +5.1 /
    #   9.5~12k +1.0   (최대 7.3, RMS 3.9. 고치기 전에는 23.9 / 12.1)
    # 남은 최대 오차는 2.5~4 kHz 의 -7.3 dB 다.
    ap.add_argument("--tilt", type=float, default=DEFAULT_TILT)
    ap.add_argument("--rd", type=float, default=DEFAULT_RD)
    args = ap.parse_args()

    cfg = Config()
    y, sr = load16(args.wav, cfg.audio.sample_rate)
    if args.t0 is not None:
        y = y[int(args.t0 * sr): int((args.t1 or len(y) / sr) * sr)]
    y = y / max(np.abs(y).max(), 1e-9)

    a = analyse(y, cfg)
    syn = PhysicalVoiceSynth(cfg, tract_mode="formant")
    c = build_controls(a, cfg, args.jitter, args.shimmer, args.tilt, args.rd)
    with torch.no_grad():
        out = syn(c)["audio"]
    out = match_envelope(out[0].numpy().astype(np.float64), a["rms"],
                         cfg.audio.hop_size)
    save_wav(args.out, torch.from_numpy(out).float(), sr)
    print(f"{args.out}  ({a['t'] * cfg.audio.hop_size / sr:.2f}s, "
          f"{a['t']} 프레임, 주기성 평균 {float(a['periodicity'].mean()):.2f})")


if __name__ == "__main__":
    main()
