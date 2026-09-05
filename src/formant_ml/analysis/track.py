"""프레임별 포먼트 추적 — 복사합성용.

`extract.lpc_formants` 는 화자 **평균** 포먼트를 낸다(프로파일 추출용).
복사합성은 프레임마다 다른 값이 필요하고, 무엇보다 **연속성**이 필요하다.
매 프레임 독립으로 근을 뽑아 정렬만 하면 포먼트가 서로 자리를 바꿔 가며
튀고, 그 궤적으로 합성하면 사람 소리가 아니라 기계음이 된다.

여기서는 (1) LPC 근을 뽑고 (2) 직전 프레임의 궤적에 **가까운 것부터 배정**하고
(3) 빈 곳을 보간한 뒤 (4) 가볍게 중앙값 평활한다.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .extract import fant_bandwidth


def _lpc(x: np.ndarray, order: int):
    r = np.correlate(x, x, "full")[len(x) - 1:][: order + 1]
    if r[0] <= 0:
        return None
    r = r / r[0]
    r[0] += 1e-6
    a = np.zeros(order + 1)
    a[0], e = 1.0, r[0]
    for i in range(1, order + 1):
        acc = r[i] + (np.dot(a[1:i], r[i - 1:0:-1]) if i > 1 else 0.0)
        k = -acc / e
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        e *= (1.0 - k * k)
        if e <= 0:
            return None
    return a


def _roots(frame: np.ndarray, sr: int, order: int, fmin: float, fmax: float,
           max_bw: float):
    a = _lpc(frame, order)
    if a is None:
        return np.empty(0), np.empty(0)
    z = np.roots(a)
    z = z[np.imag(z) > 0]
    if not len(z):
        return np.empty(0), np.empty(0)
    f = np.angle(z) * sr / (2 * np.pi)
    bw = -(sr / np.pi) * np.log(np.abs(z) + 1e-12)
    k = (f > fmin) & (f < fmax) & (bw < max_bw) & (bw > 0)
    f, bw = f[k], bw[k]
    o = np.argsort(f)
    return f[o], bw[o]


def track_formants(y: np.ndarray, sr: int = 24000, hop: int = 240,
                   win: int = 1024, n: int = 6, order: int | None = None,
                   fmin: float = 120.0, fmax: float = 9000.0,
                   max_bw: float = 900.0, preemph: float = 0.97,
                   jump_hz: float = 350.0, bw_neutral: float = 80000.0,
                   min_gap_hz: float = 800.0, gap_above_hz: float = 4000.0):
    """(T, n) 포먼트 [Hz] 와 (T, n) 대역폭 [Hz]. 연속 궤적으로 이어 붙인다.

    **가장 신뢰할 만한 프레임에서 씨를 뿌리고 양방향으로 추적한다.**
    첫 프레임부터 앞으로만 가면 무음 구간에서 잘못 잡힌 궤적이 끝까지 이어진다
    — 실제로 슬롯 5 가 무음의 8146 Hz 를 물고 있어서, 모음 구간의 실재 극
    5118 Hz 가 들어갈 자리를 잃고 통째로 버려졌다(고역이 20 dB 넘게 죽었다).

    한 프레임 안의 배정은 (1) 직전 궤적에 `jump_hz` 안으로 가장 가까운 근을
    순서를 지켜 이어 붙이고 (2) 남은 근은 **앵커 사이의 빈 자리에만** 넣는다.
    아무렇게나 채우면 한 궤적이 빠졌을 때 나머지가 한 칸씩 밀려 들어온다
    (F1 이 한 프레임 빠지자 F2 가 슬롯 0 으로 와서 출력이 11 dB 튀었다).
    """
    # 차수: 고전 경험식은 fs[kHz] + 2~4 (24 kHz -> 26) 이고, 그건 극쌍 13 개다.
    # 그런데 이 레포는 나이퀴스트까지 **포먼트 12 개**를 모델링한다(config 주석:
    # 1 kHz 당 1 개). 12 개를 담으려면 극쌍 12 개가 온전히 포먼트에 가야 하는데,
    # 13 쌍으로는 소스 기울기와 영점이 먹을 여유가 없어서 상위 슬롯이 서로
    # 겹치거나 빈다(측정: fmax=9000/order=26 에서 7~11 kHz 에 극 6 개가 몰리고
    # 그중 둘은 순서가 뒤집혔다). 극쌍 16 개(order 32)면 12 개를 채우고도 남는다.
    # 고전 경험식 fs[kHz] + 2 (24 kHz -> 26, 극쌍 13 개).
    #
    # **이 값을 올려 보고 싶어질 것이다. 재 봤고, 안 된다.** 이 레포는
    # 나이퀴스트까지 포먼트 12 개를 모델링하는데(config 주석: 1 kHz 당 1 개),
    # 차수를 32~36 으로 올리면 LPC 가 저역에서 극을 더 쪼개 찾고 그것들이
    # 12 슬롯을 다 먹는다. 그러면 최상단 극이 8.4 kHz 에 그쳐 그 위가 통째로
    # 비고, 7~11 kHz 가 -92 dB 로 무너진다(측정: '라' 토큰, order 28 -> 36 에서
    # -56 -> -98 dB). fmax 를 11 kHz 로 올려도 같은 문제가 남는다.
    #
    # 진짜 문제는 차수가 아니라 **구조**다. 8 kHz 위에는 LPC 가 안정적으로
    # 찾을 극이 없다(그 대역은 공명이 아니라 기식 노이즈다). 극으로 맞추려는
    # 것 자체가 틀렸고, 답은 Fant 의 고차극 보정이다 —
    # docs/HANDOFF_LIQUID.md §2.7.
    order = order or int(2 + sr / 1000)
    # **`fmax` 를 11 kHz 로 올려 봤고, 안 된다.** 모델링 대역과 맞추는 게
    # 맞아 보이지만(9 kHz 위에 극이 하나도 안 잡히니), 라우드니스를 맞춘 뒤
    # **절대** 대역 레벨로 재면 7~9.5 kHz 가 +8.6 -> +18.6 dB 로 오히려 더
    # 뜬다. 사용자가 탄음에서 들은 "지글거리는 고주파" 가 그 대역이다.
    # 포락선 **모양** 지표로는 11 kHz 가 나아 보이는데 그게 함정이었다 —
    # 모양 지표는 프레임 평균을 빼기 때문에 한 대역이 통째로 뜨는 것을 못 본다.
    # **판정은 절대 레벨로 해라** (`scripts/diag_hifreq.py` 의 주석 참고).
    #
    # **차수도 올리지 마라.** 26 -> 36 이면 LPC 가 저역에서 극을 더 쪼개 찾고
    # 그것들이 12 슬롯을 다 먹어, 최상단 극이 8.4 kHz 에 그치고 그 위가
    # -92 dB 로 무너진다. 슬롯이 12 개라는 것이 진짜 제약이다.
    x = lfilter([1.0, -preemph], [1.0], y)
    t = max(0, 1 + (len(x) - hop * 0 - win) // hop)
    if t == 0:
        return np.zeros((0, n)), np.zeros((0, n))
    w = np.hanning(win)
    cand = [_roots(x[i * hop: i * hop + win] * w, sr, order, fmin, fmax, max_bw)
            for i in range(t)]
    energy = np.array([np.sqrt((y[i * hop: i * hop + win] ** 2).mean())
                       for i in range(t)])

    F = np.full((t, n), np.nan)
    B = np.full((t, n), np.nan)
    # 씨 프레임은 **근이 가장 많이 잡힌 곳**으로 고른다(동수면 에너지가 큰 쪽).
    # 에너지만 보고 고르면 그 프레임이 놓친 극의 슬롯이 통째로 밀려서, 이후
    # 모든 프레임이 그 밀림을 물려받는다(3318 Hz 가 사라지고 고역이 어두워졌다).
    nroots = np.array([len(c[0]) for c in cand])
    score = nroots * 1000.0 + energy / max(energy.max(), 1e-12)
    seed = int(np.argmax(score))
    f0, b0 = cand[seed]
    m = min(n, len(f0))
    F[seed, :m], B[seed, :m] = f0[:m], b0[:m]

    for direction in (1, -1):
        prev = F[seed].copy()
        i = seed + direction
        while 0 <= i < t:
            f, bw = cand[i]
            if len(f):
                _assign(F, B, i, f, bw, prev, n, jump_hz)
                row = F[i]
                prev = np.where(np.isnan(row), prev, row)
            i += direction
    observed = ~np.isnan(F)
    F, B, found = _fill(F, B, sr)
    F = _space_out(F, min_gap_hz, gap_above_hz, sr)
    B = _tame_bandwidths(F, B)
    F, B, found = _higher_poles(F, B, found, observed, sr, min_gap_hz)
    # 못 찾은 슬롯은 **극이 없는 것**으로 둔다 — 아래 `_fill` 주석의 정책 (3).
    # `_tame_bandwidths` 뒤에 덮어써야 한다(Fant 기준으로 도로 좁히므로).
    B = np.where(found, B, bw_neutral)
    return F, B


def _higher_poles(F, B, found, observed, sr, spacing):
    """관측된 극 **위쪽** 슬롯을 Fant 의 고차극 보정으로 채운다.

    LPC 는 8~9 kHz 위에서 극을 하나도 못 찾는다(측정: 9 kHz 위 근 0.00 개
    /프레임). 그 대역은 공명이 아니라 기식 난류이기 때문이다. 그렇다고 그
    슬롯을 비우면(대역폭 중립) 9.5~12 kHz 가 -16 dB 로 무너진다 — 실제
    성도에는 그 위로도 극이 계속 서 있고, 모델이 그걸 통째로 빠뜨린 것이다.

    Fant 의 고차극 보정이 정확히 이 자리다: 모델링 대역 위의 극들은 개별로
    분해되지 않고 **완만한 셸프**로 합쳐진다. 길이 L 의 관은 극이 c/2L
    (여기서는 `spacing`) 간격으로 서므로, 그 프레임에서 **관측된 가장 높은
    극**부터 같은 간격으로 열을 연장하고 대역폭은 Fant 경험식을 쓴다.
    10 kHz 에서 2150 Hz 이므로 Q~4.7 — 휘파람이 아니라 셸프다.

    **이 방법은 예전에 한 번 기각됐다**(`_fill` 주석의 정책 (3)). 그때 실패한
    이유는 연장 자체가 아니라 그 아래가 틀렸기 때문이다 — 탐욕적 `_assign`
    이 7~8.5 kHz 에 극 4 개를 몰아 넣어 두었고(순서까지 뒤집힌 채) 그 위에
    연장을 얹으니 6 개가 됐다. `_assign` 을 순서 보존 DP 로 바꿔 몰림이
    사라진 뒤에는 연장이 제 몫을 한다.
    """
    F, B = F.copy(), B.copy()
    found = found & np.ones_like(F, bool)
    t, n = F.shape
    top = np.where(observed.any(1), observed.shape[1] - 1
                   - np.argmax(observed[:, ::-1], axis=1), -1)
    hi = sr * 0.48
    for i in range(t):
        s0 = top[i]
        if s0 < 0:
            continue
        f = F[i, s0]
        for s in range(s0 + 1, n):
            f = f + spacing
            if f >= hi:
                found[i, s] = False
                continue
            F[i, s] = f
            B[i, s] = 50.0 + 20.0 * (f / 1000.0) ** 2 + 10.0 * (f / 1000.0)
            found[i, s] = True
    return F, B, found


def _space_out(F: np.ndarray, min_gap: float, above: float, sr: int) -> np.ndarray:
    """고역에서 극이 서로 겹쳐 쌓이는 것을 막는다 (물리적 최소 간격).

    길이 L 의 관은 포먼트가 c/2L 간격으로 선다 — 17.5 cm, c=350 m/s 면 1 kHz.
    그런데 추적기는 전체 발화에서 **7.3~8.5 kHz 안에 극 4 개**를 넣어 놓았고,
    그중 하나는 순서까지 뒤집혀 있었다(8482 다음이 8373). 성도가 그럴 수는
    없다. 그 무더기가 7~9.5 kHz 에 혹을 만들고(사용자 지적: 탄음의 "지글거리는
    고주파"), 위쪽 슬롯을 다 써 버려 9.5 kHz 위는 절벽이 된다.

    **저역에는 걸지 않는다.** F1/F2 가 647 Hz 밖에 안 떨어진 프레임이 실제로
    있고(/u/ 는 357/914 로 557 Hz), 거기에 균일 간격을 걸면 실재하는 F2 를
    밀어 버린다. 몰림은 고역에서만 일어난다.

    측정 (전체 발화, 라우드니스 맞춘 절대 대역 오차):

    | min_gap | 7~9.5 kHz | 9.5~12 kHz | RMS |
    |---|---|---|---|
    | 없음 | +6.95 | -23.91 | 12.1 |
    | 600 | +1.11 | -5.85 | 7.3 |
    | **800** | **-0.88** | **-1.28** | **7.0** |
    | 1000 | -0.78 | +10.08 | 8.6 |
    | 1200 | +2.57 | +28.70 | 14.8 |

    800 Hz 는 공칭 간격(1 kHz)의 0.8 배다 — '균일관보다 조금 촘촘한 것까지는
    허용하되, 1.1 kHz 안에 4 개는 안 된다'.
    """
    F = F.copy()
    hi = sr * 0.48
    for s in range(1, F.shape[1]):
        g = np.where(F[:, s - 1] >= above, min_gap, 0.0)
        F[:, s] = np.maximum(F[:, s], F[:, s - 1] + g)
    return np.minimum(F, hi)


def _tame_bandwidths(F: np.ndarray, B: np.ndarray, lo: float = 0.5,
                     hi: float = 2.5, max_ratio: float = 1.6) -> np.ndarray:
    """LPC 근의 대역폭을 물리적 범위로 묶고 **변화율을 제한**한다.

    LPC 근의 대역폭은 못 믿는다. 실측에서 F3 의 BW 가 연속 프레임에서
    849 -> 33 -> 779 -> 71 Hz 로 뛰었다. 33 Hz 는 Q=79 짜리 면도날 공명이라
    스펙트로그램에 밝은 줄로 찍히고(사용자가 "2500 쯤 강한 진폭" 이라고 지적),
    849 Hz 프레임에서는 원본보다 +10.8 dB 튄다. 매 프레임 공명의 예리함이
    무작위로 뒤집히는 것이 고역이 지저분한 원인이다.

    Fant 경험식을 기준으로 [lo, hi] 배 안에 묶고, 프레임 사이 변화를
    `max_ratio` 배 이내로 제한한다. 주파수는 건드리지 않는다 — 그쪽은 믿을
    만하고(포먼트 오차가 작다), 대역폭만 문제다.
    """
    ref = 50.0 + 20.0 * (F / 1000.0) ** 2 + 10.0 * (F / 1000.0)
    out = np.clip(B, ref * lo, ref * hi)
    for i in range(1, len(out)):
        out[i] = np.clip(out[i], out[i - 1] / max_ratio, out[i - 1] * max_ratio)
    for i in range(len(out) - 2, -1, -1):
        out[i] = np.clip(out[i], out[i + 1] / max_ratio, out[i + 1] * max_ratio)
    return out


def _assign(F, B, i, f, bw, prev, n, jump_hz,
            drop_root: float = 2.0, drop_slot: float = 0.5,
            free_slot: float = 0.5, over_jump: float = 4.0):
    """한 프레임의 근을 궤적에 배정한다 (순서 보존, 전역 최소비용).

    **탐욕적 배정은 근을 버린다.** 예전 방식은 슬롯을 낮은 쪽부터 훑으며
    직전 궤적에서 `jump_hz` 안에 있는 첫 근을 집고, 남은 근은 이미 배정된
    이웃 사이에 빈 슬롯이 있을 때만 끼워 넣었다. 한 번 집은 포인터는 뒤로
    못 가므로, 앞 슬롯이 근 하나를 가져가 버리면 뒤 슬롯은 자기 근을 영영
    못 본다. 측정('라' 전체 발화, 활성 프레임):

    | 대역 | LPC 근 | 탐욕 배정 뒤 |
    |---|---|---|
    | 0.1~1 kHz | 0.93 | 0.91 |
    | 1~2.5 kHz | 1.97 | 1.82 |
    | **2.5~4 kHz** | **1.06** | **0.65** |
    | 4~7 kHz | 2.79 | 2.35 |

    2.5~4 kHz 에서만 39 % 가 버려진다. 그 대역이 F3/F4 자리이고, 복사합성이
    거기서 -7 dB 부족했던 것의 원인이다(docs/HANDOFF_LIQUID.md 의 미해결
    항목). 버려진 극은 `_fill` 이 보간으로 흉내 낼 뿐 되살아나지 않는다.

    그래서 **순서를 지키는 최소비용 정렬**로 바꾼다. 슬롯 열과 근 열은 둘 다
    주파수 오름차순이므로, 둘 사이의 순서 보존 부분정합은 편집거리와 같은
    꼴의 DP 로 정확히 푼다 (12 x 9 셀, 프레임당 비용 무시 가능).

    비용은 전부 `jump_hz` 배수로 준다:
      - 잇기: |f - prev| (직전 궤적이 있을 때). `jump_hz` 를 넘으면 초과분에
        `over_jump` 배 벌점 — **막지는 않는다.** 막으면 빠르게 움직이는
        전이(유음이 바로 그것이다)에서 궤적이 통째로 끊긴다.
      - 직전 궤적이 없는 슬롯에 넣기: `free_slot` x jump_hz (고정)
      - 근을 버리기: `drop_root` x jump_hz  <- 가장 비싸다
      - 슬롯을 비우기: `drop_slot` x jump_hz
    """
    m = len(f)
    if m == 0:
        return
    big = float("inf")
    cr = drop_root * jump_hz
    cs = drop_slot * jump_hz
    # cost[s][j] = 슬롯 s 에 근 j 를 넣는 비용
    cost = np.empty((n, m))
    for s in range(n):
        p = prev[s]
        if np.isnan(p):
            cost[s] = free_slot * jump_hz
        else:
            d = np.abs(f - p)
            cost[s] = np.where(d <= jump_hz, d,
                               jump_hz + (d - jump_hz) * over_jump)
    dp = np.full((n + 1, m + 1), big)
    back = np.zeros((n + 1, m + 1), np.int8)   # 0 = 정합, 1 = 슬롯 비움, 2 = 근 버림
    dp[0, 0] = 0.0
    for s in range(n + 1):
        for j in range(m + 1):
            v = dp[s, j]
            if v == big:
                continue
            if s < n and v + cs < dp[s + 1, j]:
                dp[s + 1, j], back[s + 1, j] = v + cs, 1
            if j < m and v + cr < dp[s, j + 1]:
                dp[s, j + 1], back[s, j + 1] = v + cr, 2
            if s < n and j < m and v + cost[s, j] < dp[s + 1, j + 1]:
                dp[s + 1, j + 1], back[s + 1, j + 1] = v + cost[s, j], 0
    s, j = n, m
    while s > 0 or j > 0:
        b = back[s, j]
        if b == 0:
            F[i, s - 1], B[i, s - 1] = f[j - 1], bw[j - 1]
            s, j = s - 1, j - 1
        elif b == 1:
            s -= 1
        else:
            j -= 1


def _fill(F: np.ndarray, B: np.ndarray, sr: int):
    """결측을 메우고 가볍게 평활한다. 반환 (F, B, found).

    **빈 슬롯을 0 으로 두면 안 된다.** 합성기가 f_min 으로 클램프해서 저역에
    가짜 공명기를 만들고, 그 하나하나가 -12 dB/oct 씩 감쇠를 더한다. 실제로
    빈 슬롯 4 개가 150 Hz 짜리 유령 극 4 개가 되어 고역을 40 dB 죽였다.

    빈 슬롯을 무엇으로 채울지 **세 가지를 다 재 봤다.** 다음 사람이 같은 순서로
    헤매지 않도록 결과를 적어 둔다(전체 발화 5.8 초, 무음 포함, 원본 대비):

    | 정책 | 0.1~2.5 kHz | 7~11 kHz | 무음 총에너지 |
    |---|---|---|---|
    | (1) 0.45*fs 에 '넓은' 극 | **-15.1** | **-0.1** | **-14.8** |
    | (2) 중립 (= 극 없음) | -0.3 | -16.2 | -25.1 |
    | (3) 포먼트 열의 연장 | -16.6 | -0.1 | -15.0 |
    | (원본) | -0.3 | -35.2 | -30.8 |

    (1) 은 **무해하지 않다.** `_tame_bandwidths` 와 `filt.bw_max`(800) 를 지나면서
    Q=13.5 짜리 진짜 공명이 된다. 그런 극 2 개가 7~11 kHz 를 +25.6 dB 올리고
    캐스케이드의 최대점을 1195 Hz 에서 8227 Hz 로 옮겨 놨다 — 무음 구간에서
    8 kHz 휘파람이 들렸고 발화 전체에서 저역이 15 dB 묻혔다.

    (3) 은 물리적으로는 맞는 발상이다(길이 L 의 관은 포먼트가 c/2L 간격으로
    고르게 선다). 짧은 토큰에서는 실제로 크게 좋아졌다(7~11 kHz 오차 41.7 ->
    8.3 dB). **그런데 전체 발화에서는 (1) 만큼 나쁘다.** 이유는 슬롯이 비어서가
    아니라, 추적기가 이미 7~8.5 kHz 에 극 4 개를 몰아 넣어 두었기 때문이다
    (게다가 그중 둘은 순서가 뒤집혀 있다: 8482 다음이 8373). 그 위에 연장을
    얹으면 6 개가 된다. **연장이 틀린 게 아니라, 그 아래가 이미 틀렸다.**

    그래서 지금은 (2) 를 쓴다. 사용자가 실제로 듣는 조건(무음 포함 전체 발화)
    에서 유일하게 저역을 되살리는 정책이다. 대신 짧은 발췌에서는 7~11 kHz 가
    빈다. 둘 다 해결하려면 추적기의 상위 슬롯 배정을 먼저 고쳐야 한다 —
    docs/HANDOFF_LIQUID.md §2.7.

    극이 '없는' 것은 **대역폭이 아주 넓은 것**이다(r = exp(-pi*BW/fs) -> 0).
    여기서는 `found` 마스크만 돌려주고, 실제 중립화는 호출부가
    `_tame_bandwidths` **뒤에** 한다(안 그러면 다듬기가 도로 좁힌다).
    """
    F, B = F.copy(), B.copy()
    t, n = F.shape
    idx = np.arange(t)
    found = ~np.isnan(F).all(axis=0)          # 한 프레임이라도 잡힌 슬롯
    dead = sr * 0.45
    for s in range(n):
        for A, spare in ((F, dead), (B, 1200.0)):
            col = A[:, s]
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                col[:] = spare          # 무해한 극: 아주 높고 아주 넓다
            elif ok.sum() < t:
                col[:] = np.interp(idx, idx[ok], col[ok])
            # 3점 중앙값은 **결측을 보간한 자리에만** 건다. 모든 프레임에 걸면
            # 실제 전이까지 뭉개져서 스펙트럼 변화율이 원본의 절반이 된다
            # (사용자 지적: "전이가 너무 부드럽다").
            if t >= 3 and (~ok).any():
                med3 = np.median(np.stack([col[:-2], col[1:-1], col[2:]]),
                                 axis=0)
                fix = (~ok)[1:-1]
                A[1:-1, s] = np.where(fix, med3, col[1:-1])
    return F, B, found
