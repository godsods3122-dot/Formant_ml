"""**극이 물리와 맞는가** — 적합된 궤적을 이론과 맞대어 본다.

 (1) 관 길이: F1~F4 가 함의하는 L 이 서로 일치하는가 (섭동 이론의 (2k−1)c/4L)
 (2) 극 간격: F4 위 사다리가 c/2L 인가, 그리고 그 L 이 (1) 과 같은가
 (3) 대역폭: Fant 의 손실 법칙 범위 안인가 (모달 발성 F1~F3 은 대략 40~130 Hz)
 (4) 극 개수: 16 kHz 아래 극 수 = 2L·16000/c 여야 한다
 (5) 대역폭/간격 비: 이웃 극이 녹아 버리지 않는가 (물결이 없어지는 조건)
"""
import sys, numpy as np
sys.path.insert(0, "src")
from formant_ml.engine import tract as T
C = getattr(T, "C_SOUND", 34000.0)          # cm/s

for stem in sys.argv[1:]:
    z = np.load(stem + "_track.npz", allow_pickle=True)
    nm = [str(x) for x in z["names"]]; V = np.asarray(z["values"], float)
    g = lambda n: V[:, nm.index(n)] if n in nm else None
    F = [g(f"f{i}") for i in (1, 2, 3, 4)]
    B = [g(f"bw{i}") for i in (1, 2, 3, 4)]
    ok = np.ones(len(V), bool)
    for q in F:
        ok &= np.isfinite(q) & (q > 0)
    print(f"\n== {stem}  프레임 {ok.sum()}")
    print("   (1) 각 포먼트가 함의하는 관 길이 L = (2k−1)·c / (4·F_k)   [cm]")
    Ls = []
    for k, q in enumerate(F, 1):
        L = (2*k - 1) * C / (4.0 * q[ok])
        Ls.append(L)
        print(f"       F{k}  중앙 {np.median(q[ok]):7.0f} Hz -> L {np.median(L):5.2f} cm "
              f"(25~75 % {np.percentile(L,25):4.2f}~{np.percentile(L,75):4.2f})")
    Ls = np.array(Ls)
    spread = np.median(Ls.max(0) - Ls.min(0))
    print(f"       **프레임마다 네 L 의 벌어짐 중앙 {spread:.2f} cm** "
          f"(균일관이면 0, 실제 조음은 0.5~2 cm 가 정상)")
    print("   (3) 대역폭 [Hz] — 모달 발성 실측은 F1 40~90, F2 60~120, F3 80~160")
    for k, q in enumerate(B, 1):
        if q is None: continue
        print(f"       bw{k}  5 % {np.percentile(q[ok],5):6.0f}  중앙 {np.median(q[ok]):6.0f}  "
              f"95 % {np.percentile(q[ok],95):6.0f}")
    print("   (5) 이웃 극 간격 대비 대역폭 — 1 을 넘으면 두 극이 녹아 물결이 사라진다")
    for k in range(3):
        gap = F[k+1][ok] - F[k][ok]
        rel = (B[k][ok] + B[k+1][ok]) / (2.0 * np.maximum(gap, 1.0))
        print(f"       F{k+1}-F{k+2}  간격 중앙 {np.median(gap):5.0f} Hz   "
              f"(bw 평균)/간격 중앙 {np.median(rel):5.2f}  95 % {np.percentile(rel,95):5.2f}")
    # (2)(4) 사다리
    L0 = np.median(Ls[1])                      # F2 가 주는 L 을 대표로
    sp = C / (2.0 * L0)
    print(f"   (2) 사다리 간격 c/2L = {sp:.0f} Hz  (L = {L0:.2f} cm, F2 기준)")
    print(f"   (4) 16 kHz 아래 있어야 할 극 수 = 2L·16000/c = {2*L0*16000/C:.1f} 개   "
          f"엔진의 F1~F8 + 여분 = {getattr(T,'N_FORMANTS',8)} + n_extra")
