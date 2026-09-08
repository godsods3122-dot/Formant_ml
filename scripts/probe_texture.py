"""치찰음의 '지글거림' 을 잰다 — 스펙트럼이 아니라 **포락의 변조 스펙트럼**으로.

왜 이 자인가
------------
"백색잡음 같지 않고 지글거린다" 는 것은 크기 스펙트럼의 문제가 아니라 시간 구조의
문제다. 매끄러운 난류는 포락이 저역 통과 잡음이고, 지글거리면 포락이 특정 주기로
출렁인다. 그러니 고역 대역의 포락을 뽑아 그 **변조 스펙트럼**을 본다.

두 용의자가 여기서 갈린다
-------------------------
* f0 와 그 배음에 봉우리  -> 성문 위상에 걸린 **사각파 게이트**
                            (noise.py: am = 1 − 0.5·voiced·(frac < 0.35))
* 1 kHz(= 1/프레임) 에 봉우리 -> **프레임률 제어 난동**이 선형 보간을 거쳐 삼각파가 된 것

같이 보는 것
------------
* 변조 깊이 (포락의 변동계수) — 클수록 출렁인다
* 첨도 — 가우시안 잡음은 3, 버스트성이면 그보다 크다
"""
import sys; sys.path.insert(0, "src")
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfiltfilt, hilbert, get_window
from formant_ml.engine.denoise import denoise, noise_profile


def env_modspec(x, fs, lo=4000.0, hi=12000.0):
    hi = min(hi, 0.45 * fs)
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    b = sosfiltfilt(sos, x)
    e = np.abs(hilbert(b))
    e = e - e.mean()
    n = len(e)
    w = get_window("hann", n)
    E = np.abs(np.fft.rfft(e * w)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return f, E, b


def peaks(f, E, lo, hi):
    m = (f >= lo) & (f < hi)
    if not m.any():
        return 0.0
    return float(E[m].sum())


def report(lab, x, fs, f0=None):
    x = x[np.abs(x) > 0] if (np.abs(x) > 0).any() else x
    f, E, b = env_modspec(x, fs)
    tot = peaks(f, E, 5.0, 2000.0) + 1e-30
    env = np.abs(hilbert(b))
    cv = float(env.std() / (env.mean() + 1e-20))
    kur = float(((b / (b.std() + 1e-20)) ** 4).mean())
    row = [f"{lab:>22s}", f"변조깊이 {cv:5.2f}", f"첨도 {kur:5.2f}"]
    # 대역별 변조 에너지 비율
    for lo, hi, nm in ((5, 60, "5~60"), (60, 400, "60~400"), (400, 800, "400~800"),
                       (800, 1200, "800~1200"), (1200, 2000, "1200~2000")):
        row.append(f"{nm}Hz {100*peaks(f,E,lo,hi)/tot:5.1f}%")
    print("  ".join(row), flush=True)
    if f0:
        for k in (1, 2, 3):
            band = peaks(f, E, f0 * k - 25, f0 * k + 25)
            print(f"{'':>22s}   f0×{k} ({f0*k:.0f} Hz) 근방 {100*band/tot:5.1f}%")


print("고역 4~12 kHz 포락의 변조 스펙트럼. 매끄러운 난류면 저역(5~60 Hz)에 몰린다.\n")

y, sr = sf.read("data/ref/female_yang_ilin-ilsil.wav")
if y.ndim > 1: y = y.mean(1)
y = denoise(y, sr, noise_profile(y, sr))
report("목표 여 ㅆ (씨)", y[int(1.12 * sr):int(1.26 * sr)], sr, f0=260)

for p, lab in (("out/v2_yang/04_ssa_tense.wav", "엔진 ㅆ 렌더"),
               ("out/v2_yang/03_sa_lax.wav", "엔진 ㅅ 렌더"),
               ("out/ab_male/sa_A_rec.wav", "남 /사/ 녹음"),
               ("out/ab_male/sa_B_syn.wav", "남 /사/ 합성")):
    try:
        z, zr = sf.read(p)
    except Exception:
        print(f"{lab:>22s}  (없음)"); continue
    if z.ndim > 1: z = z.mean(1)
    report(lab, z, zr)
