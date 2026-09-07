"""기준 녹음 계측 — docs/MEASUREMENTS.md 의 표를 만든 스크립트.

    python scripts/measure_ref.py f   # 여성 화자 (data/ref/female_yang_ilin-ilsil.wav)
    python scripts/measure_ref.py m   # 남성 화자 비음/치찰음 (reference/recordings/)
"""
import numpy as np, soundfile as sf, parselmouth
def load(p):
    y, sr = sf.read(p, always_2d=True); return y.mean(1), sr
def tracks(y, sr, t0, t1, step=0.005, maxf=5500, nf=5, label=""):
    snd = parselmouth.Sound(y, sr).extract_part(from_time=t0, to_time=t1, preserve_times=True)
    fm = snd.to_formant_burg(time_step=step, max_number_of_formants=nf, maximum_formant=maxf, window_length=0.02)
    it = snd.to_intensity(minimum_pitch=100, time_step=step); pt = snd.to_pitch(time_step=step, pitch_floor=70, pitch_ceiling=500)
    print(f"--- {label} {t0}-{t1}s: t(ms) dB F0 F1 F2 F3 F4")
    for t in np.arange(t0+0.01, t1-0.01, step):
        F=[fm.get_value_at_time(k,t) for k in (1,2,3,4)]; f0=pt.get_value_at_time(t)
        print(f"  {t*1000:6.0f} {it.get_value(t):5.1f} {(f0 if f0==f0 else 0):4.0f} " + " ".join(f"{(v if v==v else 0):5.0f}" for v in F))
def spectrum(y, sr, t0, t1, label, bands=((0,1000),(1000,2000),(2000,4000),(4000,6000),(6000,8000),(8000,10000),(10000,12000),(12000,16000),(16000,22000))):
    s = y[int(t0*sr):int(t1*sr)]; n=len(s); Y=np.abs(np.fft.rfft(s*np.hanning(n)))**2; f=np.fft.rfftfreq(n,1/sr)
    # smooth 1/12 oct
    P = 10*np.log10(Y+1e-20); tot=Y.sum()
    cent=(f*Y).sum()/tot; pk = f[np.argmax(np.convolve(Y, np.ones(int(n/sr*300)+1)/(int(n/sr*300)+1), 'same'))]
    print(f"--- {label} {t0:.3f}-{t1:.3f}s ({(t1-t0)*1000:.0f} ms) rms {20*np.log10(np.sqrt((s**2).mean())+1e-12):.1f} dB centroid {cent:.0f} Hz peak(300Hz smooth) {pk:.0f} Hz")
    print("    " + " ".join(f"{lo//1000}-{hi//1000}k:{10*np.log10(Y[(f>=lo)&(f<hi)].sum()/tot+1e-20):5.1f}" for lo,hi in bands if hi<=sr/2+1))
def envelope(y, sr, t0, t1, label, hp=3000, win=0.005):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, hp/(sr/2), 'high', output='sos'); e = sosfiltfilt(sos, y[int(t0*sr):int(t1*sr)])
    n=int(win*sr); env=[20*np.log10(np.sqrt((e[i:i+n]**2).mean())+1e-9) for i in range(0,len(e)-n,n)]
    print(f"--- {label} HF({hp}+) env 5ms from {t0}s:", " ".join(f"{v:.0f}" for v in env))
import sys
which = sys.argv[1]
if which == "f":
    y, sr = load("data/ref/female_yang_ilin-ilsil.wav")
    spectrum(y, sr, 2.80, 3.40, "female long /a/")
    tracks(y, sr, 2.80, 3.40, step=0.05, label="female long /a/ (50ms)", maxf=5500)
    tracks(y, sr, 0.44, 1.55, step=0.005, label="female 아 이리닐씨리래", maxf=5500)
    for a,b in ((1.10,1.22),(5.66,5.78),(6.79,6.90)):
        spectrum(y, sr, a, b, "female ㅆ region")
        envelope(y, sr, a-0.02, b+0.03, "female ㅆ", hp=4000)
    spectrum(y, sr, 1.22, 1.27, "female vowel after ㅆ (리)")
    spectrum(y, sr, 0.50, 0.64, "female 아 (initial)")
else:
    y, sr = load("reference/recordings/ko_nasal-sibilant_na-ma-sa-ja-cha_male.wav")
    for a,b,l in ((3.90,4.02,"ㄴ coda 난"),(5.92,6.02,"ㄴ coda 난#2"),(10.82,10.92,"ㅁ coda 맘"),(12.60,12.70,"ㅁ coda 맘#2"),(0.28,0.34,"ㄴ onset 나"),(4.80,4.86,"ㄴ onset"),(7.16,7.22,"ㅁ onset 마"),(11.52,11.58,"ㅁ onset")):
        spectrum(y, sr, a, b, "male "+l, bands=((0,250),(250,500),(500,1000),(1000,2000),(2000,3000),(3000,5000),(5000,8000)))
    spectrum(y, sr, 0.5, 1.0, "male vowel 나 /a/", bands=((0,250),(250,500),(500,1000),(1000,2000),(2000,3000),(3000,5000),(5000,8000)))
    tracks(y, sr, 3.80, 4.10, step=0.01, label="male 난 coda", maxf=5000)
    tracks(y, sr, 0.25, 0.50, step=0.01, label="male 나 onset", maxf=5000)
    tracks(y, sr, 7.14, 7.40, step=0.01, label="male 마 onset", maxf=5000)
    for a,b,l in ((14.24,14.40,"ㅅ #1"),(15.56,15.72,"ㅅ #2"),(16.88,17.02,"ㅅ #3"),(19.40,19.52,"ㅈ #1"),(20.68,20.80,"ㅈ #2"),(22.08,22.16,"ㅈ #3"),(24.48,24.60,"ㅊ #1"),(25.76,25.90,"ㅊ #2"),(27.16,27.30,"ㅊ #3")):
        spectrum(y, sr, a, b, "male "+l)
        envelope(y, sr, a-0.03, b+0.06, "male "+l, hp=3000)
    spectrum(y, sr, 14.5, 14.9, "male vowel after ㅅ")
    tracks(y, sr, 14.36, 14.60, step=0.01, label="male 사 onset", maxf=5000)
