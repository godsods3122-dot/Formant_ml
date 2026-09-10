"""포락 오차를 시간과 주파수로 쪼갠다 — 프레임별 멜 대역 편향.

    python scripts/probe_band_bias.py out/fix/s040 out/pole/s040

왜 필요한가
    포락 일치율은 멜 dB 거리의 **정규화 비**라, 저에너지 대역의 큰 오차에 거의
    무게를 안 준다. 장기 평균 PSD(`bandcmp.py`)는 **에너지 가중**이라 조용한
    프레임의 결손이 묻힌다. 프레임마다 대역별 dB 오차를 그냥 평균하면 둘 다
    못 보던 것이 보인다 — 실제로 8~12 kHz 의 −6~−8 dB 계통 결손이 여기서만
    드러났다 (§38.2b).

    **편향**(부호 있는 평균)이 핵심이다. |오차| 만 보면 "맞추기 어렵다" 로
    읽히지만, 편향이 −6 dB 면 그것은 **계통적으로 어둡다**는 뜻이다.
"""
import sys; sys.path.insert(0,"src")
import numpy as np, torch, soundfile as sf
from formant_ml.engine.fit import mel_bank
FS=48000.0; N=256
def mel_db(x, mel):
    w=torch.hann_window(N)
    S=torch.stft(torch.as_tensor(x,dtype=torch.float32).unsqueeze(0),n_fft=N,hop_length=N//4,
                 win_length=N,window=w,center=True,return_complex=True,pad_mode="reflect").abs()[0]
    M=mel[:,:S.shape[0]]@S
    return 20*torch.log10(M+1e-6)
mel=mel_bank(N,FS,48,fmax=0.9*FS/2)
import argparse, os
_ap=argparse.ArgumentParser(); _ap.add_argument("stems",nargs="+")
for _stem in _ap.parse_args().stems:
    run,stem=os.path.basename(os.path.dirname(_stem)),os.path.basename(_stem)
    t,fs=sf.read(f"{_stem}_target.wav"); y,_=sf.read(f"{_stem}_fit.wav")
    dt=mel_db(np.asarray(t,np.float32),mel); dy=mel_db(np.asarray(y,np.float32),mel)
    T=min(dt.shape[1],dy.shape[1]); dt,dy=dt[:,:T],dy[:,:T]
    lvl=dt.mean(0); live=lvl>float(lvl.max())-45.0
    e=(dy-dt)[:,live]
    print(f"=== {run}/{stem}  유효 프레임 {int(live.sum())}/{T},  |오차| 평균 {float(e.abs().mean()):.2f} dB")
    # 멜 밴드별
    fb=(mel*torch.arange(mel.shape[1])*FS/N).sum(1)/mel.sum(1).clamp_min(1e-9)
    q=[0,8,16,24,32,40,48]
    print("  밴드 중심 [Hz]  " + " ".join(f"{float(fb[a:b].mean()):7.0f}" for a,b in zip(q,q[1:])))
    print("  |오차| [dB]     " + " ".join(f"{float(e[a:b].abs().mean()):7.2f}" for a,b in zip(q,q[1:])))
    print("  편향 [dB]       " + " ".join(f"{float(e[a:b].mean()):+7.2f}" for a,b in zip(q,q[1:])))
    # 프레임별 상위 오차
    pf=e.abs().mean(0)
    idx=torch.argsort(pf,descending=True)[:8]
    tt=np.where(live.numpy())[0]
    print("  최악 프레임 [s]: " + " ".join(f"{tt[int(i)]*(N//4)/FS:.3f}({float(pf[i]):.1f})" for i in idx))
    # 상위 10 % 프레임이 전체 오차의 몇 %
    s=torch.sort(pf,descending=True).values
    k=max(1,len(s)//10)
    print(f"  상위 10 % 프레임이 전체 |오차| 의 {float(s[:k].sum()/s.sum())*100:.1f} %")
