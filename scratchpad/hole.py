import sys; sys.path.insert(0,"src")
import numpy as np, torch
from formant_ml.engine.tube import webster_modes, loss_bandwidths, uniform_area, C_SOUND
FS=48000.0; L=14.6; N=44
A=uniform_area(N,3.0,dtype=torch.float64)

def allpole_db(f,b,w):
    """아날로그 전 극 전달함수의 크기 [dB]. DC 에서 0 dB 로 정규화."""
    s=1j*w; H=np.ones_like(s)
    for fk,bk in zip(f,b):
        sk=-np.pi*bk+2j*np.pi*fk
        H=H*(sk*np.conj(sk))/((s-sk)*(s-np.conj(sk)))
    return 20*np.log10(np.abs(H)+1e-30)

w=2*np.pi*np.linspace(20,24000,4000)
fr=w/(2*np.pi)

# --- 관: 나이퀴스트까지 전 극 + 계산된 대역폭 ---
f,p=webster_modes(A,L); b=loss_bandwidths(A,L,f,p,wall_shape=1.3)
f=f.numpy(); b=b.numpy(); m=f<24000
ft,bt=f[m],b[m]
db_tube=allpole_db(ft,bt,w)

# --- 현재 엔진: 포먼트 8 개 (bw = 40+0.05f) + 고차 보정 4 개 (Q≈1) ---
f1=C_SOUND/(4*L)
fe=[(2*n-1)*f1 for n in range(1,9)]
be=[40.0+0.05*x for x in fe]
ex=[(2*n-1)*f1 for n in range(9,128) if (2*n-1)*f1<0.60*FS/2]
fe+=ex; be+=[max(40.0+1.0*x,800.0) for x in ex]
db_cur=allpole_db(np.array(fe),np.array(be),w)

def band(db,lo,hi): 
    s=(fr>=lo)&(fr<hi); return db[s].mean()
print(f"관 극 {len(ft)} 개 (최고 {ft[-1]:.0f} Hz),  현재 엔진 극 {len(fe)} 개 (최고 {fe[-1]:.0f} Hz)")
print(f"현재 엔진 고차보정 극: {[f'{x:.0f}' for x in ex]}")
print()
print(f"{'대역 [kHz]':>12s} {'관 dB':>8s} {'현행 dB':>8s} {'차이':>8s}")
for lo,hi in [(0.2,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(8.5,9.5),(9,10),(10,12),(12,16),(16,24)]:
    a1,a2=band(db_tube,lo*1000,hi*1000),band(db_cur,lo*1000,hi*1000)
    print(f"{lo:5.1f}-{hi:4.1f} {a1:8.1f} {a2:8.1f} {a1-a2:8.1f}")
print()
# 구멍 = 국소 포락 대비 얼마나 파였는가. 3 kHz 폭 이동평균 대비.
def hole(db):
    k=int(3000/(fr[1]-fr[0])); k+= (k+1)%2
    env=np.convolve(db,np.ones(k)/k,mode="same")
    return db-env
h_t,h_c=hole(db_tube),hole(db_cur)
print("국소 포락 대비 파임 [dB]  (음수 = 구멍)")
print(f"{'대역':>12s} {'관':>8s} {'현행':>8s}")
for lo,hi in [(6,7),(7,8),(8,9),(8.5,9.5),(9,10),(10,11),(11,12)]:
    s=(fr>=lo*1000)&(fr<hi*1000)
    print(f"{lo:5.1f}-{hi:4.1f} {h_t[s].min():8.2f} {h_c[s].min():8.2f}")
s=(fr>4000)&(fr<14000)
print()
print(f"4~14 kHz 최악의 구멍:  관 {h_t[s].min():.2f} dB @ {fr[s][h_t[s].argmin()]:.0f} Hz"
      f"   |   현행 {h_c[s].min():.2f} dB @ {fr[s][h_c[s].argmin()]:.0f} Hz")
np.save("scratchpad/hole.npy",np.stack([fr,db_tube,db_cur]))
