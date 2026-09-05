"""귀로 A/B 하기 위한 세트를 만든다 — 원본 / 이전 / 현행.

    PYTHONPATH=src python3 scripts/make_listening_set.py
    -> out/listen/<토큰>_{a_original,b_before,c_after}.wav
       out/listen/<토큰>_ABC.wav   (셋을 0.3 초 간격으로 이어 붙인 것)

'이전' 은 **예전 직사각 블록 OLA + tilt=-12/Rd=0.6** 으로 다시 렌더한다.
`diag_hifreq.ltv_rect` 가 예전 구현을 그대로 들고 있어서 가능하다.

레벨을 맞추는 데 두 번 걸렸으니 적어 둔다:
1. **전체 RMS 가 아니라 발화 구간 RMS** 로 맞춰야 한다. 무음이 긴 파일
   (5_all)이 10 dB 조용해진다.
2. **`save_wav(..., normalize=False)`** 를 줘야 한다. 기본이 피크 정규화라,
   여기서 맞춘 것을 파일 쓰는 순간 지운다.
라우드니스가 다르면 사람은 큰 쪽을 '좋다' 고 듣는다. 음색만 비교하려면 이게 먼저다.

전부 24 kHz 다 — 합성기가 12 kHz 위를 못 내므로 원본만 44.1 kHz 로 들려주면
비교가 안 된다.
"""
import sys, os
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np, torch
import formant_ml.models.synth as S
from formant_ml.analysis.acoustic import load
from formant_ml.config import Config
from formant_ml.utils import save_wav
from copysynth import (DEFAULT_RD, DEFAULT_TILT, analyse, build_controls,
                       match_envelope)
from diag_hifreq import ltv_rect

SR=24000; cfg=Config(); HOP=cfg.audio.hop_size  # 24 kHz = 나이퀴스트 12 kHz
REC="reference/recordings/ko_liquid_ra-eulla-ara_male_44k.wav"
OUT="out/listen"; os.makedirs(OUT,exist_ok=True)
TOK={"1_ra":(0.50,1.02),"2_eulla":(1.58,2.55),"3_ara":(3.20,3.86),
     "4_ara2":(5.36,6.08),"5_all":(0.40,6.20)}

def render(y,rect,tilt,rd):
    a=analyse(y,cfg)
    orig=S.ltv_filter
    if rect: S.ltv_filter=ltv_rect
    try:
        torch.manual_seed(0)
        syn=S.PhysicalVoiceSynth(cfg,tract_mode="formant")
        c=build_controls(a,cfg,0.002,0.02,tilt,rd)
        with torch.no_grad(): o=syn(c)["audio"][0].numpy().astype(np.float64)
        return match_envelope(o,a["rms"],HOP)
    finally:
        S.ltv_filter=orig

def active_rms(x, hop=240, floor_db=-35.0):
    """**발화 구간만** 의 RMS. 전체 RMS 로 맞추면 무음이 긴 파일이 조용해진다
    (5_all 이 -33 dBFS 로 나왔다)."""
    t=len(x)//hop
    fr=x[:t*hop].reshape(t,hop)
    r=np.sqrt((fr**2).mean(1))
    m=r.max()
    act=r[r>m*10**(floor_db/20)]
    return float(np.sqrt((act**2).mean())) if len(act) else float(m)

def rms_match(sigs, target=0.10):
    """세 신호를 **같은 (발화구간) RMS** 로 맞춘다. 라우드니스 차이가 음색
    비교를 덮는다.

    피크가 넘치면 **셋 모두에 같은 감쇠**를 건다 — 하나만 줄이면 RMS 매칭이
    깨진다(처음에 그렇게 해서 1_ra 가 5 dB, 5_all 이 12 dB 어긋났다).
    그리고 `save_wav` 는 기본이 피크 정규화라 `normalize=False` 가 필요하다 —
    안 주면 여기서 맞춘 것이 그대로 지워진다(두 번째로 걸린 함정).
    """
    out=[x*(target/max(active_rms(x),1e-12)) for x in sigs]
    pk=max(np.abs(x).max() for x in out)
    g=0.95/pk if pk>0.95 else 1.0
    return [x*g for x in out]

raw,_=load(REC,SR)
gap=np.zeros(int(0.30*SR))
for name,(t0,t1) in TOK.items():
    y=raw[int(t0*SR):int(t1*SR)]; y=y/max(abs(y).max(),1e-9)
    before=render(y,True,-12.0,0.6)
    after =render(y,False, DEFAULT_TILT, DEFAULT_RD)   # 기본값을 그대로 읽는다
    n=min(len(y),len(before),len(after))
    trio=rms_match([y[:n].astype(np.float64),before[:n],after[:n]])
    for tag,x in zip(("a_original","b_before","c_after"),trio):
        save_wav(f"{OUT}/{name}_{tag}.wav",torch.from_numpy(x).float(),SR,
                 normalize=False)
    seq=np.concatenate([trio[0],gap,trio[1],gap,trio[2]])
    save_wav(f"{OUT}/{name}_ABC.wav",torch.from_numpy(seq).float(),SR,
             normalize=False)
    print(f"{name}: {n/SR:.2f}s  -> {name}_ABC.wav (원본 / 이전 / 현행)")
