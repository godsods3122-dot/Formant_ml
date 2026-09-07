"""formant_ml.engine — v2 물리 음성 엔진 (시간영역, 인과, 위상 연속).

v1(`formant_ml.dsp`, `formant_ml.models`)과의 차이는 **필터를 어디서 계산하느냐**다.

* v1: 프레임별 주파수응답 -> 유한 IR -> 창 없는 블록 OLA. 응답이 10 ms 마다
  계단으로 바뀌고, 영위상 성분(tilt/skirt)이 프리에코를 만들며, 과도응답이
  없다. "라" 의 타격 지점에서 주파수별 위상이 찢어진 원인이다(docs/RIEUL.md §6).
* v2: **샘플 단위로 계수가 변하는 재귀 필터**. 상태가 샘플마다 이어지므로
  자음처럼 경계조건이 급변해도 위상은 필터 상태의 연속성으로 자동 보장되고,
  과도응답은 방정식이 낸다. 학습은 결합 스캔(associative scan)으로 시퀀스 길이에
  대해 O(N log N) 병렬이며 정확히 미분가능하다(`tviir.py`).

파이프라인 (사용자 명세 순서):

    압력 구동 성문 소스 ──┐
    노이즈 생성기(마찰/기식/과도음 템플릿) ──┤
                           ▼
    포먼트 + 위상차(올패스) 필터 -> 비강 필터 -> 잔차 보정(shifted-softplus 망) -> 음성

모듈:
    tviir     시변 2차 재귀 필터 원시연산 (scan / numpy 블록 / numba 실시간)
    control   제어 트랙(ms 단위 스크립트) <-> 샘플률 보간
    glottis   압력·내전·긴장 -> 성문 유량미분 (LF) + 기식
    noise     마찰(협착 레이놀즈), 기식, 과도음 템플릿 뱅크
    tract     포먼트 캐스케이드 + 올패스 위상 + 비강 극영점 + 방사
    residual  잔차 보정망 (상한 있는 최소위상 EQ + 템플릿 노이즈 혼합)
    nn        shifted softplus 활성/MLP/TCN
    tokens    감정 토큰 레지스트리 (학습 중 동적 생성, 식별, 후처리 분리)
    voice     VoiceEngine — 스크립트 -> 오디오, 스트리밍
    phones    한국어 음소 제스처 라이브러리 (여성 화자 기본값)
"""
try:
    from .voice import VoiceEngine, EngineConfig  # noqa: F401
except ImportError:  # 부분 조립 중
    pass
