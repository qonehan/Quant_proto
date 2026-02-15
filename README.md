# 삼성전자 주가 기울기(Slope) 예측 모델

삼성전자(005930)의 1분봉 데이터를 수집하고, 기술적 지표 피처를 생성하여
**LSTM + Temporal Attention** 기반 시계열 모델로 주가 **상승/하락 기울기**를 예측하는 프로젝트입니다.

---

## 1. 프로젝트 구조

```
Quant_proto/
├── requirements.txt                 # 의존성 패키지 목록
├── .gitignore
├── README.md
└── stock_predictor/
    ├── __init__.py
    ├── __main__.py                  # python -m stock_predictor 실행용
    ├── config.py                    # 하이퍼파라미터 정의
    ├── data_fetcher.py              # 데이터 수집 (yfinance / 한투 API)
    ├── feature_engineer.py          # 기술적 지표 피처 생성 + 시퀀스 행렬 구축
    ├── target_builder.py            # 양방향 기울기(slope) 타겟 변수 생성
    ├── model.py                     # LSTM + Attention 모델 정의
    ├── train.py                     # 학습 파이프라인
    ├── predict.py                   # 추론 클래스
    └── main.py                      # CLI 진입점 (train / predict)
```

---

## 2. 핵심 개념: 양방향 기울기(Slope) 정의

### 수식

현재 시점 `t`의 종가를 `P(t)`라 할 때:

1. 미래 시점 `t+1, t+2, ...`를 순회하며 가격이 **x% 이상 상승 또는 하락**한 첫 시점 `t+k`를 찾는다
2. **상승이 먼저 발생**: `slope = +x / k` (양의 기울기)
3. **하락이 먼저 발생**: `slope = -x / k` (음의 기울기)
4. `slope_max_window` 내에서 어느 방향으로도 x% 변동이 없으면 **slope = 0**

### 해석

| slope 값 | 의미 |
|-----------|------|
| 양수 (예: +1.0) | 급격한 상승 — 1%가 1분 만에 상승 |
| 양수 (예: +0.05) | 완만한 상승 — 1%가 20분에 걸쳐 상승 |
| 음수 (예: -1.0) | 급격한 하락 — 1%가 1분 만에 하락 |
| 음수 (예: -0.05) | 완만한 하락 — 1%가 20분에 걸쳐 하락 |
| 0 | 관측 윈도우 내에서 x% 변동이 발생하지 않음 (횡보) |

### 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `n` | 60 | 모델 입력 시퀀스 길이 (분) |
| `x` | 1.0 | 목표 변동률 (%) |
| `slope_max_window` | 60 | 기울기 산출 시 최대 관측 윈도우 (분) |

`n`과 `x`는 모델에서 **고정된 하이퍼파라미터**로 사용됩니다.

---

## 3. 데이터 구조

### 3-1. 원본 데이터 (1분봉 OHLCV)

`data_fetcher.py`에서 수집하며, 두 가지 소스를 지원합니다:

| 소스 | 용도 | 특징 |
|------|------|------|
| **yfinance** | 과거 데이터 학습 | `005930.KS`, 최대 7일 1분봉 |
| **한국투자증권 API (mojito)** | 실시간 수집 | 웹소켓 체결가 → 1분봉 집계 |

```
DataFrame (T rows):
  index: DatetimeIndex (1분 간격)
  columns: open, high, low, close, volume
```

### 3-2. 피처 행렬

`feature_engineer.py`의 `build_features()`가 원본 OHLCV에서 **29개 기술적 지표**를 생성합니다:

#### 기본 피처 (14개)

| # | 피처명 | 카테고리 | 설명 |
|---|--------|----------|------|
| 1 | `pct_change` | 가격 | 종가 변화율 |
| 2 | `ma5_gap` | 추세 | 5분 이동평균 대비 괴리율 |
| 3 | `ma10_gap` | 추세 | 10분 이동평균 대비 괴리율 |
| 4 | `ma20_gap` | 추세 | 20분 이동평균 대비 괴리율 |
| 5 | `rsi` | 모멘텀 | RSI (14분, 0~1 정규화) |
| 6 | `macd` | 추세 | MACD / 종가 |
| 7 | `macd_signal` | 추세 | MACD Signal / 종가 |
| 8 | `macd_hist` | 추세 | MACD Histogram / 종가 |
| 9 | `bb_pctb` | 변동성 | 볼린저밴드 %b |
| 10 | `bb_bandwidth` | 변동성 | 볼린저밴드 너비 / 이동평균 |
| 11 | `volume_pct` | 거래량 | 거래량 변화율 |
| 12 | `hl_range` | 변동성 | (고가 - 저가) / 종가 |
| 13 | `open_close_ratio` | 가격 | (시가 - 종가) / 종가 |
| 14 | `volume_ma5_ratio` | 거래량 | 거래량 / 거래량 5분 이동평균 |

#### 추가 피처 (15개)

| # | 피처명 | 카테고리 | 설명 |
|---|--------|----------|------|
| 15 | `stoch_k` | 모멘텀 | Stochastic Oscillator %K (0~1) |
| 16 | `stoch_d` | 모멘텀 | Stochastic Oscillator %D (0~1) |
| 17 | `williams_r` | 모멘텀 | Williams %R (-1~0) |
| 18 | `atr_ratio` | 변동성 | ATR (Average True Range) / 종가 |
| 19 | `obv_change` | 거래량 | OBV (On Balance Volume) 변화율 |
| 20 | `cci` | 추세/모멘텀 | CCI (Commodity Channel Index) / 200 |
| 21 | `roc` | 모멘텀 | ROC (Rate of Change, 10분) |
| 22 | `mfi` | 거래량 | MFI (Money Flow Index, 0~1) |
| 23 | `vwap_gap` | 가격 | VWAP 대비 괴리율 |
| 24 | `candle_body_ratio` | 캔들스틱 | 캔들 몸통 비율: \|open-close\| / (high-low) |
| 25 | `upper_shadow` | 캔들스틱 | 윗꼬리 비율 |
| 26 | `lower_shadow` | 캔들스틱 | 아래꼬리 비율 |
| 27 | `price_position` | 가격 | 가격 위치: (close-low) / (high-low) |
| 28 | `high_pct_change` | 가격 | 고가 변화율 |
| 29 | `low_pct_change` | 가격 | 저가 변화율 |

#### 추가 피처 설정 (config.py)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `stoch_period` | 14 | Stochastic Oscillator 기간 |
| `stoch_smooth` | 3 | Stochastic %D 스무딩 기간 |
| `atr_period` | 14 | ATR 기간 |
| `cci_period` | 20 | CCI 기간 |
| `roc_period` | 10 | ROC 기간 |
| `mfi_period` | 14 | MFI 기간 |

NaN이 발생하는 초기 워밍업 구간은 자동 제거됩니다.
무한대(inf) 값도 자동으로 제거됩니다.

### 3-3. 시퀀스 행렬 (모델 입력)

`build_sequence_matrix()`가 피처 DataFrame을 슬라이딩 윈도우 방식으로 3D 텐서로 변환합니다:

```
입력 shape: (samples, n, 29)
                │      │   │
                │      │   └── 피처 수 (29개 기술적 지표)
                │      └────── 시퀀스 길이 (n분, 기본 60)
                └───────────── 샘플 수 (T - warmup - n + 1)
```

각 샘플은 **연속 n분의 피처 벡터**로 구성된 행렬입니다.
학습 시 `StandardScaler`로 피처 차원별 정규화가 적용됩니다.

### 3-4. 타겟 (slope)

`target_builder.py`가 각 시퀀스 윈도우의 **마지막 시점**에서의 slope를 타겟으로 할당합니다:

```
타겟 shape: (samples,)
                │
                └── 각 윈도우의 마지막 시점에서 계산된 slope 값
                    양수: 상승 기울기
                    음수: 하락 기울기
                    0:    변동 없음 (횡보)
```

---

## 4. 모델 아키텍처

```
Input (batch, n, 29)
        │
   ┌────▼────┐
   │LayerNorm│  ← 입력 정규화
   └────┬────┘
        │
   ┌────▼────┐
   │  LSTM   │  ← 2층, hidden_size=128
   │(2-layer)│    dropout=0.2
   └────┬────┘
        │
        ▼
  (batch, n, 128)
        │
   ┌────▼─────────┐
   │   Temporal    │  ← 타임스텝별 어텐션 가중치 학습
   │  Attention    │    Linear → Tanh → Linear → Softmax
   └────┬──────────┘
        │
        ▼
  (batch, 128)     ← 가중합으로 컨텍스트 벡터 생성
        │
   ┌────▼────┐
   │ FC Head │  ← Linear(128→64) → ReLU → Dropout → Linear(64→1)
   └────┬────┘
        │
        ▼
  (batch,)         ← 예측 기울기 (slope): 양수=상승, 음수=하락
```

### 모델 파라미터

| 설정 | 기본값 |
|------|--------|
| LSTM hidden_size | 128 |
| LSTM num_layers | 2 |
| Dropout | 0.2 |
| Bidirectional | False |
| 총 파라미터 수 | ~230,000 |

---

## 5. 학습 파이프라인

`train.py`의 `train()` 함수가 아래 흐름을 실행합니다:

```
데이터 로드 → 피처 생성 (29개) → 타겟 생성 (양방향 slope)
    → 시퀀스 행렬 구축 → StandardScaler 정규화
    → Train/Val/Test 분할 (80/10/10)
    → LSTM 학습 (Adam, MSE Loss, Gradient Clipping)
    → ReduceLROnPlateau + Early Stopping
    → 최적 모델 저장 → 테스트 평가
```

| 학습 설정 | 기본값 |
|-----------|--------|
| Optimizer | Adam (lr=1e-3) |
| Loss | MSE |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early Stopping patience | 10 epochs |
| Gradient Clipping | max_norm=1.0 |
| 데이터 분할 | 시계열 순서 유지 (Train 80% / Val 10% / Test 10%) |

### 평가 지표

| 지표 | 설명 |
|------|------|
| MSE | 평균 제곱 오차 |
| MAE | 평균 절대 오차 |
| Direction Accuracy | 부호(상승/하락/횡보) 일치 비율 (`np.sign` 기반) |

### 체크포인트 저장 내용

`checkpoints/best_model.pt`에 아래 정보가 저장됩니다:

```python
{
    "model_state":  모델 가중치,
    "scaler_mean":  StandardScaler mean,
    "scaler_scale": StandardScaler scale,
    "num_features": 피처 수 (29),
    "params":       HyperParams 인스턴스,
}
```

---

## 6. 사용법

### 설치

```bash
pip install -r requirements.txt
```

### 학습

```bash
# yfinance 과거 데이터로 학습
python -m stock_predictor train

# 모의 데이터로 학습 테스트
python -m stock_predictor train --mock --mock-minutes 500

# 하이퍼파라미터 변경
python -m stock_predictor train --n 30 --x 0.5 --epochs 200 --lr 0.0005
```

### 추론

```bash
# 학습된 모델로 현재 기울기 예측
python -m stock_predictor predict

# 특정 체크포인트 사용
python -m stock_predictor predict --checkpoint checkpoints/best_model.pt
```

### 출력 예시

```
삼성전자 (005930) 기울기 예측
  설정: n=60분, x=1.0%
  예측 기울기: 0.0832
  방향: 상승
  해석: 1.0% 상승에 약 12.0분 소요 예상
```

```
삼성전자 (005930) 기울기 예측
  설정: n=60분, x=1.0%
  예측 기울기: -0.1250
  방향: 하락
  해석: 1.0% 하락에 약 8.0분 소요 예상
```

---

## 7. 전체 데이터 흐름

```
[삼성전자 1분봉 OHLCV]
        │
        ▼
┌─────────────────┐
│  data_fetcher   │  yfinance (과거) 또는 한투API (실시간)
└────────┬────────┘
         │  DataFrame (T, 5): open/high/low/close/volume
         ▼
┌─────────────────┐
│feature_engineer │  29개 기술적 지표 생성
└────────┬────────┘    가격(6) + 추세(5) + 모멘텀(6) + 변동성(4)
         │            + 거래량(4) + 캔들스틱(4)
         │  DataFrame (T', 29)  ← NaN/inf 구간 제거
         ▼
┌─────────────────┐            ┌──────────────┐
│build_sequence   │            │target_builder│
│    _matrix      │            │ (양방향 slope)│
└────────┬────────┘            └──────┬───────┘
         │                            │
    (samples, n, 29)           (samples,) slope
         │                     (+:상승, -:하락, 0:횡보)
         ▼                            ▼
┌─────────────────────────────────────────────┐
│              StandardScaler                 │
│         Train / Val / Test 분할              │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│        SlopePredictorLSTM                   │
│   LayerNorm → LSTM(2) → Attention → FC     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
              예측 slope 값
         (+:상승 예측, -:하락 예측)
```

---

## 8. 의존성

| 패키지 | 용도 |
|--------|------|
| `torch` | LSTM 모델 학습/추론 |
| `numpy` | 수치 연산 |
| `pandas` | 시계열 데이터 처리 |
| `yfinance` | 삼성전자 과거 1분봉 데이터 수집 |
| `scikit-learn` | StandardScaler 정규화 |
| `mojito2` | 한국투자증권 Open API (실시간 수집) |
| `websockets` | 실시간 웹소켓 연결 |
