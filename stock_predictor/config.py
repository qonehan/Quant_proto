"""
하이퍼파라미터 및 설정값 정의.

핵심 하이퍼파라미터:
  n: 모델 입력으로 사용할 과거 분봉 수 (lookback window)
  x: 기울기 계산에 사용할 목표 변동률 (%)

기울기 정의:
  현재 시점에서 가격이 x% 변동하는 데 걸린 시간(분)을 t라 할 때,
  상승: slope = +x / t  (양의 기울기)
  하락: slope = -x / t  (음의 기울기)
  상승/하락 중 먼저 발생한 방향을 채택.
  관측 윈도우 내에서 x% 변동이 없으면 slope = 0
"""

from dataclasses import dataclass, field


@dataclass
class HyperParams:
    # --- 핵심 하이퍼파라미터 ---
    n: int = 60               # 입력 시퀀스 길이 (분)
    x: float = 1.0            # 목표 변동률 (%)

    # --- 기울기 계산 ---
    slope_max_window: int = 60  # 기울기 산출 시 최대 관측 윈도우 (분)

    # --- 종목 ---
    ticker: str = "005930"     # 삼성전자 종목코드
    ticker_yf: str = "005930.KS"  # yfinance용 티커

    # --- 피처 관련 ---
    ma_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    stoch_period: int = 14       # Stochastic Oscillator 기간
    stoch_smooth: int = 3        # Stochastic %D 스무딩
    atr_period: int = 14         # ATR 기간
    cci_period: int = 20         # CCI 기간
    roc_period: int = 10         # ROC 기간
    mfi_period: int = 14         # MFI 기간

    # --- 모델 ---
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # --- 학습 ---
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    patience: int = 10        # early stopping patience
    train_ratio: float = 0.8
    val_ratio: float = 0.1    # 나머지가 test

    # --- 데이터 수집 ---
    yf_period: str = "5d"     # yfinance 데이터 기간
    yf_interval: str = "1m"   # yfinance 분봉 간격
