"""
비트코인 기울기 예측 모델 하이퍼파라미터 및 설정값 정의.

데이터 소스:
  1. Binance REST API — OHLCV 1분봉, 펀딩비, 미결제약정 (무료, API키 불필요)
  2. CoinMetrics Community API — MVRV, NVT, 활성 주소 등 온체인 지표 (무료)
  3. Alternative.me API — Fear & Greed Index (무료)

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

    # --- 심볼 ---
    symbol: str = "BTCUSDT"        # Binance 거래쌍
    symbol_display: str = "BTC"    # 표시용 심볼

    # --- 피처 관련 (기술적 지표) ---
    ma_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    stoch_period: int = 14
    stoch_smooth: int = 3
    atr_period: int = 14
    cci_period: int = 20
    roc_period: int = 10
    mfi_period: int = 14

    # --- 온체인 데이터 ---
    use_onchain: bool = True       # 온체인 지표 사용 여부
    use_exchange_metrics: bool = True  # 거래소 지표 (펀딩비, OI) 사용 여부
    use_fear_greed: bool = True    # Fear & Greed 지수 사용 여부

    # --- 모델 ---
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # --- 학습 ---
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    patience: int = 10
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # --- 데이터 수집 ---
    binance_interval: str = "1m"   # Binance 캔들 간격
    fetch_days: int = 5            # 수집할 일수
