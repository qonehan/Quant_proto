"""
1분봉 OHLCV 데이터를 기술적 지표 피처 행렬로 변환하는 모듈.

생성되는 피처:
  - 가격 변화율 (close pct change)
  - 이동평균 (MA5, MA10, MA20) 대비 괴리율
  - RSI
  - MACD, MACD signal, MACD histogram
  - 볼린저밴드 %b, bandwidth
  - 거래량 변화율
  - 고가-저가 범위 비율 (high-low range / close)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import HyperParams


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line, signal, histogram


def build_features(df: pd.DataFrame, params: HyperParams) -> pd.DataFrame:
    """원본 OHLCV DataFrame에 기술적 지표 피처를 추가하여 반환한다.

    Args:
        df: columns = [open, high, low, close, volume]
        params: 하이퍼파라미터

    Returns:
        피처가 추가된 DataFrame (NaN이 생기는 초기 구간은 제거됨)
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    volume = df["volume"]

    # 가격 변화율
    feat["pct_change"] = close.pct_change()

    # 이동평균 괴리율
    for w in params.ma_windows:
        ma = close.rolling(w).mean()
        feat[f"ma{w}_gap"] = (close - ma) / ma

    # RSI
    feat["rsi"] = _rsi(close, params.rsi_period) / 100.0  # 0~1 정규화

    # MACD
    macd_line, macd_signal, macd_hist = _macd(close)
    feat["macd"] = macd_line / close  # 가격 대비 비율로 정규화
    feat["macd_signal"] = macd_signal / close
    feat["macd_hist"] = macd_hist / close

    # 볼린저밴드
    bb_ma = close.rolling(params.bb_period).mean()
    bb_std = close.rolling(params.bb_period).std()
    upper = bb_ma + params.bb_std * bb_std
    lower = bb_ma - params.bb_std * bb_std
    band_width = upper - lower
    feat["bb_pctb"] = (close - lower) / band_width.replace(0, np.nan)
    feat["bb_bandwidth"] = band_width / bb_ma

    # 거래량 변화율
    feat["volume_pct"] = volume.pct_change()

    # 고가-저가 범위 비율
    feat["hl_range"] = (df["high"] - df["low"]) / close

    # OHLCV 자체도 변화율 기반으로 포함
    feat["open_close_ratio"] = (df["open"] - close) / close
    feat["volume_ma5_ratio"] = volume / volume.rolling(5).mean()

    # NaN 제거 (초기 워밍업 구간)
    feat.dropna(inplace=True)

    return feat


def build_sequence_matrix(
    features: pd.DataFrame,
    n: int,
) -> np.ndarray:
    """피처 DataFrame을 (samples, n, num_features) 형태의 3D 행렬로 변환.

    각 샘플은 연속 n분의 피처 벡터로 구성된 슬라이딩 윈도우이다.

    Args:
        features: (T, F) 피처 DataFrame
        n: lookback window 크기

    Returns:
        (T - n + 1, n, F) 형태의 numpy 배열
    """
    arr = features.values.astype(np.float32)
    T, F = arr.shape

    if T < n:
        raise ValueError(
            f"데이터 길이({T})가 윈도우 크기 n={n}보다 작습니다."
        )

    num_samples = T - n + 1
    matrix = np.empty((num_samples, n, F), dtype=np.float32)

    for i in range(num_samples):
        matrix[i] = arr[i: i + n]

    return matrix
