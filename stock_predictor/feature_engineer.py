"""
1분봉 OHLCV 데이터를 기술적 지표 피처 행렬로 변환하는 모듈.

생성되는 피처:
  [기존]
  - 가격 변화율 (close pct change)
  - 이동평균 (MA5, MA10, MA20) 대비 괴리율
  - RSI
  - MACD, MACD signal, MACD histogram
  - 볼린저밴드 %b, bandwidth
  - 거래량 변화율
  - 고가-저가 범위 비율 (high-low range / close)
  - 시가-종가 비율
  - 거래량 MA5 비율

  [추가]
  - Stochastic Oscillator (%K, %D)
  - Williams %R
  - ATR (Average True Range) / close
  - OBV (On Balance Volume) 변화율
  - CCI (Commodity Channel Index)
  - ROC (Rate of Change)
  - MFI (Money Flow Index)
  - VWAP 괴리율
  - 캔들스틱 패턴: body ratio, upper shadow, lower shadow
  - 가격 위치 (일봉 내 close 위치)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import HyperParams


# ---------------------------------------------------------------------------
# 개별 지표 헬퍼 함수
# ---------------------------------------------------------------------------

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


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    smooth: int,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator %K, %D를 계산한다."""
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    pct_k = 100 * (close - lowest_low) / denom
    pct_d = pct_k.rolling(smooth).mean()
    return pct_k, pct_d


def _williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Williams %R을 계산한다. (-100 ~ 0 범위)"""
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return -100 * (highest_high - close) / denom


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Average True Range를 계산한다."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume을 계산한다."""
    direction = close.diff().apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    return (volume * direction).cumsum()


def _cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Commodity Channel Index를 계산한다."""
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int,
) -> pd.Series:
    """Money Flow Index를 계산한다. (거래량 가중 RSI)"""
    tp = (high + low + close) / 3
    raw_money_flow = tp * volume
    tp_diff = tp.diff()

    positive_flow = pd.Series(0.0, index=close.index)
    negative_flow = pd.Series(0.0, index=close.index)
    positive_flow[tp_diff > 0] = raw_money_flow[tp_diff > 0]
    negative_flow[tp_diff < 0] = raw_money_flow[tp_diff < 0]

    pos_sum = positive_flow.rolling(period).sum()
    neg_sum = negative_flow.rolling(period).sum()
    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + money_ratio))


def _vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """VWAP (Volume Weighted Average Price)을 계산한다.

    분봉 데이터이므로 누적 VWAP을 사용한다.
    """
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


# ---------------------------------------------------------------------------
# 메인 피처 빌드 함수
# ---------------------------------------------------------------------------

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
    high = df["high"]
    low = df["low"]
    opn = df["open"]
    volume = df["volume"]

    # ===== 기존 피처 (14개) =====

    # 1. 가격 변화율
    feat["pct_change"] = close.pct_change()

    # 2-4. 이동평균 괴리율
    for w in params.ma_windows:
        ma = close.rolling(w).mean()
        feat[f"ma{w}_gap"] = (close - ma) / ma

    # 5. RSI
    feat["rsi"] = _rsi(close, params.rsi_period) / 100.0  # 0~1 정규화

    # 6-8. MACD
    macd_line, macd_signal, macd_hist = _macd(close)
    feat["macd"] = macd_line / close  # 가격 대비 비율로 정규화
    feat["macd_signal"] = macd_signal / close
    feat["macd_hist"] = macd_hist / close

    # 9-10. 볼린저밴드
    bb_ma = close.rolling(params.bb_period).mean()
    bb_std = close.rolling(params.bb_period).std()
    upper = bb_ma + params.bb_std * bb_std
    lower = bb_ma - params.bb_std * bb_std
    band_width = upper - lower
    feat["bb_pctb"] = (close - lower) / band_width.replace(0, np.nan)
    feat["bb_bandwidth"] = band_width / bb_ma

    # 11. 거래량 변화율
    feat["volume_pct"] = volume.pct_change()

    # 12. 고가-저가 범위 비율
    feat["hl_range"] = (high - low) / close

    # 13-14. OHLCV 자체도 변화율 기반으로 포함
    feat["open_close_ratio"] = (opn - close) / close
    feat["volume_ma5_ratio"] = volume / volume.rolling(5).mean()

    # ===== 추가 피처 =====

    # 15-16. Stochastic Oscillator (%K, %D)
    stoch_k, stoch_d = _stochastic(
        high, low, close, params.stoch_period, params.stoch_smooth
    )
    feat["stoch_k"] = stoch_k / 100.0   # 0~1 정규화
    feat["stoch_d"] = stoch_d / 100.0

    # 17. Williams %R
    feat["williams_r"] = _williams_r(
        high, low, close, params.stoch_period
    ) / 100.0  # -1~0 정규화

    # 18. ATR (평균 실질 변동폭) / close — 변동성 지표
    feat["atr_ratio"] = _atr(high, low, close, params.atr_period) / close

    # 19. OBV 변화율
    obv = _obv(close, volume)
    obv_ma = obv.rolling(5).mean()
    feat["obv_change"] = obv.pct_change().clip(-10, 10)  # 극단값 클리핑

    # 20. CCI (Commodity Channel Index)
    feat["cci"] = _cci(high, low, close, params.cci_period) / 200.0  # 정규화

    # 21. ROC (Rate of Change)
    feat["roc"] = close.pct_change(periods=params.roc_period)

    # 22. MFI (Money Flow Index)
    feat["mfi"] = _mfi(
        high, low, close, volume, params.mfi_period
    ) / 100.0  # 0~1 정규화

    # 23. VWAP 괴리율
    vwap = _vwap(high, low, close, volume)
    feat["vwap_gap"] = (close - vwap) / vwap

    # 24. 캔들 body 비율 — |open - close| / (high - low)
    candle_range = (high - low).replace(0, np.nan)
    feat["candle_body_ratio"] = (opn - close).abs() / candle_range

    # 25. Upper shadow 비율 — (high - max(open, close)) / (high - low)
    feat["upper_shadow"] = (high - pd.concat([opn, close], axis=1).max(axis=1)) / candle_range

    # 26. Lower shadow 비율 — (min(open, close) - low) / (high - low)
    feat["lower_shadow"] = (pd.concat([opn, close], axis=1).min(axis=1) - low) / candle_range

    # 27. 가격 위치 — (close - low) / (high - low)
    feat["price_position"] = (close - low) / candle_range

    # 28. 고가 변화율
    feat["high_pct_change"] = high.pct_change()

    # 29. 저가 변화율
    feat["low_pct_change"] = low.pct_change()

    # NaN 제거 (초기 워밍업 구간)
    feat.dropna(inplace=True)

    # 무한대 값 제거
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
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
