"""
비트코인 데이터를 기술적 지표 + 암호화폐 특화 피처 행렬로 변환하는 모듈.

생성되는 피처 카테고리:

  [기술적 지표 — OHLCV 기반, 29개]
  가격: pct_change, open_close_ratio, high_pct_change, low_pct_change
  추세: MA 괴리율 (5/10/20), MACD/signal/histogram
  모멘텀: RSI, Stochastic K/D, Williams %R, ROC, CCI
  변동성: BB %b/bandwidth, ATR ratio, hl_range
  거래량: volume_pct, volume_ma5_ratio, OBV change, MFI
  캔들: body ratio, upper/lower shadow, price position

  [암호화폐 특화 피처 — Binance 거래소 데이터, 최대 5개]
  taker_buy_ratio: 테이커 매수 비율 (매수 압력 지표)
  trade_intensity: 체결 강도 (거래 건수 변화율)
  funding_rate: 펀딩비 (선물 시장 방향성)
  oi_pct_change: 미결제약정 변화율
  quote_vol_ratio: 거래대금 / 거래대금 5분 MA

  [온체인 지표 — CoinMetrics 일별, 최대 5개]
  mvrv_norm: MVRV 비율 (정규화)
  nvt_norm: NVT 비율 (정규화)
  active_addr_change: 활성 주소 변화율
  tx_count_change: 트랜잭션 수 변화율
  hash_rate_change: 해시레이트 변화율

  [센티먼트 — Alternative.me 일별, 1개]
  fear_greed_norm: Fear & Greed Index (0~1 정규화)

  총 최대 40개 피처 (보조 지표 미수집 시 29개)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import HyperParams


# ---------------------------------------------------------------------------
# 기술적 지표 헬퍼 함수
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
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
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
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


# ---------------------------------------------------------------------------
# 메인 피처 빌드 함수
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, params: HyperParams) -> pd.DataFrame:
    """통합 DataFrame에서 모든 피처를 생성한다.

    Args:
        df: columns에 open, high, low, close, volume 필수.
            선택적으로 taker_buy_ratio, trades, quote_volume,
            funding_rate, open_interest, mvrv, nvt,
            active_addresses, tx_count, hash_rate, fear_greed 포함.
        params: 하이퍼파라미터

    Returns:
        피처 DataFrame (NaN/inf 구간 제거됨)
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    opn = df["open"]
    volume = df["volume"]

    # ===== 기술적 지표 (29개) =====

    # 1. 가격 변화율
    feat["pct_change"] = close.pct_change()

    # 2-4. 이동평균 괴리율
    for w in params.ma_windows:
        ma = close.rolling(w).mean()
        feat[f"ma{w}_gap"] = (close - ma) / ma

    # 5. RSI
    feat["rsi"] = _rsi(close, params.rsi_period) / 100.0

    # 6-8. MACD
    macd_line, macd_signal, macd_hist = _macd(close)
    feat["macd"] = macd_line / close
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

    # 13-14. 시가-종가 비율, 거래량 MA 비율
    feat["open_close_ratio"] = (opn - close) / close
    feat["volume_ma5_ratio"] = volume / volume.rolling(5).mean()

    # 15-16. Stochastic Oscillator
    stoch_k, stoch_d = _stochastic(
        high, low, close, params.stoch_period, params.stoch_smooth
    )
    feat["stoch_k"] = stoch_k / 100.0
    feat["stoch_d"] = stoch_d / 100.0

    # 17. Williams %R
    feat["williams_r"] = _williams_r(
        high, low, close, params.stoch_period
    ) / 100.0

    # 18. ATR / close
    feat["atr_ratio"] = _atr(high, low, close, params.atr_period) / close

    # 19. OBV 변화율
    obv = _obv(close, volume)
    feat["obv_change"] = obv.pct_change().clip(-10, 10)

    # 20. CCI
    feat["cci"] = _cci(high, low, close, params.cci_period) / 200.0

    # 21. ROC
    feat["roc"] = close.pct_change(periods=params.roc_period)

    # 22. MFI
    feat["mfi"] = _mfi(
        high, low, close, volume, params.mfi_period
    ) / 100.0

    # 23. VWAP 괴리율
    vwap = _vwap(high, low, close, volume)
    feat["vwap_gap"] = (close - vwap) / vwap

    # 24-26. 캔들스틱 패턴
    candle_range = (high - low).replace(0, np.nan)
    feat["candle_body_ratio"] = (opn - close).abs() / candle_range
    feat["upper_shadow"] = (high - pd.concat([opn, close], axis=1).max(axis=1)) / candle_range
    feat["lower_shadow"] = (pd.concat([opn, close], axis=1).min(axis=1) - low) / candle_range

    # 27. 가격 위치
    feat["price_position"] = (close - low) / candle_range

    # 28-29. 고가/저가 변화율
    feat["high_pct_change"] = high.pct_change()
    feat["low_pct_change"] = low.pct_change()

    # ===== 암호화폐 특화 피처 =====

    # 30. 테이커 매수 비율 (매수 압력 지표)
    if "taker_buy_ratio" in df.columns:
        feat["taker_buy_ratio"] = df["taker_buy_ratio"]

    # 31. 체결 강도 (거래 건수 변화율)
    if "trades" in df.columns:
        feat["trade_intensity"] = df["trades"].pct_change().clip(-10, 10)

    # 32. 거래대금 비율
    if "quote_volume" in df.columns:
        qv = df["quote_volume"]
        feat["quote_vol_ratio"] = qv / qv.rolling(5).mean()

    # 33. 펀딩비
    if "funding_rate" in df.columns:
        feat["funding_rate"] = df["funding_rate"]

    # 34. 미결제약정 변화율
    if "open_interest" in df.columns:
        feat["oi_pct_change"] = df["open_interest"].pct_change().clip(-10, 10)

    # ===== 온체인 지표 =====

    # 35. MVRV (정규화: 중앙값 기준 편차)
    if "mvrv" in df.columns:
        mvrv = df["mvrv"]
        feat["mvrv_norm"] = ((mvrv - mvrv.rolling(100, min_periods=1).median()) /
                             mvrv.rolling(100, min_periods=1).std().replace(0, np.nan)).clip(-3, 3)

    # 36. NVT (정규화)
    if "nvt" in df.columns:
        nvt = df["nvt"]
        feat["nvt_norm"] = ((nvt - nvt.rolling(100, min_periods=1).median()) /
                            nvt.rolling(100, min_periods=1).std().replace(0, np.nan)).clip(-3, 3)

    # 37. 활성 주소 변화율
    if "active_addresses" in df.columns:
        feat["active_addr_change"] = df["active_addresses"].pct_change().clip(-1, 1)

    # 38. 트랜잭션 수 변화율
    if "tx_count" in df.columns:
        feat["tx_count_change"] = df["tx_count"].pct_change().clip(-1, 1)

    # 39. 해시레이트 변화율
    if "hash_rate" in df.columns:
        feat["hash_rate_change"] = df["hash_rate"].pct_change().clip(-1, 1)

    # ===== 센티먼트 =====

    # 40. Fear & Greed Index (0~1)
    if "fear_greed" in df.columns:
        feat["fear_greed_norm"] = df["fear_greed"] / 100.0

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
