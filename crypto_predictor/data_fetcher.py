"""
비트코인 데이터 수집 모듈.

데이터 소스:
  1. Binance REST API — OHLCV 1분봉 + 테이커 매수 비율 (무료, API키 불필요)
  2. Binance Futures API — 펀딩비, 미결제약정 (무료, API키 불필요)
  3. CoinMetrics Community API — MVRV, NVT, 활성 주소, 해시레이트 (무료)
  4. Alternative.me API — Fear & Greed Index (무료)
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests

if TYPE_CHECKING:
    from .config import HyperParams

logger = logging.getLogger(__name__)

BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4"
FEAR_GREED_BASE = "https://api.alternative.me/fng"


# ---------------------------------------------------------------------------
# 1) Binance OHLCV 수집
# ---------------------------------------------------------------------------

def fetch_binance_ohlcv(
    params: HyperParams,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Binance REST API에서 BTC/USDT 1분봉 OHLCV 데이터를 수집한다.

    Returns:
        DataFrame with columns: open, high, low, close, volume, taker_buy_ratio
        index: DatetimeIndex (UTC)
    """
    symbol = params.symbol
    interval = params.binance_interval
    days = params.fetch_days

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)

    all_candles = []
    current_start = start_ms

    while current_start < end_ms:
        url = f"{BINANCE_SPOT_BASE}/api/v3/klines"
        api_params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        resp = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=api_params, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Binance API 요청 실패 (시도 %d/%d): %s. %d초 후 재시도",
                                   attempt + 1, max_retries, e, wait)
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Binance API 요청 {max_retries}회 실패: {e}"
                    ) from e

        candles = resp.json()
        if not candles:
            break

        all_candles.extend(candles)

        # 다음 페이지: 마지막 캔들의 close_time + 1
        last_close_time = candles[-1][6]
        current_start = last_close_time + 1

        if len(candles) < 1000:
            break

    if not all_candles:
        raise RuntimeError(
            f"Binance에서 {symbol} 데이터를 가져올 수 없습니다."
        )

    # 파싱: [open_time, open, high, low, close, volume, close_time,
    #         quote_volume, trades, taker_buy_base_vol, taker_buy_quote_vol, _]
    rows = []
    for c in all_candles:
        total_vol = float(c[5])
        taker_buy_vol = float(c[9])
        taker_buy_ratio = taker_buy_vol / total_vol if total_vol > 0 else 0.5

        rows.append({
            "timestamp": pd.Timestamp(c[0], unit="ms", tz="UTC"),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": total_vol,
            "quote_volume": float(c[7]),
            "trades": int(c[8]),
            "taker_buy_ratio": taker_buy_ratio,
        })

    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # 결측치 처리
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    logger.info("Binance OHLCV 로드 완료: %d rows, 기간 %s ~ %s",
                len(df), df.index[0], df.index[-1])
    return df


# ---------------------------------------------------------------------------
# 2) Binance Futures — 펀딩비, 미결제약정
# ---------------------------------------------------------------------------

def fetch_funding_rate(params: HyperParams) -> pd.DataFrame:
    """Binance Futures에서 펀딩비 이력을 가져온다.

    Returns:
        DataFrame with columns: funding_rate
        index: DatetimeIndex (UTC)
    """
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
    api_params = {"symbol": params.symbol, "limit": 1000}

    try:
        resp = requests.get(url, params=api_params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("펀딩비 데이터 수집 실패: %s", e)
        return pd.DataFrame(columns=["funding_rate"])

    if not data:
        return pd.DataFrame(columns=["funding_rate"])

    rows = [{
        "timestamp": pd.Timestamp(d["fundingTime"], unit="ms", tz="UTC"),
        "funding_rate": float(d["fundingRate"]),
    } for d in data]

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    logger.info("펀딩비 데이터 로드 완료: %d rows", len(df))
    return df


def fetch_open_interest_hist(params: HyperParams) -> pd.DataFrame:
    """Binance Futures에서 미결제약정 이력을 가져온다.

    Returns:
        DataFrame with columns: open_interest
        index: DatetimeIndex (UTC)
    """
    url = f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist"
    api_params = {"symbol": params.symbol, "period": "5m", "limit": 500}

    try:
        resp = requests.get(url, params=api_params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("미결제약정 데이터 수집 실패: %s", e)
        return pd.DataFrame(columns=["open_interest"])

    if not data:
        return pd.DataFrame(columns=["open_interest"])

    rows = [{
        "timestamp": pd.Timestamp(d["timestamp"], unit="ms", tz="UTC"),
        "open_interest": float(d["sumOpenInterest"]),
    } for d in data]

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    logger.info("미결제약정 데이터 로드 완료: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 3) CoinMetrics Community API — 온체인 지표
# ---------------------------------------------------------------------------

def fetch_onchain_metrics(params: HyperParams) -> pd.DataFrame:
    """CoinMetrics Community API에서 일별 온체인 지표를 가져온다.

    수집 지표:
      - CapMVRVCur: MVRV (Market Value to Realized Value)
      - NVTAdj: NVT (Network Value to Transactions) 조정값
      - AdrActCnt: 활성 주소 수
      - TxCnt: 일일 트랜잭션 수
      - HashRate: 해시레이트

    Returns:
        DataFrame with columns: mvrv, nvt, active_addresses, tx_count, hash_rate
        index: DatetimeIndex (UTC, daily)
    """
    metrics = "CapMVRVCur,NVTAdj,AdrActCnt,TxCnt,HashRate"
    start_date = (dt.datetime.utcnow() - dt.timedelta(days=params.fetch_days + 30)).strftime("%Y-%m-%d")

    url = f"{COINMETRICS_BASE}/timeseries/asset-metrics"
    api_params = {
        "assets": "btc",
        "metrics": metrics,
        "frequency": "1d",
        "start_time": start_date,
        "page_size": 10000,
    }

    try:
        resp = requests.get(url, params=api_params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("CoinMetrics 온체인 데이터 수집 실패: %s", e)
        return pd.DataFrame()

    rows_list = data.get("data", [])
    if not rows_list:
        logger.warning("CoinMetrics 응답에 데이터가 없습니다.")
        return pd.DataFrame()

    rows = []
    for r in rows_list:
        rows.append({
            "timestamp": pd.Timestamp(r.get("time", ""), tz="UTC"),
            "mvrv": _safe_float(r.get("CapMVRVCur")),
            "nvt": _safe_float(r.get("NVTAdj")),
            "active_addresses": _safe_float(r.get("AdrActCnt")),
            "tx_count": _safe_float(r.get("TxCnt")),
            "hash_rate": _safe_float(r.get("HashRate")),
        })

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df.ffill(inplace=True)

    logger.info("온체인 데이터 로드 완료: %d rows, 지표 %s",
                len(df), list(df.columns))
    return df


def _safe_float(val) -> float:
    """None 또는 빈 문자열을 NaN으로 변환."""
    if val is None or val == "":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# ---------------------------------------------------------------------------
# 4) Alternative.me — Fear & Greed Index
# ---------------------------------------------------------------------------

def fetch_fear_greed(days: int = 60) -> pd.DataFrame:
    """Alternative.me에서 Crypto Fear & Greed Index를 가져온다.

    Returns:
        DataFrame with columns: fear_greed
        index: DatetimeIndex (UTC, daily)
    """
    url = FEAR_GREED_BASE
    api_params = {"limit": days, "format": "json"}

    try:
        resp = requests.get(url, params=api_params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("Fear & Greed 데이터 수집 실패: %s", e)
        return pd.DataFrame(columns=["fear_greed"])

    entries = data.get("data", [])
    if not entries:
        return pd.DataFrame(columns=["fear_greed"])

    rows = [{
        "timestamp": pd.Timestamp(int(e["timestamp"]), unit="s", tz="UTC").normalize(),
        "fear_greed": int(e["value"]),
    } for e in entries]

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df.drop_duplicates(inplace=True)

    logger.info("Fear & Greed 데이터 로드 완료: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 5) 데이터 통합
# ---------------------------------------------------------------------------

def merge_auxiliary_data(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    oi: pd.DataFrame,
    onchain: pd.DataFrame,
    fear_greed: pd.DataFrame,
) -> pd.DataFrame:
    """OHLCV 데이터에 보조 지표를 병합한다.

    저빈도 데이터(일별, 8시간별)는 forward-fill로 분봉에 맞춘다.
    """
    df = ohlcv.copy()

    # 펀딩비 병합 (8시간 → 분봉 forward-fill)
    if not funding.empty:
        df = df.join(funding[["funding_rate"]], how="left")
        df["funding_rate"].ffill(inplace=True)
        df["funding_rate"].fillna(0.0, inplace=True)

    # 미결제약정 병합 (5분 → 분봉 forward-fill)
    if not oi.empty:
        df = df.join(oi[["open_interest"]], how="left")
        df["open_interest"].ffill(inplace=True)
        df["open_interest"].bfill(inplace=True)

    # 온체인 데이터 병합 (일별 → 분봉 forward-fill)
    if not onchain.empty:
        # 날짜 기준으로 매칭
        ohlcv_dates = df.index.normalize()
        for col in onchain.columns:
            date_map = onchain[col].to_dict()
            df[col] = ohlcv_dates.map(lambda d: date_map.get(d, np.nan))
            df[col].ffill(inplace=True)
            df[col].bfill(inplace=True)

    # Fear & Greed 병합 (일별 → 분봉 forward-fill)
    if not fear_greed.empty:
        date_map = fear_greed["fear_greed"].to_dict()
        ohlcv_dates = df.index.normalize()
        df["fear_greed"] = ohlcv_dates.map(lambda d: date_map.get(d, np.nan))
        df["fear_greed"].ffill(inplace=True)
        df["fear_greed"].bfill(inplace=True)

    return df


def fetch_all_data(params: HyperParams) -> pd.DataFrame:
    """모든 데이터 소스에서 데이터를 수집하고 통합한다.

    Returns:
        통합된 DataFrame (OHLCV + 보조 지표)
    """
    # 필수: Binance OHLCV
    ohlcv = fetch_binance_ohlcv(params)

    # 선택적: 거래소 지표
    funding = pd.DataFrame(columns=["funding_rate"])
    oi = pd.DataFrame(columns=["open_interest"])
    if params.use_exchange_metrics:
        try:
            funding = fetch_funding_rate(params)
        except Exception as e:
            logger.warning("펀딩비 수집 건너뜀: %s", e)
        try:
            oi = fetch_open_interest_hist(params)
        except Exception as e:
            logger.warning("미결제약정 수집 건너뜀: %s", e)

    # 선택적: 온체인 지표
    onchain = pd.DataFrame()
    if params.use_onchain:
        try:
            onchain = fetch_onchain_metrics(params)
        except Exception as e:
            logger.warning("온체인 데이터 수집 건너뜀: %s", e)

    # 선택적: Fear & Greed
    fear_greed_df = pd.DataFrame(columns=["fear_greed"])
    if params.use_fear_greed:
        try:
            fear_greed_df = fetch_fear_greed(days=params.fetch_days + 30)
        except Exception as e:
            logger.warning("Fear & Greed 수집 건너뜀: %s", e)

    # 통합
    df = merge_auxiliary_data(ohlcv, funding, oi, onchain, fear_greed_df)
    logger.info("데이터 통합 완료: %d rows x %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# 6) 모의 데이터 생성
# ---------------------------------------------------------------------------

def generate_mock_data(minutes: int = 500, params: HyperParams | None = None) -> pd.DataFrame:
    """테스트용 비트코인 모의 데이터를 생성한다.

    실제 BTC 가격 특성을 모사한 OHLCV + 보조 지표를 생성한다.
    """
    rng = np.random.default_rng(42)
    now = dt.datetime.now(tz=dt.timezone.utc).replace(second=0, microsecond=0)

    base_price = 95000.0
    rows = []

    for i in range(minutes):
        ts = now - dt.timedelta(minutes=minutes - i)
        # BTC 특성: 높은 변동성
        change_pct = rng.normal(0, 0.0003)
        base_price *= (1 + change_pct)

        volatility = abs(rng.normal(0, base_price * 0.0002))
        o = base_price + rng.normal(0, base_price * 0.0001)
        h = max(o, base_price) + volatility
        l = min(o, base_price) - volatility
        c = base_price
        v = float(rng.uniform(10, 500))  # BTC volume
        qv = v * c  # quote volume
        trades = int(rng.integers(500, 10000))
        taker_buy = rng.uniform(0.35, 0.65)

        rows.append({
            "timestamp": ts,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": round(v, 4),
            "quote_volume": round(qv, 2),
            "trades": trades,
            "taker_buy_ratio": round(taker_buy, 4),
            # 거래소 보조 지표
            "funding_rate": round(rng.normal(0.0001, 0.0003), 6),
            "open_interest": round(rng.uniform(50000, 80000), 2),
            # 온체인 지표 (일별이므로 하루 단위로 변경)
            "mvrv": round(rng.uniform(1.5, 3.5), 4),
            "nvt": round(rng.uniform(30, 120), 2),
            "active_addresses": float(int(rng.integers(700000, 1100000))),
            "tx_count": float(int(rng.integers(300000, 600000))),
            "hash_rate": round(rng.uniform(500e6, 800e6), 0),
            # 센티먼트
            "fear_greed": float(int(rng.integers(20, 80))),
        })

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    logger.info("모의 데이터 생성 완료: %d rows", len(df))
    return df
