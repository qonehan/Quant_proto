"""
삼성전자 1분봉 주가 데이터 수집 모듈.

두 가지 소스를 지원:
1. yfinance  — 과거 1분봉 데이터 (최대 7일, 학습용)
2. 한국투자증권 Open API (mojito) — 실시간 체결가 웹소켓 수신
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import HyperParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) yfinance 기반 과거 데이터 수집
# ---------------------------------------------------------------------------

def fetch_historical(params: HyperParams) -> pd.DataFrame:
    """yfinance로 삼성전자 1분봉 과거 데이터를 가져온다.

    Returns:
        DataFrame with columns: open, high, low, close, volume
        index: DatetimeIndex (KST)
    """
    import yfinance as yf
    ticker = yf.Ticker(params.ticker_yf)
    df = ticker.history(period=params.yf_period, interval=params.yf_interval)

    if df.empty:
        raise RuntimeError(
            f"yfinance에서 {params.ticker_yf} 데이터를 가져올 수 없습니다. "
            "장 운영 시간을 확인하세요."
        )

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # 결측치 처리: 전방 채움 후 남은 결측은 후방 채움
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    logger.info("yfinance 데이터 로드 완료: %d rows, 기간 %s ~ %s",
                len(df), df.index[0], df.index[-1])
    return df


# ---------------------------------------------------------------------------
# 2) 한국투자증권 Open API (mojito) 실시간 체결가 수집
# ---------------------------------------------------------------------------

class RealtimeCollector:
    """실시간 체결가를 1분봉으로 집계하여 DataFrame 형태로 누적한다.

    사용법:
        collector = RealtimeCollector(params, api_key, api_secret, account_no)
        await collector.run(duration_minutes=120)
        df = collector.to_dataframe()

    한국투자증권 Open API 키가 필요합니다.
    발급: https://apiportal.koreainvestment.com/
    """

    def __init__(
        self,
        params: HyperParams,
        api_key: str,
        api_secret: str,
        account_no: str,
        mock: bool = False,
    ):
        self.params = params
        self.api_key = api_key
        self.api_secret = api_secret
        self.account_no = account_no
        self.mock = mock

        # 1분봉 집계를 위한 임시 버퍼
        self._tick_buffer: list[dict] = []
        # 완성된 1분봉 저장
        self._bars: list[dict] = []

    # ------------------------------------------------------------------

    async def run(self, duration_minutes: int = 60) -> None:
        """실시간 데이터를 duration_minutes 동안 수집한다."""
        if self.mock:
            await self._run_mock(duration_minutes)
            return

        try:
            import mojito
        except ImportError:
            raise ImportError(
                "mojito2 패키지가 필요합니다: pip install mojito2"
            )

        broker = mojito.KoreaInvestment(
            api_key=self.api_key,
            api_secret=self.api_secret,
            acc_no=self.account_no,
            mock=True,  # 모의투자 환경
        )
        ws = broker.create_websocket()

        end_time = dt.datetime.now() + dt.timedelta(minutes=duration_minutes)
        current_minute = None

        logger.info("실시간 수집 시작: %d분간 진행", duration_minutes)

        async def _on_tick(price: float, volume: int, timestamp: dt.datetime):
            nonlocal current_minute
            minute_key = timestamp.replace(second=0, microsecond=0)

            if current_minute is None:
                current_minute = minute_key

            if minute_key != current_minute:
                self._flush_minute(current_minute)
                current_minute = minute_key

            self._tick_buffer.append({
                "price": price,
                "volume": volume,
                "timestamp": timestamp,
            })

        # 웹소켓 체결가 수신
        ws.subscribe(self.params.ticker, on_tick=_on_tick)

        try:
            while dt.datetime.now() < end_time:
                await asyncio.sleep(1)
        finally:
            ws.close()
            # 마지막 버퍼 플러시
            if self._tick_buffer and current_minute:
                self._flush_minute(current_minute)

        logger.info("실시간 수집 완료: %d개 분봉 생성", len(self._bars))

    async def _run_mock(self, duration_minutes: int) -> None:
        """테스트용 모의 데이터 생성."""
        rng = np.random.default_rng(42)
        base_price = 72000.0
        now = dt.datetime.now().replace(second=0, microsecond=0)

        for i in range(duration_minutes):
            ts = now + dt.timedelta(minutes=i)
            change = rng.normal(0, 100)
            o = base_price + change
            h = o + abs(rng.normal(0, 50))
            l = o - abs(rng.normal(0, 50))
            c = o + rng.normal(0, 30)
            v = int(rng.integers(1000, 50000))
            base_price = c

            self._bars.append({
                "timestamp": ts,
                "open": round(o),
                "high": round(h),
                "low": round(l),
                "close": round(c),
                "volume": v,
            })

    def _flush_minute(self, minute_key: dt.datetime) -> None:
        """틱 버퍼를 1분봉 하나로 변환하여 _bars에 추가."""
        if not self._tick_buffer:
            return

        prices = [t["price"] for t in self._tick_buffer]
        volumes = [t["volume"] for t in self._tick_buffer]

        self._bars.append({
            "timestamp": minute_key,
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(volumes),
        })
        self._tick_buffer.clear()

    def to_dataframe(self) -> pd.DataFrame:
        if not self._bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(self._bars)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
