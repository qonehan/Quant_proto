"""
실시간 비트코인 기울기 예측 엔진.

기능:
  - Binance REST API에서 최신 캔들 폴링
  - 학습된 모델로 매 캔들마다 기울기 예측
  - 예측값과 실제 결과 비교 (x% 변동 발생 시 해결)
  - 예측 이력 및 정확도 통계 관리
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

from .config import HyperParams
from .predict import SlopePredictor

logger = logging.getLogger(__name__)

BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"


# ---------------------------------------------------------------------------
# 예측 기록
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """개별 예측 기록."""
    timestamp: datetime
    price: float
    predicted_slope: float

    # 해결 후 채워지는 필드
    actual_slope: float | None = None
    resolution_time: datetime | None = None
    resolution_minutes: int | None = None
    resolved: bool = False

    @property
    def predicted_direction(self) -> str:
        if self.predicted_slope > 0.001:
            return "상승"
        elif self.predicted_slope < -0.001:
            return "하락"
        return "횡보"

    @property
    def actual_direction(self) -> str | None:
        if self.actual_slope is None:
            return None
        if self.actual_slope > 0.001:
            return "상승"
        elif self.actual_slope < -0.001:
            return "하락"
        return "횡보"

    @property
    def is_correct(self) -> bool | None:
        if not self.resolved:
            return None
        return np.sign(round(self.predicted_slope, 3)) == np.sign(round(self.actual_slope, 3))


# ---------------------------------------------------------------------------
# 실시간 엔진
# ---------------------------------------------------------------------------

class LiveEngine:
    """실시간 데이터 수집, 예측, 비교를 관리하는 엔진."""

    def __init__(self, predictor: SlopePredictor):
        self.predictor = predictor
        self.params = predictor.params

        self.candle_buffer: pd.DataFrame = pd.DataFrame()
        self.auxiliary_data: dict[str, float] = {}
        self.predictions: list[PredictionRecord] = []
        self.last_fetch_time: datetime | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """초기 데이터를 로드한다. 성공 시 True."""
        try:
            candles = self._fetch_candles(limit=300)
            if candles.empty:
                return False
            self.candle_buffer = candles

            # 보조 지표 수집 (한 번만)
            self._fetch_auxiliary()

            # 보조 지표를 캔들 버퍼에 병합
            self._merge_auxiliary()

            self._initialized = True
            self.last_fetch_time = datetime.now(tz=timezone.utc)
            logger.info("LiveEngine 초기화 완료: %d candles", len(self.candle_buffer))
            return True
        except Exception as e:
            logger.error("LiveEngine 초기화 실패: %s", e)
            return False

    # ------------------------------------------------------------------
    # 업데이트 사이클
    # ------------------------------------------------------------------

    def update(self) -> dict | None:
        """최신 캔들을 가져오고 예측을 실행한다.

        Returns:
            새 예측 정보 dict 또는 None (업데이트 없음)
        """
        if not self._initialized:
            if not self.initialize():
                return None

        # 최신 캔들 가져오기
        new_candles = self._fetch_candles(limit=5)
        if new_candles.empty:
            return None

        # 새 캔들만 추가
        new_mask = ~new_candles.index.isin(self.candle_buffer.index)
        if new_mask.any():
            new_rows = new_candles[new_mask].copy()
            # 보조 지표 컬럼 추가
            for col, val in self.auxiliary_data.items():
                if col not in new_rows.columns:
                    new_rows[col] = val
            self.candle_buffer = pd.concat([self.candle_buffer, new_rows])
        else:
            # 마지막 캔들 업데이트 (현재 진행 중인 캔들)
            last_idx = new_candles.index[-1]
            if last_idx in self.candle_buffer.index:
                for col in ["open", "high", "low", "close", "volume",
                            "quote_volume", "trades", "taker_buy_ratio"]:
                    if col in new_candles.columns and col in self.candle_buffer.columns:
                        self.candle_buffer.loc[last_idx, col] = new_candles.loc[last_idx, col]

        # 버퍼 크기 제한 (최근 500개)
        if len(self.candle_buffer) > 500:
            self.candle_buffer = self.candle_buffer.tail(500)

        # 미해결 예측 확인
        self._check_resolutions()

        # 새 예측 실행
        prediction = self._run_prediction()

        self.last_fetch_time = datetime.now(tz=timezone.utc)
        return prediction

    # ------------------------------------------------------------------
    # 데이터 수집
    # ------------------------------------------------------------------

    def _fetch_candles(self, limit: int = 200) -> pd.DataFrame:
        """Binance에서 최근 캔들을 가져온다."""
        url = f"{BINANCE_SPOT_BASE}/api/v3/klines"
        api_params = {
            "symbol": self.params.symbol,
            "interval": self.params.binance_interval,
            "limit": limit,
        }

        try:
            resp = requests.get(url, params=api_params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("캔들 데이터 수집 실패: %s", e)
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        rows = []
        for c in data:
            total_vol = float(c[5])
            taker_buy_vol = float(c[9])
            rows.append({
                "timestamp": pd.Timestamp(c[0], unit="ms", tz="UTC"),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": total_vol,
                "quote_volume": float(c[7]),
                "trades": int(c[8]),
                "taker_buy_ratio": taker_buy_vol / total_vol if total_vol > 0 else 0.5,
            })

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    def _fetch_auxiliary(self):
        """보조 지표를 수집한다 (펀딩비, 미결제약정 등)."""
        # 펀딩비
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex",
                params={"symbol": self.params.symbol}, timeout=10
            )
            if resp.ok:
                data = resp.json()
                self.auxiliary_data["funding_rate"] = float(data.get("lastFundingRate", 0))
        except Exception:
            self.auxiliary_data["funding_rate"] = 0.0

        # 미결제약정
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest",
                params={"symbol": self.params.symbol}, timeout=10
            )
            if resp.ok:
                data = resp.json()
                self.auxiliary_data["open_interest"] = float(data.get("openInterest", 0))
        except Exception:
            self.auxiliary_data["open_interest"] = 0.0

        # Fear & Greed
        try:
            resp = requests.get(
                "https://api.alternative.me/fng",
                params={"limit": 1}, timeout=10
            )
            if resp.ok:
                data = resp.json()
                entries = data.get("data", [])
                if entries:
                    self.auxiliary_data["fear_greed"] = float(entries[0]["value"])
        except Exception:
            self.auxiliary_data["fear_greed"] = 50.0

    def _merge_auxiliary(self):
        """보조 지표를 캔들 버퍼에 병합한다."""
        for col, val in self.auxiliary_data.items():
            if col not in self.candle_buffer.columns:
                self.candle_buffer[col] = val

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    def _run_prediction(self) -> dict | None:
        """현재 데이터로 기울기를 예측한다."""
        if len(self.candle_buffer) < self.params.n + 30:
            logger.warning("예측에 필요한 데이터 부족: %d rows", len(self.candle_buffer))
            return None

        try:
            slope = self.predictor.predict(self.candle_buffer)
            current_price = float(self.candle_buffer["close"].iloc[-1])
            ts = self.candle_buffer.index[-1].to_pydatetime()

            record = PredictionRecord(
                timestamp=ts,
                price=current_price,
                predicted_slope=slope,
            )
            self.predictions.append(record)

            return {
                "timestamp": ts,
                "price": current_price,
                "slope": slope,
                "direction": record.predicted_direction,
            }
        except Exception as e:
            logger.warning("예측 실패: %s", e)
            return None

    # ------------------------------------------------------------------
    # 예측 해결 (실제값 비교)
    # ------------------------------------------------------------------

    def _check_resolutions(self):
        """미해결 예측들의 실제 결과를 확인한다."""
        if self.candle_buffer.empty:
            return

        close_series = self.candle_buffer["close"]
        x = self.params.x
        max_w = self.params.slope_max_window

        for pred in self.predictions:
            if pred.resolved:
                continue

            # 예측 시점 이후의 캔들들
            future = close_series[close_series.index > pred.timestamp]
            if future.empty:
                continue

            base_price = pred.price
            up_threshold = base_price * (1 + x / 100.0)
            down_threshold = base_price * (1 - x / 100.0)

            for i, (ts, price) in enumerate(future.items()):
                minutes = i + 1

                if price >= up_threshold:
                    pred.actual_slope = x / minutes
                    pred.resolved = True
                    pred.resolution_time = ts.to_pydatetime()
                    pred.resolution_minutes = minutes
                    break
                elif price <= down_threshold:
                    pred.actual_slope = -(x / minutes)
                    pred.resolved = True
                    pred.resolution_time = ts.to_pydatetime()
                    pred.resolution_minutes = minutes
                    break
                elif minutes >= max_w:
                    pred.actual_slope = 0.0
                    pred.resolved = True
                    pred.resolution_time = ts.to_pydatetime()
                    pred.resolution_minutes = minutes
                    break

    # ------------------------------------------------------------------
    # 통계
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """예측 정확도 통계를 반환한다."""
        resolved = [p for p in self.predictions if p.resolved]
        if not resolved:
            return {
                "total_predictions": len(self.predictions),
                "resolved": 0,
                "pending": len(self.predictions),
                "accuracy": None,
                "up_accuracy": None,
                "down_accuracy": None,
                "flat_accuracy": None,
            }

        correct = sum(1 for p in resolved if p.is_correct)
        total = len(resolved)

        # 방향별 통계
        up_preds = [p for p in resolved if p.predicted_direction == "상승"]
        down_preds = [p for p in resolved if p.predicted_direction == "하락"]
        flat_preds = [p for p in resolved if p.predicted_direction == "횡보"]

        def _dir_accuracy(preds):
            if not preds:
                return None
            return sum(1 for p in preds if p.is_correct) / len(preds)

        return {
            "total_predictions": len(self.predictions),
            "resolved": total,
            "pending": len(self.predictions) - total,
            "accuracy": correct / total if total > 0 else None,
            "correct": correct,
            "wrong": total - correct,
            "up_accuracy": _dir_accuracy(up_preds),
            "up_count": len(up_preds),
            "down_accuracy": _dir_accuracy(down_preds),
            "down_count": len(down_preds),
            "flat_accuracy": _dir_accuracy(flat_preds),
            "flat_count": len(flat_preds),
            "avg_resolution_minutes": np.mean(
                [p.resolution_minutes for p in resolved if p.resolution_minutes]
            ) if resolved else None,
        }

    def get_predictions_df(self) -> pd.DataFrame:
        """예측 이력을 DataFrame으로 반환한다."""
        if not self.predictions:
            return pd.DataFrame()

        rows = []
        for p in reversed(self.predictions):  # 최신 순
            rows.append({
                "시각": p.timestamp.strftime("%H:%M"),
                "가격": f"${p.price:,.0f}",
                "예측 기울기": f"{p.predicted_slope:+.4f}",
                "예측 방향": p.predicted_direction,
                "실제 기울기": f"{p.actual_slope:+.4f}" if p.actual_slope is not None else "⏳ 대기중",
                "실제 방향": p.actual_direction or "⏳",
                "소요(분)": str(p.resolution_minutes) if p.resolution_minutes else "-",
                "결과": "✅" if p.is_correct else ("❌" if p.is_correct is False else "⏳"),
            })

        return pd.DataFrame(rows)

    def get_candle_df(self) -> pd.DataFrame:
        """차트용 캔들 데이터를 반환한다."""
        if self.candle_buffer.empty:
            return pd.DataFrame()
        return self.candle_buffer[["open", "high", "low", "close", "volume"]].copy()

    def get_slope_series(self) -> pd.DataFrame:
        """예측/실제 기울기 시계열을 반환한다."""
        if not self.predictions:
            return pd.DataFrame()

        rows = []
        for p in self.predictions:
            rows.append({
                "timestamp": p.timestamp,
                "predicted": p.predicted_slope,
                "actual": p.actual_slope if p.resolved else None,
            })
        return pd.DataFrame(rows).set_index("timestamp")


# ---------------------------------------------------------------------------
# 데모 모드 (모의 데이터)
# ---------------------------------------------------------------------------

class DemoLiveEngine(LiveEngine):
    """모의 데이터로 동작하는 데모 엔진."""

    def __init__(self, predictor: SlopePredictor):
        super().__init__(predictor)
        self._demo_step = 0

    def initialize(self) -> bool:
        from .data_fetcher import generate_mock_data
        self.candle_buffer = generate_mock_data(300, self.params)
        self._initialized = True
        self.last_fetch_time = datetime.now(tz=timezone.utc)
        logger.info("Demo 엔진 초기화 완료: %d candles", len(self.candle_buffer))
        return True

    def update(self) -> dict | None:
        if not self._initialized:
            self.initialize()

        # 모의 신규 캔들 추가
        rng = np.random.default_rng(int(time.time()))
        last = self.candle_buffer.iloc[-1]
        ts = self.candle_buffer.index[-1] + timedelta(minutes=1)

        change = rng.normal(0, last["close"] * 0.0003)
        c = last["close"] + change
        h = max(c, last["close"]) + abs(rng.normal(0, last["close"] * 0.0002))
        l = min(c, last["close"]) - abs(rng.normal(0, last["close"] * 0.0002))
        v = rng.uniform(10, 500)

        new_row = pd.DataFrame([{
            "open": last["close"],
            "high": h, "low": l, "close": c,
            "volume": v,
            "quote_volume": v * c,
            "trades": int(rng.integers(500, 10000)),
            "taker_buy_ratio": rng.uniform(0.35, 0.65),
            "funding_rate": last.get("funding_rate", 0.0001),
            "open_interest": last.get("open_interest", 60000),
            "mvrv": last.get("mvrv", 2.5),
            "nvt": last.get("nvt", 60),
            "active_addresses": last.get("active_addresses", 900000),
            "tx_count": last.get("tx_count", 400000),
            "hash_rate": last.get("hash_rate", 600e6),
            "fear_greed": last.get("fear_greed", 55),
        }], index=[ts])

        self.candle_buffer = pd.concat([self.candle_buffer, new_row]).tail(500)

        self._check_resolutions()
        return self._run_prediction()
