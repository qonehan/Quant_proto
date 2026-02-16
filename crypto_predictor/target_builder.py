"""
양방향 기울기(slope) 타겟 변수 생성 모듈.

기울기 정의:
  현재 시점 t의 종가를 P(t)라 할 때,
  미래 시점 t+1, t+2, ... 를 순회하며 가격이 x% 이상 상승 또는 하락한
  첫 번째 시점 t+k를 찾는다.

  상승 (양의 기울기):
    P(t+k) >= P(t) * (1 + x/100) 일 때 slope = +x / k
  하락 (음의 기울기):
    P(t+k) <= P(t) * (1 - x/100) 일 때 slope = -x / k

  상승과 하락 중 먼저 발생한 방향을 채택한다.
  slope_max_window 내에서 어느 방향으로도 x% 변동이 없으면 slope = 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import HyperParams


def compute_slopes(
    close_prices: pd.Series,
    params: HyperParams,
) -> np.ndarray:
    """각 시점에 대해 양방향 기울기(slope) 값을 계산한다.

    Args:
        close_prices: 1분봉 종가 시계열
        params: x (목표 변동률 %), slope_max_window (최대 관측 분)

    Returns:
        (T,) 형태의 slope 배열.
    """
    prices = close_prices.values.astype(np.float64)
    T = len(prices)
    x = params.x
    max_w = params.slope_max_window

    slopes = np.full(T, np.nan, dtype=np.float64)

    for t in range(T - 1):
        p0 = prices[t]
        if p0 == 0:
            continue

        up_threshold = p0 * (1 + x / 100.0)
        down_threshold = p0 * (1 - x / 100.0)

        end = min(t + max_w + 1, T)

        found = False
        for k_idx in range(t + 1, end):
            pk = prices[k_idx]
            k = k_idx - t

            if pk >= up_threshold:
                slopes[t] = x / k
                found = True
                break
            elif pk <= down_threshold:
                slopes[t] = -(x / k)
                found = True
                break

        if not found:
            slopes[t] = 0.0

    return slopes


def build_targets_from_raw(
    df: pd.DataFrame,
    features: pd.DataFrame,
    params: HyperParams,
) -> np.ndarray:
    """원본 DataFrame과 피처 DataFrame에서 타겟 배열을 생성한다.

    Args:
        df: 원본 OHLCV DataFrame (close 컬럼 필요)
        features: build_features 결과
        params: 하이퍼파라미터

    Returns:
        (num_samples,) 형태의 slope 타겟 배열
    """
    all_slopes = compute_slopes(df["close"], params)

    feat_positions = [df.index.get_loc(idx) for idx in features.index]
    feat_slopes = np.array([all_slopes[p] for p in feat_positions], dtype=np.float32)

    n = params.n
    num_samples = len(features) - n + 1
    targets = feat_slopes[n - 1: n - 1 + num_samples]

    targets = np.nan_to_num(targets, nan=0.0)

    return targets
