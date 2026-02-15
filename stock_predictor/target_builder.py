"""
기울기(slope) 타겟 변수 생성 모듈.

기울기 정의:
  현재 시점 t의 종가를 P(t)라 할 때,
  미래 시점 t+1, t+2, ... 를 순회하며 가격이 x% 이상 하락한
  첫 번째 시점 t+k를 찾는다.

  slope = x / k   (k = 하락에 걸린 분 수)

  만약 slope_max_window 내에서 x% 하락이 발생하지 않으면:
    slope = 0  (하락하지 않음 → 기울기 없음)

  slope 값이 클수록 급격한 하락을 의미한다.
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
    """각 시점에 대해 기울기(slope) 값을 계산한다.

    Args:
        close_prices: 1분봉 종가 시계열
        params: x (목표 하락률 %), slope_max_window (최대 관측 분)

    Returns:
        (T,) 형태의 slope 배열.
        마지막 slope_max_window 개의 값은 미래 데이터가 부족하므로 NaN.
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

        threshold = p0 * (1 - x / 100.0)
        found = False

        end = min(t + max_w + 1, T)
        for k_idx in range(t + 1, end):
            if prices[k_idx] <= threshold:
                k = k_idx - t  # 걸린 분 수
                slopes[t] = x / k
                found = True
                break

        if not found:
            slopes[t] = 0.0

    return slopes


def align_targets_with_features(
    slopes: np.ndarray,
    features: pd.DataFrame,
    n: int,
) -> np.ndarray:
    """피처 행렬의 각 윈도우에 대응하는 타겟(slope) 배열을 정렬한다.

    build_sequence_matrix로 생성된 행렬의 i번째 샘플은
    features[i : i+n] 구간에 해당하며,
    해당 윈도우의 마지막 시점(i + n - 1)에서의 slope를 타겟으로 사용한다.

    Args:
        slopes: compute_slopes로 계산된 전체 slope 배열
        features: NaN이 제거된 피처 DataFrame
        n: lookback window 크기

    Returns:
        (num_samples,) 형태의 타겟 배열
    """
    # features에서 NaN이 제거되었으므로 원본 인덱스를 기준으로 slope를 매핑
    aligned_slopes = slopes[features.index.map(
        lambda idx: features.index.get_loc(idx)
    )]

    # 시퀀스의 마지막 시점 기준으로 타겟 추출
    num_samples = len(features) - n + 1
    targets = np.empty(num_samples, dtype=np.float32)

    for i in range(num_samples):
        target_idx = i + n - 1
        targets[i] = aligned_slopes[target_idx]

    # NaN 타겟을 0으로 대체 (윈도우 끝 부분)
    targets = np.nan_to_num(targets, nan=0.0)

    return targets


def build_targets_from_raw(
    df: pd.DataFrame,
    features: pd.DataFrame,
    params: HyperParams,
) -> np.ndarray:
    """원본 OHLCV DataFrame과 피처 DataFrame에서 타겟 배열을 생성한다.

    features의 인덱스가 df 인덱스의 부분집합이어야 한다.

    Args:
        df: 원본 OHLCV DataFrame (close 컬럼 필요)
        features: build_features 결과
        params: 하이퍼파라미터

    Returns:
        (num_samples,) 형태의 slope 타겟 배열
    """
    # 전체 종가에 대해 slope 계산
    all_slopes = compute_slopes(df["close"], params)

    # features 인덱스에 해당하는 slope 추출
    feat_positions = [df.index.get_loc(idx) for idx in features.index]
    feat_slopes = np.array([all_slopes[p] for p in feat_positions], dtype=np.float32)

    # 시퀀스 윈도우에 맞춰 타겟 정렬 (마지막 시점 기준)
    n = params.n
    num_samples = len(features) - n + 1
    targets = feat_slopes[n - 1: n - 1 + num_samples]

    targets = np.nan_to_num(targets, nan=0.0)

    return targets
