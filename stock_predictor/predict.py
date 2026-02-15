"""
학습된 모델을 사용한 추론 모듈.

사용법:
  predictor = SlopePredictor.from_checkpoint("checkpoints/best_model.pt")
  slope = predictor.predict(latest_ohlcv_df)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import HyperParams
from .feature_engineer import build_features
from .model import create_model


class SlopePredictor:
    """저장된 체크포인트로부터 기울기를 예측하는 추론 클래스."""

    def __init__(
        self,
        model: torch.nn.Module,
        scaler: StandardScaler,
        params: HyperParams,
        device: torch.device | None = None,
    ):
        self.model = model
        self.scaler = scaler
        self.params = params
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | pathlib.Path,
    ) -> SlopePredictor:
        """저장된 체크포인트에서 SlopePredictor를 복원한다."""
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

        params = ckpt["params"]
        num_features = ckpt["num_features"]

        model = create_model(num_features, params)
        model.load_state_dict(ckpt["model_state"])

        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]
        scaler.n_features_in_ = num_features

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, scaler, params, device)

    def predict(self, ohlcv_df: pd.DataFrame) -> float:
        """최근 OHLCV 데이터로부터 기울기를 예측한다.

        Args:
            ohlcv_df: 최소 (n + warmup) 행의 OHLCV DataFrame.
                      columns: open, high, low, close, volume

        Returns:
            예측된 기울기(slope) 값.
            양수 → 하락 기울기 존재 (값이 클수록 급격한 하락)
            0에 가까움 → 하락 기울기 미미
        """
        features = build_features(ohlcv_df, self.params)

        if len(features) < self.params.n:
            raise ValueError(
                f"피처 데이터({len(features)}행)가 윈도우 크기 "
                f"n={self.params.n}보다 작습니다. 더 많은 데이터가 필요합니다."
            )

        # 마지막 n개 피처만 사용
        recent = features.iloc[-self.params.n:]
        arr = recent.values.astype(np.float32)

        # 스케일링
        arr = self.scaler.transform(arr).astype(np.float32)

        # (1, n, F)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            slope = self.model(tensor).item()

        return slope

    def predict_batch(self, ohlcv_df: pd.DataFrame) -> np.ndarray:
        """전체 데이터에 대해 슬라이딩 윈도우로 기울기를 예측한다.

        Returns:
            (num_windows,) 형태의 slope 예측 배열
        """
        features = build_features(ohlcv_df, self.params)

        if len(features) < self.params.n:
            raise ValueError(
                f"피처 데이터({len(features)}행)가 윈도우 크기 "
                f"n={self.params.n}보다 작습니다."
            )

        arr = features.values.astype(np.float32)
        arr = self.scaler.transform(arr).astype(np.float32)

        T, F = arr.shape
        num_windows = T - self.params.n + 1

        # 시퀀스 배치 생성
        sequences = np.empty((num_windows, self.params.n, F), dtype=np.float32)
        for i in range(num_windows):
            sequences[i] = arr[i: i + self.params.n]

        tensor = torch.from_numpy(sequences).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor).cpu().numpy()

        return preds
