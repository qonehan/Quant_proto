"""
학습 및 평가 파이프라인.

흐름:
  1. 데이터 로드 (yfinance 또는 실시간 수집)
  2. 피처 엔지니어링
  3. 타겟(slope) 생성
  4. Train / Val / Test 분할
  5. LSTM 모델 학습 (Early Stopping)
  6. 평가 및 모델 저장
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .data_fetcher import fetch_historical
from .feature_engineer import build_features, build_sequence_matrix
from .model import create_model
from .target_builder import build_targets_from_raw

if TYPE_CHECKING:
    from .config import HyperParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset 구성
# ---------------------------------------------------------------------------

def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
):
    """시계열 순서를 유지하며 train/val/test로 분할."""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        (X[:train_end], y[:train_end]),
        (X[train_end:val_end], y[train_end:val_end]),
        (X[val_end:], y[val_end:]),
    )


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train(
    params: HyperParams,
    save_dir: str | pathlib.Path = "checkpoints",
    data_df=None,
) -> dict:
    """전체 학습 파이프라인을 실행한다.

    Args:
        params: 하이퍼파라미터
        save_dir: 모델 체크포인트 저장 디렉토리
        data_df: 외부 데이터프레임 (None이면 yfinance에서 로드)

    Returns:
        {"model": trained model, "scaler": fitted scaler, "metrics": dict}
    """
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("학습 디바이스: %s", device)

    # 1) 데이터 로드
    if data_df is None:
        df = fetch_historical(params)
    else:
        df = data_df
    logger.info("원본 데이터: %d rows", len(df))

    # 2) 피처 엔지니어링
    features = build_features(df, params)
    logger.info("피처 생성 완료: %d rows x %d features", *features.shape)

    # 3) 타겟 생성
    targets = build_targets_from_raw(df, features, params)
    logger.info("타겟 생성 완료: %d samples", len(targets))

    # 4) 시퀀스 행렬 생성
    X = build_sequence_matrix(features, params.n)
    y = targets
    logger.info("시퀀스 행렬: %s, 타겟: %s", X.shape, y.shape)

    assert len(X) == len(y), (
        f"X({len(X)})와 y({len(y)}) 길이가 다릅니다."
    )

    # 5) 정규화 (피처 차원별 StandardScaler)
    num_samples, seq_len, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat).astype(np.float32)
    X = X_flat.reshape(num_samples, seq_len, num_features)

    # 6) Train/Val/Test 분할
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = _split_data(
        X, y, params.train_ratio, params.val_ratio
    )
    logger.info("분할: train=%d, val=%d, test=%d",
                len(X_train), len(X_val), len(X_test))

    train_loader = _make_loader(X_train, y_train, params.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, params.batch_size, shuffle=False)
    test_loader = _make_loader(X_test, y_test, params.batch_size, shuffle=False)

    # 7) 모델 생성
    model = create_model(num_features, params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    logger.info("모델 파라미터 수: %d",
                sum(p.numel() for p in model.parameters()))

    # 8) 학습 루프
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = save_dir / "best_model.pt"

    for epoch in range(1, params.epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else float("inf")
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
                epoch, params.epochs, train_loss, val_loss,
                optimizer.param_groups[0]["lr"],
            )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "num_features": num_features,
                "params": params,
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= params.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # 9) 최적 모델로 테스트 평가
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_preds = []
    test_trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            test_preds.append(pred.cpu().numpy())
            test_trues.append(yb.numpy())

    test_preds = np.concatenate(test_preds) if test_preds else np.array([])
    test_trues = np.concatenate(test_trues) if test_trues else np.array([])

    metrics = {}
    if len(test_preds) > 0:
        mse = float(np.mean((test_preds - test_trues) ** 2))
        mae = float(np.mean(np.abs(test_preds - test_trues)))
        # 방향 정확도: slope > 0 여부가 일치하는 비율
        dir_acc = float(np.mean(
            (test_preds > 0).astype(int) == (test_trues > 0).astype(int)
        ))
        metrics = {"mse": mse, "mae": mae, "direction_accuracy": dir_acc}
        logger.info("테스트 결과 — MSE: %.6f, MAE: %.6f, 방향정확도: %.2f%%",
                     mse, mae, dir_acc * 100)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "best_model_path": str(best_model_path),
    }
