"""
삼성전자 주가 기울기 예측 모델 — 메인 진입점.

사용법:
  # 학습 (yfinance 과거 데이터)
  python -m stock_predictor.main train

  # 학습 (모의 실시간 데이터)
  python -m stock_predictor.main train --mock --mock-minutes 300

  # 추론 (최신 데이터로 현재 기울기 예측)
  python -m stock_predictor.main predict

  # 하이퍼파라미터 변경
  python -m stock_predictor.main train --n 30 --x 0.5 --epochs 200
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import pandas as pd

from .config import HyperParams
from .data_fetcher import RealtimeCollector, fetch_historical
from .predict import SlopePredictor
from .train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="삼성전자 주가 기울기(slope) 예측 모델"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    tr = sub.add_parser("train", help="모델 학습")
    tr.add_argument("--n", type=int, default=60,
                     help="입력 시퀀스 길이 (분)")
    tr.add_argument("--x", type=float, default=1.0,
                     help="목표 하락률 (%%)")
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--layers", type=int, default=2)
    tr.add_argument("--save-dir", type=str, default="checkpoints")
    tr.add_argument("--mock", action="store_true",
                     help="모의 데이터로 학습")
    tr.add_argument("--mock-minutes", type=int, default=500,
                     help="모의 데이터 길이 (분)")

    # --- predict ---
    pr = sub.add_parser("predict", help="기울기 예측")
    pr.add_argument("--checkpoint", type=str,
                     default="checkpoints/best_model.pt")
    pr.add_argument("--n", type=int, default=60)

    return parser.parse_args()


def cmd_train(args):
    params = HyperParams(
        n=args.n,
        x=args.x,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden,
        num_layers=args.layers,
    )

    data_df = None

    if args.mock:
        logger.info("모의 데이터 생성 중 (%d분)...", args.mock_minutes)
        collector = RealtimeCollector(
            params, api_key="", api_secret="", account_no="", mock=True
        )
        asyncio.run(collector._run_mock(args.mock_minutes))
        data_df = collector.to_dataframe()
        logger.info("모의 데이터 생성 완료: %d rows", len(data_df))

    result = train(params, save_dir=args.save_dir, data_df=data_df)

    print("\n" + "=" * 60)
    print("학습 완료")
    print(f"  모델 저장: {result['best_model_path']}")
    if result["metrics"]:
        m = result["metrics"]
        print(f"  MSE:  {m['mse']:.6f}")
        print(f"  MAE:  {m['mae']:.6f}")
        print(f"  방향 정확도: {m['direction_accuracy']:.2%}")
    print("=" * 60)


def cmd_predict(args):
    predictor = SlopePredictor.from_checkpoint(args.checkpoint)
    params = predictor.params

    logger.info("최신 데이터 로드 중...")
    df = fetch_historical(params)

    slope = predictor.predict(df)

    print("\n" + "=" * 60)
    print(f"삼성전자 ({params.ticker}) 기울기 예측")
    print(f"  설정: n={params.n}분, x={params.x}%")
    print(f"  예측 기울기: {slope:.4f}")
    if slope > 0:
        estimated_minutes = params.x / slope
        print(f"  해석: {params.x}% 하락에 약 {estimated_minutes:.1f}분 소요 예상")
    else:
        print(f"  해석: {params.x}% 하락 가능성 낮음")
    print("=" * 60)


def main():
    args = parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
