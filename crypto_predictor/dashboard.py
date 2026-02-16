"""
비트코인 실시간 기울기 예측 대시보드.

실행:
  streamlit run crypto_predictor/dashboard.py

  # 데모 모드 (모의 데이터, 학습된 모델 불필요)
  streamlit run crypto_predictor/dashboard.py -- --demo

  # 체크포인트 지정
  streamlit run crypto_predictor/dashboard.py -- --checkpoint checkpoints_crypto/best_model.pt
"""

from __future__ import annotations

import sys
import pathlib
import logging

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BTC Slope Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 스타일
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background-color: #0e1117;
    }

    /* 메트릭 카드 */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #151926 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #8b92a5 !important;
        font-size: 13px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* 정확도 박스 */
    .stat-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #151926 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        text-align: center;
    }
    .stat-box .label {
        color: #8b92a5;
        font-size: 13px;
        margin-bottom: 4px;
    }
    .stat-box .value {
        font-size: 32px;
        font-weight: 700;
    }
    .stat-box .sub {
        color: #8b92a5;
        font-size: 12px;
    }
    .green { color: #00d26a; }
    .red { color: #f23645; }
    .yellow { color: #f7931a; }
    .blue { color: #4da6ff; }
    .gray { color: #8b92a5; }

    /* 테이블 스타일 */
    .prediction-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    .prediction-table th {
        background: #1a1f2e;
        color: #8b92a5;
        padding: 10px 12px;
        text-align: center;
        border-bottom: 2px solid #2d3348;
        font-weight: 600;
    }
    .prediction-table td {
        padding: 8px 12px;
        text-align: center;
        border-bottom: 1px solid #1e2235;
        color: #e1e5ee;
    }
    .prediction-table tr:hover {
        background: #1a1f2e;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-up { background: rgba(0, 210, 106, 0.15); color: #00d26a; }
    .badge-down { background: rgba(242, 54, 69, 0.15); color: #f23645; }
    .badge-flat { background: rgba(139, 146, 165, 0.15); color: #8b92a5; }
    .badge-pending { background: rgba(247, 147, 26, 0.15); color: #f7931a; }
    .badge-correct { background: rgba(0, 210, 106, 0.15); color: #00d26a; }
    .badge-wrong { background: rgba(242, 54, 69, 0.15); color: #f23645; }

    /* 제목 */
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #f7931a;
        margin-bottom: 0;
    }
    .subtitle {
        color: #8b92a5;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------

def get_args():
    """CLI 인자를 파싱한다."""
    args = {"demo": False, "checkpoint": "checkpoints_crypto/best_model.pt"}
    argv = sys.argv[1:]
    # streamlit은 -- 이후의 인자를 스크립트에 전달
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    for i, arg in enumerate(argv):
        if arg == "--demo":
            args["demo"] = True
        elif arg == "--checkpoint" and i + 1 < len(argv):
            args["checkpoint"] = argv[i + 1]
    return args


# ---------------------------------------------------------------------------
# 엔진 초기화
# ---------------------------------------------------------------------------

def init_engine():
    """LiveEngine을 초기화하여 session_state에 저장한다."""
    if "engine" in st.session_state and st.session_state.engine is not None:
        return

    args = get_args()

    # 프로젝트 루트를 path에 추가
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    from crypto_predictor.config import HyperParams
    from crypto_predictor.predict import SlopePredictor
    from crypto_predictor.live_engine import LiveEngine, DemoLiveEngine

    checkpoint_path = pathlib.Path(args["checkpoint"])

    if args["demo"] or not checkpoint_path.exists():
        if not checkpoint_path.exists():
            st.session_state.mode = "demo"
            st.sidebar.warning("학습된 모델이 없어 데모 모드로 실행합니다.")
        else:
            st.session_state.mode = "demo"

        # 데모 모드: 모의 모델 학습
        from crypto_predictor.data_fetcher import generate_mock_data
        from crypto_predictor.train import train as train_model

        params = HyperParams(n=60, x=1.0, epochs=5)
        mock_data = generate_mock_data(500, params)
        result = train_model(params, save_dir="checkpoints_crypto", data_df=mock_data)

        predictor = SlopePredictor.from_checkpoint(result["best_model_path"])
        engine = DemoLiveEngine(predictor)
    else:
        st.session_state.mode = "live"
        predictor = SlopePredictor.from_checkpoint(str(checkpoint_path))
        engine = LiveEngine(predictor)

    engine.initialize()
    st.session_state.engine = engine


# ---------------------------------------------------------------------------
# 차트 생성
# ---------------------------------------------------------------------------

def create_price_chart(candles: pd.DataFrame, predictions: list) -> go.Figure:
    """캔들스틱 + 예측 마커 차트를 생성한다."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=None,
    )

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=candles.index,
        open=candles["open"],
        high=candles["high"],
        low=candles["low"],
        close=candles["close"],
        name="BTC",
        increasing_line_color="#00d26a",
        decreasing_line_color="#f23645",
        increasing_fillcolor="#00d26a",
        decreasing_fillcolor="#f23645",
    ), row=1, col=1)

    # 예측 마커 (해결된 것만)
    for p in predictions:
        if not p.resolved:
            continue
        color = "#00d26a" if p.is_correct else "#f23645"
        symbol = "triangle-up" if p.predicted_direction == "상승" else (
            "triangle-down" if p.predicted_direction == "하락" else "diamond"
        )
        fig.add_trace(go.Scatter(
            x=[p.timestamp],
            y=[p.price],
            mode="markers",
            marker=dict(size=10, color=color, symbol=symbol, line=dict(width=1, color="white")),
            name=f"{'✅' if p.is_correct else '❌'} {p.predicted_direction}",
            showlegend=False,
            hovertemplate=(
                f"<b>{p.timestamp.strftime('%H:%M')}</b><br>"
                f"가격: ${p.price:,.0f}<br>"
                f"예측: {p.predicted_slope:+.4f} ({p.predicted_direction})<br>"
                f"실제: {p.actual_slope:+.4f} ({p.actual_direction})<br>"
                f"결과: {'정확' if p.is_correct else '오류'}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # 거래량
    colors = ["#00d26a" if c >= o else "#f23645"
              for c, o in zip(candles["close"], candles["open"])]
    fig.add_trace(go.Bar(
        x=candles.index,
        y=candles["volume"],
        marker_color=colors,
        name="Volume",
        showlegend=False,
        opacity=0.5,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        font=dict(color="#e1e5ee"),
    )
    fig.update_xaxes(gridcolor="#1e2235", showgrid=True)
    fig.update_yaxes(gridcolor="#1e2235", showgrid=True)

    return fig


def create_slope_chart(slope_df: pd.DataFrame) -> go.Figure:
    """예측 vs 실제 기울기 시계열 차트를 생성한다."""
    fig = go.Figure()

    if slope_df.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=450,
            annotations=[dict(text="예측 데이터 수집 중...", showarrow=False,
                             font=dict(size=16, color="#8b92a5"),
                             xref="paper", yref="paper", x=0.5, y=0.5)],
        )
        return fig

    # 예측 기울기
    fig.add_trace(go.Scatter(
        x=slope_df.index,
        y=slope_df["predicted"],
        mode="lines+markers",
        name="예측 기울기",
        line=dict(color="#4da6ff", width=2),
        marker=dict(size=4),
    ))

    # 실제 기울기
    actual = slope_df.dropna(subset=["actual"])
    if not actual.empty:
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual["actual"],
            mode="lines+markers",
            name="실제 기울기",
            line=dict(color="#f7931a", width=2),
            marker=dict(size=4),
        ))

    # 0 라인
    fig.add_hline(y=0, line_dash="dash", line_color="#8b92a5", opacity=0.5)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#e1e5ee"),
        yaxis_title="기울기 (slope)",
    )
    fig.update_xaxes(gridcolor="#1e2235")
    fig.update_yaxes(gridcolor="#1e2235")

    return fig


# ---------------------------------------------------------------------------
# HTML 헬퍼
# ---------------------------------------------------------------------------

def stat_box(label: str, value: str, color: str = "blue", sub: str = "") -> str:
    return f"""
    <div class="stat-box">
        <div class="label">{label}</div>
        <div class="value {color}">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """


def direction_badge(direction: str) -> str:
    cls = {"상승": "badge-up", "하락": "badge-down", "횡보": "badge-flat"}.get(direction, "badge-pending")
    return f'<span class="badge {cls}">{direction}</span>'


def result_badge(result: str) -> str:
    if result == "✅":
        return '<span class="badge badge-correct">정확</span>'
    elif result == "❌":
        return '<span class="badge badge-wrong">오류</span>'
    return '<span class="badge badge-pending">대기</span>'


def build_prediction_table(df: pd.DataFrame) -> str:
    """예측 이력 DataFrame을 HTML 테이블로 변환한다."""
    if df.empty:
        return '<p style="color:#8b92a5;text-align:center;padding:40px;">예측 이력이 없습니다. 데이터 수집 중...</p>'

    rows_html = ""
    for _, row in df.head(30).iterrows():
        slope_color = "green" if row["예측 기울기"].startswith("+") else (
            "red" if row["예측 기울기"].startswith("-") else "gray"
        )
        actual_color = ""
        if row["실제 기울기"] != "⏳ 대기중":
            actual_color = "green" if row["실제 기울기"].startswith("+") else (
                "red" if row["실제 기울기"].startswith("-") else "gray"
            )

        rows_html += f"""
        <tr>
            <td>{row['시각']}</td>
            <td style="font-family:monospace;">{row['가격']}</td>
            <td class="{slope_color}" style="font-family:monospace;">{row['예측 기울기']}</td>
            <td>{direction_badge(row['예측 방향'])}</td>
            <td class="{actual_color}" style="font-family:monospace;">{row['실제 기울기']}</td>
            <td>{direction_badge(row['실제 방향']) if row['실제 방향'] != '⏳' else '<span class="badge badge-pending">대기</span>'}</td>
            <td>{row['소요(분)']}</td>
            <td>{result_badge(row['결과'])}</td>
        </tr>
        """

    return f"""
    <table class="prediction-table">
        <thead>
            <tr>
                <th>시각</th>
                <th>가격</th>
                <th>예측 기울기</th>
                <th>예측 방향</th>
                <th>실제 기울기</th>
                <th>실제 방향</th>
                <th>소요(분)</th>
                <th>결과</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# 메인 대시보드
# ---------------------------------------------------------------------------

def render_dashboard():
    """대시보드 전체를 렌더링한다."""
    # 엔진 초기화
    init_engine()

    engine = st.session_state.engine

    # 사이드바
    with st.sidebar:
        mode_label = "🟢 실시간" if st.session_state.get("mode") == "live" else "🟡 데모"
        st.markdown(f"### ₿ BTC Slope Predictor")
        st.markdown(f"모드: **{mode_label}**")
        st.divider()

        refresh_sec = st.selectbox(
            "새로고침 간격",
            options=[10, 30, 60],
            index=1,
            format_func=lambda x: f"{x}초",
        )
        auto_refresh = st.toggle("자동 새로고침", value=True)

        if st.button("🔄 수동 새로고침", use_container_width=True):
            engine.update()
            st.rerun()

        st.divider()
        st.markdown("**모델 설정**")
        st.markdown(f"- 시퀀스 길이: `{engine.params.n}분`")
        st.markdown(f"- 목표 변동률: `{engine.params.x}%`")
        st.markdown(f"- 관측 윈도우: `{engine.params.slope_max_window}분`")
        st.markdown(f"- 심볼: `{engine.params.symbol}`")

        st.divider()
        st.markdown("**사용법**")
        st.markdown("""
        ```
        # 실시간 모드
        python -m crypto_predictor train
        streamlit run crypto_predictor/dashboard.py

        # 데모 모드
        streamlit run crypto_predictor/dashboard.py -- --demo
        ```
        """)

    # 데이터 업데이트
    engine.update()

    # 데이터 준비
    stats = engine.get_stats()
    pred_df = engine.get_predictions_df()
    candle_df = engine.get_candle_df()
    slope_df = engine.get_slope_series()

    current_price = float(candle_df["close"].iloc[-1]) if not candle_df.empty else 0
    latest_pred = engine.predictions[-1] if engine.predictions else None

    # === 헤더 ===
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <span class="main-title">₿ BTC/USDT 실시간 기울기 예측</span>
    </div>
    """, unsafe_allow_html=True)

    # === Row 1: 핵심 지표 ===
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        price_delta = None
        if not candle_df.empty and len(candle_df) > 1:
            prev = float(candle_df["close"].iloc[-2])
            price_delta = f"{(current_price - prev):+,.0f}"
        st.metric("현재가", f"${current_price:,.0f}", delta=price_delta)

    with col2:
        if latest_pred:
            slope_val = f"{latest_pred.predicted_slope:+.4f}"
            st.metric("예측 기울기", slope_val)
        else:
            st.metric("예측 기울기", "-")

    with col3:
        if latest_pred:
            direction = latest_pred.predicted_direction
            icon = {"상승": "📈", "하락": "📉", "횡보": "➡️"}.get(direction, "")
            st.metric("예측 방향", f"{icon} {direction}")
        else:
            st.metric("예측 방향", "-")

    with col4:
        acc = stats.get("accuracy")
        acc_str = f"{acc:.1%}" if acc is not None else "-"
        st.metric("전체 정확도", acc_str, delta=f"{stats['resolved']}건 해결")

    with col5:
        st.metric("예측 수", f"{stats['total_predictions']}",
                  delta=f"대기 {stats['pending']}건")

    # === Row 2: 차트 ===
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### 📊 가격 차트")
        if not candle_df.empty:
            # 최근 120개 캔들만 표시
            display_candles = candle_df.tail(120)
            fig_price = create_price_chart(display_candles, engine.predictions)
            st.plotly_chart(fig_price, key="price_chart")
        else:
            st.info("가격 데이터 로딩 중...")

    with chart_col2:
        st.markdown("##### 📈 기울기 예측 vs 실제")
        fig_slope = create_slope_chart(slope_df)
        st.plotly_chart(fig_slope, key="slope_chart")

    # === Row 3: 통계 + 테이블 ===
    st.markdown("---")
    stat_col, table_col = st.columns([1, 3])

    with stat_col:
        st.markdown("##### 📊 정확도 통계")

        # 전체 정확도
        acc_val = f"{stats['accuracy']:.1%}" if stats['accuracy'] is not None else "-"
        acc_color = "green" if stats.get('accuracy', 0) and stats['accuracy'] >= 0.5 else "red"
        correct = stats.get('correct', 0)
        wrong = stats.get('wrong', 0)
        st.markdown(stat_box("전체 정확도", acc_val, acc_color,
                             f"{correct}건 정확 / {wrong}건 오류"),
                    unsafe_allow_html=True)

        # 상승 예측 정확도
        up_acc = stats.get("up_accuracy")
        up_val = f"{up_acc:.1%}" if up_acc is not None else "-"
        st.markdown(stat_box("상승 예측", up_val, "green",
                             f"{stats.get('up_count', 0)}건"),
                    unsafe_allow_html=True)

        # 하락 예측 정확도
        down_acc = stats.get("down_accuracy")
        down_val = f"{down_acc:.1%}" if down_acc is not None else "-"
        st.markdown(stat_box("하락 예측", down_val, "red",
                             f"{stats.get('down_count', 0)}건"),
                    unsafe_allow_html=True)

        # 횡보 예측 정확도
        flat_acc = stats.get("flat_accuracy")
        flat_val = f"{flat_acc:.1%}" if flat_acc is not None else "-"
        st.markdown(stat_box("횡보 예측", flat_val, "gray",
                             f"{stats.get('flat_count', 0)}건"),
                    unsafe_allow_html=True)

        # 평균 해결 시간
        avg_min = stats.get("avg_resolution_minutes")
        avg_val = f"{avg_min:.1f}분" if avg_min else "-"
        st.markdown(stat_box("평균 해결 시간", avg_val, "blue"),
                    unsafe_allow_html=True)

    with table_col:
        st.markdown("##### 📋 예측 이력 (최근 30건)")
        table_html = build_prediction_table(pred_df)
        st.markdown(table_html, unsafe_allow_html=True)

    # === 자동 새로고침 ===
    if auto_refresh:
        st.markdown(
            f'<meta http-equiv="refresh" content="{refresh_sec}">',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_dashboard()
else:
    render_dashboard()
