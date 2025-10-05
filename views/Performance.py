# -------------------
# Imports
# -------------------
import streamlit as st
import pandas as pd
from class_functions import (
    C_AllocationStrategy,
    F_PerformanceMetrics,
    F_PlotCumulativePerformance,
    F_PlotReturns
)

# ------------------- 
# Page setup
# -------------------
st.set_page_config(page_title="Strategy Performance", layout="wide")
st.title("📈 Strategy Historical Performance")

# ------------------- 
# Sidebar Inputs
# -------------------
initial_portfolio = st.sidebar.number_input("💰 Initial Portfolio ($)", 100, 100000, 1000, step=100)

# Minimal dark mode toggle pinned at bottom
st.sidebar.markdown(
    """
    <style>
    .css-1d391kg {  /* Sidebar content wrapper */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100vh;
    }
    .dark-mode-container {
        margin-top: auto;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar.container():
    st.markdown('<div class="dark-mode-container">', unsafe_allow_html=True)
    dark_mode = st.checkbox("🌙 Dark Mode", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- 
# Apply Theme Styling
# -------------------
if dark_mode:
    primary_bg = "#0e1117"
    secondary_bg = "#1f2937"
    text_color = "#ffffff"
    card_bg = "#1f2937"
else:
    primary_bg = "#ffffff"
    secondary_bg = "#f0f2f6"
    text_color = "#000000"
    card_bg = "#f9fafb"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {primary_bg};
            color: {text_color};
        }}
        .metric-card {{
            background-color: {card_bg};
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.15);
            width: 100%;
            margin-bottom: 20px;  /* vertical spacing between cards */
        }}
        .metric-title {{
            font-size: 14px;
            color: {text_color};
            opacity: 0.85;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: {text_color};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- 
# Session Model / Data
# -------------------
if "model" not in st.session_state:
    st.error("Model not loaded. Please go to the Overview page first.")
    st.stop()

filtered_data = st.session_state["filtered_data"]
model = st.session_state["model"]
scaler = st.session_state["scaler"]
feature_cols = st.session_state["feature_cols"]

# ------------------- 
# Allocation Engine
# -------------------
feature_cols_raw = (
    filtered_data.filter(regex="^SP500_Volume|^SP500_ADX|^VIX_ADX").columns.tolist()
    + ["sum_exec_ret_last7", "signals_last7d"]
)

strategy = C_AllocationStrategy(main_dataset=filtered_data, feature_cols=feature_cols_raw)
strategy.model = model
strategy.scaler = scaler

with st.spinner("Running strategy backtest..."):
    backtested_trades = strategy.F_RunStrategy(
        main_dataset=filtered_data,
        initial_portfolio=initial_portfolio,
        window_days=7,
        model_path="saved_model.pkl",
        lookback_days=7,
    )

if backtested_trades is None or backtested_trades.empty:
    st.warning("No trades executed. Try adjusting parameters.")
    st.stop()

# ------------------- 
# Performance Metrics
# -------------------
metrics = F_PerformanceMetrics(backtested_trades, initial_portfolio=initial_portfolio)
st.markdown("### ✅ Metrics Summary")

metric_items = list(metrics.items())
cards_per_row = 3
horizontal_spacing = 0.02  # fraction of column width for horizontal spacers

# Layout KPI cards with proper spacing
for i in range(0, len(metric_items), cards_per_row):
    row_metrics = metric_items[i:i+cards_per_row]
    columns_widths = []
    for _ in range(len(row_metrics)):
        columns_widths.append(1)
        columns_widths.append(horizontal_spacing)
    columns_widths = columns_widths[:-1]
    cols = st.columns(columns_widths)

    for j, (key, val) in enumerate(row_metrics):
        col_index = j*2  # account for spacers
        with cols[col_index]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{key}</div>
                    <div class="metric-value">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ------------------- 
# Plots
# -------------------
st.markdown("### 📊 Portfolio Evolution")

col1, col2 = st.columns(2)
with col1:
    fig_portfolio = F_PlotCumulativePerformance(backtested_trades, dark_mode=dark_mode, initial_portfolio=initial_portfolio)
    st.pyplot(fig_portfolio, use_container_width=True)
with col2:
    fig_trades = F_PlotReturns(backtested_trades, dark_mode=dark_mode)
    st.pyplot(fig_trades, use_container_width=True)

# ------------------- 
# Expandable Raw Data
# -------------------
with st.expander("🔍 Backtest Trades Data"):
    st.dataframe(
        backtested_trades[['signal_date', 'entry_date', 'invested','return', 'portfolio_after_exit']],
        use_container_width=True
    )
