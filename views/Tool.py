# -------------------
# Imports
# -------------------
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from class_functions import C_AllocationStrategy

# ------------------- 
# Page setup
# -------------------
st.set_page_config(
    page_title="Allocation Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)
st.title("⚙️ Tool")

# ------------------- 
# Sidebar Inputs
# -------------------
# Date selector
selected_date = st.sidebar.date_input("Select date", value=date.today())

# Dark Mode Toggle at bottom of sidebar
st.sidebar.markdown(
    """
    <style>
    /* Push dark mode checkbox to bottom */
    .css-1d391kg {
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
primary_bg = "#0e1117" if dark_mode else "#ffffff"
card_bg = "#1f2937" if dark_mode else "#f9fafb"
text_color = "#ffffff" if dark_mode else "#000000"

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
            margin-bottom: 20px;
        }}
        .metric-title {{
            font-size: 14px;
            color: {text_color};
            opacity: 0.85;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: {text_color};
        }}
        .dataframe th {{
            background-color: {card_bg};
            color: {text_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- 
# Load session data
# -------------------
if "model" not in st.session_state:
    st.error("Model not loaded. Please go to the Overview page first.")
    st.stop()

filtered_data = st.session_state["filtered_data"]
model = st.session_state["model"]
scaler = st.session_state["scaler"]
feature_cols = st.session_state["feature_cols"]

# ------------------- 
# Initialize Allocation Strategy
# -------------------
alloc = C_AllocationStrategy(main_dataset=filtered_data, feature_cols=feature_cols)
alloc.model = model
alloc.scaler = scaler

# ------------------- 
# Generate trading signals
# -------------------
signals_df = alloc.F_GenerateSignals(filtered_data)

# Ensure engineered features exist
for f in ["sum_exec_ret_last7", "signals_last7d"]:
    if f not in signals_df.columns:
        results_sum, results_count = [], []
        for i, row in signals_df.iterrows():
            sig = row["signal_date"]
            mask = (signals_df["signal_date"] >= (sig - pd.Timedelta(days=7))) & (signals_df["signal_date"] < sig)
            results_sum.append(signals_df.loc[mask, "entry_exec_ret"].sum(skipna=True))
            results_count.append(mask.sum())
        signals_df["sum_exec_ret_last7"] = results_sum
        signals_df["signals_last7d"] = results_count

# Ensure all feature columns exist
for f in feature_cols:
    if f not in signals_df.columns:
        if f in filtered_data.columns:
            signals_df[f] = filtered_data[f].values[:len(signals_df)]
        else:
            signals_df[f] = 0.0

# Predict allocation probabilities
signals_df = alloc.F_ClassificationPredictions(signals_df)

# ------------------- 
# Suggested allocation for selected date
# -------------------
signal_row = signals_df[signals_df['signal_date'] == pd.Timestamp(selected_date)]
alloc_pct = float(signal_row['allocation_pct']) if not signal_row.empty else 0.0

# Display as card
card_color = "#16a34a" if alloc_pct > 0 else card_bg  # green for signal

st.markdown(
    f"""
    <div class="metric-card" style="background-color: {card_color};">
        <div class="metric-title">Suggested Allocation</div>
        <div class="metric-value">{alloc_pct:.2f}%</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------- 
# Feature values at selected date
# -------------------
selected_ts = pd.Timestamp(selected_date)

# Allocation features
row_features = filtered_data[filtered_data['Date'] == selected_ts]
if row_features.empty:
    row_features = pd.DataFrame([{f: np.nan for f in feature_cols}])
else:
    row_features = row_features[[f for f in feature_cols if f in row_features.columns]]

# Add engineered features
for f in ["sum_exec_ret_last7", "signals_last7d"]:
    if f in signals_df.columns:
        sig_row = signals_df[signals_df['signal_date'] == selected_ts]
        row_features[f] = sig_row.iloc[0][f] if not sig_row.empty else 0.0

row_features = row_features.reindex(columns=feature_cols)

# Signal features
signal_features = ['VIX_Z_60', 'SP500_trend_5', 'VIX_Z_60_roc1', 'VIX_Z_60_roc3']
signal_data = filtered_data[filtered_data['Date'] == selected_ts][signal_features]
if signal_data.empty:
    signal_data = pd.DataFrame([{f: np.nan for f in signal_features}])

# Merge allocation + signal features
full_features = pd.concat([signal_data, row_features], axis=1)
display_df = full_features.T.reset_index()
display_df.columns = ["Feature", "Value at Selected Date"]
display_df["Type"] = ["Signal" if f in signal_features else "Allocation" for f in display_df["Feature"]]


# ------------------- 
# Expandable Raw Data
# -------------------
with st.expander("🔍 Features' values at selected date"):
    st.dataframe(
    display_df.style.format({"Value at Selected Date": "{:,.4f}"}).set_table_styles(
        [{"selector": "th", "props": [("background-color", card_bg), ("color", text_color)]}]
    ), use_container_width=True
)
