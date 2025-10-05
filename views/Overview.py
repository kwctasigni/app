# -------------------
# Imports
# -------------------
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from class_functions import C_Preprocessing, F_LoadAllData

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="Overview",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# -------------------
# Sidebar Inputs
# ------------------
st.sidebar.markdown(
    """
    <style>
    /* Force sidebar content to fill full height and push dark mode checkbox to bottom */
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
# -------------------
# Drark Mode Toggle
# -------------------
with st.sidebar.container():
    st.markdown('<div class="dark-mode-container">', unsafe_allow_html=True)
    dark_mode = st.checkbox("🌙 Dark Mode", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------
# Apply global theme
# -------------------
if dark_mode:
    primary_bg = "#0e1117"
    secondary_bg = "#1f2937"
    text_color = "#ffffff"
    link_color = "#3b82f6"
else:
    primary_bg = "#ffffff"
    secondary_bg = "#f0f2f6"
    text_color = "#000000"
    link_color = "#1d4ed8"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {primary_bg};
            color: {text_color};
        }}
        .stMarkdown a {{
            color: {link_color};
            text-decoration: none;
        }}
        .stDataFrame, .stTable {{
            background-color: {secondary_bg} !important;
            color: {text_color} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------
# Page Title
# -------------------
st.title("📗 Overview: SVXY Trading Strategy")

# -------------------
# Products & Context in 3 Columns
# -------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### S&P 500 (SPX)")
    st.markdown("""
    - Market-cap weighted index of 500 large US stocks.  
    - Serves as a benchmark for the US equity market.  
    - Used in this strategy to measure trends, returns, and volume changes.  
    - [S&P 500 on Yahoo Finance](https://finance.yahoo.com/quote/%5EGSPC/)
    """)

with col2:
    st.markdown("### VIX (CBOE Volatility Index)")
    st.markdown("""
    - Measures **implied volatility of S&P 500 options**, also called the “fear gauge”.  
    - High VIX → expected large market moves; Low VIX → calm market.  
    - **Persistent behavior**: VIX often exhibits spike-after-spike patterns, clustering volatility rather than isolated events.  
    - [VIX on Yahoo Finance](https://finance.yahoo.com/quote/%5EVIX/)
    """)

with col3:
    st.markdown("### SVXY (ProShares Short VIX Short-Term Futures ETF)")
    st.markdown("""
    - Tracks **inverse performance of the VIX short-term futures index**.  
    - Profits when volatility falls; loses when volatility rises.  
    - High risk and leveraged, intended for short-term trading.  
    - [SVXY on Yahoo Finance](https://finance.yahoo.com/quote/SVXY/)
    """)

# -------------------
# Strategy Comments
# -------------------
st.markdown("""
## Navigation

The dashboard is divided into two interconnected analytical components:

**Performance** — Evaluates the historical effectiveness of the allocation strategy using realized and simulated trading data. It reports key risk-adjusted performance indicators (e.g., cumulative return, drawdown, Sharpe ratio) and visualizes how the portfolio evolved in response to market volatility.  

**Tool** — Provides a forward-looking interface that allows users to inspect the model’s suggested SVXY allocation on any selected date, along with the underlying feature values that drive the decision. This enables transparency, interpretability, and traceability of the model’s daily predictions.

---


## Summary

Overall, the allocation framework integrates:
1. **Feature engineering** for volatility dynamics.
2. **Signal gneneration** based on historical VIX and SPX patterns.
3. **Sequential (walk-forward) Lasso learning** with out-of-sample validation for allocation.
4. **Overlay risk and exposure bounding** for regime control and due to the small sample of the classification model.
5. **Monte Carlo robustness checks** ensuring statistical and temporal validity.

---

## Signal Design

The model’s predictive engine is based on a structured set of volatility and trend-sensitive indicators engineered from the **VIX** and **S&P 500** time series. The goal is to quantify **short-term volatility momentum**, **trend alignment**, and **market stress conditions** to anticipate periods of favorable or adverse environments for inverse volatility exposure (SVXY). 
Potentially, this logic can be adapted to other volatility-linked products with similar characteristics.

The **VIX Z-score** is the primary volatility signal. It measures how far the VIX index deviates from its recent average over multiple horizons (e.g., 60 days). This helps identify when implied volatility is abnormally high or low relative to its history. Periods of high positive Z-scores often indicate elevated fear or risk aversion, while negative values suggest complacency or mean reversion potential.  

To capture dynamic changes in volatility conditions, the model introduces **rate-of-change transformations** of the Z-score — such as one-day (`roc1`) and three-day (`roc3`) changes. These detect sudden accelerations or decelerations in volatility, serving as leading indicators for potential reversals.  

The **S&P 500 trend** feature complements volatility-based metrics by reflecting directional market momentum. It is typically calculated using short-horizon moving averages or ADX components, enabling the model to align or filter signals based on broader equity market direction. For instance, if volatility spikes occur in an already downward-trending market, the model treats such signals differently from volatility spikes in a stable or rising market environment.  

Volume-based indicators (e.g., `SP500_Volume`, `SP500_ADX`, `VIX_ADX`) are also integrated to distinguish between meaningful price–volatility movements and low-liquidity noise. High volume combined with volatility expansion typically indicates stronger signal conviction, reflected through the allocation strategy.  

Finally, two short-memory indicators — **`sum_exec_ret_last7`** and **`signals_last7d`** — are computed to measure recent signal activity and trading outcomes. These help the model recognize volatility clustering and avoid overreacting to isolated spikes or short-lived reversals.  

The resulting feature set creates a **multi-horizon volatility map**, combining absolute (Z-score) and relative (rate-of-change) measures to form a robust foundation for the allocation model.

---

## **Allocation Strategy**

The allocation process is driven by a **Lasso-regularized logistic classification**, which transforms the engineered feature set into probabilistic allocation values whenever a signal is triggered. Lasso (L1-regularization) imposes sparsity by penalizing weak predictors, ensuring that only statistically meaningful features contribute to the model. This promotes interpretability and prevents overfitting, especially in this setup where the number of training signals is limited.

To emulate real-time deployment, the classifier is trained using a **walk-forward validation framework**, where at each iteration, the model is fitted only on past data and validated on the immediately following time segment.  
This procedure, implemented via `TimeSeriesSplit`, avoids lookahead bias and ensures that predictive performance is measured under realistic sequential conditions. Within each training window, all features are standardized and passed through a logistic model with L1 penalty (`liblinear` solver). The Lasso penalty automatically filters out redundant or noisy volatility predictors, leaving only the strongest structural drivers.

An **out-of-sample validation subset** is held out completely from the walk-forward folds, ensuring that threshold tuning (for accuracy or F1 optimization) reflects unseen data behavior. The model never uses future information from test or validation samples, maintaining **temporal causality** throughout the entire process.

Once probabilities are estimated, they are mapped into **dynamic allocation percentages** according to:
""")

st.latex(r"\text{allocation} = (1 - p_{\text{negative return}}) \times 100")

st.markdown("""
This transformation implies that higher probabilities of favorable outcomes yield larger target exposures, while increased risk of negative returns automatically scales down allocations.

To enhance robustness, the system applies an **overlay risk control** adjustment that accounts for recent signal behavior.  
If the short-memory indicators (`sum_exec_ret_last7`, `signals_last7d`) suggest instability — such as consecutive large VIX spikes or clustered losses — allocations are automatically reduced. This prevents overexposure during turbulence and enforces smoother exposure transitions, effectively integrating short-term volatility memory into the risk management process.

In addition, the allocation output is **bounded within predefined exposure limits**, enforcing realistic portfolio management constraints. Even under strong bullish or bearish conditions, exposure cannot exceed or fall below the operational bounds (e.g., 0%–100%). These constraints, implemented in the `adjust_allocation()` procedure, ensure that the model retains a controlled risk footprint and avoids excessive leverage or complete disengagement from the market.

The overall validation framework combines **in-sample Monte Carlo testing**:  
- The first validation layer employs a **Monte Carlo permutation test** on the raw signal logic to assess statistical significance. By repeatedly shuffling Open and Close sequences while preserving return characteristics, it measures whether the observed Profit Factor materially exceeds what would occur by random chance. A low permutation-based p-value confirms that the signal–return linkage reflects genuine predictive structure rather than noise. 
- The second Monte Carlo applies **temporal block resampling** that preserves continuity and serial dependence, repeatedly drawing contiguous sub-periods of signals to verify allocation stability under diverse market conditions.

Together, these complementary validation layers confirm that the allocation strategy captures **persistent, economically interpretable patterns** in volatility dynamics rather than transient, data-mined artifacts.

Overall, the allocation framework integrates:
1. **Feature engineering** for volatility dynamics,  
2. **Sequential (walk-forward) Lasso learning** with out-of-sample validation,  
3. **Overlay risk and exposure bounding** for regime control, and  
4. **Monte Carlo robustness checks** ensuring statistical and temporal validity.

This cohesive structure allows the model to systematically manage **SVXY exposure** through objective, data-driven assessments of market volatility and signal persistence.
""", unsafe_allow_html=True)

# -------------------
# Feature Dictionary Table
# -------------------
st.markdown("### Feature Dictionary")

features = {
    'VIX_Z_60': 'VIX Z-score over 60-day window',
    'SP500_trend_5': 'S&P 500 pct change over 5 days',
    'VIX_Z_60_roc1': '1-day difference of VIX Z-score (60-day)',
    'VIX_Z_60_roc3': '3-day difference of VIX Z-score (60-day)',
    'SP500_Volume': 'S&P 500 daily volume',
    'SP500_Volume_60': 'Relative SP500 volume over 60 days',
    'SP500_Volume_30': 'Relative SP500 volume over 30 days',
    'SP500_Volume_10': 'Relative SP500 volume over 10 days',
    'SP500_Volume_5': 'Relative SP500 volume over 5 days',
    'SP500_ADX_60': 'ADX trend strength for SP500 (60-day)',
    'VIX_ADX_60': 'ADX trend strength for VIX (60-day)',
    'SP500_ADX_30': 'ADX trend strength for SP500 (30-day)',
    'VIX_ADX_30': 'ADX trend strength for VIX (30-day)',
    'SP500_ADX_10': 'ADX trend strength for SP500 (10-day)',
    'VIX_ADX_10': 'ADX trend strength for VIX (10-day)',
    'SP500_ADX_5': 'ADX trend strength for SP500 (5-day)',
    'VIX_ADX_5': 'ADX trend strength for VIX (5-day)',
    'sum_exec_ret_last7': 'Sum of SVXY execution returns over last 7 days',
    'signals_last7d': 'Count of signals in last 7 days'
}

feature_df = pd.DataFrame({
    'Feature': list(features.keys()),
    'Description': list(features.values())
})

st.dataframe(feature_df.style.set_properties(**{'background-color': secondary_bg, 'color': text_color}))


# -------------------
# Load once and put into session_state
# -------------------

if "model" not in st.session_state:
    try:
        filtered_data, model, scaler, feature_cols = F_LoadAllData("saved_model.pkl")
        st.session_state["filtered_data"] = filtered_data
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["feature_cols"] = feature_cols
    except Exception as e:
        st.error(f"Failed to load data/model: {e}")
        st.stop()
