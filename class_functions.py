import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

# Class imports
class C_Preprocessing:
    def __init__(self, tickers=None, start_date='2007-01-01', end_date=None, vix_window=60):
        """
        Initialize preprocessing class with tickers, date range, and parameters.
        """
        if tickers is None:
            tickers = ['^GSPC', '^VIX', 'SVXY']
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = pd.Timestamp.today().strftime('%Y-%m-%d') if end_date is None else end_date
        self.vix_window = vix_window
        print(f"Initialized C_Preprocessing for tickers {self.tickers} from {self.start_date} to {self.end_date}")

    def F_LoadData(self):
        """
        Load OHLCV data from yfinance.
        
        - Features (X): Use Close prices for SP500 and VIX (for trend, z-scores, etc.).
        - Trading instrument (SVXY): Keep both Open and Close prices.
        - Entry = SVXY Open (next day after signal).
        - Exit = SVXY Close (same day as exit signal).
        """

        raw = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval='1d'
        )[['Open', 'High', 'Low', 'Close','Volume']]

        df = pd.DataFrame({
            # Feature universe (based on closes)
            'SP500_Close': raw['Close']['^GSPC'],
            'VIX_Close': raw['Close']['^VIX'],
            'SP500_Volume': raw['Volume']['^GSPC'],
            'SP500_High': raw['High']['^GSPC'],
            'SP500_Low': raw['Low']['^GSPC'],
            'VIX_High': raw['High']['^VIX'],
            'VIX_Low': raw['Low']['^VIX'],

            # Trading instrument (SVXY)
            'SVXY_Open': raw['Open']['SVXY'],
            'SVXY_Close': raw['Close']['SVXY']
        })

        df.dropna(inplace=True)
        df.reset_index(inplace=True)   # keep 'Date' as a column
        df.rename(columns={'index':'Date'}, inplace=True)

        return df

    def F_ComputeADX(self, df, high_col='High', low_col='Low', close_col='Close', period=14):
        """
        Compute ADX, +DI, -DI for a given OHLC dataframe.
        """
        high = df[high_col]
        low = df[low_col]
        close = df[close_col]

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        # Directional movements
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Smoothed averages (EMA)
        tr_smooth = tr.ewm(span=period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()

        # DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        return pd.DataFrame({'ADX':adx})

    def F_ComputeFeatures(self):
        """
        Compute VIX stats, SP500 trend, returns, and ADX for SP500 and VIX.
        - Features are based on CLOSE prices (signal consistency).
        - SVXY execution returns can later be derived as Open→Close.
        """
        df = self.F_LoadData()

        # --- Log returns based on CLOSE prices (for features) ---
        df['SP500_log_ret'] = np.log(df['SP500_Close'] / df['SP500_Close'].shift(1))
        df['VIX_log_ret']   = np.log(df['VIX_Close'] / df['VIX_Close'].shift(1))
        df['SVXY_log_ret']  = np.log(df['SVXY_Close'] / df['SVXY_Close'].shift(1))
        df['SVXY_exec_ret'] = np.log(df['SVXY_Close'] / df['SVXY_Open'])

        # VIX metrics for multiple periods
        periods = [self.vix_window, self.vix_window // 2, self.vix_window // 6, self.vix_window // 12]
        for p in periods:
            df[f'VIX_MA_{p}'] = df['VIX_Close'].rolling(window=p).mean()
            df[f'VIX_STD_{p}'] = df['VIX_Close'].rolling(window=p).std()
            df[f'VIX_Z_{p}'] = (df['VIX_Close'] - df[f'VIX_MA_{p}']) / df[f'VIX_STD_{p}']
            df[f'VIX_Z_{p}_roc1'] = df[f'VIX_Z_{p}'].diff()
            df[f'VIX_Z_{p}_roc3'] = df[f'VIX_Z_{p}'].diff(3)
            df[f'SP500_Volume_{p}'] = df['SP500_Volume'] / df['SP500_Volume'].rolling(window=p).median()

        # ADX for SP500 and VIX
        for p in periods:
            # SP500 ADX
            adx_sp500 = self.F_ComputeADX(df,
                high_col='SP500_High',
                low_col='SP500_Low',
                close_col='SP500_Close',
                period=p
            )
            df[f'SP500_ADX_{p}'] = adx_sp500['ADX']

            # VIX ADX
            adx_vix = self.F_ComputeADX(df,
                high_col='VIX_High',
                low_col='VIX_Low',
                close_col='VIX_Close',
                period=p
            )
            df[f'VIX_ADX_{p}'] = adx_vix['ADX']

            # SP500 trend (close-to-close percentage change over p days)
            df[f'SP500_trend_{p}'] = df['SP500_Close'].pct_change(periods=p)

        print("Features computed.")
        return df


class C_AllocationStrategy:
    def __init__(self, main_dataset, feature_cols, target_col='log_ret'):
        """
        Allocation strategy class combining walk-forward logistic regression
        with adjustable allocation rules.
        main_dataset: full dataset from which signals will be generated
        """
        self.main_dataset = main_dataset.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model = None          # trained or loaded classifier
        self.scaler = None         # fitted or loaded scaler
        self.final_df = None
        self.summary = None
        self.best_thresholds = None
        self.trades_df = None

    # -------------------
    # 1) TRAIN + SAVE
    # -------------------
    def F_ClassificationModel(self, signals_df,min_train_size=20, n_splits=5,
                               threshold_metric='accuracy', export_model_path=None):
        """
        Walk-forward logistic regression with threshold selection.
        Stores predictions, probabilities, allocation percentages, and optionally exports model.
        """
        df = signals_df.copy()
        X = df[self.feature_cols]

        
        # Warn if any feature_cols are missing (do not fail)
        missing_features = [f for f in self.feature_cols if f not in df.columns]
        if missing_features:
            print(f"⚠️ The following required feature columns are missing in the input data: {missing_features}")
            
        y = (df[self.target_col] <= 0).astype(int)  # 1 = negative return

        tscv = TimeSeriesSplit(n_splits=n_splits)
        df['y_true'] = y
        df['y_proba'] = np.nan
        df['y_pred'] = np.nan
        df['set'] = 'train'

        # Walk-forward training
        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) < min_train_size:
                continue

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
            clf.fit(X_train_scaled, y_train)

            # Save last fold model & scaler for future predictions
            self.model = clf
            self.scaler = scaler

            df.loc[train_idx, 'y_proba'] = clf.predict_proba(X_train_scaled)[:, 1]
            df.loc[test_idx, 'y_proba'] = clf.predict_proba(X_test_scaled)[:, 1]
            df.loc[test_idx, 'set'] = 'test'

        # Threshold search
        all_true = df.loc[df['set'] == 'test', 'y_true'].values
        all_probas = df.loc[df['set'] == 'test', 'y_proba'].values

        thresholds = np.linspace(0, 1, 101)
        acc_scores, f1_scores = [], []

        for t in thresholds:
            preds_t = (all_probas >= t).astype(int)
            acc_scores.append((t, accuracy_score(all_true, preds_t)))
            f1_scores.append((t, f1_score(all_true, preds_t)))

        best_acc_t, best_acc = max(acc_scores, key=lambda x: x[1])
        best_f1_t, best_f1 = max(f1_scores, key=lambda x: x[1])
        final_threshold = best_f1_t if threshold_metric == 'f1' else best_acc_t

        df['y_pred'] = (df['y_proba'] >= final_threshold).astype(int)
        df['allocation_pct'] = (1 - df['y_proba']) * 100

        self.final_df = df
        self.summary = {
            "best_acc_threshold": best_acc_t,
            "accuracy": best_acc,
            "best_f1_threshold": best_f1_t,
            "f1_score": best_f1,
            "used_threshold": final_threshold,
            "weighted_return": (df[self.target_col] * df['allocation_pct'] / 100).sum()
        }

        if export_model_path is not None:
            joblib.dump({'model': self.model, 'scaler': self.scaler,
                         'feature_cols': self.feature_cols}, export_model_path)
            print(f"✅ Model and scaler saved to {export_model_path}")

        self.F_PlotConfusionMatrix(all_true, all_probas, best_acc_t, best_f1_t)
        return df, self.summary

    # -------------------
    # 2) LOAD + PREDICT
    # -------------------
    def F_ClassificationPredictions(self, new_df, model_path=None):
        """
        Predict probabilities on new data using either:
        - the trained model in memory (if model_path=None),
        - or a saved model+scaler from joblib (if model_path given).
        """
        if model_path is not None:
            saved = joblib.load(model_path)
            self.model = saved['model']
            self.scaler = saved['scaler']
            self.feature_cols = saved['feature_cols']
            print(f"✅ Loaded model from {model_path}")

        if self.model is None or self.scaler is None:
            raise ValueError("No model available. Train or load a model first.")

        X_new = new_df[self.feature_cols]
        X_new_scaled = self.scaler.transform(X_new)
        new_df = new_df.copy()
        new_df['y_proba'] = self.model.predict_proba(X_new_scaled)[:, 1]
        new_df['allocation_pct'] = (1 - new_df['y_proba']) * 100
        return new_df

    # -------------------
    # Allocation adjust - Overlay
    # -------------------
    def F_AllocationOverlay(self, final_df, prob_col, alloc_col,
                          thr_rule1=0.3, thr_rule2=0.6,
                          thr_low=0.1, thr_high=0.8,
                          window_days=7, require_non_na=True, verbose=False):
        if final_df is None:
            raise ValueError("Need predictions first!")

        df = final_df.copy().sort_values('signal_date').reset_index(drop=True)
        df[alloc_col + '_orig'] = df[alloc_col].astype(float)
        df[alloc_col + '_adj'] = df[alloc_col].astype(float)
        df['rule_applied'] = None
        dates = pd.to_datetime(df['signal_date'])

        for i in range(len(df)):
            cur_date = dates.iloc[i]
            window_start = cur_date - pd.Timedelta(days=window_days)
            prior_mask = (dates >= window_start) & (dates < cur_date)
            prior_idx = list(df.index[prior_mask])
            n_prior = len(prior_idx)
            cur_p = df.at[i, prob_col]

            if pd.isna(cur_p):
                continue

            if cur_p < thr_low:
                df.at[i, alloc_col + '_adj'] = 100.0
                df.at[i, 'rule_applied'] = 'rule3'
                continue
            if cur_p > thr_high:
                df.at[i, alloc_col + '_adj'] = 0.0
                df.at[i, 'rule_applied'] = 'rule4'
                continue
            if n_prior >= 2:
                last2_idx = prior_idx[-2:]
                p_last2 = df.loc[last2_idx, prob_col].values
                if not (require_non_na and np.any(pd.isna(p_last2))):
                    if (cur_p < thr_rule1) and np.all(p_last2 < thr_rule1):
                        df.at[i, alloc_col + '_adj'] = 100.0
                        df.at[i, 'rule_applied'] = 'rule1'
                        continue
            if n_prior >= 1:
                last1_idx = prior_idx[-1]
                p_last1 = df.at[last1_idx, prob_col]
                if not (pd.isna(p_last1) and require_non_na):
                    if (cur_p >= thr_rule2) and (p_last1 >= thr_rule2):
                        df.at[i, alloc_col + '_adj'] = 0.0
                        df.at[i, 'rule_applied'] = 'rule2'
                        continue

        df['pnl_orig'] = df['log_ret'] * df[alloc_col + '_orig']
        df['pnl_adj'] = df['log_ret'] * df[alloc_col + '_adj']
        self.final_df = df
        return df

    @staticmethod
    def F_PlotConfusionMatrix(all_true, all_probas, best_acc_t, best_f1_t):
        cm_acc = confusion_matrix(all_true, (all_probas >= best_acc_t).astype(int))
        cm_f1 = confusion_matrix(all_true, (all_probas >= best_f1_t).astype(int))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm_acc, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Confusion Matrix (Accuracy Thr {best_acc_t:.2f})')
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
        sns.heatmap(cm_f1, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title(f'Confusion Matrix (F1 Thr {best_f1_t:.2f})')
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
        plt.tight_layout(); plt.show()

    def F_LoadTrainedClassification(self, model_path):
        """Load a saved model + scaler + feature columns from joblib."""
        saved = joblib.load(model_path)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.feature_cols = saved['feature_cols']
        print(f"✅ Loaded model from {model_path}")

    # -----------------------------
    # 3) Strategy B execution on new data
    # -----------------------------
    def F_RunStrategy(self, main_dataset, initial_portfolio=1000, window_days=7, model_path=None, lookback_days=7, train_size=35, n_splits=3, threshold_metric="accuracy", export_model_path=None):
        # Load model if path provided
        df = self.main_dataset.copy()

        # --- Signal generation ---
        signals_df = self.F_GenerateSignals(df)
        
        # --- Lookback features ---
        results_sum, results_count = [], []
        for i, row in signals_df.iterrows():
            sig = row["signal_date"]
            mask = (signals_df["signal_date"] >= (sig - pd.Timedelta(days=lookback_days))) & (signals_df["signal_date"] < sig)
            results_sum.append(signals_df.loc[mask, "entry_exec_ret"].sum(skipna=True))
            results_count.append(mask.sum())

        signals_df["sum_exec_ret_last7"] = results_sum
        signals_df["signals_last7d"] = results_count

        if model_path is not None:
            self.F_LoadTrainedClassification(model_path)
        else:
            df_pred, summary = self.F_ClassificationModel(signals_df=signals_df,min_train_size=train_size, n_splits=n_splits,
                                                   threshold_metric=threshold_metric,
                                                   export_model_path=export_model_path)


        # --- Probability prediction using the loaded or trained model --- 
        signals_df = self.F_ClassificationPredictions(signals_df)

        # --- Exit logic --- 
        signals_df = self.F_ExitStrategy(df, signals_df)

        # --- Allocation adjustment --- 
        signals_df = self.F_AllocationOverlay(signals_df, prob_col="y_proba", alloc_col="allocation_pct",
                                            window_days=window_days)

        

        # --- Cash backtest ---
        backtested = self.F_CashBacktest(signals_df, initial_portfolio)
        return backtested

    # -----------------------------
    # Helpers
    # -----------------------------
    def F_GenerateSignals(self, df):
        """
        Generate trading signals and add lookback features in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with market data and feature columns.
        lookback_days : int
            Window in days for lookback features.

        Returns
        -------
        signals_df : pd.DataFrame
            Signals with feature values and lookback columns.
        """
        
        signals = []
        for i in range(len(df) - 1):
            row = df.iloc[i]
            # Example signal logic
            if (row['VIX_Z_60'] > 2.2 and row['SP500_trend_5'] < 0 and
                row['VIX_Z_60_roc1'] <= 2.1 and row['VIX_Z_60_roc3'] <= 3.5):
                
                signal = {
                    "signal_idx": i,
                    "signal_date": df.at[i, "Date"],
                    "entry_idx": i + 1,
                    "entry_date": df.at[i + 1, "Date"],
                    "entry_open": df.at[i + 1, "SVXY_Open"],
                    "entry_exec_ret": df.at[i + 1, "SVXY_exec_ret"],
                }
                # Add only existing feature_cols values at signal date
                for f in self.feature_cols:
                    if f in df.columns:
                        signal[f] = df.at[i, f]

                signals.append(signal)

        signals_df = pd.DataFrame(signals).sort_values("signal_date").reset_index(drop=True)


        return signals_df

    def F_ExitStrategy(self, df, signals_df):
        exit_idx, exit_date, exit_close, log_ret, simple_ret = [], [], [], [], []
        n = len(df)

        for _, row in signals_df.iterrows():
            entry_idx, entry_open = int(row["entry_idx"]), float(row["entry_open"])
            e_idx, e_close = None, None

            for j in range(0, 8):  # up to 7 days
                k = entry_idx + j
                if k >= n:
                    break
                current_close = df.at[k, "SVXY_Close"]
                simple_ret_now = (current_close - entry_open) / entry_open
                if (df.at[k, "VIX_Z_60"] < 0.2 or 
                    simple_ret_now <= -0.1 or 
                    simple_ret_now >= 0.1 or 
                    j == 7):
                    e_idx, e_close = k, current_close
                    break

            if e_idx is None:
                e_idx = min(entry_idx + 7, n - 1)
                e_close = df.at[e_idx, "SVXY_Close"]

            # Store results
            exit_idx.append(e_idx)
            exit_date.append(df.at[e_idx, "Date"])
            exit_close.append(e_close)
            log_ret.append(np.log(e_close / entry_open))
            simple_ret.append(np.exp(np.log(e_close / entry_open)) - 1)

        # Add all columns to signals_df
        signals_df["exit_idx"] = exit_idx
        signals_df["exit_date"] = exit_date
        signals_df["exit_close"] = exit_close
        signals_df["log_ret"] = log_ret
        signals_df["simple_ret"] = simple_ret

        return signals_df


    def F_CashBacktest(self, signals_df, initial_portfolio=1000):
        """
        Locked-cash backtest using actual entry and exit prices.
        Uses allocation_pct_adj if available, otherwise falls back to allocation_pct.
        """
        signals_df = signals_df.copy().reset_index(drop=True)

        # Track active trades and cash
        active_trades = []
        cash_available = initial_portfolio

        invested_list, pnl_list, return_list, portfolio_after_exit_list = [], [], [], []

        for idx, row in signals_df.iterrows():
            # Release cash from trades that have exited by current entry date
            still_active = []
            for t in active_trades:
                if t['exit_date'] <= row['entry_date']:
                    cash_available += t['invested'] + t['pnl']
                else:
                    still_active.append(t)
            active_trades = still_active

            # Current total portfolio
            current_portfolio = cash_available + sum(t['invested'] + t['pnl'] for t in active_trades)

            # Allocation: prefer adj if available
            alloc_pct = row.get('allocation_pct_adj', row.get('allocation_pct', 100)) / 100
            target_invest = alloc_pct * current_portfolio
            invested = min(cash_available, target_invest)

            # PnL based on actual entry and exit prices
            pnl = invested * (row['exit_close'] - row['entry_open']) / row['entry_open'] if invested > 0 else 0.0

            # Lock capital until exit
            active_trades.append({'exit_date': row['exit_date'], 'invested': invested, 'pnl': pnl})
            cash_available -= invested

            # Record trade stats
            invested_list.append(invested)
            pnl_list.append(pnl)
            return_list.append((pnl / invested * 100) if invested > 0 else 0.0)
            portfolio_after_exit_list.append(cash_available + sum(t['invested'] + t['pnl'] for t in active_trades))

        signals_df['invested'] = invested_list
        signals_df['pnl'] = pnl_list
        signals_df['return'] = return_list
        signals_df['portfolio_after_exit'] = portfolio_after_exit_list

        return signals_df


# -------------------
# Function for streamlit caching
# -------------------
@st.cache_data
def F_LoadAllData(model_path: str):
    prep = C_Preprocessing(start_date="2020-01-01", vix_window=60)
    df = prep.F_ComputeFeatures()
    filtered = df[df['Date'] >= pd.Timestamp("2020-01-01")].copy().reset_index(drop=True)

    saved = joblib.load(model_path)
    model = saved['model']
    scaler = saved['scaler']
    feature_cols = saved['feature_cols']

    return filtered, model, scaler, feature_cols


# Functions for Performance.py
# -------------------
# KPI Calculation Function
# -------------------
def F_PerformanceMetrics(df: pd.DataFrame, initial_portfolio: bool) -> dict:
    """
    Compute key performance metrics using PnL and invested capital.
    """
    # Only count active trades
    active = df[df["invested"] > 0].copy()
    if active.empty:
        return {"No trades executed": np.nan}

    # --- Core metrics ---
    total_pnl = active["pnl"].sum()
    wins = active.loc[active["pnl"] > 0, "pnl"].sum()
    losses = -active.loc[active["pnl"] < 0, "pnl"].sum()
    profit_factor = wins / losses if losses != 0 else np.nan
    n_trades = len(active)
    win_rate = (active["pnl"] > 0).mean() * 100

    # --- Portfolio-level metrics ---
    portfolio = df["portfolio_after_exit"].replace(0, np.nan).dropna()
    total_return = (portfolio.iloc[-1] / initial_portfolio - 1) * 100

    cummax = portfolio.cummax()
    drawdown = (portfolio / cummax - 1) * 100
    max_drawdown = drawdown.min()

    return {
        "Total Return (%)": round(total_return, 2),
        "Profit Factor": round(profit_factor, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Number of Trades": int(n_trades),
        "Total PnL": round(total_pnl, 2)
    }

# -------------------
# Plot: Portfolio Cumulative Line Chart
# -------------------
def F_PlotCumulativePerformance(df: pd.DataFrame, dark_mode: bool = True, initial_portfolio: float = 1000):
    """
    Static cumulative portfolio line chart.
    Colors and background adjust to dark or light mode.
    """

    portfolio = df["portfolio_after_exit"].replace(0, np.nan).dropna()
    dates = pd.to_datetime(df["signal_date"][:len(portfolio)])

    # --- Theme ---
    if dark_mode:
        plt.style.use("dark_background")
        bg_color = "#0e1117"
        line_color = "#00C853"
        grid_color = "#444"
        text_color = "white"
    else:
        plt.style.use("default")
        bg_color = "#ffffff"
        line_color = "#00B050"
        grid_color = "#cccccc"
        text_color = "black"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, portfolio, color=line_color, linewidth=3)
    ax.set_facecolor(bg_color)

    # --- Title & labels ---
    ax.set_title(f"Cumulative Portfolio Value (Starting at ${initial_portfolio:,.0f})",
                 fontsize=18, fontweight="bold", color=text_color, pad=15)
    ax.set_xlabel("Date", fontsize=12, color=text_color)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12, color=text_color)

    # --- Grid ---
    ax.grid(True, linestyle="--", alpha=0.4, color=grid_color)

    # --- Date formatting ---
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Axis color adjustments
    ax.tick_params(axis="x", colors=text_color, rotation=45)
    ax.tick_params(axis="y", colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    plt.tight_layout()
    return fig


# -------------------
# Plot: Trade Return Bar Chart
# -------------------
def F_PlotReturns(df: pd.DataFrame, dark_mode: bool = True):
    """
    Static, high-contrast bar chart for trade returns.
    Positive trades = green, negative = red.
    Dates formatted as 'Jan 2021', 'Jul 2021', etc.
    """


    trades = df[df["return"] != 0].copy()
    if trades.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No trades to display", ha="center", va="center", fontsize=12)
        return fig

    # --- Theme colors ---
    if dark_mode:
        plt.style.use("dark_background")
        bg_color = "#0e1117"
        grid_color = "#444"
        pos_color = "#00C853"
        neg_color = "#FF3D00"
        text_color = "white"
    else:
        plt.style.use("default")
        bg_color = "#ffffff"
        grid_color = "#cccccc"
        pos_color = "#00B050"
        neg_color = "#D32F2F"
        text_color = "black"

    trades["color"] = trades["return"].apply(lambda x: pos_color if x > 0 else neg_color)
    trades["signal_date"] = pd.to_datetime(trades["signal_date"])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(trades["signal_date"], trades["return"], color=trades["color"], width=5)

    ax.set_facecolor(bg_color)
    ax.grid(True, axis="y", color=grid_color, linestyle="--", alpha=0.4)

    ax.set_title("Trade Returns (%)", fontsize=16, fontweight="bold", color=text_color, pad=15)
    ax.set_xlabel("Trade Date", fontsize=12, color=text_color)
    ax.set_ylabel("Return (%)", fontsize=12, color=text_color)

    # Format y-axis
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # --- Format x-axis as 'Jan 2021' etc. ---
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Axis color adjustments
    ax.tick_params(axis="x", colors=text_color, rotation=45)
    ax.tick_params(axis="y", colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    plt.tight_layout()
    return fig
