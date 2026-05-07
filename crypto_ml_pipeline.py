"""
╔══════════════════════════════════════════════════════════════════╗
║   CRYPTOCURRENCY SENTIMENT ANALYSIS & PRICE PREDICTION          ║
║   Full ML Pipeline: FinBERT + LSTM + Random Forest              ║
║   BITS Pilani — ML for EE Course Project                        ║
╚══════════════════════════════════════════════════════════════════╝

DATASETS REQUIRED (place in same folder as this script):
  - outputs/tweets_clean.parquet   ← your 3.7M processed tweets  (2021–2023)
  - BTC-USD__2014-2024_.csv        ← BTC OHLC prices             (2014–2024)
  - ETH-USD__2017-2024_.csv        ← ETH OHLC prices             (2017–2024)

DATE OVERLAP:
  Tweets span roughly 2021–2023. Price data covers far longer.
  The pipeline automatically trims price data to the tweet date window,
  giving ~2 years of aligned daily data (~500+ feature rows).

RUN:
  python crypto_ml_pipeline.py

OUTPUT:
  outputs/  → all plots, CSVs, model results
"""

import warnings
warnings.filterwarnings("ignore")

import os, gc, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, classification_report, confusion_matrix)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
})
COIN_COLORS = {"BTC": "#F7931A", "ETH": "#627EEA"}

print("=" * 60)
print("  CRYPTO SENTIMENT + PRICE PREDICTION PIPELINE")
print("=" * 60)


# ════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading datasets...")

# ── Load 3.7M parquet tweets ────────────────────────────────────
PARQUET_PATH = "outputs/tweets_clean.parquet"
if not Path(PARQUET_PATH).exists():
    raise FileNotFoundError(
        f"❌ {PARQUET_PATH} not found!\n"
        "Run 01_data_pipeline.py first to generate this file."
    )

print(f"  Loading tweets from {PARQUET_PATH} ...")
tweets_df = pd.read_parquet(PARQUET_PATH,
                            columns=["date", "text_clean", "crypto_label"])
tweets_df["date"] = pd.to_datetime(tweets_df["date"])
tweets_df["date_day"] = tweets_df["date"].dt.normalize()
print(f"  Tweets loaded    : {len(tweets_df):,}")
print(f"  Date range       : {tweets_df.date_day.min().date()} → {tweets_df.date_day.max().date()}")
print(f"  Unique days      : {tweets_df.date_day.nunique()}")
print(f"  Crypto labels    :\n{tweets_df.crypto_label.value_counts().to_string()}")

# ── Load price CSVs (full history 2014-2024 / 2017-2024) ────────
# File names match what Yahoo Finance exports
BTC_CSV = "BTC-USD__2014-2024_.csv"
ETH_CSV = "ETH-USD__2017-2024_.csv"

for fpath in [BTC_CSV, ETH_CSV]:
    if not Path(fpath).exists():
        raise FileNotFoundError(
            f"❌ {fpath} not found!\n"
            "Place the CSV in the same folder as this script."
        )

btc_df = pd.read_csv(BTC_CSV)
eth_df = pd.read_csv(ETH_CSV)

# Drop null rows (Yahoo Finance sometimes adds blank row at end)
btc_df = btc_df.dropna(subset=["Close"])
eth_df = eth_df.dropna(subset=["Close"])

btc_df["Date"] = pd.to_datetime(btc_df["Date"])
eth_df["Date"] = pd.to_datetime(eth_df["Date"])

print(f"\n  BTC price rows (full) : {len(btc_df)} ({btc_df.Date.min().date()} → {btc_df.Date.max().date()})")
print(f"  ETH price rows (full) : {len(eth_df)} ({eth_df.Date.min().date()} → {eth_df.Date.max().date()})")

# ── Trim price data to tweet date window ─────────────────────────
# Tweets only cover a subset of the full price history.
# We restrict prices to that window so the inner join gives max rows.
tweet_start = tweets_df.date_day.min()
tweet_end   = tweets_df.date_day.max()
print(f"\n  Tweet window          : {tweet_start.date()} → {tweet_end.date()}")

btc_df = btc_df[(btc_df.Date >= tweet_start) & (btc_df.Date <= tweet_end)].copy()
eth_df = eth_df[(eth_df.Date >= tweet_start) & (eth_df.Date <= tweet_end)].copy()

print(f"  BTC rows in window    : {len(btc_df)}")
print(f"  ETH rows in window    : {len(eth_df)}")


# ════════════════════════════════════════════════════════════════
# STEP 2: VADER SENTIMENT (fast, runs on full 3.7M)
# ════════════════════════════════════════════════════════════════
print("\n[STEP 2] Running VADER sentiment on 3.7M tweets...")

VADER_CACHE = OUT / "vader_daily_sentiment.csv"

if VADER_CACHE.exists():
    print(f"  Cache found → loading from {VADER_CACHE}")
    daily_sent = pd.read_csv(VADER_CACHE, parse_dates=["date_day"])
else:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # Crypto-enhanced lexicon
    CRYPTO_LEXICON = {
        "moon": 3.5, "mooning": 3.5, "moonshot": 3.5,
        "hodl": 2.5, "bullish": 3.0, "bearish": -3.0,
        "rekt": -3.5, "dump": -2.5, "dumping": -2.5,
        "pumping": 2.5, "fud": -2.5, "fomo": 1.5,
        "ath": 3.0, "dip": -1.5, "wagmi": 3.0,
        "ngmi": -3.0, "rug": -3.5, "rugged": -3.5,
        "lambo": 2.5, "bagholder": -2.5, "rally": 2.5,
        "crash": -3.0, "crashed": -3.0, "recovery": 2.0,
        "scam": -3.5, "hack": -3.0, "ban": -2.5,
        "adoption": 2.0, "bullrun": 3.5, "correction": -1.0,
        "safu": 2.5, "based": 2.0, "cope": -1.5,
    }
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(CRYPTO_LEXICON)
    print(f"  VADER loaded with {len(CRYPTO_LEXICON)} crypto terms")

    # Score all tweets in chunks to save memory
    CHUNK = 200_000
    records = []
    texts   = tweets_df["text_clean"].fillna("").values
    dates   = tweets_df["date_day"].values
    labels  = tweets_df["crypto_label"].values

    print(f"  Scoring {len(texts):,} tweets...")
    for i in range(0, len(texts), CHUNK):
        t_chunk = texts[i:i+CHUNK]
        d_chunk = dates[i:i+CHUNK]
        l_chunk = labels[i:i+CHUNK]
        for text, date, label in zip(t_chunk, d_chunk, l_chunk):
            s = sia.polarity_scores(str(text))
            c = s["compound"]
            records.append({
                "date_day":     date,
                "crypto_label": label,
                "compound":     c,
                "is_positive":  1 if c >= 0.05 else 0,
                "is_negative":  1 if c <= -0.05 else 0,
                "is_neutral":   1 if -0.05 < c < 0.05 else 0,
            })
        print(f"  Scored {min(i+CHUNK, len(texts)):,}/{len(texts):,}")
        gc.collect()

    scored_df = pd.DataFrame(records)

    # Aggregate daily
    daily_sent = scored_df.groupby("date_day").agg(
        avg_sentiment   = ("compound",     "mean"),
        sentiment_std   = ("compound",     "std"),
        tweet_volume    = ("compound",     "count"),
        pct_positive    = ("is_positive",  "mean"),
        pct_negative    = ("is_negative",  "mean"),
        pct_neutral     = ("is_neutral",   "mean"),
    ).reset_index()
    daily_sent["sentiment_std"] = daily_sent["sentiment_std"].fillna(0)
    daily_sent["date_day"] = pd.to_datetime(daily_sent["date_day"])

    daily_sent.to_csv(VADER_CACHE, index=False)
    print(f"  ✅ Saved daily sentiment → {VADER_CACHE}")
    del scored_df, records; gc.collect()

print(f"  Daily sentiment rows : {len(daily_sent)}")
print(f"  Date range           : {daily_sent.date_day.min().date()} → {daily_sent.date_day.max().date()}")
print(f"  Avg sentiment        : {daily_sent.avg_sentiment.mean():.4f}")


# ════════════════════════════════════════════════════════════════
# STEP 3: FINBERT ON SAMPLE (for model comparison)
# ════════════════════════════════════════════════════════════════
print("\n[STEP 3] FinBERT sentiment on 5k-tweet sample (for comparison)...")

FINBERT_CACHE = OUT / "finbert_sample_sentiment.csv"

if FINBERT_CACHE.exists():
    print(f"  Cache found → loading from {FINBERT_CACHE}")
    fb_sample = pd.read_csv(FINBERT_CACHE)
else:
    try:
        from transformers import pipeline as hf_pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        print(f"  Device: {'GPU' if device==0 else 'CPU'}")
        finbert = hf_pipeline("text-classification",
                              model="ProsusAI/finbert",
                              tokenizer="ProsusAI/finbert",
                              device=device, truncation=True,
                              max_length=512, top_k=None)

        sample = tweets_df.sample(5000, random_state=42)
        texts  = sample["text_clean"].fillna("").tolist()
        rows   = []
        for i in range(0, len(texts), 32):
            batch = [t[:512] for t in texts[i:i+32]]
            preds = finbert(batch)
            for pred in preds:
                sm = {p["label"].lower(): p["score"] for p in pred}
                label = max(sm, key=sm.get)
                rows.append({"finbert_label": label,
                             "finbert_score": sm.get("positive",0) - sm.get("negative",0)})
            if i % 1000 == 0:
                print(f"  FinBERT: {min(i+32, len(texts))}/{len(texts)}")

        fb_sample = pd.DataFrame(rows)
        fb_sample.to_csv(FINBERT_CACHE, index=False)
        print(f"  ✅ FinBERT sample saved")
    except Exception as e:
        print(f"  ⚠️  FinBERT skipped: {e}")
        fb_sample = pd.DataFrame({"finbert_label": [], "finbert_score": []})

print(f"  FinBERT sample size: {len(fb_sample)}")
if len(fb_sample) > 0:
    print(f"  Label distribution :\n{fb_sample.finbert_label.value_counts().to_string()}")


# ════════════════════════════════════════════════════════════════
# STEP 4: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════
print("\n[STEP 4] Engineering features...")

def build_features(price_df, sent_df, coin):
    price = price_df.rename(columns={"Date": "date_day"}).copy()
    price = price.sort_values("date_day").reset_index(drop=True)

    df = pd.merge(price, sent_df, on="date_day", how="left")
    sent_cols = ["avg_sentiment","tweet_volume","pct_positive","pct_negative",
             "sentiment_std","pct_neutral"]
    for col in sent_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method="ffill", limit=3).fillna(method="bfill", limit=3)
            df[col] = df[col].fillna(df[col].mean())
    print(f"  {coin}: {len(df)} days after merge (kept all price days)")

    # Price features
    df["daily_return"]     = df["Close"].pct_change()
    df["price_range"]      = (df["High"] - df["Low"]) / df["Close"]
    df["volume_change"]    = df["Volume"].pct_change()
    df["target_return"]    = df["Close"].shift(-1) / df["Close"] - 1
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    df["target_price"]     = df["Close"].shift(-1)

    # Lag features
    for lag in [1, 2, 3]:
        df[f"close_lag{lag}"]     = df["Close"].shift(lag)
        df[f"return_lag{lag}"]    = df["daily_return"].shift(lag)
        df[f"sentiment_lag{lag}"] = df["avg_sentiment"].shift(lag)
        df[f"volume_lag{lag}"]    = df["tweet_volume"].shift(lag)

    # Rolling averages
    for w in [3, 7, 14]:
        df[f"close_ma{w}"]     = df["Close"].rolling(w).mean()
        df[f"return_ma{w}"]    = df["daily_return"].rolling(w).mean()
        df[f"sent_ma{w}"]      = df["avg_sentiment"].rolling(w).mean()
        df[f"vol_ma{w}"]       = df["tweet_volume"].rolling(w).mean()

    # Volatility
    df["volatility_7d"]  = df["daily_return"].rolling(7).std()
    df["volatility_14d"] = df["daily_return"].rolling(14).std()

    # Sentiment momentum
    df["sentiment_change"]    = df["avg_sentiment"].diff()
    df["tweet_volume_change"] = df["tweet_volume"].pct_change()

    df = df.dropna().reset_index(drop=True)
    df["coin"] = coin
    print(f"  {coin}: {len(df)} rows after lag/rolling (dropped NaN)")
    return df

btc_features = build_features(btc_df, daily_sent, "BTC")
eth_features = build_features(eth_df, daily_sent, "ETH")

btc_features.to_csv(OUT / "btc_features.csv", index=False)
eth_features.to_csv(OUT / "eth_features.csv", index=False)
print(f"  ✅ Feature tables saved")
print(f"  BTC rows: {len(btc_features)} | ETH rows: {len(eth_features)}")


# ════════════════════════════════════════════════════════════════
# STEP 5: EDA PLOTS
# ════════════════════════════════════════════════════════════════
print("\n[STEP 5] Generating EDA plots...")

# Plot A: Price + Sentiment over time
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

for ax, df, coin in zip(axes[:2], [btc_features, eth_features], ["BTC","ETH"]):
    color = COIN_COLORS[coin]
    ax2 = ax.twinx()
    ax.plot(df["date_day"], df["Close"], color=color, lw=2, label=f"{coin} Price")
    ax2.plot(df["date_day"], df["avg_sentiment"], color="#2ecc71",
             lw=1.2, alpha=0.8, label="Sentiment")
    ax2.axhline(0, color="#aaa", lw=0.7, linestyle="--")
    ax2.fill_between(df["date_day"], 0, df["avg_sentiment"],
                     where=df["avg_sentiment"] >= 0, alpha=0.2, color="#2ecc71")
    ax2.fill_between(df["date_day"], 0, df["avg_sentiment"],
                     where=df["avg_sentiment"] < 0, alpha=0.2, color="#e74c3c")
    ax.set_ylabel(f"{coin} Price (USD)", color=color)
    ax2.set_ylabel("Avg Sentiment", color="#2ecc71")
    ax.set_title(f"{coin} Price vs Twitter Sentiment ({len(df)} trading days)")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

# Tweet volume
axes[2].bar(daily_sent["date_day"], daily_sent["tweet_volume"],
            color="#3498db", alpha=0.7, width=1)
axes[2].set_title(f"Daily Tweet Volume ({daily_sent.tweet_volume.sum():,.0f} total tweets)")
axes[2].set_ylabel("Tweets / Day")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

plt.tight_layout()
plt.savefig(OUT / "01_price_vs_sentiment.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_price_vs_sentiment.png")

# Plot B: Correlation heatmap (BTC)
feat_cols = ["Close","daily_return","avg_sentiment","tweet_volume",
             "pct_positive","pct_negative","volatility_7d",
             "sentiment_change","close_ma7","target_price"]
feat_cols = [c for c in feat_cols if c in btc_features.columns]
corr = btc_features[feat_cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, ax=ax, linewidths=0.5)
ax.set_title("BTC Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT / "02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_correlation_heatmap.png")

# Plot C: Sentiment distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
labels_s = ["positive","neutral","negative"]
colors_s = ["#2ecc71","#95a5a6","#e74c3c"]

# VADER labels
vals = [daily_sent["pct_positive"].mean()*100,
        daily_sent["pct_neutral"].mean()*100 if "pct_neutral" in daily_sent.columns else
        (1-daily_sent["pct_positive"].mean()-daily_sent["pct_negative"].mean())*100,
        daily_sent["pct_negative"].mean()*100]
bars = axes[0].bar(labels_s, vals, color=colors_s, alpha=0.85)
axes[0].set_title("VADER Sentiment Distribution\n(avg across all days)")
axes[0].set_ylabel("Percentage (%)")
for b, v in zip(bars, vals):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                 f"{v:.1f}%", ha="center", fontweight="bold")

# FinBERT if available
if len(fb_sample) > 0:
    vals2 = [fb_sample["finbert_label"].eq(l).mean()*100 for l in labels_s]
    bars2 = axes[1].bar(labels_s, vals2, color=colors_s, alpha=0.85)
    axes[1].set_title("FinBERT Sentiment Distribution\n(5k tweet sample)")
    axes[1].set_ylabel("Percentage (%)")
    for b, v in zip(bars2, vals2):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                     f"{v:.1f}%", ha="center", fontweight="bold")
else:
    axes[1].text(0.5, 0.5, "FinBERT not run", ha="center", va="center",
                 transform=axes[1].transAxes, fontsize=12)
    axes[1].set_title("FinBERT Distribution\n(not available)")

# Daily sentiment trend
axes[2].plot(daily_sent["date_day"], daily_sent["avg_sentiment"],
             color="#3498db", lw=1.5)
axes[2].axhline(0, color="red", lw=1, linestyle="--", label="Neutral (0)")
axes[2].fill_between(daily_sent["date_day"], 0, daily_sent["avg_sentiment"],
                     where=daily_sent["avg_sentiment"]>=0, alpha=0.2, color="#2ecc71")
axes[2].fill_between(daily_sent["date_day"], 0, daily_sent["avg_sentiment"],
                     where=daily_sent["avg_sentiment"]<0,  alpha=0.2, color="#e74c3c")
axes[2].set_title("Daily Avg Sentiment Trend")
axes[2].set_ylabel("VADER Compound Score")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
axes[2].legend()

plt.tight_layout()
plt.savefig(OUT / "03_sentiment_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_sentiment_distribution.png")


# ════════════════════════════════════════════════════════════════
# STEP 6: LSTM
# ════════════════════════════════════════════════════════════════
print("\n[STEP 6] Training LSTM models...")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
    print(f"  TensorFlow {tf.__version__} loaded")
except Exception:
    HAS_TF = False
    print("  ⚠️  TensorFlow not available — skipping LSTM")

LSTM_FEATURES = [
    "Close", "avg_sentiment", "tweet_volume", "pct_positive", "pct_negative",
    "close_ma7", "close_ma14", "sent_ma7", "sent_ma14",
    "daily_return", "volatility_7d", "sentiment_change",
    "return_lag1", "sentiment_lag1", "tweet_volume_change",
]
SEQ_LEN = 14   # 14-day lookback window

def make_sequences(df, feature_cols, target_col, seq_len):
    X, y = [], []
    vals = df[feature_cols].values
    tgt  = df[target_col].values
    for i in range(seq_len, len(df)):
        X.append(vals[i-seq_len:i])
        y.append(tgt[i])
    return np.array(X), np.array(y)

lstm_results = {}

for df, coin in [(btc_features, "BTC"), (eth_features, "ETH")]:
    print(f"\n  ── {coin} LSTM ──")
    feats = [f for f in LSTM_FEATURES if f in df.columns]
    print(f"    Features: {len(feats)} | Rows: {len(df)}")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    df_s = df.copy()
    df_s[feats] = scaler_X.fit_transform(df[feats])
    df_s["target_scaled"] = scaler_y.fit_transform(df[["target_price"]])

    X, y = make_sequences(df_s, feats, "target_scaled", SEQ_LEN)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    if not HAS_TF or len(X_train) < 10:
        if len(X_train) < 10:
            print(f"    ⚠️  Too few samples ({len(X_train)}) for LSTM")
        lstm_results[coin] = None
        continue

    model = Sequential([
        Input(shape=(SEQ_LEN, len(feats))),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=8),
    ]
    history = model.fit(X_train, y_train, validation_split=0.15,
                        epochs=150, batch_size=8, callbacks=callbacks, verbose=0)
    print(f"    Trained {len(history.history['loss'])} epochs")

    y_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"    RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")

    lstm_results[coin] = {
        "history": history, "y_true": y_true, "y_pred": y_pred,
        "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
    }

# LSTM plots
if HAS_TF and any(v is not None for v in lstm_results.values()):
    valid = {k: v for k, v in lstm_results.items() if v}
    fig, axes = plt.subplots(2, len(valid), figsize=(14, 10))
    if len(valid) == 1:
        axes = axes.reshape(2, 1)
    for col, (coin, res) in enumerate(valid.items()):
        axes[0][col].plot(res["history"].history["loss"],     label="Train")
        axes[0][col].plot(res["history"].history["val_loss"], label="Val")
        axes[0][col].set_title(f"{coin} LSTM Loss")
        axes[0][col].legend()

        axes[1][col].plot(res["y_true"], label="Actual", color=COIN_COLORS[coin], lw=2)
        axes[1][col].plot(res["y_pred"], label="Predicted", color="#e74c3c",
                          lw=1.5, linestyle="--")
        axes[1][col].set_title(f"{coin} LSTM — Actual vs Predicted\n"
                               f"RMSE={res['rmse']:.0f}  R²={res['r2']:.3f}  MAPE={res['mape']:.1f}%")
        axes[1][col].legend()
    plt.tight_layout()
    plt.savefig(OUT / "04_lstm_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: 04_lstm_results.png")


# ════════════════════════════════════════════════════════════════
# STEP 7: RANDOM FOREST
# ════════════════════════════════════════════════════════════════
print("\n[STEP 7] Training Random Forest models...")

RF_FEATURES = [
    "avg_sentiment", "tweet_volume", "pct_positive", "pct_negative",
    "sentiment_change", "tweet_volume_change",
    "return_lag1", "return_lag2", "return_lag3",
    "sentiment_lag1", "sentiment_lag2",
    "close_ma7", "close_ma14", "sent_ma7",
    "daily_return", "volatility_7d", "volatility_14d", "price_range",
]

rf_results = {}

for df, coin in [(btc_features, "BTC"), (eth_features, "ETH")]:
    print(f"\n  ── {coin} Random Forest ──")
    feats = [f for f in RF_FEATURES if f in df.columns]
    X = df[feats].values
    y_price = df["target_price"].values
    y_dir   = df["target_direction"].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    yp_train, yp_test = y_price[:split], y_price[split:]
    yd_train, yd_test = y_dir[:split],   y_dir[split:]
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    # Regressor
    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=10,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, yp_train)
    yp_pred = rf_reg.predict(X_test)
    rmse = math.sqrt(mean_squared_error(yp_test, yp_pred))
    mae  = mean_absolute_error(yp_test, yp_pred)
    r2   = r2_score(yp_test, yp_pred)
    mape = np.mean(np.abs((yp_test - yp_pred) / yp_test)) * 100
    print(f"    Regressor  → RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f} | MAPE: {mape:.1f}%")

    # Classifier
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                    min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, yd_train)
    yd_pred = rf_clf.predict(X_test)
    acc = (yd_pred == yd_test).mean()
    print(f"    Classifier → Direction Accuracy: {acc*100:.1f}%")
    print(classification_report(yd_test, yd_pred, target_names=["Down","Up"]))

    importances = pd.Series(rf_reg.feature_importances_, index=feats).sort_values(ascending=False)

    rf_results[coin] = {
        "rf_reg": rf_reg, "rf_clf": rf_clf,
        "yp_true": yp_test, "yp_pred": yp_pred,
        "yd_true": yd_test, "yd_pred": yd_pred,
        "rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "acc": acc,
        "importances": importances,
        "dates": df["date_day"].values[split:],
    }

# RF plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for col, (coin, res) in enumerate(rf_results.items()):
    axes[0][col].plot(res["yp_true"], label="Actual",    color=COIN_COLORS[coin], lw=2)
    axes[0][col].plot(res["yp_pred"], label="Predicted", color="#e74c3c", lw=1.5, linestyle="--")
    axes[0][col].set_title(f"{coin} RF — Price Prediction\n"
                           f"RMSE={res['rmse']:.0f}  R²={res['r2']:.3f}  MAPE={res['mape']:.1f}%")
    axes[0][col].set_xlabel("Test Day"); axes[0][col].set_ylabel("Price (USD)")
    axes[0][col].legend()

    imp = res["importances"].head(12)
    axes[1][col].barh(imp.index[::-1], imp.values[::-1], color=COIN_COLORS[coin], alpha=0.85)
    axes[1][col].set_title(f"{coin} RF — Top 12 Feature Importances")
    axes[1][col].set_xlabel("Importance")

plt.tight_layout()
plt.savefig(OUT / "05_rf_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved: 05_rf_results.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (coin, res) in zip(axes, rf_results.items()):
    cm = confusion_matrix(res["yd_true"], res["yd_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Down","Up"], yticklabels=["Down","Up"])
    ax.set_title(f"{coin} RF Direction — Confusion Matrix\nAccuracy: {res['acc']*100:.1f}%")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(OUT / "06_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 06_confusion_matrix.png")


# ════════════════════════════════════════════════════════════════
# STEP 8: SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n[STEP 8] Model comparison summary...")

rows = []
for coin in ["BTC","ETH"]:
    if lstm_results.get(coin):
        lr = lstm_results[coin]
        rows.append({"Coin":coin,"Model":"LSTM",
                     "RMSE":f"{lr['rmse']:.2f}","MAE":f"{lr['mae']:.2f}",
                     "R2":f"{lr['r2']:.4f}","MAPE":f"{lr['mape']:.1f}%",
                     "Dir_Acc":"N/A"})
    rr = rf_results[coin]
    rows.append({"Coin":coin,"Model":"RF Regressor",
                 "RMSE":f"{rr['rmse']:.2f}","MAE":f"{rr['mae']:.2f}",
                 "R2":f"{rr['r2']:.4f}","MAPE":f"{rr['mape']:.1f}%",
                 "Dir_Acc":"N/A"})
    rows.append({"Coin":coin,"Model":"RF Classifier",
                 "RMSE":"N/A","MAE":"N/A","R2":"N/A","MAPE":"N/A",
                 "Dir_Acc":f"{rr['acc']*100:.1f}%"})

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "model_comparison.csv", index=False)
print(summary.to_string(index=False))

# RMSE bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, coin in zip(axes, ["BTC","ETH"]):
    models_p, vals_p, cols_p = [], [], []
    if lstm_results.get(coin):
        models_p.append("LSTM"); vals_p.append(lstm_results[coin]["rmse"]); cols_p.append("#3498db")
    models_p.append("RF"); vals_p.append(rf_results[coin]["rmse"]); cols_p.append(COIN_COLORS[coin])
    bars = ax.bar(models_p, vals_p, color=cols_p, alpha=0.85)
    ax.set_title(f"{coin} — RMSE Comparison"); ax.set_ylabel("RMSE (USD)")
    for b, v in zip(bars, vals_p):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f"${v:.0f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "07_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 07_model_comparison.png")


# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PIPELINE COMPLETE")
print("="*60)
print(f"\n  Input data:")
print(f"    Price files     : BTC-USD__2014-2024_.csv, ETH-USD__2017-2024_.csv")
print(f"    Tweet window    : {tweet_start.date()} → {tweet_end.date()}")
print(f"    Tweets (parquet): {len(tweets_df):,}")
print(f"    Days with tweets: {tweets_df.date_day.nunique()}")
print(f"    Daily sent rows : {len(daily_sent)}")
print(f"    BTC feature rows: {len(btc_features)}")
print(f"    ETH feature rows: {len(eth_features)}")
print(f"\n  VADER Sentiment (daily avg):")
print(f"    Mean score  : {daily_sent.avg_sentiment.mean():.4f}")
print(f"    % Positive  : {daily_sent.pct_positive.mean()*100:.1f}%")
print(f"    % Negative  : {daily_sent.pct_negative.mean()*100:.1f}%")
print(f"\n  Model Results:")
print(summary.to_string(index=False))
print(f"\n  Output files:")
for f in sorted(OUT.glob("*")):
    print(f"    {f.name}  ({f.stat().st_size/1e3:.0f} KB)")
