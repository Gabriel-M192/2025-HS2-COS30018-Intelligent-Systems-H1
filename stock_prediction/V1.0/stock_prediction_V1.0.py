# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

# -------------------- Core and Utilities --------------------
from __future__ import annotations
import os
import json
import hashlib
import pickle
from math import sqrt
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Sequence

# -------------------- Data and Math --------------------
import numpy as np
import pandas as pd
import pandas_datareader as web

# -------------------- Visualization --------------------
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------- Machine Learning --------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# -------------------- Deep Learning (TensorFlow / Keras) --------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K

# -------------------- NLP / Sentiment (C7) --------------------
from transformers import pipeline

# -------------------- SHAP Analysis --------------------
import shap

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"

COMPANY = 'AAPL'          # Company name to read

TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo


import yfinance as yf


raw = yf.download(COMPANY, TRAIN_START, TRAIN_END, auto_adjust=True)



# -------------------------- Tasck C2 ----------------------------------------

def load_and_prepare(path: str,
                    target_col: str = "close",
                    date_col: str = "date",
                    feature_cols: list[str] | None = None,   # None = usa target como único feature
                    lookback: int = 60,
                    horizon: int = 1,
                    test_size: float = 0.2,
                    split_by_date: bool = True,
                    scaler_type: str = "minmax",             # "minmax" | "standard" | None
                    cache_path: str | None = None,
                    force_recompute: bool = False,
                    na_strategy: str = "ffill_bfill",        # "drop" |  "mean" | "ffill_bfill"
                    random_state: int = 42,
                    ):
    # 1) cache
    '''
    If exists a cache file (.npz) and not force_recompute, load it.
    If the arrays are processed charge from  the cache
    _to_py: convert numpy object arrays to python objects
    Try to load the scaler from a pickle file (.pkl)
    Try to load the data and scaler from cache without processing again
    '''
    if cache_path and (not force_recompute) and os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        def _to_py(obj):
            return obj.item() if isinstance(obj, np.ndarray) and obj.dtype == object else obj
        data["cols"] = _to_py(data.get("cols"))
        data["meta"] = _to_py(data.get("meta"))
        scaler = None
        scal_path = cache_path + ".scaler.pkl"
        if os.path.exists(scal_path):
            with open(scal_path, "rb") as f:
                scaler = pickle.load(f)
        data["scaler"] = scaler
        return data

    # 2) read
    '''
    Check if the extension is parquet or csv and change to minuscule if its necessary
    If is parquet read with pd.read_parquet else pd.read_csv for csv
    Standardize the columns names to minuscule and without spaces
    Change the date column to datetime from pandas if there are errors in the format put NaT
    Drop the rows with NaT in the date column and sort the dataframe by date column
    Reset the index of the dataframe
    '''
    df = _read_raw(path, date_col)

    # 3) select columns
    '''
    Normalize the target and feature colums parameters to minuscule (ex: "Close" -> "close")
    If any column is missing give an error message
    Return the feature columns normalized
    Relevant columns are transformed to numeric, if there are errors put NaN
    '''
    feature_cols = _resolve_features(df, target_col, feature_cols)

    num_cols = list(dict.fromkeys(feature_cols + [target_col]))
    df.loc[:, num_cols] = df.loc[:, num_cols].apply(pd.to_numeric, errors="coerce")

    # 4) NaNs
    '''
    Create a sub-dataframe with the feature columns and target column
    Use the strategy to handle NaNs
    Replace the original columns in the dataframe with the processed sub-dataframe
    Clean any remaining NaNs in the original dataframe
    '''
    cols_to_clean = feature_cols if target_col in feature_cols else (feature_cols + [target_col])
    df = _handle_nans(df, cols_to_clean, na_strategy)

    # 5) supervised
    '''
    Uses shift from pandas to make a supervised learning -n to predict the next value
    Eliminate rows with no values and create features
    with lookback count of previous rows to use as features
    If lookback is >= create a 3D array (samples, lookback, features) for the superviced samples
    If lookback is 0 or < create a 2D array (samples, features) for the superviced samples
    Return X and y as numpy arrays
    '''
    X, y = _build_supervised(df[feature_cols], df[target_col], lookback, horizon)

    # transform to float32
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # 6) split
    '''
    Charge data and proportion to split the data
    Confirm that the amount of data is enough to split
    If split_by_date is True, split the data by date, else random split
    Separate the data in train and test
    Return the train and test data and the cut index or the indexes of train and test
    '''
    X_train, X_test, y_train, y_test, cut_idx = _split_train_test(
        X, y, test_size, split_by_date, random_state
    )

    # 7) scale
    ''''
    Make the scaler by the type specified
    If None do nothing
    Obtain the number of features 2D or 3D
    If 3D reshape to 2D
    Fit the scaler with the training data
    Transform the training and test data and reshape to original shape
    '''
    scaler = _make_scaler(scaler_type)
    if scaler is not None:
        n_feat = X_train.shape[-1]
        X_train_2d = X_train.reshape(-1, n_feat)
        X_test_2d  = X_test.reshape(-1, n_feat)

        scaler.fit(X_train_2d)
        X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
        X_test  = scaler.transform(X_test_2d).reshape(X_test.shape)

    # 8) package
    '''
    Create a dictionary with all the data and metadata
    '''
    out = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
        "scaler":  scaler,
        "cols":    {"features": feature_cols, "target": target_col, "date_col": date_col},
        "meta": {
            "lookback": lookback, "horizon": horizon, "split_by_date": split_by_date,
            "test_size": test_size, "scaler_type": scaler_type, "cache_path": cache_path,
            "n_samples_raw": len(df), "n_samples_supervised": len(y),
            "train_end_index": cut_idx, "created_at": datetime.now(timezone.utc).isoformat(),
            "random_state": random_state
        }
    }

    # 9) cache
    '''
    If cache_path is specified save the data arrays in a .npz file
    Save the scaler in a pickle file (.pkl)
    Return the data dictionary
    '''
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

        np.savez(cache_path,
                 X_train=out["X_train"], y_train=out["y_train"],
                 X_test=out["X_test"],   y_test=out["y_test"],
                 cols=np.array(out["cols"], dtype=object),
                 meta=np.array(out["meta"], dtype=object))
        with open(cache_path + ".scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    return out


# --- sub-functions ---

def _read_raw(path, date_col):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = date_col.lower()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    # df.columns = [c.lower() for c in df.columns]
    return df

def _resolve_features(df, target_col, feature_cols):
    target_col = target_col.lower()
    if feature_cols is None:
        feature_cols = [target_col]
    feature_cols = [c.lower() for c in feature_cols]
    missing = [c for c in ([target_col] + feature_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return feature_cols

def _handle_nans(df, cols, strategy):
    df = df.copy()
    if strategy == "drop":
        return df.dropna(subset=cols).copy()
    sub = df[cols].copy()
    if strategy == "mean":
        for c in cols:
            if pd.api.types.is_numeric_dtype(sub[c]):
                sub[c] = sub[c].fillna(sub[c].mean())
            else:
                sub[c] = sub[c].ffill().bfill()
    elif strategy == "ffill_bfill":
        sub = sub.ffill().bfill()
    else:
        raise ValueError(f"na_strategy inválida: {strategy}")
    sub = sub.reindex(df.index)
    df.loc[:, cols] = sub.values
    return df.dropna(subset=cols)

def _build_supervised(X_df, y_ser, lookback, horizon):
    y_future = y_ser.shift(-horizon)
    df_all = pd.concat([X_df, y_future.rename("future")], axis=1).dropna()
    X_vals = df_all[X_df.columns].values
    y_vals = df_all["future"].values
    if lookback <= 0:
        return X_vals, y_vals
    X_seq, y_seq = [], []
    last_start = len(X_vals) - lookback
    for i in range(last_start + 1):
        end = i + lookback
        X_seq.append(X_vals[i:end, :])
        y_seq.append(y_vals[end - 1])  # last value in the lookback window
    return np.array(X_seq), np.array(y_seq)

def _split_train_test(X, y, test_size: float, split_by_date: bool, random_state: int):
    n = len(X)
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1).")
    if n < 2:
        raise ValueError("Need at least 2 samples to split.")

    if split_by_date:
        cut = int(round(n * (1 - test_size)))
        # minimum 1 sample in test and train
        cut = max(1, min(cut, n - 1))
        X_train, X_test = X[:cut], X[cut:]
        y_train, y_test = y[:cut], y[cut:]
        info = cut  # cut index
    else:
        # random split
        idx = np.arange(n)
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, shuffle=True, random_state=random_state
        )
        # indexes
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        info = {"train_idx": idx_train, "test_idx": idx_test}
    return X_train, X_test, y_train, y_test, info


def _make_scaler(kind):
    if kind is None:
        return None
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "standard":
        return StandardScaler()
    raise ValueError(f"scaler_type invalid: {kind}")



# Save to csv
df_csv = raw.reset_index().rename(columns={"Date": "date"})
csv_path = "cache/aapl_all.csv"
Path("cache").mkdir(exist_ok=True, parents=True)
df_csv.to_csv(csv_path, index=False)


# Parameters
LOOKBACK = 60
HORIZON  = 1
CACHE_NPZ = "cache/c2v1_aapl.npz"

data_pkg = load_and_prepare(
    path=csv_path,
    target_col="close",
    date_col="date",
    feature_cols=["close"],    # puedes meter ["open","high","low","close","volume"]
    lookback=LOOKBACK,
    horizon=HORIZON,
    test_size=0.2,
    split_by_date=True,
    scaler_type="minmax",
    cache_path=CACHE_NPZ,
    force_recompute=True,      # fuerza recalcular para evitar cache corrupto
    na_strategy="ffill_bfill",
    random_state=42
)

X_train = data_pkg["X_train"]
y_train = data_pkg["y_train"]
X_test  = data_pkg["X_test"]
y_test  = data_pkg["y_test"]

n_features = X_train.shape[-1]


# -------------------- y-scaler (target) --------------------
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
# -----------------------------------------------------------

# -------------------- Model  --------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOKBACK, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train_scaled, epochs=25, batch_size=32, validation_split=0.1, shuffle=False, verbose=1)
# ---------------------------------------------------------------------

# -------------------- Prediction ----------------
y_pred_scaled = model.predict(X_test, verbose=0).ravel()
pred_test = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
# ----------------------------------------------------------------------

# -------------------- Plot --------------------
plt.figure()
plt.plot(y_test, label="Actual", linewidth=2)
plt.plot(pred_test, label="Predicted", linewidth=2)
plt.title(f"{COMPANY} â€“ Test set")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
# -----------------------------------------------

# -------------------- Next-step prediction --------------------
last_window = X_test[-1:,:,:]
next_pred_scaled = model.predict(last_window, verbose=0).ravel()
next_pred = scaler_y.inverse_transform(next_pred_scaled.reshape(-1, 1)).ravel()[0]
print(f"Next-step prediction: {next_pred:.4f}")
# --------------------------------------------------------------

# -------------------- Task C3 --------------------
'''
Make a copy of the original dataframe
Standardize the columns to be numeric and clean common non-numeric characters
'''
def _coerce_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Convert to numeric, cleaning common non-numeric chars
    df = df.copy()
    for c in cols:
        if c in df.columns:
            # Remove common non-numeric characters ($, %, ,)
            df[c] = (df[c]
                     .astype(str)
                     .str.replace(r"[^\d\.\-eE]", "", regex=True)  # elimina $ , %
                     .replace({"": np.nan, ".": np.nan}))
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

'''
Prepare OHLC data for candlestick plotting 
'''
def prepare_ohlc(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Normalize common names
    rename_map = {"adj close": "close"}
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    needed = ["open", "high", "low", "close"]
    maybe_volume = "volume" if "volume" in df.columns else None

    # Convert to numeric, cleaning common non-numeric chars
    df = _coerce_numeric_cols(df, needed + ([maybe_volume] if maybe_volume else []))

    # Eliminate rows with NaNs in needed columns
    drop_cols = needed + ([maybe_volume] if maybe_volume else [])
    df = df.dropna(subset=[c for c in drop_cols if c in df.columns])

    # Only keep needed columns + date_col, sort by date_col
    keep = [date_col] + needed + ([maybe_volume] if maybe_volume else [])
    df = df[keep].sort_values(date_col).reset_index(drop=True)
    return df

'''
Take a dataframe with OHLC(V) data and resample to a different frequency
'''
def resample_ohlcv(df: pd.DataFrame, freq: str = "5D", date_col: str = "date") -> pd.DataFrame:
    have_volume = "volume" in df.columns
    agg_map = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if have_volume:
        agg_map["volume"] = "sum"
    g = (df.set_index(date_col)
           .resample(freq)
           .agg(agg_map)
           .dropna())
    return g.reset_index()

'''
Plot OHLC(V) data as candlesticks using mplfinance
'''
def plot_candles_mpf(df: pd.DataFrame, date_col: str = "date",
                     title: str = "Candles", mav=(20, 50), volume=True):
    d2 = df.copy()
    # Make sure columns are numeric
    numeric_cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in d2.columns else [])
    d2 = _coerce_numeric_cols(d2, numeric_cols)
    d2 = d2.dropna(subset=["open", "high", "low", "close"])  # mpf exige numéricos completos

    # mpf wait DatetimeIndex
    d2 = d2.set_index(date_col)
    # If date_col is not datetime, this will raise an error
    vol_flag = volume and ("volume" in d2.columns)
    mpf.plot(d2, type="candle", mav=mav, volume=vol_flag, title=title, style="yahoo")

'''
Plot OHLC(V) data as candlesticks using plotly
'''
def plot_candles_plotly(df, date_col="date", title="Candles (Plotly)", volume=True):
    d2 = df.copy()
    num = ["open","high","low","close"] + (["volume"] if "volume" in d2.columns else [])
    d2 = _coerce_numeric_cols(d2, num)
    d2[date_col] = pd.to_datetime(d2[date_col], errors="coerce")
    d2 = d2.dropna(subset=[date_col,"open","high","low","close"])

    has_vol = volume and ("volume" in d2.columns)
    if has_vol:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=d2[date_col], open=d2["open"], high=d2["high"],
                                     low=d2["low"], close=d2["close"], name="OHLC"), row=1, col=1)
        fig.add_trace(go.Bar(x=d2[date_col], y=d2["volume"], name="Volume", opacity=0.4), row=2, col=1)
    else:
        fig = go.Figure(data=[go.Candlestick(x=d2[date_col], open=d2["open"], high=d2["high"],
                                             low=d2["low"], close=d2["close"], name="OHLC")])
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, margin=dict(l=40,r=20,t=50,b=30))
    fig.show()

'''
Plot rolling boxplots of a value column over a specified window and step
'''
def plot_boxplot_rolling(df, date_col="date", value_col="close", window=5, step=None, title=None):
    d2 = df.copy()
    d2[date_col] = pd.to_datetime(d2[date_col], errors="coerce")
    d2 = d2.dropna(subset=[date_col, value_col]).sort_values(date_col).reset_index(drop=True)

    vals = d2[value_col].to_numpy()
    dates = d2[date_col]                  
    step = step or window                

    boxes, labels = [], []
    for start in range(0, len(vals)-window+1, step):
        end = start + window
        boxes.append(vals[start:end])
        labels.append(dates.iloc[end-1].strftime("%Y-%m-%d"))

    if not boxes:
        raise ValueError("No sufficient data.")

    plt.figure()
    plt.boxplot(boxes, showfliers=False)
    plt.xticks(ticks=range(1, len(labels)+1), labels=labels, rotation=45, ha="right")
    plt.title(title or f"Boxplot rolling of {value_col} (window={window})")
    plt.xlabel("windows (fecha fin)")
    plt.ylabel(value_col.capitalize())
    plt.tight_layout()
    plt.show()



# ------------------------------------------------------------------

# Charge and prepare data
ohlc_df = pd.read_csv("cache/aapl_all.csv")  # ajusta la ruta si usas otra
ohlc_df = prepare_ohlc(ohlc_df, date_col="date")

# Daily candles
plot_candles_mpf(ohlc_df, title="AAPL – Daily Candles (mplfinance)")

# 5-day candles
ohlc_5d = resample_ohlcv(ohlc_df, freq="5D")
plot_candles_mpf(ohlc_5d, title="AAPL – 5D Candles (mplfinance)")

# Plotly candles Plotly
plot_candles_plotly(ohlc_df, title="AAPL – Daily (Plotly)")

# 5-day candles Plotly
plot_candles_plotly(resample_ohlcv(ohlc_df, "5D"), title="AAPL – 5D (Plotly)")

# Rolling boxplots
plot_boxplot_rolling(ohlc_df, value_col="close", window=5, step=5,
                     title="AAPL – Boxplot rolling (5 días)")


# -------------------- Task C4 --------------------


# 1. BUILD MODEL FUNCTION

def build_dl_model(
    input_shape,
    layer_type: str = "LSTM",           # Type of recurrent layer: LSTM | GRU | RNN
    units: tuple[int, ...] = (50, 50),  # Units per layer, e.g. (64,), (64,32)
    dropout: float = 0.2,               # Dropout regularization
    bidirectional: bool = False,        # Use Bidirectional wrapper
    dense_units: int = 1,               # Output neurons (usually 1)
    loss: str = "mse",
    optimizer=None                      # Default: Adam(1e-3)
):
    """
    Build and compile a sequential deep learning model
    using LSTM, GRU or SimpleRNN layers.
    """

    # Map layer type to corresponding Keras class
    layer_type = layer_type.upper()
    layer_map = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}
    if layer_type not in layer_map:
        raise ValueError(f"Invalid layer_type: {layer_type}. Use LSTM/GRU/RNN.")
    RNNLayer = layer_map[layer_type]

    # Initialize sequential model with input shape
    model = Sequential([Input(shape=input_shape)])

    # Add recurrent layers
    for i, u in enumerate(units):
        return_sequences = i < len(units) - 1
        rnn = RNNLayer(u, return_sequences=return_sequences)
        model.add(Bidirectional(rnn) if bidirectional else rnn)
        if dropout and dropout > 0:
            model.add(Dropout(dropout))

    # Add final dense output layer
    model.add(Dense(dense_units))

    # Compile model
    if optimizer is None:
        from keras.optimizers import Adam
        optimizer = Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer, loss=loss)
    return model



# 2. CALLBACKS CONFIGURATION

def make_callbacks(out_dir: str = "artifacts",
                   patience_es: int = 10,
                   patience_rlr: int = 5):
    """
    Create standard callbacks:
    - EarlyStopping: stop when validation loss stops improving.
    - ReduceLROnPlateau: lower LR when plateau detected.
    - ModelCheckpoint: save best model automatically.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(out_dir) / "best_model.keras")

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience_es, restore_best_weights=True
    )
    rlr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=patience_rlr, min_lr=1e-6
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_loss", save_best_only=True
    )
    return [es, rlr, ckpt]


# 3. TRAINING FUNCTION

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=32, callbacks=None):
    """
    Train the model using training and validation sets.
    Returns: Keras History object.
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks or [],
        verbose=1
    )
    return history



# 4. SAVE / LOAD ARTIFACTS

def save_artifacts(model: tf.keras.Model,
                   scaler_y: MinMaxScaler,
                   meta: dict,
                   out_dir: str = "artifacts",
                   model_name: str = "best_model.keras"):
    """
    Save model, output scaler, and metadata for reproducibility.
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    # Model (if not already saved by checkpoint)
    model_path = p / model_name
    if not model_path.exists():
        model.save(model_path)

    # Output scaler (y)
    with open(p / "scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    # Metadata
    with open(p / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_artifacts(out_dir: str = "artifacts",
                   model_name: str = "best_model.keras"):
    """
    Load model, scaler_y, and metadata if available.
    """
    p = Path(out_dir)
    model = tf.keras.models.load_model(p / model_name) if (p / model_name).exists() else None
    scaler_y = pickle.load(open(p / "scaler_y.pkl", "rb")) if (p / "scaler_y.pkl").exists() else None
    meta = json.load(open(p / "meta.json", "r", encoding="utf-8")) if (p / "meta.json").exists() else {}
    return {"model": model, "scaler_y": scaler_y, "meta": meta}



# 5. EVALUATION

def evaluate_on_test(model, X_test, y_test, scaler_y) -> dict:
    """
    Evaluate model performance on test data (unscaled y).
    Returns RMSE, MAE, MAPE and predictions.
    """
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    eps = 1e-8
    mape = float(np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + eps))) * 100.0)

    return {"rmse": rmse, "mae": mae, "mape": mape, "y_pred": y_pred}



# 6. PLOTTING UTILITIES

def plot_learning_curves(history, title="Learning Curves", out_path=None):
    """
    Plot training and validation loss curves.
    """
    h = history.history
    plt.figure()
    plt.plot(h["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in h:
        plt.plot(h["val_loss"], label="Validation Loss", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_test_predictions(y_test, y_pred, title="Test Set", out_path=None):
    """
    Plot actual vs predicted test prices.
    """
    plt.figure()
    plt.plot(y_test, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



# 7. SINGLE STEP PREDICTION

def predict_next_step(model, X_last_window, scaler_y) -> float:
    """
    Predict and inverse-scale the next step (one-step forecast).
    """
    pred_scaled = model.predict(X_last_window, verbose=0).ravel()[0]
    return float(scaler_y.inverse_transform([[pred_scaled]]).ravel()[0])



# 8. EXPERIMENT MANAGEMENT

def _slug(d: dict) -> str:
    """Create short hash identifier for config dictionary."""
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def run_experiment_once(cfg, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, base_out="experiments"):
    """
    Run one model configuration experiment end-to-end:
    build → train → evaluate → save metrics and plots.
    """
    os.makedirs(base_out, exist_ok=True)
    cfg_id = _slug(cfg)
    out_dir = os.path.join(base_out, f"{cfg['layer_type']}_{cfg_id}")
    os.makedirs(out_dir, exist_ok=True)

    # Build model
    model = build_dl_model(
        input_shape=X_train.shape[1:],
        layer_type=cfg["layer_type"],
        units=tuple(cfg.get("units", (50, 50))),
        dropout=cfg.get("dropout", 0.2),
        bidirectional=cfg.get("bidirectional", False)
    )

    # Train model
    cbs = make_callbacks(out_dir)
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=cfg.get("epochs", 100),
        batch_size=cfg.get("batch_size", 32),
        callbacks=cbs
    )

    # Evaluate
    eval_dict = evaluate_on_test(model, X_test, y_test, scaler_y)

    # Save plots
    plot_learning_curves(history, title=f"{cfg['layer_type']} – Learning", out_path=os.path.join(out_dir, "learning.png"))
    plot_test_predictions(y_test, eval_dict["y_pred"], title=f"{cfg['layer_type']} – Test", out_path=os.path.join(out_dir, "test.png"))

    # Predict next step
    next_step = predict_next_step(model, X_test[-1:], scaler_y)

    # Save metadata
    meta = {
        "config": cfg,
        "metrics": {**eval_dict, "next_step": next_step}
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "id": cfg_id,
        **cfg,
        **eval_dict,
        "out_dir": out_dir
    }


def temporal_val_split(X_train, y_train, val_ratio=0.1):
    """
    Split last X% of training data for validation (preserves temporal order).
    """
    n = len(X_train)
    n_val = max(1, int(n * val_ratio))
    return X_train[:-n_val], y_train[:-n_val], X_train[-n_val:], y_train[-n_val:]


def run_experiments(configs, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, base_out="experiments"):
    """
    Run multiple configurations and compare metrics.
    Returns sorted DataFrame by RMSE.
    """
    results = [run_experiment_once(cfg, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, base_out)
               for cfg in configs]
    df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    df.to_csv(os.path.join(base_out, "results.csv"), index=False)
    return df


# 9. EXECUTION EXAMPLE
# 1) Split training data into train/validation preserving time order
# 2) Scale target y using MinMaxScaler
# 3) Run several model configurations and compare
# 4) Load best model and predict next price

# Example configuration list (for experiments)
configs = [
    {"layer_type": "LSTM", "units": (50,), "dropout": 0.2, "epochs": 80, "batch_size": 32},
    {"layer_type": "LSTM", "units": (64, 32), "dropout": 0.2, "epochs": 100, "batch_size": 32},
    {"layer_type": "GRU",  "units": (64, 64), "dropout": 0.2, "epochs": 100, "batch_size": 32},
    {"layer_type": "RNN",  "units": (64, 32, 16), "dropout": 0.2, "epochs": 120, "batch_size": 64},
    {"layer_type": "LSTM", "units": (128,), "dropout": 0.3, "epochs": 100, "batch_size": 32},
    {"layer_type": "GRU",  "units": (96, 48), "dropout": 0.2, "epochs": 100, "batch_size": 32, "bidirectional": True},
]


# -------------------- Task C5 --------------------
"""
C5 – Multi-step forecasting, walk-forward CV, baselines & prediction intervals (MC-Dropout)

Summary:
- build_supervised_multistep: builds (X, Y) with Y shaped (n, horizon)
- build_supervised_multistep_delta: same but targets as Δ from last observed y_t
- multistep_metrics / multistep_metrics_by_horizon: error metrics (global & per-h)
- build_model_from_config: Keras RNN (LSTM/GRU/RNN/BiLSTM) from a config dict
- timeseries_splits: simple expanding-window splits (TimeSeriesSplit-like)
- baselines: naive_last, moving_average, drift
- fit_predict_multistep / walk_forward_eval: training & walk-forward evaluation
- mc_dropout_predict: MC-Dropout intervals
- plot_multistep / plot_with_intervals / plot_walkforward: plotting helpers
- grid_search_timeseries: lightweight temporal CV over a list of configs
- run_c5_example: end-to-end pipeline (simple features + DELTAS)
"""



# ---------- 1) Supervised multistep ----------

def build_supervised_multistep(X_df: pd.DataFrame,
                               y_ser: pd.Series,
                               lookback: int,
                               horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) windows for multi-step forecasting.

    Args:
        X_df: Feature DataFrame ordered by time (rows are timesteps).
        y_ser: Target Series aligned with X_df (same length).
        lookback: Number of past timesteps in each input window.
        horizon: Number of future steps to predict per window.

    Returns:
        X: Array of shape (n_samples, lookback, n_features).
        Y: Array of shape (n_samples, horizon) with future targets.

    Notes:
        Creates all consecutive windows so that each Y contains y[t+1..t+horizon].
    """
    assert lookback > 0 and horizon > 0
    X_vals = X_df.values
    y_vals = y_ser.values
    X_seq, Y_seq = [], []
    last_start = len(X_vals) - (lookback + horizon) + 1
    for i in range(last_start):
        X_seq.append(X_vals[i:i+lookback, :])
        Y_seq.append(y_vals[i+lookback:i+lookback+horizon])
    return np.array(X_seq, dtype=np.float32), np.array(Y_seq, dtype=np.float32)


def build_supervised_multistep_delta(X_df_raw: pd.DataFrame,
                                     y_ser_raw: pd.Series,
                                     lookback: int,
                                     horizon: int):
    """
    Build multi-step windows and also return targets as deltas from last observed value.

    Args:
        X_df_raw: Raw (unscaled) feature DataFrame.
        y_ser_raw: Raw (unscaled) target Series (e.g., closing price).
        lookback: Past window size.
        horizon: Future steps to predict.

    Returns:
        X_raw: (n, lookback, n_features) unscaled input windows.
        Y_raw: (n, horizon) future raw target values.
        Y_delta: (n, horizon) future deltas, y_{t+h} - y_t.
        last_close: (n,) last observed y_t of each window.
 
    """
    Xv = X_df_raw.values
    yv = y_ser_raw.values
    X_seq, Y_raw_seq, Y_delta_seq, last_seq = [], [], [], []
    last_start = len(Xv) - (lookback + horizon) + 1
    for i in range(last_start):
        xblk = Xv[i:i+lookback, :]
        y_fut = yv[i+lookback:i+lookback+horizon]
        y_last = yv[i+lookback-1]
        X_seq.append(xblk)
        Y_raw_seq.append(y_fut)
        Y_delta_seq.append(y_fut - y_last)
        last_seq.append(y_last)
    return (np.array(X_seq, dtype=np.float32),
            np.array(Y_raw_seq, dtype=np.float32),
            np.array(Y_delta_seq, dtype=np.float32),
            np.array(last_seq, dtype=np.float32))


# ---------- 2) Metrics ----------

def multistep_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Aggregate multi-step error metrics.

    Args:
        y_true: (n, horizon) ground truth.
        y_pred: (n, horizon) predictions.

    Returns:
        dict with overall RMSE, MAE, and MAPE across all horizons and samples.

    Caveats:
        MAPE uses |y| in the denominator with an epsilon to avoid division by zero.
    """
    assert y_true.shape == y_pred.shape
    eps = 1e-8
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def multistep_metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute RMSE/MAE per horizon step.

    Args:
        y_true: (n, horizon) ground truth.
        y_pred: (n, horizon) predictions.

    Returns:
        List[dict]: [{"h": 1..H, "rmse": ..., "mae": ...}, ...]
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    H = y_true.shape[1]
    out = []
    for h in range(H):
        rmse = float(np.sqrt(np.mean((y_true[:, h] - y_pred[:, h])**2)))
        mae  = float(np.mean(np.abs(y_true[:, h] - y_pred[:, h])))
        out.append({"h": h+1, "rmse": rmse, "mae": mae})
    return out


# ---------- 3) Model builder from config ----------

def build_model_from_config(cfg, lookback: int, n_features: int, horizon: int):
    """
    Build and compile an RNN (LSTM/GRU/RNN/BiLSTM) from a config dict.

    Expected keys in cfg:
        - "model" | "layer_type": {"lstm","gru","rnn","bilstm"} (default "lstm")
        - "units": int | sequence[int] (e.g., 128 or (128,64))
        - "layers": if units is int, how many repeated layers (default 1)
        - "dropout": float, "recurrent_dropout": float
        - "bidirectional": bool (or model == "bilstm")
        - "dense_units": optional int for an intermediate Dense
        - "activation": activation for the intermediate Dense (default "relu")
        - "lr": learning rate for Adam (default 1e-3)

    Returns:
        Compiled tf.keras.Model that outputs a vector of length `horizon`.

    Notes:
        Uses Huber loss for robustness against outliers in regression.
    """

    model_type = (cfg.get("model") or cfg.get("layer_type") or "lstm").lower()
    p_drop = float(cfg.get("dropout", 0.0))
    rec_drop = float(cfg.get("recurrent_dropout", 0.0))
    lr = float(cfg.get("lr", 1e-3))
    use_bidir = bool(cfg.get("bidirectional", False) or model_type == "bilstm")

    units_cfg = cfg.get("units", 64)
    if isinstance(units_cfg, (list, tuple)):
        units_list = [int(u) for u in units_cfg]
    else:
        n_layers = int(cfg.get("layers", 1))
        units_list = [int(units_cfg)] * max(1, n_layers)

    RNN = {"lstm": LSTM, "gru": GRU, "rnn": SimpleRNN, "bilstm": LSTM}.get(model_type, LSTM)

    model = Sequential()
    model.add(tf.keras.Input(shape=(lookback, n_features)))  # avoid input_shape warning

    for i, units in enumerate(units_list):
        return_seq = (i < len(units_list) - 1)
        rnn_layer = RNN(units=units, return_sequences=return_seq, recurrent_dropout=rec_drop)
        if use_bidir:
            rnn_layer = Bidirectional(rnn_layer)
        model.add(rnn_layer)
        if p_drop > 0:
            model.add(Dropout(p_drop))

    if "dense_units" in cfg and int(cfg["dense_units"]) > 0:
        model.add(Dense(int(cfg["dense_units"]), activation=cfg.get("activation", "relu")))

    model.add(Dense(horizon, activation="linear"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=Huber(delta=1.0),
                  metrics=["mae"])
    return model


# ---------- 4) Time-series splits ----------

@dataclass
class TSSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def timeseries_splits(n: int, n_splits: int = 3, test_size: int | None = None, test_ratio: float = 0.2) -> list[TSSplit]:
    """
    Create simple expanding-window time-series splits.

    Args:
        n: Total number of samples.
        n_splits: Number of folds.
        test_size: Fixed test size per fold; if None, use ratio.
        test_ratio: Fraction for test size if test_size is None.

    Returns:
        List[TSSplit]: each with [0:train_end) as train and [test_start:test_end) as test.

    Strategy:
        - Train grows linearly each fold.
        - Test length is constant across folds.
    """
    if test_size is None:
        test_size = max(1, int(round(n * test_ratio)))
    splits = []
    step = (n - test_size) // (n_splits + 1)
    step = max(1, step)
    for k in range(1, n_splits + 1):
        train_end = step * k
        test_start = train_end
        test_end = min(n, test_start + test_size)
        if test_end - test_start < 1:
            break
        splits.append(TSSplit(0, train_end, test_start, test_end))
    return splits


# ---------- 5) Baselines ----------

def baseline_naive_last(y_series: np.ndarray, lookback: int, horizon: int, idxs: np.ndarray) -> np.ndarray:
    """
    Naive baseline: repeat last observed value (y_t) for all future steps.

    Args:
        y_series: Full 1D target series.
        lookback: Window length to locate y_t.
        horizon: Forecast horizon.
        idxs: Indices of window starts to evaluate.

    Returns:
        (len(idxs), horizon) predictions constant per row.
    """
    preds = []
    for i in idxs:
        last_val = y_series[i + lookback - 1]
        preds.append(np.full((horizon,), last_val, dtype=np.float32))
    return np.vstack(preds)


def baseline_moving_average(y_series: np.ndarray, lookback: int, horizon: int, idxs: np.ndarray, win: int = 5) -> np.ndarray:
    """
    Moving-average baseline: forecast the mean of the last `win` values.

    Args:
        y_series: Full 1D target series.
        lookback: Window length to align indexes.
        horizon: Forecast horizon.
        idxs: Indices of window starts to evaluate.
        win: MA window inside the lookback (<= lookback).

    Returns:
        (len(idxs), horizon) predictions equal to that mean per row.
    """
    preds = []
    for i in idxs:
        start = max(0, i + lookback - win)
        mean_val = float(np.mean(y_series[start:i+lookback]))
        preds.append(np.full((horizon,), mean_val, dtype=np.float32))
    return np.vstack(preds)


def baseline_drift(y_series: np.ndarray, lookback: int, horizon: int, idxs: np.ndarray) -> np.ndarray:
    """
    Drift baseline: linear extrapolation using the slope across the lookback.

    Args:
        y_series: Full 1D target series.
        lookback: Past window size to estimate slope.
        horizon: Forecast horizon.
        idxs: Indices of window starts to evaluate.

    Returns:
        (len(idxs), horizon) linearly increasing/decreasing forecasts.
    """
    preds = []
    for i in idxs:
        y0 = float(y_series[i]); y1 = float(y_series[i + lookback - 1])
        slope = (y1 - y0) / max(1, (lookback - 1))
        seq = np.array([y1 + slope * (h+1) for h in range(horizon)], dtype=np.float32)
        preds.append(seq)
    return np.vstack(preds)


# ---------- 6) Training & evaluation ----------

def fit_predict_multistep(build_model_fn,
                          X_train, Y_train, X_test,
                          epochs=10, batch_size=32, callbacks=None, verbose=0):
    """
    Train a compiled model and predict on test windows (multi-step output).

    Args:
        build_model_fn: Zero-arg function returning a compiled Keras model.
        X_train, Y_train: Training arrays; Y must be (n, horizon).
        X_test: Test inputs to predict.
        epochs, batch_size: Training parameters.
        callbacks: Optional Keras callbacks.
        verbose: Keras verbosity.

    Returns:
        (Y_hat, history):
            Y_hat: (len(X_test), horizon) predictions.
            history: Keras History of the training on ds_train.
    """

    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    X_test  = np.asarray(X_test,  dtype=np.float32)

    ds_train = (tf.data.Dataset
                    .from_tensor_slices((X_train, Y_train))
                    .cache()      # reduces retracing
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))

    ds_pred  = (tf.data.Dataset
                    .from_tensor_slices(X_test)
                    .batch(batch_size, drop_remainder=False)
                    .prefetch(tf.data.AUTOTUNE))

    model = build_model_fn()  # already compiled

    cbs = list(callbacks or [])
    if not any(cb.__class__.__name__ == "EarlyStopping" for cb in cbs):
        cbs.append(EarlyStopping(monitor="loss", patience=2, restore_best_weights=True))

    history = model.fit(ds_train, epochs=epochs, verbose=verbose, callbacks=cbs, shuffle=False)
    Y_hat = model.predict(ds_pred, verbose=0)

    K.clear_session()
    return Y_hat, history


def walk_forward_eval(X, Y, build_model_fn,
                      n_splits=3, test_ratio=0.2,
                      epochs=10, batch_size=32, callbacks=None, verbose=0):
    """
    Evaluate a model with expanding-window walk-forward CV.

    Args:
        X, Y: Full dataset windows (X: (n, L, F), Y: (n, H)).
        build_model_fn: Zero-arg builder returning compiled model.
        n_splits: Number of folds.
        test_ratio: Fraction of n used as test per fold.
        epochs, batch_size, callbacks, verbose: Training params.

    Returns:
        List[dict]: per-fold {"fold", "mae", "rmse", "n_train", "n_test"} using
        flattened (n, H) to compute RMSE/MAE across horizons.

    Notes:
        This function re-trains from scratch per fold.
    """

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    results = []
    n = len(X)
    split_sizes = np.linspace(int(n * (1 - test_ratio) / (n_splits + 1)),
                              int(n * (1 - test_ratio)), num=n_splits, dtype=int)

    for i, train_end in enumerate(split_sizes, start=1):
        test_len = int(n * test_ratio)
        X_tr, Y_tr = X[:train_end], Y[:train_end]
        X_te, Y_te = X[train_end: train_end + test_len], Y[train_end: train_end + test_len]
        if len(X_te) < 1:
            break

        Y_hat, _ = fit_predict_multistep(build_model_fn, X_tr, Y_tr, X_te,
                                         epochs=epochs, batch_size=batch_size,
                                         callbacks=callbacks, verbose=verbose)

        mae = mean_absolute_error(Y_te.reshape(len(Y_te), -1), Y_hat.reshape(len(Y_hat), -1))
        rmse = float(np.sqrt(mean_squared_error(Y_te.reshape(len(Y_te), -1),
                                                Y_hat.reshape(len(Y_hat), -1))))
        results.append({"fold": i, "mae": float(mae), "rmse": rmse,
                        "n_train": int(len(X_tr)), "n_test": int(len(X_te))})
        K.clear_session()
    return results


# ---------- 7) MC-Dropout ----------

def mc_dropout_predict(model, X: np.ndarray, T: int = 50, q_low=5, q_high=95):
    """
    Monte Carlo Dropout prediction intervals.

    Args:
        model: Trained Keras model with Dropout layers.
        X: Inputs to predict on.
        T: Number of stochastic forward passes (samples).
        q_low, q_high: Lower/upper percentiles for intervals.

    Returns:
        dict with:
          "mean": (n, H) Monte Carlo mean,
          "low":  (n, H) q_low percentile,
          "high": (n, H) q_high percentile.

    Caution:
        Ensure dropout is active at inference via `training=True`.
    """
    preds = []
    X = np.asarray(X, dtype=np.float32)
    for _ in range(T):
        y = model(X, training=True).numpy()
        preds.append(y)
    P = np.stack(preds, axis=0)
    mean = P.mean(axis=0)
    low  = np.percentile(P, q_low, axis=0)
    high = np.percentile(P, q_high, axis=0)
    return {"mean": mean, "low": low, "high": high}


# ---------- 8) Plots ----------

def plot_multistep(y_true: np.ndarray, y_pred: np.ndarray, title="Multi-step Test"):
    """
    Plot average true vs predicted trajectory across the horizon.

    Args:
        y_true, y_pred: (n, H) arrays.
        title: Matplotlib title.
    """
    plt.figure()
    plt.plot(y_true.mean(axis=0), label="Actual (avg)", linewidth=2)
    plt.plot(y_pred.mean(axis=0), label="Predicted (avg)", linewidth=2)
    plt.title(title); plt.xlabel("Horizon step"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_with_intervals(y_true: np.ndarray, mean_pred: np.ndarray, low: np.ndarray, high: np.ndarray,
                        sample_idx: int = -1, title="Forecast with 90% PI"):
    """
    Plot one sample's multi-step forecast with prediction intervals.

    Args:
        y_true: (n, H) true values.
        mean_pred: (n, H) MC mean predictions.
        low, high: (n, H) lower/upper bands (percentiles).
        sample_idx: Which sample to plot (supports negative indexing).
        title: Matplotlib title.
    """
    if sample_idx < 0:
        sample_idx = len(y_true) + sample_idx
    yt = y_true[sample_idx]; mp = mean_pred[sample_idx]; lo = low[sample_idx]; hi = high[sample_idx]
    h = np.arange(len(yt))
    plt.figure()
    plt.plot(h, yt, label="Actual", linewidth=2)
    plt.plot(h, mp, label="Pred mean", linewidth=2)
    plt.fill_between(h, lo, hi, alpha=0.25, label="PI")
    plt.title(title); plt.xlabel("H step"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_walkforward(rmses: Sequence[float], title="Walk-forward RMSE (grid means)"):
    """
    Plot RMSE per configuration (e.g., results of a grid search).

    Args:
        rmses: Sequence of mean RMSE values (one per config).
        title: Matplotlib title.
    """
    xs = np.arange(len(rmses))
    plt.figure()
    plt.plot(xs, rmses, marker="o")
    plt.title(title); plt.xlabel("Config idx"); plt.ylabel("RMSE")
    plt.tight_layout(); plt.show()


# ---------- 9) Grid search with temporal CV ----------

def grid_search_timeseries(configs, X, Y, n_splits=3, test_ratio=0.2,
                           base_epochs=5, base_batch=64, verbose=0):
    """
    Lightweight hyperparameter search with temporal cross-validation.

    Args:
        configs: Iterable of config dicts (see build_model_from_config).
        X, Y: Full dataset windows (X: (n, L, F), Y: (n, H)).
        n_splits: Number of walk-forward folds.
        test_ratio: Fraction of data used for test per fold.
        base_epochs, base_batch: Default training params if not in cfg.
        verbose: Verbosity forwarded to training.

    Returns:
        pd.DataFrame where each row corresponds to a config with:
            - all original cfg keys,
            - "mean_rmse": mean RMSE across folds (Y flattened),
            - "mean_mae": mean MAE across folds (Y flattened),
            - "folds": list of per-fold dicts from walk_forward_eval.

    Notes:
        - Builds a fresh model per fold and per config (no weight leakage).
        - Uses EarlyStopping(monitor="loss") as a minimal guard; you can
          pass stricter callbacks via cfg if needed.
    """

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    lookback = int(X.shape[1])
    n_features = int(X.shape[2]) if X.ndim == 3 else 1
    horizon = int(Y.shape[1]) if Y.ndim == 2 else 1

    rows = []
    early = EarlyStopping(monitor="loss", patience=1, restore_best_weights=True)

    for cfg in configs:
        def make_builder(cfg_):
            def builder():
                return build_model_from_config(cfg_, lookback, n_features, horizon)
            return builder

        res = walk_forward_eval(
            X, Y,
            build_model_fn=make_builder(cfg),
            n_splits=n_splits,
            test_ratio=test_ratio,
            epochs=cfg.get("epochs", base_epochs),
            batch_size=cfg.get("batch_size", base_batch),
            callbacks=[early],
            verbose=verbose
        )
        mean_rmse = float(np.mean([r["rmse"] for r in res])) if len(res) else np.inf
        mean_mae  = float(np.mean([r["mae"]  for r in res])) if len(res) else np.inf
        rows.append({**cfg, "mean_rmse": mean_rmse, "mean_mae": mean_mae, "folds": res})

    return pd.DataFrame(rows)


# ---------- 10) C5 example run (deltas + features) ----------


def run_c5_example(csv_path="cache/aapl_all.csv",
                   lookback: int = 60, horizon: int = 5,
                   n_splits: int = 3, test_ratio: float = 0.2):
    """
    End-to-end C5 demo:
      - Load OHLCV, create simple features (1-step return, SMA5, SMA10).
      - Build supervised windows (Δ target = y_{t+h} - y_t).
      - Scale X with MinMax and Δy with StandardScaler.
      - Grid-search a few RNN configs with walk-forward CV.
      - Train the best config on the last split with a small val slice.
      - Inverse-transform deltas and reconstruct prices; report metrics.
      - Estimate prediction intervals via MC-Dropout and plot bands.
      - Compare against naïve/moving-average/drift baselines.

    Args:
        csv_path: Path to cached CSV (expects 'date' and OHLC columns).
        lookback: Past window length for inputs.
        horizon: Multi-step output length.
        n_splits: Number of CV folds.
        test_ratio: Test fraction per fold.

    Returns:
        dict with {"best_config", "best_metrics", "grid"} for reporting.
    """
    Path("experiments_c5").mkdir(parents=True, exist_ok=True)

    # 1) Load & features
    df = pd.read_csv(csv_path)
    df = prepare_ohlc(df, date_col="date")  # external helper in your codebase
    d2 = df.copy()
    d2["ret1"] = d2["close"].pct_change().fillna(0.0)
    d2["sma5"] = d2["close"].rolling(5).mean().bfill()
    d2["sma10"] = d2["close"].rolling(10).mean().bfill()
    feats = ["close", "ret1", "sma5", "sma10"]

    X_df_raw = d2[feats].astype(np.float32)
    y_raw_ser = d2["close"].astype(np.float32)

    # 2) Windows in deltas
    X_raw, Y_raw, Y_delta_raw, last_close = build_supervised_multistep_delta(
        X_df_raw, y_raw_ser, lookback, horizon
    )

    # 3) Scaling
    sx  = MinMaxScaler()
    sdy = StandardScaler()
    n_features = X_raw.shape[2]
    X_flat = X_raw.reshape(len(X_raw), lookback * n_features)
    X_scaled = sx.fit_transform(X_flat).reshape(len(X_raw), lookback, n_features)
    Y_delta_scaled = sdy.fit_transform(Y_delta_raw.reshape(-1, horizon)).reshape(Y_delta_raw.shape)

    # 4) Small grid (extend as needed)
    configs = [
        {"layer_type": "LSTM", "units": (64, 32),  "dropout": 0.2, "epochs": 16, "batch_size": 64, "lr": 1e-3, "dense_units": 64},
        {"layer_type": "GRU",  "units": (96,),     "dropout": 0.2, "epochs": 20, "batch_size": 64, "lr": 1e-3, "dense_units": 64},
        {"layer_type": "LSTM", "units": (128, 64), "dropout": 0.2, "epochs": 25, "batch_size": 64, "lr": 1e-3, "bidirectional": True, "dense_units": 64},
    ]
    df_res = grid_search_timeseries(configs, X_scaled, Y_delta_scaled,
                                    n_splits=n_splits, test_ratio=test_ratio, verbose=0)
    print(df_res[["layer_type","units","mean_rmse","mean_mae"]])  # Δ-scale (std units)

    # 5) Train best on last split + small val
    best = df_res.sort_values("mean_rmse").iloc[0].to_dict()

    def builder_best():
        return build_model_from_config(best, X_scaled.shape[1], X_scaled.shape[2], Y_delta_scaled.shape[1])

    splits = timeseries_splits(len(X_scaled), n_splits=n_splits, test_ratio=test_ratio)
    sp = splits[-1]
    X_tr, Yd_tr = X_scaled[sp.train_start:sp.train_end], Y_delta_scaled[sp.train_start:sp.train_end]
    X_te, Yd_te = X_scaled[sp.test_start:sp.test_end],  Y_delta_scaled[sp.test_start:sp.test_end]
    Y_true_raw = Y_raw[sp.test_start:sp.test_end]
    last_close_te = last_close[sp.test_start:sp.test_end]

    # Temporal validation (last 10% of train)
    val_len = max(1, int(0.1 * len(X_tr)))
    X_tr_fit, Yd_tr_fit = X_tr[:-val_len], Yd_tr[:-val_len]
    X_val,    Yd_val    = X_tr[-val_len:], Yd_tr[-val_len:]

    # Callbacks
    try:
        cbs = make_callbacks(out_dir="artifacts_c5_best")  # external helper in your codebase
    except NameError:
        cbs = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
        ]

    model = builder_best()
    model.fit(X_tr_fit, Yd_tr_fit,
              epochs=int(best.get("epochs", 25)),
              batch_size=int(best.get("batch_size", 64)),
              validation_data=(X_val, Yd_val),
              callbacks=cbs, verbose=0, shuffle=False)

    # 6) Predict Δ -> inverse scale -> reconstruct price
    Yd_hat_scaled = model.predict(X_te, verbose=0)
    Yd_hat = sdy.inverse_transform(Yd_hat_scaled.reshape(-1, horizon)).reshape(Yd_hat_scaled.shape)
    Y_hat_inv = (last_close_te[:, None] + Yd_hat).astype(np.float32)
    Y_te_inv  = Y_true_raw.astype(np.float32)

    m = multistep_metrics(Y_te_inv, Y_hat_inv)
    print("Best fold metrics (reconstructed):", m)
    print("Per-horizon:", multistep_metrics_by_horizon(Y_te_inv, Y_hat_inv))

    # 7) MC-Dropout bands (Δ -> price)
    mc = mc_dropout_predict(model, X_te, T=60, q_low=5, q_high=95)
    mc_mean_d = sdy.inverse_transform(mc["mean"].reshape(-1, horizon)).reshape(mc["mean"].shape)
    mc_low_d  = sdy.inverse_transform(mc["low"].reshape(-1, horizon)).reshape(mc["low"].shape)
    mc_high_d = sdy.inverse_transform(mc["high"].reshape(-1, horizon)).reshape(mc["high"].shape)
    mc_mean_p = last_close_te[:, None] + mc_mean_d
    mc_low_p  = last_close_te[:, None] + mc_low_d
    mc_high_p = last_close_te[:, None] + mc_high_d

    plot_multistep(Y_te_inv, Y_hat_inv, title="C5 – Multi-step (avg) [price]")
    plot_with_intervals(Y_te_inv, mc_mean_p, mc_low_p, mc_high_p,
                        sample_idx=-1, title="C5 – Forecast with PI [price]")

    # 8) RMSE per config (Δ-scale; indicative)
    try:
        rmse_series = df_res["mean_rmse"].to_numpy()
    except KeyError:
        rmse_series = df_res.filter(regex="rmse", axis=1).iloc[:, 0].to_numpy()
    plot_walkforward(rmse_series, title="C5 – RMSE (grid means, Δ-scale)")

    # 9) Baselines in real price space
    idxs_test = np.arange(sp.test_start, sp.test_end)
    y_whole = y_raw_ser.values.astype(np.float32)
    print("Baseline (Last): ",  multistep_metrics(Y_te_inv, baseline_naive_last(y_whole, lookback, horizon, idxs_test)))
    print("Baseline (MA5):  ",  multistep_metrics(Y_te_inv, baseline_moving_average(y_whole, lookback, horizon, idxs_test, win=5)))
    print("Baseline (Drift):",  multistep_metrics(Y_te_inv, baseline_drift(y_whole, lookback, horizon, idxs_test)))

    return {"best_config": best, "best_metrics": m, "grid": df_res}


# --- Quick C5 demo ---
# C5_example= run_c5_example(csv_path="cache/aapl_all.csv", lookback=60, horizon=5, n_splits=3, test_ratio=0.2)


# -------------------- Task C6 --------------------
"""
Task C6 – Machine Learning 3

Main features:
  - Transformer model for temporal attention.
  - Ensemble combining LSTM, GRU, BiLSTM, and Transformer.
  - Explainability: SHAP/gradients + attention heatmaps.
  - Evaluation with RMSE, MAE, MAPE, R², and prediction variance.
"""



# ---------- Helper: R² ----------
def r2_score_np(y_true, y_pred):
    """Compute R² (coefficient of determination) for NumPy arrays."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


# ---------- Transformer model ----------
def _transformer_encoder_block(x, num_heads=4, key_dim=32, ff_dim=128, dropout=0.1):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ff = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(x.shape[-1])
    ])(x)
    x = tf.keras.layers.Add()([x, ff])
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)


def build_attention_model(lookback, n_features, horizon,
                          num_layers=2, num_heads=4, key_dim=32, ff_dim=128,
                          dropout=0.1, lr=1e-3):
    """Light Transformer encoder for multi-step regression."""
    inp = tf.keras.Input(shape=(lookback, n_features))
    x = tf.keras.layers.Dense(ff_dim, activation="relu")(inp)
    for _ in range(num_layers):
        x = _transformer_encoder_block(x, num_heads, key_dim, ff_dim, dropout)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(horizon, activation="linear")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=Huber(),
        metrics=["mae"]
    )
    return model


# ---------- Ensemble training ----------
def fit_single_model(model, X_train, Y_train, X_val, Y_val,
                     epochs=10, batch_size=32, verbose=0):
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=False,
        callbacks=[es]
    )
    return history


def build_ensemble_models(X_train, Y_train, X_val, Y_val, X_test,
                          lookback, n_features, horizon,
                          epochs=10, batch_size=64, verbose=0):
    """Train LSTM, GRU, BiLSTM, Transformer, and combine predictions."""
    configs = [
        ("LSTM",   build_model_from_config({"model": "lstm",  "units": (64, 32), "dropout": 0.2, "lr": 1e-3},
                                            lookback, n_features, horizon)),
        ("GRU",    build_model_from_config({"model": "gru",   "units": (96,),    "dropout": 0.2, "lr": 1e-3},
                                            lookback, n_features, horizon)),
        ("BiLSTM", build_model_from_config({"model": "bilstm","units": (128,),   "dropout": 0.2, "lr": 1e-3},
                                            lookback, n_features, horizon)),
        ("Transformer", build_attention_model(lookback, n_features, horizon,
                                              num_layers=2, num_heads=4, key_dim=32, ff_dim=128,
                                              dropout=0.1, lr=1e-3))
    ]
    preds = []
    for name, model in configs:
        print(f"Training {name}...")
        fit_single_model(model, X_train, Y_train, X_val, Y_val,
                         epochs=epochs, batch_size=batch_size, verbose=verbose)
        preds.append(model.predict(X_test, verbose=0))
        K.clear_session()

    preds = np.stack(preds, axis=0)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred, std_pred


# ---------- Evaluation ----------
def evaluate_ensemble(y_true, y_pred_mean, y_pred_std):
    metrics = multistep_metrics(y_true, y_pred_mean)
    metrics["r2"] = r2_score_np(y_true, y_pred_mean)
    metrics["mean_pred_std"] = float(np.mean(y_pred_std))
    return metrics


# ---------- Explainability ----------
def explain_predictions(model, X_sample):
    """Compute SHAP or gradient-based feature importance."""
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        return np.abs(shap_values.values).mean(axis=(0, 1))
    except Exception as e:
        print(f"SHAP failed ({e}); using gradient method.")
        X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            preds = model(X_tensor, training=False)
        grads = tape.gradient(preds, X_tensor)
        return np.mean(np.abs(grads.numpy()), axis=(0, 1))


# ---------- Visualization ----------
def plot_ensemble_predictions(y_true, y_pred_mean, y_pred_std, title="C6 – Ensemble Forecast"):
    plt.figure()
    plt.plot(y_true.mean(axis=0), label="Actual (avg)", linewidth=2)
    plt.plot(y_pred_mean.mean(axis=0), label="Ensemble (avg)", linewidth=2)
    plt.fill_between(np.arange(y_pred_mean.shape[1]),
                     y_pred_mean.mean(axis=0) - y_pred_std.mean(axis=0),
                     y_pred_mean.mean(axis=0) + y_pred_std.mean(axis=0),
                     alpha=0.3, label="Std band")
    plt.title(title)
    plt.xlabel("Horizon step"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_feature_importance(importances, feature_names):
    plt.figure()
    idx = np.argsort(importances)[::-1]
    plt.bar(np.array(feature_names)[idx], importances[idx])
    plt.title("C6 – Feature Importance (SHAP/Gradients)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ---------- Incremental learning ----------
def train_incremental_model(model, X_new, Y_new, epochs=3, batch_size=32):
    """Fine-tune model with new data (incremental update)."""
    model.fit(X_new, Y_new, epochs=epochs, batch_size=batch_size,
              verbose=0, shuffle=False)
    return model


# ---------- Full pipeline ----------
def run_c6_experiment(df: pd.DataFrame,
                      target_col="close",
                      lookback=60,
                      horizon=5,
                      epochs=10,
                      batch_size=64,
                      verbose=1,
                      out_dir="plots_c6",
                      attn_kwargs=None):
    """
    Run Task C6 pipeline: ensemble + transformer + explainability.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = prepare_ohlc(df, date_col="date")
    df["ret1"] = df["close"].pct_change().fillna(0.0)
    df["sma5"] = df["close"].rolling(5).mean().bfill()
    df["sma10"] = df["close"].rolling(10).mean().bfill()
    feats = ["close", "ret1", "sma5", "sma10"]

    X_df = df[feats].astype(np.float32)
    y_ser = df[target_col].astype(np.float32)
    X_raw, Y_raw, Y_delta, last_close = build_supervised_multistep_delta(X_df, y_ser, lookback, horizon)

    # Scale
    sx, sy = MinMaxScaler(), StandardScaler()
    n_feat = X_raw.shape[2]
    Xs = sx.fit_transform(X_raw.reshape(len(X_raw), lookback * n_feat)).reshape(len(X_raw), lookback, n_feat)
    Ys = sy.fit_transform(Y_delta.reshape(-1, horizon)).reshape(Y_delta.shape)

    # Split train/val/test
    splits = timeseries_splits(len(Xs), n_splits=3, test_ratio=0.2)
    sp = splits[-1]
    X_train, Y_train = Xs[sp.train_start:sp.train_end], Ys[sp.train_start:sp.train_end]
    X_test, Y_test = Xs[sp.test_start:sp.test_end], Y_raw[sp.test_start:sp.test_end]
    val_len = max(1, int(0.1 * len(X_train)))
    X_val, Y_val = X_train[-val_len:], Y_train[-val_len:]
    X_tr = X_train[:-val_len]; Y_tr = Y_train[:-val_len]

    # Ensemble
    mean_pred_d, std_pred_d = build_ensemble_models(
        X_tr, Y_tr, X_val, Y_val, X_test,
        lookback, n_feat, horizon,
        epochs=epochs, batch_size=batch_size, verbose=verbose
    )

    # Inverse transform deltas -> prices
    mean_pred = sy.inverse_transform(mean_pred_d.reshape(-1, horizon)).reshape(mean_pred_d.shape)
    std_pred  = sy.inverse_transform(std_pred_d.reshape(-1, horizon)).reshape(std_pred_d.shape)
    mean_pred_p = last_close[sp.test_start:sp.test_end, None] + mean_pred
    std_pred_p  = std_pred

    # Evaluate
    metrics = evaluate_ensemble(Y_test, mean_pred_p, std_pred_p)
    print("C6 metrics:", metrics)

    # Explainability on first model (retrain one for SHAP)
    model_explain = build_attention_model(lookback, n_feat, horizon, **(attn_kwargs or {}))
    fit_single_model(model_explain, X_tr, Y_tr, X_val, Y_val, epochs=3, verbose=0)
    importances = explain_predictions(model_explain, X_val[:32])

    # Plots
    plot_ensemble_predictions(Y_test, mean_pred_p, std_pred_p)
    plot_feature_importance(importances, feats)

    results = {"metrics": metrics, "plots": {"ensemble": out_dir}}
    return results


# ---------- C6 demo ----------
# df = pd.read_csv("cache/aapl_all.csv")
# res_c6 = run_c6_experiment(df, lookback=60, horizon=5, epochs=8, batch_size=64)
# print("Final metrics (C6):", res_c6["metrics"])











# -------------------- Task C7 --------------------
"""
Task C7 – Sentiment-Based Stock Price Movement Prediction
Goal:
  Extend the stock price forecasting pipeline by incorporating sentiment
  data from news or social media to classify whether the next day's price
  will rise (1) or fall (0).

Main steps:
  1. Data Collection & Preprocessing – Load price and sentiment data.
  2. Sentiment Analysis – Compute daily sentiment scores (synthetic or real).
  3. Feature Engineering & Modelling – Combine financial + sentiment features.
  4. Evaluation – Accuracy, Precision, Recall, F1, Confusion Matrix.
  5. Independent Research Component – Optional FinBERT integration.
  6. Reporting – Visualization and metrics summary.
"""



# ---------- 1) Data collection & sentiment scoring ----------

def load_sentiment_data(csv_prices: str = "cache/aapl_all.csv",
                        csv_news: str | None = None) -> pd.DataFrame:
    """
    Load stock prices and optional news sentiment data, compute daily sentiment score.

    Args:
        csv_prices: Path to price data (expects 'date' and 'close').
        csv_news: Optional CSV containing columns ['date', 'sentiment_score'].

    Returns:
        DataFrame merged with 'date', 'close', and 'sentiment_score' columns.

    Notes:
        - If no external sentiment file is provided, random synthetic sentiment
          scores are generated for demonstration purposes.
        - Sentiment range: [-1, 1].
    """
    df = pd.read_csv(csv_prices)
    df["date"] = pd.to_datetime(df["date"])
    df = prepare_ohlc(df, date_col="date")

    # Load or simulate sentiment data
    if csv_news and os.path.exists(csv_news):
        news = pd.read_csv(csv_news)
        news["date"] = pd.to_datetime(news["date"])
        if "sentiment_score" not in news.columns:
            raise ValueError("csv_news must contain 'sentiment_score' column.")
    else:
        news = df[["date"]].copy()
        np.random.seed(42)
        news["sentiment_score"] = np.random.uniform(-0.5, 0.5, size=len(news))

    df = df.merge(news[["date", "sentiment_score"]], on="date", how="left")
    df["sentiment_score"].fillna(0, inplace=True)
    return df


# ---------- 2) Build binary target ----------

def build_binary_target(df: pd.DataFrame, target_col: str = "close") -> pd.DataFrame:
    """
    Compute binary label for next-day price movement.

    Args:
        df: DataFrame with target column (e.g., 'close').
        target_col: Name of target price column.

    Returns:
        DataFrame with an added column 'target' (1 = price rises, 0 = falls).
    """
    df = df.copy()
    df["target"] = (df[target_col].shift(-1) > df[target_col]).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df


# ---------- 3) Model definition ----------

def build_sentiment_model(input_dim: int, lr: float = 1e-3) -> tf.keras.Model:
    """
    Define a simple fully connected neural network for binary classification.

    Args:
        input_dim: Number of input features.
        lr: Learning rate.

    Returns:
        Compiled Keras Sequential model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


# ---------- 4) Full pipeline ----------

def run_c7_experiment(csv_prices: str = "cache/aapl_all.csv",
                      csv_news: str | None = None,
                      test_size: float = 0.2,
                      epochs: int = 20,
                      batch_size: int = 32):
    """
    Run the complete Task C7 pipeline:
      1. Load price + sentiment data.
      2. Compute binary labels (rise/fall).
      3. Engineer features (financial + sentiment).
      4. Train classifier and evaluate performance.

    Args:
        csv_prices: Path to CSV with price data.
        csv_news: Optional sentiment data file.
        test_size: Fraction of test data.
        epochs: Training epochs.
        batch_size: Mini-batch size.

    Returns:
        dict with metrics and model training history.
    """
    # --- 1. Load and preprocess ---
    df = load_sentiment_data(csv_prices, csv_news)
    df = build_binary_target(df)

    # --- 2. Feature engineering ---
    df["ret1"] = df["close"].pct_change().fillna(0.0)
    df["sma5"] = df["close"].rolling(5).mean().bfill()
    df["sma10"] = df["close"].rolling(10).mean().bfill()
    feats = ["close", "ret1", "sma5", "sma10", "sentiment_score"]

    X = df[feats].astype(np.float32)
    y = df["target"].astype(np.float32)

    # --- 3. Split train/test ---
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # --- 4. Scale features ---
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- 5. Build and train model ---
    model = build_sentiment_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=0, shuffle=False)

    # --- 6. Evaluate ---
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    print("C7 metrics:", json.dumps(metrics, indent=2))

    # --- 7. Visualization ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("C7 – Confusion Matrix (Up/Down)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("C7 – Training Curves")
    plt.xlabel("Epoch"); plt.ylabel("Binary Crossentropy")
    plt.legend(); plt.tight_layout(); plt.show()

    return {"metrics": metrics, "history": history.history, "model": model, "scaler": scaler}


# ---------- 5) Independent Research Component (FinBERT demo) ----------

def build_finbert_sentiment(texts: list[str]) -> np.ndarray:
    """
    Example placeholder for FinBERT sentiment extraction.
    Requires transformers library.

    Args:
        texts: List of text strings (news headlines).

    Returns:
        np.ndarray of sentiment scores.
    """
    try:
        classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
        res = [classifier(t[:512])[0]["score"] for t in texts]
        return np.array(res)
    except Exception as e:
        print(f"FinBERT sentiment extraction failed: {e}")
        return np.zeros(len(texts), dtype=np.float32)

# -------------------- Next-Day Price Forecast --------------------


def predict_next_price(csv_prices="cache/ls.csv", model=None, scaler=None):
    """
    Estimate the next-day close price based on the most recent data window.
    Uses the same features as in run_c7_experiment().
    """
    df = pd.read_csv(csv_prices)
    df = prepare_ohlc(df, date_col="date")

    # Rebuild features
    df["ret1"] = df["close"].pct_change().fillna(0.0)
    df["sma5"] = df["close"].rolling(5).mean().bfill()
    df["sma10"] = df["close"].rolling(10).mean().bfill()
    df["sentiment_score"] = np.random.uniform(-0.5, 0.5, size=len(df))  # synthetic

    feats = ["close", "ret1", "sma5", "sma10", "sentiment_score"]

    # Take last sample
    last = df[feats].iloc[-1:].astype(np.float32)

    if scaler:
        last = scaler.transform(last)

    # Predict probability (rise/fall)
    prob = model.predict(last, verbose=0).ravel()[0]
    trend = "rise" if prob >= 0.5 else "fall"

    # Print formatted result

    print("Next-Day Forecast Summary (Task C7)")
    print(f"Predicted close price (approx): {df['close'].iloc[-1]:.2f} USD → next ≈ {df['close'].iloc[-1]*(1 + (prob - 0.5)*0.02):.2f} USD")
    print(f"Trend classification: {trend.upper()}  (probability = {prob:.2f})")

    return prob, trend


# ---------- 6) Quick demo ----------

if __name__ == "__main__":
    res_c7 = run_c7_experiment("cache/aapl_all.csv")
    print("Final metrics (C7):", res_c7["metrics"])

# --- Next-day prediction ---
    model = res_c7["model"]
    scaler = res_c7["scaler"]
    prob, trend = predict_next_price("cache/aapl_all.csv", model=model, scaler=scaler)
