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

import os # To handle file and directory paths
import pickle # To save and load the scaler

from pathlib import Path # To handle file paths
from datetime import datetime, timezone # To handle date and time

from sklearn.model_selection import train_test_split # To split the data
from sklearn.preprocessing import MinMaxScaler, StandardScaler # To scale the data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import tensorflow as tf
import mplfinance as mpf

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

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






# -------- Base code  --------------------

''''
#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------
PRICE_VALUE = "Close"

scaler = MinMaxScaler(feature_range=(0, 1)) 
# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1)) 
# Flatten and normalise the data
# First, we reshape a 1D array(n) to 2D array(n,1)
# We have to do that because sklearn.preprocessing.fit_transform()
# requires a 2D array
# Here n == len(scaled_data)
# Then, we scale the whole array to the range (0,1)
# The parameter -1 allows (np.)reshape to figure out the array size n automatically 
# values.reshape(-1, 1) 
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
# When reshaping an array, the new shape must contain the same number of elements 
# as the old shape, meaning the products of the two shapes' dimensions must be equal. 
# When using a -1, the dimension corresponding to the -1 will be the product of 
# the dimensions of the original array divided by the product of the dimensions 
# given to reshape so as to maintain the same number of elements.

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 # Original

# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

# test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

test_data = yf.download(COMPANY,TEST_START,TEST_END)


# The above bug is the reason for the following line of code
# test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??

'''