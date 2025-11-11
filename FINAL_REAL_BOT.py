import os
import re
import glob
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
import warnings
import quantstats as qs
import datetime
from datetime import datetime
import math

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from skopt import gp_minimize
from skopt.space import Real
from numpy.fft import fft, fftfreq
from scipy.stats import kstest, normaltest, shapiro
from scipy.special import comb
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
from importlib import util

warnings.simplefilter('ignore')


###ROOSTOO FUNCTOIONS DO NOT MODIFY###

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import hashlib
import hmac
import time

# API_KEY = "UqIHb7BC5QgKVMjZKAhzARLUa6jWvvgO3GO1OxdVMqXBWVPJsqsha33VIvh6KFrx"
# SECRET = "KBpaEeVmYVussNCI5ewdTv7jTmVJ6S0ZGWxvIkpKz5xMoLGsVwWYMKek0a5XeRAD"

"""REAL API"""
API_KEY = "pR6tM2yNaV9bC4DfH1jK7LxWoG3qS8EeT5uZ0iYcnB2vF6PrX7wD1mQaJ4lUh9Sz"
SECRET = "C9vB1nM5qW7eRtY3uI9oPaS1dF3gHjK7lL2ZxC6V8bN4mQwE0rT4yUiP6oA8"

BASE_URL = "https://mock-api.roostoo.com"


def generate_signature(params):
    query_string = '&'.join(["{}={}".format(k, params[k])
                             for k in sorted(params.keys())])
    us = SECRET.encode('utf-8')
    m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
    return m.hexdigest()


def get_server_time():
    r = requests.get(
        BASE_URL + "/v3/serverTime",
    )
    print(r.status_code, r.text)
    return r.json()


def get_ex_info():
    r = requests.get(
        BASE_URL + "/v3/exchangeInfo",
    )
    print(r.status_code, r.text)
    return r.json()


def get_ticker(pair=None):
    payload = {
        "timestamp": int(time.time()),
    }
    if pair:
        payload["pair"] = pair

    r = requests.get(
        BASE_URL + "/v3/ticker",
        params=payload,
    )
    print(r.status_code, r.text)
    return r.json()


def get_balance():
    payload = {
        "timestamp": int(time.time()) * 1000,
    }

    r = requests.get(
        BASE_URL + "/v3/balance",
        params=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)
    return r.json()


def place_order(coin, side, qty, price=None):
    payload = {
        "timestamp": int(time.time()) * 1000,
        "pair": coin + "/USD",
        "side": side,
        "quantity": qty,
    }

    if not price:
        payload['type'] = "MARKET"
    else:
        payload['type'] = "LIMIT"
        payload['price'] = price

    r = requests.post(
        BASE_URL + "/v3/place_order",
        data=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)


def cancel_order():
    payload = {
        "timestamp": int(time.time()) * 1000,
        # "order_id": 77,
        "pair": "BTC/USD",
    }

    r = requests.post(
        BASE_URL + "/v3/cancel_order",
        data=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)


def query_order():
    payload = {
        "timestamp": int(time.time())*1000,
        # "order_id": 77,
        # "pair": "DASH/USD",
        # "pending_only": True,
    }

    r = requests.post(
        BASE_URL + "/v3/query_order",
        data=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)


def pending_count():
    payload = {
        "timestamp": int(time.time()) * 1000,
    }

    r = requests.get(
        BASE_URL + "/v3/pending_count",
        params=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)
    return r.json()

##################################################################### DO NOT MODIFY ABOVE THIS LINE ###


# ----- ALL STRATS ----- #

short_period = 7
long_period = 14

class MovingAverageStrategy:
    def __init__(self, short=short_period, long=long_period, resample='D'):
        self.short = short
        self.long = long
        self.resample = resample
    def signal(self, index, series):
        series_daily = series.resample(self.resample).last()
        index_daily = series_daily.index
        x = series_daily.values.astype(float)
        methods = ['EMA','SMA','WMA','DEMA','TEMA','TRIMA','KAMA']
        vals = []
        for m in methods:
            if m == 'EMA':
                a = talib.EMA(x, timeperiod=self.short)
                b = talib.EMA(x, timeperiod=self.long)
            elif m == 'SMA':
                a = talib.SMA(x, timeperiod=self.short)
                b = talib.SMA(x, timeperiod=self.long)
            elif m == 'WMA':
                a = talib.WMA(x, timeperiod=self.short)
                b = talib.WMA(x, timeperiod=self.long)
            elif m == 'DEMA':
                a = talib.DEMA(x, timeperiod=self.short)
                b = talib.DEMA(x, timeperiod=self.long)
            elif m == 'TEMA':
                a = talib.TEMA(x, timeperiod=self.short)
                b = talib.TEMA(x, timeperiod=self.long)
            elif m == 'TRIMA':
                a = talib.TRIMA(x, timeperiod=self.short)
                b = talib.TRIMA(x, timeperiod=self.long)
            elif m == 'KAMA':
                a = talib.KAMA(x, timeperiod=self.short)
                b = talib.KAMA(x, timeperiod=self.long)
            vals.append(((a - b) > 0).astype(int))
        avg = np.nanmean(np.column_stack(vals), axis=1)
        df_daily = pd.DataFrame({'value': avg}, index=index_daily)
        return df_daily.reindex(index).ffill().fillna(0.0)


lookback_period = 6
zscore_lookback_period = 24
signal_threshold = 0.6

class BetaRegressionStrategy:
    def __init__(self, beta_lookback_period=lookback_period,
                 zscore_lookback_period=zscore_lookback_period,
                 signal_threshold=signal_threshold,
                 resample='H'):
        self.beta_lookback_period = beta_lookback_period
        self.zscore_lookback_period = zscore_lookback_period
        self.signal_threshold = signal_threshold
        self.resample = resample

    def signal(self, index, low_series, high_series):
        x = low_series.resample(self.resample).min()
        y = high_series.resample(self.resample).max()
        idx = x.index
        w = self.beta_lookback_period
        mx = x.rolling(w).mean(); my = y.rolling(w).mean()
        mxx = (x * x).rolling(w).mean(); myy = (y * y).rolling(w).mean()
        mxy = (x * y).rolling(w).mean()
        varx = mxx - mx * mx; vary = myy - my * my; cov = mxy - mx * my
        beta_s = cov / varx
        corr = cov / (np.sqrt(varx) * np.sqrt(vary))
        r2_s = corr * corr
        z = (beta_s - beta_s.rolling(self.zscore_lookback_period).mean()) / beta_s.rolling(self.zscore_lookback_period).std()
        strength = z * r2_s
        weight = strength.where(beta_s > self.signal_threshold, 0.0)
        df_hourly = pd.DataFrame({'value': weight}, index=idx)
        return df_hourly.reindex(index).ffill().fillna(0.0)


threshold = -1


class CLV:
    def __init__(self, threshold=threshold):
        self.threshold = threshold

    def signal(self, index, clv_series):
        y = (clv_series == self.threshold).fillna(0)
        return pd.DataFrame({'value': y.astype(int)}, index=index)


ma_long = 14
ma_short = 7
breakout_lookback = 7

class BreakoutStrategy:
    def __init__(self, ma_long=ma_long, ma_short=ma_short,
                 breakout_lookback=breakout_lookback, resample='D'):
        self.ma_long = ma_long
        self.ma_short = ma_short
        self.breakout_lookback = breakout_lookback
        self.resample = resample

    def signal(self, index: pd.DatetimeIndex,
               low_series: pd.Series,
               high_series: pd.Series) -> pd.DataFrame:
        low = low_series.resample(self.resample).min()
        high = high_series.resample(self.resample).max()
        close = (low + high) / 2.0

        ma_long = close.rolling(self.ma_long).mean()
        ma_short = close.rolling(self.ma_short).mean()

        high_n = close.rolling(self.breakout_lookback).max()
        low_n = close.rolling(self.breakout_lookback).min()

        prev_close = close.shift()
        tr1 = high_n - low_n
        tr2 = (high_n - prev_close).abs()
        tr3 = (low_n - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        breakout_up = (close > ma_long) & (close > high_n.shift(1))
        breakout_down = (close < ma_short) & (close < low_n.shift(1))

        sig = pd.Series(0.0, index=close.index)
        sig[breakout_up] = 1.0
        sig[breakout_down] = -1.0

        sig_smooth = sig.rolling(window=self.ma_short, min_periods=1).mean().fillna(0.0)
        df_daily = pd.DataFrame({'value': sig_smooth.astype(float)}, index=close.index)
        return df_daily.reindex(index).ffill().fillna(0.0)


time_series_datapoints = 288
low_energy_threshold = 0.5
mid_energy_threshold = 0.1
low_slope_threshold = 0.0
mid_slope_threshold = 0.0


class FourierTransform:
    def __init__(self,
                 time_series_datapoints=time_series_datapoints,
                 low_energy_threshold=low_energy_threshold,
                 mid_energy_threshold=mid_energy_threshold,
                 low_slope_threshold=low_slope_threshold,
                 mid_slope_threshold=mid_slope_threshold):
        self.recents = int(time_series_datapoints)
        self.low_energy_threshold = float(low_energy_threshold)
        self.mid_energy_threshold = float(mid_energy_threshold)
        self.low_slope_threshold = float(low_slope_threshold)
        self.mid_slope_threshold = float(mid_slope_threshold)
        self.low_grid = np.arange(0.01, 0.09, 0.01)
        self.high_grid = np.arange(0.06, 0.21, 0.01)

    def discrete_fourier_transform(self, s: pd.Series, recents: int) -> pd.DataFrame:
        x = s.dropna().sort_index().tail(recents)
        n = len(x)
        if n == 0:
            return pd.DataFrame({"frequency": [], "magnitude": []})
        dt = 1.0
        y = x.values
        yf = fft(y)
        xf = fftfreq(n, d=dt)
        pos = xf > 0
        xf = xf[pos]
        mag = np.abs(yf)[pos]
        return pd.DataFrame({"frequency": xf, "magnitude": mag})

    def reconstruct_time_series(self, s: pd.Series,
                                low_filter: float,
                                high_filter: float,
                                band: str) -> pd.Series:
        x = s.dropna().sort_index()
        y = x.values
        n = len(x)
        if n == 0:
            return pd.Series([], dtype=float)

        F = np.fft.rfft(y, n=n)
        f = np.fft.rfftfreq(n, d=1.0)

        if band == "low":
            mask = f < low_filter
        elif band == "middle":
            mask = (f >= low_filter) & (f <= high_filter)
        else:
            mask = f > high_filter

        y_band = np.fft.irfft(F * mask, n=n)
        return pd.Series(y_band, index=x.index)

    def frequency_filter(self, spec_df: pd.DataFrame,
                         low_filter: float,
                         high_filter: float):
        d = spec_df.dropna()
        low = d[d["frequency"] < low_filter]
        mid = d[(d["frequency"] >= low_filter) & (d["frequency"] <= high_filter)]
        high = d[d["frequency"] > high_filter]
        return low, mid, high

    def normality_test(self, arr):
        x = np.asarray(arr)
        if len(x) < 8 or np.allclose(x.std(ddof=1), 0):
            return 1.0, 1.0, 1.0
        z = (x - x.mean()) / x.std(ddof=1)
        return (
            kstest(z, "norm").pvalue,
            normaltest(x).pvalue,
            shapiro(x).pvalue
        )

    def seasonality_test(self, spec_df: pd.DataFrame) -> float:
        d = spec_df.dropna()
        if len(d) == 0:
            return 1.0
        p = (d["magnitude"].values ** 2)
        mx = float(p.max())
        tot = float(p.sum())
        g = mx / tot if tot > 0 else 0.0
        m = len(p)
        if g <= 0 or m <= 1:
            return 1.0
        kmax = int(np.floor(1.0 / g)) if g > 0 else 0
        pv = 0.0
        for k in range(1, kmax + 1):
            pv += ((-1)**(k-1)) * comb(m, k, exact=False) * (1 - k*g)**(m-1)
        return float(pv)

    def compute_energy(self, spec_df: pd.DataFrame,
                       low_filter: float,
                       high_filter: float):
        d = spec_df.dropna()
        if len(d) == 0:
            return 0.0, 0.0, 0.0
        f = d["frequency"].values
        m2 = (d["magnitude"].values ** 2)
        low = float(m2[f < low_filter].sum())
        mid = float(m2[(f >= low_filter) & (f < high_filter)].sum())
        high = float(m2[f >= high_filter].sum())
        tot = low + mid + high
        if tot > 0:
            low /= tot
            mid /= tot
            high /= tot
        return low, mid, high

    def compute_slope(self, s: pd.Series, inst_regression: int) -> float:
        y = s.dropna().tail(inst_regression).values
        if len(y) < 2:
            return 0.0
        X = np.arange(len(y)).reshape(-1, 1)
        return float(LinearRegression().fit(X, y).coef_[0])

    def choose_filter_bound(self, s: pd.Series,
                            spec_df: pd.DataFrame,
                            low_grid,
                            high_grid):
        for low in low_grid:
            for high in high_grid:
                if high <= low:
                    continue
                hf = self.reconstruct_time_series(s, low, high, "high")
                r = pd.Series(hf).diff().dropna()
                p1, p2, p3 = self.normality_test(r)

                _, mid_spec, _ = self.frequency_filter(spec_df, low, high)
                pv_season = self.seasonality_test(mid_spec)

                if (p1 < 0.05) and (p2 < 0.05) and (p3 < 0.05) and (pv_season < 0.05):
                    return float(low), float(high)
        return None

    def signal(self, index: pd.DatetimeIndex,
               vwap_series: pd.Series) -> pd.DataFrame:
        s = vwap_series.dropna().sort_index()
        s = np.log(s[s > 0])
        if len(s) < self.recents:
            return pd.DataFrame({'value': pd.Series(0.0, index=index)})
        spec_last = self.discrete_fourier_transform(s, self.recents)
        chosen = self.choose_filter_bound(s, spec_last, self.low_grid, self.high_grid)
        if chosen is None:
            return pd.DataFrame({'value': pd.Series(0.0, index=index)})
        low_f, high_f = chosen
        weights = pd.Series(0.0, index=s.index, dtype=float)
        step = self.recents
        inst_reg = 3
        for t in range(self.recents, len(s) + 1, step):
            sw = s.iloc[:t].tail(self.recents)
            spec_t = self.discrete_fourier_transform(sw, self.recents)
            el, em, eh = self.compute_energy(spec_t, low_f, high_f)
            low_ts = self.reconstruct_time_series(sw, low_f, high_f, "low")
            mid_ts = self.reconstruct_time_series(sw, low_f, high_f, "middle")
            sl = self.compute_slope(low_ts, inst_reg)
            sm = self.compute_slope(mid_ts, inst_reg)
            ok = ((el >= self.low_energy_threshold) and
                  (em >= self.mid_energy_threshold) and
                  (sl >= self.low_slope_threshold) and
                  (sm >= self.mid_slope_threshold))
            if ok:
                w = max(0.0, 1.0 - (eh / el if el > 0 else 1.0) - (eh / em if em > 0 else 1.0))
            else:
                w = 0.0
            weights.iloc[t - 1] = float(max(w, 0.0))
        weights = weights.replace(0, np.nan).ffill().fillna(0.0)
        full = pd.Series(0.0, index=index, dtype=float)
        full.loc[weights.index] = weights.values
        full = full.ffill().fillna(0.0)
        return pd.DataFrame({'value': full}, index=index)


# AR Parameter
p = 1

class AutoRegressiveStrategy:
    def __init__(self, p=p, method='AR'):
        self.p = int(p)
        self.method = method.upper()

    def signal(self, index, series):
        y = series.dropna().astype(float)

        if len(y) < self.p + 2:
            return pd.DataFrame({'value': [np.nan] * len(series)}, index=index)

        if y.autocorr(lag=1) > 0.95:
            y = y.diff().dropna()

        if len(y) < self.p + 1:
            return pd.DataFrame({'value': [np.nan] * len(series)}, index=index)

        model = sm.tsa.AutoReg(y, lags=self.p, trend='c').fit()
        resid = model.resid
        sig = (resid > 0).astype(int)
        out = pd.DataFrame({'value': sig}, index=index)
        return out



# ARMA Parameter
p = 1
q = 1

class ARMAStrategy:
    def __init__(self, p=p, q=q, method='ARMA'):
        self.p = int(p)
        self.q = int(q)
        self.method = method.upper()

    def signal(self, index, series):
        y = series.dropna().astype(float)

        min_obs = max(self.p, self.q) + 2
        if len(y) < min_obs:
            return pd.DataFrame({'value': [np.nan] * len(series)}, index=index)

        if y.autocorr(lag=1) > 0.95:
            y = y.diff().dropna()

        if len(y) < min_obs:
            return pd.DataFrame({'value': [np.nan] * len(series)}, index=index)

        model = sm.tsa.ARIMA(y, order=(self.p, 0, self.q), trend='c').fit()
        resid = model.resid

        sig = (resid > 0).astype(int)

        out = pd.DataFrame({'value': sig}, index=index)
        return out

# Mean Reversion Parameter
window = 288
z_entry = -1.5
z_exit = -0.3

class MeanReversionStrategy:
    def __init__(self, window=window, z_entry=z_entry, z_exit=z_exit):
        self.window = int(window)
        self.z_entry = float(z_entry)
        self.z_exit  = float(z_exit)

    def signal(self, index, series):
        y = series.dropna().astype(float)

        if y.autocorr(lag=1) > 0.95:
            y = y.diff().dropna()

        if len(y) < self.window + 1:
            return pd.DataFrame({'value': [np.nan] * len(series)}, index=index)

        roll_mean = y.rolling(self.window).mean()
        roll_std  = y.rolling(self.window).std()
        z = (y - roll_mean) / roll_std

        long_flag = (z < self.z_entry)
        exit_flag = (z > self.z_exit)

        position = pd.Series(np.nan, index=z.index)
        position.iloc[self.window] = 0
        for t in range(self.window + 1, len(z)):
            if long_flag.iloc[t]:
                position.iloc[t] = 1
            elif exit_flag.iloc[t]:
                position.iloc[t] = 0
            else:
                position.iloc[t] = position.iloc[t-1]

        out = pd.DataFrame({'value': position}, index=index)
        return out



window = 10
sigma = 1

class VWAPStrategy:
    def __init__(self, window=window , sigma=sigma):
        self.window = int(window)
        self.sigma = float(sigma)

    def signal(self, index, series, volume):
        close  = series.dropna().astype(float)
        volume = volume.reindex(close.index).astype(float)

        df = pd.concat({'close': close, 'vol': volume}, axis=1).dropna()
        if len(df) < self.window:
            return pd.DataFrame({'value': [np.nan] * len(series)}, index=index)

        typical_price = df['close']
        tpv = typical_price * df['vol']
        rolling_tpv = tpv.rolling(self.window).sum()
        rolling_vol   = df['vol'].rolling(self.window).sum()
        rolling_vwap  = rolling_tpv / rolling_vol

        spread = df['close'] - rolling_vwap
        roll_std = spread.rolling(self.window).std()
        z = spread / roll_std

        sig = (z < -self.sigma).astype(int)

        out = pd.DataFrame({'value': sig}, index=index)
        return out



N = 72
resample = '1H'

class Donchian_Breakout:
    def __init__(self, N=N, resample=resample):
        self.N = int(N)
        self.resample = resample
        self.requires_panel = True

    def signal(self, index, open, high, low, close, volume, vwap, clv):
        h = high.resample(self.resample).max()
        l = low.resample(self.resample).min()
        c = close.resample(self.resample).last()
        up = h.rolling(self.N).max().shift(1)
        lo = l.rolling(self.N).min().shift(1)
        pos = pd.DataFrame(0, index=c.index, columns=c.columns, dtype=int)
        for col in c.columns:
            cc = c[col].to_numpy()
            uh = up[col].to_numpy()
            ll = lo[col].to_numpy()
            p = np.zeros(len(cc), dtype=int)
            for i in range(1, len(cc)):
                p[i] = p[i-1]
                if p[i-1] == 0 and cc[i] > uh[i]:
                    p[i] = 1
                elif p[i-1] == 1 and cc[i] < ll[i]:
                    p[i] = 0
            pos[col] = p
        picks = pos.reindex(index).ffill().fillna(0).astype(int)
        return picks



lookback = 14
top_frac = 0.2
vol_threshold = 5000000
vol_window = 7
risk_shift = 1

class Cross_Sectional_Momentum:
    def __init__(self, lookback=lookback, top_frac=top_frac, vol_threshold=vol_threshold, vol_window=vol_window, risk_shift=risk_shift):
        self.lookback = lookback
        self.top_frac = top_frac
        self.vol_threshold = vol_threshold
        self.vol_window = vol_window
        self.risk_shift = risk_shift
        self.requires_panel = True

    def signal(self, index, open, high, low, close, volume, vwap, clv):
        daily_close = close.resample('1D').last()
        daily_vol = volume.resample('1D').sum()
        dollar_vol = daily_close * daily_vol
        liquid = dollar_vol.rolling(self.vol_window).mean() >= self.vol_threshold

        mom = daily_close.pct_change(self.lookback)
        mom = mom.where(liquid)

        rank = mom.rank(axis=1, ascending=False, pct=True)
        picks = (rank <= self.top_frac).astype(int).shift(self.risk_shift)

        picks = picks.reindex(index, method='ffill').fillna(0).astype(int)
        return picks

# ----- MULTISTRAT ------ #

@dataclass
class StrategyConfig:
    strategy: object
    series_names: List[str] = None   # used for per-symbol strategies
    weight: float = 1.0
    long_only: bool = True


class MultiStrategyEngine:
    def __init__(self, configs: List[StrategyConfig], combine: str = 'weighted_sum'):
        if not configs:
            raise ValueError("MultiStrategyEngine requires at least one StrategyConfig.")
        self.configs = configs
        self.combine = combine

    def _combine(self, sig_list, weights):
        """Combine list of pd.Series into one ensemble series."""
        if not sig_list:
            return pd.Series(0.0)

        base_index = sig_list[0].index
        sig_list = [s.reindex(base_index).fillna(0.0) for s in sig_list]

        if self.combine == 'mean':
            return sum(sig_list) / float(len(sig_list))

        w_sum = float(sum(weights))
        if w_sum == 0:
            return sum(sig_list)

        out = pd.Series(0.0, index=base_index)
        for s, w in zip(sig_list, weights):
            out += s * (w / w_sum)
        return out

    def signal_matrix(
        self,
        index: pd.DatetimeIndex,
        series_mats: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Returns DataFrame [time x symbols] of combined signals.

        - Per-symbol strategies:
            cfg.series_names -> pulled column-wise per symbol.
        - Panel strategies (strategy.requires_panel == True):
            strategy.signal(...) called once with full matrices,
            then symbol column is used inside symbol loop.
        """
        # Use first config to infer symbol universe
        first_cfg = self.configs[0]
        if first_cfg.series_names:
            base_mat = series_mats[first_cfg.series_names[0]]
        else:
            # fallback: use 'close' if first strategy is panel-only
            base_mat = series_mats['close']
        symbols = base_mat.columns

        # Precompute panel strategy outputs to avoid recomputing per symbol
        panel_cache = {}
        for cfg in self.configs:
            if getattr(cfg.strategy, "requires_panel", False):
                # Call with full panel data; strategies will pick what they need
                panel_sig = cfg.strategy.signal(
                    index,
                    series_mats.get('open'),
                    series_mats.get('high'),
                    series_mats.get('low'),
                    series_mats.get('close'),
                    series_mats.get('volume'),
                    series_mats.get('vwap'),
                    series_mats.get('clv'),
                )
                # Ensure DataFrame
                if isinstance(panel_sig, pd.Series):
                    panel_sig = panel_sig.to_frame(name='value')
                panel_cache[id(cfg)] = panel_sig

        out = {}

        for sym in symbols:
            sigs = []
            wts = []

            for cfg in self.configs:
                if getattr(cfg.strategy, "requires_panel", False):
                    # Use precomputed panel signal for this strategy
                    panel_sig = panel_cache[id(cfg)]

                    # If strategy returns positions per symbol as columns (Donchian, XSec Mom)
                    # pick its column; otherwise assume 'value' column.
                    if sym in panel_sig.columns:
                        s = panel_sig[sym].astype(float)
                    elif 'value' in panel_sig.columns:
                        s = panel_sig['value'].astype(float)
                    else:
                        # fallback: all zeros if format unexpected
                        s = pd.Series(0.0, index=index)

                else:
                    # Standard per-symbol strategy using series_names
                    if not cfg.series_names:
                        # If misconfigured, skip
                        s = pd.Series(0.0, index=index)
                    else:
                        series_map = {
                            name: series_mats[name][sym]
                            for name in cfg.series_names
                        }
                        df = cfg.strategy.signal(index, *series_map.values())
                        if isinstance(df, pd.Series):
                            s = df.astype(float)
                        else:
                            s = df['value'].astype(float)

                if cfg.long_only:
                    s = s.clip(lower=0.0)

                sigs.append(s)
                wts.append(cfg.weight)

            combined = self._combine(sigs, wts)
            out[sym] = combined

        sig_df = pd.DataFrame(out, index=index).fillna(0.0)
        return sig_df
# =========================
# LIVE DATA PIPELINE
# =========================

import ccxt
import pandas as pd
import numpy as np
import time

LOG_FILE = "trade_log.csv"
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,coin,side,qty,price\n")

def append_trade(coin, side, qty, price):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()},{coin},{side},{qty},{price}\n")

class LiveDataPipeline:
    def __init__(self,
                 exchange_id='coinbase',
                 symbols=None,
                 timeframe='5m',
                 limit=500,
                 exchange_params=None):
        params = {'enableRateLimit': True}
        if exchange_params:
            params.update(exchange_params)
        self.exchange = getattr(ccxt, exchange_id)(params)
        self.exchange.load_markets()
        self.timeframe = timeframe
        self.limit = int(limit)
        supported = set(self.exchange.symbols or [])
        requested = symbols or ['BTC/USD', 'ETH/USD', 'SOL/USD']
        self.symbols = [s for s in requested if s in supported]
        if not self.symbols:
            raise RuntimeError("No requested symbols supported by Coinbase.")
        self.data = {}

    def _fetch_ohlcv_one(self, symbol):
        ohlcv = self.exchange.fetch_ohlcv(
            symbol,
            timeframe=self.timeframe,
            limit=self.limit
        )
        # ohlcv columns: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

    def update(self):
        """
        Fetch latest OHLCV for all symbols and rebuild self.data.
        Call this each cycle before computing signals.
        """
        new_data = {}
        for sym in self.symbols:
            base = sym.split('/')[0]
            df = self._fetch_ohlcv_one(sym)

            # Approx VWAP (no per-trade data): using close as proxy.
            df['vwap'] = df['close']

            # CLV proxy in [-1,1]: (close - low)/(high - low)*2 - 1
            span = (df['high'] - df['low']).replace(0, np.nan)
            clv = ((df['close'] - df['low']) / span) * 2.0 - 1.0
            df['clv'] = clv.fillna(0.0)

            new_data[base] = df[['open', 'high', 'low', 'close', 'volume', 'vwap', 'clv']]

        self.data = new_data

    def build_matrix(self, col: str) -> pd.DataFrame:
        cols = []
        for sym, df in self.data.items():
            if col in df.columns:
                cols.append(df[col].rename(sym))
        if not cols:
            raise ValueError("No column '%s' found in live data." % col)
        return pd.concat(cols, axis=1).sort_index()

    def get_series_mats(self):
        """
        Return dict of matrices compatible with MultiStrategyEngine.signal_matrix.
        """
        open_m  = self.build_matrix('open')
        high_m  = self.build_matrix('high')
        low_m   = self.build_matrix('low')
        close_m = self.build_matrix('close')
        vol_m   = self.build_matrix('volume')
        vwap_m  = self.build_matrix('vwap')
        clv_m   = self.build_matrix('clv')

        return {
            'open': open_m,
            'high': high_m,
            'low': low_m,
            'close': close_m,
            'volume': vol_m,
            'vwap': vwap_m,
            'clv': clv_m,
        }


# =========================
# ROOSTOO CLIENT (using functions above)
# =========================

QUOTE_CURRENCY = "USD"
BASE_LOT_SIZES = {
    "BTC": 0.0001, "ETH": 0.001, "ARB": 0.1, "APT": 0.01, "AVAX": 0.01, "ADA": 1.0,
    "AAVE": 0.001, "BNB": 0.001, "CAKE": 0.1, "DOGE": 1.0, "DOT": 0.1, "EIGEN": 0.1,
    "ENA": 0.1, "FET": 0.1, "FIL": 0.01, "ICP": 0.01, "LINK": 0.01, "LTC": 0.001,
    "NEAR": 0.1, "ONDO": 0.1, "PENDLE": 0.01, "PENGU": 1.0, "POL": 0.1, "SEI": 0.1,
    "SOL": 0.001, "SUI": 0.01, "TAO": 0.0001, "UNI": 0.01, "WIF": 0.01, "WLD": 0.01,
    "XLM": 1.0, "XRP": 0.1, "ZEC": 0.001, "ZEN": 0.001
}

_EX_STEP = {}

def init_step_sizes():
    info = get_ex_info() or {}
    instruments = info.get("Instruments") or []
    for inst in instruments:
        pair = (inst.get("Pair") or "").upper()
        base = pair.split("/")[0] if "/" in pair else pair
        step = (inst.get("QuantityStepSize") or inst.get("StepSize") or inst.get("QtyStep"))
        if base:
            try:
                _EX_STEP[base] = float(step)
            except:
                _EX_STEP[base] = float(BASE_LOT_SIZES.get(base, 0.01))

def _round_step(base_symbol, qty):
    if not _EX_STEP:
        init_step_sizes()
    step = float(_EX_STEP.get(base_symbol.upper(), BASE_LOT_SIZES.get(base_symbol.upper(), 0.01)))
    units = int(qty // step)
    return round(units * step, 8)

def real_roostoo_place_market_order(base_symbol, side, qty, price=None):
    base = base_symbol.upper()
    side = side.upper()
    adj = _round_step(base, float(qty))
    if adj <= 0:
        return None
    place_order(base, side, f"{adj:.8f}")
    return {"OrderDetail": {"FillPrice": price}}

def roostoo_get_quote_balance_and_positions():
    data = get_balance() or {}
    quote = 0.0
    positions = {}
    # try common shapes from the mock API
    spot = data.get("SpotWallet") or {}
    if spot:
        for asset, vals in spot.items():
            free = float(vals.get("Free", 0) or 0)
            asset = asset.upper()
            if asset == QUOTE_CURRENCY:
                quote += free
            else:
                positions[asset] = positions.get(asset, 0.0) + free
        return quote, positions
    items = data.get("Balances") or data.get("Data") or data.get("data") or []
    for it in items:
        asset = (it.get('Asset') or it.get('asset') or it.get('Symbol') or it.get('symbol') or '').upper()
        free = it.get('Free')
        if free is None:
            free = it.get('Available', it.get('available', it.get('Total', it.get('total', 0.0))))
        amt = float(free or 0.0)
        if asset == QUOTE_CURRENCY:
            quote += amt
        elif asset:
            positions[asset] = amt
    return quote, positions

# default live order function alias
roostoo_place_market_order = real_roostoo_place_market_order

# To force never sell: comment SELL branch inside run_realtime_loop or flip long_only for BreakoutStrategy
# =========================
# EXECUTOR
# =========================

import numpy as np
import pandas as pd
from typing import Optional

# Assumes:
# - StrategyConfig
# - MultiStrategyEngine
# - All strategy classes
# are already defined above in the same file.

########################
# 1. Strategy Weights  #
########################

"""OLD WEIGHTS"""
# strategies_config = [
#     StrategyConfig(CLV(),                      ['clv'],            weight=0.094325, long_only=True),
#     StrategyConfig(FourierTransform(),         ['vwap'],           weight=0.057814, long_only=True),
#     StrategyConfig(MeanReversionStrategy(),    ['close'],          weight=0.178602, long_only=True),
#     StrategyConfig(Donchian_Breakout(),        None,               weight=0.422030, long_only=True),
#     StrategyConfig(Cross_Sectional_Momentum(), None,               weight=0.187230, long_only=True),

#     StrategyConfig(MovingAverageStrategy(),    ['vwap'],           weight=0.01,     long_only=True),
#     StrategyConfig(BreakoutStrategy(),         ['low', 'high'],    weight=0.01,     long_only=False),
#     StrategyConfig(BetaRegressionStrategy(),   ['low', 'high'],    weight=0.01,     long_only=True),
#     StrategyConfig(AutoRegressiveStrategy(),   ['close'],          weight=0.01,     long_only=True),
#     StrategyConfig(ARMAStrategy(),             ['close'],          weight=0.01,     long_only=True),
#     StrategyConfig(VWAPStrategy(),             ['close', 'volume'],weight=0.01,     long_only=True),
# ]

strategies_config = [
    # Positive-alpha dominated
    StrategyConfig(CLV(),                      ['clv'],            weight=0.81, long_only=True),
    StrategyConfig(FourierTransform(),         ['vwap'],           weight=0.51, long_only=True),
    StrategyConfig(MeanReversionStrategy(),    ['close'],          weight=0.84, long_only=True),
    StrategyConfig(Donchian_Breakout(),        None,               weight=7.16, long_only=True),
    StrategyConfig(Cross_Sectional_Momentum(), None,               weight=6.83, long_only=True),
    StrategyConfig(MovingAverageStrategy(),    ['vwap'],           weight=1.58, long_only=True),
    StrategyConfig(BreakoutStrategy(),         ['low','high'],     weight=1.07, long_only=False),
    StrategyConfig(BetaRegressionStrategy(),   ['low','high'],     weight=2.9,  long_only=True),
]

# Normalize strategy weights to sum to 1
w_sum_cfg = sum(cfg.weight for cfg in strategies_config)
if w_sum_cfg > 0:
    for cfg in strategies_config:
        cfg.weight = float(cfg.weight) / float(w_sum_cfg)

engine = MultiStrategyEngine(strategies_config, combine='weighted_sum')

########################
# 2. Live Config       #
########################
UNIVERSE_BASES = [
    "ZEN","AAVE","ADA","APT","ARB","AVAX","BNB","BTC","CAKE","DOGE","DOT","EIGEN","ENA",
    "ETH","FET","FIL","ICP","LINK","LTC","NEAR","OMNI","ONDO","PENDLE","PENGU","POL",
    "SEI","SOL","SUI","TAO","UNI","WIF","WLD","XLM","XRP","ZEC"
]

# Symbols for ccxt (Coinbase or other USD-quoted exchange)
SYMBOLS = [f"{base}/USD" for base in UNIVERSE_BASES]

COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_SECRET_KEY = os.getenv("COINBASE_SECRET_KEY")
COINBASE_PASSWORD = os.getenv("COINBASE_PASSWORD")

pipeline = LiveDataPipeline(
    exchange_id='coinbase',
    exchange_params={
        'apiKey': COINBASE_API_KEY,
        'secret': COINBASE_SECRET_KEY,
        'password': COINBASE_PASSWORD
    },
    symbols=SYMBOLS,
    timeframe='5m',
    limit=500
)



LAMBDA = 0.8          # EW memory for target weights
THRESHOLD = 0.02      # L1 diff threshold to trigger rebalance
MIN_NOTIONAL = 10.0   # Minimum USD value per trade


########################
# 3. Helpers           #
########################

_prev_target_weights = None  # persistent between iterations

def warm_strategies_with_history(days=14, timeframe='5m'):
    global _prev_target_weights
    ex = pipeline.exchange
    syms = pipeline.symbols

    step_ms = int(ex.parse_timeframe(timeframe) * 1000)
    need_bars = int(days * 24 * 60 * 60 * 1000 // step_ms)

    new_data = {}
    last_ts_list = []

    for sym in syms:
        base = sym.split('/')[0]

        # 1) Start from the latest window
        ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=500)
        if not ohlcv:
            continue
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # 2) Backfill older chunks until we reach target history
        while len(df) < need_bars:
            first_ms = int(df.index[0].timestamp() * 1000)
            prev_since = first_ms - 500 * step_ms
            chunk = ex.fetch_ohlcv(sym, timeframe=timeframe, since=prev_since, limit=500)
            if not chunk:
                break
            df2 = pd.DataFrame(chunk, columns=['timestamp','open','high','low','close','volume'])
            df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms', utc=True)
            df2.set_index('timestamp', inplace=True)
            df2.sort_index(inplace=True)
            df2 = df2[df2.index < df.index[0]]  # keep only strictly older bars
            if df2.empty:
                break
            df = pd.concat([df2, df]).sort_index()

        # 3) Add derived fields
        df['vwap'] = df['close']
        span = (df['high'] - df['low']).replace(0, np.nan)
        df['clv'] = (((df['close'] - df['low']) / span) * 2.0 - 1.0).fillna(0.0)

        # 4) Save and report freshness
        new_data[base] = df[['open','high','low','close','volume','vwap','clv']]
        last_ts = df.index[-1]
        last_ts_list.append(last_ts)
        now_ut = pd.Timestamp.now('UTC')
        age_min = (now_ut - last_ts).total_seconds() / 60.0
        print(f"WARM {base:>6} last={last_ts} age_min={age_min:.1f}")

    if last_ts_list:
        gmin = min(last_ts_list)
        gmax = max(last_ts_list)
        now_ut = pd.Timestamp.now('UTC')
        print(f"WARM RANGE first={gmin} last={gmax} now={now_ut} lag_last_min={(now_ut-gmax).total_seconds()/60:.1f}")
        last_5m = now_ut.floor('5T')
        if gmax >= last_5m:
            print(f"SUCCESS: OHLCV is updated to the nearest 5 min mark (expected={last_5m}, latest={gmax})")
        else:
            lag5 = (last_5m - gmax).total_seconds() / 60.0
            print(f"ERROR: OHLCV not up to date (expected={last_5m}, latest={gmax}, lag_min={lag5:.1f})")

    if not new_data:
        return

    old = pipeline.data
    pipeline.data = new_data
    series_mats = pipeline.get_series_mats()
    idx = series_mats['close'].index
    raw = engine.signal_matrix(idx, series_mats).iloc[-1].clip(lower=0.0)
    _prev_target_weights = (raw / raw.sum()).fillna(0.0) if raw.sum() > 0 else raw
    pipeline.data = old


def compute_target_weights(series_mats: dict) -> pd.Series:
    """
    Use MultiStrategyEngine to get ensemble signal for latest bar,
    then convert to normalized long-only weights.
    """
    index = series_mats['close'].index
    sig_mat = engine.signal_matrix(index, series_mats)
    raw = sig_mat.iloc[-1].clip(lower=0.0)

    if raw.sum() == 0:
        return pd.Series(0.0, index=raw.index)

    return raw / raw.sum()


def smooth_weights(new_w: pd.Series,
                   prev_w: Optional[pd.Series] = None) -> pd.Series:
    """
    Exponential smoothing between previous and new target weights.
    Uses LAMBDA as smoothing factor.
    """
    if prev_w is None:
        return new_w

    blended = LAMBDA * new_w + (1.0 - LAMBDA) * prev_w
    blended = blended.clip(lower=0.0)
    s = float(blended.sum())
    if s == 0.0:
        return prev_w
    return blended / s


def should_rebalance(target: pd.Series, current: pd.Series) -> bool:
    l1 = (target - current).abs().sum()
    return l1 >= THRESHOLD


def get_current_weights_from_balance(quote_balance,
                                     positions,
                                     prices) -> pd.Series:
    """
    Compute current portfolio weights from Roostoo balances,
    restricted to bases we have prices for. Always returns 0 instead of NaN.
    """
    assets = sorted(prices.keys())
    if not assets:
        return pd.Series(dtype=float)

    values = {}
    total = float(quote_balance)

    for base in assets:
        qty = float(positions.get(base, 0.0))
        px = float(prices.get(base, 0.0))
        val = qty * px
        values[base] = val
        total += val

    if total <= 0:
        return pd.Series(0.0, index=assets)

    w = pd.Series({a: values[a] / total for a in assets}, index=assets)
    return w.fillna(0.0)



def fetch_last_prices(series_mats: dict) -> dict:
    close_m = series_mats['close']
    last = close_m.iloc[-1]
    return {col: float(last[col]) for col in close_m.columns}


def process_trade_queue(queue, positions, quote_balance, limit=5, cooldown=60):
    pending = list(queue)
    while pending:
        batch = pending[:limit]
        pending = pending[limit:]
        retry = []
        for task in batch:
            base = task["base"]
            side = task["side"]
            qty = task["qty"]
            price = task["price"]
            res = roostoo_place_market_order(base, side, qty, price)
            if res:
                value = qty * price
                detail = res.get("OrderDetail") if isinstance(res, dict) else {}
                fill_price = price
                if isinstance(detail, dict):
                    fill_price = float(detail.get("FillPrice") or fill_price)
                append_trade(base, side, qty, fill_price)
                if side == "SELL":
                    quote_balance += value
                    positions[base] = max(0.0, positions.get(base, 0.0) - qty)
                else:
                    quote_balance -= value
                    positions[base] = positions.get(base, 0.0) + qty
            else:
                retry.append(task)
            time.sleep(0.25)
        pending = retry + pending
        if pending:
            print("Hit trade limit; pausing before next batch.")
            time.sleep(cooldown)
    return quote_balance


########################
# 4. Main Loop         #
########################

def run_realtime_loop(sleep_seconds=86400): ##CHANGE THIS TO CHANGE REBALANCE FREQUENCY ##
    global _prev_target_weights
    sell_queue = []
    buy_queue = []

    while True:
        try:
            # 1) Pull live market data
            pipeline.update()
            series_mats = pipeline.get_series_mats()
            prices = fetch_last_prices(series_mats)  # { 'AAVE': 123.4, ... }
            bases = sorted(prices.keys())

            if not bases:
                print("No live prices available, skipping this cycle.")
                time.sleep(sleep_seconds)
                continue

            # 2) Compute target weights from strategies
            index = series_mats['close'].index
            raw_target = engine.signal_matrix(index, series_mats).iloc[-1].clip(lower=0.0)

            if raw_target.sum() == 0:
                print("All strategy signals are zero; skipping rebalance.")
                time.sleep(sleep_seconds)
                continue

            target_base = (raw_target / raw_target.sum()).reindex(bases).fillna(0.0)

            # 3) Smooth weights vs previous
            target_smoothed = smooth_weights(target_base, _prev_target_weights)
            if _prev_target_weights is None:
                _prev_target_weights = target_smoothed

            # 4) Get Roostoo portfolio
            quote_balance, positions = roostoo_get_quote_balance_and_positions()

            # compute current weights ONLY on assets we have prices for (bases)
            current_w = get_current_weights_from_balance(quote_balance, positions, prices)

            # Debug prints
            print("Target weights:", target_smoothed.to_dict())
            print("Current weights:", current_w.to_dict())
            print("Quote balance:", quote_balance, "Positions:", positions)

            # 5) Rebalance decision
            tw = target_smoothed.reindex(bases).fillna(0.0)
            cw = current_w.reindex(bases).fillna(0.0)
            l1 = (tw - cw).abs().sum()
            print("L1 diff:", float(l1))

            if l1 < THRESHOLD:
                print("No rebalance needed.")
                time.sleep(sleep_seconds)
                continue

            # 6) Compute total equity using only tradable bases
            total_equity = float(quote_balance)
            for base in bases:
                qty = float(positions.get(base, 0.0))
                px = float(prices.get(base, 0.0))
                if qty > 0 and px > 0:
                    total_equity += qty * px

            if total_equity <= 0:
                print("No equity; skipping.")
                time.sleep(sleep_seconds)
                continue

            # 7) Queue trades to honor API rate limits
            sell_pool = 0.0
            buy_pool = 0.0
            for base in bases:
                px = float(prices.get(base, 0.0))
                if not np.isfinite(px) or px <= 0:
                    continue

                t_w = float(tw.get(base, 0.0))
                if not np.isfinite(t_w) or t_w < 0:
                    t_w = 0.0

                cur_qty = float(positions.get(base, 0.0))
                if not np.isfinite(cur_qty) or cur_qty < 0:
                    cur_qty = 0.0

                cur_val = cur_qty * px
                tgt_val = t_w * total_equity

                if not np.isfinite(tgt_val) or not np.isfinite(cur_val):
                    continue
                diff_val = tgt_val - cur_val
                if not np.isfinite(diff_val):
                    continue

                # Tiny adjustment -> skip
                if abs(diff_val) < MIN_NOTIONAL:
                    continue

                if diff_val > 0:
                    available_cash = quote_balance + sell_pool - buy_pool
                    if available_cash <= 0:
                        continue
                    buy_val = min(diff_val, available_cash)
                    if buy_val < MIN_NOTIONAL:
                        continue
                    qty = buy_val / px
                    buy_queue.append({
                        "base": base,
                        "side": "BUY",
                        "qty": qty,
                        "price": px
                    })
                    buy_pool += buy_val
                else:
                    desired_sell_val = -diff_val
                    max_sell_val = cur_val            # don't sell more than we own
                    sell_val = min(desired_sell_val, max_sell_val)
                    if sell_val < MIN_NOTIONAL:
                        continue
                    qty = sell_val / px
                    sell_queue.append({
                        "base": base,
                        "side": "SELL",
                        "qty": qty,
                        "price": px
                    })
                    sell_pool += sell_val

            trade_queue = sell_queue + buy_queue
            if not trade_queue:
                print("No qualified trades after thresholds.")
                time.sleep(sleep_seconds)
                continue

            quote_balance = process_trade_queue(trade_queue, positions, quote_balance)
            _prev_target_weights = tw

        except Exception as e:
            # Do NOT silently die; show the error so you know why no trades happen
            print("Error in realtime loop:", repr(e))

        time.sleep(sleep_seconds)



# =========================
# START EXECUTOR
# =========================
# (Orders are sent via Roostoo functions defined above)

DRY_RUN = False
if __name__ == "__main__":
    if DRY_RUN:
        def mock_place_market_order(base_symbol, side, qty, price=None):
            print("[DRY RUN]", side, f"{qty:.6f}", base_symbol, "@", price)
            return {"OrderDetail": {"FillPrice": price}}
        roostoo_place_market_order = mock_place_market_order
        print("Starting Multi-Strategy live executor in DRY RUN mode...")
    else:
        print("Starting Multi-Strategy live executor in LIVE mode (REAL ORDERS will be sent via Roostoo)...")

    # Warm strategies with 5m data before live loop
    warm_strategies_with_history(days=14, timeframe='5m')


    run_realtime_loop(sleep_seconds=86400)
