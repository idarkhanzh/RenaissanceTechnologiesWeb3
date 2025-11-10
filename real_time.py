import os
import re
import glob
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
import warnings
import quantstats as qs

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

warnings.simplefilter('ignore')


# ----- ALL STRATS ----- #

short_period = 6
long_period = 12

class MovingAverageStrategy:
    def __init__(self, short=short_period, long=long_period):
        self.short = short
        self.long = long
    def signal(self, index, series):
        x = series.values.astype(float)
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
            sig = (a - b) > 0
            vals.append(sig.astype(int))
        avg = np.nanmean(np.column_stack(vals), axis=1)
        return pd.DataFrame({'value': avg}, index=index)


lookback_period = 12
zscore_lookback_period = 72
signal_threshold = 0.75


class BetaRegressionStrategy:
    def __init__(self, beta_lookback_period=lookback_period, zscore_lookback_period=zscore_lookback_period,
                 signal_threshold=signal_threshold):
        self.beta_lookback_period = beta_lookback_period
        self.zscore_lookback_period = zscore_lookback_period
        self.signal_threshold = signal_threshold

    def signal(self, index, low_series, high_series):
        x = pd.Series(low_series.values.astype(float), index=index)
        y = pd.Series(high_series.values.astype(float), index=index)
        w = self.beta_lookback_period
        mx = x.rolling(w).mean()
        my = y.rolling(w).mean()
        mxx = (x * x).rolling(w).mean()
        myy = (y * y).rolling(w).mean()
        mxy = (x * y).rolling(w).mean()
        varx = mxx - mx * mx
        vary = myy - my * my
        cov = mxy - mx * my
        beta_s = cov / varx
        corr = cov / (np.sqrt(varx) * np.sqrt(vary))
        r2_s = corr * corr
        z = (beta_s - beta_s.rolling(self.zscore_lookback_period).mean()) / beta_s.rolling(
            self.zscore_lookback_period).std()
        strength = z * r2_s
        weight = strength.where(beta_s > self.signal_threshold, 0.0)
        return pd.DataFrame({'value': weight}, index=index)


threshold = -1


class CLV:
    def __init__(self, threshold=threshold):
        self.threshold = threshold

    def signal(self, index, clv_series):
        y = (clv_series == self.threshold).fillna(0)
        return pd.DataFrame({'value': y.astype(int)}, index=index)


ma_long = 50
ma_short = 20
breakout_lookback = 20
atr_period = 14

class BreakoutStrategy:
    def __init__(self, ma_long=ma_long, ma_short=ma_short, breakout_lookback=breakout_lookback, atr_period=atr_period):
        self.ma_long = ma_long
        self.ma_short = ma_short
        self.breakout_lookback = breakout_lookback
        self.atr_period = atr_period

    def signal(self, index: pd.DatetimeIndex,
               low_series: pd.Series,
               high_series: pd.Series) -> pd.DataFrame:
        close = (low_series + high_series) / 2.0

        ma_long = close.rolling(self.ma_long).mean()
        ma_short = close.rolling(self.ma_short).mean()

        high_n = close.rolling(self.breakout_lookback).max()
        low_n = close.rolling(self.breakout_lookback).min()

        prev_close = close.shift()
        tr1 = high_n - low_n
        tr2 = (high_n - prev_close).abs()
        tr3 = (low_n - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        breakout_up = (close > ma_long) & (close > high_n.shift(1))
        breakout_down = (close < ma_short) & (close < low_n.shift(1))

        sig = pd.Series(0.0, index=index)
        sig.loc[breakout_up] = 1.0
        sig.loc[breakout_down] = -1.0

        sig_smooth = sig.rolling(window=self.ma_short, min_periods=1).mean().fillna(0.0)

        return pd.DataFrame({'value': sig_smooth.astype(float)}, index=index)


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
        inst_reg = max(8, self.recents // 6)

        for t in range(self.recents, len(s) + 1, step):
            sw = s.iloc[:t].tail(self.recents)
            spec_t = self.discrete_fourier_transform(sw, self.recents)
            el, em, eh = self.compute_energy(spec_t, low_f, high_f)

            low_ts = self.reconstruct_time_series(sw, low_f, high_f, "low")
            mid_ts = self.reconstruct_time_series(sw, low_f, high_f, "middle")

            sl = self.compute_slope(low_ts, inst_reg)
            sm = self.compute_slope(mid_ts, inst_reg)

            ok = (
                (el >= self.low_energy_threshold) and
                (em >= self.mid_energy_threshold) and
                (sl >= self.low_slope_threshold) and
                (sm >= self.mid_slope_threshold)
            )

            if ok:
                w = max(
                    0.0,
                    1.0 - (eh / el if el > 0 else 1.0) - (eh / em if em > 0 else 1.0)
                )
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
window = 252
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



lookback = 30
top_frac = 0.2
vol_threshold = 5000000
vol_window = 15
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

# live_data.py

import ccxt
import pandas as pd
import numpy as np
import time

class LiveDataPipeline:
    """
    Fetches live OHLCV via ccxt and builds the same structure
    your MultiStrategyEngine expects: matrices for open/high/low/close/volume/vwap/clv.
    """

    def __init__(self,
                 exchange_id='binance',
                 symbols=None,
                 timeframe='5m',
                 limit=1000):
        self.exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        self.timeframe = timeframe
        self.limit = limit
        # symbols like ['BTC/USDT','ETH/USDT',...]
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        # internal cache
        self.data = {}  # { 'BTC': DataFrame, ... }

    def _fetch_ohlcv_one(self, symbol):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.limit)
        # ohlcv columns: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
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

            # Approx VWAP (no per-trade data): price * volume / volume over bar
            # Here VWAP == close as an approximation, can refine if needed.
            df['vwap'] = df['close']

            # Simple CLV proxy in [-1,1]
            # (close - low)/(high - low) *2 -1 ; handle div0
            span = (df['high'] - df['low']).replace(0, np.nan)
            clv = ((df['close'] - df['low']) / span) * 2 - 1
            df['clv'] = clv.fillna(0.0)

            new_data[base] = df[['open','high','low','close','volume','vwap','clv']]

        self.data = new_data

    def build_matrix(self, col: str) -> pd.DataFrame:
        cols = []
        for sym, df in self.data.items():
            if col in df.columns:
                cols.append(df[col].rename(sym))
        if not cols:
            raise ValueError(f"No column '{col}' found in live data.")
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


# roostoo_client.py

import requests
import hashlib
import hmac
import time
import os

BASE_URL = "https://mock-api.roostoo.com"

class RoostooClient:
    def __init__(self,
                 api_key=None,
                 secret=None,
                 base_url=BASE_URL,
                 quote_currency="USD"):
        self.api_key = api_key or os.getenv("ROOSTOO_API_KEY")
        self.secret = secret or os.getenv("ROOSTOO_SECRET")
        self.base_url = base_url
        self.quote = quote_currency

    def _sign(self, params: dict) -> str:
        query_string = '&'.join(["{}={}".format(k, params[k])
                                 for k in sorted(params.keys())])
        us = self.secret.encode('utf-8')
        m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
        return m.hexdigest()

    def _auth_headers(self, payload: dict):
        return {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": self._sign(payload)
        }

    def get_balance(self):
        payload = {
            "timestamp": int(time.time()) * 1000,
        }
        r = requests.get(
            self.base_url + "/v3/balance",
            params=payload,
            headers=self._auth_headers(payload),
            timeout=5,
        )
        r.raise_for_status()
        return r.json()

    def get_quote_balance_and_positions(self):
        """
        Parse Roostoo balance into:
          - quote_balance: float (e.g. USD)
          - positions: { 'BTC': qty, ... }

        Expected schema (as seen in logs):
        {
          "Success": True,
          "ErrMsg": "",
          "SpotWallet": {
             "BTC": {"Free": 0.00094, "Lock": 0},
             "ETH": {"Free": 0.0278, "Lock": 0},
             "USD": {"Free": 46141.04, "Lock": 0},
             ...
          },
          "MarginWallet": { ... }
        }
        """
        data = self.get_balance()
        print("Roostoo raw balance:", data)  # keep for visibility

        quote_balance = 0.0
        positions = {}

        if not isinstance(data, dict) or not data.get("Success", False):
            return 0.0, {}

        spot = data.get("SpotWallet", {}) or {}

        for asset, vals in spot.items():
            # vals like {"Free": 0.446, "Lock": 0}
            free = vals.get("Free", 0) or 0
            try:
                free = float(free)
            except (TypeError, ValueError):
                free = 0.0

            if free <= 0:
                continue

            asset_up = asset.upper()
            if asset_up == self.quote.upper():
                # e.g. USD
                quote_balance += free
            else:
                positions[asset_up] = positions.get(asset_up, 0.0) + free

        return quote_balance, positions

    def place_market_order(self, base_symbol: str, side: str, qty: float):
        if qty <= 0:
            return

        payload = {
            "timestamp": int(time.time()) * 1000,
            "pair": f"{base_symbol}/{self.quote}",
            "side": side.upper(),
            "quantity": float(qty),
            "type": "MARKET",
        }

        r = requests.post(
            self.base_url + "/v3/place_order",
            data=payload,
            headers=self._auth_headers(payload),
            timeout=5,
        )
        # You may want to handle errors/logging here.
        try:
            r.raise_for_status()
        except Exception as e:
            print("Order error:", e, r.text)
        else:
            print("Order placed:", payload, r.text)


# executor.py

import time
import numpy as np
import pandas as pd

# ^ replace `your_multistrat_module` with the actual module where StrategyConfig & MultiStrategyEngine live
# and import your strategy classes from wherever you defined them.

########################
# 1. Strategy Weights  #
########################

# Use the alpha-based weights you derived earlier (no zeros).
strategies_config = [
    StrategyConfig(CLV(),                      ['clv'],            weight=0.094325, long_only=True),
    StrategyConfig(FourierTransform(),         ['vwap'],           weight=0.057814, long_only=True),
    StrategyConfig(MeanReversionStrategy(),    ['close'],          weight=0.178602, long_only=True),
    StrategyConfig(Donchian_Breakout(),        None,               weight=0.422030, long_only=True),
    StrategyConfig(Cross_Sectional_Momentum(), None,               weight=0.187230, long_only=True),

    StrategyConfig(MovingAverageStrategy(),    ['vwap'],           weight=0.01,     long_only=True),
    StrategyConfig(BreakoutStrategy(),         ['low','high'],     weight=0.01,     long_only=False),
    StrategyConfig(BetaRegressionStrategy(),   ['low','high'],     weight=0.01,     long_only=True),
    StrategyConfig(AutoRegressiveStrategy(),   ['close'],          weight=0.01,     long_only=True),
    StrategyConfig(ARMAStrategy(),             ['close'],          weight=0.01,     long_only=True),
    StrategyConfig(VWAPStrategy(),             ['close','volume'], weight=0.01,     long_only=True),
]

engine = MultiStrategyEngine(strategies_config, combine='weighted_sum')

########################
# 2. Live Config       #
########################

# Symbols must match ccxt & your internal naming (base symbol = left side).
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # extend as desired

pipeline = LiveDataPipeline(
    exchange_id='binance',
    symbols=SYMBOLS,
    timeframe='5m',
    limit=1500
)

roostoo = RoostooClient()

LAMBDA = 0.8          # exponential weight memory
THRESHOLD = 0.02      # turnover trigger (L1 diff in weights)
MIN_NOTIONAL = 10.0   # minimal USD per trade to avoid dust


########################
# 3. Helpers           #
########################

_prev_target_weights = None  # across loop iterations

def compute_target_weights(series_mats: dict) -> pd.Series:
    """
    Use MultiStrategyEngine to get ensemble signal for latest bar,
    then map to long-only normalized weights.
    """
    index = series_mats['close'].index
    sig_mat = engine.signal_matrix(index, series_mats)

    # last row = latest ensemble scores for each asset
    raw = sig_mat.iloc[-1].clip(lower=0.0)

    if raw.sum() == 0:
        # no conviction -> stay as we are (handled at caller)
        return pd.Series(0.0, index=raw.index)

    return raw / raw.sum()


def smooth_weights(new_w: pd.Series, prev_w: pd.Series | None) -> pd.Series:
    if prev_w is None:
        return new_w
    blended = LAMBDA * new_w + (1 - LAMBDA) * prev_w
    s = blended.clip(lower=0.0).sum()
    if s == 0:
        return prev_w
    return blended / s


def should_rebalance(target: pd.Series, current: pd.Series) -> bool:
    l1 = (target - current).abs().sum()
    return l1 >= THRESHOLD


def get_current_weights_from_balance(quote_balance, positions, prices) -> pd.Series:
    """
    Compute current portfolio weights from Roostoo balances.
    positions: { 'BTC': qty, ...}
    prices: { 'BTC': last_price, ...}
    """
    assets = sorted(prices.keys())
    values = {}
    total = quote_balance
    for base in assets:
        if base in prices:  # only BTC/ETH/SOL
            qty = positions.get(base, 0.0)
            val = qty * prices[base]
            values[base] = val
            total += val

    if total <= 0:
        return pd.Series(0.0, index=assets)

    return pd.Series({a: values[a] / total for a in assets}, index=assets)


def fetch_last_prices(series_mats: dict) -> dict:
    close_m = series_mats['close']
    last = close_m.iloc[-1]
    return {col: float(last[col]) for col in close_m.columns}


########################
# 4. Main Loop         #
########################

def run_realtime_loop(sleep_seconds=300):
    global _prev_target_weights

    while True:
        try:
            # 1) Update data
            pipeline.update()
            series_mats = pipeline.get_series_mats()
            prices = fetch_last_prices(series_mats)
            bases = sorted(prices.keys())

            # 2) Compute target weights from signals
            target_base = compute_target_weights(series_mats)

            # 3) Smooth vs previous targets
            target_smoothed = smooth_weights(target_base, _prev_target_weights)
            if _prev_target_weights is None:
                _prev_target_weights = target_smoothed

            # 4) Get current Roostoo portfolio
            quote_balance, positions = roostoo.get_quote_balance_and_positions()
            current_w = get_current_weights_from_balance(quote_balance, positions, prices)

            # 5) Decide if we rebalance
            print("Target weights:", target_smoothed.to_dict())
            print("Current weights:", current_w.to_dict())
            print("Quote balance:", quote_balance, "Positions:", positions)
            if not should_rebalance(target_smoothed.reindex(current_w.index).fillna(0.0),
                                    current_w.reindex(target_smoothed.index).fillna(0.0)):
                print("No rebalance needed.")
                time.sleep(sleep_seconds)
                continue

            # 6) Compute target USD allocation & order sizes
            total_equity = quote_balance + sum(
                positions.get(b, 0.0) * prices[b] for b in bases
            )
            if total_equity <= 0:
                print("No equity; skipping.")
                time.sleep(sleep_seconds)
                continue

            # ensure alignment
            target_smoothed = target_smoothed.reindex(bases).fillna(0.0)
            current_w = current_w.reindex(bases).fillna(0.0)

            for base in bases:
                tgt_val = float(target_smoothed[base]) * total_equity
                cur_qty = float(positions.get(base, 0.0))
                cur_val = cur_qty * prices[base]
                diff_val = tgt_val - cur_val

                if abs(diff_val) < MIN_NOTIONAL:
                    continue

                qty = abs(diff_val) / prices[base]
                side = "BUY" if diff_val > 0 else "SELL"

                roostoo.place_market_order(base, side, qty)

            _prev_target_weights = target_smoothed

        except Exception as e:
            print("Error in realtime loop:", e)

        time.sleep(sleep_seconds)

# ====== CONFIGURE ROOSTOO + START LIVE EXECUTOR ======

# 1) Set your Roostoo API credentials here
#    (or set ROOSTOO_API_KEY / ROOSTOO_SECRET as env vars and leave these as None)

ROOSTOO_API_KEY = "UqIHb7BC5QgKVMjZKAhzARLUa6jWvvgO3GO1OxdVMqXBWVPJsqsha33VIvh6KFrx"   # <-- replace if needed
ROOSTOO_SECRET = "KBpaEeVmYVussNCI5ewdTv7jTmVJ6S0ZGWxvIkpKz5xMoLGsVwWYMKek0a5XeRAD"   # <-- replace if needed

# Override the earlier generic client (if any) with an explicitly keyed one
roostoo = RoostooClient(
    api_key=ROOSTOO_API_KEY or os.getenv("ROOSTOO_API_KEY"),
    secret=ROOSTOO_SECRET or os.getenv("ROOSTOO_SECRET"),
)

if __name__ == "__main__":
    # ========= LIVE / DRY-RUN SWITCH =========
    DRY_RUN = False  # <<-- SET TO False TO SEND REAL ORDERS

    if DRY_RUN:
        # Monkey-patch to log instead of sending
        _real_place_order = roostoo.place_market_order

        def _mock_place_market_order(base_symbol, side, qty):
            print(f"[DRY RUN] {side} {qty:.6f} {base_symbol}")

        roostoo.place_market_order = _mock_place_market_order
        print("Starting Multi-Strategy live executor in DRY RUN mode...")
    else:
        print("Starting Multi-Strategy live executor in LIVE mode (REAL ORDERS will be sent)...")

    # Align loop sleep with your timeframe (5m candles)
    run_realtime_loop(sleep_seconds=300)

