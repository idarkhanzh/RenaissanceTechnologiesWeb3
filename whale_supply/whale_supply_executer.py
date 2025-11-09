import time
import torch
import requests
import datetime
import numpy as np
import pandas as pd
import logging, os
from sklearn.preprocessing import StandardScaler
import joblib  # pip install joblib

class WhaleSupplySignal:
    HORUS_API_KEY   = "c17248ed7a55f8e016bd95dfe0e5900be290db72b4ca533ced39b0aeb54111c1"
    HORUS_FLOW_URL  = "https://api-horus.com/addresses/whale_net_flow"
    HORUS_PRICE_URL = "https://api-horus.com/market/price"
    CHAIN, ASSET, INTERVAL = "bitcoin", "BTC", "1d"
    MODEL_PATH      = "models/whale_btc_model.pt"
    SCALER_PATH     = "models/whale_scaler.pkl"

    BUY_TH  = 0.51
    SELL_TH = 0.49

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        self.model = self._load_model()
        self.scaler = None  # set during backtest or loaded if available

    # --- model ---
    def _load_model(self):
        class WhaleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(1, 16), torch.nn.ReLU(),
                    torch.nn.Linear(16, 8), torch.nn.ReLU(),
                    torch.nn.Linear(8, 1), torch.nn.Sigmoid()
                )
            def forward(self, x): return self.net(x)
        m = WhaleNet()
        m.load_state_dict(torch.load(self.MODEL_PATH, map_location="cpu"))
        m.eval()
        return m

    # --- fetch ---
    def _get_whale_data(self, start_ts, end_ts):
        params  = {"chain": self.CHAIN, "interval": self.INTERVAL, "start": start_ts, "end": end_ts, "format":"json"}
        headers = {"X-API-Key": self.HORUS_API_KEY}
        r = requests.get(self.HORUS_FLOW_URL, headers=headers, params=params, timeout=30); r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.rename(columns={"whale_net_flow": "flow"})
        return df[["timestamp","flow"]]

    def _get_price_data(self, start_ts, end_ts):
        params  = {"asset": self.ASSET, "interval": self.INTERVAL, "start": start_ts, "end": end_ts, "format":"json"}
        headers = {"X-API-Key": self.HORUS_API_KEY}
        r = requests.get(self.HORUS_PRICE_URL, headers=headers, params=params, timeout=30); r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.rename(columns={"price": "close"})
        return df[["timestamp","close"]]

    # --- inference helpers ---
    def _ensure_scaler(self, fit_series: np.ndarray):
        """
        Prefer loading the persisted scaler. If missing, fit on provided training series and save it.
        """
        if os.path.exists(self.SCALER_PATH):
            self.scaler = joblib.load(self.SCALER_PATH)
            logging.info("Loaded scaler from disk.")
            return
        self.scaler = StandardScaler().fit(fit_series.reshape(-1,1))
        os.makedirs(os.path.dirname(self.SCALER_PATH), exist_ok=True)
        joblib.dump(self.scaler, self.SCALER_PATH)
        logging.info("Fitted and saved scaler from training window.")

    def _predict_signal_vec(self, flow_values: np.ndarray) -> np.ndarray:
        """
        Vectorized prediction for a 1D numpy array of flows.
        Assumes self.scaler is ready.
        """
        Xs = self.scaler.transform(flow_values.reshape(-1,1))
        with torch.no_grad():
            out = self.model(torch.tensor(Xs, dtype=torch.float32)).numpy().flatten()
        return out

    # --- backtest ---
    def backtest(self):
        start_date = datetime.datetime(2024, 8, 12)
        end_date   = datetime.datetime.now()
        start_ts, end_ts = int(start_date.timestamp()), int(end_date.timestamp())

        # fetch
        df_flow  = self._get_whale_data(start_ts, end_ts)
        df_price = self._get_price_data(start_ts, end_ts)
        df = pd.merge_asof(df_flow.sort_values("timestamp"),
                           df_price.sort_values("timestamp"),
                           on="timestamp", direction="nearest").dropna()

        # time-based split to fit scaler on training portion (first 80%)
        t0, t1 = df["timestamp"].min(), df["timestamp"].max()
        split_time = t0 + (t1 - t0) * 0.8
        train_mask = df["timestamp"] <= split_time
        val_mask   = df["timestamp"] >  split_time

        # set scaler (load if available; else fit on training window’s flows)
        self._ensure_scaler(df.loc[train_mask, "flow"].values)

        # predict signals for all rows, then we’ll only trade on validation rows
        df["signal_strength"] = self._predict_signal_vec(df["flow"].values)
        print("\nSignal strength distribution (all rows):")
        print(df["signal_strength"].describe())

        # threshold to discrete signals
        df["signal"] = np.where(df["signal_strength"] > self.BUY_TH, 1,
                         np.where(df["signal_strength"] < self.SELL_TH, 0, np.nan))

        # only execute in validation (like live)
        df.loc[train_mask, "signal"] = np.nan

        # simulate trades (no shorting)
        trades, in_pos = [], False
        buy_t = buy_p = None
        for i, row in df.iterrows():
            sig = row["signal"];  ts = row["timestamp"];  p = row["close"]
            if np.isnan(sig):
                continue
            if not in_pos and sig == 1:
                in_pos, buy_t, buy_p = True, ts, p
            elif in_pos and sig == 0:
                in_pos = False
                sell_t, sell_p = ts, p
                ret = (sell_p / buy_p - 1) * 100
                trades.append({
                    "buy_date":  buy_t.strftime("%Y-%m-%d"),
                    "buy_price": round(buy_p, 2),
                    "sell_date": sell_t.strftime("%Y-%m-%d"),
                    "sell_price": round(sell_p, 2),
                    "return_%":  round(ret, 2)
                })

        trade_df = pd.DataFrame(trades)
        if trade_df.empty:
            print("\n⚠️ No trades executed. (If signal still flat, lower gap: try BUY 0.5005 / SELL 0.4995)")
            return trade_df

        print("\n=== Backtest Trade Log (validation only) ===")
        print(trade_df.to_string(index=False))
        total_ret = trade_df["return_%"].sum()
        print(f"\n💰 Total Return: {total_ret:.2f}% over {len(trade_df)} trades.")
        return trade_df

if __name__ == "__main__":
    WhaleSupplySignal().backtest()
