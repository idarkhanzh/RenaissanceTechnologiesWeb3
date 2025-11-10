#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import hashlib
import hmac
import time
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
import ccxt
import os


API_KEY = "UqIHb7BC5QgKVMjZKAhzARLUa6jWvvgO3GO1OxdVMqXBWVPJsqsha33VIvh6KFrx"
SECRET = "KBpaEeVmYVussNCI5ewdTv7jTmVJ6S0ZGWxvIkpKz5xMoLGsVwWYMKek0a5XeRAD"

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
    payload = { "timestamp": int(time.time()) * 1000 }  # ms
    if pair: payload["pair"] = pair
    r = requests.get(BASE_URL + "/v3/ticker", params=payload)
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
    try:
        return r.json()
    except:
        return None


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
    

####################### DO NOT CHANGE ANY OF THE FUNCTIONS ABOVE THIS LINE ############################

"""DONCHIAN BREAKOUT STRATEGY LOGIC FROM IPYNB"""
class Donchian_Breakout:
    def __init__(self, N=72):
        self.N = N
        self.requires_panel = True  # ensure build_weights sends full matrices

    def signal(self, index, open, high, low, close, volume, vwap, clv):
        h = high.resample('1h').max()
        l = low.resample('1h').min()
        c = close.resample('1h').last()
        dc_high = h.rolling(self.N).max().shift(1)
        dc_low  = l.rolling(self.N).min().shift(1)
        pos = pd.DataFrame(0, index=c.index, columns=c.columns, dtype=int)
        for col in c.columns:
            close_arr = c[col].to_numpy()
            dch = dc_high[col].to_numpy()
            dcl = dc_low[col].to_numpy()
            p = np.zeros(len(close_arr), dtype=int)
            for i in range(1, len(close_arr)):
                if p[i-1] <= 0 and close_arr[i] > dch[i]:
                    p[i] = 1
                elif p[i-1] > 0:
                    p[i] = 1
                    if close_arr[i] < dcl[i]:
                        p[i] = 0
            pos[col] = p
        picks = pos.reindex(index, method='ffill').fillna(0).astype(int)
        return picks

##############################################################


#39 coins total is correct
"""COINS"""  
# Step 1: coin universe (exclude 1000CHEEMSUSDT)
COINS = ["ZEN","AAVE","ADA","APT","ARB","AVAX","BNB","BTC","CAKE","CFX","DOGE","DOT","EIGEN","ENA",
         "ETH","FET","FIL","ICP","LINK","LISTA","LTC","NEAR","OMNI","ONDO","PENDLE","PENGU","POL",
         "SEI","SOL","SUI","TAO","TON","TRX","UNI","WIF","WLD","XLM","XRP","ZEC"]

# Step 2: ccxt exchange
EX = ccxt.coinbase({"enableRateLimit": True})

# Step 3: fetch latest prices (USD)
def fetch_prices():
    syms = [f"{c}/USD" for c in COINS]
    tks = EX.fetch_tickers(syms)
    return {c: float(tks[f"{c}/USD"]["last"]) for c in COINS if f"{c}/USD" in tks and tks[f"{c}/USD"].get("last")}

# Step 4: build hourly ohlc
def build_hourly(df):
    o = df.resample("1h").first()
    h = df.resample("1h").max()
    l = df.resample("1h").min()
    c = df.resample("1h").last()
    return o, h, l, c

# Step 5: warm start hourly (≈1 month)
def warm_hourly_ticks(hours=720):
    EX.load_markets()
    frames = []
    limit = min(hours, 720)
    for c in COINS:
        sym = f"{c}/USD"
        if sym not in EX.markets: continue
        data = EX.fetch_ohlcv(sym, timeframe="1h", limit=limit)
        rows = []
        for k in data:
            ts = pd.to_datetime(k[0], unit="ms", utc=True)
            h_, l_, cl_ = k[2], k[3], k[4]
            rows.append((ts + pd.Timedelta(minutes=1), h_))
            rows.append((ts + pd.Timedelta(minutes=2), l_))
            rows.append((ts + pd.Timedelta(minutes=3), cl_))
        dfc = pd.DataFrame(rows, columns=["ts", c]).set_index("ts")
        frames.append(dfc)
    if not frames: return pd.DataFrame()
    base = frames[0]
    for f in frames[1:]: base = base.join(f, how="outer")
    return base.sort_index()

# Step 6: sizing
def size_for_coin(usd_free, price, usd_per_trade=50.0, amount_precision=3):
    v = min(usd_per_trade, usd_free * 0.1)
    raw = v / max(price, 1e-9)
    step = 1.0 if amount_precision <= 0 else 10 ** (-amount_precision)
    qty = np.floor(raw / step) * step
    if qty < step:
        return 0.0
    return float(round(qty, max(amount_precision, 0)))

# Step 7: trade log
LOG_FILE = "trade_log.csv"
def init_trade_log():
    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "a") as f:
            f.write("timestamp,coin,side,price,qty,dc_high,dc_low,pos_before,pos_after,status,msg\n")
def log_trade(coin, side, price, qty, dc_high, dc_low, pos_before, pos_after, status, msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()},{coin},{side},{price},{qty},{dc_high},{dc_low},{pos_before},{pos_after},{status},{msg}\n")

# Step 8: multi-coin bot
class MultiDonchianBot:
    def __init__(self, lookback=72, poll_sec=30, usd_per_trade=50.0):
        self.N = lookback
        self.poll = poll_sec
        self.usd_per_trade = usd_per_trade
        self.strategy = Donchian_Breakout(N=lookback)
        self.tick_df = warm_hourly_ticks(720)
        self.pos = {c: 0 for c in COINS}
        self.amount_precision = {c: 3 for c in COINS}
        self._load_precisions()
        init_trade_log()
        print(f"Init coins={len(COINS)} N={lookback} warm_rows={len(self.tick_df)}")

    def _load_precisions(self):
        try:
            info = get_ex_info()
            tp = info.get("TradePairs", {}) if isinstance(info, dict) else {}
            for coin in COINS:
                meta = tp.get(f"{coin}/USD")
                if meta and isinstance(meta, dict):
                    ap = meta.get("AmountPrecision")
                    if isinstance(ap, int):
                        self.amount_precision[coin] = ap
        except Exception as e:
            print("precision load error", e)

    def update_ticks(self):
        prices = fetch_prices()
        if not prices: return
        ts = pd.Timestamp.utcnow()
        self.tick_df = pd.concat([self.tick_df, pd.DataFrame([prices], index=[ts])])
        if len(self.tick_df) > 15000: self.tick_df = self.tick_df.tail(12000)

    def compute_signal(self):
        if self.tick_df.empty: return None
        o,h,l,c = build_hourly(self.tick_df)
        if len(c.index) < self.N + 2: return None
        vol = c.copy(); vol[:] = 0
        vwap = c.copy()
        clv = c.copy(); clv[:] = 0
        return self.strategy.signal(c.index, o, h, l, c, vol, vwap, clv)

    def trade(self, sig):
        if sig is None: return
        o,h,l,c = build_hourly(self.tick_df)
        last_idx = sig.index[-1]
        row = sig.loc[last_idx]
        bal = get_balance()
        # Adjust balance parsing: real schema prints SpotWallet with assets list; fallback to previous structure
        usd_free = 0.0
        if isinstance(bal, dict):
            if "SpotWallet" in bal:  # newer observed schema
                w = bal.get("SpotWallet", {})
                # try direct USD free
                if isinstance(w, dict) and "USD" in w:
                    usd_free = float(w["USD"].get("Free", 0) or 0)
                # or iterate list of assets
                assets = w.get("Assets") if isinstance(w, dict) else None
                if assets and usd_free == 0:
                    for a in assets:
                        if a.get("Asset") == "USD":
                            usd_free = float(a.get("Free", 0) or 0)
                            break
            elif bal.get("Success"):
                usd_free = float(bal.get("Data", {}).get("USD", {}).get("Free", 0) or 0)
        prices_now = self.tick_df.iloc[-1].to_dict()
        for coin in COINS:
            p_new = int(row.get(coin, 0))
            p_old = self.pos[coin]
            price = prices_now.get(coin)
            if price is None: continue
            dc_high = h[coin].rolling(self.N).max().shift(1).iloc[-1]
            dc_low = l[coin].rolling(self.N).min().shift(1).iloc[-1]
            if p_old == 0 and p_new == 1:
                qty = size_for_coin(usd_free, price, self.usd_per_trade, self.amount_precision.get(coin, 3))
                if qty <= 0:
                    log_trade(coin, "BUY", price, qty, dc_high, dc_low, 0, 0, "SKIP_SMALL", "qty below step")
                else:
                    resp = place_order(coin, "BUY", qty)
                    status = "SUCCESS" if isinstance(resp, dict) and resp.get("Success", False) else "ERROR"
                    msg = resp.get("ErrMsg") if isinstance(resp, dict) else "no-json"
                    log_trade(coin, "BUY", price, qty, dc_high, dc_low, 0, 1 if status=="SUCCESS" else 0, status, msg)
                    if status == "SUCCESS":
                        self.pos[coin] = 1
                time.sleep(0.25)
            elif p_old == 1 and p_new == 0:
                bal2 = get_balance()
                free_q = 0.0
                if isinstance(bal2, dict):
                    if "SpotWallet" in bal2:
                        w2 = bal2.get("SpotWallet", {})
                        # direct coin dict
                        if isinstance(w2, dict) and coin in w2:
                            free_q = float(w2[coin].get("Free", 0) or 0)
                        assets2 = w2.get("Assets") if isinstance(w2, dict) else None
                        if assets2 and free_q == 0:
                            for a in assets2:
                                if a.get("Asset") == coin:
                                    free_q = float(a.get("Free", 0) or 0)
                                    break
                    elif bal2.get("Success"):
                        free_q = float(bal2.get("Data", {}).get(coin, {}).get("Free", 0) or 0)
                if free_q > 0:
                    step = 1.0 if self.amount_precision.get(coin, 3) <= 0 else 10 ** (-self.amount_precision.get(coin, 3))
                    free_q = np.floor(free_q / step) * step
                    resp = place_order(coin, "SELL", float(round(free_q, self.amount_precision.get(coin, 3))))
                    status = "SUCCESS" if isinstance(resp, dict) and resp.get("Success", False) else "ERROR"
                    msg = resp.get("ErrMsg") if isinstance(resp, dict) else "no-json"
                    log_trade(coin, "SELL", price, free_q, dc_high, dc_low, 1, 0 if status=="SUCCESS" else 1, status, msg)
                else:
                    log_trade(coin, "SELL", price, 0, dc_high, dc_low, 1, 0, "NO_BALANCE", "forced flat")
                self.pos[coin] = 0
                time.sleep(0.25)

    def run(self):
        print("Start loop")
        while True:
            self.update_ticks()
            sig = self.compute_signal()
            self.trade(sig)
            time.sleep(self.poll)

# Step 9: entry point
if __name__ == "__main__":
    bot = MultiDonchianBot(lookback=72, poll_sec=30, usd_per_trade=50.0)
    bot.run()