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

class Donchian_Breakout:
    def __init__(self, N=72):
        self.N = N
        self.requires_panel = True  # ensure build_weights sends full matrices

    def signal(self, index, open, high, low, close, volume, vwap, clv):
        h = high.resample('1H').max()
        l = low.resample('1H').min()
        c = close.resample('1H').last()
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

# --- Minimal live Donchian trading loop using hourly highs/lows ---
LOG_FILE = "trade_log.csv"

def _ensure_log():
    try:
        import os
        if not os.path.isfile(LOG_FILE):
            with open(LOG_FILE, "a", newline="") as f:
                f.write("timestamp,side,price,qty,dc_high,dc_low,position_before,position_after,status,message\n")
    except Exception as e:
        print(f"Log init error: {e}")

def _log_trade(side, price, qty, dc_high, dc_low, pos_before, pos_after, status, message):
    try:
        with open(LOG_FILE, "a", newline="") as f:
            ts = datetime.utcnow().isoformat()
            f.write(f"{ts},{side},{price},{qty},{dc_high},{dc_low},{pos_before},{pos_after},{status},{message}\n")
    except Exception as e:
        print(f"Log write error: {e}")

def _last_price(pair):
    t = get_ticker(pair)
    if not t or not t.get("Success"):
        return None
    d = t.get("Data", {}).get(pair, {})
    p = d.get("LastPrice")
    try:
        return float(p)
    except:
        return None

def _free_coin(symbol, balance_json):
    try:
        return float(balance_json.get("Data", {}).get(symbol, {}).get("Free", 0))
    except:
        return 0.0

def _free_usd(balance_json):
    try:
        return float(balance_json.get("Data", {}).get("USD", {}).get("Free", 0))
    except:
        return 0.0

if __name__ == '__main__':
    # Config
    COIN = "BNB"
    PAIR = f"{COIN}/USD"
    N = 72
    POLL_SEC = 30
    TRADE_QTY = 0.05

    _ensure_log()
    # State for hourly bars
    period_highs = []    # completed-hour highs
    period_lows = []     # completed-hour lows
    current_hour = None
    cur_high = None
    cur_low = None
    pos = 0              # 0 = flat, 1 = long

    print(f"Starting Donchian breakout bot ({PAIR}, N={N})")
    while True:
        try:
            price = _last_price(PAIR)
            if price is None:
                time.sleep(POLL_SEC)
                continue

            hour_key = int(time.time() // 3600)
            if current_hour is None:
                current_hour = hour_key
                cur_high = price
                cur_low = price
            elif hour_key == current_hour:
                # Update forming hour
                if price > cur_high: cur_high = price
                if price < cur_low: cur_low = price
            else:
                # Hour rolled: store completed-hour high/low
                period_highs.append(cur_high)
                period_lows.append(cur_low)
                current_hour = hour_key
                cur_high = price
                cur_low = price

            # Only trade using fully completed hours, exclude current forming hour
            if len(period_highs) >= N:
                dc_high = max(period_highs[-N:])
                dc_low = min(period_lows[-N:])

                # Entry: price > dc_high and we are flat
                if pos == 0 and price > dc_high:
                    print(f"{datetime.now()} BUY signal: price {price:.4f} > dc_high {dc_high:.4f}")
                    pos_before = pos
                    r = place_order(COIN, "BUY", TRADE_QTY)
                    if r and r.get("Success"):
                        pos = 1
                        print("Entered LONG.")
                        _log_trade("BUY", price, TRADE_QTY, dc_high, dc_low, pos_before, pos, "SUCCESS", "Breakout entry")
                    else:
                        print("BUY failed, staying FLAT.")
                        _log_trade("BUY", price, TRADE_QTY, dc_high, dc_low, pos_before, pos, "FAIL", f"{r}")

                # Exit: price < dc_low and we are long
                elif pos == 1 and price < dc_low:
                    print(f"{datetime.now()} SELL signal: price {price:.4f} < dc_low {dc_low:.4f}")
                    pos_before = pos
                    bal = get_balance()
                    qty = _free_coin(COIN, bal)
                    if qty > 0:
                        r = place_order(COIN, "SELL", qty)
                        if r and r.get("Success"):
                            pos = 0
                            print("Exited LONG.")
                            _log_trade("SELL", price, qty, dc_high, dc_low, pos_before, pos, "SUCCESS", "Breakdown exit")
                        else:
                            print("SELL failed, staying LONG.")
                            _log_trade("SELL", price, qty, dc_high, dc_low, pos_before, pos, "FAIL", f"{r}")
                    else:
                        pos = 0
                        print("No balance to sell; setting FLAT.")
                        _log_trade("SELL", price, 0, dc_high, dc_low, pos_before, pos, "NO_BALANCE", "Forced flat")

            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("Stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(POLL_SEC)