#  Multi-Strategy Crypto Trading Bot

A modular, live-trading **multi-strategy** engine for USD-quoted spot crypto, with:

* Robust **signal ensemble** (trend, momentum, mean-reversion, AR/ARMA, Fourier/seasonality, beta, VWAP/CLV).
* Exchange-agnostic market data via **ccxt**.
* Live order routing to **Roostoo** mock/live API with **lot-size rounding** and safety checks.
* Simple **dry-run** mode, CSV trade logs, and backtesting notebooks.

> ⚠️ **Disclaimer**: This software is for educational use. Crypto trading is risky. Use at your own risk.

---

## Repository Structure

```
.
├─ deployed_bot.py           # Live trading entrypoint (dry run or live to Roostoo)
├─ backtester.ipynb          # Backtesting / experiments (notebook)
├─ multistrat_engine.ipynb   # Strategy engine exploration (notebook)
├─ Final Model.ipynb         # Final modeling / research notes (notebook)
├─ whale_supply/             # Whale supply signals / utilities (placeholder folder)
└─ README.md                 # You are here
```

---

## What It Does

### Live data

* Pulls OHLCV from the configured **ccxt** exchange (default: `coinbase`) at `5m` resolution for a configurable USD universe.

### Signal ensemble

The bot builds an **ensemble** from multiple strategies and converts it into **long-only target weights**:

* **Donchian_Breakout (panel)** — Breaks out above N-bar highs/lows on resampled bars (default: 1H / N=72).
* **Cross_Sectional_Momentum (panel)** — Ranks symbols by lookback returns, filters by dollar volume, buys the top fraction.
* **MeanReversionStrategy** — Z-score of price vs. rolling mean; enters below `z_entry`, exits over `z_exit`.
* **MovingAverageStrategy** — Majority vote over EMA/SMA/WMA/DEMA/TEMA/TRIMA/KAMA short vs. long crossovers.
* **BreakoutStrategy** — Price vs. MA bands + N-day high/low + ATR smoothing.
* **BetaRegressionStrategy** — Rolling beta/corr/z-score between low/high series.
* **VWAPStrategy** — Deviations from rolling VWAP with volume weighting.
* **CLV** — Close Location Value classifier from -1..+1.
* **AutoRegressiveStrategy / ARMAStrategy** — Residual-sign classifier from AR(p)/ARMA(p,q).
* **FourierTransform** — DFT + band-pass/energy/seasonality tests; constructs a weight from low/mid/high energy and recent slopes.

Each strategy contributes a **[0..1]** signal; the engine combines them by **weighted sum**, normalizes to weights, then **exponential-smooths** them over time.

### Portfolio + execution

* Computes **current** weights from Roostoo balances (USD + coin holdings × last price).
* Compares to **target**; if L1 distance ≥ threshold, computes trades.
* **Safety rails**:

  * Long-only (no shorts).
  * Don’t sell more than you own.
  * Skip notionals `< MIN_NOTIONAL`.
  * Respect **exchange lot sizes** (step sizes) with rounding **down**.
* Sends **MARKET** orders to Roostoo with signed requests (HMAC-SHA256).
* Writes trades to `trade_log.csv`.

---

## Quickstart

### 1) Python & dependencies

* Python **3.11+** recommended (works on 3.13 as well).
* TA-Lib requires system libs.

```bash
# Linux/macOS (example)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib requests ccxt statsmodels scikit-learn scikit-optimize scipy quantstats

# TA-Lib (choose one, OS-specific)
# macOS (brew):  brew install ta-lib
# Ubuntu:        sudo apt-get install -y ta-lib
# then:
pip install ta-lib
```

### 2) Environment variables (Roostoo)

Provide your Roostoo API credentials via env vars **or** in code:

```bash
export ROOSTOO_API_KEY="your_api_key"
export ROOSTOO_SECRET="your_secret"
```

> In `deployed_bot.py`, the client reads:
>
> ```python
> roostoo = RoostooClient(
>     api_key=ROOSTOO_API_KEY or os.getenv("ROOSTOO_API_KEY"),
>     secret=ROOSTOO_SECRET or os.getenv("ROOSTOO_SECRET"),
> )
> ```

### 3) Configure universe (optional)

Edit `UNIVERSE_BASES` in `deployed_bot.py`. Default includes:

```
"ZEN","AAVE","ADA","APT","ARB","AVAX","BNB","BTC","CAKE","DOGE","DOT","EIGEN",
"ENA","ETH","FET","FIL","ICP","LINK","LTC","NEAR","OMNI","ONDO","PENDLE","PENGU",
"POL","SEI","SOL","SUI","TAO","UNI","WIF","WLD","XLM","XRP","ZEC"
```

Make sure your **ccxt exchange** actually lists these as `BASE/USD` (e.g., `ARB/USD`). If not, remove/adjust symbols or swap the exchange in `LiveDataPipeline(exchange_id='coinbase')`.

### 4) Dry-run or Live

In `deployed_bot.py`:

```python
if __name__ == "__main__":
    DRY_RUN = False   # set True for paper trading (console prints only)
    ...
```

Run it:

```bash
python deployed_bot.py
```

* **Dry run** prints `[DRY RUN] BUY/SELL ...` without hitting the API.
* **Live** signs and submits orders to Roostoo’s `/v3/place_order`.

---

## Configuration Knobs (in `deployed_bot.py`)

* **Execution**

  * `MIN_NOTIONAL = 10.0`      # skip tiny fills
  * `THRESHOLD = 0.02`         # L1 diff vs current weights to rebalance
  * `LAMBDA = 0.8`             # target weight smoothing (higher = more reactive)
  * `sleep_seconds = 300`      # loop interval

* **Data**

  * `exchange_id='coinbase'`   # ccxt exchange id
  * `timeframe='5m'`           # OHLCV interval
  * `limit=1500`               # bars to pull

* **Lot sizes**

  * `RoostooClient.base_lot_sizes` — per-asset step sizes (e.g., BTC=0.0001, ADA=1.0).
  * Orders are **rounded down** to the nearest step; 0 after rounding ⇒ **skipped**.

* **Strategy weights**

  * `strategies_config = [ StrategyConfig(..., weight=..., long_only=...), ... ]`
  * Combine policy: `'weighted_sum'` (default) or `'mean'`.

---

## How It Works (Flow)

```
ccxt (OHLCV) ──► LiveDataPipeline ──► series_mats (open/high/low/close/vol/vwap/clv)
                                        │
                                        ▼
                             MultiStrategyEngine (panel + per-symbol strategies)
                                        │
                                        ▼
                             target weights (normalized & smoothed)
                                        │                  ▲
        Roostoo /balance ──► current weights ◄────────────┘
                                        │
                              Rebalance decision (L1 ≥ THRESHOLD)
                                        │
                                        ▼
                       Trade list with notional & lot-size rounding
                                        │
                          Roostoo /v3/place_order (signed HMAC)
                                        │
                                        ▼
                               trade_log.csv (append-only)
```

---

## Notebooks

* **`backtester.ipynb`**
  Load historical OHLCV (via ccxt or your data), reproduce the `series_mats` dict, call the `MultiStrategyEngine` to compute signals and simulate PnL. Use `quantstats` for tear-sheets.

* **`multistrat_engine.ipynb`**
  Isolated strategy exploration: tune parameters, visualize signals, combine policies, sensitivity analysis.

* **`Final Model.ipynb`**
  Final exploration / results summary (use as research log).

> Tip: When backtesting, keep strategy logic deterministic and side-effect free. The ensemble already expects per-symbol `Series` and panel `DataFrame`s.

---

## Whale Supply

`whale_supply/` is a staging area for whale inflow/outflow or net-position metrics (e.g., from Horus/Glassnode-like sources). Integrate these as an **additional strategy** by:

1. Building a per-symbol time series, aligned to your OHLCV index.
2. Adding a new `StrategyConfig(YourWhaleStrategy(), ['whale_signal'], weight=..., long_only=...)`.
3. Merging the column into `series_mats` (e.g., extend `LiveDataPipeline` to fetch it).

---

## Common Errors & Fixes

* **`[RoostooClient] WARNING: API key/secret not set.`**
  Export `ROOSTOO_API_KEY` / `ROOSTOO_SECRET` or hard-code them for testing.

* **`"quantity step size error"` (Roostoo response)**
  Increase precision or adjust `base_lot_sizes` for that symbol. The client **rounds down**; if rounding→0, trade is skipped.

* **No trades placed**

  * Signals may all be zero (ensemble yields no allocation).
  * `L1 diff` below `THRESHOLD`.
  * Notional per order < `MIN_NOTIONAL`.
  * Your universe has symbols not supported by the exchange.

* **TA-Lib install issues**
  Install system libraries first (brew/apt), then `pip install ta-lib`.

* **`ccxt` symbol not found**
  Check actual symbols on your exchange; adapt to `BASE/USD` pairs that exist. Some venues use USDT or different tickers.

---

## Extending the System

* **Add a new strategy**
  Implement a class with a `signal(...) -> DataFrame|Series` returning a column `'value'` in `[0..1]` (or any real, the engine clips long-only).
  For **panel** strategies, set `requires_panel = True` and implement:

  ```python
  def signal(self, index, open, high, low, close, volume, vwap, clv) -> pd.DataFrame
  ```

  Return a DataFrame with symbol columns → integer/float picks or weights.

* **Change combine policy**
  In `MultiStrategyEngine(combine='mean'|'weighted_sum')`.

* **Risk overlays**
  Add volatility targeting, drawdown caps, or cash buffers at the rebalance step before order creation.

---

## Logging

* Trades appended to `trade_log.csv`:

  ```
  timestamp,coin,side,qty,price
  2025-...Z,BTC,BUY,0.005,105000
  ```
* Console prints:

  * Target/current weights, `L1 diff`, per-order results:

    * `ORDER OK ...` on success.
    * `ROOSTOO ERROR ...` with API response.
    * `SKIP ...` when below step size or notional.

---

## Citation / Credit

Built with ❤️ on top of:

* [ccxt](https://github.com/ccxt/ccxt) for market data
* [TA-Lib](https://mrjbq7.github.io/ta-lib/) for indicators
* [statsmodels](https://www.statsmodels.org/)
* [scikit-learn](https://scikit-learn.org/)
* [scikit-optimize](https://scikit-optimize.github.io/stable/)
* [quantstats](https://github.com/ranaroussi/quantstats)

