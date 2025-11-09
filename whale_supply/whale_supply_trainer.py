import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import datetime, time, os
from sklearn.preprocessing import StandardScaler

# ==== CONFIG ====
API_KEY = "c17248ed7a55f8e016bd95dfe0e5900be290db72b4ca533ced39b0aeb54111c1"
BASE_URL_WHALE = "https://api-horus.com/addresses/whale_net_flow"
BASE_URL_PRICE = "https://api-horus.com/market/price"
CHAIN, ASSET, INTERVAL = "bitcoin", "BTC", "1d"

start_date = datetime.datetime(2024, 8, 12)
end_date   = datetime.datetime.now()
start_ts, end_ts = int(start_date.timestamp()), int(end_date.timestamp())

# ==== FETCH FUNCTIONS ====
def get_whale_data():
    params = {"chain": CHAIN, "interval": INTERVAL, "start": start_ts, "end": end_ts, "format": "json"}
    headers = {"X-API-Key": API_KEY}
    resp = requests.get(BASE_URL_WHALE, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if "whale_net_flow" not in df.columns:
        raise ValueError("Unexpected keys: " + ", ".join(df.columns))
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df[["timestamp", "whale_net_flow"]]

def get_price_data():
    params = {"asset": ASSET, "interval": INTERVAL, "start": start_ts, "end": end_ts, "format": "json"}
    headers = {"X-API-Key": API_KEY}
    resp = requests.get(BASE_URL_PRICE, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if "price" not in df.columns:
        raise ValueError("Unexpected keys: " + ", ".join(df.columns))
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.rename(columns={"price": "close"}, inplace=True)
    return df[["timestamp", "close"]]

# ==== PREPARE DATA ====
df = pd.merge_asof(
    get_whale_data().sort_values("timestamp"),
    get_price_data().sort_values("timestamp"),
    on="timestamp", direction="nearest"
).dropna(subset=["whale_net_flow", "close"])

df["future_return"] = df["close"].pct_change().shift(-1)
df["signal"] = (df["future_return"] > 0).astype(int)
df.dropna(inplace=True)

X = df[["whale_net_flow"]].values
y = df["signal"].values
X_scaled = StandardScaler().fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ==== TIME-BASED SPLIT ====
start_time, end_time = df["timestamp"].min(), df["timestamp"].max()
split_time = start_time + (end_time - start_time) * 0.8
train_mask, val_mask = df["timestamp"] <= split_time, df["timestamp"] > split_time

X_train, y_train = X_tensor[train_mask.values], y_tensor[train_mask.values]
X_val, y_val     = X_tensor[val_mask.values],   y_tensor[val_mask.values]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

print(f"🕒 Train: {start_time.date()} → {split_time.date()}")
print(f"🧪 Validate: {split_time.date()} → {end_time.date()}")

# ==== MODEL ====
class WhaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

model = WhaleNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==== TRAIN ====
for epoch in range(80):
    model.train()
    total = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        total += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/80 | Loss: {total/len(train_loader):.4f}")

# ==== VALIDATE & SIMULATE ====
model.eval()
with torch.no_grad():
    preds_val = model(X_val).numpy().flatten()

# model outputs continuous 'greatness of signal'
df.loc[val_mask, "signal_strength"] = preds_val
df.loc[train_mask, "signal_strength"] = np.nan

# execute only when signal_strength > 0.7
df["model_signal"] = np.where(df["signal_strength"] > 0.5, 1, 0)
df.loc[train_mask, "model_signal"] = np.nan

# ---- Simulation ----
capital, holding, portfolio_value = 1.0, 0.0, 1.0
entered_market = False
capital_curve = []

for i in range(len(df)):
    price, sig = df["close"].iloc[i], df["model_signal"].iloc[i]
    if np.isnan(sig):
        capital_curve.append(portfolio_value)
        continue
    if not entered_market:
        if sig == 1:
            holding, capital = capital / price, 0
            entered_market = True
        capital_curve.append(portfolio_value)
        continue
    if sig == 1 and capital > 0:
        holding, capital = capital / price, 0
    elif sig == 0 and holding > 0:
        capital, holding = holding * price, 0
    portfolio_value = capital + holding * price
    capital_curve.append(portfolio_value)
df["portfolio_value"] = capital_curve

# ==== TRADE LOG ====
trades, in_pos = [], False
for i in range(len(df)):
    if np.isnan(df["model_signal"].iloc[i]): continue
    sig, t, p = int(df["model_signal"].iloc[i]), df["timestamp"].iloc[i], float(df["close"].iloc[i])
    if not in_pos and sig == 1:
        in_pos, buy_t, buy_p = True, t, p
    elif in_pos and sig == 0:
        in_pos = False
        ret = (p/buy_p - 1)*100
        trades.append({
            "buy_time": buy_t,
            "buy_price": buy_p,
            "sell_time": t,
            "sell_price": p,
            "trade_return_%": round(ret, 2)
        })

trade_df = pd.DataFrame(trades)
print("\n=== Trade Log (Validation) ===")
print(trade_df.to_string(index=False))

# ==== PERFORMANCE ====
val_df = df[val_mask].copy()
first_act = val_df["portfolio_value"].ne(val_df["portfolio_value"].iloc[0]).idxmax()
if pd.isna(first_act):
    total_ret, sharpe = 0, 0
else:
    active = df.loc[first_act:].copy()
    total_ret = (active["portfolio_value"].iloc[-1] / active["portfolio_value"].iloc[0]) - 1
    rets = active["portfolio_value"].pct_change().dropna()
    sharpe = 0 if rets.std()==0 else (rets.mean()/rets.std())*np.sqrt(365)
print(f"\nTotal Return: {total_ret*100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")

# ==== PLOT ====
plt.figure(figsize=(12,6))
price_norm = df["close"]/df["close"].iloc[0]
portfolio_norm = df["portfolio_value"]/df["portfolio_value"].iloc[0]
plt.plot(df["timestamp"], price_norm, label="BTC Price", color="gray", alpha=0.6)
plt.plot(df["timestamp"], portfolio_norm, label="Portfolio", color="green")
plt.axvline(split_time, color="black", ls="--", lw=1, alpha=0.6, label="Train/Val Split")
buy_pts = df[(df["model_signal"]==1)&(df["model_signal"].shift(1)==0)]
sell_pts= df[(df["model_signal"]==0)&(df["model_signal"].shift(1)==1)]
plt.scatter(buy_pts["timestamp"], price_norm.loc[buy_pts.index], marker="^", c="blue", s=100)
plt.scatter(sell_pts["timestamp"], price_norm.loc[sell_pts.index], marker="v", c="red", s=100)
plt.legend(); plt.grid(alpha=0.3)
plt.title("BTC Whale Net Flow – Signal>0.7 Trade Execution")
plt.tight_layout(); plt.show()

# ==== SAVE ====
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/whale_btc_model.pt")
print("✅ Model saved to models/whale_btc_model.pt")
