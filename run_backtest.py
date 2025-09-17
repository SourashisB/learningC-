import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import subprocess
import os

# === Load Data ===
eth = pd.read_csv("eth.csv")
sol = pd.read_csv("sol.csv")

eth['Open time'] = pd.to_datetime(eth['Open time'].str.strip())
sol['Open time'] = pd.to_datetime(sol['Open time'].str.strip())

eth = eth[['Open time', 'Close']].rename(columns={'Close': 'ETH_Close'})
sol = sol[['Open time', 'Close']].rename(columns={'Close': 'SOL_Close'})

df = pd.merge(eth, sol, on='Open time', how='inner')
df['ETH_Return'] = df['ETH_Close'].pct_change()
df['SOL_Return'] = df['SOL_Close'].pct_change()
df.dropna(inplace=True)

# === Markowitz Optimization ===
def mean_variance_optimization(returns_df, target_return=None):
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    ones = np.ones(len(mean_returns))
    mu = mean_returns.values
    cov = cov_matrix.values
    inv_cov = np.linalg.inv(cov)
    A = ones @ inv_cov @ ones
    B = ones @ inv_cov @ mu
    C = mu @ inv_cov @ mu
    D = A * C - B**2
    if target_return is None:
        weights = inv_cov @ mu / (ones @ inv_cov @ mu)
    else:
        lam = (C - B * target_return) / D
        gam = (A * target_return - B) / D
        weights = lam * (inv_cov @ ones) + gam * (inv_cov @ mu)
    return dict(zip(returns_df.columns, weights))

returns_df = df[['ETH_Return', 'SOL_Return']]
optimal_weights_markowitz = mean_variance_optimization(returns_df)
with open("allocations_markowitz.json", "w") as f:
    json.dump(optimal_weights_markowitz, f, indent=2)

# === Load Trained ML Model ===
window_size = 30

class AllocationNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)

# Prepare features for rolling backtest
features = []
for i in range(window_size, len(df) - 1):
    past_eth = df['ETH_Return'].iloc[i-window_size:i].values
    past_sol = df['SOL_Return'].iloc[i-window_size:i].values
    vol_eth = np.std(past_eth)
    vol_sol = np.std(past_sol)
    feat = np.concatenate([past_eth, past_sol, [vol_eth, vol_sol]])
    features.append(feat)

# Load model weights
model = AllocationNet(len(features[0]))
model.load_state_dict(torch.load("ml_model.pth"))
model.eval()

# === Rolling Backtest ===
def rolling_backtest(df, window_size, model, markowitz_func):
    dates = []
    eq_value = mv_value = ml_value = 1.0
    eq_values, mv_values, ml_values = [], [], []

    for i in range(window_size, len(df)-1):
        past_returns = df[['ETH_Return', 'SOL_Return']].iloc[i-window_size:i]
        eq_weights = np.array([0.5, 0.5])
        mv_weights = np.array(list(markowitz_func(past_returns).values()))
        past_eth = past_returns['ETH_Return'].values
        past_sol = past_returns['SOL_Return'].values
        vol_eth = np.std(past_eth)
        vol_sol = np.std(past_sol)
        feat = np.concatenate([past_eth, past_sol, [vol_eth, vol_sol]])
        feat_tensor = torch.tensor(feat.reshape(1, -1), dtype=torch.float32)
        ml_weights = model(feat_tensor).detach().numpy().flatten()
        next_ret = df[['ETH_Return', 'SOL_Return']].iloc[i+1].values
        eq_value *= (1 + np.dot(eq_weights, next_ret))
        mv_value *= (1 + np.dot(mv_weights, next_ret))
        ml_value *= (1 + np.dot(ml_weights, next_ret))
        dates.append(df['Open time'].iloc[i+1])
        eq_values.append(eq_value)
        mv_values.append(mv_value)
        ml_values.append(ml_value)

    return pd.DataFrame({
        'Date': dates,
        'Equal_Weight': eq_values,
        'Markowitz': mv_values,
        'ML_Model': ml_values
    })

backtest_df = rolling_backtest(df, window_size, model, mean_variance_optimization)
backtest_df.to_csv("backtest_results.csv", index=False)

# === Plot Results ===
plt.figure(figsize=(10, 6))
plt.plot(backtest_df['Date'], backtest_df['Equal_Weight'], label='Equal Weight', linestyle='--')
plt.plot(backtest_df['Date'], backtest_df['Markowitz'], label='Markowitz')
plt.plot(backtest_df['Date'], backtest_df['ML_Model'], label='ML Model')
plt.title("Cumulative Portfolio Value")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("portfolio_cumulative_returns.png", dpi=300)
plt.close()

def rolling_sharpe(series, window=60):
    return series.pct_change().rolling(window).mean() / series.pct_change().rolling(window).std()

plt.figure(figsize=(10, 6))
plt.plot(backtest_df['Date'], rolling_sharpe(backtest_df['Equal_Weight']), label='Equal Weight', linestyle='--')
plt.plot(backtest_df['Date'], rolling_sharpe(backtest_df['Markowitz']), label='Markowitz')
plt.plot(backtest_df['Date'], rolling_sharpe(backtest_df['ML_Model']), label='ML Model')
plt.title("Rolling Sharpe Ratio (60-period)")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("portfolio_rolling_sharpe.png", dpi=300)
plt.close()

# === Run C++ Risk Engine (Windows-friendly) ===
def run_risk_engine(alloc_json, returns_csv, output_json, threads=8):
    cpp_dir = os.path.join(os.path.dirname(__file__), "risk_engine")
    build_dir = os.path.join(cpp_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    print("Configuring C++ risk engine with CMake...")
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)

    print("Building C++ risk engine...")
    subprocess.run(["cmake", "--build", "."], cwd=build_dir, check=True)

    exe_name = "risk_engine.exe" if os.name == "nt" else "risk_engine"
    exe_path = None
    for root, dirs, files in os.walk(build_dir):
        if exe_name in files:
            exe_path = os.path.join(root, exe_name)
            break
    if exe_path is None:
        raise FileNotFoundError(f"{exe_name} not found in {build_dir}")

    print(f"Running risk engine with {threads} threads...")
    subprocess.run(
        [exe_path, alloc_json, returns_csv, output_json, str(threads)],
        check=True
    )

alloc_json_path = "allocations_ml.json"
returns_csv_path = "backtest_results.csv"
risk_json_path = "risk_metrics.json"

run_risk_engine(alloc_json_path, returns_csv_path, risk_json_path, threads=8)
print("Risk metrics saved to", risk_json_path)