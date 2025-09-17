import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

# === Load & Preprocess Data ===
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

# === Prepare Features & Targets ===
window_size = 30
features = []
targets = []

for i in range(window_size, len(df) - 1):
    past_eth = df['ETH_Return'].iloc[i-window_size:i].values
    past_sol = df['SOL_Return'].iloc[i-window_size:i].values
    vol_eth = np.std(past_eth)
    vol_sol = np.std(past_sol)
    feat = np.concatenate([past_eth, past_sol, [vol_eth, vol_sol]])
    features.append(feat)
    next_ret = df[['ETH_Return', 'SOL_Return']].iloc[i+1].values
    targets.append(next_ret)

features = np.array(features)
targets = np.array(targets)

X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.float32)

# === Model Definition ===
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

def negative_sharpe_loss(weights, future_returns, risk_free=0.0):
    port_returns = torch.sum(weights * future_returns, dim=1)
    mean_ret = torch.mean(port_returns) - risk_free
    std_ret = torch.std(port_returns) + 1e-8
    sharpe = mean_ret / std_ret
    return -sharpe

# === Train Model ===
model = AllocationNet(X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    weights = model(X)
    loss = negative_sharpe_loss(weights, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# === Save Allocation & Model Weights ===
latest_feat_tensor = torch.tensor(features[-1:], dtype=torch.float32)
predicted_weights = model(latest_feat_tensor).detach().numpy().flatten()
allocation_ml = {"ETH": float(predicted_weights[0]), "SOL": float(predicted_weights[1])}

with open("allocations_ml.json", "w") as f:
    json.dump(allocation_ml, f, indent=2)

torch.save(model.state_dict(), "ml_model.pth")
print("Model trained and saved to ml_model.pth")
print("Allocation saved to allocations_ml.json")