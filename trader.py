import socket
import json
import threading
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import csv
import time
from datetime import datetime

HOST = "127.0.0.1"
PORT = 5000
USER_ID = "bot1"

# Trading parameters
START_WALLET = 10000.0
TARGET_MULTIPLIER = 1.5
wallet = START_WALLET
positions = {}  # symbol -> (qty, entry_price)

# Price history per symbol
symbols = ["AAPL", "TSLA", "AMZN", "GOOG"]
price_history = {sym: deque(maxlen=50) for sym in symbols}

# Training data
training_data = []
labels = []

model = None
training_complete = False

# CSV Logging Setup
LOG_FILE = "trade_log.csv"
with open(LOG_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "symbol", "side", "quantity", "price", "wallet_after_trade"])

def connect_to_market():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    return sock

def log_trade(symbol, side, qty, price):
    """Log trade to CSV and console."""
    global wallet
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write to CSV
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, symbol, side, qty, f"{price:.2f}", f"{wallet:.2f}"])

    # Print to console
    print(f"[{timestamp}] TRADE: {side} {qty} {symbol} @ {price:.2f} | Wallet: {wallet:.2f}")

def send_trade(sock, symbol, side, qty, price):
    """Send trade command to server and log it."""
    cmd = f"TRADE {USER_ID} {symbol} {side} {qty}\n"
    sock.sendall(cmd.encode())
    log_trade(symbol, side, qty, price)

def handle_market_data(sock):
    global wallet, training_complete

    while True:
        data = sock.recv(4096).decode().strip()
        if not data:
            continue
        try:
            market_state = json.loads(data)
        except json.JSONDecodeError:
            continue

        # Update each symbol's price history
        for eq in market_state["equities"]:
            sym = eq["symbol"]
            price_history[sym].append(eq["price"])

        # Data collection phase
        if not training_complete:
            for sym, hist in price_history.items():
                if len(hist) == hist.maxlen:
                    X = np.array(hist).reshape(1, -1, 1)
                    y = 1 if hist[-1] > hist[-2] else 0
                    training_data.append(X)
                    labels.append(y)

            if len(training_data) > 200:
                train_model()
            continue

        # Live trading phase
        for eq in market_state["equities"]:
            sym = eq["symbol"]
            price = eq["price"]
            hist = price_history[sym]

            if len(hist) < hist.maxlen:
                continue

            X = np.array(hist).reshape(1, -1, 1)
            pred = model.predict(X, verbose=0)[0][0]

            # Buy condition
            if pred > 0.52 and wallet >= price:
                qty = max(1, int((wallet * 0.1) // price))  # 10% wallet per trade
                send_trade(sock, sym, "BUY", qty, price)
                wallet -= qty * price
                if sym in positions:
                    positions[sym] = (positions[sym][0] + qty, price)
                else:
                    positions[sym] = (qty, price)

            # Sell condition
            elif pred < 0.48 and sym in positions:
                qty, entry_price = positions.pop(sym)
                send_trade(sock, sym, "SELL", qty, price)
                wallet += qty * price

        # Stop if wallet target reached
        if wallet >= START_WALLET * TARGET_MULTIPLIER:
            print(f"🎯 Target reached! Final wallet: {wallet:.2f}")
            exit(0)

def train_model():
    global model, training_complete

    X_train = np.vstack(training_data)
    y_train = np.array(labels)

    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    print("📈 Starting LSTM training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=16,
        verbose=1  # shows per-epoch logs
    )
    acc = history.history["accuracy"][-1]
    loss = history.history["loss"][-1]
    print(f"✅ Training complete. Accuracy: {acc:.2%} | Loss: {loss:.4f}")

    if acc >= 0.75:
        print("🚀 Accuracy threshold met. Starting live trading...")
        training_complete = True
    else:
        print("⚠ Accuracy below threshold. Continuing data collection...")

if __name__ == "__main__":
    sock = connect_to_market()
    threading.Thread(target=handle_market_data, args=(sock,), daemon=True).start()

    while True:
        time.sleep(1)