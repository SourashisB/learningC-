import socket
import json
import threading
import numpy as np
from collections import deque
from keras.api.models import Sequential, clone_model
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.optimizers import Adam
import csv
import time
from datetime import datetime

# Market connection
HOST = "127.0.0.1"
PORT = 5000
USER_ID = "bot1"

# Wallet and trading parameters
START_WALLET = 10000.0
TARGET_MULTIPLIER = 1.5
wallet = START_WALLET
positions = {}  # symbol -> dict(qty, entry_price, entry_tick)

# Exposure limits
MAX_TRADE_PCT = 0.1   # Max 10% of wallet per trade
MAX_TOTAL_PCT = 0.5   # Max 50% of wallet invested at once

# Symbols and price history
symbols = ["AAPL", "TSLA", "AMZN", "GOOG"]
price_history = {sym: deque(maxlen=50) for sym in symbols}

# Rolling window training data
MAX_TRAIN_SAMPLES = 1000
training_data = deque(maxlen=MAX_TRAIN_SAMPLES)
labels = deque(maxlen=MAX_TRAIN_SAMPLES)

# Model state
model = None
training_complete = False
last_retrain_tick = 0
best_accuracy = 0.0
best_model = None

# CSV Logging
LOG_FILE = "trade_log.csv"
with open(LOG_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "symbol", "side", "quantity", "price", "wallet_after_trade"])

def connect_to_market():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    return sock

def log_trade(symbol, side, qty, price):
    global wallet
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, symbol, side, qty, f"{price:.2f}", f"{wallet:.2f}"])
    print(f"[{timestamp}] TRADE: {side} {qty} {symbol} @ {price:.2f} | Wallet: {wallet:.2f}")

def send_trade(sock, symbol, side, qty, price):
    cmd = f"TRADE {USER_ID} {symbol} {side} {qty}\n"
    sock.sendall(cmd.encode())
    log_trade(symbol, side, qty, price)

def compute_label(hist):
    """Simple trend label from moving averages."""
    short_ma = np.mean(list(hist)[-5:])
    long_ma = np.mean(list(hist)[-20:])
    return 1 if short_ma > long_ma else 0

def handle_market_data(sock):
    global wallet, training_complete, last_retrain_tick

    while True:
        data = sock.recv(4096).decode().strip()
        if not data:
            continue
        try:
            market_state = json.loads(data)
        except json.JSONDecodeError:
            continue

        tick = market_state["tick"]

        # Update price history
        for eq in market_state["equities"]:
            sym = eq["symbol"]
            price_history[sym].append(eq["price"])

        # Collect data before training starts
        if not training_complete:
            for sym, hist in price_history.items():
                if len(hist) == hist.maxlen:
                    training_data.append(np.array(hist).reshape(1, -1, 1))
                    labels.append(compute_label(hist))
            if len(training_data) >= 300:
                train_model()
            continue

        # Periodic retraining
        if tick - last_retrain_tick >= 200:
            print("♻ Retraining model with latest data...")
            train_model()
            last_retrain_tick = tick

        # Live trading
        for eq in market_state["equities"]:
            sym = eq["symbol"]
            price = eq["price"]
            hist = price_history[sym]
            if len(hist) < hist.maxlen:
                continue

            # Keep collecting rolling training data
            training_data.append(np.array(hist).reshape(1, -1, 1))
            labels.append(compute_label(hist))

            pred = model.predict(np.array(hist).reshape(1, -1, 1), verbose=0)[0][0]

            # Current total invested capital
            current_invested = sum(
                positions[s]["qty"] * price_history[s][-1]
                for s in positions if len(price_history[s]) > 0
            )

            # Buy logic with risk management
            if pred > 0.55 and wallet >= price and sym not in positions:
                if current_invested / (wallet + current_invested) >= MAX_TOTAL_PCT:
                    continue  # too much invested already
                trade_capital = wallet * MAX_TRADE_PCT
                qty = max(1, int(trade_capital // price))
                if qty > 0:
                    send_trade(sock, sym, "BUY", qty, price)
                    wallet -= qty * price
                    positions[sym] = {
                        "qty": qty,
                        "entry_price": price,
                        "entry_tick": tick
                    }

            # Sell logic
            elif sym in positions:
                pos = positions[sym]
                change_pct = (price - pos["entry_price"]) / pos["entry_price"]
                held_ticks = tick - pos["entry_tick"]

                if (
                    pred < 0.45 or
                    change_pct >= 0.05 or    # take profit
                    change_pct <= -0.03 or   # stop loss
                    held_ticks >= 20         # max hold time
                ):
                    send_trade(sock, sym, "SELL", pos["qty"], price)
                    wallet += pos["qty"] * price
                    del positions[sym]

        # Stop if profit target reached
        if wallet >= START_WALLET * TARGET_MULTIPLIER:
            print(f"🎯 Target reached! Final wallet: {wallet:.2f}")
            exit(0)

def train_model():
    global model, training_complete, best_accuracy, best_model

    if len(training_data) < 50:
        return

    X_train = np.vstack(training_data)
    y_train = np.array(labels)

    # Build improved stacked LSTM model
    temp_model = Sequential()
    temp_model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    temp_model.add(Dropout(0.2))
    temp_model.add(LSTM(32))
    temp_model.add(Dense(16, activation="relu"))
    temp_model.add(Dense(1, activation="sigmoid"))

    temp_model.compile(optimizer=Adam(learning_rate=0.001),
                       loss="binary_crossentropy",
                       metrics=["accuracy"])

    print(f"📈 Training on {len(X_train)} samples...")
    history = temp_model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=32,
        verbose=0
    )
    acc = history.history["accuracy"][-1]
    loss = history.history["loss"][-1]
    print(f"✅ Training complete. Accuracy: {acc:.2%} | Loss: {loss:.4f}")

    # Model selection logic
    if acc >= best_accuracy:
        print(f"🏆 New best model! Accuracy improved from {best_accuracy:.2%} to {acc:.2%}")
        best_accuracy = acc
        best_model = clone_model(temp_model)
        best_model.set_weights(temp_model.get_weights())
        model = best_model
    else:
        print(f"⚠ New model accuracy {acc:.2%} is worse than best {best_accuracy:.2%}. Keeping old model.")

    # Start live trading once threshold is met
    if not training_complete and best_accuracy >= 0.6:
        print("🚀 Accuracy threshold met. Starting live trading...")
        training_complete = True
    elif not training_complete:
        print("⚠ Accuracy below threshold. Continuing data collection...")

if __name__ == "__main__":
    sock = connect_to_market()
    threading.Thread(target=handle_market_data, args=(sock,), daemon=True).start()

    while True:
        time.sleep(1)