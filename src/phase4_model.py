# ============================================================
# PHASE 4 — Splitting, Sequences, Architecture & Training
# ============================================================
# GOAL: Split data correctly → create sequences → build LSTM
#       → train with early stopping → save best model
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

# Create output directory
os.makedirs('saved_models', exist_ok=True)

# ── 1. Load featured data ─────────────────────────────────────
df = pd.read_csv('outputs/data_features.csv',
                 index_col='datetime',
                 parse_dates=True)

print(f"Loaded: {df.shape}")

TARGET      = 'PM2.5'
FEATURE_COLS = [c for c in df.columns if c != TARGET]

print(f"Target column  : {TARGET}")
print(f"Feature columns: {len(FEATURE_COLS)}")

# ── 2. Chronological split ────────────────────────────────────
# Rule: ALWAYS split time-series by time, never randomly
# 70% train | 15% validation | 15% test

n        = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

df_train = df.iloc[:train_end]
df_val   = df.iloc[train_end:val_end]
df_test  = df.iloc[val_end:]

print(f"\n=== Data Split ===")
print(f"Train:      {len(df_train):,} rows | "
      f"{df_train.index[0].date()} → {df_train.index[-1].date()}")
print(f"Validation: {len(df_val):,} rows | "
      f"{df_val.index[0].date()} → {df_val.index[-1].date()}")
print(f"Test:       {len(df_test):,} rows | "
      f"{df_test.index[0].date()} → {df_test.index[-1].date()}")

# ── 3. Scaling ────────────────────────────────────────────────
# fit_transform on TRAIN: learns min/max AND scales in one step
# transform only on VAL and TEST: uses the min/max from training
#
# Why two separate scalers?
#   scaler_X: for the input features
#   scaler_y: for the PM2.5 target
#   We need scaler_y separately so we can inverse-transform
#   predictions back to real μg/m³ values for evaluation

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train = scaler_X.fit_transform(df_train[FEATURE_COLS])
X_val   = scaler_X.transform(df_val[FEATURE_COLS])
X_test  = scaler_X.transform(df_test[FEATURE_COLS])

# Note: we pass [[TARGET]] with double brackets to keep 2D shape
# required by the scaler (it expects shape (n_rows, n_columns))
y_train = scaler_y.fit_transform(df_train[[TARGET]])
y_val   = scaler_y.transform(df_val[[TARGET]])
y_test  = scaler_y.transform(df_test[[TARGET]])

# Save scalers — we need them for deployment later
joblib.dump(scaler_X, 'saved_models/scaler_X.pkl')
joblib.dump(scaler_y, 'saved_models/scaler_y.pkl')

print("\nScalers saved.")
print(f"X_train scaled shape: {X_train.shape}")

# ── 4. Create LSTM sequences ──────────────────────────────────
# This is the most important transformation before feeding to LSTM.
#
# The LSTM needs windows of consecutive hours as input.
# LOOKBACK=48: use the past 48 hours as input
# HORIZON =24: predict the next 24 hours
#
# Example with LOOKBACK=3, HORIZON=2:
#   Raw:       [10, 15, 20, 25, 30, 35]
#   Seq 1 in:  [10, 15, 20]  →  Seq 1 out: [25, 30]
#   Seq 2 in:  [15, 20, 25]  →  Seq 2 out: [30, 35]

LOOKBACK = 48    # hours of history to show the model
HORIZON  = 24    # hours into the future to predict

def create_sequences(X: np.ndarray,
                     y: np.ndarray,
                     lookback: int,
                     horizon: int):
    """
    Convert flat arrays into overlapping LSTM sequences.

    Parameters:
        X        : features, shape (n_rows, n_features)
        y        : target,   shape (n_rows, 1)
        lookback : input window size
        horizon  : output window size

    Returns:
        Xs: shape (n_sequences, lookback, n_features)
        ys: shape (n_sequences, horizon)
    """
    Xs, ys = [], []

    # We need at least (lookback + horizon) rows to form one sequence
    for i in range(lookback, len(X) - horizon + 1):
        # Input: the lookback window before position i
        Xs.append(X[i - lookback : i])

        # Output: the horizon window starting at position i
        # .flatten() converts shape (horizon, 1) → (horizon,)
        ys.append(y[i : i + horizon].flatten())

    return np.array(Xs), np.array(ys)


X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, LOOKBACK, HORIZON)
X_vl_seq, y_vl_seq = create_sequences(X_val,   y_val,   LOOKBACK, HORIZON)
X_te_seq, y_te_seq = create_sequences(X_test,  y_test,  LOOKBACK, HORIZON)

print(f"\n=== Sequence Shapes ===")
print(f"X_train: {X_tr_seq.shape}  ← (sequences, 48_hours, features)")
print(f"y_train: {y_tr_seq.shape}  ← (sequences, 24_hours)")
print(f"X_val:   {X_vl_seq.shape}")
print(f"X_test:  {X_te_seq.shape}")


# ── 5. Convert to PyTorch Tensors ─────────────────────────────
# PyTorch works with Tensor objects, not NumPy arrays.
# FloatTensor creates a 32-bit float tensor — standard for neural nets.
# .to(device) moves the tensor to GPU if available, else stays on CPU.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining device: {device}")

def to_tensor(arr):
    return torch.FloatTensor(arr).to(device)

X_tr_t = to_tensor(X_tr_seq)
y_tr_t = to_tensor(y_tr_seq)
X_vl_t = to_tensor(X_vl_seq)
y_vl_t = to_tensor(y_vl_seq)
X_te_t = to_tensor(X_te_seq)
y_te_t = to_tensor(y_te_seq)

# DataLoader batches the data and handles shuffling
# shuffle=True for training only — never for val/test
BATCH_SIZE = 32    # smaller batch = better for CPU

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_vl_t, y_vl_t),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches  : {len(val_loader)}")

# ── 6. LSTM Model Architecture ────────────────────────────────
# The model has three sections:
#
# Section 1 — LSTM layers:
#   Reads the 48-hour sequence step by step.
#   At each step, updates its internal memory (hidden state).
#   After the last step, we take the final hidden state —
#   a compressed summary of everything the model saw.
#
# Section 2 — Dropout:
#   Randomly turns off neurons during training.
#   Forces the network to not rely on any single path.
#   This is our main weapon against overfitting.
#
# Section 3 — Fully Connected (FC) layers:
#   Takes the LSTM summary (128 numbers) and maps it to
#   24 numbers (one prediction per forecast hour).

class AirQualityLSTM(nn.Module):

    def __init__(self,
                 input_size:  int,
                 hidden_size: int,
                 num_layers:  int,
                 output_size: int,
                 dropout:     float = 0.2):
        super(AirQualityLSTM, self).__init__()

        # LSTM layer
        # input_size  = number of features per time step
        # hidden_size = size of the LSTM's memory (hyperparameter)
        # num_layers  = how many LSTM layers stacked on top of each other
        # batch_first = True means input shape is (batch, time, features)
        # dropout     = applied between LSTM layers (only if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )

        # Dropout applied after LSTM before FC layers
        self.dropout = nn.Dropout(dropout)

        # Fully connected output head
        # Maps LSTM output → 24 predictions
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),  # compress 128 → 64
            nn.ReLU(),                    # activation: zero-out negatives
            nn.Dropout(dropout),          # regularize again
            nn.Linear(64, output_size)   # 64 → 24 predictions
        )

    def forward(self, x):
        # x shape: (batch_size, lookback=48, input_size)

        # Pass sequence through LSTM
        # lstm_out: output at every time step, shape (batch, 48, hidden)
        # We don't need the hidden/cell state here, so we use _
        lstm_out, _ = self.lstm(x)

        # Take ONLY the last time step
        # lstm_out[:, -1, :] means:
        #   : = all batches
        #   -1 = last time step (hour 48)
        #   : = all hidden units
        last_step = lstm_out[:, -1, :]

        # Apply dropout for regularization
        last_step = self.dropout(last_step)

        # Pass through FC layers to get 24 predictions
        output = self.fc(last_step)

        return output    # shape: (batch_size, 24)


# ── 7. Instantiate the model ──────────────────────────────────
INPUT_SIZE  = X_tr_seq.shape[2]   # = number of feature columns
HIDDEN_SIZE = 128                  # LSTM memory units
NUM_LAYERS  = 2                    # stacked LSTM layers
OUTPUT_SIZE = HORIZON              # = 24

model = AirQualityLSTM(
    input_size  = INPUT_SIZE,
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    output_size = OUTPUT_SIZE,
    dropout     = 0.2
).to(device)

# Count trainable parameters
n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"\nModel created.")
print(f"Input size  : {INPUT_SIZE}")
print(f"Parameters  : {n_params:,}")
print(model)

# ── 8. Training configuration ─────────────────────────────────
EPOCHS    = 100
LR        = 0.001
PATIENCE  = 15       # early stopping patience

# Loss function: Mean Squared Error
# For each prediction: squares the error (actual - predicted)²
# Takes the average across all predictions in the batch
criterion = nn.MSELoss()

# Optimizer: Adam
# Automatically adapts the learning rate per parameter
# weight_decay: L2 regularization — penalizes large weights
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5
)

# Learning rate scheduler
# When validation loss stops improving for 7 epochs,
# divide the learning rate by 2 (factor=0.5)
# This lets the model take smaller, more careful steps
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=7,
)

# ── 9. Training loop ──────────────────────────────────────────
history          = {'train_loss': [], 'val_loss': []}
best_val_loss    = float('inf')    # start with infinity
patience_counter = 0

print("\nStarting training...")
print("-" * 65)

for epoch in range(1, EPOCHS + 1):

    # ── Training phase ──────────────────────────────────────
    # model.train() enables dropout (needed during training)
    model.train()
    batch_losses = []

    for X_batch, y_batch in train_loader:

        # Zero gradients from the previous batch
        # PyTorch accumulates gradients — we reset each batch
        optimizer.zero_grad()

        # Forward pass: compute predictions
        predictions = model(X_batch)

        # Compute loss: how wrong are we?
        loss = criterion(predictions, y_batch)

        # Backward pass: compute gradients
        # This calculates how each weight should change
        loss.backward()

        # Gradient clipping: prevents exploding gradients
        # A common issue with LSTMs on long sequences
        # Clips all gradients so their combined norm ≤ 1.0
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights using the optimizer
        optimizer.step()

        batch_losses.append(loss.item())

    avg_train_loss = np.mean(batch_losses)

    # ── Validation phase ────────────────────────────────────
    # model.eval() disables dropout (we want deterministic output)
    model.eval()

    # torch.no_grad() tells PyTorch not to compute gradients
    # This saves memory and speeds up the forward pass
    with torch.no_grad():
        val_preds = model(X_vl_t)
        val_loss  = criterion(val_preds, y_vl_t).item()

    # Record history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)

    # Step the scheduler
    scheduler.step(val_loss)

    # ── Early stopping ──────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        # Save the best model weights to disk
        torch.save(model.state_dict(), 'saved_models/best_model.pth')
    else:
        patience_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  |  "
              f"Train: {avg_train_loss:.5f}  |  "
              f"Val: {val_loss:.5f}  |  "
              f"Patience: {patience_counter}/{PATIENCE}  |  "
              f"Best: {best_val_loss:.5f}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

# Load the best weights (not necessarily from the last epoch)
model.load_state_dict(torch.load('saved_models/best_model.pth'))
print("\nBest model loaded from checkpoint.")

# ── 10. Plot training curves ──────────────────────────────────
# This plot tells you whether your model is healthy
#
# GOOD:        Both curves decrease and end close together
# OVERFIT:     Train loss keeps falling, val loss starts rising
# UNDERFIT:    Both losses are high and barely decreasing

plt.figure(figsize=(11, 5))
epochs_ran = range(1, len(history['train_loss']) + 1)

plt.plot(epochs_ran, history['train_loss'],
         label='Training loss',   color='steelblue', linewidth=2)
plt.plot(epochs_ran, history['val_loss'],
         label='Validation loss', color='tomato',    linewidth=2)

# Mark where early stopping triggered
plt.axvline(x=len(history['train_loss']),
            color='gray', linestyle=':', alpha=0.7,
            label=f'Stopped at epoch {len(history["train_loss"])}')

plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss — Is the model healthy?')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/charts/straining_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: outputs/charts/training_curve.png")

# Save feature column names for deployment
import json
with open('saved_models/Charts/feature_cols.json', 'w') as f:
    json.dump(FEATURE_COLS, f)
print("Feature columns saved.")