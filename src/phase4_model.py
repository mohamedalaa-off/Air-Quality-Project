import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import json

# ── 1. Create necessary directories ─────────────────────────────
os.makedirs('saved_models/Charts', exist_ok=True)
os.makedirs('outputs/charts', exist_ok=True)

# ── 2. Load featured data ─────────────────────────────────────
df = pd.read_csv('outputs/data_features.csv', index_col='datetime', parse_dates=True)
print(f"Loaded data shape: {df.shape}")

TARGET = 'PM2.5'
FEATURE_COLS = df.columns.tolist()  # PM2.5 is included as a feature

print(f"Target column  : {TARGET}")
print(f"Feature columns: {len(FEATURE_COLS)}")

# ── 3. Chronological split (70% Train | 15% Val | 15% Test) ───
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

df_train = df.iloc[:train_end]
df_val   = df.iloc[train_end:val_end]
df_test  = df.iloc[val_end:]

print(f"\n=== Data Split ===")
print(f"Train:      {len(df_train):,} rows")
print(f"Validation: {len(df_val):,} rows")
print(f"Test:       {len(df_test):,} rows")

# ── 4. Scaling ────────────────────────────────────────────────
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train = scaler_X.fit_transform(df_train[FEATURE_COLS])
X_val   = scaler_X.transform(df_val[FEATURE_COLS])
X_test  = scaler_X.transform(df_test[FEATURE_COLS])

y_train = scaler_y.fit_transform(df_train[[TARGET]])
y_val   = scaler_y.transform(df_val[[TARGET]])
y_test  = scaler_y.transform(df_test[[TARGET]])

# Save scalers for deployment
joblib.dump(scaler_X, 'saved_models/scaler_X.pkl')
joblib.dump(scaler_y, 'saved_models/scaler_y.pkl')
print("\nScalers saved.")

# ── 5. Create LSTM sequences ──────────────────────────────────
LOOKBACK = 48    # 48 hours of history
HORIZON  = 1     # Predict 1 hour ahead (for recursive forecasting)

def create_sequences(X, y, lookback, horizon):
    Xs, ys = [], []
    for i in range(lookback, len(X) - horizon + 1):
        Xs.append(X[i - lookback : i])
        ys.append(y[i : i + horizon].flatten())
    return np.array(Xs), np.array(ys)

X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, LOOKBACK, HORIZON)
X_vl_seq, y_vl_seq = create_sequences(X_val,   y_val,   LOOKBACK, HORIZON)
X_te_seq, y_te_seq = create_sequences(X_test,  y_test,  LOOKBACK, HORIZON)

print(f"\nSequence Shapes (Batch, TimeSteps, Features):")
print(f"X_train: {X_tr_seq.shape} | y_train: {y_tr_seq.shape}")

# ── 6. Convert to PyTorch Tensors ─────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining on device: {device}")

def to_tensor(arr):
    return torch.FloatTensor(arr).to(device)

BATCH_SIZE = 32

train_loader = DataLoader(TensorDataset(to_tensor(X_tr_seq), to_tensor(y_tr_seq)), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(to_tensor(X_vl_seq), to_tensor(y_vl_seq)), batch_size=BATCH_SIZE, shuffle=False)

# ── 7. LSTM Model Architecture ────────────────────────────────
class AirQualityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(AirQualityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = self.dropout(lstm_out[:, -1, :])
        return self.fc(last_step)

INPUT_SIZE  = X_tr_seq.shape[2]
HIDDEN_SIZE = 32
NUM_LAYERS  = 1
OUTPUT_SIZE = HORIZON

model = AirQualityLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, dropout=0.3).to(device)

# ── 8. Training Configuration ─────────────────────────────────
EPOCHS   = 100
LR       = 0.001
PATIENCE = 15

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

# ── 9. Training Loop ──────────────────────────────────────────
history = {'train_loss': [], 'val_loss': []}
best_val_loss = float('inf')
patience_counter = 0

print("\nStarting training...")
print("-" * 65)

for epoch in range(1, EPOCHS + 1):
    
    # Training phase
    model.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())
        
    avg_train_loss = np.mean(batch_losses)

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_preds = model(to_tensor(X_vl_seq))
        val_loss  = criterion(val_preds, to_tensor(y_vl_seq)).item()

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'saved_models/best_model.pth')
    else:
        patience_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {avg_train_loss:.5f} | Val: {val_loss:.5f} | Patience: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

print("\nBest model saved to: saved_models/best_model.pth")

# ── 10. Save configurations & plots ───────────────────────────
# Save feature columns
with open('saved_models/Charts/feature_cols.json', 'w') as f:
    json.dump(FEATURE_COLS, f)
print("Feature columns saved.")

# Plot learning curves
plt.figure(figsize=(10, 5))
epochs_ran = range(1, len(history['train_loss']) + 1)
plt.plot(epochs_ran, history['train_loss'], label='Training loss', color='steelblue', linewidth=2)
plt.plot(epochs_ran, history['val_loss'], label='Validation loss', color='tomato', linewidth=2)
plt.axvline(x=len(history['train_loss']), color='gray', linestyle=':', alpha=0.7, label=f'Stopped at epoch {len(history["train_loss"])}')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/charts/training_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved chart: outputs/charts/training_curve.png")