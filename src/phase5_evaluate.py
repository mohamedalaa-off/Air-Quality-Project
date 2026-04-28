# ============================================================
# PHASE 5 — Model Evaluation
# ============================================================
# GOAL: Measure exactly how well the model performs on
#       data it has never seen before (the test set).
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             r2_score)
import torch
import torch.nn as nn
import joblib

# ── Model Architecture ───────────────────────────────────────
class AirQualityLSTM(nn.Module):

    def __init__(self,
                 input_size:  int,
                 hidden_size: int,
                 num_layers:  int,
                 output_size: int,
                 dropout:     float = 0.2):
        super(AirQualityLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        last_step = self.dropout(last_step)
        output = self.fc(last_step)
        return output


# ── Reload model, scalers, and test data from Phase 4 ───

# Load scalers
scaler_X = joblib.load('saved_models/scaler_X.pkl')
scaler_y = joblib.load('saved_models/scaler_y.pkl')

# Load featured data
df = pd.read_csv('outputs/data_features.csv',
                 index_col='datetime',
                 parse_dates=True)

TARGET = 'PM2.5'
FEATURE_COLS = [c for c in df.columns if c != TARGET]

# Chronological split (same as Phase 4)
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

# Scale test data
X_test = scaler_X.transform(df_test[FEATURE_COLS])
y_test = scaler_y.transform(df_test[[TARGET]])

# Create sequences (same LOOKBACK and HORIZON as Phase 4)
LOOKBACK = 48
HORIZON = 24

def create_sequences(X, y, lookback, horizon):
    Xs, ys = [], []
    for i in range(lookback, len(X) - horizon + 1):
        Xs.append(X[i - lookback : i])
        ys.append(y[i : i + horizon].flatten())
    return np.array(Xs), np.array(ys)

X_te_seq, y_te_seq = create_sequences(X_test, y_test, LOOKBACK, HORIZON)

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_te_t = torch.FloatTensor(X_te_seq).to(device)

# Reconstruct and load the model
INPUT_SIZE = X_te_seq.shape[2]
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = HORIZON

model = AirQualityLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE,
    dropout=0.2
).to(device)

model.load_state_dict(torch.load('saved_models/best_model.pth'))
print("Model loaded from saved_models/best_model.pth")

# ── 1. Generate predictions on test set ──────────────────────
model.eval()
with torch.no_grad():
    # X_te_t is the test tensor from Phase 4
    y_pred_scaled = model(X_te_t).cpu().numpy()

# Inverse transform: convert scaled [0,1] back to real μg/m³
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_te_seq)

# y_pred shape: (n_test_sequences, 24)
# y_true shape: (n_test_sequences, 24)

print(f"Predictions shape: {y_pred.shape}")
print(f"True values shape: {y_true.shape}")

# ── 2. Compute overall metrics ────────────────────────────────
# We flatten all 24 hours × all test sequences into one array
# to get a single overall score

y_pred_flat = y_pred.flatten()
y_true_flat = y_true.flatten()

mae  = mean_absolute_error(y_true_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
r2   = r2_score(y_true_flat, y_pred_flat)

# MAPE: we add a tiny value (1e-8) to the denominator
# to avoid division by zero when actual PM2.5 = 0
mape = np.mean(
    np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))
) * 100

print("=" * 50)
print("TEST SET RESULTS")
print("=" * 50)
print(f"  MAE    : {mae:.2f} μg/m³")
print(f"  RMSE   : {rmse:.2f} μg/m³")
print(f"  R²     : {r2:.4f}")
print(f"  MAPE   : {mape:.2f}%")
print("=" * 50)

# Interpretation guide
print("\nInterpretation:")
if mae < 15:
    print(f"  MAE {mae:.1f} → Excellent (< 15 is great for this dataset)")
elif mae < 25:
    print(f"  MAE {mae:.1f} → Good (< 25 is acceptable)")
else:
    print(f"  MAE {mae:.1f} → Needs improvement (try more features or tuning)")

if r2 > 0.85:
    print(f"  R² {r2:.3f} → Model explains {r2*100:.1f}% of variance — strong")
elif r2 > 0.70:
    print(f"  R² {r2:.3f} → Model explains {r2*100:.1f}% of variance — acceptable")
else:
    print(f"  R² {r2:.3f} → Model explains {r2*100:.1f}% of variance — weak")

# ── 3. Per-horizon analysis ───────────────────────────────────
# Does accuracy degrade as we forecast further into the future?
# We expect hour 1 to be more accurate than hour 24.

horizon_results = []
for h in range(HORIZON):
    mae_h  = mean_absolute_error(y_true[:, h], y_pred[:, h])
    rmse_h = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
    horizon_results.append({
        'hour': h + 1,
        'MAE':  mae_h,
        'RMSE': rmse_h
    })

df_horizon = pd.DataFrame(horizon_results)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(df_horizon['hour'], df_horizon['MAE'],
             'o-', color='steelblue', linewidth=2, markersize=5)
axes[0].set_title('MAE per Forecast Hour\n(should increase over time)')
axes[0].set_xlabel('Forecast hour ahead')
axes[0].set_ylabel('MAE (μg/m³)')
axes[0].set_xticks(range(1, 25, 2))
axes[0].grid(alpha=0.3)
axes[0].axhline(y=mae, color='tomato', linestyle='--',
                label=f'Overall MAE = {mae:.1f}')
axes[0].legend()

axes[1].plot(df_horizon['hour'], df_horizon['RMSE'],
             'o-', color='coral', linewidth=2, markersize=5)
axes[1].set_title('RMSE per Forecast Hour')
axes[1].set_xlabel('Forecast hour ahead')
axes[1].set_ylabel('RMSE (μg/m³)')
axes[1].set_xticks(range(1, 25, 2))
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('saved_models/horizon_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 4. Visual comparison: actual vs predicted ─────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

# Pick 4 interesting test samples
# We look for one good, one bad, one high pollution, one low
sample_indices = [10, 50, 100, 200]

for i, idx in enumerate(sample_indices):
    ax = axes[i]
    ax.plot(range(1, 25), y_true[idx],
            'o-',  color='steelblue',
            linewidth=2, markersize=5,
            label='Actual PM2.5')
    ax.plot(range(1, 25), y_pred[idx],
            's--', color='tomato',
            linewidth=2, markersize=5,
            label='Predicted PM2.5')
    ax.fill_between(range(1, 25),
                    y_true[idx], y_pred[idx],
                    alpha=0.12, color='gray',
                    label='Error region')

    sample_mae = mean_absolute_error(y_true[idx], y_pred[idx])
    ax.set_title(f'Test Sample {idx} | MAE = {sample_mae:.1f} μg/m³')
    ax.set_xlabel('Hours ahead')
    ax.set_ylabel('PM2.5 (μg/m³)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Actual vs Predicted — 24-Hour Forecasts', fontsize=13)
plt.tight_layout()
plt.savefig('saved_models/predictions_sample.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: saved_models/predictions_sample.png")

# ── 5. Simple baseline comparison ────────────────────────────
# A good model should beat a trivial baseline.
# Baseline: "predict same PM2.5 as 24 hours ago" (persistence model)
# If your LSTM doesn't beat this, something is wrong.

y_baseline = y_true[:, 0:1].repeat(HORIZON, axis=1)
mae_baseline = mean_absolute_error(y_true.flatten(), y_baseline.flatten())

print(f"\n=== Baseline Comparison ===")
print(f"Persistence baseline MAE : {mae_baseline:.2f} μg/m³")
print(f"Your LSTM MAE            : {mae:.2f} μg/m³")
print(f"Improvement              : {((mae_baseline - mae)/mae_baseline*100):.1f}%")

