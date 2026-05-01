import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import joblib
import os

# Create directory for saving evaluation charts
os.makedirs('outputs/charts', exist_ok=True)

# ── 1. Model Architecture (Must match Phase 4 exactly) ───────
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

# ── 2. Reload Data and Scalers ───────────────────────────────
scaler_X = joblib.load('saved_models/scaler_X.pkl')
scaler_y = joblib.load('saved_models/scaler_y.pkl')

df = pd.read_csv('outputs/data_features.csv', index_col='datetime', parse_dates=True)

TARGET = 'PM2.5'
FEATURE_COLS = df.columns.tolist()  # PM2.5 is included!
pm25_idx = FEATURE_COLS.index(TARGET)

# Split data (using the same 85% index as Phase 4 for Test)
n = len(df)
val_end = int(n * 0.85)
df_test = df.iloc[val_end:]

# Scale test data
X_test = scaler_X.transform(df_test[FEATURE_COLS])
y_test = scaler_y.transform(df_test[[TARGET]])

LOOKBACK = 48
HORIZON = 24  # Evaluate on a full 24-hour horizon

def create_eval_sequences(X, y, lookback, horizon):
    Xs, ys, last_known = [], [], []
    for i in range(lookback, len(X) - horizon + 1):
        Xs.append(X[i - lookback : i + horizon]) # 48 past + 24 future hours
        ys.append(y[i : i + horizon].flatten())  # 24 hours true target
        last_known.append(y[i - 1])              # For persistence baseline
    return np.array(Xs), np.array(ys), np.array(last_known)

X_full_seq, y_te_seq, y_last_known = create_eval_sequences(X_test, y_test, LOOKBACK, HORIZON)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 3. Load Trained Model ────────────────────────────────────
INPUT_SIZE  = X_full_seq.shape[2]
HIDDEN_SIZE = 32
NUM_LAYERS  = 1
OUTPUT_SIZE = 1  # Model predicts 1 step at a time!

model = AirQualityLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, dropout=0.3).to(device)
model.load_state_dict(torch.load('saved_models/best_model.pth', map_location=device))
model.eval()
print(f"Model loaded on {device}. Starting Recursive Forecasting...")

# ── 4. Recursive Forecasting Loop ────────────────────────────
y_pred_scaled = []

with torch.no_grad():
    for i in range(len(X_full_seq)):
        # Start with the 48-hour history
        current_window = X_full_seq[i, :LOOKBACK, :].copy() 
        sample_preds = []

        for h in range(HORIZON):
            # Predict the next hour
            seq_tensor = torch.FloatTensor(current_window).unsqueeze(0).to(device)
            pred = model(seq_tensor).cpu().numpy()[0, 0]
            sample_preds.append(pred)

            # Recursive Update: Shift window and inject prediction
            if h < HORIZON - 1:
                next_hour_features = X_full_seq[i, LOOKBACK + h, :].copy()
                next_hour_features[pm25_idx] = pred  # Inject predicted PM2.5
                current_window = np.vstack((current_window[1:], next_hour_features))

        y_pred_scaled.append(sample_preds)

y_pred_scaled = np.array(y_pred_scaled)

# ── 5. Standard Metrics & Baseline Comparison ────────────────
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_te_seq)

mae  = mean_absolute_error(y_true.flatten(), y_pred.flatten())
rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
r2   = r2_score(y_true.flatten(), y_pred.flatten())

print("\n" + "=" * 50)
print("  TEST SET RESULTS (24-Hour Recursive)")
print("=" * 50)
print(f"  MAE    : {mae:.2f} μg/m³")
print(f"  RMSE   : {rmse:.2f} μg/m³")
print(f"  R²     : {r2:.4f}")
print("=" * 50)

# True Baseline (Persistence)
last_known_pm25 = scaler_y.inverse_transform(y_last_known)
y_baseline = np.repeat(last_known_pm25, HORIZON, axis=1)
mae_baseline = mean_absolute_error(y_true.flatten(), y_baseline.flatten())

print(f"\n=== Baseline Comparison ===")
print(f"  Persistence baseline MAE : {mae_baseline:.2f} μg/m³")
print(f"  Recursive LSTM MAE       : {mae:.2f} μg/m³")
improvement = ((mae_baseline - mae) / mae_baseline) * 100
print(f"  Improvement              : {improvement:.1f}%\n")

# ── 6. Custom Accuracy Metrics ───────────────────────────────
ACCEPTABLE_ERROR = 15.0

# Threshold Accuracy
absolute_errors = np.abs(y_true - y_pred)
correct_threshold = (absolute_errors <= ACCEPTABLE_ERROR)
threshold_acc_per_hour = np.mean(correct_threshold, axis=0) * 100

# Directional Accuracy (starting from hour 2)
diff_true = np.diff(y_true, axis=1)
diff_pred = np.diff(y_pred, axis=1)
correct_direction = (np.sign(diff_true) == np.sign(diff_pred))
directional_acc_per_hour = np.mean(correct_direction, axis=0) * 100

print(f"=== Custom Accuracy ===")
print(f"  Overall Threshold Accuracy (±{ACCEPTABLE_ERROR} μg/m³) : {np.mean(threshold_acc_per_hour):.1f}%")
print(f"  Overall Directional Accuracy             : {np.mean(directional_acc_per_hour):.1f}%\n")

# ── 7. Visualizations ────────────────────────────────────────

# Plot A: Actual vs Predicted Samples
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
sample_indices = [10, 50, 100, 200]

for i, idx in enumerate(sample_indices):
    ax = axes[i]
    ax.plot(range(1, 25), y_true[idx], 'o-', color='steelblue', linewidth=2, label='Actual PM2.5')
    ax.plot(range(1, 25), y_pred[idx], 's--', color='tomato', linewidth=2, label='Predicted PM2.5')
    ax.set_title(f'Test Sample {idx} | MAE = {mean_absolute_error(y_true[idx], y_pred[idx]):.1f}')
    ax.set_xlabel('Hours Ahead')
    ax.set_ylabel('PM2.5 (μg/m³)')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/charts/predictions_sample.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot B: Accuracy Metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Accuracy Metrics over 24-Hour Horizon', fontsize=14, fontweight='bold')

# Threshold Accuracy
hours = np.arange(1, 25)
axes[0].plot(hours, threshold_acc_per_hour, 'o-', color='mediumseagreen', linewidth=2)
axes[0].set_title(f'Threshold Accuracy\n(% predictions within ±{ACCEPTABLE_ERROR} μg/m³)')
axes[0].set_xlabel('Forecast hour ahead')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 105)
axes[0].set_xticks(range(1, 25, 2))
axes[0].grid(alpha=0.3)
axes[0].axhline(y=np.mean(threshold_acc_per_hour), color='gray', linestyle='--', label=f'Avg: {np.mean(threshold_acc_per_hour):.1f}%')
axes[0].legend()

# Directional Accuracy
hours_diff = np.arange(2, 25)
axes[1].plot(hours_diff, directional_acc_per_hour, 's-', color='cornflowerblue', linewidth=2)
axes[1].set_title('Directional Accuracy\n(% correctly guessed UP or DOWN)')
axes[1].set_xlabel('Forecast hour ahead')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(0, 105)
axes[1].set_xticks(range(2, 25, 2))
axes[1].grid(alpha=0.3)
axes[1].axhline(y=np.mean(directional_acc_per_hour), color='gray', linestyle='--', label=f'Avg: {np.mean(directional_acc_per_hour):.1f}%')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/charts/custom_accuracies.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved evaluation charts to: outputs/charts/")