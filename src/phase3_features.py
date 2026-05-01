import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the clean data from Phase 2
df_clean = pd.read_csv('outputs/data_clean.csv', index_col='datetime', parse_dates=True)
print(f"Loaded clean data: {df_clean.shape}")

df_feat = df_clean.copy()

# ── 1. Cyclical Time Encoding ─────────────────────────────────
# WHY: The hour column (0–23) has a discontinuity:
#   hour 23 and hour 0 are consecutive but numerically far apart.
# HOW: We encode hour as TWO values — sine and cosine.

df_feat['hour']        = df_feat.index.hour
df_feat['day_of_week'] = df_feat.index.dayofweek    # 0=Monday, 6=Sunday
df_feat['month']       = df_feat.index.month
df_feat['is_weekend']  = df_feat['day_of_week'].isin([5, 6]).astype(int)

# Cyclical encoding for hour (24-hour cycle)
df_feat['hour_sin']  = np.sin(2 * np.pi * df_feat['hour']  / 24)
df_feat['hour_cos']  = np.cos(2 * np.pi * df_feat['hour']  / 24)

# Cyclical encoding for month (12-month cycle)
df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)

# Cyclical encoding for day of week (7-day cycle)
df_feat['dow_sin']   = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
df_feat['dow_cos']   = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)

print("Cyclical features added: hour, month, dow (sin/cos encoding)")

# ── 2. Rolling Statistics ─────────────────────────────────────
# WHY: Captures recent trends and volatility without forcing
#      the LSTM to memorize exact historical points (lags).

# 24-hour rolling mean — captures daily average level
df_feat['PM25_roll_24h'] = df_feat['PM2.5'].rolling(window=24, min_periods=1).mean()

# 6-hour rolling std — captures how stable PM2.5 has been recently
df_feat['PM25_roll_std_6h'] = df_feat['PM2.5'].rolling(window=6, min_periods=1).std().fillna(0)

print("Rolling features added: PM25_roll_24h, PM25_roll_std_6h")

# ── 3. Drop NaN rows ──────────────────────────────────────────
# Clean up any initial rows that might have NaN from rolling calculations

rows_before = len(df_feat)
df_feat = df_feat.dropna()
rows_after = len(df_feat)

print(f"\nRows dropped due to NaN: {rows_before - rows_after}")
print(f"Final shape: {df_feat.shape}")
print(f"\nAll active features ({df_feat.shape[1]} total):")
for i, col in enumerate(df_feat.columns, 1):
    print(f"  {i:2d}. {col}")

# ── 4. Visualize the most important new features ──────────────
# Adjusted to a 1x3 layout to neatly display the active features

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Feature Engineering — Visual Verification', fontsize=14, fontweight='bold')

sample = df_feat.iloc[:500]    # first 500 rows for visual clarity

# Plot A: PM2.5 vs its rolling mean
axes[0].plot(sample.index, sample['PM2.5'], alpha=0.5, linewidth=1, label='Raw PM2.5', color='gray')
axes[0].plot(sample.index, sample['PM25_roll_24h'], linewidth=2, label='24h rolling mean', color='tomato')
axes[0].set_title('PM2.5 vs 24h Rolling Mean')
axes[0].legend(fontsize=9)
axes[0].tick_params(axis='x', rotation=30)
axes[0].grid(alpha=0.2)

# Plot B: Cyclical hour encoding
hours = np.arange(24)
axes[1].plot(hours, np.sin(2*np.pi*hours/24), 'o-', label='hour_sin', color='mediumpurple')
axes[1].plot(hours, np.cos(2*np.pi*hours/24), 's-', label='hour_cos', color='steelblue')
axes[1].set_title('Cyclical Hour Encoding')
axes[1].set_xlabel('Hour of day')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot C: Rolling std — captures volatility
axes[2].plot(sample.index, sample['PM25_roll_std_6h'], color='coral', linewidth=1.5)
axes[2].set_title('6h Rolling Std (PM2.5 Volatility)')
axes[2].set_ylabel('Standard Deviation')
axes[2].tick_params(axis='x', rotation=30)
axes[2].grid(alpha=0.2)

plt.tight_layout()
plt.savefig('outputs/charts/feature_engineering.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved chart: outputs/charts/feature_engineering.png")

# ── 5. Save the featured data ─────────────────────────────────
df_feat.to_csv('outputs/data_features.csv')
print(f"Featured data saved to: outputs/data_features.csv")