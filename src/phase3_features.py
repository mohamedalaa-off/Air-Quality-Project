# ============================================================
# PHASE 3 — Feature Engineering
# ============================================================
# GOAL: Create new informative features from existing columns.
# Every feature we add should have a clear reason why it helps.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the clean data from Phase 2
df_clean = pd.read_csv('outputs/data_clean.csv', index_col='datetime', parse_dates=True)

print(f"Loaded clean data: {df_clean.shape}")

df_feat = df_clean.copy()

# ── 1. Cyclical Time Encoding ─────────────────────────────────
#
# WHY: The hour column (0–23) has a discontinuity:
#   hour 23 and hour 0 are consecutive but numerically far apart.
#   The model would think 23 and 0 are unrelated.
#
# HOW: We encode hour as TWO values — sine and cosine.
#   On a circle, hour 23 and hour 0 are adjacent.
#   The model now understands the circular nature of time.
#
# Formula: sin(2π × hour / 24), cos(2π × hour / 24)

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

print("Cyclical features added: hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos")

# ── 2. Rolling Statistics ─────────────────────────────────────
#
# WHY: PM2.5 at this hour is heavily influenced by what
#   happened in the previous hours. A rising trend in the
#   past 6 hours strongly suggests PM2.5 will keep rising.
#
# HOW: For each row, we look back N hours and compute:
#   - mean: the average level (is it generally high or low?)
#   - std:  the volatility (is it stable or wildly changing?)
#
# min_periods=1 means: compute even if we have fewer than
#   'window' rows available (handles the start of the dataset)

# 3-hour rolling mean — captures very recent trend
df_feat['PM25_roll_3h']  = (
    df_feat['PM2.5'].rolling(window=3,  min_periods=1).mean()
)

# 6-hour rolling mean — captures short-term trend
df_feat['PM25_roll_6h']  = (
    df_feat['PM2.5'].rolling(window=6,  min_periods=1).mean()
)

# 24-hour rolling mean — captures daily average level
df_feat['PM25_roll_24h'] = (
    df_feat['PM2.5'].rolling(window=24, min_periods=1).mean()
)

# 6-hour rolling std — captures how stable PM2.5 has been recently
# High std = volatile / rapidly changing conditions
df_feat['PM25_roll_std_6h'] = (
    df_feat['PM2.5'].rolling(window=6, min_periods=1).std().fillna(0)
)

# CO rolling — CO is a strong proxy for traffic/industrial pollution
df_feat['CO_roll_6h'] = (
    df_feat['CO'].rolling(window=6, min_periods=1).mean()
)

print("Rolling features added: PM25_roll_3h, 6h, 24h, std_6h, CO_roll_6h")

# ── 3. Lag Features ───────────────────────────────────────────
#
# WHY: PM2.5 has a very strong daily cycle.
#   Knowing what the air was like at the same hour yesterday
#   is one of the single best predictors of today's reading.
#
# HOW: shift(N) moves all values DOWN by N rows.
#   So at row i, the value equals what was at row i-N.
#   This gives the model "memory" without needing the full sequence.
#
# Note: shift creates NaN in the first N rows.
#   We will drop those NaN rows at the end.

# 1 hour ago
df_feat['PM25_lag_1h']  = df_feat['PM2.5'].shift(1)

# 3 hours ago
df_feat['PM25_lag_3h']  = df_feat['PM2.5'].shift(3)

# 6 hours ago
df_feat['PM25_lag_6h']  = df_feat['PM2.5'].shift(6)

# 24 hours ago (same hour yesterday) — captures daily cycle
df_feat['PM25_lag_24h'] = df_feat['PM2.5'].shift(24)

# 48 hours ago (same hour 2 days ago) — reinforces daily pattern
df_feat['PM25_lag_48h'] = df_feat['PM2.5'].shift(48)

# Wind speed 3 hours ago — wind takes time to clear pollution
df_feat['WSPM_lag_3h']  = df_feat['WSPM'].shift(3)

print("Lag features added: PM25_lag_1h, 3h, 6h, 24h, 48h, WSPM_lag_3h")

# ── 4. Drop NaN rows ──────────────────────────────────────────
# The lag and rolling features create NaN in the first N rows.
# We drop them here — they cannot be used for training anyway.

rows_before = len(df_feat)
df_feat = df_feat.dropna()
rows_after = len(df_feat)

print(f"\nRows dropped due to NaN from lags: {rows_before - rows_after}")
print(f"Final shape: {df_feat.shape}")
print(f"\nAll features ({df_feat.shape[1]} total):")
for i, col in enumerate(df_feat.columns, 1):
    print(f"  {i:2d}. {col}")

# ── 5. Visualize the most important new features ──────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle('Feature Engineering — Visual Verification', fontsize=13)

sample = df_feat.iloc[:500]    # first 500 rows for clarity

# Plot A: PM2.5 vs its rolling mean
axes[0,0].plot(sample.index, sample['PM2.5'],
               alpha=0.5, linewidth=0.7,
               label='Raw PM2.5', color='gray')
axes[0,0].plot(sample.index, sample['PM25_roll_24h'],
               linewidth=2, label='24h rolling mean', color='tomato')
axes[0,0].set_title('PM2.5 vs 24h Rolling Mean')
axes[0,0].legend(fontsize=9)
axes[0,0].tick_params(axis='x', rotation=30)

# Plot B: Scatter PM2.5 vs lag_24h
# If this shows a strong diagonal, the lag feature is valuable
axes[0,1].scatter(df_feat['PM25_lag_24h'], df_feat['PM2.5'],
                  alpha=0.05, s=2, color='steelblue')
axes[0,1].set_xlabel('PM2.5 (24h ago)')
axes[0,1].set_ylabel('PM2.5 (now)')
axes[0,1].set_title('Does yesterday predict today?')
corr_lag = df_feat['PM25_lag_24h'].corr(df_feat['PM2.5'])
axes[0,1].text(0.05, 0.92, f'Correlation: {corr_lag:.3f}',
               transform=axes[0,1].transAxes, fontsize=10,
               color='tomato', fontweight='bold')

# Plot C: Cyclical hour encoding
hours = np.arange(24)
axes[1,0].plot(hours, np.sin(2*np.pi*hours/24),
               'o-', label='hour_sin', color='mediumpurple')
axes[1,0].plot(hours, np.cos(2*np.pi*hours/24),
               's-', label='hour_cos', color='steelblue')
axes[1,0].set_title('Cyclical Hour Encoding')
axes[1,0].set_xlabel('Hour of day')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# Plot D: Rolling std — captures volatility
axes[1,1].plot(sample.index, sample['PM25_roll_std_6h'],
               color='coral', linewidth=0.8)
axes[1,1].set_title('6h Rolling Std (PM2.5 Volatility)')
axes[1,1].set_ylabel('Standard Deviation')
axes[1,1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('outputs/charts/feature_engineering.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: feature_engineering.png")

# ── 6. Save the featured data ─────────────────────────────────
df_feat.to_csv('outputs/data_features.csv')
print(f"\nFeatured data saved to: data_features.csv")
