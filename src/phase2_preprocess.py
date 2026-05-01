# ── 02_preprocess.py ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import glob
import os

# (Assume df_all is already loaded from Phase 1)
DATA_PATH = "data\PRSA_Data_20130301-20170228"

all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

print(f"Files found: {len(all_files)}")

dfs = []
for filepath in all_files:
    df = pd.read_csv(filepath)

    # Extract station name from filename
    # Example: PRSA_Data_Aotizhongxin_20130301-20170228.csv
    # .split('_')[2] gives 'Aotizhongxin'
    station = os.path.basename(filepath).split('_')[2]
    df['station'] = station
    dfs.append(df)

# pd.concat stacks all DataFrames vertically into one big table
# ignore_index=True resets row numbers from 0 to N
df_all = pd.concat(dfs, ignore_index=True)
# Select one station to start — we'll generalize later
STATION = 'Tiantan'

df = df_all[df_all['station'] == STATION].copy()

# Create datetime index
# pd.to_datetime converts a dict of year/month/day/hour columns into a real datetime
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df = df.sort_values('datetime').reset_index(drop=True)
df = df.set_index('datetime')  # make datetime the index (like a label for each row)

# Keep only the columns we need for modeling
FEATURES = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
            'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

df = df[FEATURES].copy()

print(f"Working dataset shape: {df.shape}")
print(f"Date range: {df.index[0]} → {df.index[-1]}")
print(f"Missing values:\n{df.isnull().sum()}")

# ── Step 2.2: Fill missing values ─────────────────────────────────────────────

df_clean = df.copy()

# --- Strategy 1: Linear Interpolation ---
# For each missing value, estimate it based on neighbors
# limit=6 means: only interpolate gaps of 6 or fewer consecutive missing hours
# If the gap is longer than 6 hours, leave it as NaN for now

df_clean = df_clean.interpolate(
    method='linear',
    limit=6,          # don't interpolate gaps longer than 6 hours
    limit_direction='forward'
)

# --- Strategy 2: Forward fill then backward fill for remaining NaN ---
# ffill: copy the last valid value forward
# bfill: copy the next valid value backward
# This handles edge cases (gaps at the start or end of the dataset)

df_clean = df_clean.ffill().bfill()

# Verify no missing values remain
remaining_missing = df_clean.isnull().sum().sum()
print(f"Total missing values remaining: {remaining_missing}")

if remaining_missing == 0:
    print("✅ All missing values handled")
else:
    print("⚠️ Some missing values remain — investigate further")


# ── Step 2.3: Remove outliers ──────────────────────────────────────────────────

def cap_outliers_iqr(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """
    Replace outliers with NaN, then interpolate.
    
    We 'cap' rather than delete — removing rows would break the time sequence.
    """
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Count outliers before replacing
    n_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    
    # Replace outliers with NaN
    # .where keeps values that satisfy the condition, replaces others with NaN
    cleaned = series.where(
        (series >= lower_bound) & (series <= upper_bound),
        other=np.nan
    )
    
    # Interpolate to fill the NaN we just created
    cleaned = cleaned.interpolate(method='linear')
    
    return cleaned, n_outliers


# Apply to all columns
print("=== Outlier Removal Summary ===")
for col in FEATURES:
    df_clean[col], n = cap_outliers_iqr(df_clean[col])
    if n > 0:
        print(f"  {col:<12}: {n} outliers replaced")

print("\n✅ Outlier removal complete")

print(df_all[df_all['station'] == STATION].head())

# ── 6. Visualize before vs after cleaning ────────────────────
# Always verify your cleaning worked correctly

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Before (using original df)
axes[0].plot(df.index, df['PM2.5'],
             color='tomato', alpha=0.6, linewidth=0.5)
axes[0].set_title('PM2.5 — BEFORE Cleaning')
axes[0].set_ylabel('PM2.5 (μg/m³)')

# After (using df_clean)
axes[1].plot(df_clean.index, df_clean['PM2.5'],
             color='steelblue', alpha=0.6, linewidth=0.5)
axes[1].set_title('PM2.5 — AFTER Cleaning')
axes[1].set_ylabel('PM2.5 (μg/m³)')

plt.tight_layout()
plt.savefig('outputs/charts/cleaning_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: cleaning_comparison.png")

# ── 7. Save the clean data ────────────────────────────────────
df_clean.to_csv('outputs/data_clean.csv')
print(f"\nClean data saved to: data_clean.csv")
print(f"Final shape: {df_clean.shape}")
print(f"Missing values: {df_clean.isnull().sum().sum()}")
