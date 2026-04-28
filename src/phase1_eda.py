# ============================================================
# PHASE 1 — Exploratory Data Analysis
# ============================================================
# GOAL: Understand the dataset before touching it.
# We look at shape, missing values, distributions, and patterns.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ── 1. Load all 12 station files ─────────────────────────────
# glob.glob finds every file matching a pattern
# The * wildcard matches any filename in that folder

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

print(f"\nCombined shape   : {df_all.shape}")
print(f"Columns          : {list(df_all.columns)}")

# ── 2. First look ─────────────────────────────────────────────
print("\n=== First 5 rows ===")
print(df_all.head())

print("\n=== Data types & non-null counts ===")
print(df_all.info())

print("\n=== Statistical summary ===")
print(df_all.describe().round(2))

# ── 3. Missing value analysis ─────────────────────────────────
# .isnull() returns True/False for each cell
# .sum() counts the Trues (= missing values)
# We divide by total rows and multiply by 100 to get percentage

missing_count = df_all.isnull().sum()
missing_pct   = (missing_count / len(df_all) * 100).round(2)

missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing %'    : missing_pct
}).sort_values('Missing %', ascending=False)

# Show only columns that actually have missing values
missing_df = missing_df[missing_df['Missing Count'] > 0]

print("\n=== Missing Values ===")
print(missing_df)

# Missing per station — some stations are cleaner than others
print("\n=== Missing PM2.5 per Station ===")
missing_by_station = (
    df_all.groupby('station')['PM2.5']
    .apply(lambda x: x.isnull().sum())
    .reset_index()
)
missing_by_station.columns = ['station', 'missing_pm25']
missing_by_station['missing_%'] = (
    missing_by_station['missing_pm25'] /
    df_all.groupby('station').size().values * 100
).round(2)
print(missing_by_station.sort_values('missing_%'))

# ── 4. Build a datetime column ────────────────────────────────
# The dataset has separate year, month, day, hour columns
# pd.to_datetime merges them into a single proper datetime object
# This lets us sort, filter, and plot by time correctly

df_all['datetime'] = pd.to_datetime(df_all[['year', 'month', 'day', 'hour']])

print(f"\nDate range: {df_all['datetime'].min()} → {df_all['datetime'].max()}")

# ── 5. Work with one station for deep analysis ────────────────
# We use Tiantan — central Beijing, usually has clean data
# We will generalize to all stations later

STATION = 'Tiantan'
df_station = df_all[df_all['station'] == STATION].copy()
df_station = df_station.sort_values('datetime').reset_index(drop=True)
df_station = df_station.set_index('datetime')

print(f"\n{STATION} station shape: {df_station.shape}")
print(f"Missing PM2.5 in {STATION}: {df_station['PM2.5'].isnull().sum()}")

# ── 6. Visualization ──────────────────────────────────────────
# Plot 1: PM2.5 distribution + time series

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'EDA — {STATION} Station', fontsize=15, fontweight='bold')

# ---- Plot A: Histogram of PM2.5 ----
# Shows us the shape of the distribution
# Most values should cluster low, with a long tail to the right
axes[0, 0].hist(
    df_station['PM2.5'].dropna(),
    bins=80,
    color='steelblue',
    edgecolor='white',
    alpha=0.85
)
axes[0, 0].axvline(x=75,  color='orange', linestyle='--',
                   linewidth=1.5, label='Unhealthy threshold (75)')
axes[0, 0].axvline(x=150, color='red',    linestyle='--',
                   linewidth=1.5, label='Hazardous threshold (150)')
axes[0, 0].set_title('PM2.5 Distribution')
axes[0, 0].set_xlabel('PM2.5 (μg/m³)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend(fontsize=9)

# ---- Plot B: PM2.5 over time ----
# Shows seasonal patterns — winter should be much worse
axes[0, 1].plot(
    df_station.index,
    df_station['PM2.5'],
    color='steelblue',
    alpha=0.5,
    linewidth=0.4
)
axes[0, 1].set_title('PM2.5 Over Time (2013–2017)')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('PM2.5 (μg/m³)')
axes[0, 1].axhline(y=75, color='orange', linestyle='--',
                   linewidth=1, label='Unhealthy (75)')
axes[0, 1].legend(fontsize=9)

# ---- Plot C: Average PM2.5 by hour of day ----
# Shows daily cycle — do mornings or evenings have more pollution?
hourly_avg = df_station.groupby(df_station.index.hour)['PM2.5'].mean()
axes[1, 0].bar(
    hourly_avg.index,
    hourly_avg.values,
    color='mediumpurple',
    alpha=0.85,
    edgecolor='white'
)
axes[1, 0].set_title('Average PM2.5 by Hour of Day')
axes[1, 0].set_xlabel('Hour (0 = midnight)')
axes[1, 0].set_ylabel('Average PM2.5 (μg/m³)')
axes[1, 0].set_xticks(range(0, 24, 2))

# ---- Plot D: Average PM2.5 by month ----
# Shows seasonal patterns — winter months (Nov-Feb) burn more coal
monthly_avg = df_station.groupby(df_station.index.month)['PM2.5'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
colors = ['#d73027' if m in [11,12,1,2] else 'steelblue'
          for m in monthly_avg.index]
axes[1, 1].bar(
    monthly_avg.index,
    monthly_avg.values,
    color=colors,
    alpha=0.85,
    edgecolor='white'
)
axes[1, 1].set_title('Average PM2.5 by Month\n(red = winter heating season)')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Average PM2.5 (μg/m³)')
axes[1, 1].set_xticks(range(1, 13))
axes[1, 1].set_xticklabels(month_names, rotation=45)

plt.tight_layout()
plt.savefig('outputs/charts/eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: eda_overview.png")

# Plot 2: Correlation heatmap
# Correlation tells us: when PM2.5 goes up, does this other variable go up or down?
# +1 = perfect positive relationship
# -1 = perfect negative relationship (opposites)
#  0 = no relationship

numeric_cols = ['PM2.5','PM10','SO2','NO2','CO','O3',
                'TEMP','PRES','DEWP','RAIN','WSPM']

corr = df_station[numeric_cols].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(
    corr,
    annot=True,      # write the number in each cell
    fmt='.2f',       # 2 decimal places
    cmap='RdYlGn',   # red=negative, yellow=zero, green=positive
    center=0,        # center the colors at 0
    square=True,
    linewidths=0.5,
    annot_kws={'size': 9}
)
plt.title(f'Correlation Matrix — {STATION}', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/charts/eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: eda_correlation.png")

# Print the top correlations with PM2.5 specifically
print("\n=== Correlations with PM2.5 (sorted) ===")
pm25_corr = corr['PM2.5'].drop('PM2.5').sort_values(ascending=False)
print(pm25_corr.round(3))

# ── 7. Summary statistics — what to document ─────────────────
print("\n" + "=" * 50)
print("DOCUMENT THESE FINDINGS IN YOUR DOCUMENTATION")
print("=" * 50)
print(f"Total rows                  : {len(df_all):,}")
print(f"Total stations              : {df_all['station'].nunique()}")
print(f"Date range                  : {df_all['datetime'].min().date()} → {df_all['datetime'].max().date()}")
print(f"PM2.5 mean (all stations)   : {df_all['PM2.5'].mean():.1f} μg/m³")
print(f"PM2.5 max (all stations)    : {df_all['PM2.5'].max():.1f} μg/m³")
print(f"PM2.5 missing (all)         : {df_all['PM2.5'].isnull().sum():,} ({df_all['PM2.5'].isnull().mean()*100:.1f}%)")
print(f"Worst month (avg PM2.5)     : {month_names[monthly_avg.idxmax()-1]} ({monthly_avg.max():.1f} μg/m³)")
print(f"Best month  (avg PM2.5)     : {month_names[monthly_avg.idxmin()-1]} ({monthly_avg.min():.1f} μg/m³)")
print(f"Strongest + correlation     : {pm25_corr.idxmax()} ({pm25_corr.max():.3f})")
print(f"Strongest - correlation     : {pm25_corr.idxmin()} ({pm25_corr.min():.3f})")