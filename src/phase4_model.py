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