# ============================================================
# PHASE 7 — Full End-to-End Pipeline (Recursive + Fuzzy)
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ── 1. Setup Fuzzy Logic System ───────────────────────────────
pm25_var = ctrl.Antecedent(np.arange(0, 501, 1), 'pm25')
risk_var = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

pm25_var['good']           = fuzz.trimf(pm25_var.universe, [0,   0,   35])
pm25_var['moderate']       = fuzz.trimf(pm25_var.universe, [15,  50,  75])
pm25_var['unhealthy_sens'] = fuzz.trimf(pm25_var.universe, [55,  115, 150])
pm25_var['unhealthy']      = fuzz.trimf(pm25_var.universe, [115, 200, 250])
pm25_var['very_hazardous'] = fuzz.trimf(pm25_var.universe, [200, 350, 500])

risk_var['low']       = fuzz.trimf(risk_var.universe, [0,   0,   30])
risk_var['medium']    = fuzz.trimf(risk_var.universe, [20,  40,  60])
risk_var['high']      = fuzz.trimf(risk_var.universe, [50,  70,  85])
risk_var['critical']  = fuzz.trimf(risk_var.universe, [75,  90,  100])
risk_var['emergency'] = fuzz.trimf(risk_var.universe, [90,  100, 100])

rule1 = ctrl.Rule(pm25_var['good'],           risk_var['low'])
rule2 = ctrl.Rule(pm25_var['moderate'],       risk_var['medium'])
rule3 = ctrl.Rule(pm25_var['unhealthy_sens'], risk_var['high'])
rule4 = ctrl.Rule(pm25_var['unhealthy'],      risk_var['critical'])
rule5 = ctrl.Rule(pm25_var['very_hazardous'], risk_var['emergency'])

air_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
air_sim  = ctrl.ControlSystemSimulation(air_ctrl)

def translate_pm25(pm25_value: float) -> dict:
    pm25_value = float(np.clip(pm25_value, 0.0, 500.0))
    air_sim.input['pm25'] = pm25_value
    air_sim.compute()
    risk_score = air_sim.output['risk']

    if pm25_value <= 35:
        return {'pm25': round(pm25_value, 1), 'risk_score': round(risk_score, 1), 'category': 'Good', 'advice': 'Air quality is excellent.', 'mask': False}
    elif pm25_value <= 75:
        return {'pm25': round(pm25_value, 1), 'risk_score': round(risk_score, 1), 'category': 'Moderate', 'advice': 'Acceptable. Sensitive individuals take note.', 'mask': False}
    elif pm25_value <= 150:
        return {'pm25': round(pm25_value, 1), 'risk_score': round(risk_score, 1), 'category': 'Unhealthy for Sensitive', 'advice': 'Sensitive groups face risks.', 'mask': True}
    elif pm25_value <= 250:
        return {'pm25': round(pm25_value, 1), 'risk_score': round(risk_score, 1), 'category': 'Unhealthy', 'advice': 'Serious health effects for all.', 'mask': True}
    else:
        return {'pm25': round(pm25_value, 1), 'risk_score': round(risk_score, 1), 'category': 'Very Hazardous', 'advice': 'Health emergency!', 'mask': True}

# ── 2. Load Model & Artifacts ─────────────────────────────────
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

device = torch.device('cpu')
scaler_X = joblib.load('saved_models/scaler_X.pkl')
scaler_y = joblib.load('saved_models/scaler_y.pkl')

with open('saved_models/Charts/feature_cols.json', 'r') as f:
    FEATURE_COLS = json.load(f)

INPUT_SIZE = len(FEATURE_COLS)
model = AirQualityLSTM(INPUT_SIZE, 32, 1, 1, 0.3).to(device)
model.load_state_dict(torch.load('saved_models/best_model.pth', map_location=device))
model.eval()

# ── 3. The Recursive Pipeline Function ───────────────────────

def forecast_air_quality(raw_data_with_future: np.ndarray) -> dict:
    """
    raw_data_with_future: shape (48 + 24, n_features)
    First 48 rows: Past history.
    Next 24 rows: Future features (like hour, temp) with dummy PM2.5 to be overwritten.
    """
    HORIZON = 24
    pm25_idx = FEATURE_COLS.index('PM2.5')
    
    # Scale entire input
    scaled_data = scaler_X.transform(raw_data_with_future)
    current_window = scaled_data[:48, :].copy()
    
    pred_scaled_list = []
    
    with torch.no_grad():
        for h in range(HORIZON):
            tensor_in = torch.FloatTensor(current_window).unsqueeze(0).to(device)
            pred_scaled = model(tensor_in).numpy()[0, 0]
            pred_scaled_list.append([pred_scaled])
            
            # Recursive update
            if h < HORIZON - 1:
                next_features = scaled_data[48 + h].copy()
                next_features[pm25_idx] = pred_scaled  # Inject prediction
                current_window = np.vstack((current_window[1:], next_features))

    # Inverse transform predictions
    pred_pm25 = scaler_y.inverse_transform(pred_scaled_list).flatten()
    
    # Apply fuzzy logic
    hourly_forecast = []
    for idx, pm25_val in enumerate(pred_pm25):
        advisory = translate_pm25(pm25_val)
        advisory['forecast_hour'] = idx + 1
        hourly_forecast.append(advisory)

    # Summary
    pm25_vals = [h['pm25'] for h in hourly_forecast]
    worst_hour = hourly_forecast[np.argmax(pm25_vals)]
    
    summary = {
        'avg_pm25': round(np.mean(pm25_vals), 1),
        'max_pm25': max(pm25_vals),
        'worst_hour': worst_hour['forecast_hour'],
        'worst_category': worst_hour['category'],
        'mask_recommended': any(h['mask'] for h in hourly_forecast)
    }
    
    return {'hourly_forecast': hourly_forecast, 'summary': summary}

# ── 4. Demo Run ──────────────────────────────────────────────
# Load raw data to test
df = pd.read_csv('outputs/data_features.csv', index_col='datetime', parse_dates=True)
# Take last 72 hours (48 past + 24 future)
raw_input = df[FEATURE_COLS].values[-72:] 

result = forecast_air_quality(raw_input)

print("\n" + "=" * 65)
print("  24-HOUR AIR QUALITY FORECAST")
print("=" * 65)
for h in result['hourly_forecast']:
    print(f" +{h['forecast_hour']:2d}h | PM2.5: {h['pm25']:5.1f} | Risk: {h['risk_score']:4.1f} | {h['category']}")

print("=" * 65)
s = result['summary']
print(f"SUMMARY: Avg PM2.5 = {s['avg_pm25']} | Worst = {s['max_pm25']} at hour +{s['worst_hour']} | Mask? {'YES' if s['mask_recommended'] else 'NO'}")