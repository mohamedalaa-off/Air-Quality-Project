# ============================================================
# PHASE 8 — FastAPI Deployment (Recursive & Fuzzy Logic)
# ============================================================
# Run with: uvicorn app:app --reload
# Then open: http://127.0.0.1:8000/docs
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from typing import List
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# ── 1. Lifespan (Startup & Shutdown) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # الجزء ده بيشتغل أول ما السيرفر يقوم (Startup)
    global model, scaler_X, scaler_y, FEATURE_COLS, pm25_idx

    with open('saved_models/Charts/feature_cols.json', 'r') as f:
        FEATURE_COLS = json.load(f)
    
    pm25_idx = FEATURE_COLS.index('PM2.5')
    scaler_X = joblib.load('saved_models/scaler_X.pkl')
    scaler_y = joblib.load('saved_models/scaler_y.pkl')

    INPUT_SIZE  = len(FEATURE_COLS)
    HIDDEN_SIZE = 32
    NUM_LAYERS  = 1
    OUTPUT_SIZE = 1

    device = torch.device('cpu')
    model = AirQualityLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, dropout=0.3).to(device)
    model.load_state_dict(torch.load('saved_models/best_model.pth', map_location=device))
    model.eval()
    print("All artifacts loaded successfully.")
    
    yield  # هنا السيرفر بيفضل شغال وبيستقبل الطلبات
    
    # الجزء ده بيشتغل لما تقفل السيرفر (Shutdown)
    print("Shutting down API...")

# ── 2. App setup ─────────────────────────────────────────────────
app = FastAPI(
    title       = "Air Quality Forecasting API",
    description = "Predicts PM2.5 for 24 hours (Recursive) and returns health advisories",
    version     = "2.0",
    lifespan    = lifespan  
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # بيسمح لأي ويب سايت يكلمه
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Model Architecture & Fuzzy System ─────────────────────────
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

# Setup Fuzzy Logic Rules
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


# ── 4. Recursive Pipeline Function ───────────────────────────────
def forecast_air_quality(raw_data_with_future: np.ndarray) -> dict:
    HORIZON = 24
    scaled_data = scaler_X.transform(raw_data_with_future)
    current_window = scaled_data[:48, :].copy()
    
    pred_scaled_list = []
    device = torch.device('cpu')
    
    with torch.no_grad():
        for h in range(HORIZON):
            tensor_in = torch.FloatTensor(current_window).unsqueeze(0).to(device)
            pred_scaled = model(tensor_in).numpy()[0, 0]
            pred_scaled_list.append([pred_scaled])
            
            if h < HORIZON - 1:
                next_features = scaled_data[48 + h].copy()
                next_features[pm25_idx] = pred_scaled
                current_window = np.vstack((current_window[1:], next_features))

    pred_pm25 = scaler_y.inverse_transform(pred_scaled_list).flatten()
    
    hourly_forecast = []
    for idx, pm25_val in enumerate(pred_pm25):
        advisory = translate_pm25(pm25_val)
        advisory['forecast_hour'] = idx + 1
        hourly_forecast.append(advisory)

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


# ── 5. Request and response schemas ──────────────────────────────
class ForecastRequest(BaseModel):
    # 72 rows total (48 past + 24 future weather/time features)
    sensor_readings: List[List[float]]

class HourlyForecast(BaseModel):
    forecast_hour: int
    pm25:          float
    risk_score:    float
    category:      str
    advice:        str
    mask:          bool

class ForecastResponse(BaseModel):
    hourly_forecast: List[HourlyForecast]
    summary:         dict


# ── 6. Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "AirQualityLSTM Recursive v2.0"}

@app.post("/forecast", response_model=ForecastResponse)
def get_forecast(request: ForecastRequest):
    try:
        data = np.array(request.sensor_readings)
        expected_shape = (72, len(FEATURE_COLS)) # 48 history + 24 future
        
        if data.shape != expected_shape:
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape {expected_shape}, got {data.shape}. "
                       f"Need 72 rows (48 past + 24 future) and {len(FEATURE_COLS)} features."
            )

        result = forecast_air_quality(data)
        
        return ForecastResponse(
            hourly_forecast=[HourlyForecast(**h) for h in result['hourly_forecast']],
            summary=result['summary']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Air Quality Forecasting API is running!",
        "docs":    "Visit /docs to test the API"
    }

@app.get("/forecast-latest", response_model=ForecastResponse)
def get_latest_forecast():
    """
    Endpoint مخصص للويب: بيقرأ آخر داتا أوتوماتيك من الملف ويرجع التوقع
    """
    try:
        # بيقرأ الداتا أوتوماتيك من السيرفر
        df = pd.read_csv('outputs/data_features.csv', index_col='datetime', parse_dates=True)
        test_data = df[FEATURE_COLS].values[-72:]
        
        result = forecast_air_quality(test_data)
        
        return ForecastResponse(
            hourly_forecast=[HourlyForecast(**h) for h in result['hourly_forecast']],
            summary=result['summary']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")