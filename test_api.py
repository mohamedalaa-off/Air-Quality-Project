import pandas as pd
import requests
import json

print("Preparing data to send to API...")

# 1. Load the feature columns to ensure correct order
with open('saved_models/Charts/feature_cols.json', 'r') as f:
    FEATURE_COLS = json.load(f)

# 2. Load the actual data
df = pd.read_csv('outputs/data_features.csv', index_col='datetime', parse_dates=True)

# 3. Get the last 72 hours (48 history + 24 future)
# Convert to a list of lists so it can be sent as JSON
test_data = df[FEATURE_COLS].values[-72:].tolist()

# 4. Prepare the request payload
payload = {
    "sensor_readings": test_data
}

# 5. Send the POST request to our FastAPI server
url = "http://127.0.0.1:8000/forecast"
print(f"Sending request to {url}...\n")

response = requests.post(url, json=payload)

# 6. Check the response
if response.status_code == 200:
    result = response.json()
    
    print("=" * 50)
    print(" API FORECAST RESPONSE RECEIVED SUCCESSFULLY! ")
    print("=" * 50)
    
    # Print a few hours to check
    print("First 3 Hours Forecast:")
    for h in result['hourly_forecast'][:3]:
        print(f" Hour +{h['forecast_hour']}: PM2.5 = {h['pm25']} -> {h['category']}")
        
    print("\nSUMMARY:")
    summary = result['summary']
    for key, value in summary.items():
        print(f" - {key}: {value}")
        
else:
    print(f"Error {response.status_code}: {response.text}")