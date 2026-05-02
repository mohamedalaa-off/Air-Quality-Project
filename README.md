# 🌬️ AirQ AI: Intelligent Air Quality & Health Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![scikit-fuzzy](https://img.shields.io/badge/scikit--fuzzy-Logic-orange)

An end-to-end Machine Learning pipeline that predicts PM2.5 air pollution levels for the next 24 hours and translates raw predictions into actionable, human-readable health advisories.

## 🚀 Project Overview
Air quality directly impacts public health, but raw sensor data (e.g., "PM2.5 is 85 μg/m³") is often meaningless to the average person. **AirQ AI** bridges this gap by:
1. Forecasting PM2.5 levels recursively for the next 24 hours using a specialized **LSTM Neural Network**.
2. Translating those predictions into specific risk scores and health advice using a **Fuzzy Logic Inference System**.
3. Serving the entire pipeline via a blazing-fast **FastAPI backend** and a modern, interactive **Web Dashboard**.

---

## 🧠 Architecture & Methodology

### 1. Feature Engineering
- **Cyclical Time Encoding:** Converted hours and days into `sine` and `cosine` waves to help the model understand the circular nature of time.
- **Rolling Volatility:** Engineered rolling standard deviations (e.g., 6h window) to capture environmental instability without forcing the LSTM to overfit on direct lags.

### 2. Recursive LSTM Forecasting
Unlike traditional direct-multi-step models that suffer from degradation, this project uses a **Recursive (Autoregressive) approach**:
- The model (`HORIZON = 1`) predicts the very next hour.
- This prediction is injected back into the feature sequence as a "known" value to predict the subsequent hour, repeated 24 times.
- **Result:** Outperformed the persistence baseline by **> 57%** and achieved an **R² score of 0.88**.

### 3. Fuzzy Logic Translation Layer
A rules-based inference system (`skfuzzy`) acts as the bridge between AI and humans. It maps raw PM2.5 forecasts to a continuous Risk Score (0-100) and categorizes the threat level (Good, Moderate, Unhealthy, Hazardous), outputting specific advice like "N95 Mask Recommended."

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch (`nn.LSTM`)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Expert System:** Scikit-Fuzzy
* **API Backend:** FastAPI, Uvicorn
* **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JS, Chart.js, PapaParse

---

## 📊 Performance Metrics (Test Set)
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **MAE** | ~ 18.92 μg/m³ | High precision for a 24-hour window |
| **R² Score** | 0.88 | Model explains 88% of PM2.5 variance |
| **Baseline Improvement** | +57.2% | Beats the persistence baseline significantly |
| **Threshold Accuracy** | ~ 59.4% | Forecasts within a strict ±15 μg/m³ margin |

---

## 💻 How to Run the Project

### 1. Clone the Repository
```bash
git clone [https://github.com/mohamedalaa-off/Air-Quality-Project.git](https://github.com/mohamedalaa-off/Air-Quality-Project.git)
cd Air-Quality-Project

## 2. Install Dependencies
