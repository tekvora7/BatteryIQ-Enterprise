# 🔋 BatteryIQ Pro — Battery RUL Prediction System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-ee4c2c)

An end-to-end **Predictive Maintenance ML System** that estimates the Remaining Useful Life (RUL) of Li-ion battery cells using the NASA ARC Battery Dataset.

---

## 🚀 Live Dashboard

> Run locally with `streamlit run src/app.py`

---

## 📸 Dashboard Preview

| Tab | Description |
|-----|-------------|
| 📉 Degradation Analysis | Capacity fade curves for all 4 batteries |
| 📊 Model Performance | MAE/RMSE comparison across models |
| 🔮 Live RUL Prediction | Real-time RUL prediction with sliders |
| 🔍 Feature Analysis | Feature importance + correlation heatmap |
| 🚗 Fleet Comparison | Side-by-side battery health comparison |

---

## 📊 Dataset

- **Source:** NASA Prognostics Center of Excellence (PCoE)
- **Batteries:** B0005, B0006, B0007, B0018
- **Data:** Li-Ion battery charge/discharge cycles at various temperatures
- **Download:** [NASA Battery Dataset](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip)

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **ML Models:** Random Forest, XGBoost, LSTM (PyTorch)
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Data Parsing:** SciPy (.mat files)

---

## 🧠 ML Pipeline
[Raw .mat Files] → [Feature Engineering] → [Model Training] → [RUL Prediction]

(V, I, Temp)        (SOH, Capacity Fade,    (RF / XGBoost /    (Cycles Remaining)

RollMean, FadeRate)      LSTM)

---

## 📈 Model Results

| Model | MAE | RMSE |
|-------|-----|------|
| 🌲 Random Forest | 7.43 | 9.44 |
| ⚡ XGBoost | 7.93 | 9.68 |
| 🧠 LSTM (PyTorch) | ~10.38 | ~14.24 |

✅ **Best Model: Random Forest** with RMSE of 9.44

---

## ⚙️ Feature Engineering

| Feature | Description |
|---------|-------------|
| Capacity | Discharge capacity per cycle |
| Capacity Fade | Degradation from initial capacity |
| SOH | State of Health (0 to 1) |
| RUL | Remaining cycles until 80% capacity threshold |
| Capacity RollMean | Rolling average over 5 cycles |
| Fade Rate | Rate of capacity loss per cycle |

---

## 📁 Project Structure
BatteryIQ-Enterprise/

├── data/               # NASA battery .mat files

├── models/             # Saved .pkl models

├── notebooks/          # Jupyter notebooks

├── src/

│   └── app.py          # Streamlit dashboard

├── results/            # CSV outputs

└── README.md

---

## 🏃 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost streamlit scipy matplotlib seaborn torch

# Run dashboard
streamlit run src/app.py
```

---

## 📝 Resume Description

> **Battery Remaining Useful Life (RUL) Prediction System** | Python, Scikit-Learn, XGBoost, PyTorch, Streamlit
> - Built an end-to-end predictive maintenance ML pipeline using NASA Li-ion Battery Dataset (4 batteries, 636 cycles)
> - Engineered time-series features including SOH, Capacity Fade, Rolling Mean, and Fade Rate
> - Trained and compared Random Forest, XGBoost, and LSTM models — Random Forest achieved best RMSE of 9.44
> - Built an interactive dark-themed Streamlit dashboard (BatteryIQ Pro) with 5 tabs for degradation analysis, live RUL prediction, and fleet comparison

---

## 🙏 Citation

B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA