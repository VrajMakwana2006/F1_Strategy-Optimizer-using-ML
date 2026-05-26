# 🏁 F1 Pit Wall Pro — AI Strategy Engine

> A machine learning powered Formula 1 race strategy optimizer that predicts the optimal pit stop window and tyre compound selection in real time, based on live race inputs and historical multi-year F1 telemetry data.
---
## 🚀 Live Demo

**[👉 Open F1 Pit Wall Pro](https://distributedsystemsimulation-z5x7rghxwutjkwxocjz2gq.streamlit.app/)**

---

## 🧠 What It Does

F1 Pit Wall Pro simulates the role of a real-world pit wall strategist. You feed it the current race situation — lap number, tyre compound, tyre age, track temperature, flag status, weather, and engine mode — and the AI engine computes the mathematically optimal pit stop strategy for the remainder of the race.

The ML model at its core has been trained on **3 years of F1 race data (2022–2024)** across **10 circuits**, learning the nuanced patterns of tyre degradation, fuel load effects, and lap time evolution that define real-world race strategy.

---

## ✨ Features

- **Live Pit Wall** — Input real-time race telemetry and get instant strategy recommendations
- **Optimal Pit Window** — Dynamic programming engine computes the lowest total race time path
- **Tyre Compound Advice** — Recommends SOFT / MEDIUM / HARD based on remaining laps and conditions
- **Safety Car & VSC Awareness** — Strategy adjusts automatically for green, VSC, SC, and red flag conditions
- **Pace Projection Chart** — Visualizes predicted lap times for the rest of the race
- **Tyre Degradation Analysis** — Per-compound degradation rate charts per circuit
- **Stint Map** — Gantt-style visualization of planned tyre stints
- **Strategy Comparison** — A vs B head-to-head race time comparison with delta chart
- **F1 Archives & Compendium** — Interactive glossary of F1 technical terms and historical timeline

---

## 🤖 How the ML Model Works

The brain of the app is an `AdvancedRacePaceModel` — a **Gradient Boosting Regressor** (scikit-learn) trained on clean lap data fetched via the **FastF1 API**.

### Features used for prediction:
| Feature | Description |
|---|---|
| `LapNumber` | Race lap — proxy for fuel load and track rubber |
| `TyreLife` | Laps on current set of tyres |
| `Compound_enc` | Ordinal encoding: SOFT=0, MEDIUM=1, HARD=2 |
| `FuelLoad` | Estimated fuel remaining (110kg at lap 1, burns ~1.6kg/lap) |
| `TrackTemp` | Track surface temperature in °C |
| `Track_*` | One-hot encoded circuit columns (10 circuits) |

### Training data:
- **Seasons:** 2022, 2023, 2024
- **Circuits:** Bahrain, Saudi Arabia, Australia, Monaco, Silverstone, Monza, Spa, Suzuka, Singapore, Abu Dhabi
- **Filtering:** DNFs removed, pit-in/pit-out laps stripped, outliers clipped at 1st–97th percentile
- **Model:** GradientBoostingRegressor — 400 estimators, learning rate 0.07, max depth 5

The trained model is serialized to `f1_brain.pkl` so the app loads instantly without retraining on every startup.

### Strategy optimization:
The `UniversalStrategyEngine` uses **dynamic programming** to evaluate every possible pit stop path from the current lap to the end of the race, returning the sequence with the lowest predicted total race time. It supports 1-stop, 2-stop, and 3-stop strategies and accounts for pit lane time loss, safety car windows, and engine mode deltas.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| ML Model | scikit-learn — Gradient Boosting Regressor |
| Data Source | FastF1 API (official F1 telemetry) |
| Strategy Engine | Custom dynamic programming optimizer |
| Data Processing | pandas, NumPy |
| Visualizations | Plotly |
| Model Persistence | pickle |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
F1_Strategy-Optimizer-using-ML/
│
├── dashboard.py          # Streamlit UI — main entry point
├── tire_model.py         # GBM ML model — training, saving, prediction
├── realtime_engine.py    # Dynamic programming strategy optimizer
├── simulation_engine.py  # Lap-by-lap race simulator
├── strategy_optimizer.py # Brute-force 1-stop / 2-stop sweep optimizer
├── data_ingestion.py     # FastF1 API data fetching and cleaning
├── data_processing.py    # Lap data processor and stint extractor
├── train_local.py        # Offline training script (run locally)
├── f1_brain.pkl          # Pre-trained serialized model
└── requirements.txt      # Python dependencies
```

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/VrajMakwana2006/F1_Strategy-Optimizer-using-ML.git
cd F1_Strategy-Optimizer-using-ML
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux / WSL
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```


### 4. Run the app
```bash
streamlit run dashboard.py
```

The app will open at `http://localhost:8501`

---

## 📊 Supported Circuits

| Circuit | Country | Race Laps |
|---|---|---|
| Bahrain | 🇧🇭 | 57 |
| Saudi Arabia | 🇸🇦 | 50 |
| Australia | 🇦🇺 | 58 |
| Monaco | 🇲🇨 | 78 |
| Silverstone | 🇬🇧 | 52 |
| Monza | 🇮🇹 | 53 |
| Spa | 🇧🇪 | 44 |
| Suzuka | 🇯🇵 | 53 |
| Singapore | 🇸🇬 | 62 |
| Abu Dhabi | 🇦🇪 | 58 |

---

## ⚠️ Disclaimer

F1 Pit Wall Pro is an independent fan project built for educational and analytical purposes. It is not affiliated with, endorsed by, or connected to the FIA, Formula One Group, or any F1 team. All data is sourced from the publicly available FastF1 API.

---

## 👤 Author

**Vraj Makwana**
[GitHub](https://github.com/VrajMakwana2006)
