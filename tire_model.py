import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import pickle, os

COMPOUND_DEGRADATION = {"SOFT": 0.18, "MEDIUM": 0.10, "HARD": 0.055, "INTERMEDIATE": 0.08, "WET": 0.04}
COMPOUND_BASE_OFFSET = {"SOFT": -1.2, "MEDIUM": 0.0, "HARD": 1.3, "INTERMEDIATE": 8.0, "WET": 15.0}


class AdvancedRacePaceModel:
    """
    Gradient Boosted model trained on multi-year, multi-track F1 data.

    Features used:
      - LapNumber          (fuel-corrected race position proxy)
      - TyreLife           (laps on current set)
      - Compound_enc       (ordinal: SOFT=0, MEDIUM=1, HARD=2, …)
      - FuelLoad           (estimated kg remaining, 110 kg at lap 1 → burns ~1.6 kg/lap)
      - TrackTemp          (°C — filled with per-track mean when not available in historical data)
      - Track_*            (one-hot encoded circuit columns)

    The model is serialised to `f1_brain.pkl` so the dashboard can hot-load
    without re-training on every restart.
    """

    FUEL_START_KG = 110.0
    FUEL_BURN_PER_LAP = 1.6          # average kg/lap
    FUEL_TIME_EFFECT = 0.035         # seconds per kg of fuel

    COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

    def __init__(self, model_path: str = "f1_brain.pkl"):
        self.model_path = model_path
        self.model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.07, max_depth=5,
            subsample=0.8, random_state=42
        )
        self.feature_columns: list = []
        self.track_mean_laptime: dict = {}  # fallback per-track median
        self.is_trained = False

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _compound_enc(self, compound: str) -> int:
        c = compound.upper()
        return self.COMPOUND_ORDER.index(c) if c in self.COMPOUND_ORDER else 1

    def _fuel_load(self, lap_number: int) -> float:
        return max(0.0, self.FUEL_START_KG - self.FUEL_BURN_PER_LAP * (lap_number - 1))

    def _build_row(self, track: str, lap_number: int, tire_age: int,
                   compound: str = "MEDIUM", track_temp: float = 35.0) -> dict:
        return {
            "LapNumber":   lap_number,
            "TyreLife":    tire_age,
            "Compound_enc": self._compound_enc(compound),
            "FuelLoad":    self._fuel_load(lap_number),
            "TrackTemp":   track_temp,
            "Track":       track,
        }

    def _encode(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        enc = pd.get_dummies(df_raw, columns=["Track"], drop_first=False)
        for col in self.feature_columns:
            if col not in enc.columns:
                enc[col] = 0
        return enc[self.feature_columns]

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train_global(self, df: pd.DataFrame):
        """
        Expects columns: Year, Track, Driver, LapNumber, TyreLife, Compound, LapTimeSeconds
        Optional:        TrackTemp
        """
        df = df.copy()

        # Estimated fuel load
        df["FuelLoad"] = df["LapNumber"].apply(self._fuel_load)

        # Compound encoding
        df["Compound_enc"] = df["Compound"].apply(self._compound_enc)

        # Track temp — use column if present, else fill per-track mean (35 °C default)
        if "TrackTemp" not in df.columns:
            df["TrackTemp"] = 35.0

        # Per-track median lap time (used as fallback in predict)
        self.track_mean_laptime = df.groupby("Track")["LapTimeSeconds"].median().to_dict()

        X_raw = df[["LapNumber", "TyreLife", "Compound_enc", "FuelLoad", "TrackTemp", "Track"]]
        y = df["LapTimeSeconds"]

        X_enc = pd.get_dummies(X_raw, columns=["Track"], drop_first=False)
        self.feature_columns = X_enc.columns.tolist()

        print(f"Training GBM on {len(df):,} laps across "
              f"{df['Track'].nunique()} circuits ({df['Year'].nunique()} seasons)…")
        self.model.fit(X_enc, y)
        self.is_trained = True
        print("Model trained successfully.")
        self.save()

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_columns": self.feature_columns,
                "track_mean_laptime": self.track_mean_laptime,
            }, f)
        print(f"Model saved → {self.model_path}")

    def load(self) -> bool:
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.feature_columns = data["feature_columns"]
                self.track_mean_laptime = data.get("track_mean_laptime", {})
                self.is_trained = True
                print(f"Model loaded from {self.model_path}")
                return True
            except Exception as e:
                print(f"Could not load cached model ({e}); retraining from scratch.")
                return False
        return False

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict_lap_time(self, track: str, lap_number: int, tire_age: int,
                         compound: str = "MEDIUM", track_temp: float = 35.0) -> float:
        if not self.is_trained:
            # Graceful fallback before training
            base = self.track_mean_laptime.get(track, 92.0)
            return base + COMPOUND_BASE_OFFSET.get(compound.upper(), 0) + tire_age * COMPOUND_DEGRADATION.get(compound.upper(), 0.10)

        row = self._build_row(track, lap_number, tire_age, compound, track_temp)
        df_raw = pd.DataFrame([row])
        X = self._encode(df_raw)
        return float(self.model.predict(X)[0])

    # Legacy shim for simulation_engine.py (single-track mode, no compound/temp args)
    def predict_lap_time_simple(self, lap_number: int, tire_age: int,
                                track: str = "Bahrain", compound: str = "MEDIUM") -> float:
        return self.predict_lap_time(track, lap_number, tire_age, compound)
