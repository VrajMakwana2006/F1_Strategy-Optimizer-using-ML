import numpy as np
from tire_model import AdvancedRacePaceModel, COMPOUND_BASE_OFFSET, COMPOUND_DEGRADATION

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Compound switch restrictions (can't switch to same compound on green flag)
ALLOWED_NEXT = {
    "SOFT":   ["MEDIUM", "HARD"],
    "MEDIUM": ["SOFT", "HARD"],
    "HARD":   ["SOFT", "MEDIUM"],
}


class UniversalStrategyEngine:
    """
    Dynamic-programming race strategy calculator.

    The global_ml_cache is keyed on (track, lap, tire_age, compound)
    for O(1) lookups during slider interactions.
    """

    def __init__(self, global_pace_model: AdvancedRacePaceModel,
                 tracks_list: list, max_laps_map: dict,
                 default_track_temp: float = 35.0):
        self.global_pace_model = global_pace_model
        self.tracks_list = tracks_list
        self.max_laps_map = max_laps_map
        self.default_track_temp = default_track_temp

        # {(track, lap, tire_age, compound): predicted_lap_time}
        self.global_ml_cache: dict = {}
        self._build_global_cache()

    # ------------------------------------------------------------------ #
    #  Cache build                                                         #
    # ------------------------------------------------------------------ #

    def _build_global_cache(self):
        """Pre-computes all (track, lap, age, compound) combinations."""
        total = 0
        for track in self.tracks_list:
            total_laps = self.max_laps_map.get(track, 60)
            for compound in COMPOUNDS:
                for lap in range(1, total_laps + 1):
                    for age in range(1, 55):
                        t = self.global_pace_model.predict_lap_time(
                            track=track, lap_number=lap, tire_age=age,
                            compound=compound, track_temp=self.default_track_temp
                        )
                        self.global_ml_cache[(track, lap, age, compound)] = t
                        total += 1
        print(f"Global cache built: {total:,} predictions cached.")

    # ------------------------------------------------------------------ #
    #  Live pace with real-time modifiers                                  #
    # ------------------------------------------------------------------ #

    def _predict_live_pace(self, track: str, lap: int, tire_age: int,
                           compound: str, track_temp: float, weather: str,
                           engine_mode: str, fuel_adjusted: bool = True) -> float:
        key = (track, lap, tire_age, compound)
        base = self.global_ml_cache.get(key, 95.0)

        # Temperature delta from training default (35 °C)
        temp_delta = (track_temp - 35.0) * 0.008  # ~0.008 s per °C above baseline

        # Weather
        weather_delta = 12.0 if weather == "WET" else 0.0

        # Engine mode
        engine_delta = {"PUSH": -0.6, "BALANCED": 0.0, "SAVE": +0.7}.get(engine_mode, 0.0)

        return base + temp_delta + weather_delta + engine_delta

    # ------------------------------------------------------------------ #
    #  Dynamic programming strategy tree                                   #
    # ------------------------------------------------------------------ #

    def calculate_optimal_path(self, track: str, current_lap: int, tire_age: int,
                               compound: str, flag_status: str, track_temp: float,
                               weather: str, engine_mode: str, base_pit_loss: float,
                               vsc_pit_loss: float, stops_remaining: int = 2,
                               memo: dict = None):
        if memo is None:
            memo = {}

        total_laps = self.max_laps_map.get(track, 57)

        # Base case
        if current_lap > total_laps:
            return 0.0, []

        # Cliff: worn-out tires → infinite time (forces a stop)
        if tire_age > 50:
            return float("inf"), []

        state = (current_lap, tire_age, compound, stops_remaining, flag_status)
        if state in memo:
            return memo[state]

        # ---- Option A: Stay out ----------------------------------------
        stay_lap_time = self._predict_live_pace(
            track, current_lap, tire_age, compound, track_temp, weather, engine_mode
        )
        time_stay, path_stay = self.calculate_optimal_path(
            track, current_lap + 1, tire_age + 1, compound,
            "GREEN", track_temp, weather, engine_mode,
            base_pit_loss, vsc_pit_loss, stops_remaining, memo
        )
        total_stay = stay_lap_time + time_stay

        # ---- Option B: Pit ---------------------------------------------
        best_pit_total = float("inf")
        best_pit_path = []

        if stops_remaining > 0 and current_lap < total_laps:
            pit_loss = vsc_pit_loss if flag_status in ("VSC", "SC") else base_pit_loss
            # Safety-car pit loses less; model it as 0 lap time for that lap
            sc_time_saving = stay_lap_time if flag_status in ("VSC", "SC") else 0.0

            next_compounds = ALLOWED_NEXT.get(compound, COMPOUNDS)
            if flag_status in ("VSC", "SC"):
                # Under safety car, can put on same compound too
                next_compounds = COMPOUNDS

            for next_comp in next_compounds:
                outlap = self._predict_live_pace(
                    track, current_lap, 1, next_comp, track_temp, weather, engine_mode
                )
                time_after, path_after = self.calculate_optimal_path(
                    track, current_lap + 1, 2, next_comp,
                    "GREEN", track_temp, weather, engine_mode,
                    base_pit_loss, vsc_pit_loss, stops_remaining - 1, memo
                )
                total_pit = pit_loss - sc_time_saving + outlap + time_after
                if total_pit < best_pit_total:
                    best_pit_total = total_pit
                    sc_tag = " [SC WINDOW 🟡]" if flag_status in ("VSC", "SC") else ""
                    best_pit_path = [{
                        "Lap": current_lap,
                        "Action": f"PIT → {next_comp}{sc_tag}",
                        "Pit Loss (s)": round(pit_loss, 1),
                        "Compound": next_comp,
                    }] + path_after

        if total_stay <= best_pit_total:
            result = (total_stay, path_stay)
        else:
            result = (best_pit_total, best_pit_path)

        memo[state] = result
        return result

    # ------------------------------------------------------------------ #
    #  Utility: lap time projection for charting                           #
    # ------------------------------------------------------------------ #

    def project_lap_times(self, track: str, strategy_actions: list,
                          current_lap: int, current_compound: str,
                          current_tire_age: int, total_laps: int,
                          track_temp: float = 35.0, weather: str = "DRY",
                          engine_mode: str = "BALANCED") -> list:
        """
        Returns a list of dicts [{lap, predicted_time, compound, tire_age}]
        for drawing the lap-time curve on the chart.
        """
        rows = []
        compound = current_compound
        age = current_tire_age
        pit_laps = {a["Lap"]: a["Compound"] for a in strategy_actions}

        for lap in range(current_lap, total_laps + 1):
            if lap in pit_laps:
                compound = pit_laps[lap]
                age = 1
            t = self._predict_live_pace(track, lap, age, compound, track_temp, weather, engine_mode)
            rows.append({"lap": lap, "time": round(t, 3), "compound": compound, "tire_age": age})
            age += 1

        return rows
