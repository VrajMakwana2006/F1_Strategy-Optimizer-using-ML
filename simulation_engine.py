import pandas as pd
import numpy as np
from tire_model import AdvancedRacePaceModel


class CarSimulationState:
    def __init__(self, starting_compound: str):
        self.current_compound = starting_compound
        self.tire_age = 1
        self.total_race_time = 0.0
        self.lap_by_lap_history = []

    def age_tires(self):
        self.tire_age += 1

    def perform_pit_stop(self, new_compound: str, pit_penalty: float):
        self.current_compound = new_compound
        self.tire_age = 1
        self.total_race_time += pit_penalty


class RaceSimulator:
    def __init__(self, total_laps: int, pit_penalty: float,
                 pace_model: AdvancedRacePaceModel,
                 track: str = "Bahrain", track_temp: float = 35.0):
        self.total_laps = total_laps
        self.pit_penalty = pit_penalty
        self.pace_model = pace_model
        self.track = track
        self.track_temp = track_temp

    def run_strategy(self, starting_compound: str, pit_windows: dict) -> tuple:
        """
        pit_windows: {lap_number: 'COMPOUND'} — pit at END of that lap.
        Returns (total_race_time_seconds, lap_history_df).
        """
        car = CarSimulationState(starting_compound)

        for current_lap in range(1, self.total_laps + 1):
            if current_lap in pit_windows:
                car.perform_pit_stop(pit_windows[current_lap], self.pit_penalty)

            lap_time = self.pace_model.predict_lap_time(
                track=self.track,
                lap_number=current_lap,
                tire_age=car.tire_age,
                compound=car.current_compound,
                track_temp=self.track_temp,
            )
            car.total_race_time += lap_time
            car.lap_by_lap_history.append({
                "LapNumber": current_lap,
                "LapTime": round(lap_time, 3),
                "TireAge": car.tire_age,
                "Compound": car.current_compound,
                "CumulativeTime": round(car.total_race_time, 3),
            })
            car.age_tires()

        return car.total_race_time, pd.DataFrame(car.lap_by_lap_history)
