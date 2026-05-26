import pandas as pd
import numpy as np
from itertools import product as itertools_product


class StrategyOptimizer:
    """
    Brute-force sweep optimizer supporting 1-stop and 2-stop strategies.
    """

    def __init__(self, simulator):
        self.simulator = simulator

    # ------------------------------------------------------------------ #
    #  1-Stop                                                              #
    # ------------------------------------------------------------------ #

    def optimize_1_stop(self, start_search: int, end_search: int,
                        starting_compound: str,
                        available_choices: list) -> tuple:
        best = {"pit_lap": -1, "compound": None, "time": float("inf")}
        log = []

        for compound in available_choices:
            for pit_lap in range(start_search, end_search + 1):
                total, _ = self.simulator.run_strategy(
                    starting_compound, {pit_lap: compound}
                )
                log.append({
                    "Strategy": "1-stop",
                    "StartCompound": starting_compound,
                    "Pit1Lap": pit_lap,
                    "Pit1Compound": compound,
                    "Pit2Lap": None,
                    "Pit2Compound": None,
                    "TotalRaceTime": round(total, 3),
                })
                if total < best["time"]:
                    best = {"pit_lap": pit_lap, "compound": compound, "time": total}

        return best, pd.DataFrame(log)

    # ------------------------------------------------------------------ #
    #  2-Stop                                                              #
    # ------------------------------------------------------------------ #

    def optimize_2_stop(self, window1: tuple, window2: tuple,
                        starting_compound: str,
                        available_choices: list) -> tuple:
        """
        window1/window2: (start_lap, end_lap) search ranges.
        Enumerates all (lap1, comp1, lap2, comp2) combinations where lap1 < lap2.
        """
        best = {"pit1_lap": -1, "pit1_comp": None,
                "pit2_lap": -1, "pit2_comp": None, "time": float("inf")}
        log = []

        for comp1, comp2 in itertools_product(available_choices, repeat=2):
            for pit1 in range(*window1):
                for pit2 in range(*window2):
                    if pit2 <= pit1 + 3:  # minimum stint length
                        continue
                    total, _ = self.simulator.run_strategy(
                        starting_compound, {pit1: comp1, pit2: comp2}
                    )
                    log.append({
                        "Strategy": "2-stop",
                        "StartCompound": starting_compound,
                        "Pit1Lap": pit1,
                        "Pit1Compound": comp1,
                        "Pit2Lap": pit2,
                        "Pit2Compound": comp2,
                        "TotalRaceTime": round(total, 3),
                    })
                    if total < best["time"]:
                        best = {"pit1_lap": pit1, "pit1_comp": comp1,
                                "pit2_lap": pit2, "pit2_comp": comp2, "time": total}

        return best, pd.DataFrame(log)

    # ------------------------------------------------------------------ #
    #  Legacy shim                                                         #
    # ------------------------------------------------------------------ #

    def optimize_best_1_stop_compound(self, start_search, end_search,
                                      starting_compound, available_choices):
        best, df = self.optimize_1_stop(start_search, end_search,
                                        starting_compound, available_choices)
        return (best["pit_lap"], best["compound"], best["time"], df)
