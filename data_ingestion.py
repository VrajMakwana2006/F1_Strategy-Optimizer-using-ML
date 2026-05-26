import fastf1
import pandas as pd
import os

_CACHE_DIR = "f1_api_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(_CACHE_DIR)


def fetch_global_clean_dataset(
    years: list = [2022, 2023, 2024],
    tracks: list = ["Bahrain", "Saudi Arabia", "Australia", "Monaco",
                    "Silverstone", "Monza", "Spa", "Suzuka", "Singapore", "Abu Dhabi"],
) -> pd.DataFrame:
    """
    Downloads Race sessions for every (year, track) pair, strips DNFs,
    filters pit-in/pit-out laps, and returns a clean master DataFrame.

    Columns returned:
        Year, Track, Driver, LapNumber, TyreLife, Compound,
        LapTimeSeconds, TrackTemp
    """
    all_laps = []

    for year in years:
        for track in tracks:
            try:
                print(f"  ↳ {year} {track} …", end=" ", flush=True)
                session = fastf1.get_session(year, track, "R")
                session.load(laps=True, telemetry=False, weather=True)
                laps = session.laps

                if laps is None or laps.empty:
                    print("no data, skipped.")
                    continue

                # --- Finisher filter ----------------------------------------
                max_lap = laps["LapNumber"].max()
                results = session.results

                finished = (
                    results[results["Status"] == "Finished"]["Abbreviation"].tolist()
                )
                if not finished:
                    counts = laps.groupby("Driver").size()
                    finished = counts[counts >= max_lap * 0.9].index.tolist()

                clean = laps[laps["Driver"].isin(finished)].copy()

                # --- Strip pit transition laps ------------------------------
                clean = clean[clean["PitInTime"].isna() & clean["PitOutTime"].isna()]

                # --- Core numeric columns -----------------------------------
                clean["LapTimeSeconds"] = clean["LapTime"].dt.total_seconds()
                clean = clean.dropna(subset=["LapTimeSeconds", "LapNumber", "TyreLife", "Compound"])

                # Remove obvious outliers (SC laps, red-flag laps, etc.)
                q_lo = clean["LapTimeSeconds"].quantile(0.01)
                q_hi = clean["LapTimeSeconds"].quantile(0.97)
                clean = clean[(clean["LapTimeSeconds"] >= q_lo) & (clean["LapTimeSeconds"] <= q_hi)]

                # --- Track temperature (mean over session) ------------------
                try:
                    weather_data = session.weather_data
                    if weather_data is not None and not weather_data.empty and "TrackTemp" in weather_data.columns:
                        mean_track_temp = weather_data["TrackTemp"].dropna().mean()
                    else:
                        mean_track_temp = 35.0
                except Exception:
                    mean_track_temp = 35.0

                clean["TrackTemp"] = mean_track_temp
                clean["Track"] = track
                clean["Year"] = year

                subset = clean[["Year", "Track", "Driver", "LapNumber",
                                "TyreLife", "Compound", "LapTimeSeconds", "TrackTemp"]]
                all_laps.append(subset)
                print(f"✓ {len(subset):,} laps")

            except Exception as exc:
                print(f"⚠ skipped ({exc})")
                continue

    if not all_laps:
        raise ValueError(
            "No lap data could be fetched. Check FastF1 cache or internet connection."
        )

    master = pd.concat(all_laps, ignore_index=True)
    print(f"\nMaster dataset: {len(master):,} clean laps across "
          f"{master['Track'].nunique()} circuits, {master['Year'].nunique()} seasons.")
    return master


# ---- Legacy shim (keeps simulation_engine.py working) ----
class F1DataFetcher:
    def __init__(self):
        pass

    def fetch_race_laps(self, year: int, track: str):
        session = fastf1.get_session(year, track, "R")
        session.load(laps=True, telemetry=False, weather=False)
        return session.laps
