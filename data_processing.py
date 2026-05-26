import pandas as pd

class F1DataProcessor:
    def __init__(self, raw_laps: pd.DataFrame):
        """Initializes the processor with the raw lap data."""
        self.raw_laps = raw_laps

    def clean_race_laps(self) -> pd.DataFrame:
        """
        Filters the raw data to keep only representative 'flying laps' 
        under green flag conditions.
        """
        # 1. Make a copy so we don't modify the original data accidentally
        clean_laps = self.raw_laps.copy()

        # 2. Remove laps where the lap time is missing (e.g., driver retired)
        clean_laps = clean_laps.dropna(subset=['LapTime'])

        # 3. Filter out In-laps (PitInTime is not null) and Out-laps (PitOutTime is not null)
        clean_laps = clean_laps[pd.isnull(clean_laps['PitInTime'])]
        clean_laps = clean_laps[pd.isnull(clean_laps['PitOutTime'])]

        # 4. Filter for green flag conditions. TrackStatus '1' means clear track.
        # We keep laps that only have a '1' in their track status string.
        clean_laps = clean_laps[clean_laps['TrackStatus'] == '1']

        # 5. Convert timedelta objects to float (seconds) for math/ML operations
        clean_laps['LapTime_sec'] = clean_laps['LapTime'].dt.total_seconds()
        
        return clean_laps

    def get_driver_stint(self, clean_laps: pd.DataFrame, driver_code: str, stint_number: int) -> pd.DataFrame:
        """
        Isolates the lap data for a specific driver and a specific tire stint.
        """
        # Filter for the specific driver (e.g., 'VER', 'LEC')
        driver_laps = clean_laps[clean_laps['Driver'] == driver_code]
        
        # Filter for the specific stint (1 = first set of tires, 2 = second set, etc.)
        stint_laps = driver_laps[driver_laps['Stint'] == stint_number]
        
        return stint_laps

if __name__ == "__main__":
    # To test this, we need to import our fetcher from the previous step
    from data_ingestion import F1DataFetcher
    
    print("Fetching raw data...")
    fetcher = F1DataFetcher()
    raw_laps = fetcher.fetch_race_laps(2024, "Bahrain")
    
    print("Processing and cleaning data...")
    processor = F1DataProcessor(raw_laps)
    clean_laps = processor.clean_race_laps()
    
    print(f"\nRaw laps count: {len(raw_laps)}")
    print(f"Clean flying laps count: {len(clean_laps)}")
    print(f"Removed {len(raw_laps) - len(clean_laps)} invalid/slow laps.")
    
    # Let's extract Max Verstappen's ('VER') second stint
    ver_stint_2 = processor.get_driver_stint(clean_laps, "VER", 2)
    
    print("\nMax Verstappen's Stint 2 Data (First 5 valid laps):")
    # Display just the Lap Number, Tire Life, and our new LapTime_sec column
    print(ver_stint_2[['LapNumber', 'Compound', 'TyreLife', 'LapTime_sec']].head())