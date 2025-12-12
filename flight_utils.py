import pandas as pd
from typing import Dict
import datetime

class FlightUtils:
    """Utility class for flight-related operations"""
    
    def __init__(self, flights_df: pd.DataFrame):
        self.flights = flights_df
    
    def get_day(self, fid: str) -> datetime.date:
        return int(self.flights[self.flights['pairing_id'] == fid]['start date'].values[0])
    
    def get_route_makeup(self, fid: str) -> str:
        return self.flights[self.flights['pairing_id'] == fid]['duty_route_makeup'].values[0]
    
    def get_start_utc(self, fid: str):
        return self.flights[self.flights['pairing_id'] == fid]['start_utc'].values[0]
    
    def get_end_utc(self, fid: str):
        return self.flights[self.flights['pairing_id'] == fid]['end_utc'].values[0]
    
    def get_FDP_length_numeric(self, fid: str) -> int:
        return int(self.flights[self.flights['pairing_id'] == fid]['FDPLengthNumeric'].values[0])
    
    def get_operational_sectors(self, fid: str) -> str:
        return self.flights[self.flights['pairing_id'] == fid]['OperationalSectors'].values[0]
