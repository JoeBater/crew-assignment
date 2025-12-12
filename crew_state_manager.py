import pandas as pd
from typing import Dict

from flight_utils import FlightUtils


class CrewStateManager:
    """Manages crew state tracking and location logic"""
    
    def __init__(self, crew_df: pd.DataFrame, flight_utils: FlightUtils):
        # Keep original DataFrame but determine canonical column names for crew id and role
        self.crew = crew_df
        # Detect crew id column (allow 'id' or 'crew_id')
        for col in ['id', 'crew_id', 'crewId']:
            if col in self.crew.columns:
                self.crew_id_col = col
                break
        else:
            raise KeyError(f"Crew DataFrame missing crew identifier column. Available columns: {self.crew.columns.tolist()}")

        # Detect role column
        for col in ['role', 'crew_role', 'crewRole']:
            if col in self.crew.columns:
                self.role_col = col
                break
        else:
            # Role isn't strictly required here but warn by raising to make debugging explicit
            raise KeyError(f"Crew DataFrame missing role column. Available columns: {self.crew.columns.tolist()}")
        self.flight_utils = flight_utils
    
    def initialize_crew_state(self, assignment: Dict) -> Dict:
        """Initialize crew state from assignment"""
        # Build crew_state keyed by the detected crew id column
        crew_state = {crew_id: [] for crew_id in self.crew[self.crew_id_col]}

        for fid, roles in assignment.items():
            flight_info = self._create_flight_info(fid)

            if roles.get('captain') is not None:
                flight_info_copy = flight_info.copy()
                flight_info_copy['role'] = 'captain'
                crew_state[roles['captain']].append(flight_info_copy)

            if roles.get('first_officer') is not None:
                flight_info_copy = flight_info.copy()
                flight_info_copy['role'] = 'first_officer'
                crew_state[roles['first_officer']].append(flight_info_copy)

            for dh in roles.get('dead_heading', []):
                flight_info_copy = flight_info.copy()
                flight_info_copy['role'] = 'dead_heading'
                crew_state[dh].append(flight_info_copy)

        # Sort crew flights chronologically
        for crew_id in crew_state:
            crew_state[crew_id].sort(key=lambda x: (x['day'], x.get('depart', None)))

        return crew_state
    
    def _create_flight_info(self, fid: str) -> Dict:
        """Create flight info dictionary for a flight"""
        return {
            'day': self.flight_utils.get_day(fid),
            'flight': fid,
            'route': self.flight_utils.get_route_makeup(fid),
            'start_utc': self.flight_utils.get_start_utc(fid),
            'end_utc': self.flight_utils.get_end_utc(fid),
            'FDP_length_numeric': self.flight_utils.get_FDP_length_numeric(fid),
            'operational_sectors': self.flight_utils.get_operational_sectors(fid)
        }
    
    