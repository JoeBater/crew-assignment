"""
Repair operators for Large Neighborhood Search in crew assignment optimization.

This module contains various repair operators that rebuild assignments after the destroy phase
to create feasible solutions in the LNS algorithm.
"""

import copy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, TYPE_CHECKING

# Import for type hints only to avoid circular imports
if TYPE_CHECKING:
    from crew_optimizer import FlightUtils, CrewStateManager


class RepairOperator(ABC):
    """Abstract base class for repair operators"""
    
    @abstractmethod
    def repair(self, assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame, 
               flight_utils: 'FlightUtils', crew_manager: 'CrewStateManager') -> Tuple[Dict, Dict]:
        """
        Repair assignments and return modified assignment and crew_state
        
        Args:
            assignment: Current flight assignments (possibly with gaps)
            crew_state: Current crew scheduling state
            crew_df: DataFrame containing crew information
            flight_utils: Utility class for flight operations
            crew_manager: Utility class for crew state management
            
        Returns:
            Tuple of (repaired_assignment, repaired_crew_state)
        """
        pass


class RandomRepairOperator(RepairOperator):
    """Random repair - assigns any qualified crew to unassigned flights"""
    
    def repair(self, assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame, 
               flight_utils: 'FlightUtils', crew_manager: 'CrewStateManager') -> Tuple[Dict, Dict]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        for fid, roles in new_assignment.items():
            if roles['captain'] is not None and roles['first_officer'] is not None:
                continue
            
            # Repair captain
            if roles['captain'] is None:
                possible_captains = crew_df[
                    (crew_df['crew_role'] == 'captain') 
                ]['crew_id'].values
                
                if len(possible_captains) > 0:
                    repair_captain = np.random.choice(possible_captains.tolist())
                    new_assignment[fid]['captain'] = repair_captain
                    
                    # Add to crew state
                    flight_info = crew_manager._create_flight_info(fid)
                    flight_info['role'] = 'captain'
                    new_crew_state[repair_captain].append(flight_info)
                    new_crew_state[repair_captain].sort(key=lambda f: (f['day'], f['start_utc']))
            
            # Repair first officer
            if roles['first_officer'] is None:
                possible_first_officers = crew_df[
                    (crew_df['crew_role'] == 'first_officer') 
                ]['crew_id'].values
                
                if len(possible_first_officers) > 0:
                    repair_first_officer = np.random.choice(possible_first_officers.tolist())
                    new_assignment[fid]['first_officer'] = repair_first_officer
                    
                    # Add to crew state
                    flight_info = crew_manager._create_flight_info(fid)
                    flight_info['role'] = 'first_officer'
                    new_crew_state[repair_first_officer].append(flight_info)
                    new_crew_state[repair_first_officer].sort(key=lambda f: (f['day'], f['start_utc']))
        
        return new_assignment, new_crew_state

