import pandas as pd
import numpy as np
import copy
from datetime import datetime, timedelta, time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional


class FlightUtils:
    """Utility class for flight-related operations"""
    
    def __init__(self, flights_df: pd.DataFrame):
        self.flights = flights_df
    
    def get_day(self, fid: str) -> int:
        return int(self.flights[self.flights['id'] == fid]['day'].values[0])
    
    def get_origin(self, fid: str) -> str:
        return self.flights[self.flights['id'] == fid]['origin'].values[0]
    
    def get_destination(self, fid: str) -> str:
        return self.flights[self.flights['id'] == fid]['dest'].values[0]
    
    def get_departure(self, fid: str):
        return self.flights[self.flights['id'] == fid]['dep'].values[0]
    
    def get_arrival(self, fid: str):
        return self.flights[self.flights['id'] == fid]['arr'].values[0]
    
    def get_duration(self, fid: str) -> int:
        return int(self.flights[self.flights['id'] == fid]['duration'].values[0])
    
    def get_aircraft_type(self, fid: str) -> str:
        return self.flights[self.flights['id'] == fid]['type'].values[0]


class CrewStateManager:
    """Manages crew state tracking and location logic"""
    
    def __init__(self, crew_df: pd.DataFrame, flight_utils: FlightUtils):
        self.crew = crew_df
        self.flight_utils = flight_utils
    
    def initialize_crew_state(self, assignment: Dict) -> Dict:
        """Initialize crew state from assignment"""
        crew_state = {crew_id: [] for crew_id in self.crew['id']}
        
        for fid, roles in assignment.items():
            flight_info = self._create_flight_info(fid)
            
            if roles['captain'] is not None:
                flight_info_copy = flight_info.copy()
                flight_info_copy['role'] = 'captain'
                crew_state[roles['captain']].append(flight_info_copy)
            
            if roles['first_officer'] is not None:
                flight_info_copy = flight_info.copy()
                flight_info_copy['role'] = 'first_officer'
                crew_state[roles['first_officer']].append(flight_info_copy)
            
            for dh in roles['dead_heading']:
                flight_info_copy = flight_info.copy()
                flight_info_copy['role'] = 'dead_heading'
                crew_state[dh].append(flight_info_copy)
        
        # Sort crew flights chronologically
        for crew_id in crew_state:
            crew_state[crew_id].sort(key=lambda x: (x['day'], x['depart']))
        
        return crew_state
    
    def _create_flight_info(self, fid: str) -> Dict:
        """Create flight info dictionary for a flight"""
        return {
            'day': self.flight_utils.get_day(fid),
            'flight': fid,
            'origin': self.flight_utils.get_origin(fid),
            'destination': self.flight_utils.get_destination(fid),
            'depart': self.flight_utils.get_departure(fid),
            'arrive': self.flight_utils.get_arrival(fid),
            'duration': self.flight_utils.get_duration(fid)
        }
    
    def is_crew_available_at_location(self, crew_id: str, target_location: str, 
                                    target_day: int, target_time, crew_state: Dict) -> bool:
        """Check if crew member is available at target location before target time"""
        crew_flights = crew_state.get(crew_id, [])
        crew_base = self.crew[self.crew['id'] == crew_id].iloc[0]['base']
        current_location = crew_base
        
        if crew_flights:
            sorted_flights = sorted(crew_flights, key=lambda x: (x['day'], x['depart']))
            
            for flight in sorted_flights:
                if flight['day'] < target_day:
                    current_location = flight['destination']
                elif flight['day'] == target_day and flight['depart'] < target_time:
                    current_location = flight['destination']
                elif flight['day'] == target_day and flight['depart'] >= target_time:
                    break
        
        return current_location == target_location


class DestroyOperator(ABC):
    """Abstract base class for destroy operators"""
    
    @abstractmethod
    def destroy(self, assignment: Dict, crew_state: Dict, num_to_destroy: int, 
                flights_df: pd.DataFrame, **kwargs) -> Tuple[Dict, Dict, List, List]:
        """
        Destroy assignments and return modified assignment, crew_state, 
        destroyed flights, and affected crew
        """
        pass


class RandomDestroyOperator(DestroyOperator):
    """Randomly destroys flights"""
    
    def destroy(self, assignment: Dict, crew_state: Dict, num_to_destroy: int, 
                flights_df: pd.DataFrame, **kwargs) -> Tuple[Dict, Dict, List, List]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        flights_to_destroy = np.random.choice(flights_df['id'].tolist(), num_to_destroy, replace=False)
        affected_crew = set()
        
        for ftd in flights_to_destroy:
            # Track affected crew
            if new_assignment[ftd]['captain'] is not None:
                affected_crew.add(new_assignment[ftd]['captain'])
            if new_assignment[ftd]['first_officer'] is not None:
                affected_crew.add(new_assignment[ftd]['first_officer'])
            for dh in new_assignment[ftd]['dead_heading']:
                affected_crew.add(dh)
            
            # Remove from assignment
            new_assignment[ftd]['captain'] = None
            new_assignment[ftd]['first_officer'] = None
            new_assignment[ftd]['dead_heading'] = []
            
            # Remove from crew state
            for crew_id in affected_crew:
                if crew_id in new_crew_state:
                    new_crew_state[crew_id] = [
                        f for f in new_crew_state[crew_id] 
                        if f['flight'] not in flights_to_destroy
                    ]
        
        return new_assignment, new_crew_state, flights_to_destroy.tolist(), list(affected_crew)


class OverlapDestroyOperator(DestroyOperator):
    """Destroys overlapping flights for crew members"""
    
    def destroy(self, assignment: Dict, crew_state: Dict, num_to_destroy: int, 
                flights_df: pd.DataFrame, **kwargs) -> Tuple[Dict, Dict, List, List]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        flights_to_destroy = []
        affected_crew = set()
        
        for crew_id, crew_flights in new_crew_state.items():
            if len(crew_flights) <= 1:
                continue
            
            remaining_flights = []
            sorted_flights = sorted(crew_flights, key=lambda x: (x['day'], x['depart']))
            
            for i, current_flight in enumerate(sorted_flights):
                has_overlap = False
                
                for j, other_flight in enumerate(sorted_flights):
                    if i != j and self._flights_overlap(current_flight, other_flight):
                        has_overlap = True
                        break
                
                if has_overlap:
                    ftd = current_flight['flight']
                    role = current_flight['role']
                    affected_crew.add(crew_id)
                    flights_to_destroy.append(ftd)
                    
                    # Remove from assignment based on role
                    if role == 'captain':
                        new_assignment[ftd]['captain'] = None
                    elif role == 'first_officer':
                        new_assignment[ftd]['first_officer'] = None
                    elif role == 'dead_heading':
                        if crew_id in new_assignment[ftd]['dead_heading']:
                            new_assignment[ftd]['dead_heading'].remove(crew_id)
                else:
                    remaining_flights.append(current_flight)
            
            new_crew_state[crew_id] = remaining_flights
        
        return new_assignment, new_crew_state, flights_to_destroy, list(affected_crew)
    
    def _flights_overlap(self, flight1: Dict, flight2: Dict) -> bool:
        """Check if two flights overlap in time"""
        return (flight1['day'] == flight2['day'] and 
                flight1['depart'] < flight2['arrive'] and 
                flight2['depart'] < flight1['arrive'])


class RepairOperator(ABC):
    """Abstract base class for repair operators"""
    
    @abstractmethod
    def repair(self, assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame, 
               flight_utils: FlightUtils, crew_manager: CrewStateManager) -> Tuple[Dict, Dict]:
        """Repair assignments and return modified assignment and crew_state"""
        pass


class RandomRepairOperator(RepairOperator):
    """Random repair - assigns any qualified crew to unassigned flights"""
    
    def repair(self, assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame, 
               flight_utils: FlightUtils, crew_manager: CrewStateManager) -> Tuple[Dict, Dict]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        for fid, roles in new_assignment.items():
            if roles['captain'] is not None and roles['first_officer'] is not None:
                continue
            
            aircraft = flight_utils.get_aircraft_type(fid)
            
            # Repair captain
            if roles['captain'] is None:
                possible_captains = crew_df[
                    (crew_df['role'] == 'captain') &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                if len(possible_captains) > 0:
                    repair_captain = np.random.choice(possible_captains.tolist())
                    new_assignment[fid]['captain'] = repair_captain
                    
                    # Add to crew state
                    flight_info = crew_manager._create_flight_info(fid)
                    flight_info['role'] = 'captain'
                    new_crew_state[repair_captain].append(flight_info)
                    new_crew_state[repair_captain].sort(key=lambda f: (f['day'], f['depart']))
            
            # Repair first officer
            if roles['first_officer'] is None:
                possible_first_officers = crew_df[
                    (crew_df['role'] == 'first_officer') &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                if len(possible_first_officers) > 0:
                    repair_first_officer = np.random.choice(possible_first_officers.tolist())
                    new_assignment[fid]['first_officer'] = repair_first_officer
                    
                    # Add to crew state
                    flight_info = crew_manager._create_flight_info(fid)
                    flight_info['role'] = 'first_officer'
                    new_crew_state[repair_first_officer].append(flight_info)
                    new_crew_state[repair_first_officer].sort(key=lambda f: (f['day'], f['depart']))
        
        return new_assignment, new_crew_state


class LocationAwareRepairOperator(RepairOperator):
    """Location-aware repair - only assigns crew who are at the right location"""
    
    def repair(self, assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame, 
               flight_utils: FlightUtils, crew_manager: CrewStateManager) -> Tuple[Dict, Dict]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        for fid, roles in new_assignment.items():
            if roles['captain'] is not None and roles['first_officer'] is not None:
                continue
            
            aircraft = flight_utils.get_aircraft_type(fid)
            origin = flight_utils.get_origin(fid)
            depart = flight_utils.get_departure(fid)
            flight_day = flight_utils.get_day(fid)
            
            # Repair captain
            if roles['captain'] is None:
                possible_captains = crew_df[
                    (crew_df['role'] == 'captain') &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                for crew_id in possible_captains:
                    if crew_manager.is_crew_available_at_location(
                        crew_id, origin, flight_day, depart, new_crew_state):
                        
                        new_assignment[fid]['captain'] = crew_id
                        
                        # Add to crew state
                        flight_info = crew_manager._create_flight_info(fid)
                        flight_info['role'] = 'captain'
                        new_crew_state[crew_id].append(flight_info)
                        new_crew_state[crew_id].sort(key=lambda f: (f['day'], f['depart']))
                        break
            
            # Repair first officer
            if roles['first_officer'] is None:
                possible_first_officers = crew_df[
                    (crew_df['role'] == 'first_officer') &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                for crew_id in possible_first_officers:
                    if crew_manager.is_crew_available_at_location(
                        crew_id, origin, flight_day, depart, new_crew_state):
                        
                        new_assignment[fid]['first_officer'] = crew_id
                        
                        # Add to crew state
                        flight_info = crew_manager._create_flight_info(fid)
                        flight_info['role'] = 'first_officer'
                        new_crew_state[crew_id].append(flight_info)
                        new_crew_state[crew_id].sort(key=lambda f: (f['day'], f['depart']))
                        break
        
        return new_assignment, new_crew_state


class CrewOptimizer:
    """Main optimizer class that orchestrates destroy and repair operations"""
    
    def __init__(self, flights_file: str, crew_file: str, verbose: bool = False):
        # Load data
        self.flights = pd.read_json(flights_file)
        self.flights['dep'] = pd.to_datetime(self.flights['dep']).dt.time
        self.flights['arr'] = pd.to_datetime(self.flights['arr']).dt.time
        
        self.crew = pd.read_json(crew_file)
        
        # Initialize components
        self.flight_utils = FlightUtils(self.flights)
        self.crew_manager = CrewStateManager(self.crew, self.flight_utils)
        
        # State
        self.assignment = None
        self.crew_state = None
        self.verbose = verbose
        
        # Operators
        self.destroy_operators = [
            RandomDestroyOperator(),
            OverlapDestroyOperator()
        ]
        
        self.repair_operators = [
            RandomRepairOperator(),
            LocationAwareRepairOperator()
        ]
    
    def initial_assignment(self) -> Dict:
        """Create initial random assignment"""
        self.assignment = {
            fid: {"captain": None, "first_officer": None, "dead_heading": []}
            for fid in self.flights['id']
        }
        
        for _, flight in self.flights.iterrows():
            fid = flight['id']
            origin = flight['origin']
            aircraft = flight['type']
            
            # Find eligible crew
            eligible_captains = self.crew[
                (self.crew['role'] == 'captain') &
                (self.crew['base'] == origin) &
                (self.crew['qualified'].apply(lambda q: aircraft in q))
            ]['id'].values
            
            eligible_first_officers = self.crew[
                (self.crew['role'] == 'first_officer') &
                (self.crew['base'] == origin) &
                (self.crew['qualified'].apply(lambda q: aircraft in q))
            ]['id'].values
            
            # Assign randomly
            if len(eligible_captains) > 0:
                self.assignment[fid]['captain'] = np.random.choice(eligible_captains.tolist())
            if len(eligible_first_officers) > 0:
                self.assignment[fid]['first_officer'] = np.random.choice(eligible_first_officers.tolist())
        
        return self.assignment
    
    def initial_crew_state(self) -> Dict:
        """Initialize crew state from assignment"""
        if self.assignment is None:
            raise ValueError("Call initial_assignment() first")
        
        self.crew_state = self.crew_manager.initialize_crew_state(self.assignment)
        return self.crew_state
    
    def destroy(self, num_to_destroy: int, operator_index: Optional[int] = None) -> Tuple[Dict, Dict, List, List]:
        """Apply destroy operator"""
        if self.assignment is None or self.crew_state is None:
            raise ValueError("Call initial_assignment() and initial_crew_state() first")
        
        if operator_index is None:
            operator_index = np.random.randint(0, len(self.destroy_operators))
        
        operator = self.destroy_operators[operator_index]
        
        if self.verbose:
            print(f"Destroy - using operator {operator_index} => {operator.__class__.__name__}")
        
        return operator.destroy(self.assignment, self.crew_state, num_to_destroy, self.flights)
    
    def repair(self, assignment: Dict, crew_state: Dict, operator_index: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Apply repair operator"""
        if operator_index is None:
            operator_index = np.random.randint(0, len(self.repair_operators))
        
        operator = self.repair_operators[operator_index]
        
        if self.verbose:
            print(f"Repair - using operator {operator_index} => {operator.__class__.__name__}")
        
        return operator.repair(assignment, crew_state, self.crew, self.flight_utils, self.crew_manager)
    
    def add_destroy_operator(self, operator: DestroyOperator):
        """Add a custom destroy operator"""
        self.destroy_operators.append(operator)
    
    def add_repair_operator(self, operator: RepairOperator):
        """Add a custom repair operator"""
        self.repair_operators.append(operator)