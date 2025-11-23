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
               flight_utils: 'FlightUtils', crew_manager: 'CrewStateManager') -> Tuple[Dict, Dict]:
        
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


class BaseMatchingRepairOperator(RepairOperator):
    """Repair operator that prioritizes crew from the same base as flight origin"""
    
    def repair(self, assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame, 
               flight_utils: 'FlightUtils', crew_manager: 'CrewStateManager') -> Tuple[Dict, Dict]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        for fid, roles in new_assignment.items():
            if roles['captain'] is not None and roles['first_officer'] is not None:
                continue
            
            aircraft = flight_utils.get_aircraft_type(fid)
            origin = flight_utils.get_origin(fid)
            
            # Repair captain - prioritize same base
            if roles['captain'] is None:
                # First try: same base crew
                possible_captains = crew_df[
                    (crew_df['role'] == 'captain') &
                    (crew_df['base'] == origin) &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                # Second try: any qualified crew
                if len(possible_captains) == 0:
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
            
            # Similar logic for first officer
            if roles['first_officer'] is None:
                # First try: same base crew
                possible_first_officers = crew_df[
                    (crew_df['role'] == 'first_officer') &
                    (crew_df['base'] == origin) &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                # Second try: any qualified crew
                if len(possible_first_officers) == 0:
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


class QualificationFirstRepairOperator(RepairOperator):
    """Repair operator that prioritizes qualification matches and handles rare aircraft types"""
    
    def repair(self, assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame, 
               flight_utils: 'FlightUtils', crew_manager: 'CrewStateManager') -> Tuple[Dict, Dict]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        # Sort unassigned flights by aircraft type rarity
        unassigned_flights = [fid for fid, roles in new_assignment.items() 
                             if roles['captain'] is None or roles['first_officer'] is None]
        
        # Count aircraft types to prioritize rare ones
        aircraft_counts = {}
        for fid in unassigned_flights:
            aircraft = flight_utils.get_aircraft_type(fid)
            aircraft_counts[aircraft] = aircraft_counts.get(aircraft, 0) + 1
        
        # Sort by aircraft rarity (ascending - rarest first)
        unassigned_flights.sort(key=lambda fid: aircraft_counts[flight_utils.get_aircraft_type(fid)])
        
        for fid in unassigned_flights:
            roles = new_assignment[fid]
            aircraft = flight_utils.get_aircraft_type(fid)
            origin = flight_utils.get_origin(fid)
            flight_day = flight_utils.get_day(fid)
            depart = flight_utils.get_departure(fid)
            
            # Repair captain
            if roles['captain'] is None:
                # Find qualified captains first, then check availability
                qualified_captains = crew_df[
                    (crew_df['role'] == 'captain') &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                for crew_id in qualified_captains:
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
                qualified_first_officers = crew_df[
                    (crew_df['role'] == 'first_officer') &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                for crew_id in qualified_first_officers:
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


class GreedyCostRepairOperator(RepairOperator):
    """Repair operator that selects assignments with lowest cost impact"""
    
    def repair(self, assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame, 
               flight_utils: 'FlightUtils', crew_manager: 'CrewStateManager') -> Tuple[Dict, Dict]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        for fid, roles in new_assignment.items():
            if roles['captain'] is not None and roles['first_officer'] is not None:
                continue
            
            aircraft = flight_utils.get_aircraft_type(fid)
            origin = flight_utils.get_origin(fid)
            flight_day = flight_utils.get_day(fid)
            depart = flight_utils.get_departure(fid)
            
            # Repair captain with cost consideration
            if roles['captain'] is None:
                qualified_captains = crew_df[
                    (crew_df['role'] == 'captain') &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                best_captain = None
                best_cost = float('inf')
                
                for crew_id in qualified_captains:
                    if crew_manager.is_crew_available_at_location(
                        crew_id, origin, flight_day, depart, new_crew_state):
                        
                        # Calculate assignment cost
                        crew_member = crew_df[crew_df['id'] == crew_id].iloc[0]
                        cost = 0
                        
                        # Base mismatch penalty
                        if crew_member['base'] != origin:
                            cost += 500
                        
                        # Workload penalty (more flights = higher cost)
                        current_workload = len(new_crew_state.get(crew_id, []))
                        cost += current_workload * 50
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_captain = crew_id
                
                if best_captain is not None:
                    new_assignment[fid]['captain'] = best_captain
                    
                    # Add to crew state
                    flight_info = crew_manager._create_flight_info(fid)
                    flight_info['role'] = 'captain'
                    new_crew_state[best_captain].append(flight_info)
                    new_crew_state[best_captain].sort(key=lambda f: (f['day'], f['depart']))
            
            # Similar logic for first officer
            if roles['first_officer'] is None:
                qualified_first_officers = crew_df[
                    (crew_df['role'] == 'first_officer') &
                    (crew_df['qualified'].apply(lambda q: aircraft in q))
                ]['id'].values
                
                best_first_officer = None
                best_cost = float('inf')
                
                for crew_id in qualified_first_officers:
                    if crew_manager.is_crew_available_at_location(
                        crew_id, origin, flight_day, depart, new_crew_state):
                        
                        # Calculate assignment cost
                        crew_member = crew_df[crew_df['id'] == crew_id].iloc[0]
                        cost = 0
                        
                        # Base mismatch penalty
                        if crew_member['base'] != origin:
                            cost += 500
                        
                        # Workload penalty
                        current_workload = len(new_crew_state.get(crew_id, []))
                        cost += current_workload * 50
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_first_officer = crew_id
                
                if best_first_officer is not None:
                    new_assignment[fid]['first_officer'] = best_first_officer
                    
                    # Add to crew state
                    flight_info = crew_manager._create_flight_info(fid)
                    flight_info['role'] = 'first_officer'
                    new_crew_state[best_first_officer].append(flight_info)
                    new_crew_state[best_first_officer].sort(key=lambda f: (f['day'], f['depart']))
        
        return new_assignment, new_crew_state