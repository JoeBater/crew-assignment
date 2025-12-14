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


class DeadheadingRepairOperator(RepairOperator):
    """
    Deadheading repair operator that fills empty crew slots by deadheading 
    appropriately qualified crew members to the flight's origin on a preceding flight.
    
    This operator:
    1. Identifies empty crew slots (unassigned captain or first officer)
    2. Finds qualified crew members currently at other locations
    3. Schedules a deadhead flight (or uses existing flight) to move crew to the origin
    4. Makes the crew available for the target flight
    
    Deadheading is a cost-effective way to position crew without revenue-generating flights.
    """
    
    def repair(self, assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame, 
               flight_utils: 'FlightUtils', crew_manager: 'CrewStateManager') -> Tuple[Dict, Dict]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        # Process each unassigned flight
        for fid, roles in new_assignment.items():
            if roles['captain'] is not None and roles['first_officer'] is not None:
                continue  # Skip fully assigned flights
            
            aircraft = flight_utils.get_aircraft_type(fid)
            origin = flight_utils.get_origin(fid)
            flight_day = flight_utils.get_day(fid)
            depart = flight_utils.get_departure(fid)
            
            # Repair captain
            if roles['captain'] is None:
                repaired_captain = self._find_crew_with_deadhead(
                    'captain', aircraft, origin, flight_day, depart, 
                    new_assignment, new_crew_state, crew_df, flight_utils, crew_manager
                )
                
                if repaired_captain is not None:
                    new_assignment[fid]['captain'] = repaired_captain
                    
                    # Add to crew state
                    flight_info = crew_manager._create_flight_info(fid)
                    flight_info['role'] = 'captain'
                    new_crew_state[repaired_captain].append(flight_info)
                    new_crew_state[repaired_captain].sort(key=lambda f: (f['day'], f['depart']))
            
            # Repair first officer
            if roles['first_officer'] is None:
                repaired_first_officer = self._find_crew_with_deadhead(
                    'first_officer', aircraft, origin, flight_day, depart, 
                    new_assignment, new_crew_state, crew_df, flight_utils, crew_manager
                )
                
                if repaired_first_officer is not None:
                    new_assignment[fid]['first_officer'] = repaired_first_officer
                    
                    # Add to crew state
                    flight_info = crew_manager._create_flight_info(fid)
                    flight_info['role'] = 'first_officer'
                    new_crew_state[repaired_first_officer].append(flight_info)
                    new_crew_state[repaired_first_officer].sort(key=lambda f: (f['day'], f['depart']))
        
        return new_assignment, new_crew_state
    
    def _find_crew_with_deadhead(self, role: str, aircraft: str, target_origin: str, 
                                  target_day: int, target_depart_time: Tuple[int, int],
                                  assignment: Dict, crew_state: Dict, crew_df: pd.DataFrame,
                                  flight_utils: 'FlightUtils', crew_manager: 'CrewStateManager') -> str:
        """
        Find a qualified crew member and potentially deadhead them to the target location.
        
        Args:
            role: 'captain' or 'first_officer'
            aircraft: Aircraft type for the target flight
            target_origin: Origin airport of the target flight
            target_day: Day of the target flight
            target_depart_time: Departure time of the target flight (tuple of hours/minutes)
            assignment: Current flight assignments
            crew_state: Current crew scheduling state
            crew_df: Crew information dataframe
            flight_utils: Flight utility functions
            crew_manager: Crew state management utility
            
        Returns:
            Crew ID if found (either already at location or deadheadable), None otherwise
        """
        
        # Find all qualified crew for this role and aircraft
        qualified_crew = crew_df[
            (crew_df['role'] == role) &
            (crew_df['qualified'].apply(lambda q: aircraft in q))
        ]['id'].values
        
        # First, try to find crew already at the target location
        for crew_id in qualified_crew:
            if crew_manager.is_crew_available_at_location(
                crew_id, target_origin, target_day, target_depart_time, crew_state):
                return crew_id
        
        # If no crew at location, try to find someone who can be deadheaded
        for crew_id in qualified_crew:
            crew_flights = crew_state.get(crew_id, [])
            
            if len(crew_flights) == 0:
                # Crew has no flights yet; can be based at origin on target day
                current_location = crew_df[crew_df['id'] == crew_id].iloc[0]['base']
                if current_location == target_origin:
                    return crew_id
                else:
                    # Could deadhead from base, but no incoming flight needed
                    return crew_id
            
            # Get crew's last flight before the target flight
            last_flight = crew_flights[-1]
            last_flight_day = last_flight['day']
            last_arrival_time = last_flight['arrive']
            last_arrival_location = last_flight['destination']
            
            # Check if crew could deadhead to target origin
            if last_flight_day == target_day and last_arrival_location != target_origin:
                # Look for a connecting deadhead flight on the same day
                deadhead_flight = self._find_deadhead_flight(
                    last_arrival_location, target_origin, target_day, 
                    last_arrival_time, target_depart_time, assignment, flight_utils
                )
                
                if deadhead_flight is not None:
                    # Can deadhead via this flight
                    return crew_id
            elif last_flight_day < target_day and last_arrival_location == target_origin:
                # Crew is already at target location from a prior flight
                return crew_id
            elif last_flight_day < target_day:
                # Crew is at a different location but has time to deadhead back to origin
                # (This could be extended to find deadhead paths, but for simplicity,
                # we only support same-day connections)
                pass
        
        return None
    
    def _find_deadhead_flight(self, origin: str, destination: str, day: int, 
                              earliest_departure: Tuple[int, int], 
                              latest_arrival: Tuple[int, int],
                              assignment: Dict, 
                              flight_utils: 'FlightUtils') -> str:
        """
        Find a deadhead (positioning) flight between origin and destination on a given day.
        
        A valid deadhead flight must:
        - Depart from origin after the crew's last flight arrives
        - Arrive at destination before the target flight departs
        - Have empty crew slots available (i.e., unassigned captain or first officer)
        
        Args:
            origin: Starting airport for deadhead
            destination: Ending airport for deadhead
            day: Day of deadhead
            earliest_departure: Earliest the deadhead can depart (tuple of hours/minutes)
            latest_arrival: Latest the deadhead can arrive (tuple of hours/minutes)
            assignment: Current flight assignments
            flight_utils: Flight utility functions
            
        Returns:
            Flight ID of a suitable deadhead flight, or None if not found
        """
        
        for fid, roles in assignment.items():
            # Only consider flights on the correct day and route
            if flight_utils.get_day(fid) != day:
                continue
            if flight_utils.get_origin(fid) != origin or flight_utils.get_destination(fid) != destination:
                continue
            
            flight_depart = flight_utils.get_departure(fid)
            flight_arrival = flight_utils.get_arrival(fid)
            
            # Check timing constraints
            if flight_depart < earliest_departure:
                continue
            if flight_arrival > latest_arrival:
                continue
            
            # Check if flight has empty slots (crew can deadhead as passenger/non-operating crew)
            # In this simplified model, we allow deadheading on any flight with any capacity
            # In a more complex model, you might track aircraft capacity or specific deadhead slots
            
            return fid
        
        return None