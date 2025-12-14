"""
Destroy operators for Large Neighborhood Search in crew assignment optimization.

This module contains various destroy operators that remove assignments from the current solution
to create neighborhoods for the LNS algorithm.
"""

import copy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


class DestroyOperator(ABC):
    """Abstract base class for destroy operators"""
    
    @abstractmethod
    def destroy(self, assignment: Dict, crew_state: Dict, num_to_destroy: int, 
                flights_df: pd.DataFrame, **kwargs) -> Tuple[Dict, Dict, List, List]:
        """
        Destroy assignments and return modified assignment, crew_state, 
        destroyed flights, and affected crew
        
        Args:
            assignment: Current flight assignments
            crew_state: Current crew scheduling state
            num_to_destroy: Number of flights to destroy
            flights_df: DataFrame containing flight information
            **kwargs: Additional parameters for specific operators
            
        Returns:
            Tuple of (new_assignment, new_crew_state, destroyed_flights, affected_crew)
        """
        pass


class RandomDestroyOperator(DestroyOperator):
    """Randomly destroys flights"""
    
    def destroy(self, assignment: Dict, crew_state: Dict, num_to_destroy: int, 
                flights_df: pd.DataFrame, **kwargs) -> Tuple[Dict, Dict, List, List]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        flights_to_destroy = np.random.choice(flights_df['pairing_id'].tolist(), num_to_destroy, replace=False)
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
            sorted_flights = sorted(crew_flights, key=lambda x: (x['day'], x['start_utc']))
            
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
                flight1['start_utc'] < flight2['end_utc'] and 
                flight2['start_utc'] < flight1['end_utc'])


class FatigueBasedDestroyOperator(DestroyOperator):
    """Destroys flights assigned to overworked crew members"""
    
    def __init__(self, fatigue_threshold: int = 3):
        """
        Initialize fatigue-based destroy operator
        
        Args:
            fatigue_threshold: Maximum number of flights before considering crew overworked
        """
        self.fatigue_threshold = fatigue_threshold
    
    def destroy(self, assignment: Dict, crew_state: Dict, num_to_destroy: int, 
                flights_df: pd.DataFrame, **kwargs) -> Tuple[Dict, Dict, List, List]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        # Find overworked crew members
        overworked_crew = []
        for crew_id, flights in crew_state.items():
            if len(flights) > self.fatigue_threshold:
                overworked_crew.append((crew_id, len(flights)))
        
        # Sort by workload (most overworked first)
        overworked_crew.sort(key=lambda x: x[1], reverse=True)
        
        flights_to_destroy = []
        affected_crew = set()
        
        # Target flights from overworked crew
        for crew_id, workload in overworked_crew:
            if len(flights_to_destroy) >= num_to_destroy:
                break
                
            crew_flights = [f['flight'] for f in crew_state[crew_id]]
            # Remove some flights from this overworked crew member
            flights_to_remove = min(workload - self.fatigue_threshold, 
                                   num_to_destroy - len(flights_to_destroy))
            
            selected_flights = np.random.choice(crew_flights, flights_to_remove, replace=False)
            flights_to_destroy.extend(selected_flights)
            affected_crew.add(crew_id)
        
        # If not enough flights from overworked crew, fall back to random
        if len(flights_to_destroy) < num_to_destroy:
            remaining_flights = [fid for fid in assignment.keys() if fid not in flights_to_destroy]
            if remaining_flights:
                additional_flights = np.random.choice(
                    remaining_flights,
                    min(num_to_destroy - len(flights_to_destroy), len(remaining_flights)),
                    replace=False
                )
                flights_to_destroy.extend(additional_flights)
        
        # Remove selected flights
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
        
        return new_assignment, new_crew_state, flights_to_destroy, list(affected_crew)

