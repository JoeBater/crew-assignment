import pandas as pd
import numpy as np
import copy
import json
from datetime import datetime, timedelta, time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

# Import operators from separate modules
from destroy_operators import (
    DestroyOperator, RandomDestroyOperator, OverlapDestroyOperator,
    FatigueBasedDestroyOperator, AircraftTypeDestroyOperator
)
from repair_operators import (
    RepairOperator, RandomRepairOperator, LocationAwareRepairOperator,
    DeadheadingRepairOperator,
    BaseMatchingRepairOperator, QualificationFirstRepairOperator, GreedyCostRepairOperator
)


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
        return int(self.flights[self.flights['id'] == fid]['dur'].values[0])
    
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
        self.assignment: Optional[Dict[str, Dict[str, Any]]] = None
        self.crew_state: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self.verbose = verbose
        
        # Operators
        self.destroy_operators = [
            RandomDestroyOperator(),
            OverlapDestroyOperator(),
            FatigueBasedDestroyOperator(),
            AircraftTypeDestroyOperator(),
        ]
        
        self.repair_operators = [
            RandomRepairOperator(),
            LocationAwareRepairOperator(),
            BaseMatchingRepairOperator(),
            QualificationFirstRepairOperator(),
            GreedyCostRepairOperator(),
            DeadheadingRepairOperator(),
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


    def compute_assignment_cost(self) -> Tuple[float, Dict[str, Any]]:
        if self.assignment is None:
            raise ValueError("No assignment available. Call initial_assignment() first.")
            
        total_cost = 0
        diagnostics = {
            "unassigned_flights": 0,
            "base_mismatches": 0,
            "qualification_mismatches": 0,
            "deadheading_count": 0,
            "duplicate_roles": 0,
            "crew_usage": {},  # crew_id → count
        }

        for fid, roles in self.assignment.items():
            flight = self.flights[self.flights['id'] == fid].iloc[0]
            origin = flight['origin']
            aircraft = flight['type']

            for role in ['captain', 'first_officer']:
                crew_id = roles[role]
                if crew_id is None:
                    diagnostics["unassigned_flights"] += 1
                    total_cost += 1000
                    continue

                crew_member = self.crew[self.crew['id'] == crew_id].iloc[0]

                # Base mismatch
                if crew_member['base'] != origin:
                    diagnostics["base_mismatches"] += 1
                    total_cost += 500

                # Qualification mismatch
                if aircraft not in crew_member['qualified']:
                    diagnostics["qualification_mismatches"] += 1
                    total_cost += 1000

                # Track usage
                diagnostics["crew_usage"][crew_id] = diagnostics["crew_usage"].get(crew_id, 0) + 1

            # Role duplication
            if roles['captain'] == roles['first_officer']:
                diagnostics["duplicate_roles"] += 1
                total_cost += 2000

            # Deadheading cost
            dh_count = len(roles['dead_heading'])
            diagnostics["deadheading_count"] += dh_count
            total_cost += dh_count * 100

        return total_cost, diagnostics

    def compute_crew_cost(self) -> Tuple[float, Dict[str, Any]]:
        if self.crew_state is None:
            raise ValueError("No crew state available. Call initial_crew_state() first.")
            
        total_cost = 0
        diagnostics = {}

        for cid, trace in self.crew_state.items():
            trace = sorted(trace, key=lambda x: (x['day'], x['depart']))  # chronological
            crew_member = self.crew[self.crew['id'] == cid].iloc[0]
            base = crew_member['base']
            last_location = base
            last_arrival = time(0, 0, 0)
            day_flights = {}

            for entry in trace:
                day = entry['day']
                role = entry['role']
                origin = entry['origin']
                destination = entry['destination']
                depart = entry['depart']
                arrive = entry['arrive']

                # Track flights per day
                day_flights[day] = day_flights.get(day, 0) + 1

                # Location mismatch
                if origin != last_location:
                    total_cost += 500  # repositioning penalty

                # Overlap check (simplified)
                if depart < last_arrival:
                    total_cost += 2000  # impossible schedule

                # Deadheading cost
                if role == "dead_heading":
                    total_cost += 100

                last_location = destination
                last_arrival = arrive

            # Fatigue penalty
            for d, count in day_flights.items():
                if count > 3:
                    total_cost += (count - 3) * 300

            # Base return incentive
            if last_location == base:
                total_cost -= 200

            diagnostics[cid] = {
                "flights": len(trace),
                "deadheads": sum(1 for t in trace if t['role'] == "dead_heading"),
                "base_return": last_location == base,
                "fatigue_days": sum(1 for c in day_flights.values() if c > 3),
            }

        return total_cost, diagnostics
    
    def optimize_lns(self, max_iterations: int = 100, destroy_size_ratio: float = 0.1, 
                     temperature: float = 1000, cooling_rate: float = 0.95) -> Dict:
        """
        Large Neighborhood Search optimization algorithm
        
        Args:
            max_iterations: Maximum number of LNS iterations
            destroy_size_ratio: Fraction of flights to destroy (0.0-1.0)
            temperature: Initial temperature for simulated annealing acceptance
            cooling_rate: Temperature cooling rate per iteration
            
        Returns:
            Best solution found and optimization statistics
        """
        if self.assignment is None or self.crew_state is None:
            raise ValueError("Call initial_assignment() and initial_crew_state() first")
        
        # Track best solution
        best_assignment = copy.deepcopy(self.assignment)
        best_crew_state = copy.deepcopy(self.crew_state)
        best_cost = self._total_cost()
        
        # Current solution
        current_assignment = copy.deepcopy(self.assignment)
        current_crew_state = copy.deepcopy(self.crew_state)
        current_cost = best_cost
        
        # Optimization statistics
        stats = {
            'iterations': 0,
            'improvements': 0,
            'accepts': 0,
            'cost_history': [current_cost],
            'best_cost_history': [best_cost]
        }
        
        current_temp = temperature
        destroy_size = max(1, int(len(self.flights) * destroy_size_ratio))
        
        for iteration in range(max_iterations):
            if self.verbose:
                print(f"Iteration {iteration + 1}/{max_iterations}, Cost: {current_cost:.0f}, Best: {best_cost:.0f}, Temp: {current_temp:.1f}")
            
            # Temporarily set current solution for destroy/repair operations
            self._temp_update_solution(current_assignment, current_crew_state)
            
            # Destroy and repair
            new_assignment, new_crew_state, destroyed_flights, affected_crew = self.destroy(destroy_size)
            new_assignment, new_crew_state = self.repair(new_assignment, new_crew_state)
            
            # Evaluate new solution
            self._temp_update_solution(new_assignment, new_crew_state)
            new_cost = self._total_cost()
            
            # Acceptance criterion (simulated annealing)
            cost_delta = new_cost - current_cost
            accept = False
            
            if cost_delta <= 0:
                # Better solution - always accept
                accept = True
                stats['improvements'] += 1
            else:
                accept = False
            # elif current_temp > 0:
            #     # Worse solution - accept with probability
            #     probability = np.exp(-cost_delta / current_temp)
            #     accept = np.random.random() < probability
            
            if accept:
                current_assignment = new_assignment
                current_crew_state = new_crew_state
                current_cost = new_cost
                stats['accepts'] += 1
                
                # Update best solution if needed
                if new_cost < best_cost:
                    best_assignment = copy.deepcopy(new_assignment)
                    best_crew_state = copy.deepcopy(new_crew_state)
                    best_cost = new_cost
            
            # Cool down temperature
            current_temp *= cooling_rate
            
            # Update statistics
            stats['iterations'] += 1
            stats['cost_history'].append(current_cost)
            stats['best_cost_history'].append(best_cost)
        
        # Update optimizer state with best solution
        self.assignment = best_assignment
        self.crew_state = best_crew_state
        
        if self.verbose:
            print(f"\nOptimization complete!")
            print(f"Best cost: {best_cost:.0f}")
            print(f"Improvements: {stats['improvements']}")
            print(f"Acceptance rate: {stats['accepts']/stats['iterations']:.2%}")
        
        return {
            'assignment': best_assignment,
            'crew_state': best_crew_state,
            'cost': best_cost,
            'stats': stats
        }
    
    def _total_cost(self) -> float:
        """Compute total cost of current solution"""
        assignment_cost, _ = self.compute_assignment_cost()
        crew_cost, _ = self.compute_crew_cost()
        return assignment_cost + crew_cost
    
    def _temp_update_solution(self, assignment: Dict, crew_state: Dict):
        """Temporarily update solution for cost evaluation"""
        self.assignment = assignment
        self.crew_state = crew_state

    def write_assignment(self, assignmentfilename: str):
        """
        Write the current assignment to a JSON file
        
        Args:
            assignmentfilename: Path to the output JSON file
            
        Raises:
            ValueError: If no assignment exists
            IOError: If file cannot be written
        """
        if self.assignment is None:
            raise ValueError("No assignment to write. Call initial_assignment() first.")
        
        # Create output data structure with metadata
        output_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_flights': len(self.assignment),
                'num_crew': len(self.crew),
                'optimizer_version': '1.0.0'
            },
            'assignment': self.assignment,
            'crew_state': self.crew_state if self.crew_state is not None else {},
            'cost_summary': {}
        }
        
        # Add cost information if available
        try:
            assignment_cost, assignment_diag = self.compute_assignment_cost()
            crew_cost, crew_diag = self.compute_crew_cost()
            
            output_data['cost_summary'] = {
                'total_cost': assignment_cost + crew_cost,
                'assignment_cost': assignment_cost,
                'crew_cost': crew_cost,
                'diagnostics': {
                    'assignment': assignment_diag,
                    'crew': crew_diag
                }
            }
        except Exception as e:
            output_data['cost_summary'] = {
                'error': f"Could not compute costs: {str(e)}"
            }
        
        # Write to JSON file
        try:
            with open(assignmentfilename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            if self.verbose:
                print(f"✅ Assignment written to {assignmentfilename}")
                if self.assignment is not None:
                    print(f"   • Flights: {len(self.assignment)}")
                if 'total_cost' in output_data['cost_summary']:
                    print(f"   • Total cost: {output_data['cost_summary']['total_cost']:,.0f}")
                    
        except IOError as e:
            raise IOError(f"Failed to write assignment to {assignmentfilename}: {str(e)}")

    def read_assignment(self, assignmentfilename: str):
        """
        Read an assignment from a JSON file
        
        Args:
            assignmentfilename: Path to the input JSON file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            with open(assignmentfilename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate file format
            if 'assignment' not in data:
                raise ValueError("Invalid assignment file format: missing 'assignment' key")
            
            # Load assignment and crew state
            self.assignment = data['assignment']
            self.crew_state = data.get('crew_state', {})
            
            if self.verbose:
                print(f"✅ Assignment loaded from {assignmentfilename}")
                if self.assignment is not None:
                    print(f"   • Flights: {len(self.assignment)}")
                if 'metadata' in data and 'created_at' in data['metadata']:
                    print(f"   • Created: {data['metadata']['created_at']}")
                if 'cost_summary' in data and 'total_cost' in data['cost_summary']:
                    print(f"   • Saved cost: {data['cost_summary']['total_cost']:,.0f}")
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Assignment file not found: {assignmentfilename}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {assignmentfilename}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load assignment from {assignmentfilename}: {str(e)}")
