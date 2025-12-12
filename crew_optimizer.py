import pandas as pd
import numpy as np
import copy
import json
from datetime import datetime, timedelta, time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

from flight_utils import FlightUtils
from crew_state_manager import CrewStateManager

# Import operators from separate modules
from destroy_operators import (
    DestroyOperator, RandomDestroyOperator, OverlapDestroyOperator,
    FatigueBasedDestroyOperator, AircraftTypeDestroyOperator
)
from repair_operators import (
    RepairOperator, RandomRepairOperator, LocationAwareRepairOperator,
    QualificationFirstRepairOperator, GreedyCostRepairOperator
)


class CrewOptimizer:
    """Main optimizer class that orchestrates destroy and repair operations"""
    
    def __init__(self, base: str, flights_file: str, crew_file: str, verbose: bool = False):
        # Load data
        self.base = base
        self.flights = pd.read_csv(flights_file)

        # Normalise common column names so other modules can rely on canonical names
        # Drop obviously malformed rows if these common datetime columns exist
        drop_subset = [c for c in ['duty_id', 'duty_start_datetime_utc', 'duty_end_datetime_utc'] if c in self.flights.columns]
        if drop_subset:
            self.flights = self.flights.dropna(subset=drop_subset)

        # Normalise some common datetime/time columns 
        self.flights['start_utc'] = pd.to_datetime(self.flights['start_utc']).dt.time
        self.flights['end_utc'] = pd.to_datetime(self.flights['end_utc']).dt.time
        self.flights['start date'] = pd.to_datetime(self.flights['start date']).dt.date

        # Filter by base using a flexible column name (expect 'base' commonly)
        base_candidates = ['base', 'home_base', 'duty_start_location']
        base_col = next((c for c in base_candidates if c in self.flights.columns), None)
        if base_col is not None:
            self.flights = self.flights[self.flights[base_col] == self.base]
        else:
            if verbose:
                print("Warning: no base-like column found in flights; skipping base filter")

        # Load crew and normalise column names
        self.crew = pd.read_csv(crew_file)

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
            for fid in self.flights['pairing_id']
        }

        for _, flight in self.flights.iterrows():
            fid = flight['pairing_id']
            
            # Find eligible crew
            eligible_captains = self.crew[
                (self.crew['crew_role'] == 'captain') 
            ]['crew_id'].values
            
            eligible_first_officers = self.crew[
                (self.crew['crew_role'] == 'first_officer') 
            ]['crew_id'].values
            
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
            "deadheading_count": 0,
            "duplicate_roles": 0,
            "crew_usage": {},  # crew_id → count
        }

        for fid, roles in self.assignment.items():
            flight = self.flights[self.flights['pairing_id'] == fid].iloc[0]
            origin = flight['origin']
            aircraft = flight['type']

            for role in ['captain', 'first_officer']:
                crew_id = roles[role]
                if crew_id is None:
                    diagnostics["unassigned_flights"] += 1
                    total_cost += 1000
                    continue

                crew_member = self.crew[self.crew['id'] == crew_id].iloc[0]

                

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

            

            diagnostics[cid] = {
                "flights": len(trace),
                "deadheads": sum(1 for t in trace if t['role'] == "dead_heading"),
                
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
