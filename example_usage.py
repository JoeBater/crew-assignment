"""
Example usage of the new modular crew optimizer system
"""

from crew_optimizer import CrewOptimizer
from destroy_operators import DestroyOperator
from repair_operators import RepairOperator
import numpy as np
import copy
from typing import Dict, List, Tuple


# Example: Custom destroy operator that targets specific aircraft types
class AircraftTypeDestroyOperator(DestroyOperator):
    """Destroys flights for a specific aircraft type"""
    
    def __init__(self, target_aircraft: str):
        self.target_aircraft = target_aircraft
    
    def destroy(self, assignment: Dict, crew_state: Dict, num_to_destroy: int, 
                flights_df, **kwargs) -> Tuple[Dict, Dict, List, List]:
        
        new_assignment = copy.deepcopy(assignment)
        new_crew_state = copy.deepcopy(crew_state)
        
        # Find flights with target aircraft type
        target_flights = flights_df[flights_df['type'] == self.target_aircraft]['id'].tolist()
        
        if len(target_flights) == 0:
            return new_assignment, new_crew_state, [], []
        
        # Select random flights to destroy from target aircraft
        num_to_destroy = min(num_to_destroy, len(target_flights))
        flights_to_destroy = np.random.choice(target_flights, num_to_destroy, replace=False)
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


# Example: Custom repair operator that prioritizes base-matching
class BaseMatchingRepairOperator(RepairOperator):
    """Repair operator that prioritizes crew from the same base as flight origin"""
    
    def repair(self, assignment: Dict, crew_state: Dict, crew_df, 
               flight_utils, crew_manager) -> Tuple[Dict, Dict]:
        
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


def main():
    """Example usage of the modular crew optimizer"""
    
    # Initialize optimizer
    optimizer = CrewOptimizer('data/flights.json', 'data/crew.json', verbose=True)
    
    # Create initial assignment
    print("Creating initial assignment...")
    assignment = optimizer.initial_assignment()
    crew_state = optimizer.initial_crew_state()
    
    print(f"Initial assignment has {len(assignment)} flights")
    
    # Add custom operators
    print("\nAdding custom operators...")
    optimizer.add_destroy_operator(AircraftTypeDestroyOperator("A320"))
    optimizer.add_repair_operator(BaseMatchingRepairOperator())
    
    print(f"Available destroy operators: {len(optimizer.destroy_operators)}")
    print(f"Available repair operators: {len(optimizer.repair_operators)}")
    
    # Destroy-repair cycle
    print("\nPerforming destroy-repair cycle...")
    
    # Destroy 5 flights using random operator selection
    d_assignment, d_crew_state, destroyed_flights, affected_crew = optimizer.destroy(5)
    print(f"Destroyed {len(destroyed_flights)} flights, affected {len(affected_crew)} crew")
    
    # Repair using location-aware operator (index 1)
    r_assignment, r_crew_state = optimizer.repair(d_assignment, d_crew_state, operator_index=1)
    print("Repair completed using location-aware operator")
    
    # Try custom aircraft destroy operator (index 2)
    print("\nTrying custom aircraft destroy operator...")
    d_assignment2, d_crew_state2, destroyed_flights2, affected_crew2 = optimizer.destroy(
        3, operator_index=2)
    print(f"Aircraft-specific destroy affected {len(destroyed_flights2)} flights")
    
    # Try custom base-matching repair operator (index 2)
    r_assignment2, r_crew_state2 = optimizer.repair(d_assignment2, d_crew_state2, operator_index=2)
    print("Repair completed using base-matching operator")
    
    # Demonstrate full LNS optimization
    print("\n" + "="*50)
    print("DEMONSTRATING FULL LNS OPTIMIZATION")
    print("="*50)
    
    # Reset to initial solution
    optimizer.assignment = assignment
    optimizer.crew_state = crew_state
    
    # Calculate initial cost
    initial_cost = optimizer._total_cost()
    print(f"Initial solution cost: {initial_cost:.0f}")
    
    # Run LNS optimization
    print("\nRunning LNS optimization with custom operators...")
    result = optimizer.optimize_lns(
        max_iterations=20,
        destroy_size_ratio=0.15,  # Destroy 15% of flights each iteration
        temperature=1000,
        cooling_rate=0.95
    )
    
    # Show results
    improvement = initial_cost - result['cost']
    print(f"\nOptimization Results:")
    print(f"  Final cost: {result['cost']:.0f}")
    print(f"  Improvement: {improvement:.0f} ({improvement/initial_cost*100:.1f}%)")
    print(f"  Iterations: {result['stats']['iterations']}")
    print(f"  Improvements found: {result['stats']['improvements']}")
    print(f"  Solutions accepted: {result['stats']['accepts']}")
    print(f"  Acceptance rate: {result['stats']['accepts']/result['stats']['iterations']:.1%}")
    
    # Show final diagnostics
    final_assignment_cost, final_assignment_diag = optimizer.compute_assignment_cost()
    final_crew_cost, final_crew_diag = optimizer.compute_crew_cost()
    
    print(f"\nFinal Solution Quality:")
    print(f"  Assignment cost: {final_assignment_cost:.0f}")
    print(f"  Crew cost: {final_crew_cost:.0f}")
    print(f"  Unassigned flights: {final_assignment_diag['unassigned_flights']}")
    print(f"  Base mismatches: {final_assignment_diag['base_mismatches']}")
    print(f"  Qualification mismatches: {final_assignment_diag['qualification_mismatches']}")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()