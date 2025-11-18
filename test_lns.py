#!/usr/bin/env python3
"""
Test the complete LNS optimization system
"""

import numpy as np
from crew_optimizer import CrewOptimizer

def main():
    print("Testing LNS Crew Assignment Optimizer")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = CrewOptimizer(
        flights_file="data/flights.json",
        crew_file="data/crew.json",
        verbose=True
    )
    
    print(f"Loaded {len(optimizer.flights)} flights and {len(optimizer.crew)} crew members")
    
    # Create initial solution
    print("\n1. Creating initial assignment...")
    assignment = optimizer.initial_assignment()
    crew_state = optimizer.initial_crew_state()
    
    # Evaluate initial solution
    assignment_cost, assignment_diag = optimizer.compute_assignment_cost()
    crew_cost, crew_diag = optimizer.compute_crew_cost()
    initial_cost = assignment_cost + crew_cost
    
    print(f"\nInitial Solution:")
    print(f"  Assignment cost: {assignment_cost:.0f}")
    print(f"  Crew cost: {crew_cost:.0f}")
    print(f"  Total cost: {initial_cost:.0f}")
    print(f"  Unassigned flights: {assignment_diag['unassigned_flights']}")
    print(f"  Base mismatches: {assignment_diag['base_mismatches']}")
    
    # Run LNS optimization
    print("\n2. Running LNS optimization...")
    result = optimizer.optimize_lns(
        max_iterations=1000,
        destroy_size_ratio=0.15,  # Destroy 15% of flights each iteration
        temperature=2000,
        cooling_rate=0.98
    )
    
    # Show final results
    print(f"\nFinal Solution:")
    print(f"  Total cost: {result['cost']:.0f}")
    print(f"  Improvement: {initial_cost - result['cost']:.0f} ({((initial_cost - result['cost'])/initial_cost)*100:.1f}%)")
    print(f"  Iterations: {result['stats']['iterations']}")
    print(f"  Improvements found: {result['stats']['improvements']}")
    print(f"  Acceptance rate: {result['stats']['accepts']/result['stats']['iterations']:.1%}")
    
    # Final diagnostics
    final_assignment_cost, final_assignment_diag = optimizer.compute_assignment_cost()
    final_crew_cost, final_crew_diag = optimizer.compute_crew_cost()
    
    print(f"\nFinal Diagnostics:")
    print(f"  Assignment cost: {final_assignment_cost:.0f}")
    print(f"  Crew cost: {final_crew_cost:.0f}")
    print(f"  Unassigned flights: {final_assignment_diag['unassigned_flights']}")
    print(f"  Base mismatches: {final_assignment_diag['base_mismatches']}")
    print(f"  Qualification mismatches: {final_assignment_diag['qualification_mismatches']}")
    print(f"  Deadheading assignments: {final_assignment_diag['deadheading_count']}")
    
    # Show cost progression
    print(f"\nCost History (first 10 and last 10):")
    cost_history = result['stats']['cost_history']
    best_history = result['stats']['best_cost_history']
    
    print("  Iteration | Current Cost | Best Cost")
    print("  --------- | ------------ | ---------")
    
    # First 10 iterations
    for i in range(min(10, len(cost_history))):
        print(f"  {i:8d} | {cost_history[i]:11.0f} | {best_history[i]:8.0f}")
    
    if len(cost_history) > 20:
        print("      ...   |      ...     |   ...")
        
        # Last 10 iterations  
        for i in range(max(10, len(cost_history) - 10), len(cost_history)):
            print(f"  {i:8d} | {cost_history[i]:11.0f} | {best_history[i]:8.0f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()