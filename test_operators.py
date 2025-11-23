#!/usr/bin/env python3
"""
Test the refactored operator modules
"""

import numpy as np
from crew_optimizer import CrewOptimizer
from destroy_operators import (
    FatigueBasedDestroyOperator, AircraftTypeDestroyOperator
)
from repair_operators import (
    BaseMatchingRepairOperator, QualificationFirstRepairOperator, GreedyCostRepairOperator
)

def test_refactored_operators():
    """Test the new modular operator system"""
    print("🧪 Testing Refactored Operator System")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = CrewOptimizer("data/flights.json", "data/crew.json", verbose=False)
    
    print(f"✅ Optimizer loaded with {len(optimizer.flights)} flights and {len(optimizer.crew)} crew")
    
    # Test that default operators work
    assignment = optimizer.initial_assignment()
    crew_state = optimizer.initial_crew_state()
    
    print(f"✅ Initial assignment created")
    
    # Test default operators
    print(f"\n🔧 Testing Default Operators:")
    print(f"   • Destroy operators: {len(optimizer.destroy_operators)}")
    print(f"   • Repair operators: {len(optimizer.repair_operators)}")
    
    for i, op in enumerate(optimizer.destroy_operators):
        print(f"     - Destroy {i}: {op.__class__.__name__}")
        
    for i, op in enumerate(optimizer.repair_operators):
        print(f"     - Repair {i}: {op.__class__.__name__}")
    
    # Add new operators from the separate modules
    print(f"\n➕ Adding New Operators:")
    
    # Add new destroy operators
    fatigue_destroyer = FatigueBasedDestroyOperator(fatigue_threshold=4)
    aircraft_destroyer = AircraftTypeDestroyOperator(target_aircraft="A320")
    
    optimizer.add_destroy_operator(fatigue_destroyer)
    optimizer.add_destroy_operator(aircraft_destroyer)
    
    # Add new repair operators
    base_repairer = BaseMatchingRepairOperator()
    qualification_repairer = QualificationFirstRepairOperator()
    greedy_repairer = GreedyCostRepairOperator()
    
    optimizer.add_repair_operator(base_repairer)
    optimizer.add_repair_operator(qualification_repairer)
    optimizer.add_repair_operator(greedy_repairer)
    
    print(f"   • Total destroy operators: {len(optimizer.destroy_operators)}")
    print(f"   • Total repair operators: {len(optimizer.repair_operators)}")
    
    # Test each new operator
    print(f"\n🔥 Testing New Operators:")
    
    # Test fatigue-based destroy
    damaged_assignment, damaged_crew_state, destroyed_flights, affected_crew = optimizer.destroy(5, operator_index=2)
    print(f"   • Fatigue destroyer: removed {len(destroyed_flights)} flights, affected {len(affected_crew)} crew")
    
    # Test aircraft type destroy
    damaged_assignment2, damaged_crew_state2, destroyed_flights2, affected_crew2 = optimizer.destroy(3, operator_index=3)
    print(f"   • Aircraft destroyer: removed {len(destroyed_flights2)} flights")
    
    # Test base matching repair
    repaired_assignment, repaired_crew_state = optimizer.repair(damaged_assignment, damaged_crew_state, operator_index=2)
    print(f"   • Base matching repair: completed")
    
    # Test qualification first repair
    repaired_assignment2, repaired_crew_state2 = optimizer.repair(damaged_assignment2, damaged_crew_state2, operator_index=3)
    print(f"   • Qualification first repair: completed")
    
    # Test greedy cost repair
    repaired_assignment3, repaired_crew_state3 = optimizer.repair(damaged_assignment2, damaged_crew_state2, operator_index=4)
    print(f"   • Greedy cost repair: completed")
    
    print(f"\n✅ All operator tests passed!")
    
    # Quick LNS test with new operators
    print(f"\n🚀 Testing LNS with Extended Operator Set:")
    result = optimizer.optimize_lns(
        max_iterations=10,
        destroy_size_ratio=0.1,
        temperature=1000,
        cooling_rate=0.95
    )
    
    print(f"   • LNS completed: cost {result['cost']:,.0f}")
    print(f"   • Used {len(optimizer.destroy_operators)} destroy and {len(optimizer.repair_operators)} repair operators")
    
    print(f"\n🎉 Refactored operator system working perfectly!")

if __name__ == "__main__":
    np.random.seed(42)
    test_refactored_operators()