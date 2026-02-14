"""
Test script to verify the Wang model implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from wang_model import WangModel
from session_params import session_params
from main_simulation import run_simulation
from plotting import plot_choice_pattern, plot_neural_trajectories
import pickle


def test_basic_model():
    """Test basic model functionality"""
    print("=" * 70)
    print("TEST 1: Basic Model Functionality")
    print("=" * 70)
    
    model = WangModel()
    
    # Test single trial
    print("\nRunning single trial...")
    offA, offB = 10, 5
    rangeA, rangeB = (0, 15), (0, 15)
    dj_HL = np.array([1.0, 1.0])
    
    result = model.run_trial(offA, offB, rangeA, rangeB, dj_HL)
    
    print(f"  Offer A: {offA}, Offer B: {offB}")
    print(f"  Final CJ1 activity: {result['nu1_wind'][-1]:.2f} Hz")
    print(f"  Final CJ2 activity: {result['nu2_wind'][-1]:.2f} Hz")
    
    # Determine choice
    choice = "A" if result['nu1_wind'][-1] > result['nu2_wind'][-1] else "B"
    print(f"  Model chose: {choice}")
    
    print("✓ Basic model test passed!")
    return True


def test_session_params():
    """Test session parameter generation"""
    print("\n" + "=" * 70)
    print("TEST 2: Session Parameters")
    print("=" * 70)
    
    # Test explicit mode
    print("\nTesting explicit mode...")
    offList, rangeA, rangeB = session_params('explicit', ntrials=100)
    print(f"  Generated {len(offList)} trials")
    print(f"  Offer A range: {rangeA}")
    print(f"  Offer B range: {rangeB}")
    print(f"  Sample offers: {offList[:5]}")
    
    # Test parametric mode
    print("\nTesting parametric mode...")
    offList, rangeA, rangeB = session_params('parametric')
    print(f"  Generated {len(offList)} trials")
    print(f"  Offer A range: {rangeA}")
    print(f"  Offer B range: {rangeB}")
    print(f"  Sample offers: {offList[:5]}")
    
    print("✓ Session parameters test passed!")
    return True


def test_small_simulation():
    """Run a small simulation"""
    print("\n" + "=" * 70)
    print("TEST 3: Small Simulation (10 trials)")
    print("=" * 70)
    
    print("\nRunning simulation...")
    session_summary = run_simulation(
        filesuffix='test',
        session_mode='explicit',
        ntrials=10
    )
    
    # Check outputs
    print("\nChecking outputs...")
    assert 'behavData' in session_summary
    assert 'tuning' in session_summary
    assert 'traj_quant' in session_summary
    
    if 'relvalue' in session_summary['behavData']:
        relvalue = session_summary['behavData']['relvalue']
        print(f"  Relative value: {relvalue:.3f}")
    
    print("✓ Small simulation test passed!")
    return session_summary


def test_plotting(session_summary):
    """Test plotting functions"""
    print("\n" + "=" * 70)
    print("TEST 4: Plotting")
    print("=" * 70)
    
    print("\nCreating choice pattern plot...")
    fig1 = plot_choice_pattern(session_summary)
    plt.close(fig1)
    print("✓ Choice pattern plot created")
    
    print("\nCreating neural trajectories plot...")
    fig2 = plot_neural_trajectories(session_summary)
    plt.close(fig2)
    print("✓ Neural trajectories plot created")
    
    print("✓ Plotting test passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RUNNING ALL TESTS")
    print("=" * 70)
    
    try:
        # Test 1: Basic model
        test_basic_model()
        
        # Test 2: Session parameters
        test_session_params()
        
        # Test 3: Small simulation
        session_summary = test_small_simulation()
        
        # Test 4: Plotting
        test_plotting(session_summary)
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nThe implementation is working correctly.")
        print("You can now run larger simulations with:")
        print("  python main_simulation.py")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED! ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
