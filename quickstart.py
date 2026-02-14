"""
Quick Start Script for Wang's Neuroeconomic Model

This is a minimal example to get you started quickly.
"""

import numpy as np
import matplotlib.pyplot as plt
from wang_model import WangModel
from main_simulation import run_simulation
import os

# Create output directory
os.makedirs('outputs', exist_ok=True)


def quick_demo():
    """Quick demonstration of the model"""
    
    print("=" * 70)
    print("WANG'S NEUROECONOMIC MODEL - QUICK DEMO")
    print("=" * 70)
    
    # 1. Run a single trial
    print("\n1. Running single trial...")
    model = WangModel()
    result = model.run_trial(
        offA=10, offB=8,
        rangeA=(0, 15), rangeB=(0, 15),
        dj_HL=np.array([1.0, 1.0])
    )
    
    # Check the choice
    choice = "A" if result['nu1_wind'][-1] > result['nu2_wind'][-1] else "B"
    print(f"   Offers: 10A vs 8B → Model chose: {choice}")
    
    # 2. Run a small simulation
    print("\n2. Running simulation with 50 trials...")
    session = run_simulation(
        filesuffix='quickstart',
        session_mode='explicit',
        ntrials=50
    )
    
    # 3. Show results
    print("\n3. Results:")
    print(f"   Relative value: {session['behavData']['relvalue']:.3f}")
    print(f"   Choice variability: {session['behavData']['width']:.3f}")
    
    # 4. Create a simple plot
    print("\n4. Creating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    time = np.arange(len(result['nu1_wind'])) * 5 - 1000
    ax.plot(time, result['nu1_wind'], 'r-', linewidth=2, label='CJ1 (A)')
    ax.plot(time, result['nu2_wind'], 'b-', linewidth=2, label='CJ2 (B)')
    ax.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time from offer (ms)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title('Neural Activity During Decision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/quickstart_demo.png', dpi=150, bbox_inches='tight')
    print("   Plot saved: outputs/quickstart_demo.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run examples.py for more detailed examples")
    print("  - Run test_model.py to verify installation")
    print("  - Check README.md for full documentation")


if __name__ == "__main__":
    quick_demo()
