"""
Example Usage of Wang's Neuroeconomic Model

This script demonstrates various ways to use the model implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from wang_model import WangModel
from session_params import session_params
from main_simulation import run_simulation
from plotting import plot_session_summary, plot_choice_pattern, plot_neural_trajectories


# Create output directory in current working directory
OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def example_1_single_trial():
    """
    Example 1: Run a single trial and examine the results
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Trial")
    print("=" * 70)

    # Initialize model
    model = WangModel()

    # Define trial parameters
    offA = 12  # Offer A: 12 units
    offB = 8  # Offer B: 8 units
    rangeA = (0, 15)
    rangeB = (0, 15)
    dj_HL = np.array([1.0, 1.0])  # Hebbian learning weights

    # Run trial
    print(f"\nRunning trial with offers: A={offA}, B={offB}")
    result = model.run_trial(offA, offB, rangeA, rangeB, dj_HL)

    # Extract results
    nu1 = result["nu1_wind"]  # CJ1 cell (chosen juice A)
    nu2 = result["nu2_wind"]  # CJ2 cell (chosen juice B)
    nuI = result["nuI_wind"]  # CV cell (chosen value)

    # Determine choice
    final_nu1 = nu1[-1]
    final_nu2 = nu2[-1]
    choice = "A" if final_nu1 > final_nu2 else "B"

    print(f"Final activity - CJ1: {final_nu1:.2f} Hz, CJ2: {final_nu2:.2f} Hz")
    print(f"Model chose: option {choice}")

    # Plot trajectory
    time = np.arange(len(nu1)) * 5 - 1000  # Time in ms from offer onset

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot CJ cells
    axes[0].plot(time, nu1, "r-", linewidth=2, label="CJ1 (Chosen Juice A)")
    axes[0].plot(time, nu2, "b-", linewidth=2, label="CJ2 (Chosen Juice B)")
    axes[0].axvline(0, color="k", linestyle=":", alpha=0.5, label="Offer onset")
    axes[0].set_xlabel("Time from offer onset (ms)")
    axes[0].set_ylabel("Firing rate (Hz)")
    axes[0].set_title(f"Choice Selectivity Cells (Chose {choice})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot CV cells
    axes[1].plot(time, nuI, "orange", linewidth=2, label="CV (Chosen Value)")
    axes[1].axvline(0, color="k", linestyle=":", alpha=0.5, label="Offer onset")
    axes[1].set_xlabel("Time from offer onset (ms)")
    axes[1].set_ylabel("Firing rate (Hz)")
    axes[1].set_title("Chosen Value Cells")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "example_1_single_trial.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {save_path}")
    plt.close()


def example_2_compare_offers():
    """
    Example 2: Compare model behavior across different offer types
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Comparing Different Offers")
    print("=" * 70)

    model = WangModel()
    rangeA = (0, 15)
    rangeB = (0, 15)
    dj_HL = np.array([1.0, 1.0])

    # Test different offer types
    offers = [
        (10, 5, "Easy (A preferred)"),
        (5, 10, "Easy (B preferred)"),
        (10, 10, "Equal"),
        (12, 10, "Difficult"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (offA, offB, description) in enumerate(offers):
        print(f"\nTesting: {description} (A={offA}, B={offB})")

        result = model.run_trial(offA, offB, rangeA, rangeB, dj_HL)

        nu1 = result["nu1_wind"]
        nu2 = result["nu2_wind"]
        time = np.arange(len(nu1)) * 5 - 1000

        choice = "A" if nu1[-1] > nu2[-1] else "B"
        print(f"  → Model chose: {choice}")

        axes[idx].plot(time, nu1, "r-", linewidth=2, label="CJ1 (A)")
        axes[idx].plot(time, nu2, "b-", linewidth=2, label="CJ2 (B)")
        axes[idx].axvline(0, color="k", linestyle=":", alpha=0.5)
        axes[idx].set_xlabel("Time (ms)")
        axes[idx].set_ylabel("Firing rate (Hz)")
        axes[idx].set_title(f"{description}\nChose: {choice}")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "example_2_compare_offers.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {save_path}")
    plt.close()


def example_3_full_session():
    """
    Example 3: Run a full session and analyze behavior
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Full Session (50 trials)")
    print("=" * 70)

    # Run simulation
    session_summary = run_simulation(
        filesuffix="example", session_mode="explicit", ntrials=300
    )

    # Create all plots
    print("\nCreating summary plots...")
    plots_dir = os.path.join(OUTPUT_DIR, "example_3_plots")
    plot_session_summary(session_summary, save_dir=plots_dir)
    print(f"Plots saved to: {plots_dir}/")

    # Display behavioral metrics
    relvalue = session_summary["behavData"]["relvalue"]
    width = session_summary["behavData"]["width"]

    print("\nBehavioral Analysis:")
    print(f"  Relative value: {relvalue:.3f}")
    print(f"  Choice variability: {width:.3f}")

    # Analyze choice consistency
    table01 = session_summary["behavData"]["table01"]
    print(f"\n  Number of offer types: {len(table01)}")

    return session_summary


def example_4_custom_analysis():
    """
    Example 4: Custom analysis of tuning properties
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Tuning Analysis")
    print("=" * 70)

    # Load or run session
    session_file = "sessionSumm_example.pkl"
    try:
        with open(session_file, "rb") as f:
            session_summary = pickle.load(f)
        print("Loaded existing session")
    except:
        print("Running new session...")
        session_summary = run_simulation("example", "explicit", 30)

    # Analyze OV cell tuning
    tuning_OV1 = session_summary["tuning"]["rOV1"]
    trial_types = tuning_OV1["bytrialtype"]

    if len(trial_types) > 0:
        # Extract offer values and firing rates
        offA_values = trial_types[:, 0]
        offB_values = trial_types[:, 1]
        firing_rates = trial_types[:, 3]

        # Create 2D tuning map
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot as scatter with size indicating firing rate
        scatter = ax.scatter(
            offB_values,
            offA_values,
            c=firing_rates,
            s=200,
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
        )

        ax.set_xlabel("Offer B", fontsize=12)
        ax.set_ylabel("Offer A", fontsize=12)
        ax.set_title(
            "OV1 Cell Tuning Map (Offer Value A)", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Firing Rate (Hz)", fontsize=12)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "example_4_tuning_map.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Tuning map saved: {save_path}")
        plt.close()

    # Analyze choice-related activity
    tuning_r1 = session_summary["tuning"]["r1"]
    bytrial = tuning_r1["bytrial"]

    if len(bytrial) > 0:
        chose_A = bytrial[bytrial[:, 3] == 1, 4]
        chose_B = bytrial[bytrial[:, 3] == -1, 4]

        print(f"\nCJ1 Cell Activity:")
        print(f"  When A chosen: {np.mean(chose_A):.2f} ± {np.std(chose_A):.2f} Hz")
        print(f"  When B chosen: {np.mean(chose_B):.2f} ± {np.std(chose_B):.2f} Hz")
        print(f"  Selectivity index: {(np.mean(chose_A) - np.mean(chose_B)):.2f} Hz")


def run_all_examples():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("WANG'S NEUROECONOMIC MODEL - EXAMPLES")
    print("=" * 70)
    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")

    # Example 1: Single trial
    example_1_single_trial()

    # Example 2: Compare offers
    example_2_compare_offers()

    # Example 3: Full session
    session_summary = example_3_full_session()

    # Example 4: Custom analysis
    example_4_custom_analysis()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files in: {os.path.abspath(OUTPUT_DIR)}")
    print("  - example_1_single_trial.png")
    print("  - example_2_compare_offers.png")
    print("  - example_3_plots/ (directory with multiple plots)")
    print("  - example_4_tuning_map.png")
    print("  - sessionSumm_example.pkl (saved data)")


if __name__ == "__main__":
    run_all_examples()
