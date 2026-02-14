"""
Plotting Module for Wang's Neuroeconomic Model

Visualizes choice patterns, tuning curves, and neural trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from scipy.optimize import curve_fit
import pickle


def plot_choice_pattern(session_summary, save_path=None):
    """
    Plot choice pattern (psychometric curve) - CLEAN VERSION

    Excludes forced-choice trials (where offA=0 or offB=0)
    Focuses on the psychometric curve with readable x-axis

    Parameters:
    -----------
    session_summary : dict
        Session summary dictionary
    save_path : str, optional
        Path to save figure
    """
    table01 = session_summary["behavData"]["table01"]

    # Only include non-forced choices (both offers > 0)
    table1mod = table01[(table01[:, 0] > 0) & (table01[:, 1] > 0)]

    # Prepare data
    xx = np.log(table1mod[:, 1] / table1mod[:, 0])  # log(B/A)
    yy = table1mod[:, 2]  # % chose B

    # Fit sigmoid
    def sigmoid(x, x0, w):
        return norm.cdf(x, x0, w)

    try:
        popt, _ = curve_fit(sigmoid, xx, yy, p0=[0, 1])
        x0, w = popt
        relvalue = np.exp(x0)
        width = np.exp(w)

        # Plot fitted sigmoid
        x_fit = np.linspace(min(xx) - 0.5, max(xx) + 0.5, 200)
        y_fit = sigmoid(x_fit, x0, w)
    except:
        relvalue = 1.0
        width = 1.0
        x_fit = None
        y_fit = None

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot fitted sigmoid
    if x_fit is not None:
        ax.plot(
            x_fit, y_fit * 100, "k-", linewidth=2.5, label="Fitted sigmoid", zorder=1
        )

    # Plot choice data
    ax.plot(
        xx,
        yy * 100,
        "ko",
        markersize=10,
        markerfacecolor="black",
        markeredgewidth=1.5,
        label="Observed choices",
        zorder=2,
    )

    # Prepare x-axis labels
    unique_xx = np.unique(xx)
    xlab = []

    for x in unique_xx:
        idx = np.where(xx == x)[0][0]
        xlab.append(f"{int(table1mod[idx, 1])}:{int(table1mod[idx, 0])}")

    # Set x-axis
    ax.set_xticks(unique_xx)
    ax.set_xticklabels(xlab, rotation=45, ha="right", fontsize=10)
    ax.set_xlim([min(unique_xx) - 0.3, max(unique_xx) + 0.3])

    # Set y-axis
    ax.set_ylim([-5, 105])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% Choice B", fontsize=13, fontweight="bold")
    ax.set_xlabel("Offer (B:A)", fontsize=13, fontweight="bold")

    # Add text with fit parameters
    if x_fit is not None:
        ax.text(
            0.05,
            0.95,
            f"μ = {relvalue:.3f}\nσ = {width:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

    # Add reference lines
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1, zorder=0)
    ax.axhline(25, color="gray", linestyle=":", alpha=0.3, linewidth=0.5, zorder=0)
    ax.axhline(75, color="gray", linestyle=":", alpha=0.3, linewidth=0.5, zorder=0)

    ax.set_title(
        "Choice Pattern (Non-Forced Trials Only)",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    return fig


def plot_choice_pattern2(session_summary, save_path=None):
    """
    Plot choice pattern with equidistant x-axis ticks regardless of numerical value.
    """
    table01 = session_summary["behavData"]["table01"]

    # --- 1. PREPARE DATA ---
    # Separate non-forced and forced choices
    # table1mod: [A, B, Choice]
    table1mod = table01[(table01[:, 0] > 0) & (table01[:, 1] > 0)]
    forcedAtab = table01[(table01[:, 0] > 0) & (table01[:, 1] == 0)]
    forcedBtab = table01[(table01[:, 0] == 0) & (table01[:, 1] > 0)]

    nfA = len(forcedAtab)
    nfB = len(forcedBtab)

    # Calculate real log-ratios for the choice trials
    # We use these for the fitting logic
    xx_real = np.log(table1mod[:, 1] / table1mod[:, 0])
    yy = table1mod[:, 2]

    # --- 2. DEFINE THE 'REAL' X-AXIS COORDINATES FOR FORCED TRIALS ---
    # We need artificial log-values for forced choices to maintain the sort order
    # (Forced A < Lowest Ratio ... Highest Ratio < Forced B)

    unique_xx_real = np.unique(xx_real)

    # Generate artificial coordinates for forced A (decreasing to the left)
    if nfA > 0:
        dist = np.log(2)  # Arbitrary spacing step for calculation
        start_A = min(unique_xx_real) if len(unique_xx_real) > 0 else 0
        xx_forcedA_real = start_A - dist * np.arange(nfA, 0, -1)
    else:
        xx_forcedA_real = np.array([])

    # Generate artificial coordinates for forced B (increasing to the right)
    if nfB > 0:
        dist = np.log(2)
        end_B = max(unique_xx_real) if len(unique_xx_real) > 0 else 0
        xx_forcedB_real = end_B + dist * np.arange(1, nfB + 1)
    else:
        xx_forcedB_real = np.array([])

    # Combine all real x-coordinates into one sorted master list
    # This represents the "True" mathematical order
    all_x_real = np.concatenate([xx_forcedA_real, unique_xx_real, xx_forcedB_real])

    # --- 3. CREATE THE EQUIDISTANT MAPPING ---
    # We map every real value to an integer: 0, 1, 2, 3...
    # This creates the "Equal Spacing" visual.
    all_x_indices = np.arange(len(all_x_real))

    # Create a helper to map real values to indices for the raw data points
    # We use interpolation to find the index, though for exact points it returns integers
    xx_mapped_indices = np.interp(xx_real, all_x_real, all_x_indices)

    # --- 4. FIT THE SIGMOID (SCIENTIFIC CALCULATION) ---
    def sigmoid(x, x0, w):
        return norm.cdf(x, x0, w)

    popt = None
    try:
        # Fit on the REAL log values, not the indices
        popt, _ = curve_fit(sigmoid, xx_real, yy, p0=[0, 1])
        x0, w = popt
        relvalue = np.exp(x0)
        width = np.exp(w)
    except:
        relvalue = 1.0
        width = 1.0
        x0, w = 0, 1  # Fallback

    # --- 5. PLOTTING ---
    fig, ax = plt.subplots(figsize=(12, 6))  # Slightly wider for labels

    # A. Plot the Fitted Curve (Warped)
    if popt is not None:
        # 1. Generate dense points in REAL space (to get the S-shape)
        x_fit_real = np.linspace(min(all_x_real), max(all_x_real), 500)
        y_fit = sigmoid(x_fit_real, x0, w)

        # 2. Map these dense real points to the INTEGER index space
        # This "squeezes" the curve into the equal-spacing layout
        x_fit_indices = np.interp(x_fit_real, all_x_real, all_x_indices)

        ax.plot(x_fit_indices, y_fit, "k-", linewidth=2, label="Fitted sigmoid")

    # B. Plot Choice Data
    # Plot using the mapped indices
    ax.plot(
        xx_mapped_indices,
        yy,
        "ko",
        markersize=8,
        markerfacecolor="black",
        label="Choices",
    )

    # C. Plot Forced Choices
    # They are simply the first nfA indices and last nfB indices
    if nfA > 0:
        ax.plot(
            all_x_indices[:nfA],
            np.zeros(nfA),
            "ko",
            markersize=8,
            markerfacecolor="black",
        )

    if nfB > 0:
        ax.plot(
            all_x_indices[-nfB:],
            np.ones(nfB),
            "ko",
            markersize=8,
            markerfacecolor="black",
        )

    # --- 6. LABELS AND TICKS ---
    # Construct labels exactly as before
    xlab = []

    # Forced A Labels
    for i in range(nfA):
        xlab.append(
            f"{int(forcedAtab[nfA - i - 1, 1])}:{int(forcedAtab[nfA - i - 1, 0])}"
        )

    # Choice Labels
    for x in unique_xx_real:
        # Find one instance in original table to get the label
        idx = np.where(xx_real == x)[0][0]
        xlab.append(f"{int(table1mod[idx, 1])}:{int(table1mod[idx, 0])}")

    # Forced B Labels
    for i in range(nfB):
        xlab.append(
            f"{int(forcedBtab[nfB - i - 1, 1])}:{int(forcedBtab[nfB - i - 1, 0])}"
        )

    # Apply the equidistant ticks
    ax.set_xticks(all_x_indices)
    ax.set_xticklabels(xlab, rotation=45, ha="right")

    # Add padding to x-axis
    ax.set_xlim([min(all_x_indices) - 0.5, max(all_x_indices) + 0.5])

    # Standard Y-axis setup
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("% Choice B", fontsize=12)
    ax.set_xlabel("Offer (B:A)", fontsize=12)

    # Stats Box
    if popt is not None:
        ax.text(
            0.05,
            0.95,
            f"μ = {relvalue:.3f}\nσ = {width:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_title("Choice Pattern", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_choice_pattern_orig(session_summary, save_path=None):
    """
    Plot choice pattern (psychometric curve)

    Parameters:
    -----------
    session_summary : dict
        Session summary dictionary
    save_path : str, optional
        Path to save figure
    """
    table01 = session_summary["behavData"]["table01"]

    # Separate non-forced and forced choices
    table1mod = table01[(table01[:, 0] > 0) & (table01[:, 1] > 0)]
    forcedAtab = table01[(table01[:, 0] > 0) & (table01[:, 1] == 0)]
    forcedBtab = table01[(table01[:, 0] == 0) & (table01[:, 1] > 0)]

    nfA = len(forcedAtab)
    nfB = len(forcedBtab)

    # Prepare data
    xx = np.log(table1mod[:, 1] / table1mod[:, 0])
    yy = table1mod[:, 2]

    # Fit sigmoid
    def sigmoid(x, x0, w):
        return norm.cdf(x, x0, w)

    try:
        popt, _ = curve_fit(sigmoid, xx, yy, p0=[0, 1])
        x0, w = popt
        relvalue = np.exp(x0)
        width = np.exp(w)

        # Plot fitted sigmoid
        x_fit = np.linspace(min(xx) - np.log(2) / 2, max(xx) + np.log(2) / 2, 100)
        y_fit = sigmoid(x_fit, x0, w)
    except:
        relvalue = 1.0
        width = 1.0
        x_fit = None
        y_fit = None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot fitted sigmoid
    if x_fit is not None:
        ax.plot(x_fit, y_fit, "k-", linewidth=2, label="Fitted sigmoid")

    # Plot choice data
    ax.plot(xx, yy, "ko", markersize=8, markerfacecolor="black", label="Choices")

    # Plot forced choices
    if nfA > 0:
        xx_forcedA = min(xx) - np.log(2) * np.arange(nfA, 0, -1)
        ax.plot(xx_forcedA, np.zeros(nfA), "ko", markersize=8, markerfacecolor="black")

    if nfB > 0:
        xx_forcedB = max(xx) + np.log(2) * np.arange(1, nfB + 1)
        ax.plot(xx_forcedB, np.ones(nfB), "ko", markersize=8, markerfacecolor="black")

    # Prepare x-axis labels
    unique_xx = np.unique(xx)
    xlab = []

    for i in range(nfA):
        xlab.append(
            f"{int(forcedAtab[nfA - i - 1, 1])}:{int(forcedAtab[nfA - i - 1, 0])}"
        )

    for x in unique_xx:
        idx = np.where(xx == x)[0][0]
        xlab.append(f"{int(table1mod[idx, 1])}:{int(table1mod[idx, 0])}")

    for i in range(nfB):
        xlab.append(
            f"{int(forcedBtab[nfB - i - 1, 1])}:{int(forcedBtab[nfB - i - 1, 0])}"
        )

    # Set x-axis
    if nfA > 0:
        all_x = np.concatenate([xx_forcedA, unique_xx])
    else:
        all_x = unique_xx

    if nfB > 0:
        xx_forcedB_sorted = max(xx) + np.log(2) * np.arange(1, nfB + 1)
        all_x = np.concatenate([all_x, xx_forcedB_sorted])

    ax.set_xticks(all_x)
    ax.set_xticklabels(xlab, rotation=45, ha="right")
    ax.set_xlim([min(all_x) - 0.5, max(all_x) + 0.5])

    # Set y-axis
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("% Choice B", fontsize=12)
    ax.set_xlabel("Offer (B:A)", fontsize=12)

    # Add text with fit parameters
    if x_fit is not None:
        ax.text(
            0.05,
            0.95,
            f"μ = {relvalue:.3f}\nσ = {width:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_title("Choice Pattern", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_neural_trajectories(session_summary, save_path=None):
    """
    Plot neural activity trajectories for different cell populations

    Parameters:
    -----------
    session_summary : dict
        Session summary dictionary
    save_path : str, optional
        Path to save figure
    """
    traj_quant = session_summary["traj_quant"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # OV1 cells (Offer Value A)
    ax = axes[0, 0]
    binTimes = traj_quant["rOV1"][:, 0]
    colors = cm.Reds(np.linspace(0.4, 1.0, 3))
    for i in range(3):
        ax.plot(
            binTimes,
            traj_quant["rOV1"][:, i + 1],
            color=colors[i],
            linewidth=2,
            label=f"Quantile {i + 1}",
        )
    ax.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlim([-500, 1000])
    ax.set_xlabel("Time from offer onset (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("OV1 Cells (Offer Value A)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # OV2 cells (Offer Value B)
    ax = axes[0, 1]
    colors = cm.Blues(np.linspace(0.4, 1.0, 3))
    for i in range(3):
        ax.plot(
            binTimes,
            traj_quant["rOV2"][:, i + 1],
            color=colors[i],
            linewidth=2,
            label=f"Quantile {i + 1}",
        )
    ax.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlim([-500, 1000])
    ax.set_xlabel("Time from offer onset (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("OV2 Cells (Offer Value B)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CJ1 cells (Chosen Juice A)
    ax = axes[0, 2]
    binTimes = traj_quant["r1"][:, 0]
    ax.plot(binTimes, traj_quant["r1"][:, 1], "r-", linewidth=2, label="A chosen")
    ax.plot(binTimes, traj_quant["r1"][:, 2], "g-", linewidth=2, label="B chosen")
    ax.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlim([-500, 1000])
    ax.set_xlabel("Time from offer onset (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("CJ1 Cells (Chosen Juice A)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CJ2 cells (Chosen Juice B)
    ax = axes[1, 0]
    binTimes = traj_quant["r2"][:, 0]
    ax.plot(binTimes, traj_quant["r2"][:, 1], "r-", linewidth=2, label="A chosen")
    ax.plot(binTimes, traj_quant["r2"][:, 2], "g-", linewidth=2, label="B chosen")
    ax.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlim([-500, 1000])
    ax.set_xlabel("Time from offer onset (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("CJ2 Cells (Chosen Juice B)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CV cells (Chosen Value)
    ax = axes[1, 1]
    binTimes = traj_quant["rI"][:, 0]
    colors = cm.Oranges(np.linspace(0.4, 1.0, 3))
    for i in range(3):
        ax.plot(
            binTimes,
            traj_quant["rI"][:, i + 1],
            color=colors[i],
            linewidth=2,
            label=f"Quantile {i + 1}",
        )
    ax.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlim([-500, 1000])
    ax.set_xlabel("Time from offer onset (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("CV Cells (Chosen Value)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NS cells (Non-selective)
    ax = axes[1, 2]
    binTimes = traj_quant["r3"][:, 0]
    colors = cm.Greys(np.linspace(0.4, 1.0, 3))
    for i in range(3):
        ax.plot(
            binTimes,
            traj_quant["r3"][:, i + 1],
            color=colors[i],
            linewidth=2,
            label=f"Quantile {i + 1}",
        )
    ax.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlim([-500, 1000])
    ax.set_xlabel("Time from offer onset (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("NS Cells (Non-selective)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Neural Population Trajectories", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_CJ_easysplit(session_summary, save_path=None):
    """
    Plot CJ cell activity for easy vs split trials

    Parameters:
    -----------
    session_summary : dict
        Session summary dictionary
    save_path : str, optional
        Path to save figure
    """
    CJ_traj = session_summary.get("CJ_traj_easysplit", {})

    if not CJ_traj:
        print("No CJ easy/split data available")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "Eeasy": [0, 0, 0.75],
        "Esplit": [0.6, 0.6, 1],
        "Osplit": [1, 0.6, 0.6],
        "Oeasy": [0.8, 0, 0],
    }

    labels = {
        "Eeasy": "Easy - Encoded option",
        "Esplit": "Split - Encoded option",
        "Osplit": "Split - Other option",
        "Oeasy": "Easy - Other option",
    }

    for key in ["Eeasy", "Esplit", "Osplit", "Oeasy"]:
        if key in CJ_traj:
            binTimes = CJ_traj[key][:, 0]
            traj = CJ_traj[key][:, 1]
            ax.plot(
                binTimes, traj, "-", color=colors[key], linewidth=2, label=labels[key]
            )

    ax.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlim([-500, 1000])
    ax.set_xlabel("Time from offer onset (ms)", fontsize=12)
    ax.set_ylabel("Firing rate (Hz)", fontsize=12)
    ax.set_title(
        "CJ Cell Activity: Easy vs Split Trials", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_session_summary(session_summary, save_dir=None):
    """
    Create all summary plots for a session

    Parameters:
    -----------
    session_summary : dict
        Session summary dictionary
    save_dir : str, optional
        Directory to save figures
    """
    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Plot choice pattern
    fig1 = plot_choice_pattern(
        session_summary,
        save_path=os.path.join(save_dir, "choice_pattern.png") if save_dir else None,
    )

    # Plot neural trajectories
    fig2 = plot_neural_trajectories(
        session_summary,
        save_path=os.path.join(save_dir, "neural_trajectories.png")
        if save_dir
        else None,
    )

    # Plot CJ easy/split
    fig3 = plot_CJ_easysplit(
        session_summary,
        save_path=os.path.join(save_dir, "CJ_easysplit.png") if save_dir else None,
    )

    return fig1, fig2, fig3


if __name__ == "__main__":
    # Load results and create plots
    import sys

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "sessionSumm_JNP2015.pkl"

    print(f"Loading results from {filename}...")
    with open(filename, "rb") as f:
        session_summary = pickle.load(f)

    print("Creating plots...")
    plot_session_summary(session_summary, save_dir="plots")

    print("Done! Plots saved to 'plots' directory")
    plt.show()
