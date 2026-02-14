"""
Main Simulation Script for Wang's Neuroeconomic Model

Runs the complete simulation and saves results
"""

import numpy as np
import pickle
from wang_model import WangModel
from session_params import session_params
from tuning_analysis import get_tuning


def run_simulation(filesuffix="JNP2015", session_mode="explicit", ntrials=None):
    """
    Run the complete neuroeconomic model simulation

    Parameters:
    -----------
    filesuffix : str
        Suffix for saved files
    session_mode : str
        'explicit' or 'parametric'
    ntrials : int or None
        Number of trials (if None, use default from session_params)
    """
    print("=" * 70)
    print("Wang's Neuroeconomic Decision-Making Model")
    print("=" * 70)

    # Initialize model
    model = WangModel()

    # Generate offer list
    if ntrials is None:
        offList, rangeA, rangeB = session_params(session_mode)
    else:
        offList, rangeA, rangeB = session_params(session_mode, ntrials)

    offA = offList[:, 0]
    offB = offList[:, 1]
    ntrials = len(offList)

    print(f"\nSession mode: {session_mode}")
    print(f"Number of trials: {ntrials}")
    print(f"Offer A range: {rangeA}")
    print(f"Offer B range: {rangeB}")

    # Hebbian learning weights
    dj_HL = np.array([rangeA[1] / rangeB[1], 1.0])

    # Initialize trajectory storage
    # Will be filled after first trial
    rOV1_traj = None
    rOV2_traj = None
    r1_traj = None
    r2_traj = None
    r3_traj = None
    rI_traj = None
    s1_traj = None
    s2_traj = None
    s3_traj = None
    sampa1_traj = None
    sampa2_traj = None
    sampa3_traj = None
    sgaba_traj = None

    # Run trials
    print("\nRunning simulation...")
    implement_hysteresis = False  # Set to True to implement choice hysteresis

    for ww in range(ntrials):
        # Print progress every 10 trials
        if ww % 10 == 0:
            print(f"Processing trial {ww + 1}/{ntrials}...")
        # Determine initial state
        if ww == 0 or not implement_hysteresis:
            initial_state = None
        else:
            initial_state = {
                "s1": s1_traj[ww - 1, -1],
                "s2": s2_traj[ww - 1, -1],
                "s3": s3_traj[ww - 1, -1],
                "sampa1": sampa1_traj[ww - 1, -1],
                "sampa2": sampa2_traj[ww - 1, -1],
                "sampa3": sampa3_traj[ww - 1, -1],
                "sgaba": sgaba_traj[ww - 1, -1],
                "nu1": r1_traj[ww - 1, -1],
                "nu2": r2_traj[ww - 1, -1],
                "nu3": r3_traj[ww - 1, -1],
                "nuI": rI_traj[ww - 1, -1],
            }

        # Run trial
        result = model.run_trial(
            offA[ww], offB[ww], rangeA, rangeB, dj_HL, initial_state
        )

        # Initialize arrays on first trial
        if ww == 0:
            nwind = len(result["nu1_wind"])
            rOV1_traj = np.zeros((ntrials, nwind))
            rOV2_traj = np.zeros((ntrials, nwind))
            r1_traj = np.zeros((ntrials, nwind))
            r2_traj = np.zeros((ntrials, nwind))
            r3_traj = np.zeros((ntrials, nwind))
            rI_traj = np.zeros((ntrials, nwind))
            s1_traj = np.zeros((ntrials, nwind))
            s2_traj = np.zeros((ntrials, nwind))
            s3_traj = np.zeros((ntrials, nwind))
            sampa1_traj = np.zeros((ntrials, nwind))
            sampa2_traj = np.zeros((ntrials, nwind))
            sampa3_traj = np.zeros((ntrials, nwind))
            sgaba_traj = np.zeros((ntrials, nwind))

        # Store results
        rOV1_traj[ww] = result["nuOV1_wind"]
        rOV2_traj[ww] = result["nuOV2_wind"]
        r1_traj[ww] = result["nu1_wind"]
        r2_traj[ww] = result["nu2_wind"]
        r3_traj[ww] = result["nu3_wind"]
        rI_traj[ww] = result["nuI_wind"]
        s1_traj[ww] = result["s1_wind"]
        s2_traj[ww] = result["s2_wind"]
        s3_traj[ww] = result["s3_wind"]
        sampa1_traj[ww] = result["sampa1_wind"]
        sampa2_traj[ww] = result["sampa2_wind"]
        sampa3_traj[ww] = result["sampa3_wind"]
        sgaba_traj[ww] = result["sgaba_wind"]

    print("\nSimulation complete!")

    # Create session summary
    dt = 0.5
    time_wind = int(50 / dt)
    slide_wind = int(5 / dt)
    T_total = int(3000 / dt + time_wind)
    T_stimon = int(1000 / dt)
    T_stimdur = int(500 / dt)

    session_summary = {
        "params": {
            "simul": {
                "Tstimon_ms": T_stimon * dt,
                "Tstimdur_ms": T_stimdur * dt,
                "Ttotal_ms": T_total * dt,
                "binsize_ms": slide_wind * dt,
            },
            "LIF": {
                "Im": model.Im,
                "g": model.g,
                "c": model.c,
                "ImI": model.ImI,
                "gI": model.gI,
                "cI": model.cI,
            },
            "behav": {"rangeA": rangeA, "rangeB": rangeB},
        },
        "behavData": {
            "offerList": np.column_stack([np.arange(1, ntrials + 1), offA, offB])
        },
        "trajectory": {
            "rOV1_traj": rOV1_traj,
            "rOV2_traj": rOV2_traj,
            "r1_traj": r1_traj,
            "r2_traj": r2_traj,
            "r3_traj": r3_traj,
            "rI_traj": rI_traj,
            "s1_traj": s1_traj,
            "s2_traj": s2_traj,
            "s3_traj": s3_traj,
            "sampa1_traj": sampa1_traj,
            "sampa2_traj": sampa2_traj,
            "sampa3_traj": sampa3_traj,
            "sgaba_traj": sgaba_traj,
        },
    }

    # Compute tuning curves and behavioral analyses
    print("\nComputing tuning curves and behavioral analyses...")
    session_summary = get_tuning(session_summary)

    # Remove full trajectories to save space (as in MATLAB code)
    session_summary_light = session_summary.copy()
    del session_summary_light["trajectory"]

    # Save results
    filename = f"sessionSumm_{filesuffix}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(session_summary_light, f)

    print(f"\nResults saved to: {filename}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if "relvalue" in session_summary["behavData"]:
        relvalue = session_summary["behavData"]["relvalue"]
        width = session_summary["behavData"]["width"]
        print(f"Relative value (μ): {relvalue:.3f}")
        print(f"Choice width (σ): {width:.3f}")

    offerList = session_summary["behavData"]["offerList"]
    if offerList.shape[1] > 3:
        choices = offerList[:, 3]
        print(f"Total trials: {len(choices)}")
        print(f"Chose A: {np.sum(choices == 1)} ({100 * np.mean(choices == 1):.1f}%)")
        print(f"Chose B: {np.sum(choices == 0)} ({100 * np.mean(choices == 0):.1f}%)")

    print("=" * 70)

    return session_summary


if __name__ == "__main__":
    # Run simulation with fewer trials for testing
    session_summary = run_simulation(
        filesuffix="JNP2015",
        session_mode="explicit",
        ntrials=1000,  # Use fewer trials for faster testing
    )
