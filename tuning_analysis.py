"""
Tuning Analysis Module

Computes tuning curves and behavioral analyses for the neuroeconomic model
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Any


def get_tuning(session_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute tuning curves and behavioral analyses

    Parameters:
    -----------
    session_summary : dict
        Dictionary containing simulation results

    Returns:
    --------
    session_summary : dict
        Updated dictionary with tuning and behavioral data
    """
    # Extract parameters
    Tstimon_ms = session_summary["params"]["simul"]["Tstimon_ms"]
    Tstimdur_ms = session_summary["params"]["simul"]["Tstimdur_ms"]
    Ttotal_ms = session_summary["params"]["simul"]["Ttotal_ms"]
    binsize_ms = session_summary["params"]["simul"]["binsize_ms"]

    rangeA = session_summary["params"]["behav"]["rangeA"]
    rangeB = session_summary["params"]["behav"]["rangeB"]

    offerList = session_summary["behavData"]["offerList"]
    offA = offerList[:, 1]
    offB = offerList[:, 2]

    # Extract trajectories
    rOV1_traj = session_summary["trajectory"]["rOV1_traj"]
    rOV2_traj = session_summary["trajectory"]["rOV2_traj"]
    r1_traj = session_summary["trajectory"]["r1_traj"]
    r2_traj = session_summary["trajectory"]["r2_traj"]
    r3_traj = session_summary["trajectory"]["r3_traj"]
    rI_traj = session_summary["trajectory"]["rI_traj"]

    # Define time windows
    choicebins = np.round((Tstimon_ms + np.array([400, 600])) / binsize_ms).astype(int)
    timewindow_OV_ms = Tstimon_ms + np.array([0, 500])
    timewindow_CJ_ms = Tstimon_ms + np.array([500, 1000])
    timewindow_CV_ms = Tstimon_ms + np.array([0, 500])
    timewindow_NS_ms = Tstimon_ms + np.array([0, 500])

    # Extract choices from CJ cells
    nbins = r1_traj.shape[1]
    Afinal = np.mean(r1_traj[:, choicebins[0] : choicebins[1]], axis=1)
    Bfinal = np.mean(r2_traj[:, choicebins[0] : choicebins[1]], axis=1)
    Ach = Afinal > Bfinal
    Bch = Afinal < Bfinal

    # Add choices to offerList
    offerList = np.column_stack([offerList, Ach.astype(int)])

    # Remove error trials (forced choice trials where model selected 0)
    error_trials = (~offA.astype(bool) & Ach) | (~offB.astype(bool) & Bch)
    valid_trials = ~error_trials

    # Filter data
    offA = offA[valid_trials]
    offB = offB[valid_trials]
    Ach = Ach[valid_trials]
    Bch = Bch[valid_trials]
    offerList = offerList[valid_trials]
    rOV1_traj = rOV1_traj[valid_trials]
    rOV2_traj = rOV2_traj[valid_trials]
    r1_traj = r1_traj[valid_trials]
    r2_traj = r2_traj[valid_trials]
    r3_traj = r3_traj[valid_trials]
    rI_traj = rI_traj[valid_trials]

    # Compute table01
    table01 = compute_table01(offA, offB, Bch)

    # Compute relative value from choices
    relvalue, width = compute_relvalue(table01)

    # Compute chosen value
    chval = relvalue * offA * Ach.astype(float) + offB * Bch.astype(float)

    # Behavioral output
    behav = {"table01": table01, "relvalue": relvalue, "width": width}

    # Compute tuning curves
    tuning = {}
    tuning["table01"] = table01
    tuning["rOV1"] = compute_tuning_curve(
        rOV1_traj, offerList, table01, timewindow_OV_ms, binsize_ms
    )
    tuning["rOV2"] = compute_tuning_curve(
        rOV2_traj, offerList, table01, timewindow_OV_ms, binsize_ms
    )
    tuning["r1"] = compute_tuning_curve(
        r1_traj, offerList, table01, timewindow_CJ_ms, binsize_ms
    )
    tuning["r2"] = compute_tuning_curve(
        r2_traj, offerList, table01, timewindow_CJ_ms, binsize_ms
    )
    tuning["rI"] = compute_tuning_curve(
        rI_traj, offerList, table01, timewindow_CV_ms, binsize_ms
    )
    tuning["r3"] = compute_tuning_curve(
        r3_traj, offerList, table01, timewindow_NS_ms, binsize_ms
    )

    # Compute average quantile trajectories
    binTimes = np.arange(nbins) * binsize_ms - Tstimon_ms
    traj_quant = compute_traj_quant(
        offA,
        offB,
        Ach,
        Bch,
        chval,
        relvalue,
        rOV1_traj,
        rOV2_traj,
        r1_traj,
        r2_traj,
        r3_traj,
        rI_traj,
        binTimes,
    )

    # Compute CJ trajectories for easy/split trials
    CJ_traj_easysplit = compute_CJ_easysplit(tuning, r1_traj, r2_traj, binTimes)

    # Compute CV trajectories
    CV_traj_chX = compute_CV_chX(
        session_summary, tuning, rI_traj, binTimes, rangeA, rangeB
    )

    # Update session summary
    session_summary["behavData"]["offerList"] = offerList
    session_summary["behavData"]["table01"] = behav["table01"]
    session_summary["behavData"]["relvalue"] = behav["relvalue"]
    session_summary["behavData"]["width"] = behav["width"]
    session_summary["tuning"] = tuning
    session_summary["traj_quant"] = traj_quant
    session_summary["CJ_traj_easysplit"] = CJ_traj_easysplit
    session_summary["CV_traj_chX"] = CV_traj_chX

    # Update trajectories
    session_summary["trajectory"]["rOV1_traj"] = rOV1_traj
    session_summary["trajectory"]["rOV2_traj"] = rOV2_traj
    session_summary["trajectory"]["r1_traj"] = r1_traj
    session_summary["trajectory"]["r2_traj"] = r2_traj
    session_summary["trajectory"]["r3_traj"] = r3_traj
    session_summary["trajectory"]["rI_traj"] = rI_traj

    return session_summary


def compute_table01(offA: np.ndarray, offB: np.ndarray, Bch: np.ndarray) -> np.ndarray:
    """
    Compute table of offer types with choice statistics

    Returns:
    --------
    table01 : np.ndarray
        Array with columns: [offA, offB, perc_Bch, ntrials_offtype]
    """
    offtypes = np.unique(np.column_stack([offA, offB]), axis=0)

    # Sort offtypes
    eps = 0.001
    aux = offtypes + eps
    sort_idx = np.argsort(aux[:, 1] / aux[:, 0])
    offtypes = offtypes[sort_idx]

    nofftypes = offtypes.shape[0]
    ntrials_offtype = np.zeros(nofftypes)
    perc_Bch = np.zeros(nofftypes)

    for i in range(nofftypes):
        mask = (offA == offtypes[i, 0]) & (offB == offtypes[i, 1])
        ntrials_offtype[i] = np.sum(mask)
        perc_Bch[i] = np.sum(mask & Bch) / np.sum(mask) if np.sum(mask) > 0 else 0

    table01 = np.column_stack([offtypes, perc_Bch, ntrials_offtype])
    return table01


def compute_relvalue(table01: np.ndarray) -> Tuple[float, float]:
    """
    Compute relative value from choice behavior

    Returns:
    --------
    relvalue : float
        Relative value (in units of B)
    width : float
        Width of choice distribution
    """
    # Filter for non-forced choice trials
    table1mod = table01[(table01[:, 0] > 0) & (table01[:, 1] > 0)]

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
    except:
        relvalue = 1.0
        width = 1.0

    return relvalue, width


def compute_tuning_curve(
    rtraj: np.ndarray,
    offerList: np.ndarray,
    table01: np.ndarray,
    timewindow_ms: np.ndarray,
    binsize_ms: float,
) -> Dict[str, np.ndarray]:
    """
    Compute tuning curves for neural activity

    Returns:
    --------
    tuning : dict
        Dictionary with 'bytrial', 'byoffertype', 'bytrialtype'
    """
    offtypes = table01[:, :2].astype(int)
    nofftypes = offtypes.shape[0]

    twin_bin = np.round(timewindow_ms / binsize_ms).astype(int)

    tuncurve_offer = np.zeros((nofftypes, 5))
    tuncurve_trial_list = []
    activity_list = []

    for iofftype in range(nofftypes):
        # Tuning curve by offer type
        mask = (offerList[:, 1] == offtypes[iofftype, 0]) & (
            offerList[:, 2] == offtypes[iofftype, 1]
        )

        if np.sum(mask) > 0:
            act = np.mean(rtraj[mask, twin_bin[0] : twin_bin[1]], axis=1)
            tuncurve_offer[iofftype] = [
                offtypes[iofftype, 0],
                offtypes[iofftype, 1],
                np.mean(act),
                np.std(act),
                np.sum(mask),
            ]

        # Tuning curve by trial type (split by choice)
        for Ach in [1, 0]:
            choice_sign = 2 * Ach - 1
            mask = (
                (offerList[:, 1] == offtypes[iofftype, 0])
                & (offerList[:, 2] == offtypes[iofftype, 1])
                & (offerList[:, 3] == Ach)
            )

            if np.sum(mask) > 0:
                act = np.mean(rtraj[mask, twin_bin[0] : twin_bin[1]], axis=1)
                tuncurve_trial_list.append(
                    [
                        offtypes[iofftype, 0],
                        offtypes[iofftype, 1],
                        choice_sign,
                        np.mean(act),
                        np.std(act),
                        np.sum(mask),
                    ]
                )

                # Activity trial by trial
                for trial_idx in np.where(mask)[0]:
                    activity_list.append(
                        [
                            offerList[trial_idx, 0],
                            offerList[trial_idx, 1],
                            offerList[trial_idx, 2],
                            choice_sign,
                            np.mean(rtraj[trial_idx, twin_bin[0] : twin_bin[1]]),
                        ]
                    )

    tuning = {
        "bytrial": np.array(activity_list) if activity_list else np.array([]),
        "byoffertype": tuncurve_offer,
        "bytrialtype": np.array(tuncurve_trial_list)
        if tuncurve_trial_list
        else np.array([]),
    }

    return tuning


def compute_traj_quant(
    offA,
    offB,
    Ach,
    Bch,
    chval,
    relvalue,
    rOV1_traj,
    rOV2_traj,
    r1_traj,
    r2_traj,
    r3_traj,
    rI_traj,
    binTimes,
):
    """Compute average quantile trajectories"""
    nbins = len(binTimes)
    nquant = 3

    traj_quant = {}

    # OV1 - quantiles by offA
    _, bin_edges = np.histogram(offA, bins=nquant)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    traj_quant["rOV1"] = np.zeros((nbins, nquant + 1))
    traj_quant["rOV1"][:, 0] = binTimes
    for iquant in range(nquant):
        mask = (offA >= bin_edges[iquant]) & (offA < bin_edges[iquant + 1])
        if np.sum(mask) > 0:
            traj_quant["rOV1"][:, iquant + 1] = np.mean(rOV1_traj[mask], axis=0)

    # OV2 - quantiles by offB
    _, bin_edges = np.histogram(offB, bins=nquant)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    traj_quant["rOV2"] = np.zeros((nbins, nquant + 1))
    traj_quant["rOV2"][:, 0] = binTimes
    for iquant in range(nquant):
        mask = (offB >= bin_edges[iquant]) & (offB < bin_edges[iquant + 1])
        if np.sum(mask) > 0:
            traj_quant["rOV2"][:, iquant + 1] = np.mean(rOV2_traj[mask], axis=0)

    # CJ cells - by choice
    traj_quant["r1"] = np.zeros((nbins, 3))
    traj_quant["r1"][:, 0] = binTimes
    traj_quant["r1"][:, 1] = np.mean(r1_traj[Ach], axis=0) if np.sum(Ach) > 0 else 0
    traj_quant["r1"][:, 2] = np.mean(r1_traj[Bch], axis=0) if np.sum(Bch) > 0 else 0

    traj_quant["r2"] = np.zeros((nbins, 3))
    traj_quant["r2"][:, 0] = binTimes
    traj_quant["r2"][:, 1] = np.mean(r2_traj[Ach], axis=0) if np.sum(Ach) > 0 else 0
    traj_quant["r2"][:, 2] = np.mean(r2_traj[Bch], axis=0) if np.sum(Bch) > 0 else 0

    # CV and NS cells - by chosen value
    _, bin_edges = np.histogram(chval, bins=nquant)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    traj_quant["rI"] = np.zeros((nbins, nquant + 1))
    traj_quant["rI"][:, 0] = binTimes
    traj_quant["r3"] = np.zeros((nbins, nquant + 1))
    traj_quant["r3"][:, 0] = binTimes

    for iquant in range(nquant):
        mask = (chval >= bin_edges[iquant]) & (chval < bin_edges[iquant + 1])
        if np.sum(mask) > 0:
            traj_quant["rI"][:, iquant + 1] = np.mean(rI_traj[mask], axis=0)
            traj_quant["r3"][:, iquant + 1] = np.mean(r3_traj[mask], axis=0)

    return traj_quant


def compute_CJ_easysplit(tuning, r1_traj, r2_traj, binTimes):
    """Compute CJ trajectories for easy vs split trials"""
    # Identify split offtypes
    neuract = tuning["r1"]["bytrialtype"]
    iiA = neuract[:, 2] == 1
    iiB = neuract[:, 2] == -1

    offtypes_A = neuract[iiA, :2]
    offtypes_B = neuract[iiB, :2]

    # Find common offtypes (split trials)
    offtypes_split = []
    for ot in offtypes_A:
        if np.any(np.all(offtypes_B == ot, axis=1)):
            offtypes_split.append(ot)
    offtypes_split = np.array(offtypes_split)

    # Identify trials
    neuract = tuning["r1"]["bytrial"]
    if len(neuract) == 0:
        return {}

    sort_idx = np.argsort(neuract[:, 0])
    neuract = neuract[sort_idx]

    iiA = neuract[:, 3] == 1
    iiB = neuract[:, 3] == -1

    # Check which trials are from split offtypes
    is_split = np.zeros(len(neuract), dtype=bool)
    for ot in offtypes_split:
        is_split |= np.all(neuract[:, 1:3] == ot, axis=1)

    iiA_easy = iiA & ~is_split
    iiA_split = iiA & is_split
    iiB_easy = iiB & ~is_split
    iiB_split = iiB & is_split

    # Compute trajectories
    rE_traj_easy = (
        np.vstack([r1_traj[iiA_easy], r2_traj[iiB_easy]])
        if (np.sum(iiA_easy) > 0 or np.sum(iiB_easy) > 0)
        else np.array([])
    )
    rE_traj_split = (
        np.vstack([r1_traj[iiA_split], r2_traj[iiB_split]])
        if (np.sum(iiA_split) > 0 or np.sum(iiB_split) > 0)
        else np.array([])
    )
    rO_traj_split = (
        np.vstack([r1_traj[iiB_split], r2_traj[iiA_split]])
        if (np.sum(iiA_split) > 0 or np.sum(iiB_split) > 0)
        else np.array([])
    )
    rO_traj_easy = (
        np.vstack([r1_traj[iiB_easy], r2_traj[iiA_easy]])
        if (np.sum(iiA_easy) > 0 or np.sum(iiB_easy) > 0)
        else np.array([])
    )

    CJ_traj = {}
    if len(rE_traj_easy) > 0:
        CJ_traj["Eeasy"] = np.column_stack([binTimes, np.mean(rE_traj_easy, axis=0)])
    if len(rE_traj_split) > 0:
        CJ_traj["Esplit"] = np.column_stack([binTimes, np.mean(rE_traj_split, axis=0)])
    if len(rO_traj_split) > 0:
        CJ_traj["Osplit"] = np.column_stack([binTimes, np.mean(rO_traj_split, axis=0)])
    if len(rO_traj_easy) > 0:
        CJ_traj["Oeasy"] = np.column_stack([binTimes, np.mean(rO_traj_easy, axis=0)])

    return CJ_traj


def compute_CV_chX(session_summary, tuning, rI_traj, binTimes, rangeA, rangeB):
    """Compute CV trajectories split by chosen option"""
    CV_traj_chX = {"chA": {"binTimes": binTimes}, "chB": {"binTimes": binTimes}}

    for iX, X in enumerate(["A", "B"]):
        signX = 1 if X == "A" else -1
        rangeX = rangeA if X == "A" else rangeB

        offX_list = []
        easy_list = []
        split_list = []

        for offX in range(rangeX[0], rangeX[1] + 1):
            neuract = tuning["rI"]["bytrialtype"]
            if len(neuract) == 0:
                continue

            iiX = (neuract[:, iX] == offX) & (neuract[:, 2] == signX)
            iiY = (neuract[:, iX] == offX) & (neuract[:, 2] == -signX)

            if not (np.sum(iiX) > 0 and np.sum(iiY) > 0):
                continue

            offtypes_X = neuract[iiX, :2]
            offtypes_Y = neuract[iiY, :2]

            # Easy trials
            offtypes_easy = []
            for ot in offtypes_X:
                if not np.any(np.all(offtypes_Y == ot, axis=1)):
                    offtypes_easy.append(ot)

            # Split trials
            offtypes_split = []
            for ot in offtypes_X:
                if np.any(np.all(offtypes_Y == ot, axis=1)):
                    offtypes_split.append(ot)

            if len(offtypes_easy) == 0 or len(offtypes_split) == 0:
                continue

            offtypes_easy = np.array(offtypes_easy)
            offtypes_split = np.array(offtypes_split)

            # Identify trials
            neuract_bytrial = tuning["rI"]["bytrial"]
            if len(neuract_bytrial) == 0:
                continue

            sort_idx = np.argsort(neuract_bytrial[:, 0])
            neuract_bytrial = neuract_bytrial[sort_idx]

            is_easy = np.zeros(len(neuract_bytrial), dtype=bool)
            for ot in offtypes_easy:
                is_easy |= np.all(neuract_bytrial[:, 1:3] == ot, axis=1)

            is_split = np.zeros(len(neuract_bytrial), dtype=bool)
            for ot in offtypes_split:
                is_split |= np.all(neuract_bytrial[:, 1:3] == ot, axis=1)

            iiX_bytrial = neuract_bytrial[:, 3] == signX
            iiX_easy = iiX_bytrial & is_easy
            iiX_split = iiX_bytrial & is_split

            if np.sum(iiX_easy) > 0 and np.sum(iiX_split) > 0:
                offX_list.append(offX)
                easy_list.append(np.mean(rI_traj[iiX_easy], axis=0))
                split_list.append(np.mean(rI_traj[iiX_split], axis=0))

        if len(offX_list) > 0:
            CV_traj_chX[f"ch{X}"]["offX"] = np.array(offX_list)
            CV_traj_chX[f"ch{X}"]["easy"] = np.array(easy_list)
            CV_traj_chX[f"ch{X}"]["split"] = np.array(split_list)

    return CV_traj_chX
