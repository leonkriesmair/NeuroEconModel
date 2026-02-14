"""
Session Parameters Module

Generates lists of offers for economic choice experiments
"""

import numpy as np
from typing import Tuple


def session_params(
    session_mode: str = "explicit", ntrials: int = 2500
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Generate a list of offers for the experiment

    Parameters:
    -----------
    session_mode : str
        Either 'explicit' (cover the plane uniformly) or 'parametric' (as in experiments)
    ntrials : int
        Total number of trials (default: 2500)

    Returns:
    --------
    offList : np.ndarray
        Array of shape (ntrials, 2) with offer A and offer B for each trial
    rangeA : tuple
        (min, max) range for offer A
    rangeB : tuple
        (min, max) range for offer B
    """

    if session_mode == "explicit":
        # Value ranges in current session
        rangeA = (0, 20)
        rangeB = (0, 20)

        # Offer ranks and quantities in each trial
        # Do 2* to keep ntrials correct once we remove some trial types
        rankA = np.random.rand(2 * ntrials)
        rankB = np.random.rand(2 * ntrials)

        # Offers
        offA = np.floor(rankA * (rangeA[1] - rangeA[0] + 1)) + rangeA[0]
        offB = np.floor(rankB * (rangeB[1] - rangeB[0] + 1)) + rangeB[0]

        # Remove offers 0:0
        valid_indices = ~((offA == 0) & (offB == 0))
        offA = offA[valid_indices]
        offB = offB[valid_indices]

        # Take only the required number of trials
        offList = np.column_stack([offA[:ntrials], offB[:ntrials]])

    elif session_mode == "parametric":
        nblocks = 15

        # Define offer types: [N, offer1, offer2]
        offerTypes = np.array(
            [
                [2, 1, 0],
                [2, 5, 1],
                [2, 4, 1],
                [2, 3, 1],
                [2, 2, 1],
                [2, 1, 1],
                [2, 1, 2],
                [2, 1, 4],
                [2, 1, 6],
                [2, 1, 8],
                [2, 1, 10],
                [2, 0, 2],
            ]
        )

        ntrials_block = int(np.sum(offerTypes[:, 0]))

        offList = []
        for iblock in range(nblocks):
            offerList_block = []
            for itype in range(offerTypes.shape[0]):
                n_repetitions = int(offerTypes[itype, 0])
                for _ in range(n_repetitions):
                    offerList_block.append(offerTypes[itype, 1:])

            # Randomize order within block
            offerList_block = np.array(offerList_block)
            np.random.shuffle(offerList_block)
            offList.append(offerList_block)

        offList = np.vstack(offList)

        # Determine ranges
        rangeA = (int(np.min(offList[:, 0])), int(np.max(offList[:, 0])))
        rangeB = (int(np.min(offList[:, 1])), int(np.max(offList[:, 1])))

    else:
        raise ValueError(f"Unknown session_mode: {session_mode}")

    return offList.astype(int), rangeA, rangeB
