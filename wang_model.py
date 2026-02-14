"""
Wang's Model with 11 Variables for Economic Decision-Making

This is a Python implementation of the neuroeconomic decision-making model
from "A neuro-economical model for decision-making" (Rustichini & Padoa-Schioppa, 2015)
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import pickle
from typing import Tuple, Dict, Any


class WangModel:
    """
    Neuroeconomic decision-making model based on Wang's attractor network
    """

    def __init__(self):
        """Initialize network parameters"""
        # Network parameters
        self.CE = 1600  # Total number of connections / E-cell
        self.Cext = 800  # Total number of external connections / cell
        self.NI = 400  # Total number of I-cells
        self.f = 0.15  # Fraction of structured (potentiated) synapses
        self.nuext = 3.0  # Mean firing rates of external cells

        # Synaptic time constants (ms)
        self.Tnmda = 100  # Decay time constant of NMDA
        self.Tampa = 2  # Decay time constant of AMPA
        self.Tgaba = 5  # Decay time constant of GABA_A

        # Synaptic couplings
        self.Jpampaext = -0.1123
        self.Jiampaext = -0.0842

        # Synapses OV => 1,2
        self.JpampaOV1 = 30 * self.Jpampaext
        self.JpampaOV2 = 30 * self.Jpampaext

        # Recurrent synapses
        self.Jpampa = -0.0027
        self.Jpnmdaeff = -0.00091979
        self.Jpgaba = 0.0215  # To E-cells

        self.Jiampa = -0.0022
        self.Jinmdaeff = -0.00083446
        self.Jigaba = 0.0180  # To I-cells

        # F-I curve parameters
        self.Im = 125  # E-cell parameters
        self.g = 0.16
        self.c = 310

        self.ImI = 177  # I-cell parameters
        self.gI = 0.087
        self.cI = 615

        # Economic choice parameters
        self.dj_OT = np.array([2, 1])  # dj in connections OV => T
        self.dj_rev = np.array([1, 1])  # dj in nmda reverberating connections
        self.dj_inh = np.array([1, 1])  # dj in connections inh => T

        # Network dynamics parameters
        self.w1 = 1.75  # Self-excitatory synaptic strength
        self.w2 = 1 - self.f * (self.w1 - 1) / (1 - self.f)
        self.noise_amp = 0.020

    def get_nuOV(
        self, T_total: int, T_stimon: int, dt: float, rankE: float
    ) -> np.ndarray:
        """
        Generate offer value input signal

        Parameters:
        -----------
        T_total : int
            Total number of time steps
        T_stimon : int
            Stimulus onset time step
        dt : float
            Time step size
        rankE : float
            Rank of the offer (0 to 1)

        Returns:
        --------
        nuOV : np.ndarray
            Offer value signal over time
        """
        tt = np.arange(1, T_total + 1)
        a = T_stimon + 175 / dt
        b = 30 / dt
        c = T_stimon + 400 / dt
        d = 100 / dt

        fr0 = 0  # no baseline
        Dfr = 8

        # Sigmoid function with rise and fall
        ft = 1 / (1 + np.exp(-(tt - a) / b)) * 1 / (1 + np.exp((tt - c) / d))

        nuOV = fr0 + ft / np.max(ft) * Dfr * rankE

        return nuOV

    def f_I_curve_E(self, Isyn: float) -> float:
        """
        F-I curve for excitatory cells

        Parameters:
        -----------
        Isyn : float
            Synaptic input current

        Returns:
        --------
        phi : float
            Firing rate (Hz)
        """
        phi = (self.c * Isyn - self.Im) / (
            1 - np.exp(-self.g * (self.c * Isyn - self.Im))
        )
        return phi

    def f_I_curve_I(self, Isyn: float) -> float:
        """
        F-I curve for inhibitory cells

        Parameters:
        -----------
        Isyn : float
            Synaptic input current

        Returns:
        --------
        phi : float
            Firing rate (Hz)
        """
        phi = (self.cI * Isyn - self.ImI) / (
            1 - np.exp(-self.gI * (self.cI * Isyn - self.ImI))
        )
        return phi

    def run_trial(
        self,
        offA: int,
        offB: int,
        rangeA: Tuple[int, int],
        rangeB: Tuple[int, int],
        dj_HL: np.ndarray,
        initial_state: Dict[str, float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run a single trial of the model

        Parameters:
        -----------
        offA : int
            Offer A quantity
        offB : int
            Offer B quantity
        rangeA : tuple
            Range of offer A values (min, max)
        rangeB : tuple
            Range of offer B values (min, max)
        dj_HL : np.ndarray
            Hebbian learning weights
        initial_state : dict, optional
            Initial state for implementing hysteresis

        Returns:
        --------
        result : dict
            Dictionary containing smoothed firing rates and gating variables
        """
        # Time parameters
        dt = 0.5  # Time step in ms
        time_wind = int(50 / dt)  # Temporal window size for averaging
        slide_wind = int(5 / dt)  # Sliding step for window
        T_total = int(3000 / dt + time_wind)  # Total number of steps
        T_stimon = int(1000 / dt)  # Stimulus onset time
        T_stimdur = int(500 / dt)  # Duration of motion viewing

        # Initialize state variables
        if initial_state is None:
            s1 = 0.1
            s2 = 0.1
            s3 = 0.1
            sampa1 = 0.0
            sampa2 = 0.0
            sampa3 = 0.0
            sgaba = 0.0
            nu1 = 3.0
            nu2 = 3.0
            nu3 = 3.0
            nuI = 8.0
        else:
            s1 = initial_state["s1"]
            s2 = initial_state["s2"]
            s3 = initial_state["s3"]
            sampa1 = initial_state["sampa1"]
            sampa2 = initial_state["sampa2"]
            sampa3 = initial_state["sampa3"]
            sgaba = initial_state["sgaba"]
            nu1 = initial_state["nu1"]
            nu2 = initial_state["nu2"]
            nu3 = initial_state["nu3"]
            nuI = initial_state["nuI"]

        # Initialize arrays
        s1_arr = np.ones(T_total) * s1
        s2_arr = np.ones(T_total) * s2
        s3_arr = np.ones(T_total) * s3
        sampa1_arr = np.ones(T_total) * sampa1
        sampa2_arr = np.ones(T_total) * sampa2
        sampa3_arr = np.ones(T_total) * sampa3
        sgaba_arr = np.ones(T_total) * sgaba

        nu1_arr = np.ones(T_total) * nu1
        nu2_arr = np.ones(T_total) * nu2
        nu3_arr = np.ones(T_total) * nu3
        nuI_arr = np.ones(T_total) * nuI

        phi1_arr = np.ones(T_total) * nu1
        phi2_arr = np.ones(T_total) * nu2
        phi3_arr = np.ones(T_total) * nu3
        phiI_arr = np.ones(T_total) * nuI

        I_eta1 = np.zeros(T_total)
        I_eta2 = np.zeros(T_total)
        I_eta3 = np.zeros(T_total)
        I_etaI = np.zeros(T_total)

        # Generate offer value inputs
        rankA = offA / rangeA[1]
        rankB = offB / rangeB[1]

        nuOV1 = self.get_nuOV(T_total, T_stimon, dt, rankA)
        nuOV2 = self.get_nuOV(T_total, T_stimon, dt, rankB)

        # Time loop
        for t in range(T_total - 1):
            # External current input
            IampaextE1 = (
                -self.Jpampaext * self.Tampa * self.Cext * self.nuext / 1000 + I_eta1[t]
            )
            IampaextE2 = (
                -self.Jpampaext * self.Tampa * self.Cext * self.nuext / 1000 + I_eta2[t]
            )
            IampaextE3 = (
                -self.Jpampaext * self.Tampa * self.Cext * self.nuext / 1000 + I_eta3[t]
            )
            IampaextG = (
                -self.Jiampaext * self.Tampa * self.Cext * self.nuext / 1000 + I_etaI[t]
            )

            # Offer values with hebbian learning
            I_stim_1 = (
                nuOV1[t]
                * dj_HL[0]
                * self.dj_OT[0]
                * (-self.JpampaOV1 * self.Tampa / 1000)
            )
            I_stim_2 = (
                nuOV2[t]
                * dj_HL[1]
                * self.dj_OT[1]
                * (-self.JpampaOV2 * self.Tampa / 1000)
            )

            # Input to population 1
            Iamparec1 = (
                -self.CE * (1 - 2 * self.f) * self.Jpampa * sampa3_arr[t]
                - self.CE * self.f * self.w1 * self.Jpampa * sampa1_arr[t]
                - self.CE * self.f * self.w2 * self.Jpampa * sampa2_arr[t]
            )
            Inmda1 = (
                -self.CE * (1 - 2 * self.f) * self.Jpnmdaeff * s3_arr[t]
                - self.CE
                * self.f
                * self.w1
                * self.dj_rev[0]
                * self.Jpnmdaeff
                * s1_arr[t]
                - self.CE * self.f * self.w2 * self.Jpnmdaeff * s2_arr[t]
            )
            Igaba1 = -self.NI * self.dj_inh[0] * self.Jpgaba * sgaba_arr[t]
            Isyn1 = Inmda1 + Iamparec1 + Igaba1 + IampaextE1 + I_stim_1
            phi1_arr[t] = self.f_I_curve_E(Isyn1)

            # Input to population 2
            Iamparec2 = (
                -self.CE * (1 - 2 * self.f) * self.Jpampa * sampa3_arr[t]
                - self.CE * self.f * self.w2 * self.Jpampa * sampa1_arr[t]
                - self.CE * self.f * self.w1 * self.Jpampa * sampa2_arr[t]
            )
            Inmda2 = (
                -self.CE * (1 - 2 * self.f) * self.Jpnmdaeff * s3_arr[t]
                - self.CE * self.f * self.Jpnmdaeff * s1_arr[t]
                - self.CE
                * self.f
                * self.w1
                * self.dj_rev[1]
                * self.Jpnmdaeff
                * s2_arr[t]
            )
            Igaba2 = -self.NI * self.dj_inh[1] * self.Jpgaba * sgaba_arr[t]
            Isyn2 = Inmda2 + Iamparec2 + Igaba2 + IampaextE2 + I_stim_2
            phi2_arr[t] = self.f_I_curve_E(Isyn2)

            # Input to population 3 (non-selective)
            Iamparec3 = (
                -self.CE * (1 - 2 * self.f) * self.Jpampa * sampa3_arr[t]
                - self.CE * self.f * self.Jpampa * sampa1_arr[t]
                - self.CE * self.f * self.Jpampa * sampa2_arr[t]
            )
            Inmda3 = (
                -self.CE * (1 - 2 * self.f) * self.Jpnmdaeff * s3_arr[t]
                - self.CE * self.f * self.Jpnmdaeff * s1_arr[t]
                - self.CE * self.f * self.Jpnmdaeff * s2_arr[t]
            )
            Igaba3 = -self.NI * self.Jpgaba * sgaba_arr[t]
            Isyn3 = Inmda3 + Iamparec3 + Igaba3 + IampaextE3
            phi3_arr[t] = self.f_I_curve_E(Isyn3)

            # Input to population I (inhibitory)
            IamparecI = (
                -self.CE * (1 - 2 * self.f) * self.Jiampa * sampa3_arr[t]
                - self.CE * self.f * self.Jiampa * sampa1_arr[t]
                - self.CE * self.f * self.Jiampa * sampa2_arr[t]
            )
            InmdaI = (
                -self.CE * (1 - 2 * self.f) * self.Jinmdaeff * s3_arr[t]
                - self.CE * self.f * self.Jinmdaeff * s1_arr[t]
                - self.CE * self.f * self.Jinmdaeff * s2_arr[t]
            )
            IgabaI = -self.NI * self.Jigaba * sgaba_arr[t]
            IsynI = InmdaI + IamparecI + IgabaI + IampaextG
            phiI_arr[t] = self.f_I_curve_I(IsynI)

            # Update synaptic gating variables - NMDA
            s1_arr[t + 1] = s1_arr[t] + dt * (
                -(s1_arr[t] / self.Tnmda) + (1 - s1_arr[t]) * 0.641 * nu1_arr[t] / 1000
            )
            s2_arr[t + 1] = s2_arr[t] + dt * (
                -(s2_arr[t] / self.Tnmda) + (1 - s2_arr[t]) * 0.641 * nu2_arr[t] / 1000
            )
            s3_arr[t + 1] = s3_arr[t] + dt * (
                -(s3_arr[t] / self.Tnmda) + (1 - s3_arr[t]) * 0.641 * nu3_arr[t] / 1000
            )

            # Update synaptic gating variables - AMPA
            sampa1_arr[t + 1] = sampa1_arr[t] + dt * (
                -sampa1_arr[t] / self.Tampa + nu1_arr[t] / 1000
            )
            sampa2_arr[t + 1] = sampa2_arr[t] + dt * (
                -sampa2_arr[t] / self.Tampa + nu2_arr[t] / 1000
            )
            sampa3_arr[t + 1] = sampa3_arr[t] + dt * (
                -sampa3_arr[t] / self.Tampa + nu3_arr[t] / 1000
            )

            # Update synaptic gating variables - GABA_A
            sgaba_arr[t + 1] = sgaba_arr[t] + dt * (
                -sgaba_arr[t] / self.Tgaba + nuI_arr[t] / 1000
            )

            # Update firing rates with bounds
            if phi1_arr[t] > 500:
                nu1_arr[t + 1] = 500
                phi1_arr[t] = 500
            else:
                nu1_arr[t + 1] = nu1_arr[t] + (dt / self.Tampa) * (
                    -nu1_arr[t] + phi1_arr[t]
                )

            if phi2_arr[t] > 500:
                nu2_arr[t + 1] = 500
                phi2_arr[t] = 500
            else:
                nu2_arr[t + 1] = nu2_arr[t] + (dt / self.Tampa) * (
                    -nu2_arr[t] + phi2_arr[t]
                )

            if phi3_arr[t] > 500:
                nu3_arr[t + 1] = 500
                phi3_arr[t] = 500
            else:
                nu3_arr[t + 1] = nu3_arr[t] + (dt / self.Tampa) * (
                    -nu3_arr[t] + phi3_arr[t]
                )

            if phiI_arr[t] > 1000:
                nuI_arr[t + 1] = 1000
                phiI_arr[t] = 1000
            else:
                nuI_arr[t + 1] = nuI_arr[t] + (dt / self.Tampa) * (
                    -nuI_arr[t] + phiI_arr[t]
                )

            # Generate noise
            I_eta1[t + 1] = (
                I_eta1[t]
                + (dt / self.Tampa) * (-I_eta1[t])
                + np.sqrt(dt / self.Tampa) * self.noise_amp * np.random.randn()
            )
            I_eta2[t + 1] = (
                I_eta2[t]
                + (dt / self.Tampa) * (-I_eta2[t])
                + np.sqrt(dt / self.Tampa) * self.noise_amp * np.random.randn()
            )
            I_eta3[t + 1] = (
                I_eta3[t]
                + (dt / self.Tampa) * (-I_eta3[t])
                + np.sqrt(dt / self.Tampa) * self.noise_amp * np.random.randn()
            )
            I_etaI[t + 1] = (
                I_etaI[t]
                + (dt / self.Tampa) * (-I_etaI[t])
                + np.sqrt(dt / self.Tampa) * self.noise_amp * np.random.randn()
            )

        # Smooth the firing rates using sliding window
        nwind = int((T_total - time_wind) / slide_wind)

        def smooth_signal(signal, nwind, time_wind, slide_wind):
            """Helper function to smooth signal"""
            smoothed = np.zeros(nwind + 1)
            smoothed[0] = np.mean(signal[:time_wind])
            for jj in range(nwind):
                start = jj * slide_wind
                end = jj * slide_wind + time_wind
                smoothed[jj + 1] = np.mean(signal[start:end])
            return smoothed

        result = {
            "nuOV1_wind": smooth_signal(nuOV1, nwind, time_wind, slide_wind),
            "nuOV2_wind": smooth_signal(nuOV2, nwind, time_wind, slide_wind),
            "nu1_wind": smooth_signal(nu1_arr, nwind, time_wind, slide_wind),
            "nu2_wind": smooth_signal(nu2_arr, nwind, time_wind, slide_wind),
            "nu3_wind": smooth_signal(nu3_arr, nwind, time_wind, slide_wind),
            "nuI_wind": smooth_signal(nuI_arr, nwind, time_wind, slide_wind),
            "s1_wind": smooth_signal(s1_arr, nwind, time_wind, slide_wind),
            "s2_wind": smooth_signal(s2_arr, nwind, time_wind, slide_wind),
            "s3_wind": smooth_signal(s3_arr, nwind, time_wind, slide_wind),
            "sampa1_wind": smooth_signal(sampa1_arr, nwind, time_wind, slide_wind),
            "sampa2_wind": smooth_signal(sampa2_arr, nwind, time_wind, slide_wind),
            "sampa3_wind": smooth_signal(sampa3_arr, nwind, time_wind, slide_wind),
            "sgaba_wind": smooth_signal(sgaba_arr, nwind, time_wind, slide_wind),
            "final_state": {
                "s1": s1_arr[-1],
                "s2": s2_arr[-1],
                "s3": s3_arr[-1],
                "sampa1": sampa1_arr[-1],
                "sampa2": sampa2_arr[-1],
                "sampa3": sampa3_arr[-1],
                "sgaba": sgaba_arr[-1],
                "nu1": nu1_arr[-1],
                "nu2": nu2_arr[-1],
                "nu3": nu3_arr[-1],
                "nuI": nuI_arr[-1],
            },
        }

        return result
