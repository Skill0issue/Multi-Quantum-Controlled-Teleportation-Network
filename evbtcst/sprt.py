"""
sprt.py — Per-controller Sequential Probability Ratio Test (SPRT).

Theory (paper §4.3)
-------------------
Each verification trial produces a Bell outcome.  For an honest controller,
the probability of |Φ+⟩ is p1 = F_noise + (1-F_noise)/4.  For a coherent-flip
adversary contributing to a Pauli error, it is p0 = (1-F_noise)/4.

We track how many times each controller's bit contributed to a WRONG correction
on a failed verification trial (Bell outcome ≠ |Φ+⟩).  The log-likelihood ratio
accumulates evidence:

    LLR_k += log( p_flip(adversary) / p_flip(honest) )

where
    p_flip(honest)    = 1 - p1  = 3(1-F_noise)/4
    p_flip(adversary) = 1 - p0  = F_noise + 3(1-F_noise)/4

Stopping rule: flag controller k when LLR_k > log(A),  A = (1-beta)/alpha.
Expected trials to detection for a coherent-flip adversary: ~2.5 at p=0.01.

Usage
-----
    tracker = SPRTTracker(cfg)
    tracker.update_failed_trial(contributing_controller_ids=[2, 5])
    flagged = tracker.get_flagged_controllers()
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("sprt")


@dataclass
class ControllerSPRT:
    """SPRT state for a single controller."""
    controller_id: int
    llr: float = 0.0            # accumulated log-likelihood ratio
    n_verif_seen: int = 0       # verification trials where this controller was active
    n_contrib_flips: int = 0    # times this controller contributed to a failed verif.
    flagged: bool = False
    flagged_at_trial: Optional[int] = None


class SPRTTracker:
    """
    Tracks per-controller SPRT state across verification trials.

    Parameters
    ----------
    n_controllers : int
    F_noise       : float  — estimated noise fidelity (default 0.833 at p=0.01)
    alpha         : float  — false-positive rate
    beta          : float  — false-negative rate target
    min_k         : int    — minimum failures before flagging (avoids early false alarms)
    """

    def __init__(
        self,
        n_controllers: int,
        F_noise: float = 0.833,
        alpha: float = 0.01,
        beta: float = 0.0001,
        min_k: int = 3,
        p1: Optional[float] = None,
        p0: Optional[float] = None,
        p_flip_honest: Optional[float] = None,
        p_flip_adv: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        p1 : float, optional
            Empirical P(Phi+ | honest trial). Used to derive p_flip_honest = 1-p1
            only if p_flip_honest is not given directly.
        p0 : float, optional
            Empirical P(Phi+ | adversary trial). Used to derive p_flip_adv = 1-p0
            only if p_flip_adv is not given directly.
        p_flip_honest : float, optional
            Directly measured P(controller blamed | honest run). Overrides all
            other estimates. This is the correct observable for the SPRT.
        p_flip_adv : float, optional
            Directly measured P(controller blamed | adversary present). Overrides
            all other estimates.
        """
        self.n = n_controllers
        self.F_noise = F_noise
        self.alpha = alpha
        self.beta = beta
        self.min_k = min_k

        # Threshold: flag when LLR > log(A),  A = (1-beta)/alpha
        self.threshold = math.log((1 - beta) / alpha)

        # Two-sided SPRT increments (proper Wald SPRT)
        # Priority: p_flip_honest/adv > p1/p0 > F_noise model
        if p_flip_honest is not None and p_flip_adv is not None:
            pf_h = max(p_flip_honest, 1e-12)
            pf_a = max(p_flip_adv, 1e-12)
        elif p1 is not None and p0 is not None:
            pf_h = max(1 - p1, 1e-12)
            pf_a = max(1 - p0, 1e-12)
        else:
            pf_h = (1 - F_noise) * 3 / 4
            pf_a = F_noise + (1 - F_noise) * 3 / 4
        # LLR increment when controller IS blamed (flip observed)
        self._llr_flip    = math.log(pf_a / max(pf_h, 1e-12))
        # LLR increment when controller is NOT blamed (no flip)
        self._llr_no_flip = math.log(
            max(1 - pf_a, 1e-12) / max(1 - pf_h, 1e-12)
        )
        # Legacy alias kept for external callers
        self._llr_increment = self._llr_flip

        self._states: dict[int, ControllerSPRT] = {
            k: ControllerSPRT(k) for k in range(1, n_controllers + 1)
        }
        self._trial_count = 0

        log.debug(
            f"SPRTTracker: threshold={self.threshold:.3f}  "
            f"llr_flip={self._llr_flip:.3f}  llr_no_flip={self._llr_no_flip:.3f}  "
            f"F_noise={F_noise:.3f}"
        )

    def update_verification_trial(
        self,
        outcome: str,
        contributing_controller_ids: list[int],
        trial_id: int = -1,
    ) -> list[int]:
        """
        Process one verification trial result.

        Parameters
        ----------
        outcome                     : Bell outcome label ('Phi+', 'Phi-', etc.)
        contributing_controller_ids : controllers whose bit position could have caused a flip.
                                      On a passed trial (Phi+), this list is ignored.
        trial_id                    : for logging

        Returns
        -------
        newly_flagged : list of controller IDs newly flagged this trial
        """
        self._trial_count += 1
        passed = (outcome == "Phi+")

        contributing_set = set(contributing_controller_ids)
        newly_flagged = []
        for k, state in self._states.items():
            if state.flagged:
                continue
            state.n_verif_seen += 1

            if not passed and k in contributing_set:
                # Blamed on a failed trial → evidence for adversary
                state.n_contrib_flips += 1
                state.llr += self._llr_flip
            else:
                # Pass OR not blamed → evidence for honesty (negative increment)
                state.llr += self._llr_no_flip

            if (state.n_contrib_flips >= self.min_k
                    and state.llr >= self.threshold):
                state.flagged = True
                state.flagged_at_trial = trial_id
                newly_flagged.append(k)
                log.info(
                    f"[SPRT] Controller C{k} FLAGGED at trial {trial_id}: "
                    f"LLR={state.llr:.3f} >= threshold={self.threshold:.3f}, "
                    f"flips={state.n_contrib_flips}"
                )

        return newly_flagged

    def get_flagged_controllers(self) -> set[int]:
        return {k for k, s in self._states.items() if s.flagged}

    def get_llr(self, controller_id: int) -> float:
        return self._states[controller_id].llr

    def state_summary(self) -> dict[int, dict]:
        return {
            k: {
                "llr": round(s.llr, 4),
                "flips": s.n_contrib_flips,
                "seen": s.n_verif_seen,
                "flagged": s.flagged,
            }
            for k, s in self._states.items()
        }

    def reset(self):
        """Reset all states (e.g. between independent experiment runs)."""
        for k in self._states:
            self._states[k] = ControllerSPRT(k)
        self._trial_count = 0


def compute_per_controller_flip_contributions(
    all_outcomes: list[int],
    M_X_true: int,
    M_Z_true: int,
    M_X_used: int,
    M_Z_used: int,
    n_controllers: int,
) -> list[int]:
    """
    Identify which controller positions could have caused the observed
    correction error (M_X_used != M_X_true or M_Z_used != M_Z_true).

    A controller at position i (0-indexed in all_outcomes) contributes to:
      - M_X error if i is ODD  and M_X_used != M_X_true
      - M_Z error if i is EVEN and M_Z_used != M_Z_true

    Returns list of 1-based controller IDs (positions 2..n+1 in all_outcomes
    correspond to C1..Cn, i.e. controller k is at all_outcomes index k+1).
    """
    x_flipped = (M_X_used != M_X_true)
    z_flipped = (M_Z_used != M_Z_true)

    contributing = []
    for k in range(1, n_controllers + 1):
        pos = k + 1   # position in all_outcomes: [input(0), alice(1), C1(2), C2(3), ...]
        if pos % 2 == 1 and x_flipped:   # odd position → contributes to M_X
            contributing.append(k)
        elif pos % 2 == 0 and z_flipped:  # even position → contributes to M_Z
            contributing.append(k)

    return contributing
