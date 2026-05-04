"""
byzantine.py — Byzantine adversary model.

Each Byzantine node is assigned one strategy at selection time and applies it
consistently across ALL protocol channels (Pedersen commitment, hash VSS, and
direct measurement report).

Strategy semantics (consistent with secret_sharing.py)
-------------------------------------------------------
coherent_flip  : Commits to 1-m_k, shares poly with f(0)=1-m_k, reports 1-m_k.
                 All channels consistent. Information-theoretically undetectable.
                 Corrupts one bit of M_X or M_Z.

equivocate     : Commits to poly_a (hash), but sends poly_b shares to j > n//2.
                 Hash mismatch in Phase 3 → detected. Recovery uses committed poly_a.
                 Corrupts direct report but is caught and corrected.

withhold       : Does not publish any commitments. Direct report is honest (m_k).
                 Absence detected in Phase 1. VSS aggregate is affected but
                 direct report is correct so M_X/M_Z is not corrupted.
"""

from __future__ import annotations
import random
import logging

log = logging.getLogger("byzantine")

STRATEGIES = ("coherent_flip", "equivocate", "withhold")


def select_byzantine_nodes(n: int, f: int) -> dict[int, str]:
    """
    Randomly choose f distinct controller indices from {1…n} and assign
    each one a uniformly-random strategy.

    Returns {node_index (1-based): strategy_string}.
    Empty dict when f == 0.
    """
    if f == 0:
        return {}
    if f > n:
        raise ValueError(f"Cannot have f={f} Byzantine nodes with n={n}")
    nodes = random.sample(range(1, n + 1), f)
    return {node: random.choice(STRATEGIES) for node in nodes}


def corrupt_direct_reports(
    true_measurements: list[int],
    byzantine_nodes: dict[int, str],
) -> list[int]:
    """
    Produce the direct-reported measurement vector.

    coherent_flip  : flip m_k → 1-m_k
    equivocate     : flip m_k → 1-m_k  (also corrupts VSS; both channels flip)
    withhold       : keep m_k           (honest direct report; attacks via VSS)
    """
    reported = list(true_measurements)
    for b, strategy in byzantine_nodes.items():
        idx = b - 1
        if not (0 <= idx < len(reported)):
            continue
        if strategy in ("coherent_flip", "equivocate"):
            reported[idx] = 1 - reported[idx]
            log.debug(f"  [C{b}/{strategy}] flipped report: "
                      f"{true_measurements[idx]} → {reported[idx]}")
        else:
            log.debug(f"  [C{b}/withhold] honest direct report: {reported[idx]}")
    return reported
