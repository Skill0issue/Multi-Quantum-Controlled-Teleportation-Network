"""
analysis.py — Statistical summaries for EVBTCST simulation results.

Handles both regular and verification trial rows in the same DataFrame.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("analysis")


# ── Core statistics ───────────────────────────────────────────────────────────

def compute_statistics(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"n_trials": 0, "error": "No data"}

    reg = df[df["trial_type"] == "regular"] if "trial_type" in df.columns else df
    ver = df[df["trial_type"] == "verification"] if "trial_type" in df.columns else pd.DataFrame()

    def _det_rate(row):
        byz = set(eval(row["byzantine_nodes"]))
        det = set(eval(row["detected_nodes"]))
        if not byz:
            return 1.0
        return len(byz & det) / len(byz)

    det_rates = df.apply(_det_rate, axis=1)

    stats = {
        "n_trials_total":       len(df),
        "n_regular":            len(reg),
        "n_verification":       len(ver),
        # Fidelity (regular trials only)
        "mean_F_before":        reg["fidelity_before"].mean() if len(reg) else None,
        "std_F_before":         reg["fidelity_before"].std()  if len(reg) else None,
        "mean_F_after":         reg["fidelity_after"].mean()  if len(reg) else None,
        "std_F_after":          reg["fidelity_after"].std()   if len(reg) else None,
        "mean_F_improvement":   (reg["fidelity_after"] - reg["fidelity_before"]).mean() if len(reg) else None,
        "frac_F_after_gt_0.9":  (reg["fidelity_after"] > 0.9).mean() if len(reg) else None,
        # Correction accuracy
        "M_correct_before":     df["M_correct_before"].mean(),
        "M_correct_after":      df["M_correct_after"].mean(),
        "M_fail_before":        1 - df["M_correct_before"].mean(),
        "M_fail_after":         1 - df["M_correct_after"].mean(),
        # Detection
        "mean_detection_rate":  det_rates.mean(),
        "full_detection_rate":  (det_rates == 1.0).mean(),
        # Strategy breakdown
        "mean_n_coherent_flip": df["n_coherent_flip"].mean() if "n_coherent_flip" in df else None,
        "mean_n_equivocate":    df["n_equivocate"].mean()    if "n_equivocate"    in df else None,
        "mean_n_withhold":      df["n_withhold"].mean()      if "n_withhold"      in df else None,
        # Verification trials
        "verif_pass_rate":      ver["verification_pass"].mean() if len(ver) else None,
        "mean_p_phi_plus":      ver["p_phi_plus"].mean()        if len(ver) else None,
        # Config
        "noise_level":          df["noise_level"].iloc[0],
        "n_controllers":        df["n_controllers"].iloc[0],
        "max_byzantine":        df["max_byzantine"].iloc[0],
    }

    # XOR cancellation anomaly
    xor = xor_cancellation_rate(df)
    stats["xor_cancellation_rate"]    = xor["rate"]
    stats["xor_cancellation_count"]   = xor["count"]
    stats["xor_cancellation_theory"]  = xor["theory"]

    return stats


def xor_cancellation_rate(df: pd.DataFrame) -> dict:
    """
    Measure the XOR cancellation anomaly (paper §6.2).

    When two coherent-flip adversaries corrupt the SAME correction bit,
    their flips XOR-cancel, making M accidentally correct BEFORE recovery.
    After VSS recovery detects one of them, the cancellation breaks and
    M becomes wrong — so M_correct_after < M_correct_before for these trials.

    Returns rate, count, and theoretical prediction.
    """
    if "n_coherent_flip" not in df.columns:
        return {"rate": None, "count": 0, "theory": None}

    # Trials with exactly 2 coherent-flip adversaries
    mask_2cf = df["n_coherent_flip"] == 2
    df_2cf = df[mask_2cf]

    if len(df_2cf) == 0:
        return {"rate": 0.0, "count": 0, "theory": 0.15}

    # Cancellation: M was accidentally correct before, broken after recovery
    cancellation_mask = (
        (df_2cf["M_correct_before"] == 1) &
        (df_2cf["M_correct_after"]  == 0)
    )
    count = int(cancellation_mask.sum())

    # Rate as fraction of ALL trials (matching paper's framing)
    rate = count / len(df)

    # Theory: P(same bit) * P(2 coherent-flips among f=2 Byzantine)
    # P(same bit) = C(4,2)/C(8,2) = 6/28 ≈ 0.214 for n=8 controllers
    # (4 choose 2 for same-parity positions out of 8 choose 2 total pairs)
    # Expected fraction of all f=2 trials: ~15% per paper
    n = int(df["n_controllers"].iloc[0]) if len(df) else 8
    theory = 0.15  # paper's stated value for n=8, f=2

    return {"rate": round(rate, 4), "count": count, "theory": theory}


def per_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for val, grp in df.groupby(group_col):
        s = compute_statistics(grp)
        s[group_col] = val
        rows.append(s)
    return pd.DataFrame(rows)


def print_summary(stats: dict) -> None:
    bar = "=" * 62
    print(bar)
    print("  EVBTCST SIMULATION SUMMARY")
    print(bar)
    for k, v in stats.items():
        if v is None:
            continue
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<38} {val}")
    print(bar)


def detection_power_from_sprt(sprt_summary: dict, true_byzantine: set[int]) -> dict:
    """
    Given the SPRT state summary and the ground-truth Byzantine set,
    compute empirical detection power and false positive rate.
    """
    flagged = {k for k, s in sprt_summary.items() if s["flagged"]}
    tp = len(flagged & true_byzantine)
    fp = len(flagged - true_byzantine)
    fn = len(true_byzantine - flagged)

    power = tp / max(len(true_byzantine), 1)
    fpr   = fp / max(len(sprt_summary) - len(true_byzantine), 1)

    return {
        "true_positive":  tp,
        "false_positive": fp,
        "false_negative": fn,
        "detection_power": round(power, 4),
        "false_positive_rate": round(fpr, 4),
    }
