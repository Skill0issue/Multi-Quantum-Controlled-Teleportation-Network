"""
analysis.py — Statistical summaries of simulation results.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("analysis")


def compute_statistics(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"n_trials": 0, "error": "No data"}

    n = len(df)

    # Detection: what fraction of Byzantine nodes were caught
    def _det_rate(row):
        byz = set(eval(row["byzantine_nodes"]))
        det = set(eval(row["detected_nodes"]))
        if not byz:
            return 1.0
        return len(byz & det) / len(byz)

    det_rates = df.apply(_det_rate, axis=1)

    return {
        "n_trials":                  n,
        # Fidelity
        "mean_F_before":             df["fidelity_before"].mean(),
        "std_F_before":              df["fidelity_before"].std(),
        "mean_F_after":              df["fidelity_after"].mean(),
        "std_F_after":               df["fidelity_after"].std(),
        "mean_F_improvement":        (df["fidelity_after"] - df["fidelity_before"]).mean(),
        "frac_F_after_gt_0.9":       (df["fidelity_after"] > 0.9).mean(),
        # Correction accuracy
        "M_correct_before":          df["M_correct_before"].mean(),
        "M_correct_after":           df["M_correct_after"].mean(),
        # Detection
        "mean_detection_rate":       det_rates.mean(),
        "full_detection_rate":       (det_rates == 1.0).mean(),
        # Strategy breakdown (if present)
        "mean_n_equivocate":         df["n_equivocate"].mean() if "n_equivocate" in df else None,
        "mean_n_coherent_flip":      df["n_coherent_flip"].mean() if "n_coherent_flip" in df else None,
        "mean_n_withhold":           df["n_withhold"].mean() if "n_withhold" in df else None,
        # Config
        "noise_level":               df["noise_level"].iloc[0],
        "n_controllers":             df["n_controllers"].iloc[0],
        "max_byzantine":             df["max_byzantine"].iloc[0],
    }


def print_summary(stats: dict) -> None:
    bar = "=" * 58
    print(bar)
    print("SIMULATION SUMMARY")
    print(bar)
    for k, v in stats.items():
        if v is None:
            continue
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<35} {val}")
    print(bar)


def per_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for val, grp in df.groupby(group_col):
        s = compute_statistics(grp)
        s[group_col] = val
        rows.append(s)
    return pd.DataFrame(rows)
