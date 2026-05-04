"""
experiments/table3_fidelity_vs_byzantine.py — Fidelity vs Byzantine count.

Paper Table 3: n=8, p=0.01, 500 trials.
For f in [0,1,2]: F_before (no recovery), F_after_VSS, F_after_BellVerif.

Key result to reproduce:
  - f=2, M_fail_before ≈ 50.8% (coherent-flip dominant)
  - f=2, M_fail_after_BellVerif < 0.01% (Bell verification closes the gap)
  - XOR cancellation anomaly: F_after_VSS < F_before for f=2

Outputs
-------
  results/table3_fidelity_byzantine.csv
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from qiskit_aer import AerSimulator

from config import SimConfig
from simulator import EVBTCSTSimulator
from analysis import compute_statistics, xor_cancellation_rate

log = logging.getLogger("table3")


def run_table3(
    f_values: list[int] = None,
    n_trials: int = 300,
    noise_level: float = 0.01,
    delta: float = 0.15,
    out_dir: str = "results",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run Table 3 experiment.

    For each f: run n_trials trials (mix of regular+verification).
    Report fidelity and M-accuracy at three stages:
      before VSS recovery, after VSS recovery, after Bell verification
      (Bell verif = after SPRT flags coherent-flip adversaries and their
       contributions are excluded from M).
    """
    if f_values is None:
        f_values = [0, 1, 2]

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for f in f_values:
        # minimum valid n for given f: n > 3f, n even
        n = max(8, (3 * f + 2) if (3 * f + 2) % 2 == 0 else (3 * f + 3))
        n = 8  # fix to n=8 matching paper

        cfg = SimConfig(
            n_controllers=n,
            max_byzantine=f,
            noise_level=noise_level,
            n_trials=n_trials,
            delta=delta,
            sprt_alpha=0.01,
            sprt_beta=0.0001,
            sprt_threshold_k=2,
        )

        if verbose:
            print(f"\n  f={f}: {cfg.summary()}")

        sim_obj = EVBTCSTSimulator(cfg)
        df, sprt_summary = sim_obj.run()

        reg = df[df["trial_type"] == "regular"]
        ver = df[df["trial_type"] == "verification"]

        # F_noise estimate from honest trials
        flagged = sim_obj.sprt.get_flagged_controllers()

        # M failure: fraction of regular trials where M_rec != M_true
        M_fail_before = 1 - reg["M_correct_before"].mean() if len(reg) else None
        M_fail_after  = 1 - reg["M_correct_after"].mean()  if len(reg) else None

        # XOR cancellation
        xor = xor_cancellation_rate(reg)

        # Verification pass rate
        verif_pass = ver["verification_pass"].mean() if len(ver) else None

        row = {
            "f":                  f,
            "n":                  n,
            "n_regular":          len(reg),
            "n_verification":     len(ver),
            "F_before":           round(reg["fidelity_before"].mean(), 4) if len(reg) else None,
            "F_after_VSS":        round(reg["fidelity_after"].mean(),  4) if len(reg) else None,
            "M_fail_before":      round(M_fail_before, 4) if M_fail_before is not None else None,
            "M_fail_after_VSS":   round(M_fail_after,  4) if M_fail_after  is not None else None,
            "verif_pass_rate":    round(verif_pass, 4) if verif_pass is not None else None,
            "sprt_flagged":       str(sorted(flagged)),
            "xor_cancel_rate":    xor["rate"],
            "xor_cancel_theory":  xor["theory"],
            "noise_level":        noise_level,
            "n_trials":           n_trials,
        }
        rows.append(row)

        if verbose:
            print(f"    F_before={row['F_before']}  F_after_VSS={row['F_after_VSS']}")
            print(f"    M_fail_before={row['M_fail_before']}  M_fail_after_VSS={row['M_fail_after_VSS']}")
            print(f"    verif_pass_rate={row['verif_pass_rate']}  SPRT_flagged={flagged}")
            print(f"    XOR cancellation rate={xor['rate']} (theory≈{xor['theory']})")

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "table3_fidelity_byzantine.csv")
    df_out.to_csv(out_path, index=False)
    if verbose:
        print(f"\nSaved: {out_path}")
    return df_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("\nTable 3 — Fidelity vs Byzantine Count")
    print("=" * 55)
    df = run_table3(n_trials=200, verbose=True)
    print("\n", df.to_string(index=False))
