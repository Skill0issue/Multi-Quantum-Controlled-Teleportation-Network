"""
experiments/table4_noise_scaling.py — Fidelity and detection viability vs noise.

Paper Table 4 / §6.3: For p in [0,0.005,0.01,0.02,0.03,0.05]:
  - F_noise: mean fidelity on honest trials (no Byzantine)
  - Likelihood ratio: 1 + 4*F_noise / (1 - F_noise)
  - Viable: likelihood ratio > 12 (Bell verification useful)

Viability threshold: p ≈ 0.05 (F_noise = 3/4 → ratio = 4, borderline).

Outputs
-------
  results/table4_noise_scaling.csv
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import math
import pandas as pd
import numpy as np
from qiskit_aer import AerSimulator

from config import SimConfig
from simulator import EVBTCSTSimulator

log = logging.getLogger("table4")

VIABILITY_THRESHOLD = 12.0


def likelihood_ratio(F_noise: float) -> float:
    """1 + 4 * F_noise / (1 - F_noise)"""
    denom = max(1 - F_noise, 1e-9)
    return 1.0 + 4 * F_noise / denom


def theoretical_f_noise(p: float, n_qubits: int) -> float:
    """
    Rough theoretical F_noise for depolarising noise at rate p.
    F_noise = (1 - 4p/3)^n_sq approximation (single-qubit gates dominant).
    This is an approximation; simulation gives the exact value.
    """
    n_gates = 2 * n_qubits  # H gates + CZ gates (rough count)
    return max(0.0, (1 - 4 * p / 3) ** n_gates)


def run_table4(
    noise_values: list[float] = None,
    n_trials: int = 200,
    out_dir: str = "results",
    verbose: bool = True,
) -> pd.DataFrame:

    if noise_values is None:
        noise_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    if verbose:
        print(f"\n  {'p':>6}  {'F_noise_sim':>12}  {'F_noise_theory':>14}  "
              f"{'lik_ratio':>10}  {'viable':>8}")
        print(f"  {'-'*58}")

    for p in noise_values:
        cfg = SimConfig(
            n_controllers=8,
            max_byzantine=0,    # all honest to isolate noise effect
            noise_level=p,
            n_trials=n_trials,
            delta=0.0,          # no verification trials needed here
        )

        sim_obj = EVBTCSTSimulator(cfg)   # pre-computes DMs internally
        df, _ = sim_obj.run()

        reg = df[df["trial_type"] == "regular"]
        F_sim   = float(reg["fidelity_after"].mean()) if len(reg) else 0.0
        F_theo  = theoretical_f_noise(p, cfg.n_qubits)
        lr      = likelihood_ratio(F_sim)
        viable  = lr >= VIABILITY_THRESHOLD

        row = {
            "noise_p":          p,
            "F_noise_sim":      round(F_sim,  4),
            "F_noise_theory":   round(F_theo, 4),
            "likelihood_ratio": round(lr, 2),
            "viable":           viable,
            "n_trials":         n_trials,
        }
        rows.append(row)

        if verbose:
            v_str = "YES" if viable else "NO "
            print(f"  {p:>6.3f}  {F_sim:>12.4f}  {F_theo:>14.4f}  "
                  f"{lr:>10.2f}  {v_str:>8}")

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "table4_noise_scaling.csv")
    df_out.to_csv(out_path, index=False)
    if verbose:
        print(f"\nSaved: {out_path}")
    return df_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("\nTable 4 — Noise Scaling and Viability")
    print("=" * 60)
    df = run_table4(n_trials=100, verbose=True)
    print("\n", df.to_string(index=False))
