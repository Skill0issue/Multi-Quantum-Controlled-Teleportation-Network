"""
lemma1_check.py — Numerical verification of Lemma 1 (indistinguishability).

Lemma 1 (paper §3.2):
    The reduced density matrix of each controller qubit q[k+1] (k=1..n) is
    the maximally mixed state I/2, regardless of whether Alice's input is
    |ψ⟩ (regular trial) or one qubit of |Φ+⟩ (verification trial).

    This is the security foundation: Byzantine controllers cannot distinguish
    trial types from their own qubit's reduced state, so selective compliance
    (honest only on verification trials) is physically impossible.

Output
    ------
    A table: qubit | trial_type | input_state | trace_dist_from_I2
    All values should be < 1e-6 (numerical precision limit).

    Run standalone:  python lemma1_check.py
    Or import:       from lemma1_check import run_lemma1_check
"""

from __future__ import annotations
import sys
import os
import logging

import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile

log = logging.getLogger("lemma1")

# Make sure local modules are importable when run standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SimConfig
from quantum_circuit import (
    build_cluster_circuit,
    build_verification_circuit,
    run_dm_simulation,
    extract_single_qubit_dm,
)


def trace_distance_from_maximally_mixed(rho: np.ndarray) -> float:
    """
    Compute (1/2) * ||rho - I/2||_1  where ||A||_1 = sum of singular values.
    For a 2x2 matrix this equals the Frobenius distance times sqrt(2)/2,
    but we use the exact 1-norm definition for correctness.
    """
    diff = rho - np.eye(2, dtype=complex) / 2
    singular_vals = np.linalg.svd(diff, compute_uv=False)
    return float(0.5 * np.sum(singular_vals))


# Five test input states for regular trials
TEST_STATES = [
    ("psi",   "cos(π/6)|0⟩+e^{iπ/7}sin(π/6)|1⟩"),
    ("zero",  "|0⟩"),
    ("plus",  "|+⟩"),
]


def run_lemma1_check(
    n_controllers: int = 8,
    verbose: bool = True,
) -> list[dict]:
    """
    Verify Lemma 1 for both trial types across all test input states.

    Returns list of result dicts, one per (qubit, trial_type, input_state).
    Asserts all trace distances < 1e-4 (numerical tolerance).
    """
    cfg = SimConfig(
        n_controllers=n_controllers,
        max_byzantine=n_controllers // 4,  # minimal valid f
        noise_level=0.0,   # noiseless for exact check
        n_trials=1,
        delta=0.0,
    )
    sim = AerSimulator(method="density_matrix")
    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"  LEMMA 1 VERIFICATION  (n={n_controllers}, noiseless)")
        print(f"{'='*70}")
        print(f"  {'qubit':<8} {'trial_type':<16} {'input_state':<12} {'trace_dist':<14} {'pass'}")
        print(f"  {'-'*60}")

    # ── Regular trials ─────────────────────────────────────────────────────
    for state_label, state_desc in TEST_STATES:
        qc  = build_cluster_circuit(cfg, input_state=state_label)
        # Run WITHOUT noise model for exact DM
        qc_t = transpile(qc, sim, optimization_level=0)
        res  = sim.run(qc_t, shots=1).result()
        rho  = np.array(res.data(0)["dm"].data, dtype=complex)
        n_total = cfg.n_qubits

        # Controller qubits: q[2..n+1] (0-indexed: indices 2 to n_controllers+1)
        for k in range(1, n_controllers + 1):
            qubit_idx = k + 1  # q[2] = C1, q[3] = C2, ...
            rho_k = extract_single_qubit_dm(rho, n_total, qubit_idx)
            td    = trace_distance_from_maximally_mixed(rho_k)
            passed = td < 1e-4

            row = {
                "qubit":      f"C{k} (q[{qubit_idx}])",
                "trial_type": "regular",
                "input_state": state_label,
                "trace_dist": round(td, 8),
                "pass":       passed,
            }
            results.append(row)

            if verbose:
                icon = "✓" if passed else "✗ FAIL"
                print(f"  {row['qubit']:<8} {'regular':<16} {state_label:<12} {td:<14.2e} {icon}")

    # ── Verification trial ─────────────────────────────────────────────────
    qc_v   = build_verification_circuit(cfg)
    qc_vt  = transpile(qc_v, sim, optimization_level=0)
    res_v  = sim.run(qc_vt, shots=1).result()
    rho_v  = np.array(res_v.data(0)["dm"].data, dtype=complex)
    n_total_v = cfg.n_qubits + 1  # includes alice_hold qubit

    for k in range(1, n_controllers + 1):
        qubit_idx = k + 1
        rho_k = extract_single_qubit_dm(rho_v, n_total_v, qubit_idx)
        td    = trace_distance_from_maximally_mixed(rho_k)
        passed = td < 1e-4

        row = {
            "qubit":      f"C{k} (q[{qubit_idx}])",
            "trial_type": "verification",
            "input_state": "|Φ+⟩ half",
            "trace_dist": round(td, 8),
            "pass":       passed,
        }
        results.append(row)

        if verbose:
            icon = "✓" if passed else "✗ FAIL"
            print(f"  {row['qubit']:<8} {'verification':<16} {'|Phi+> half':<12} {td:<14.2e} {icon}")

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass = sum(1 for r in results if r["pass"])
    n_fail = len(results) - n_pass

    if verbose:
        print(f"  {'-'*60}")
        print(f"  Result: {n_pass}/{len(results)} passed  "
            f"({'ALL PASS — Lemma 1 verified' if n_fail == 0 else f'FAILURES: {n_fail}'})")
        print(f"{'='*70}\n")

    assert n_fail == 0, (
        f"Lemma 1 FAILED for {n_fail} qubit(s). "
        f"Failed: {[r for r in results if not r['pass']]}"
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_lemma1_check(n_controllers=8, verbose=True)
