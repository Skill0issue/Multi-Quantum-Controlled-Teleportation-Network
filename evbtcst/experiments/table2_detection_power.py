"""
experiments/table2_detection_power.py — Detection power vs verification trials k.

Paper Theorem 1 / Table 2: at p=0.01, alpha=0.01, k=7 verification trials,
detection power against a coherent-flip adversary > 0.9999.

Method
------
For each k in [1,2,3,5,7,10,15,20]:
  Run N_BATCHES independent batches, each with exactly k verification trials
  (force_trial_type='verification') and exactly 1 coherent-flip adversary.
  Count batches where the adversary is flagged by SPRT.

Also compute theoretical detection power using the binomial model from §3.3.

Outputs
-------
  results/table2_detection_power.csv
  console table
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from config import SimConfig
from quantum_circuit import build_noise_model, build_verification_circuit, run_dm_simulation, compute_correction_bits
from sprt import SPRTTracker, compute_per_controller_flip_contributions
from simulator import run_trial

log = logging.getLogger("table2")


# ── Module-level worker (must be at top level for ProcessPoolExecutor on Windows) ──

def _run_k_worker(args: tuple) -> dict:
    """
    Worker function for one k-value.  Reads the pre-computed DM from shared
    memory (no large serialisation through pipes — avoids WinError 1450).
    """
    (k, n_batches, n_fpr_batches,
     n_controllers, sprt_alpha, sprt_beta, sprt_threshold_k,
     F_noise_est, shm_name, rho_shape, rho_dtype_str,
     noise_level, gf_prime,
     p1_cal, p0_cal, p_flip_hon_cal, p_flip_adv_cal) = args

    # Attach to shared memory — read-only view, no copy
    shm = SharedMemory(name=shm_name)
    rho_v = np.ndarray(rho_shape, dtype=np.dtype(rho_dtype_str), buffer=shm.buf)
    rho_cache = {"verification": rho_v}

    cfg_byz = SimConfig(
        n_controllers=n_controllers, max_byzantine=1,
        noise_level=noise_level, n_trials=1, delta=1.0,
        sprt_alpha=sprt_alpha, sprt_beta=sprt_beta,
        sprt_threshold_k=sprt_threshold_k, gf_prime=gf_prime,
    )
    cfg_hon = SimConfig(
        n_controllers=n_controllers, max_byzantine=0,
        noise_level=noise_level, n_trials=1, delta=1.0,
        sprt_alpha=sprt_alpha, sprt_beta=sprt_beta,
        sprt_threshold_k=sprt_threshold_k, gf_prime=gf_prime,
    )

    # Detection power (Byzantine present)
    detected = 0
    for _ in range(n_batches):
        tracker = SPRTTracker(
            n_controllers=n_controllers, F_noise=F_noise_est,
            alpha=sprt_alpha, beta=sprt_beta, min_k=sprt_threshold_k,
            p_flip_honest=p_flip_hon_cal, p_flip_adv=p_flip_adv_cal,
        )
        byz_node     = random.randint(1, n_controllers)
        byz_override = {byz_node: "coherent_flip"}
        for t in range(k):
            run_trial(t, cfg_byz, rho_cache=rho_cache,
                      sprt_tracker=tracker,
                      force_trial_type="verification",
                      byzantine_override=byz_override)
        if byz_node in tracker.get_flagged_controllers():
            detected += 1

    # False positive rate (all honest)
    fp = 0
    for _ in range(n_fpr_batches):
        tracker = SPRTTracker(
            n_controllers=n_controllers, F_noise=F_noise_est,
            alpha=sprt_alpha, beta=sprt_beta, min_k=sprt_threshold_k,
            p_flip_honest=p_flip_hon_cal, p_flip_adv=p_flip_adv_cal,
        )
        for t in range(k):
            run_trial(t, cfg_hon, rho_cache=rho_cache,
                      sprt_tracker=tracker,
                      force_trial_type="verification",
                      byzantine_override={})
        if len(tracker.get_flagged_controllers()) > 0:
            fp += 1

    shm.close()   # detach from shared memory (don't unlink — owner does that)
    return {"k": k, "detected": detected, "fp": fp}


def theoretical_detection_power(
    k: int,
    F_noise: float,
    alpha: float = 0.01,
    beta_target: float = 0.0001,
    p1: float = None,
    p0: float = None,
) -> float:
    """
    Theoretical detection power for k verification trials.

    P(Phi+ | correct) = F_noise + (1-F_noise)/4  (or p1 if provided)
    P(Phi+ | coherent-flip) = (1-F_noise)/4       (or p0 if provided)

    Detection: count how many Phi+ outcomes in k trials.
    Reject null (honest) if count <= s*, where s* = max{s : P(S<=s|honest) <= alpha}.
    Power = P(S <= s* | adversary).
    """
    if p1 is None:
        p1 = F_noise + (1 - F_noise) / 4     # P(Phi+ | honest)
    if p0 is None:
        p0 = (1 - F_noise) / 4               # P(Phi+ | adversary)

    # Find rejection threshold s* under null hypothesis (honest)
    # S = number of Phi+ outcomes in k trials
    # Under honest: S ~ Binomial(k, p1)
    from math import comb

    def binom_cdf(s_max, n, p):
        return sum(comb(n, s) * p**s * (1-p)**(n-s) for s in range(s_max+1))

    s_star = -1
    for s in range(k + 1):
        if binom_cdf(s, k, p1) <= alpha:
            s_star = s

    if s_star < 0:
        return 0.0  # no rejection region at this alpha

    # Power = P(S <= s* | adversary)
    power = binom_cdf(s_star, k, p0)
    return float(power)


def run_table2(
    k_values: list[int] = None,
    n_batches: int = 200,
    noise_level: float = 0.01,
    out_dir: str = "results",
    verbose: bool = True,
    n_workers: int = None,
) -> pd.DataFrame:
    """
    Run Table 2 experiment.

    Optimisations
    -------------
    1. DM caching: verification circuit is simulated ONCE; all k-batches
       reuse the same density matrix (only Born-rule sampling varies).
    2. ProcessPoolExecutor: each k-value is fully independent and runs in
       a separate subprocess, giving near-linear speedup with CPU count.

    For each k:
      - n_batches   with 1 coherent-flip adversary → empirical detection power
      - n_fpr_batches with 0 Byzantine nodes       → false positive rate
    """
    if k_values is None:
        k_values = [1, 2, 3, 5, 7, 10, 15, 20]

    n_fpr_batches = max(20, n_batches // 5)
    n_controllers = 8
    sprt_alpha, sprt_beta, sprt_threshold_k = 0.01, 0.0001, 1

    os.makedirs(out_dir, exist_ok=True)

    # ── Pre-compute verification DM once (expensive, ~4 s) ─────────────────
    cfg_byz = SimConfig(
        n_controllers=n_controllers, max_byzantine=1,
        noise_level=noise_level, n_trials=1, delta=1.0,
        sprt_alpha=sprt_alpha, sprt_beta=sprt_beta,
        sprt_threshold_k=sprt_threshold_k,
    )
    nm  = build_noise_model(cfg_byz)
    sim = AerSimulator(method="density_matrix", precision="single")

    if verbose:
        print("  Pre-computing verification DM…", flush=True)
    qc_v  = build_verification_circuit(cfg_byz)
    rho_v = run_dm_simulation(cfg_byz, qc_v, sim, nm)

    # ── Calibrate SPRT blame probabilities from the DM directly ──────────────
    # The SPRT needs P(controller blamed | honest) and P(controller blamed | adversary).
    # These differ from P(fail) because failures can come from either parity group.
    if verbose:
        print("  Calibrating SPRT blame rates from simulation…", flush=True)

    cfg_hon = SimConfig(
        n_controllers=n_controllers, max_byzantine=0,
        noise_level=noise_level, n_trials=1, delta=1.0,
        sprt_alpha=sprt_alpha, sprt_beta=sprt_beta,
        sprt_threshold_k=sprt_threshold_k,
    )
    rho_cache_cal = {"verification": rho_v}
    N_cal = 300

    # Honest: P(controller 1 blamed per trial)
    hon_blamed = 0
    p1_sum = 0.0
    for t in range(N_cal):
        r = run_trial(t, cfg_hon, rho_cache=rho_cache_cal,
                      force_trial_type="verification", byzantine_override={})
        p1_sum += float(r["p_phi_plus"])
        if not r["verification_pass"]:
            circuit_all = eval(r["true_all_outcomes"])
            M_X_t, M_Z_t = compute_correction_bits(circuit_all)
            contrib = compute_per_controller_flip_contributions(
                circuit_all, M_X_t, M_Z_t, r["M_X_rec"], r["M_Z_rec"], n_controllers)
            if 1 in contrib:
                hon_blamed += 1
    p_flip_hon = hon_blamed / N_cal
    p1_actual = p1_sum / N_cal

    # Adversary C1: P(controller 1 blamed per trial)
    adv_blamed = 0
    p0_sum = 0.0
    byz_cal = {1: "coherent_flip"}
    for t in range(N_cal):
        r = run_trial(t, cfg_byz, rho_cache=rho_cache_cal,
                      force_trial_type="verification", byzantine_override=byz_cal)
        p0_sum += float(r["p_phi_plus"])
        if not r["verification_pass"]:
            circuit_all = eval(r["true_all_outcomes"])
            M_X_t, M_Z_t = compute_correction_bits(circuit_all)
            contrib = compute_per_controller_flip_contributions(
                circuit_all, M_X_t, M_Z_t, r["M_X_rec"], r["M_Z_rec"], n_controllers)
            if 1 in contrib:
                adv_blamed += 1
    p_flip_adv_cal = adv_blamed / N_cal
    p0_actual = p0_sum / N_cal

    # Back out F_noise from calibrated p1: p1 = F + (1-F)/4  →  F = (4p1-1)/3
    F_noise_est = max(0.0, min(1.0, (4 * p1_actual - 1) / 3))
    if verbose:
        print(f"  p_flip_honest={p_flip_hon:.4f}  p_flip_adv={p_flip_adv_cal:.4f}  "
              f"p1={p1_actual:.4f}  p0={p0_actual:.4f}  "
              f"F_noise_est={F_noise_est:.4f}", flush=True)

    # ── Share rho via shared memory to avoid 268 MB pipe transfers ─────────
    shm = SharedMemory(create=True, size=rho_v.nbytes)
    shm_arr = np.ndarray(rho_v.shape, dtype=rho_v.dtype, buffer=shm.buf)
    np.copyto(shm_arr, rho_v)

    worker_args = [
        (k, n_batches, n_fpr_batches,
         n_controllers, sprt_alpha, sprt_beta, sprt_threshold_k,
         F_noise_est, shm.name, rho_v.shape, rho_v.dtype.str,
         noise_level, cfg_byz.gf_prime,
         p1_actual, p0_actual, p_flip_hon, p_flip_adv_cal)
        for k in k_values
    ]

    # ── Parallel execution over k-values ───────────────────────────────────
    if n_workers is None:
        import os as _os
        import sys as _sys
        cpu = _os.cpu_count() or 4
        # Windows spawn context: Qiskit imports in each worker are heavy; cap at 2
        # to avoid process crashes from simultaneous heavy imports.
        safe_max = 2 if _sys.platform == "win32" else cpu
        n_workers = min(len(k_values), safe_max)

    if verbose:
        print(f"  Running {len(k_values)} k-values on {n_workers} workers…",
              flush=True)

    results_by_k: dict = {}
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_k_worker, a): a[0] for a in worker_args}
            for fut in as_completed(futures):
                r = fut.result()
                results_by_k[r["k"]] = r
                if verbose:
                    k       = r["k"]
                    power   = r["detected"] / n_batches
                    fpr     = r["fp"] / n_fpr_batches
                    theory  = theoretical_detection_power(k, F_noise_est, alpha=0.01,
                                                          p1=p1_actual, p0=p0_actual)
                    print(f"  k={k:2d}  empirical={power:.4f}  "
                          f"theory={theory:.4f}  FPR={fpr:.4f}", flush=True)
    finally:
        shm.close()
        shm.unlink()   # free shared memory once all workers are done

    rows = []
    for k in k_values:
        r          = results_by_k[k]
        emp_power  = r["detected"] / n_batches
        emp_fpr    = r["fp"] / n_fpr_batches
        theo_power = theoretical_detection_power(k, F_noise_est, alpha=0.01,
                                                  p1=p1_actual, p0=p0_actual)
        rows.append({
            "k":                   k,
            "empirical_power":     round(emp_power,  4),
            "theoretical_power":   round(theo_power, 4),
            "false_positive_rate": round(emp_fpr,    4),
            "n_batches":           n_batches,
            "n_fpr_batches":       n_fpr_batches,
            "F_noise_est":         round(F_noise_est, 4),
        })

    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    out_path = os.path.join(out_dir, "table2_detection_power.csv")
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"\nSaved: {out_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("\nTable 2 — Detection Power vs Verification Trials k")
    print("=" * 55)
    df = run_table2(n_batches=100, verbose=True)
    print("\n", df.to_string(index=False))
