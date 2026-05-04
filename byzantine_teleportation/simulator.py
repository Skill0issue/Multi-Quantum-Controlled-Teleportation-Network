"""
simulator.py — Full per-trial protocol orchestration.

Trial flow
----------
1.  Build cluster circuit (no measurements), run ONE DM simulation.
2.  Sample measurement outcomes coherently from that DM.
3.  Extract Bob's conditional density matrix.
4.  Apply Byzantine corruption model (unified strategy per node).
5.  VSS Phase 1: Pedersen commitments.
6.  VSS Phase 2: Hash commit + share distribution.
7.  VSS Phase 3: Hash verification / complaints.
8.  VSS Phase 4: Pedersen aggregate check + individual challenge.
9.  Build detection set (union of all mechanisms).
10. Recover corrupted measurements for detected nodes.
11. Compute correction bits M_X, M_Z from recovered measurements.
12. Compute fidelity before and after recovery.
"""

from __future__ import annotations
import logging
import random

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from config import SimConfig
from quantum_circuit import (
    build_cluster_circuit,
    run_dm_simulation,
    build_noise_model,
    get_input_state_vector,
    sample_outcomes,
    extract_bob_dm,
    compute_correction_bits,
    compute_fidelity,
)
from byzantine import select_byzantine_nodes, corrupt_direct_reports
from secret_sharing import (
    VSSState,
    pedersen_commit_phase,
    hash_commit_and_share,
    verify_received_shares,
    pedersen_aggregate_check,
    build_detection_set,
    recover_measurements,
)

log = logging.getLogger("simulator")


def run_trial(trial_id: int, cfg: SimConfig, sim: AerSimulator, nm) -> dict:
    # ── Trap setup ────────────────────────────────────────────────────────
    is_trap     = random.random() < cfg.trap_probability
    input_label = random.choice(["zero", "plus"]) if is_trap else "psi"
    ideal_psi   = get_input_state_vector(input_label)

    # ── 1-2. Single DM simulation + coherent outcome sampling ─────────────
    qc   = build_cluster_circuit(cfg, input_state=input_label)
    rho  = run_dm_simulation(cfg, qc, sim, nm)
    n_meas = cfg.n_qubits - 1    # q[0..n+1], skip Bob

    # circuit_outcomes : true quantum results (DM collapses on these)
    # reported_outcomes: circuit_outcomes + readout error (controller's display)
    circuit_outcomes, reported_outcomes, rho_post = sample_outcomes(
        rho, cfg.n_qubits, n_meas,
        # At noise_level=0 suppress readout errors too — this gives an idealised
        # experiment where the only imperfection is Byzantine behaviour.
        # At noise_level>0 the physical readout errors model detector imperfection.
        ro_01=cfg.readout_error_01 if cfg.noise_level > 0 else 0.0,
        ro_10=cfg.readout_error_10 if cfg.noise_level > 0 else 0.0,
    )
    # all_outcomes layout: [m_input, m_alice, m_C1, …, m_Cn]
    circuit_input   = circuit_outcomes[0]
    circuit_alice   = circuit_outcomes[1]
    circuit_ctrl    = circuit_outcomes[2: 2 + cfg.n_controllers]

    reported_input  = reported_outcomes[0]
    reported_alice  = reported_outcomes[1]
    reported_ctrl_base = reported_outcomes[2: 2 + cfg.n_controllers]  # pre-Byzantine

    # ── 3. Bob's conditional density matrix ───────────────────────────────
    # rho_post collapsed on circuit_outcomes → Bob holds the correct byproduct state
    rho_bob = extract_bob_dm(rho_post, cfg.n_qubits)

    # ── 4. Byzantine corruption ───────────────────────────────────────────
    # Byzantine nodes corrupt the REPORTED bits (what controllers broadcast),
    # not the circuit bits. M_true uses circuit bits; M_rep uses corrupted reported bits.
    byz = select_byzantine_nodes(cfg.n_controllers, cfg.max_byzantine)
    reported_ctrl = corrupt_direct_reports(reported_ctrl_base, byz)

    # M_true: correction bits from circuit outcomes (what Bob physically needs)
    circuit_all  = [circuit_input,  circuit_alice]  + circuit_ctrl
    # M_rep: correction bits from Byzantine-corrupted reported outcomes
    reported_all = [reported_input, reported_alice] + reported_ctrl

    M_X_true, M_Z_true = compute_correction_bits(circuit_all)
    M_X_rep,  M_Z_rep  = compute_correction_bits(reported_all)

    # ── 5-9. VSS commit-reveal protocol ──────────────────────────────────
    vss = VSSState(cfg.n_controllers, cfg.threshold)

    pedersen_commit_phase(reported_ctrl, cfg.n_controllers, byz, vss)
    hash_commit_and_share(reported_ctrl, cfg.n_controllers, cfg.threshold, byz, vss)
    verify_received_shares(cfg.n_controllers, vss)
    pedersen_aggregate_check(cfg.n_controllers, cfg.threshold, vss.hash_complaints | vss.withhold_absent, vss)
    detected = build_detection_set(vss)

    # ── 10-11. Recovery + correction ─────────────────────────────────────
    recovered_ctrl = recover_measurements(reported_ctrl, cfg.n_controllers, cfg.threshold, detected, vss)
    recovered_all  = [reported_input, reported_alice] + recovered_ctrl
    M_X_rec, M_Z_rec = compute_correction_bits(recovered_all)

    # ── 12. Fidelity before and after recovery ────────────────────────────
    F_before = compute_fidelity(rho_bob, M_X_rep, M_Z_rep, ideal_psi)
    F_after  = compute_fidelity(rho_bob, M_X_rec, M_Z_rec, ideal_psi)

    # ── 13. Trap detection (uses POST-correction fidelity on known input) ─
    # For trap teleportations the input state is known. If the recovered
    # fidelity falls below threshold, flag Byzantine activity.
    # Threshold 0.75 is conservative: noise_level=0.02 still gives F≈0.75+
    trap_detected = False
    if is_trap:
        trap_threshold = 0.75
        trap_detected = F_after < trap_threshold
        if trap_detected:
            log.debug(f"  [Trap] F_after={F_after:.3f} < {trap_threshold} → Byzantine suspected")

    if cfg.verbose:
        log.debug(
            f"[{trial_id}] byz={dict(byz)} det={detected} trap={trap_detected} "
            f"F={F_before:.3f}→{F_after:.3f}"
        )

    strat_counts = {"coherent_flip": 0, "equivocate": 0, "withhold": 0}
    for s in byz.values():
        strat_counts[s] += 1

    return {
        "trial_id":           trial_id,
        "byzantine_nodes":    str(sorted(byz.keys())),
        "byzantine_strategies": str({k: v for k, v in byz.items()}),
        "n_coherent_flip":    strat_counts["coherent_flip"],
        "n_equivocate":       strat_counts["equivocate"],
        "n_withhold":         strat_counts["withhold"],
        "detected_nodes":     str(sorted(detected)),
        "n_detected":         len(detected),
        "n_undetected_byz":   len(set(byz.keys()) - detected),
        "true_all_outcomes":  str(circuit_all),
        "reported_ctrl":      str(reported_ctrl),
        "recovered_ctrl":     str(recovered_ctrl),
        "M_X_true":   M_X_true,   "M_Z_true":  M_Z_true,
        "M_X_reported": M_X_rep,  "M_Z_reported": M_Z_rep,
        "M_X_recovered": M_X_rec, "M_Z_recovered": M_Z_rec,
        "M_correct_before": int(M_X_rep == M_X_true and M_Z_rep == M_Z_true),
        "M_correct_after":  int(M_X_rec == M_X_true and M_Z_rec == M_Z_true),
        "fidelity_before":  F_before,
        "fidelity_after":   F_after,
        "noise_level":      cfg.noise_level,
        "n_controllers":    cfg.n_controllers,
        "max_byzantine":    cfg.max_byzantine,
        "is_trap":          is_trap,
        "trap_detected":    trap_detected,
        "input_state":      input_label,
    }


class ByzantineQTSimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.nm  = build_noise_model(cfg)
        self.sim = AerSimulator(method="density_matrix")
        log.info(
            f"ByzantineQTSimulator: n={cfg.n_controllers} f={cfg.max_byzantine} "
            f"N_qubits={cfg.n_qubits} noise={cfg.noise_level} trials={cfg.n_trials}"
        )

    def run(self) -> pd.DataFrame:
        records = []
        log_every = max(1, self.cfg.n_trials // 10)
        for t in range(self.cfg.n_trials):
            try:
                records.append(run_trial(t, self.cfg, self.sim, self.nm))
            except Exception as e:
                log.error(f"Trial {t} failed: {e}", exc_info=True)
            if t % log_every == 0:
                log.info(f"  {t}/{self.cfg.n_trials} trials done")
        df = pd.DataFrame(records)
        if len(df):
            log.info(
                f"Done — F_before={df.fidelity_before.mean():.3f} "
                f"F_after={df.fidelity_after.mean():.3f} "
                f"M_correct_after={df.M_correct_after.mean():.3f}"
            )
        return df
