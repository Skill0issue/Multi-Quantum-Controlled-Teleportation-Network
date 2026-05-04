"""
simulator.py — Full per-trial protocol orchestration (EVBTCST).

Trial types
-----------
regular      : standard cluster-state teleportation (N qubits).
               Input |ψ⟩.  Output: fidelity before/after VSS recovery.

verification : Bell-pair verification trial (N+1 qubits).
               Input: half of |Φ+⟩ on q[0]; Alice retains q[N].
               After correction: Alice+Bob measure in Bell basis.
               Outcome Φ+ → correct. Any other → Pauli error detected.
               Controllers cannot distinguish from regular trials (Lemma 1).

Trial flow (both types)
-----------------------
1.  Build circuit, run ONE DM simulation.
2.  Sample measurement outcomes coherently (q[0..N-2]).
3.  Extract Bob's conditional DM.
4.  Byzantine corruption model.
5-9. VSS commit-reveal (Pedersen + hash).
10. Build detection set, recover measurements.
11. Compute M_X, M_Z from recovered outcomes.
12. Fidelity before/after (regular) OR Bell measurement (verification).
13. Update SPRT with verification outcome.
"""

from __future__ import annotations
import logging
import random
from typing import Optional

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from config import SimConfig
from quantum_circuit import (
    build_cluster_circuit,
    build_verification_circuit,
    run_dm_simulation,
    build_noise_model,
    get_input_state_vector,
    sample_outcomes,
    extract_bob_dm,
    extract_two_qubit_dm,
    compute_correction_bits,
    compute_fidelity,
    bell_basis_measurement,
    apply_bob_correction_to_dm,
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
from sprt import SPRTTracker, compute_per_controller_flip_contributions

log = logging.getLogger("simulator")


def run_trial(
    trial_id: int,
    cfg: SimConfig,
    sim: Optional[AerSimulator] = None,
    nm=None,
    sprt_tracker: Optional[SPRTTracker] = None,
    force_trial_type: Optional[str] = None,
    byzantine_override: Optional[dict] = None,
    rho_cache: Optional[dict] = None,
) -> dict:
    """
    rho_cache : dict mapping cache_key → pre-computed numpy DM.
                If the relevant key is present, the expensive DM simulation
                is skipped entirely.  Enabled by EVBTCSTSimulator by default.
    """
    """
    Run one protocol trial.

    Parameters
    ----------
    force_trial_type : 'regular' | 'verification' | None (random by delta)
    sprt_tracker     : if provided, updated with verification outcome
    """
    # ── Trial type selection ───────────────────────────────────────────────
    if force_trial_type is not None:
        trial_type = force_trial_type
    else:
        trial_type = "verification" if random.random() < cfg.delta else "regular"

    is_verification = (trial_type == "verification")
    is_trap = (not is_verification) and (random.random() < cfg.trap_probability)

    # ── Input state ────────────────────────────────────────────────────────
    if is_verification:
        input_label = "verification"
        ideal_psi   = None          # not used for fidelity on verification trials
    elif is_trap:
        input_label = random.choice(["zero", "plus"])
        ideal_psi   = get_input_state_vector(input_label)
    else:
        input_label = "psi"
        ideal_psi   = get_input_state_vector(input_label)

    # ── 1-2. DM simulation + coherent outcome sampling ─────────────────────
    if is_verification:
        n_total   = cfg.n_qubits + 1   # N + alice_hold qubit
        cache_key = "verification"
    else:
        n_total   = cfg.n_qubits
        cache_key = f"regular_{input_label}"

    if rho_cache is not None and cache_key in rho_cache:
        rho = rho_cache[cache_key]          # fast: reuse pre-computed DM
    else:
        if is_verification:
            qc = build_verification_circuit(cfg)
        else:
            qc = build_cluster_circuit(cfg, input_state=input_label)
        rho = run_dm_simulation(cfg, qc, sim, nm)  # slow: ~4 s
    n_meas = cfg.n_qubits - 1    # q[0..N-2] are measured; q[N-1]=Bob is not

    circuit_outcomes, reported_outcomes, rho_post = sample_outcomes(
        rho, n_total, n_meas,
        ro_01=cfg.readout_error_01 if cfg.noise_level > 0 else 0.0,
        ro_10=cfg.readout_error_10 if cfg.noise_level > 0 else 0.0,
    )

    circuit_input      = circuit_outcomes[0]
    circuit_alice      = circuit_outcomes[1]
    circuit_ctrl       = circuit_outcomes[2: 2 + cfg.n_controllers]
    reported_input     = reported_outcomes[0]
    reported_alice     = reported_outcomes[1]
    reported_ctrl_base = reported_outcomes[2: 2 + cfg.n_controllers]

    # ── 3. Bob's conditional DM ────────────────────────────────────────────
    # For verification: rho_post has n_total=N+1 qubits; Bob is still q[N-1]
    rho_bob = extract_bob_dm(rho_post, n_total)

    # ── 4. Byzantine corruption ────────────────────────────────────────────
    if byzantine_override is not None:
        byz = byzantine_override
    else:
        byz = select_byzantine_nodes(cfg.n_controllers, cfg.max_byzantine)
    reported_ctrl = corrupt_direct_reports(reported_ctrl_base, byz)

    circuit_all  = [circuit_input, circuit_alice] + circuit_ctrl
    reported_all = [reported_input, reported_alice] + reported_ctrl

    M_X_true, M_Z_true = compute_correction_bits(circuit_all)
    M_X_rep,  M_Z_rep  = compute_correction_bits(reported_all)

    # ── 5-9. VSS commit-reveal ─────────────────────────────────────────────
    vss = VSSState(cfg.n_controllers, cfg.threshold)
    pedersen_commit_phase(reported_ctrl, cfg.n_controllers, byz, vss)
    hash_commit_and_share(reported_ctrl, cfg.n_controllers, cfg.threshold, byz, vss)
    verify_received_shares(cfg.n_controllers, vss)
    pedersen_aggregate_check(cfg.n_controllers, cfg.threshold,
                              vss.hash_complaints | vss.withhold_absent, vss)
    detected = build_detection_set(vss)

    # ── 10-11. Recovery + correction ──────────────────────────────────────
    recovered_ctrl = recover_measurements(
        reported_ctrl, cfg.n_controllers, cfg.threshold, detected, vss)
    recovered_all  = [reported_input, reported_alice] + recovered_ctrl
    M_X_rec, M_Z_rec = compute_correction_bits(recovered_all)

    # ── 12. Outcome computation ────────────────────────────────────────────
    bell_outcome      = None
    bell_probs        = None
    verification_pass = None
    p_phi_plus        = None
    F_before          = None
    F_after           = None
    trap_detected     = False

    if is_verification:
        # Alice's retained qubit is q[N] (index cfg.n_qubits in the N+1 qubit DM)
        alice_hold = cfg.n_qubits
        bob_idx    = cfg.n_qubits - 1

        # Extract joint (alice_hold, bob) 2-qubit DM AFTER collapse on measurements
        rho_2q = extract_two_qubit_dm(rho_post, n_total, alice_hold, bob_idx)

        # Apply Bob's correction to the joint DM (correction acts only on Bob's qubit)
        # Correction: U_bob = Z^{M_Z} X^{M_X}; applied as (I_alice ⊗ U_bob) ρ (I_alice ⊗ U_bob)†
        X = np.array([[0,1],[1,0]], dtype=complex)
        Z = np.array([[1,0],[0,-1]], dtype=complex)
        U_bob = np.eye(2, dtype=complex)
        if M_X_rec: U_bob = X @ U_bob
        if M_Z_rec: U_bob = Z @ U_bob

        # In the 4x4 joint DM (bit0=alice_hold, bit1=bob), Bob is bit1
        U_joint = np.kron(np.eye(2, dtype=complex), U_bob)
        rho_2q_corrected = U_joint @ rho_2q @ U_joint.conj().T

        bell_outcome, bell_probs = bell_basis_measurement(rho_2q_corrected)
        verification_pass = (bell_outcome == "Phi+")
        p_phi_plus = bell_probs.get("Phi+", 0.0)

        # SPRT update
        if sprt_tracker is not None:
            if not verification_pass:
                contributing = compute_per_controller_flip_contributions(
                    circuit_all, M_X_true, M_Z_true, M_X_rec, M_Z_rec,
                    cfg.n_controllers,
                )
            else:
                contributing = []
            sprt_tracker.update_verification_trial(
                bell_outcome, contributing, trial_id=trial_id)

    else:
        # Regular / trap trial
        F_before = compute_fidelity(rho_bob, M_X_rep,  M_Z_rep,  ideal_psi)
        F_after  = compute_fidelity(rho_bob, M_X_rec,  M_Z_rec,  ideal_psi)
        if is_trap:
            trap_detected = F_after < 0.75

    # ── Build result record ────────────────────────────────────────────────
    strat_counts = {"coherent_flip": 0, "equivocate": 0, "withhold": 0}
    for s in byz.values():
        strat_counts[s] += 1

    if cfg.verbose:
        log.debug(
            f"[{trial_id}|{trial_type}] byz={dict(byz)} det={detected} "
            f"bell={bell_outcome} F={F_before}→{F_after}"
        )

    return {
        "trial_id":            trial_id,
        "trial_type":          trial_type,
        "input_state":         input_label,
        # Byzantine
        "byzantine_nodes":     str(sorted(byz.keys())),
        "byzantine_strategies":str({k: v for k, v in byz.items()}),
        "n_coherent_flip":     strat_counts["coherent_flip"],
        "n_equivocate":        strat_counts["equivocate"],
        "n_withhold":          strat_counts["withhold"],
        "detected_nodes":      str(sorted(detected)),
        "n_detected":          len(detected),
        "n_undetected_byz":    len(set(byz.keys()) - detected),
        # Outcomes
        "true_all_outcomes":   str(circuit_all),
        "reported_ctrl":       str(reported_ctrl),
        "recovered_ctrl":      str(recovered_ctrl),
        "M_X_true":  M_X_true,  "M_Z_true":  M_Z_true,
        "M_X_rep":   M_X_rep,   "M_Z_rep":   M_Z_rep,
        "M_X_rec":   M_X_rec,   "M_Z_rec":   M_Z_rec,
        "M_correct_before": int(M_X_rep == M_X_true and M_Z_rep == M_Z_true),
        "M_correct_after":  int(M_X_rec == M_X_true and M_Z_rec == M_Z_true),
        # Regular trial fidelity
        "fidelity_before":     F_before,
        "fidelity_after":      F_after,
        # Verification trial
        "bell_outcome":        bell_outcome,
        "verification_pass":   verification_pass,
        "p_phi_plus":          p_phi_plus,
        # Meta
        "noise_level":         cfg.noise_level,
        "n_controllers":       cfg.n_controllers,
        "max_byzantine":       cfg.max_byzantine,
        "is_trap":             is_trap,
        "trap_detected":       trap_detected,
    }


class EVBTCSTSimulator:
    """
    Full EVBTCST protocol simulator.

    Runs n_trials trials (mix of regular + verification per delta).
    Maintains a single SPRTTracker across all trials.
    Returns a DataFrame with one row per trial plus the final SPRT state.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg  = cfg
        self.nm   = build_noise_model(cfg)
        self.sim  = AerSimulator(method="density_matrix", precision="single")
        # Derive F_noise estimate from noise_level for SPRT calibration.
        # Using the CZ-only noise model: F ≈ (1 - 4p/3)^n_CZ where n_CZ ≈ n_controllers+1.
        n_cz = cfg.n_controllers + 1
        F_noise_est = max(0.5, (1.0 - 4 * cfg.noise_level / 3) ** n_cz)
        self.sprt = SPRTTracker(
            n_controllers=cfg.n_controllers,
            F_noise=F_noise_est,
            alpha=cfg.sprt_alpha,
            beta=cfg.sprt_beta,
            min_k=cfg.sprt_threshold_k,
        )
        log.info(f"EVBTCSTSimulator: {cfg.summary()} F_noise_est={F_noise_est:.4f}")

        # Pre-compute all density matrices once — reused across every trial.
        # DM is deterministic given (circuit, noise_model); randomness is only
        # in Born-rule sampling (sample_outcomes), which is done per trial.
        log.info("Pre-computing density matrices (4 circuits)…")
        self._rho_cache: dict = {}
        qc_v = build_verification_circuit(cfg)
        self._rho_cache["verification"] = run_dm_simulation(cfg, qc_v, self.sim, self.nm)
        for label in ("psi", "zero", "plus"):
            qc = build_cluster_circuit(cfg, input_state=label)
            self._rho_cache[f"regular_{label}"] = run_dm_simulation(cfg, qc, self.sim, self.nm)
        log.info("DM cache ready.")

    def run(self) -> tuple[pd.DataFrame, dict]:
        """
        Returns (df, sprt_summary).
        df           : one row per trial
        sprt_summary : final SPRT state per controller
        """
        records   = []
        log_every = max(1, self.cfg.n_trials // 10)

        for t in range(self.cfg.n_trials):
            try:
                rec = run_trial(t, self.cfg, self.sim, self.nm,
                                sprt_tracker=self.sprt,
                                rho_cache=self._rho_cache)
                records.append(rec)
            except Exception as e:
                log.error(f"Trial {t} failed: {e}", exc_info=True)
            if t % log_every == 0:
                log.info(f"  {t}/{self.cfg.n_trials} trials done")

        df = pd.DataFrame(records)
        sprt_summary = self.sprt.state_summary()

        if len(df):
            reg = df[df.trial_type == "regular"]
            ver = df[df.trial_type == "verification"]
            log.info(
                f"Done — {len(reg)} regular, {len(ver)} verification | "
                f"F_before={reg.fidelity_before.mean():.3f} "
                f"F_after={reg.fidelity_after.mean():.3f} | "
                f"verif_pass_rate={ver.verification_pass.mean():.3f} "
                f"| SPRT_flagged={self.sprt.get_flagged_controllers()}"
            )
        return df, sprt_summary
