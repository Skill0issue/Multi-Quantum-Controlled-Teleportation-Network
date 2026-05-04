"""
quantum_circuit.py — Cluster-state circuit and single-DM fidelity pipeline.

Single-DM architecture
-----------------------
Run ONE Qiskit density-matrix simulation per trial.  All downstream steps
(outcome sampling, post-selection, fidelity) operate on the SAME density
matrix, ensuring full physical consistency.

The old two-simulator approach (separate ideal-SV + noisy-shots jobs) was
incoherent: post-selecting a clean statevector on outcomes from an independent
noisy run is physically meaningless and always gives F ≈ 0.5.

Circuit layout (N = n_controllers + 3 qubits, N must be ODD)
--------------------------------------------------------------
  q[0]        input qubit      |ψ⟩             measured (X-basis)
  q[1]        Alice's qubit    |+⟩             measured (X-basis)
  q[2..n+1]   C1..Cn           |+⟩ each        measured (X-basis)
  q[n+2]      Bob's qubit      |+⟩             NOT measured → output

Correction formula (valid iff N is odd)
----------------------------------------
  all_outcomes = [m_input, m_alice, m_C1, ..., m_Cn]   length = N-1 = n+2
  M_X = XOR of outcomes at ODD  positions: 1,3,5,…  (alice, C2, C4, …)
  M_Z = XOR of outcomes at EVEN positions: 0,2,4,…  (input, C1, C3, …)
  Bob applies X^{M_X} Z^{M_Z} to recover |ψ⟩.
"""

from __future__ import annotations
import random
import logging

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from config import SimConfig

log = logging.getLogger("quantum_circuit")


# ── Noise model ───────────────────────────────────────────────────────────────

def build_noise_model(cfg: SimConfig) -> NoiseModel | None:
    if cfg.noise_level == 0:
        return None
    nm = NoiseModel()
    sq = depolarizing_error(cfg.noise_level, 1)
    for g in ("h", "x", "z", "s", "t", "rx", "ry", "rz", "u", "u1", "u2", "u3"):
        nm.add_all_qubit_quantum_error(sq, g)
    tq = depolarizing_error(cfg.tq_noise_level, 2)
    for g in ("cz", "cx", "cy", "swap"):
        nm.add_all_qubit_quantum_error(tq, g)
    ro = ReadoutError([
        [1 - cfg.readout_error_01, cfg.readout_error_01],
        [cfg.readout_error_10,     1 - cfg.readout_error_10],
    ])
    nm.add_all_qubit_readout_error(ro)
    return nm


# ── Input state ───────────────────────────────────────────────────────────────

def get_input_state_vector(label: str) -> np.ndarray:
    """
    Return the ideal input state vector.

    State choices
    -------------
    "psi"  : (cos(π/6)|0⟩ + e^{iπ/7} sin(π/6)|1⟩) — generic state, not an
             eigenstate of X, Y, Z, or any XZ product.  All four Pauli corrections
             give DISTINCT fidelities (1.0, 0.25, 0.61, 0.14), making every
             measurement outcome unambiguously identifiable.

             Previous "psi" was (|0⟩+i|1⟩)/√2 — a Y-eigenstate with eigenvalue +1.
             Because XZ·|+Y⟩ = iY|+Y⟩ = i|+Y⟩, both M=(0,1) and M=(1,0) gave
             F=1.0, making half the possible correction errors undetectable in testing.

    "zero" : |0⟩  — Z-eigenstate, useful for trap teleportations.
    "plus" : |+⟩  — X-eigenstate, useful for trap teleportations.
    """
    if label == "psi":
        th, ph = np.pi / 3, np.pi / 7
        return np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)], dtype=complex)
    if label == "zero":
        return np.array([1.0, 0.0], dtype=complex)
    if label == "plus":
        return np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    raise ValueError(f"Unknown input state: {label}")


# ── Circuit builder ───────────────────────────────────────────────────────────

def build_cluster_circuit(cfg: SimConfig, input_state: str = "psi") -> QuantumCircuit:
    """
    Build the cluster-state circuit WITHOUT measurement gates.
    Appends save_density_matrix() so Aer stores result.data(0)["dm"].

    Gate sequence:
      1. Prepare input state on q[0]
      2. H on q[1..N-1]   (cluster |+⟩ preparation)
      3. CZ(q[i], q[i+1]) for i = 0..N-2   (entanglement chain)
      4. H on q[0..N-2]   (X-basis rotation for all measured qubits incl. input)
    """
    N = cfg.n_qubits
    qr = QuantumRegister(N, "q")
    qc = QuantumCircuit(qr)

    # Input preparation — matches get_input_state_vector()
    if input_state == "psi":
        # |ψ⟩ = cos(π/6)|0⟩ + e^{iπ/7}sin(π/6)|1⟩
        # Rz(2π/7) Ry(π/3) |0⟩  (Qiskit Ry/Rz decomposition)
        qc.ry(np.pi / 3, 0)
        qc.rz(2 * np.pi / 7, 0)
    elif input_state == "plus":
        qc.h(0)    # |+⟩
    # "zero": leave as |0⟩

    # Cluster: |+⟩ on q[1..N-1]
    for q in range(1, N):
        qc.h(q)

    # CZ entanglement chain
    for q in range(N - 1):
        qc.cz(q, q + 1)

    # X-basis rotation on ALL measured qubits q[0..N-2]
    for q in range(N - 1):
        qc.h(q)

    # Save full density matrix (BEFORE any measurement)
    qc.save_density_matrix(label="dm")
    return qc


# ── DM simulation ─────────────────────────────────────────────────────────────

def run_dm_simulation(
    cfg: SimConfig,
    qc: QuantumCircuit,
    sim: AerSimulator,
    nm: NoiseModel | None,
) -> np.ndarray:
    """
    Run the circuit once and return the full N-qubit density matrix
    as a (2^N, 2^N) complex numpy array.
    """
    kwargs = {"shots": 1}
    if nm is not None:
        kwargs["noise_model"] = nm
    qc_t = transpile(qc, sim, optimization_level=0)
    result = sim.run(qc_t, **kwargs).result()
    dm_obj = result.data(0)["dm"]
    # DensityMatrix → numpy array
    return np.array(dm_obj.data, dtype=complex)


# ── Coherent outcome sampling from DM ────────────────────────────────────────

def sample_outcomes(
    rho: np.ndarray,
    n_qubits: int,
    n_measured: int,
    ro_01: float = 0.0,
    ro_10: float = 0.0,
) -> tuple[list[int], list[int], np.ndarray]:
    """
    Sample measurement outcomes for q[0..n_measured-1] from the density matrix.

    Three-way separation of physical layers:

      circuit_outcomes : TRUE quantum measurement results (Born-rule collapse).
                         These are what the quantum circuit actually produced.
                         Used for: M_true computation, DM collapse → Bob's state.

      reported_outcomes: circuit_outcomes after readout error (what the controller
                         reads off their classical display). These are the base
                         measurements before any Byzantine corruption.
                         Used for: input to corrupt_direct_reports() and VSS.

      rho_post         : DM after collapsing on circuit_outcomes. Bob's conditional
                         state is derived from this.

    Qiskit LSB convention: qubit q occupies bit position q of the state index,
    so (index >> q) & 1 extracts qubit q's value.

    Returns
    -------
    circuit_outcomes  : list[int], length n_measured
    reported_outcomes : list[int], length n_measured (includes readout errors)
    rho_post          : np.ndarray (2^N, 2^N)
    """
    N   = n_qubits
    dim = 2 ** N
    rho_cond = rho.copy()
    circuit_outcomes  = []
    reported_outcomes = []

    for q in range(n_measured):
        # Born-rule: P(qubit q = 0) from DM diagonal (Qiskit LSB convention)
        p0 = float(sum(
            rho_cond[b, b].real
            for b in range(dim)
            if ((b >> q) & 1) == 0
        ))
        p0 = max(0.0, min(1.0, p0))
        true_m = 0 if random.random() < p0 else 1
        circuit_outcomes.append(true_m)

        # Readout error: what the controller observes on their classical display
        if true_m == 0:
            reported_m = 1 if random.random() < ro_01 else 0
        else:
            reported_m = 0 if random.random() < ro_10 else 1
        reported_outcomes.append(reported_m)

        # Collapse DM on the TRUE circuit outcome (readout error is purely classical)
        mask = np.array([((b >> q) & 1) == true_m for b in range(dim)])
        rho_cond[~mask, :] = 0
        rho_cond[:, ~mask] = 0
        prob = float(rho_cond.trace().real)
        if prob > 1e-12:
            rho_cond /= prob

    return circuit_outcomes, reported_outcomes, rho_cond


# ── Extract Bob's conditional DM ──────────────────────────────────────────────

def extract_bob_dm(rho_post: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Partial trace over q[0..N-2], keeping Bob (q[N-1]).

    Qiskit LSB convention: qubit q = bit q of array index.
    In rho.reshape([2]*N + [2]*N):
      axis 0   corresponds to qubit N-1 (MSB) — this is BOB
      axis N-1 corresponds to qubit 0   (LSB) — this is input

    We sum over axes 1..N-1 (qubits 0..N-2) and keep axis 0 (Bob).
    """
    N  = n_qubits
    rt = rho_post.reshape([2] * N + [2] * N)
    rho_bob = np.zeros((2, 2), dtype=complex)
    for idx in range(2 ** (N - 1)):
        bits = [(idx >> (N - 2 - k)) & 1 for k in range(N - 1)]
        for b0 in range(2):
            for b1 in range(2):
                # Bob's index is the FIRST axis; other qubits follow
                rho_bob[b0, b1] += rt[tuple([b0] + bits) + tuple([b1] + bits)]
    return rho_bob


# ── Correction bits ───────────────────────────────────────────────────────────

def compute_correction_bits(all_outcomes: list[int]) -> tuple[int, int]:
    """
    all_outcomes = [m_input, m_alice, m_C1, …, m_Cn]

    M_X = XOR of positions 1,3,5,…  (odd:  alice, C2, C4, …)
    M_Z = XOR of positions 0,2,4,…  (even: input, C1, C3, …)
    """
    M_X = M_Z = 0
    for i, m in enumerate(all_outcomes):
        if i % 2 == 1:
            M_X ^= m
        else:
            M_Z ^= m
    return M_X, M_Z


# ── Fidelity ─────────────────────────────────────────────────────────────────

def compute_fidelity(
    rho_bob: np.ndarray,
    M_X: int,
    M_Z: int,
    ideal_psi: np.ndarray,
) -> float:
    """
    F = ⟨ψ| (Z^{M_Z} X^{M_X} ρ_bob X^{M_X} Z^{M_Z}) |ψ⟩
    Applies X first, then Z.
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    U = np.eye(2, dtype=complex)
    if M_X:
        U = X @ U
    if M_Z:
        U = Z @ U
    rho_c = U @ rho_bob @ U.conj().T
    return float(max(0.0, min(1.0, (ideal_psi.conj() @ rho_c @ ideal_psi).real)))
