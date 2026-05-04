"""
quantum_circuit.py — Cluster-state circuit, DM pipeline, and Bell-pair verification.

Single-DM architecture
-----------------------
One Qiskit density-matrix simulation per trial.  All downstream steps
operate on the SAME density matrix for physical consistency.

Circuit layout (N = n_controllers + 3 qubits, N must be ODD)
--------------------------------------------------------------
  q[0]        input qubit      |ψ⟩ or half-|Φ+⟩    measured (X-basis)
  q[1]        Alice's qubit    |+⟩                  measured (X-basis)
  q[2..n+1]   C1..Cn           |+⟩ each             measured (X-basis)
  q[n+2]      Bob's qubit      |+⟩                  NOT measured → output

Correction formula (valid iff N is odd)
----------------------------------------
  M_X = XOR of outcomes at ODD  positions: 1,3,5,…
  M_Z = XOR of outcomes at EVEN positions: 0,2,4,…
  Bob applies X^{M_X} Z^{M_Z}.

Bell-pair verification (EVBTCST §3)
-------------------------------------
  On verification trials Alice prepares q[0] as one qubit of |Φ+⟩ and
  retains the second qubit privately (q[N], outside the cluster chain).
  After Bob applies his correction, Alice and Bob jointly measure in the
  Bell basis.  Any Pauli error gives outcome ≠ |Φ+⟩ with probability 1
  (noiseless) or near-1 (noisy), allowing detection of coherent-flip.
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

# ── Bell basis projectors (4x4) ───────────────────────────────────────────────
_PHI_PLUS  = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
_PHI_MINUS = np.array([1, 0, 0,-1], dtype=complex) / np.sqrt(2)
_PSI_PLUS  = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
_PSI_MINUS = np.array([0, 1,-1, 0], dtype=complex) / np.sqrt(2)

BELL_STATES = {
    "Phi+":  _PHI_PLUS,
    "Phi-":  _PHI_MINUS,
    "Psi+":  _PSI_PLUS,
    "Psi-":  _PSI_MINUS,
}

BELL_PROJECTORS = {
    name: np.outer(v, v.conj()) for name, v in BELL_STATES.items()
}


# ── Noise model ───────────────────────────────────────────────────────────────

def build_noise_model(cfg: SimConfig) -> NoiseModel | None:
    """
    Build noise model matching the paper's channel-noise assumption:
    depolarising noise only on entangling (CZ/CX) gates — local single-qubit
    operations (H, X, Z, corrections) are treated as ideal.  This gives
    F_noise ≈ 1 − (4/3)·p·n_CZ, consistent with the paper's formula.

    Readout errors are added only when cfg.readout_error_01 / _10 are non-zero
    (default 0.0 — set explicitly to model imperfect measurement).
    """
    if cfg.noise_level == 0:
        return None
    nm = NoiseModel()
    # Entangling gate noise: two-qubit depolarising at rate tq_noise_level
    tq = depolarizing_error(cfg.tq_noise_level, 2)
    for g in ("cz", "cx", "cy", "swap"):
        nm.add_all_qubit_quantum_error(tq, g)
    # Readout errors (only if configured; default 0)
    if cfg.readout_error_01 > 0 or cfg.readout_error_10 > 0:
        ro = ReadoutError([
            [1 - cfg.readout_error_01, cfg.readout_error_01],
            [cfg.readout_error_10,     1 - cfg.readout_error_10],
        ])
        nm.add_all_qubit_readout_error(ro)
    return nm


# ── Input state ───────────────────────────────────────────────────────────────

def get_input_state_vector(label: str) -> np.ndarray:
    """
    Return ideal input state vector for regular trials.

    "psi"  : cos(π/6)|0⟩ + e^{iπ/7}sin(π/6)|1⟩  — generic, all 4 Pauli
             corrections give DISTINCT fidelities (1.0, 0.25, 0.61, 0.14).
    "zero" : |0⟩
    "plus" : |+⟩
    """
    if label == "psi":
        th, ph = np.pi / 3, np.pi / 7
        return np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)], dtype=complex)
    if label == "zero":
        return np.array([1.0, 0.0], dtype=complex)
    if label == "plus":
        return np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    raise ValueError(f"Unknown input state: {label}")


# ── Regular cluster circuit ───────────────────────────────────────────────────

def build_cluster_circuit(cfg: SimConfig, input_state: str = "psi") -> QuantumCircuit:
    """
    Build the cluster-state circuit for a REGULAR trial (N qubits).

    Gate sequence:
      1. Prepare input state on q[0]
      2. H on q[1..N-1]
      3. CZ(q[i], q[i+1]) for i=0..N-2
      4. H on q[0..N-2]  (X-basis rotation for all measured qubits)
      5. save_density_matrix()
    """
    N = cfg.n_qubits
    qr = QuantumRegister(N, "q")
    qc = QuantumCircuit(qr)

    if input_state == "psi":
        qc.ry(np.pi / 3, 0)
        qc.rz(2 * np.pi / 7, 0)
    elif input_state == "plus":
        qc.h(0)
    # "zero": leave as |0⟩

    for q in range(1, N):
        qc.h(q)
    for q in range(N - 1):
        qc.cz(q, q + 1)
    for q in range(N - 1):
        qc.h(q)

    qc.save_density_matrix(label="dm")
    return qc


# ── Bell-pair verification circuit ────────────────────────────────────────────

def build_verification_circuit(cfg: SimConfig) -> QuantumCircuit:
    """
    Build the cluster-state circuit for a VERIFICATION trial (N+1 qubits).

    Qubit layout:
      q[0..N-1] : cluster chain (identical to regular circuit)
      q[N]      : Alice's retained half of |Φ+⟩ (never entangled into cluster)

    Input preparation:
      q[N] and q[0] are prepared as |Φ+⟩ = (|00⟩+|11⟩)/√2.
      Specifically: H on q[N], CNOT(q[N], q[0]) — q[N] is control (retained),
      q[0] enters the cluster chain as the teleportation input.

    Controllers only see q[0..N-1]; q[N] is invisible to them.
    Their reduced state on q[k] is I/2 regardless of this preparation
    (Lemma 1 — verified numerically in lemma1_check.py).

    Returns circuit with N+1 qubits.  save_density_matrix covers all N+1.
    """
    N = cfg.n_qubits          # cluster qubits
    total = N + 1             # +1 for Alice's retained qubit
    alice_hold = N            # index of retained qubit

    qr = QuantumRegister(total, "q")
    qc = QuantumCircuit(qr)

    # Prepare |Φ+⟩ on (alice_hold, q[0])
    qc.h(alice_hold)
    qc.cx(alice_hold, 0)

    # Cluster chain on q[0..N-1] (same as regular circuit, q[0] already set)
    for q in range(1, N):
        qc.h(q)
    for q in range(N - 1):
        qc.cz(q, q + 1)
    for q in range(N - 1):
        qc.h(q)

    qc.save_density_matrix(label="dm")
    return qc


# ── DM simulation ─────────────────────────────────────────────────────────────

def run_dm_simulation(
    cfg: SimConfig,
    qc: QuantumCircuit,
    sim: AerSimulator,
    nm: NoiseModel | None,
) -> np.ndarray:
    """Run circuit, return full density matrix as (2^n, 2^n) complex array."""
    kwargs = {"shots": 1}
    if nm is not None:
        kwargs["noise_model"] = nm
    qc_t = transpile(qc, sim, optimization_level=0)
    result = sim.run(qc_t, **kwargs).result()
    dm_obj = result.data(0)["dm"]
    return np.array(dm_obj.data, dtype=complex)


# ── Coherent outcome sampling ─────────────────────────────────────────────────

def sample_outcomes(
    rho: np.ndarray,
    n_qubits: int,
    n_measured: int,
    ro_01: float = 0.0,
    ro_10: float = 0.0,
) -> tuple[list[int], list[int], np.ndarray]:
    """
    Sample measurement outcomes for q[0..n_measured-1] from the density matrix.

    Optimised implementation — avoids copying/zeroing the full DM.
    Only the diagonal is needed for sequential Born-rule sampling.
    The post-measurement DM is reconstructed sparsely at the end.

    Returns
    -------
    circuit_outcomes  : true quantum outcomes
    reported_outcomes : circuit_outcomes + readout error
    rho_post          : DM after collapse (sparse — only consistent block non-zero)
    """
    dim     = 2 ** n_qubits
    indices = np.arange(dim, dtype=np.int64)
    diag    = np.diag(rho).real        # only 1-D diagonal needed for sampling

    circuit_outcomes  = []
    reported_outcomes = []
    consistent        = np.ones(dim, dtype=bool)   # which states survive

    for q in range(n_measured):
        bits_q   = (indices >> q) & 1              # vectorised bit extraction
        mask_0   = consistent & (bits_q == 0)
        p_total  = float(diag[consistent].sum())
        p0       = float(diag[mask_0].sum()) / max(p_total, 1e-12)
        p0       = max(0.0, min(1.0, p0))

        true_m   = 0 if random.random() < p0 else 1
        circuit_outcomes.append(true_m)

        reported_m = true_m
        if true_m == 0 and ro_01 > 0:
            reported_m = 1 if random.random() < ro_01 else 0
        elif true_m == 1 and ro_10 > 0:
            reported_m = 0 if random.random() < ro_10 else 1
        reported_outcomes.append(reported_m)

        consistent &= (bits_q == true_m)

    # Build rho_post: extract the consistent (sparse) subspace then renormalise.
    # After n_measured measurements, len(consistent_idx) = 2^(n_qubits-n_measured).
    consistent_idx = np.where(consistent)[0]
    rho_sub        = rho[np.ix_(consistent_idx, consistent_idx)]
    prob           = float(np.trace(rho_sub).real)

    rho_post = np.zeros((dim, dim), dtype=complex)
    if prob > 1e-12:
        rho_post[np.ix_(consistent_idx, consistent_idx)] = rho_sub / prob

    return circuit_outcomes, reported_outcomes, rho_post


# ── Partial trace helpers ──────────────────────────────────────────────────────

def extract_bob_dm(rho_post: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Partial trace over q[0..N-2], keeping Bob (q[N-1]).
    Qiskit LSB: qubit q = bit q of array index.

    Vectorised: uses numpy fancy indexing instead of Python loops.
    Bob is bit N-1 of each index; rest bits 0..N-2 are traced over.
    """
    N        = n_qubits
    n_rest   = 2 ** (N - 1)
    rest     = np.arange(n_rest, dtype=np.int64)

    rho_bob = np.zeros((2, 2), dtype=complex)
    for b0 in range(2):
        for b1 in range(2):
            ket = rest | np.int64(b0 << (N - 1))
            bra = rest | np.int64(b1 << (N - 1))
            rho_bob[b0, b1] = rho_post[ket, bra].sum()
    return rho_bob


def extract_single_qubit_dm(rho: np.ndarray, n_qubits: int, qubit_idx: int) -> np.ndarray:
    """
    Partial trace keeping only qubit_idx (Qiskit LSB convention).
    Used for Lemma 1 verification and for Alice's retained qubit.
    """
    dim = 2 ** n_qubits
    rho_q = np.zeros((2, 2), dtype=complex)
    for b0 in range(2):
        for b1 in range(2):
            for rest in range(2 ** (n_qubits - 1)):
                # Build full indices with qubit_idx fixed to b0 / b1
                def _idx(bit, rest_val):
                    # Insert bit at position qubit_idx into rest_val bits
                    lo = rest_val & ((1 << qubit_idx) - 1)
                    hi = rest_val >> qubit_idx
                    return lo | (bit << qubit_idx) | (hi << (qubit_idx + 1))
                i0 = _idx(b0, rest)
                i1 = _idx(b1, rest)
                rho_q[b0, b1] += rho[i0, i1]
    return rho_q


def extract_two_qubit_dm(
    rho: np.ndarray, n_qubits: int, q_a: int, q_b: int
) -> np.ndarray:
    """
    Partial trace keeping qubits q_a and q_b (Qiskit LSB).
    Returns 4x4 density matrix ordered as |00⟩,|01⟩,|10⟩,|11⟩
    where bit 0 = q_a, bit 1 = q_b.

    Vectorised: builds index arrays with numpy instead of Python loops,
    replacing 163,840 Python function calls with 16 numpy gather operations.
    """
    other_qubits = [q for q in range(n_qubits) if q not in (q_a, q_b)]
    n_other      = len(other_qubits)
    n_rest       = 2 ** n_other
    rest         = np.arange(n_rest, dtype=np.int64)

    # Precompute "rest" contribution to indices: bit_pos → qubit bit position
    rest_bits = np.zeros((n_rest, n_other), dtype=np.int64)
    for bit_pos, q in enumerate(other_qubits):
        rest_bits[:, bit_pos] = (rest >> bit_pos) & 1

    rho_2q = np.zeros((4, 4), dtype=complex)

    for s_a0, s_b0, s_a1, s_b1 in np.ndindex(2, 2, 2, 2):
        row = s_a0 | (s_b0 << 1)
        col = s_a1 | (s_b1 << 1)

        # Build ket and bra index arrays (one entry per rest combination)
        i_arr = np.full(n_rest, np.int64(s_a0 << q_a) | np.int64(s_b0 << q_b),
                        dtype=np.int64)
        j_arr = np.full(n_rest, np.int64(s_a1 << q_a) | np.int64(s_b1 << q_b),
                        dtype=np.int64)
        for bit_pos, q in enumerate(other_qubits):
            shift = rest_bits[:, bit_pos] << q
            i_arr |= shift
            j_arr |= shift

        rho_2q[row, col] = rho[i_arr, j_arr].sum()

    return rho_2q


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
    """F = ⟨ψ| (Z^{M_Z} X^{M_X} ρ_bob X^{M_X} Z^{M_Z}) |ψ⟩"""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    U = np.eye(2, dtype=complex)
    if M_X:
        U = X @ U
    if M_Z:
        U = Z @ U
    rho_c = U @ rho_bob @ U.conj().T
    return float(max(0.0, min(1.0, (ideal_psi.conj() @ rho_c @ ideal_psi).real)))


# ── Bell basis measurement (verification trials) ──────────────────────────────

def bell_basis_measurement(
    rho_2q: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """
    Project a 4x4 two-qubit density matrix onto the four Bell states.

    The 4x4 rho_2q is ordered |q_a=0,q_b=0⟩, |q_a=0,q_b=1⟩,
    |q_a=1,q_b=0⟩, |q_a=1,q_b=1⟩  (bit 0 = q_a, bit 1 = q_b).

    Returns
    -------
    outcome    : most probable Bell outcome label ('Phi+','Phi-','Psi+','Psi-')
    probs      : dict of all four probabilities (sum ≈ 1)
    """
    probs = {}
    for name, proj in BELL_PROJECTORS.items():
        p = float(np.real(np.trace(proj @ rho_2q)))
        probs[name] = max(0.0, p)

    # Normalise (small numerical errors)
    total = sum(probs.values())
    if total > 1e-10:
        probs = {k: v / total for k, v in probs.items()}

    # Sample outcome from probabilities
    names = list(probs.keys())
    ps    = [probs[n] for n in names]
    outcome = random.choices(names, weights=ps, k=1)[0]

    return outcome, probs


def apply_bob_correction_to_dm(
    rho_bob: np.ndarray,
    M_X: int,
    M_Z: int,
) -> np.ndarray:
    """Apply X^{M_X} Z^{M_Z} correction to Bob's 2x2 DM."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    U = np.eye(2, dtype=complex)
    if M_X:
        U = X @ U
    if M_Z:
        U = Z @ U
    return U @ rho_bob @ U.conj().T
