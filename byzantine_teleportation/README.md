# Byzantine-Tolerant Cluster-State Quantum Teleportation Simulator

## Architecture

```
main.py              — Entry point, experiment runner, parameter sweeps
config.py            — SimConfig dataclass (all tunable parameters)
quantum_circuit.py   — Qiskit circuit builder + noise model
byzantine.py         — Adversary model: node selection, measurement corruption, trap detection
secret_sharing.py    — Shamir secret sharing over GF(257), polynomial consistency detection
simulator.py         — Core trial logic + ByzantineTeleportationSimulator (Monte Carlo)
analysis.py          — Statistical summaries
plotting.py          — 5 publication-quality Matplotlib/Seaborn figures
requirements.txt     — Python dependencies
```

## Qubit Layout

```
q[0]       input qubit  (carries |ψ⟩ = (|0⟩ + i|1⟩)/√2)
q[1]       Alice's cluster qubit
q[2…n+1]  C1 … Cn  (n controllers)
q[n+2]    Bob's cluster qubit
```

## Running

```bash
pip install -r requirements.txt
python main.py
```

Results are written to `results/`:
- `default_results.csv`   — 5000-trial default experiment
- `sweep_byzantine.csv`   — f sweep
- `sweep_network_size.csv`— n sweep
- `sweep_noise.csv`       — noise sweep
- `fig1_fidelity_vs_byzantine.png`
- `fig2_fidelity_improvement.png`
- `fig3_detection_rate.png`
- `fig4_success_vs_network_size.png`
- `fig5_fidelity_vs_noise.png`

## Design Decisions & Corrections Applied

### 1. Input state attachment
The spec says "attach via CZ between input qubit and Alice's cluster qubit".
`quantum_circuit.py` implements this exactly: `qc.cz(0, 1)` after preparing
`|ψ⟩` on q[0] and `|+⟩` on all cluster qubits.

### 2. Measurement outcomes from the simulator
Outcomes are extracted from `result.get_counts()` with `shots=1`.
Byzantine corruption is applied *after* the quantum simulation so that the
quantum outcomes are genuine and the classical Byzantine layer is separate.

### 3. Correction rules
Per the spec:
```
M_X = m1 ⊕ m3 ⊕ m5 ⊕ m7   (odd-indexed controllers)
M_Z = m_A ⊕ m2 ⊕ m4 ⊕ m6  (even-indexed controllers + Alice)
```
`compute_correction_bits()` in `simulator.py` implements this generically
for any n.

### 4. Bob's fidelity via partial trace
`partial_trace(dm_full, trace_out)` from `qiskit.quantum_info` extracts
ρ_B. Pauli corrections X^M_X Z^M_Z are applied as unitary conjugation.
`state_fidelity(ρ_B_corrected, |ψ⟩)` computes the final fidelity.

### 5. Secret sharing — GF(257) Lagrange interpolation
All arithmetic is done mod 257.  `modinv` uses the extended Euclidean algorithm.
`detect_inconsistent_shares` interpolates t points from controller i's shares
and verifies the remaining shares lie on the same polynomial.

### 6. Recovery
Per the spec: `m_b = 1 − f_b(0)` where f_b(0) is the reconstructed constant
of Byzantine node b.  This is implemented in `secret_sharing.recover_measurement`.

### 7. Noise model
`build_noise_model` applies:
- `depolarizing_error(p, 1)` to all single-qubit gates
- `depolarizing_error(2p, 2)` to CZ/CX/SWAP
- `ReadoutError([[0.97,0.03],[0.04,0.96]])` when p > 0
Uses `AerSimulator(method="density_matrix")` for exact mixed-state simulation.
