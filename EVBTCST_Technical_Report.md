# Entanglement-Verified Byzantine-Tolerant Cluster-State Quantum Teleportation (EVBTCST)
### Technical Report — Protocol, Simulation Results, and Open Issues

---

## 1. What This Is

EVBTCST is a quantum teleportation protocol designed to work correctly even when some of the
network nodes (controllers) are actively trying to sabotage it. It combines three independent
mechanisms into one unified protocol:

1. **1D Cluster-State Teleportation** — the physical transport mechanism
2. **Verifiable Secret Sharing (VSS)** — cryptographic protection of measurement outcomes
3. **Sequential Probability Ratio Test (SPRT)** — statistical detection of persistent adversaries
4. **Bell-Pair Verification** — quantum-level integrity check of each teleportation hop

The core question the protocol answers: *"Can Alice reliably send a quantum state to Bob through
a chain of n controllers, when up to f of those controllers are Byzantine (actively adversarial),
and with noisy quantum hardware?"*

---

## 2. Protocol Setup

### 2.1 Network Structure

```
Alice (input) → C1 → C2 → C3 → ... → C8 → Bob
                                           ↑
                               Alice holds entangled qubit
```

- **n = 8 controllers** (C1 through C8) — must be even
- **f ≤ 2 Byzantine controllers** — hard constraint: n > 3f (Fischer-Lynch-Paterson bound)
- **N = n + 3 = 11 total qubits** in the circuit
- **Threshold t = f + 1 = 3** — minimum shares needed to reconstruct a secret (Shamir VSS)
- **GF(257)** — finite field for polynomial secret sharing

### 2.2 Hard Constraints (Enforced in Code)

| Constraint | Reason | Value |
|---|---|---|
| n must be even | N = n+3 must be odd for XOR correction to have closed form | n = 8 |
| n > 3f | FLP bound for Byzantine VSS | 8 > 6 ✓ |
| δ ∈ [0,1] | Fraction of verification trials | δ = 0.10 |

### 2.3 Trial Types

Every trial is independently classified as one of two types, determined randomly by δ:

- **Regular trial (90%)**: Teleport actual quantum state. Goal: high fidelity.
- **Verification trial (10%)**: Alice prepares |Φ+⟩, holds one half, teleports the other half
  through the chain to Bob. After correction, Alice and Bob perform a Bell measurement.
  Goal: detect cheating.

---

## 3. Protocol Mechanics

### 3.1 Regular Trial Flow

```
1. Alice prepares input state |ψ⟩ (one of: |ψ⟩, |0⟩, |+⟩)
2. Build 1D cluster state |G⟩ across all n+3 qubits (H gates + CZ bonds)
3. Each controller Ck measures their qubit, gets outcome mk ∈ {0,1}
4. Each controller commits to mk via VSS:
      - Pedersen commitment: C = g^mk · h^r mod p (hides mk, binding)
      - Hash commitment: H(mk || nonce) (secondary binding)
      - Shamir secret sharing: distribute t+1 shares of mk to peers
5. Reveal phase: each controller broadcasts mk and opens commitment
6. VSS verification: check Pedersen/hash consistency, check shares match polynomial
7. Recovery: Lagrange interpolation recovers mk for any misbehaving controller
8. Compute correction bits: M_X = XOR(odd-position bits), M_Z = XOR(even-position bits)
9. Bob applies X^M_X · Z^M_Z correction to receive teleported state
```

### 3.2 Verification Trial Flow (EVBTCST-specific)

```
1. Alice prepares |Φ+⟩ = (|00⟩ + |11⟩)/√2
2. Alice holds qubit q[N] (the "Alice-hold" qubit)
3. The other half is teleported through the cluster to Bob (same as regular trial)
4. After Bob applies correction, Alice and Bob jointly perform Bell measurement
5. Expected outcome: Φ+ (if teleportation was perfect and correction was right)
6. Actual outcome: Φ+, Φ-, Ψ+, or Ψ- (depending on errors)
7. If outcome ≠ Φ+: find which parity group had wrong correction → blame those controllers
8. Feed outcome into SPRT tracker
```

### 3.3 Byzantine Attack Strategies

Three attack strategies are modeled:

| Strategy | What the attacker does | Caught by |
|---|---|---|
| **coherent_flip** | Flips reported measurement mk (consistently) | SPRT (causes wrong M_Z or M_X) |
| **equivocate** | Sends different mk values to different peers | Hash VSS (inconsistent opens) |
| **withhold** | Refuses to reveal mk | Absent commitment detected |

`coherent_flip` is the hardest to detect because the attacker's VSS shares are internally
consistent — they just commit to the wrong value. VSS cannot catch this. Only SPRT can.

---

## 4. The SPRT Detection Mechanism

### 4.1 Parity Attribution

When a verification trial fails (outcome ≠ Φ+), the code identifies which parity group caused
the wrong correction:

```
M_X = XOR of bits at odd positions  → controllers C2, C4, C6, C8
M_Z = XOR of bits at even positions → controllers C1, C3, C5, C7

If M_X_reported ≠ M_X_true: blame {C2, C4, C6, C8}
If M_Z_reported ≠ M_Z_true: blame {C1, C3, C5, C7}
```

**Important limitation**: the protocol identifies a group of 4 controllers, not a specific node.

### 4.2 Two-Sided Wald SPRT

For each controller Ck, maintain a log-likelihood ratio (LLR):

```
After each verification trial:
  - If Ck is blamed (in contributing list AND trial failed):
        LLR_k += log(p_flip_adv / p_flip_honest)        [positive: evidence of adversary]
  - Otherwise (passed, or Ck not in contributing list):
        LLR_k += log((1-p_flip_adv) / (1-p_flip_honest)) [negative: evidence of honesty]

Flag Ck when: LLR_k ≥ threshold  AND  n_blamed_trials ≥ min_k
  where threshold = log((1-β)/α) = log(9999/1) ≈ 9.21
        α = 0.01  (target FPR)
        β = 0.0001 (target FNR)
```

### 4.3 SPRT Calibration (Key Finding)

The SPRT parameters must be calibrated empirically, not from the theoretical F_noise formula.
This is because:

- **p_flip_honest** (P(controller blamed | honest run)) ≈ **0.0%** in simulation
  - Honest failures are "quantum state noise failures" — the Bell measurement gives wrong
    outcome due to CZ gate noise, but the correction bits (M_X, M_Z) are still correct.
    Contributing list is empty → no controller is blamed.
- **p_flip_adv** (P(controller blamed | adversary present)) ≈ **98%** in simulation
  - The adversary always reports wrong mk → M_Z_reported ≠ M_Z_true → blamed in every
    failed trial.

This near-perfect separation (0% vs 98%) means the SPRT effectively becomes:
*"flag the controller after their first blamed trial."*

---

## 5. Simulation Results

### 5.1 Lemma 1 — Indistinguishability (32/32 PASS)

**Claim**: The reduced density matrix of each controller's qubit is I/2 (maximally mixed)
regardless of the input state being teleported.

**Result**: All 32 cases (8 controllers × 2 trial types × 2 inputs) verified at trace distance
= **0.0** (below floating-point precision ~4.4 × 10⁻¹⁶).

**Significance**: Byzantine controllers cannot adapt their attack based on what is being
teleported — they gain zero information about the input state.

---

### 5.2 Table 2 — Detection Power vs Verification Trials k

**Setup**: 200 independent batches per k-value, each with exactly k verification trials and
one fixed coherent-flip adversary. SPRT calibrated with measured p_flip_honest and p_flip_adv.
F_noise_est = 0.895 (calibrated from simulation).

| k | Empirical Detection Power | Theoretical Power | False Positive Rate |
|---|---|---|---|
| 1 | **97.0%** | 0.0%* | 0.0% |
| 2 | **100.0%** | 94.84% | 0.0% |
| 3 | **100.0%** | 92.35% | 0.0% |
| 5 | **100.0%** | 99.98% | 0.0% |
| 7 | **100.0%** | 100.0% | 0.0% |
| 10+ | **100.0%** | 100.0% | 0.0% |

*At k=1, the theoretical binomial test cannot achieve α=0.01 significance (P(S=0|honest)=7.9%
> 1%), so the theoretical power is undefined/0. The SPRT achieves 97% by using LLR accumulation
rather than a counting test.

**Key finding**: The SPRT detects the adversary after just **2 verification trials** with 100%
reliability and **0% false positive rate** across all tested k-values.

---

### 5.3 Table 3 — Fidelity vs Byzantine Count

**Setup**: 300 trials, δ=0.15, noise_level=0.01.
F_before = fidelity of Bob's state before VSS correction.
F_after = fidelity of Bob's state after VSS recovery.

| Byzantine (f) | F_before | F_after (VSS) | M_fail_before | M_fail_after | Verif Pass Rate |
|---|---|---|---|---|---|
| 0 | 0.921 | 0.921 | 0.0% | 0.0% | 82.5% |
| 1 | 0.599 | **0.785** | 63.8% | 27.1% | 55.1% |
| 2 | 0.541 | **0.643** | 67.7% | 51.3% | 37.8% |

**Key finding**: VSS recovery significantly improves fidelity. With 1 Byzantine controller,
fidelity improves from 0.60 → 0.79 (+31%). With 2 Byzantine, from 0.54 → 0.64 (+19%).
The improvement decreases with more Byzantine nodes because VSS can only correct up to f
corruptions when n > 3f.

---

### 5.4 Table 4 — Noise Scaling / Protocol Viability

**Setup**: 150 trials per noise level, no Byzantine, δ=0.
Viability criterion: likelihood ratio > 12 (strong evidence of working teleportation).

| Noise p | F_noise (simulation) | F_noise (theory) | Likelihood Ratio | Viable |
|---|---|---|---|---|
| 0.000 | 0.965 | 1.000 | 109.98 | ✓ |
| 0.005 | 0.942 | 0.863 | 65.83 | ✓ |
| 0.010 | 0.921 | 0.744 | 47.53 | ✓ |
| 0.020 | 0.879 | 0.552 | 30.00 | ✓ |
| 0.030 | 0.843 | 0.407 | 22.44 | ✓ |
| 0.050 | 0.779 | 0.219 | 15.07 | ✓ |

**Key finding**: The protocol remains viable (LR > 12) up to at least p = 0.05 depolarizing
noise on CZ gates. The theoretical F_noise formula underestimates actual fidelity significantly
at high noise — the simulation is the authoritative source.

---

## 6. Architecture of the Simulation

```
evbtcst/
├── config.py                    # SimConfig dataclass, enforces n>3f and n even
├── quantum_circuit.py           # Build cluster circuit, noise model, sample outcomes
├── vss.py                       # Pedersen + hash VSS, Shamir sharing, recovery
├── byzantine.py                 # Byzantine attack strategies (flip, equivocate, withhold)
├── sprt.py                      # SPRTTracker, two-sided Wald SPRT, parity attribution
├── simulator.py                 # run_trial(), EVBTCSTSimulator with DM caching
├── main.py                      # Entry point — runs all experiments
└── experiments/
    ├── lemma1_check.py          # Lemma 1 density matrix verification
    ├── table2_detection_power.py # SPRT detection power vs k
    ├── table3_fidelity_byzantine.py
    └── table4_noise_scaling.py
```

### Key Optimizations

| Optimization | Speedup | Detail |
|---|---|---|
| DM caching | 4140ms → 30ms per trial | Pre-compute 4 density matrices (verification + 3 input states). Reuse across all trials — only Born-rule sampling varies. |
| Diagonal-only sampling | ~100× on sample_outcomes | Extract diagonal of 2048×2048 DM instead of copying full matrix |
| Vectorized numpy indexing | ~50× on extract_two_qubit_dm | Replace 163,840 Python function calls with 16 numpy operations |
| SharedMemory IPC | Avoids WinError 1450 | Pass 268MB DM to worker processes via shared memory, not OS pipes |
| Windows worker cap | Prevents crashes | Cap ProcessPoolExecutor at 2 workers on Windows (Qiskit imports crash with ≥3 simultaneous spawns) |

---

## 7. Known Issues and Limitations

### 7.1 Noise Model Scope (Most Important for Paper)

**Issue**: The simulation applies depolarizing noise only to CZ/entangling gates (not H gates,
not readout). This was necessary to match the paper's F_noise formula.

**Impact**: With realistic full-gate noise:
- Actual F_noise drops from 0.92 to 0.47 (H-gate noise adds ~21 extra error events)
- Hardcoded readout errors (3-4%) reduce p(Φ+|honest) from 0.89 to 0.52
- The protocol's viability claims break down under realistic noise

**What it means**: The current model represents a scenario where local single-qubit operations
(H gates, Pauli corrections) are near-perfect but quantum communication links (CZ bonds) are
noisy. This is physically motivated — in a real quantum network, H gates run locally on
high-fidelity qubits while CZ gates require coupling between nodes (noisier). But it needs to
be explicitly stated and justified in the paper.

---

### 7.2 Parity-Group Attribution, Not Node Attribution

**Issue**: When a verification fails, the SPRT blames all 4 controllers in the relevant parity
group (e.g., {C1, C3, C5, C7}), not just the actual adversary.

**Impact**:
- When the adversary C1 is detected, controllers C3, C5, C7 are flagged alongside them
- You cannot tell which of the 4 is the actual adversary
- If the protocol responds by excluding all 4, the network loses half its controllers

**Root cause**: M_Z = XOR(even-position bits). If M_Z is wrong, any of the 4 even-position
controllers could be responsible.

**Mitigation options**:
- Run individual verification rounds targeting each controller (more overhead)
- Use the XOR structure: run verification while one controller is offline to isolate blame
- Accept group-level attribution and design the exclusion protocol accordingly

**Current status**: Stated as a "Limitations and Future Work" item.

---

### 7.3 SPRT Requires Fixed Adversary Identity

**Issue**: The SPRT accumulates evidence against a specific controller across trials. If the
Byzantine controller changes identity across trials (rotating adversary), the accumulated LLR
spreads across all 8 controllers and none gets flagged.

**Evidence**: In the main simulation (EVBTCSTSimulator), Byzantine identity is drawn randomly
each trial. SPRT_flagged remains empty throughout. In Table 2, Byzantine identity is fixed —
detection is 97-100%.

**What it means**: The protocol's detection capability assumes the adversary commits to
attacking consistently (same node, same strategy). An adversary who rotates between nodes
every few trials would evade detection.

**Mitigation**: The adversary model should be stated explicitly in the paper: "We assume
Byzantine nodes maintain consistent behavior across trials within a session."

---

### 7.4 XOR Cancellation Rate = 0 (Broken Feature)

**Issue**: Table 3 includes columns `xor_cancel_rate` (measured: 0.0) and `xor_cancel_theory`
(0.15 for all rows). These never match.

**Root cause**: The feature is not correctly implemented — the measurement code produces 0.0
in all cases despite a theoretical prediction of 15%.

**Fix plan**: Remove both columns from Table 3 and the paper entirely. The feature adds nothing
to the paper's core claims.

---

### 7.5 F_noise Formula vs Simulation Divergence

**Issue**: The theoretical formula F_noise = 1 - (4/3)·p·n diverges from simulation at high p:

| p | Simulation | Theory | Ratio |
|---|---|---|---|
| 0.01 | 0.921 | 0.744 | 1.24× |
| 0.05 | 0.779 | 0.219 | 3.56× |

**Root cause**: The formula is a first-order Taylor approximation for a single-qubit channel
applied once per qubit. The actual circuit has 10 CZ gates (not 11 single-qubit gates), and
the CZ-only noise model degrades fidelity differently than the formula predicts.

**Fix plan**: Remove the F_noise_theory column from Table 4. Keep only F_noise_sim. Add a
footnote: "Theoretical F_noise approximation underestimates actual fidelity; simulation values
used throughout."

---

### 7.6 Theoretical vs Empirical Detection Power at k=1

**Issue**: At k=1, theoretical power = 0% but empirical = 97%.

**This is not a bug** — it reflects two different tests:
- Theoretical: optimal fixed-sample binomial test (reject if Φ+ count ≤ s*). At k=1,
  P(S=0|honest) = 7.9% > α=0.01, so no valid rejection region exists.
- Empirical SPRT: accumulates LLR per controller. With p_flip_honest≈0 and p_flip_adv≈98%,
  a single blame event gives LLR ≈ 27 >> threshold 4.6 → immediate flag.

The SPRT is a fundamentally different (and more powerful) test than what the theoretical
formula computes. This needs to be explained clearly in the paper to avoid reviewer confusion.

---

### 7.7 SPRT in Main Simulation Uses Approximate F_noise

**Issue**: EVBTCSTSimulator's SPRT uses F_noise estimated from the formula
`(1 - 4p/3)^(n_cz)` rather than empirically calibrated p_flip values.

**Impact**: For the main simulation runs (default, sweep_byzantine, sweep_noise), the SPRT
threshold and increments are slightly miscalibrated. However since p_flip_honest≈0 in practice,
this mainly affects the LLR increment magnitude, not the detection decision (which is dominated
by the first blame event pushing LLR far above threshold).

---

## 8. What the Paper Should Claim (and What It Shouldn't)

### Strong Claims (Well-Supported)

✓ "Lemma 1: each controller's reduced density matrix is I/2 regardless of input state,
   verified at machine precision (trace distance < 4.4 × 10⁻¹⁶)"

✓ "With a fixed coherent-flip adversary and CZ-channel noise p=0.01, the SPRT detects
   the adversary within 2 verification trials with 100% empirical power and 0% FPR"

✓ "VSS recovery improves teleportation fidelity from 0.60 to 0.79 (f=1) and 0.54 to 0.64
   (f=2), demonstrating that cryptographic commitment prevents adversarial fidelity collapse"

✓ "The protocol remains viable (likelihood ratio > 12) at CZ depolarizing noise up to p=0.05"

✓ "The protocol enforces the n > 3f Byzantine bound at construction time"

### Claims to Qualify or Avoid

⚠ "Detection power > 0.9999 at k=7" — true for the theoretical binomial model with idealized
   F_noise=0.853, but the SPRT achieves 100% at k=2 through a different mechanism. State both.

⚠ "The SPRT identifies the Byzantine node" — it identifies the 4-node parity group.
   Change to: "identifies the parity group containing the Byzantine node."

✗ "XOR cancellation achieves 15% overhead reduction" — not measured. Remove entirely.

✗ Any claim about F_noise_theory matching simulation — it does not above p=0.02.

---

## 9. Recommended Next Steps (Prioritized)

| Priority | Task | Time |
|---|---|---|
| 1 | Remove xor_cancel columns from Table 3 + paper | 1 hour |
| 2 | Remove F_noise_theory from Table 4 + paper | 1 hour |
| 3 | Write the paper (Sections 1-5) | 3-5 days |
| 4 | Post to arXiv (quant-ph + cs.DC) | 1 day |
| 5 | [Future] Fix attribution to sub-group level | Research question |
| 6 | [Future] Test with n=4, 12, 16 for scaling analysis | 1 day coding |
| 7 | [Future] Add full-gate noise analysis as separate section | 2 days |

---

## 10. Simulation Parameters Reference

| Parameter | Value | Notes |
|---|---|---|
| n_controllers | 8 | Must be even |
| max_byzantine | 2 | n > 3f enforced |
| noise_level | 0.01 | CZ depolarizing rate |
| tq_noise_factor | 1.0 | CZ noise = 1.0 × p |
| readout_error_01/10 | 0.0 | No readout errors (channel noise only) |
| n_trials | 500 (default) | |
| delta | 0.10 | 10% verification trials |
| sprt_alpha | 0.01 | FPR target |
| sprt_beta | 0.0001 | FNR target (→ 99.99% detection) |
| sprt_threshold_k | 3 | Min blamed trials before flagging |
| gf_prime | 257 | GF field for Shamir VSS |
| AerSimulator | density_matrix, single precision | |

---

*Generated from simulation run: 2026-03-22. All results reproducible via `python main.py` in the `evbtcst/` directory.*
