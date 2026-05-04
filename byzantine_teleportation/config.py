"""
config.py — SimConfig with hard constraints enforced at construction.

Constraint 1 — n_controllers MUST be EVEN.
  N = n_controllers + 3 must be ODD for Pauli byproduct corrections to
  have a closed-form XOR expression.  For even N the byproduct is a
  non-Pauli SU(2) rotation that cannot be determined from binary outcomes.

Constraint 2 — n_controllers > 3 * max_byzantine.
  The Fischer-Lynch-Paterson bound for Byzantine VSS with Shamir threshold
  t = f+1.  Below this, the protocol has no correctness guarantee.
  Both constraints together: smallest valid (n,f) pairs are
    (4,1), (8,2), (12,3), (16,4), ...
"""

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # ── Network ────────────────────────────────────────────────────────────
    n_controllers: int = 8
    max_byzantine: int = 2

    # ── Noise ──────────────────────────────────────────────────────────────
    noise_level: float = 0.01        # single-qubit depolarising p
    tq_noise_factor: float = 2.0     # two-qubit noise = factor * p
    readout_error_01: float = 0.03   # P(report 1 | true 0)
    readout_error_10: float = 0.04   # P(report 0 | true 1)

    # ── Experiment ─────────────────────────────────────────────────────────
    n_trials: int = 500
    trap_probability: float = 0.05

    # ── Crypto ─────────────────────────────────────────────────────────────
    gf_prime: int = 257              # Shamir field: GF(257)

    # ── Backend ────────────────────────────────────────────────────────────
    shots: int = 1
    verbose: bool = False

    # ── Derived ────────────────────────────────────────────────────────────
    threshold: int = field(init=False)
    n_qubits:  int = field(init=False)
    tq_noise_level: float = field(init=False)

    def __post_init__(self):
        if self.n_controllers % 2 != 0:
            raise ValueError(
                f"n_controllers must be EVEN (got {self.n_controllers}). "
                f"N = n_controllers+3 = {self.n_controllers+3} must be ODD "
                f"for Pauli byproduct corrections to exist."
            )
        if self.n_controllers <= 3 * self.max_byzantine:
            raise ValueError(
                f"Byzantine bound violated: need n > 3f, "
                f"got n={self.n_controllers}, f={self.max_byzantine}. "
                f"Min valid n for f={self.max_byzantine} is "
                f"{3*self.max_byzantine+2} (first even integer > {3*self.max_byzantine})."
            )
        self.threshold      = self.max_byzantine + 1
        self.n_qubits       = self.n_controllers + 3   # input+alice+ctrl*n+bob
        self.tq_noise_level = self.tq_noise_factor * self.noise_level
