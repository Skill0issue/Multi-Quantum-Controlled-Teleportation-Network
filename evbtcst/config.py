"""
config.py — SimConfig with hard constraints enforced at construction.

Constraint 1 — n_controllers MUST be EVEN.
  N = n_controllers + 3 must be ODD for Pauli byproduct corrections to
  have a closed-form XOR expression.

Constraint 2 — n_controllers > 3 * max_byzantine.
  Fischer-Lynch-Paterson bound for Byzantine VSS with Shamir threshold t = f+1.

New in EVBTCST:
  delta          — fraction of trials that are Bell-pair verification trials.
  sprt_alpha     — SPRT false-positive rate (Type I error).
  sprt_beta      — SPRT false-negative rate (Type II error) target.
  sprt_threshold_k — minimum verification trials before SPRT can flag.
"""

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # ── Network ────────────────────────────────────────────────────────────
    n_controllers: int = 8
    max_byzantine: int = 2

    # ── Noise ──────────────────────────────────────────────────────────────
    noise_level: float = 0.01        # depolarising noise rate p (on CZ/entangling gates)
    tq_noise_factor: float = 1.0     # two-qubit noise = factor * p (1.0 = same rate as p)
    readout_error_01: float = 0.0    # P(report 1 | true 0) — set proportional via noise_level
    readout_error_10: float = 0.0    # P(report 0 | true 1) — set proportional via noise_level

    # ── Experiment ─────────────────────────────────────────────────────────
    n_trials: int = 500
    trap_probability: float = 0.05   # legacy trap (known states) — kept for compat

    # ── Bell-pair verification (EVBTCST) ───────────────────────────────────
    delta: float = 0.10              # fraction of trials that are verification trials
    sprt_alpha: float = 0.01         # SPRT false-positive rate
    sprt_beta: float = 0.0001        # SPRT false-negative rate target (<0.9999 detection)
    sprt_threshold_k: int = 3        # min failed verif. trials before SPRT flags

    # ── Crypto ─────────────────────────────────────────────────────────────
    gf_prime: int = 257

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
                f"N = n_controllers+3 = {self.n_controllers+3} must be ODD."
            )
        if self.n_controllers <= 3 * self.max_byzantine:
            raise ValueError(
                f"Byzantine bound violated: need n > 3f, "
                f"got n={self.n_controllers}, f={self.max_byzantine}. "
                f"Min valid n for f={self.max_byzantine} is "
                f"{3*self.max_byzantine+2} (first even integer > {3*self.max_byzantine})."
            )
        if not (0.0 <= self.delta <= 1.0):
            raise ValueError(f"delta must be in [0,1], got {self.delta}")

        self.threshold      = self.max_byzantine + 1
        self.n_qubits       = self.n_controllers + 3
        self.tq_noise_level = self.tq_noise_factor * self.noise_level

    @property
    def n(self) -> int:
        return self.n_controllers

    @property
    def f(self) -> int:
        return self.max_byzantine

    def summary(self) -> str:
        return (
            f"SimConfig(n={self.n_controllers}, f={self.max_byzantine}, "
            f"N={self.n_qubits}, t={self.threshold}, "
            f"p={self.noise_level}, delta={self.delta}, "
            f"trials={self.n_trials})"
        )
