"""
secret_sharing.py — Full commit-reveal VSS protocol with hash + Pedersen commitments.

Protocol phases (called in order from simulator.py)
----------------------------------------------------
Phase 1  pedersen_commit_phase   — each controller commits to their m_k
Phase 2  hash_commit_and_share   — each controller commits to + distributes shares
Phase 3  verify_received_shares  — each receiver checks hash commitments
Phase 4  pedersen_aggregate_check— homomorphic check + individual challenge
→       build_detection_set      — union of all detected Byzantine nodes
Phase 5  recover_measurements    — for detected nodes, recover true m_k from
                                   committed-polynomial shares (honest half only)

Detection guarantees with both mechanisms active
------------------------------------------------
Strategy        | Detected by      | Recoverable
----------------|------------------|------------
coherent_flip   | NOT detectable   | NO  (information-theoretic limit)
equivocate      | Hash VSS (Ph 3)  | YES (use committed poly_a shares)
withhold        | Missing commit   | NO  (no shares to recover from)

Detection rate: ~2/3 per Byzantine node per trial (1/3 coherent_flip escapes).
"""

from __future__ import annotations
import random
import logging
from typing import Optional

from crypto import (
    pedersen_commit, pedersen_verify, pedersen_rand,
    pedersen_combine, pedersen_commit_sum,
    hash_commit, hash_verify, fresh_nonce,
)

log = logging.getLogger("secret_sharing")

# ── GF(257) arithmetic ────────────────────────────────────────────────────────
_P = 257

def _modinv(a: int, p: int = _P) -> int:
    g, x = p, 0
    a = a % p
    b, y = a, 1
    while b:
        q = g // b
        g, b = b, g - q * b
        x, y = y, x - q * y
    return x % p

def _poly_eval(coeffs: list[int], x: int) -> int:
    r = 0
    for c in reversed(coeffs):
        r = (r * x + c) % _P
    return r

def _lagrange_at_zero(points: list[tuple[int, int]]) -> int:
    """Lagrange interpolation to evaluate polynomial at x=0 over GF(257)."""
    result = 0
    for i, (xi, yi) in enumerate(points):
        num, den = yi, 1
        for j, (xj, _) in enumerate(points):
            if i != j:
                num = (num * (0 - xj)) % _P
                den = (den * (xi - xj)) % _P
        result = (result + num * _modinv(den)) % _P
    return result


# ── Data structures ───────────────────────────────────────────────────────────

class VSSState:
    """All VSS protocol state for one trial, passed between phases."""
    def __init__(self, n: int, t: int):
        self.n = n
        self.t = t
        # Phase 1
        self.pedersen_commits: dict[int, Optional[int]] = {}     # node → C_k (None if absent)
        self.pedersen_rands:   dict[int, Optional[int]] = {}     # node → r_k
        self.pedersen_vals:    dict[int, Optional[int]] = {}     # node → committed m_k (simulation only)
        # Phase 2
        self.share_hash_commits: dict[int, Optional[dict[int,str]]] = {}   # node → {j: hash}
        self.shares_sent:        dict[int, Optional[dict[int,int]]] = {}   # node → {j: actual_share}
        self.share_nonces:       dict[int, Optional[dict[int,bytes]]] = {} # node → {j: nonce}
        self.committed_shares:   dict[int, Optional[dict[int,int]]] = {}   # node → {j: COMMITTED share (poly_a)}
        # Phase 3
        self.hash_complaints: set[int] = set()   # nodes caught by hash VSS
        self.withhold_absent: set[int] = set()   # nodes that never published commitments
        # Phase 4
        self.pedersen_caught: set[int] = set()   # nodes caught by Pedersen challenge


# ── Phase 1: Pedersen commitment to individual measurements ──────────────────

def pedersen_commit_phase(
    measurements: list[int],
    n: int,
    byzantine_nodes: dict[int, str],
    state: VSSState,
) -> None:
    """
    Each controller k publishes C_k = G^{m_k} * H^{r_k}.

    Byzantine strategies:
      coherent_flip : commits to 1-m_k (wrong value, but consistently so)
      equivocate    : commits to m_k (correct — equivocation is in shares, not here)
      withhold      : publishes nothing → None in state dict (detected in Phase 3)
    """
    for k in range(1, n + 1):
        m_k = measurements[k - 1]
        strategy = byzantine_nodes.get(k, "honest")

        if strategy == "withhold":
            state.pedersen_commits[k] = None
            state.pedersen_rands[k]   = None
            state.pedersen_vals[k]    = None
            state.withhold_absent.add(k)
            log.debug(f"  [C{k}/withhold] skipped Pedersen commit")
        else:
            r_k = pedersen_rand()
            committed_val = (1 - m_k) if strategy == "coherent_flip" else m_k
            C_k = pedersen_commit(committed_val, r_k)
            state.pedersen_commits[k] = C_k
            state.pedersen_rands[k]   = r_k
            state.pedersen_vals[k]    = committed_val
            log.debug(f"  [C{k}/{strategy}] Pedersen commit to {committed_val} (true={m_k})")


# ── Phase 2: Share generation + hash commitments ──────────────────────────────

def hash_commit_and_share(
    measurements: list[int],
    n: int,
    t: int,
    byzantine_nodes: dict[int, str],
    state: VSSState,
) -> None:
    """
    Each controller k:
      1. Picks polynomial f_k with f_k(0) = m_k (or 1-m_k for coherent_flip).
      2. Publishes H_{k,j} = SHA256(k||j||f_k(j)||nonce) for ALL j.
      3. Sends (f_k(j), nonce) to controller j.

    Equivocate:
      - Commits (publishes hashes) based on poly_a (f_a(0) = 1-m_k).
      - Sends poly_a shares to j ≤ n//2, poly_b shares (f_b(0)=0) to j > n//2.
      - Hash check fails for j > n//2 → detected in Phase 3.
      - Recovery can use the hash-committed poly_a values directly.

    Withhold:
      - Publishes no hash commitments → already detected in Phase 1.
      - Sends zero shares for completeness.
    """
    half = n // 2

    for k in range(1, n + 1):
        m_k = measurements[k - 1]
        strategy = byzantine_nodes.get(k, "honest")

        if strategy == "withhold":
            state.share_hash_commits[k] = None
            state.shares_sent[k]        = {j: 0 for j in range(1, n + 1)}
            state.share_nonces[k]       = None
            state.committed_shares[k]   = None
            continue

        # Determine the polynomial secret
        if strategy == "coherent_flip":
            secret = 1 - m_k
        else:
            secret = m_k

        # Primary polynomial (poly_a)
        coeffs_a = [secret] + [random.randint(0, _P - 1) for _ in range(t - 1)]
        poly_a_shares = {j: _poly_eval(coeffs_a, j) for j in range(1, n + 1)}

        # Equivocate: secondary polynomial for second half
        if strategy == "equivocate":
            coeffs_b = [0] + [random.randint(0, _P - 1) for _ in range(t - 1)]
            poly_b_shares = {j: _poly_eval(coeffs_b, j) for j in range(1, n + 1)}

        # Generate nonces and hash commitments (always based on poly_a)
        nonces = {j: fresh_nonce() for j in range(1, n + 1)}
        hashes = {
            j: hash_commit(k, j, poly_a_shares[j], nonces[j])
            for j in range(1, n + 1)
        }

        # Actual shares sent (equivocate sends poly_b to second half)
        if strategy == "equivocate":
            actual_sent = {
                j: (poly_a_shares[j] if j <= half else poly_b_shares[j])
                for j in range(1, n + 1)
            }
        else:
            actual_sent = poly_a_shares.copy()

        state.share_hash_commits[k] = hashes
        state.shares_sent[k]        = actual_sent
        state.share_nonces[k]       = nonces
        state.committed_shares[k]   = poly_a_shares   # the committed polynomial


# ── Phase 3: Hash verification and complaint protocol ─────────────────────────

def verify_received_shares(n: int, state: VSSState) -> None:
    """
    Each controller j checks the share received from k against H_{k,j}.

    Absent commitments (withhold) were already recorded in Phase 1.
    Any hash mismatch here catches equivocation.

    After complaints:
    - For equivocate nodes: everyone learns the committed (poly_a) value for the
      complained-about receiver via the public complaint-resolution reveal.
      We model this by making committed_shares available for recovery.
    """
    for j in range(1, n + 1):
        for k in range(1, n + 1):
            if k == j:
                continue
            # Already flagged in Phase 1
            if k in state.withhold_absent:
                continue
            if state.share_hash_commits[k] is None:
                state.hash_complaints.add(k)
                continue

            received  = state.shares_sent[k][j]
            committed = state.share_hash_commits[k][j]
            nonce     = state.share_nonces[k][j]

            if not hash_verify(committed, k, j, received, nonce):
                state.hash_complaints.add(k)
                log.debug(f"  [C{j}] Hash mismatch on share from C{k} → complaint")


# ── Phase 4: Pedersen aggregate check + individual challenge ─────────────────

def pedersen_aggregate_check(
    n: int,
    t: int,
    already_detected: set[int],
    state: VSSState,
) -> None:
    """
    Among nodes NOT yet detected, run the Pedersen aggregate check.

    Steps:
    1. Collect honest nodes' shares → S_j = Σ_{honest k} f_k(j).
    2. Lagrange reconstruct F(0) = Σ_{honest k} f_k(0) (Shamir aggregate).
    3. Compute product of Pedersen commitments: ∏ C_k for honest k.
    4. Verify: G^{F(0)} * H^{Σ r_k} = ∏ C_k.
    5. If fails → run individual challenge to find the mismatch.

    Individual challenge:
    - For each honest k, compare Shamir-reconstructed f_k(0) against
      their Pedersen-committed value.
    - Mismatch → k is a coherent_flip Byzantine caught by Pedersen.

    Note: coherent_flip commits to 1-m_k AND shares polynomial with
    f_k(0) = 1-m_k, so internally consistent → passes aggregate check.
    Only an INCOHERENT variant (commit right, share wrong) is caught here.
    For this simulation all coherent_flips are undetectable by design.
    """
    honest = [k for k in range(1, n + 1)
              if k not in already_detected
              and k not in state.withhold_absent
              and state.pedersen_commits[k] is not None]

    if len(honest) < t:
        log.warning("Not enough honest nodes for Pedersen aggregate check")
        return

    # Shamir aggregate from honest shares
    agg_points = []
    for j in honest:
        S_j = sum(state.shares_sent[k][j] for k in honest) % _P
        agg_points.append((j, S_j))
    F0 = _lagrange_at_zero(agg_points[:t])

    # Pedersen product check
    commits   = [state.pedersen_commits[k] for k in honest]
    rands_sum = sum(state.pedersen_rands[k] for k in honest)

    if not pedersen_commit_sum(commits, F0, rands_sum):
        log.debug("  Pedersen aggregate check FAILED — running individual challenge")
        # Individual challenge: find k where f_k(0) ≠ committed value
        for k in honest:
            k_pts = [(j, state.shares_sent[k][j]) for j in honest if j != k]
            if len(k_pts) < t:
                continue
            f_k0 = _lagrange_at_zero(k_pts[:t])
            if f_k0 != state.pedersen_vals[k]:
                state.pedersen_caught.add(k)
                log.debug(f"  [Pedersen challenge] C{k} caught: "
                          f"f(0)={f_k0} ≠ committed {state.pedersen_vals[k]}")
    else:
        log.debug("  Pedersen aggregate check passed")


def build_detection_set(state: VSSState) -> set[int]:
    """Union of all detection mechanisms."""
    return state.withhold_absent | state.hash_complaints | state.pedersen_caught


# ── Phase 5: Recovery ─────────────────────────────────────────────────────────

def recover_measurements(
    reported_ctrl: list[int],
    n: int,
    t: int,
    detected_nodes: set[int],
    state: VSSState,
) -> list[int]:
    """
    For each detected node, attempt to recover the true measurement.

    Equivocate nodes: committed poly_a has f_a(0) = 1-m_k (flipped).
      - committed_shares[k] holds poly_a values for ALL j.
      - Use poly_a shares from HONEST reporters (not detected) only.
      - Lagrange gives f_a(0) = 1-m_k → true m_k = 1 - f_a(0) % 2.

    Withhold nodes: no polynomial information → cannot recover.
      - Leave reported value as-is (direct report was honest for withhold).

    coherent_flip (if somehow detected): same recovery as equivocate.

    Returns the recovered ctrl_bits list.
    """
    recovered = list(reported_ctrl)
    honest_reporters = [j for j in range(1, n + 1) if j not in detected_nodes]

    for b in detected_nodes:
        if not (1 <= b <= n):
            continue

        # Withhold: honest direct report, no polynomial to recover from
        if b in state.withhold_absent:
            log.debug(f"  [Recovery C{b}/withhold] No polynomial — keeping direct report")
            continue

        # For equivocate: use the COMMITTED polynomial (poly_a) shares
        # These are hash-verified values from honest collectors
        if state.committed_shares.get(b) is None:
            log.warning(f"  [Recovery C{b}] No committed shares available")
            continue

        # Gather poly_a share from honest reporters only
        pts = [
            (j, state.committed_shares[b][j])
            for j in honest_reporters
            if state.committed_shares[b].get(j) is not None
        ]

        if len(pts) < t:
            log.warning(f"  [Recovery C{b}] Not enough honest share points ({len(pts)} < {t})")
            continue

        f0 = _lagrange_at_zero(pts[:t])
        recovered_m = 1 - (f0 % 2)   # equivocate: f_a(0) = 1-m_k → m_k = 1-f_a(0)
        log.debug(f"  [Recovery C{b}] f_a(0)={f0} → recovered m={recovered_m}")
        recovered[b - 1] = recovered_m

    return recovered
