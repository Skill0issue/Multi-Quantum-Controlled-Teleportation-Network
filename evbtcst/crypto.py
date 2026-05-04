"""
crypto.py — Pedersen commitments and SHA-256 hash VSS primitives.

Pedersen group parameters (128-bit safe prime, generated + verified offline)
-----------------------------------------------------------------------------
    P  = 313136867555460223635466608969886933223   (safe prime, 128-bit)
    Q  = (P-1)/2 = 156568433777730111817733304484943466611  (prime)
    G  = 2    (generator of order-Q subgroup of Z_P*)
    H  = 93887408135706653947728378060939709224
         H = G^x mod P for some x nobody knows (generated as G^rand)

Commitment scheme
-----------------
  Pedersen:   C = G^v * H^r  mod P
              - Perfectly hiding  (r uniform random → C reveals nothing)
              - Computationally binding (breaking requires DL in Z_P*)
              - Homomorphic: C(v1,r1)*C(v2,r2) = C(v1+v2, r1+r2) mod P

  Hash VSS:   H_{k,j} = SHA256(k || j || share || nonce)
              - Computationally binding (preimage resistance)
              - Each controller commits to every share before sending it
              - Hash mismatches signal equivocation

Note: group size 128 bits is adequate for this simulation.
Production deployments should use ≥ 256-bit groups.
"""

import hashlib
import os
import secrets

# ── Pedersen group parameters ─────────────────────────────────────────────────
_P  = 313136867555460223635466608969886933223
_Q  = 156568433777730111817733304484943466611
_G  = 2
_H  = 93887408135706653947728378060939709224


# ── Pedersen commitment ───────────────────────────────────────────────────────

def pedersen_commit(value: int, rand: int) -> int:
    """
    Compute C = G^value * H^rand mod P.

    Parameters
    ----------
    value : secret value (0 or 1 for measurement bits)
    rand  : random blinding factor in [1, Q-1]
    """
    return (pow(_G, value, _P) * pow(_H, rand, _P)) % _P


def pedersen_verify(commitment: int, value: int, rand: int) -> bool:
    return commitment == pedersen_commit(value, rand)


def pedersen_rand() -> int:
    """Sample a uniform random blinding factor."""
    return secrets.randbelow(_Q - 1) + 1


def pedersen_combine(commits: list[int]) -> int:
    """Homomorphic product: C(v1+v2+...) = ∏ C(vi) mod P."""
    result = 1
    for c in commits:
        result = (result * c) % _P
    return result


def pedersen_commit_sum(commits: list[int], value_sum: int, rand_sum: int) -> bool:
    """
    Verify that ∏ C_k = C(value_sum, rand_sum).
    Used for aggregate Pedersen check: product of individual commitments
    should equal a commitment to the Shamir-reconstructed aggregate.
    """
    return pedersen_combine(commits) == pedersen_commit(value_sum % _P, rand_sum % _Q)


# ── Hash VSS commitment ───────────────────────────────────────────────────────

def hash_commit(sender: int, receiver: int, share: int, nonce: bytes) -> str:
    """
    H_{k,j} = SHA256(sender || receiver || share || nonce).
    Binds sender k to the specific share value they committed to send receiver j.
    """
    data = f"{sender}:{receiver}:{share}:".encode() + nonce
    return hashlib.sha256(data).hexdigest()


def hash_verify(commitment: str, sender: int, receiver: int,
                share: int, nonce: bytes) -> bool:
    return commitment == hash_commit(sender, receiver, share, nonce)


def fresh_nonce() -> bytes:
    """16 random bytes — sufficient for hash commitment binding."""
    return os.urandom(16)
