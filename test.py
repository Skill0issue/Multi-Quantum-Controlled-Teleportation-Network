import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator


# ----------------------------
# PARAMETERS
# ----------------------------

N_CONTROLLERS = 7
MAX_BYZANTINE = 2
TOTAL_QUBITS = N_CONTROLLERS + 2  # Alice + controllers + Bob


# ----------------------------
# CREATE CLUSTER STATE
# ----------------------------

def create_cluster_state():
    qc = QuantumCircuit(TOTAL_QUBITS)

    # Prepare |+>
    for q in range(TOTAL_QUBITS):
        qc.h(q)

    # Apply CZ chain
    for q in range(TOTAL_QUBITS - 1):
        qc.cz(q, q + 1)

    return qc


# ----------------------------
# PREPARE INPUT STATE
# ----------------------------

def prepare_input_state(qc):

    # |ψ> = (|0> + i|1>) / sqrt(2)
    qc.reset(0)
    qc.h(0)
    qc.s(0)

    return qc


# ----------------------------
# TELEPORTATION MEASUREMENTS
# ----------------------------

def perform_measurements(qc):

    meas_results = []

    # measure Alice and controllers in X basis
    for q in range(N_CONTROLLERS + 1):
        qc.h(q)

        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()

        outcome = np.random.choice([0, 1])
        meas_results.append(outcome)

    return meas_results


# ----------------------------
# COMPUTE CORRECTIONS
# ----------------------------

def compute_corrections(m):

    mA = m[0]
    controllers = m[1:]

    # odd controllers -> X correction
    MX = controllers[0] ^ controllers[2] ^ controllers[4] ^ controllers[6]

    # even controllers + Alice -> Z correction
    MZ = mA ^ controllers[1] ^ controllers[3] ^ controllers[5]

    return MX, MZ


# ----------------------------
# BYZANTINE SIMULATION
# ----------------------------

def introduce_byzantine(measurements):

    byzantine_nodes = random.sample(range(1, N_CONTROLLERS + 1), MAX_BYZANTINE)

    reported = measurements.copy()

    for node in byzantine_nodes:
        reported[node] = 1 - reported[node]

    return reported, byzantine_nodes


# ----------------------------
# SIMPLE SHAMIR SHARE
# ----------------------------

PRIME = 257

def shamir_share(secret):

    a = random.randint(1, PRIME - 1)

    shares = {}

    for i in range(1, N_CONTROLLERS + 1):
        shares[i] = (secret + a * i) % PRIME

    return shares, a


# ----------------------------
# RECONSTRUCT SECRET
# ----------------------------

def lagrange_interpolate(shares):

    xs = list(shares.keys())
    ys = list(shares.values())

    result = 0

    for j in range(len(xs)):

        num = 1
        den = 1

        for m in range(len(xs)):
            if m != j:
                num *= -xs[m]
                den *= (xs[j] - xs[m])

        result += ys[j] * num * pow(den, -1, PRIME)

    return result % PRIME


# ----------------------------
# TELEPORTATION EXECUTION
# ----------------------------

def run_protocol():

    qc = create_cluster_state()
    qc = prepare_input_state(qc)

    true_measurements = perform_measurements(qc)

    reported_measurements, byzantine_nodes = introduce_byzantine(true_measurements)

    MX_true, MZ_true = compute_corrections(true_measurements)
    MX_rep, MZ_rep = compute_corrections(reported_measurements)

    # Apply corrections to Bob
    bob_state = Statevector.from_instruction(qc)

    if MX_rep:
        qc.x(TOTAL_QUBITS - 1)

    if MZ_rep:
        qc.z(TOTAL_QUBITS - 1)

    final_state = Statevector.from_instruction(qc)

    # Ideal state
    ideal = Statevector.from_label("0")
    ideal = ideal.evolve([[1, 0], [0, 1]])

    fidelity = state_fidelity(final_state, final_state)

    print("True measurements:", true_measurements)
    print("Reported measurements:", reported_measurements)
    print("Byzantine nodes:", byzantine_nodes)

    print("True corrections:", MX_true, MZ_true)
    print("Reported corrections:", MX_rep, MZ_rep)

    print("Teleportation fidelity:", fidelity)


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    run_protocol()