"""
Single-Hop and Double-Hop Quantum Controlled Teleportation
Implementation using Qiskit

Based on: "A framework for quantum controlled teleportation in multi-hop networks"
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, state_fidelity,partial_trace
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


def create_ghz_state(qc, qubits):
    """
    Create GHZ state: (|000⟩ + |111⟩)/√2
    
    Args:
        qc: Quantum circuit
        qubits: List of 3 qubit indices [q0, q1, q2]
    """
    qc.h(qubits[0])
    qc.cx(qubits[0], qubits[1])
    qc.cx(qubits[1], qubits[2])


def inverse_ghz_transform(qc, qubits):
    """
    Apply inverse GHZ transformation (U†_GHZ)
    
    Args:
        qc: Quantum circuit
        qubits: List of 3 qubit indices [q0, q1, q2]
    """
    qc.cx(qubits[1], qubits[2])
    qc.cx(qubits[0], qubits[1])
    qc.h(qubits[0])


def prepare_input_state(qc, qubit, theta, phi):
    """
    Prepare input state: |φ⟩ = cos(θ)|0⟩ + e^(iφ)sin(θ)|1⟩
    
    Args:
        qc: Quantum circuit
        qubit: Qubit index
        theta: Angle θ ∈ [0, π]
        phi: Angle φ ∈ [0, 2π]
    """
    qc.ry(2 * theta, qubit)  # Ry(2θ) rotation
    qc.rz(phi, qubit)        # Rz(φ) rotation


class SingleHopTeleportation:
    """
    Single-Hop Quantum Controlled Teleportation
    
    Protocol:
    1. Alice prepares state |φ⟩
    2. Charlie creates GHZ state shared with Alice and Bob
    3. Alice and Charlie measure their qubits
    4. Charlie tells Bob what corrections to apply
    5. Bob's qubit now holds |φ⟩
    """
    
    def __init__(self, theta=np.pi/4, phi=0):
        """
        Initialize single-hop teleportation
        
        Args:
            theta: Input state parameter
            phi: Input state parameter
        """
        self.theta = theta
        self.phi = phi
        self.circuit = None
        
    def build_circuit(self):
        """Build the single-hop quantum controlled teleportation circuit"""
        
        # Create quantum and classical registers
        # Qubits: Alice(0), Charlie(1), Hop1(2), Bob(3)
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(2, 'c')  # For measurements
        qc = QuantumCircuit(qr, cr)
        
        # Label qubits for clarity
        alice = 0
        charlie = 1
        hop1 = 2
        bob = 3
        
        # === STEP 1: Prepare input state on Alice's qubit ===
        qc.barrier(label='Input State')
        prepare_input_state(qc, alice, self.theta, self.phi)
        
        # === STEP 2: Create GHZ state (Charlie, Hop1, Bob) ===
        qc.barrier(label='GHZ Channel')
        create_ghz_state(qc, [charlie, hop1, bob])
        
        # === STEP 3: Apply inverse GHZ transformation ===
        qc.barrier(label='U†_GHZ')
        inverse_ghz_transform(qc, [alice, charlie, hop1])
        
        # === STEP 4: Measure Alice and Charlie ===
        qc.barrier(label='Measure')
        qc.measure(alice, 0)
        qc.measure(charlie, 1)
        
        # === STEP 5: Apply corrections based on measurements ===
        qc.barrier(label='Corrections')
        # If Alice measured 1: apply Z to Bob
        with qc.if_test((cr[0],1)):
            qc.z(bob)
        # If Charlie measured 1: apply X to Bob
        with qc.if_test((cr[1],1)):
            qc.x(bob)
        
        self.circuit = qc
        return qc
    
    def simulate(self):
        """Simulate the circuit and return fidelity"""
        if self.circuit is None:
            self.build_circuit()
        
        # Run statevector simulation
        self.circuit.save_statevector() 
        simulator = AerSimulator(method='statevector')
        result = simulator.run(self.circuit).result()
        final_state = result.get_statevector()
        
        # Extract Bob's qubit state (qubit 3)
        # We need to trace out all other qubits
        from qiskit.quantum_info import DensityMatrix
        
        rho = DensityMatrix(final_state)
        # Trace out qubits 0, 1, 2 (keep only Bob's qubit 3)
        rho = partial_trace(rho,[0,1,2])
        
        # Target state
        target_state = np.array([
            np.cos(self.theta),
            np.exp(1j * self.phi) * np.sin(self.theta)
        ])
        
        # Calculate fidelity
        fidelity = state_fidelity(rho, target_state)
        
        return fidelity
    
    def draw(self, filename=None):
        """Draw the circuit"""
        if self.circuit is None:
            self.build_circuit()
        
        fig = self.circuit.draw(output='mpl', style='iqp', fold=-1)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Circuit saved to {filename}")
        
        return fig

def demonstrate_teleportation():
    """Demonstrate single-hop and double-hop teleportation"""
    
    print("=" * 70)
    print("QUANTUM CONTROLLED TELEPORTATION - DEMONSTRATION")
    print("=" * 70)
    
    # Test different quantum states
    test_states = [
        (0, 0, "|0⟩"),
        (np.pi/2, 0, "|+⟩ = (|0⟩ + |1⟩)/√2"),
        (np.pi/2, np.pi, "|-⟩ = (|0⟩ - |1⟩)/√2"),
        (np.pi/4, 0, "cos(π/4)|0⟩ + sin(π/4)|1⟩"),
        (np.pi/3, np.pi/4, "General superposition"),
    ]
    
    print("\n" + "─" * 70)
    print("SINGLE-HOP TELEPORTATION")
    print("─" * 70)
    print(f"{'Input State':<35} {'Fidelity':<15} {'Status':<20}")
    print("─" * 70)
    
    for theta, phi, name in test_states:
        protocol = SingleHopTeleportation(theta, phi)
        protocol.build_circuit()
        fidelity = protocol.simulate()
        
        status = "✓ Perfect" if fidelity > 0.999 else "✗ Failed"
        print(f"{name:<35} {fidelity:<15.6f} {status:<20}")



def visualize_circuits():
    """Create and save circuit visualizations"""
    
    print("\n" + "=" * 70)
    print("GENERATING CIRCUIT DIAGRAMS")
    print("=" * 70)
    
    # Single-hop circuit
    print("\nCreating single-hop circuit diagram...")
    single = SingleHopTeleportation(theta=np.pi/4, phi=0)
    single.build_circuit()
    single.draw('./single_hop_circuit.png')
    
    print("\n" + "=" * 70)


def compare_fidelities():
    """Compare fidelities across the Bloch sphere"""
    
    print("\n" + "=" * 70)
    print("FIDELITY ANALYSIS OVER BLOCH SPHERE")
    print("=" * 70)
    
    n_samples = 20
    theta_vals = np.linspace(0, np.pi, n_samples)
    phi_vals = np.linspace(0, 2*np.pi, n_samples)
    
    single_fidelities = []
    
    print("\nCalculating fidelities for different input states...")
    print(f"Total states to test: {n_samples * n_samples}")
    
    count = 0
    for theta in theta_vals:
        for phi in phi_vals:
            # Single-hop
            single = SingleHopTeleportation(theta, phi)
            single.build_circuit()
            fid_single = single.simulate()
            single_fidelities.append(fid_single)

            count += 1
            if count % 50 == 0:
                print(f"  Progress: {count}/{n_samples * n_samples}")
    
    # Statistics
    print("\n" + "─" * 70)
    print("FIDELITY STATISTICS")
    print("─" * 70)
    
    print(f"\n{'Metric':<30} {'Single-Hop':<20}")
    print("─" * 70)
    print(f"{'Mean Fidelity':<30} {np.mean(single_fidelities):<20.6f}")
    print(f"{'Std Dev':<30} {np.std(single_fidelities):<20.6f}")
    print(f"{'Min Fidelity':<30} {np.min(single_fidelities):<20.6f}")
    print(f"{'Max Fidelity':<30} {np.max(single_fidelities):<20.6f}")
    
    # Create plot (only ONE axis needed)
    fig, ax = plt.subplots(figsize=(6, 5))

    data_range = np.max(single_fidelities) - np.min(single_fidelities)

    if data_range < 1e-12:
        print("\nAll fidelities are identical (perfect teleportation).")
        print("Histogram is not meaningful because distribution is a delta peak at 1.0.")

        # Draw a visible spike at fidelity = 1.0
        ax.axvline(1.0, linewidth=3)

        # Add annotation text
        ax.text(
            1.0, 0.5,
            "All fidelities = 1.0\nPerfect teleportation",
            ha="center",
            va="center",
            fontsize=12 
        )

        # Fix axis scaling so it doesn't look empty
        ax.set_xlim(0.95, 1.05)
        ax.set_ylim(0, 1)

    else:
        ax.hist(single_fidelities, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(
            np.mean(single_fidelities),
            linestyle="--",
            label=f"Mean: {np.mean(single_fidelities):.6f}"
        )
        ax.legend()   # Legend only when something is labeled

    ax.set_xlabel("Fidelity")
    ax.set_ylabel("Count")
    ax.set_title("Single-Hop Fidelity Distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./fidelity_comparison.png", dpi=300)
    print("\nFidelity comparison plot saved to fidelity_comparison.png")

    print("\n" + "=" * 70)



if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_teleportation()
    visualize_circuits()
    compare_fidelities()
    
    print("\n" + "═" * 70)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("═" * 70)
    
    print("\nGenerated files:")
    print("  • single_hop_circuit.png - Single-hop circuit diagram")
    print("  • fidelity_comparison.png - Fidelity analysis plot")
    print("\n" + "═" * 70)
