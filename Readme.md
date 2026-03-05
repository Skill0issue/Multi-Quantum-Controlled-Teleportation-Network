
# Multi Quantum Controlled Teleportation Network

Research implementation of a **multi-hop quantum teleportation network protocol** enabling state transfer across distributed quantum nodes without direct quantum channels.

This project investigates routing, concurrency, and resource management in **quantum teleportation networks with intermediate control nodes (Charlies)**.

---

## Overview

Quantum teleportation allows transmission of an unknown quantum state using:

- shared entanglement
- classical communication
- local operations

Standard teleportation involves three parties:

Alice → Charlie → Bob

However, large-scale quantum networks require **multi-hop teleportation** where states traverse multiple intermediate nodes.

This project explores **multi quantum controlled teleportation (MQCT)** where:

- nodes can act as sender, receiver, or controller
- teleportation occurs across multi-hop paths
- intermediate nodes coordinate entanglement and measurements

---

## Research Goal

Design a **scalable teleportation network protocol** that supports:

- multi-hop state transfer
- concurrent teleportation sessions
- overlapping network routes
- minimal entanglement resources
- high fidelity transmission

---

## Protocol Concept

In a network of quantum nodes:

Alice ── C1 ── C2 ── ... ── Cn ── Bob

each intermediate node performs controlled teleportation operations.

The protocol uses:

- Bell state entanglement
- GHZ resources for control
- Bell-basis measurements
- classical communication for Pauli corrections

The state evolves across hops as:

ψ → X^(a_k) Z^(b_k) ψ

After multiple hops the final correction becomes:

X^(⊕ a_k) Z^(⊕ b_k)

which can be applied once at the destination node.

---

## Key Idea: Pauli Frame Aggregation

Instead of correcting at every hop, the protocol tracks Pauli corrections in a **Pauli frame**:

F = (x_total , z_total)

where

x_total = ⊕ a_k  
z_total = ⊕ b_k

This allows:

- concurrent teleportation across nodes
- reduced quantum gate overhead
- simplified intermediate node operations

---

## Network Challenges Studied

### Multi-path Routing

Teleportation requests may share intermediate nodes.

Example:

A → Y → B  
C → Y → D

The protocol studies how **shared nodes process concurrent teleportation sessions** without state interference.

### Resource Scheduling

Teleportation consumes:

- entangled Bell pairs
- classical communication bandwidth
- quantum memory

The system investigates strategies for:

- entanglement distribution
- congestion handling
- request scheduling

### Fault Tolerance

Future extensions explore robustness against:

- node failure
- entanglement loss
- decoherence

Possible approaches include:

- logical qubit teleportation
- redundant path routing
- distributed error correction

---


## Research Direction

Current work focuses on:

- designing concurrency-safe teleportation protocols
- studying network congestion under high traffic
- evaluating entanglement resource consumption
- developing scheduling strategies for quantum networks

---

## Related Concepts

This work connects to research in:

- quantum repeaters
- entanglement routing
- quantum network coding
- measurement-based quantum computation
- distributed quantum information processing

---

## Future Work

Planned extensions include:

- hybrid teleportation + entanglement swapping protocols
- reinforcement learning for teleportation routing
- logical qubit teleportation for fault tolerance
- simulation of large quantum network topologies

---

## Author

Chelamakuri Nihar Kartikeya  
B.Tech Engineering Physics  
Indian Institute of Technology Hyderabad

---

## License

Research prototype implementation for academic use.
