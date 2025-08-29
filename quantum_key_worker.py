import sys
import os
import math
import random
import binascii
import json

# Minimal imports for quantum key generation
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

def is_power_of_two(n):
    return (n & (n-1) == 0) and n != 0

def run_high_dimensional_bb84_protocol(key_length=256, dimension=4, backend=None):
    # Minimal, non-Streamlit version for subprocess
    if not is_power_of_two(dimension):
        return None
    bits_per_symbol = math.log2(dimension)
    decoy_ratio = 0.3
    initial_length = int(key_length * 3 / bits_per_symbol / (1 - decoy_ratio))
    alice_symbols = [random.randint(0, dimension-1) for _ in range(initial_length)]
    alice_bases = [random.choice([0, 1]) for _ in range(initial_length)]
    bob_bases = [random.choice([0, 1]) for _ in range(initial_length)]
    num_qubits_per_symbol = int(math.ceil(math.log2(dimension)))
    if backend is None:
        backend = AerSimulator()
    batch_size = 10
    circuits = []
    for batch_start in range(0, initial_length, batch_size):
        batch_end = min(batch_start + batch_size, initial_length)
        batch_symbols = alice_symbols[batch_start:batch_end]
        batch_alice_bases = alice_bases[batch_start:batch_end]
        batch_bob_bases = bob_bases[batch_start:batch_end]
        total_qubits = num_qubits_per_symbol * (batch_end - batch_start)
        qc = QuantumCircuit(total_qubits, total_qubits)
        for i, (symbol, alice_basis, bob_basis) in enumerate(zip(
            batch_symbols, batch_alice_bases, batch_bob_bases
        )):
            qubit_start = i * num_qubits_per_symbol
            qubit_end = (i + 1) * num_qubits_per_symbol
            symbol_bin = format(symbol, f'0{num_qubits_per_symbol}b')
            for j, bit in enumerate(symbol_bin):
                if bit == '1':
                    qc.x(qubit_start + j)
            if alice_basis == 1:
                qc.h(range(qubit_start, qubit_end))
                for j in range(num_qubits_per_symbol):
                    for k in range(j+1, num_qubits_per_symbol):
                        angle = math.pi / (2 ** (k - j))
                        qc.cp(angle, qubit_start + k, qubit_start + j)
                for j in range(num_qubits_per_symbol//2):
                    qc.swap(qubit_start + j, qubit_start + num_qubits_per_symbol - j - 1)
            if bob_basis == 1:
                for j in range(num_qubits_per_symbol//2):
                    qc.swap(qubit_start + j, qubit_start + num_qubits_per_symbol - j - 1)
                for j in range(num_qubits_per_symbol-1, -1, -1):
                    for k in range(num_qubits_per_symbol-1, j, -1):
                        angle = -math.pi / (2 ** (k - j))
                        qc.cp(angle, qubit_start + k, qubit_start + j)
                qc.h(range(qubit_start, qubit_end))
            qc.measure(range(qubit_start, qubit_end), range(qubit_start, qubit_end))
        circuits.append(qc)
    measured_symbols = []
    try:
        job = backend.run(circuits, shots=1, memory=True)
        result = job.result()
        for circuit_idx, circuit in enumerate(circuits):
            memory = result.get_memory(circuit_idx)[0]
            batch_symbol_count = circuit.num_qubits // num_qubits_per_symbol
            for i in range(batch_symbol_count):
                start_idx = i * num_qubits_per_symbol
                end_idx = (i + 1) * num_qubits_per_symbol
                measured_bits = memory[start_idx:end_idx]
                measured_symbol = int(measured_bits, 2) % dimension
                measured_symbols.append(measured_symbol)
    except Exception:
        # fallback: random symbols
        measured_symbols = [random.randint(0, dimension-1) for _ in range(initial_length)]
    # Sifting
    sifted_key_alice = []
    sifted_key_bob = []
    for i in range(initial_length):
        if alice_bases[i] == bob_bases[i]:
            sifted_key_alice.append(alice_symbols[i])
            sifted_key_bob.append(measured_symbols[i])
    # Remove sample indices
    sample_size = min(50, len(sifted_key_alice) // 2)
    sample_indices = random.sample(range(len(sifted_key_alice)), sample_size) if sample_size > 0 else []
    final_key_alice = []
    final_key_bob = []
    for i in range(len(sifted_key_alice)):
        if i not in sample_indices:
            final_key_alice.append(sifted_key_alice[i])
            final_key_bob.append(sifted_key_bob[i])
    symbol_bits = int(math.log2(dimension))
    alice_binary = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in final_key_alice)
    bob_binary = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in final_key_bob)
    key_str = alice_binary
    if len(key_str) < key_length:
        key_str += ''.join(random.choice('01') for _ in range(key_length - len(key_str)))
    elif len(key_str) > key_length:
        key_str = key_str[:key_length]
    key_bytes = int(key_str, 2).to_bytes((len(key_str) + 7) // 8, byteorder='big')
    security_parameter = 128
    hkdf = HKDF(
        algorithm=hashes.SHA3_256(),
        length=(key_length + security_parameter) // 8,
        salt=None,
        info=b'high-dim-bb84-reconciled-key',
        backend=default_backend()
    )
    final_key = hkdf.derive(key_bytes)[:key_length//8]
    return binascii.hexlify(final_key).decode()

if __name__ == "__main__":
    key_length = 256
    if len(sys.argv) > 1:
        try:
            key_length = int(sys.argv[1])
        except Exception:
            pass
    key = run_high_dimensional_bb84_protocol(key_length, 4)
    if key:
        print(key)
    else:
        print("", end="")
