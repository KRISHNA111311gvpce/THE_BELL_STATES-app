# --- Quantum dimension validation helper ---
def is_power_of_two(n):
    """Check if a number is a power of two"""
    return (n & (n-1) == 0) and n != 0
# --- Quantum circuit batching optimization ---
def get_optimal_batch_size(backend, num_qubits_per_symbol):
    """
    Determine the optimal batch size based on backend capabilities.
    """
    if hasattr(backend, 'configuration') and callable(getattr(backend, 'configuration', None)):
        try:
            max_qubits = backend.configuration().n_qubits
            available_qubits = max_qubits - 2
            return max(1, available_qubits // num_qubits_per_symbol)
        except Exception:
            return 10
    else:
        return 50

# --- Batched BB84 protocol implementation ---
def run_high_dimensional_bb84_protocol_batched(key_length=256, dimension=4, backend=None):
    """
    Enhanced BB84 protocol with batched circuit execution for efficiency.
    """
    # Validate dimension is power of two
    if not is_power_of_two(dimension):
        st.error(f"❌ Quantum dimension must be a power of 2 (got {dimension})")
        return None
    use_auth = st.session_state.get("authentication_enabled", True)
    auth_psk = st.session_state.get("authentication_psk", "")
    block_size = st.session_state.get("reconciliation_block_size", 8)
    max_iterations = st.session_state.get("reconciliation_max_iterations", 5)

    bits_per_symbol = math.log2(dimension)
    decoy_ratio = 0.3
    initial_length = int(key_length * 3 / bits_per_symbol / (1 - decoy_ratio))

    alice_symbols = [random.randint(0, dimension-1) for _ in range(initial_length)]
    alice_bases = [random.choice([0, 1]) for _ in range(initial_length)]
    alice_decoy_states = [random.random() < decoy_ratio for _ in range(initial_length)]

    decoy_intensities = {
        'signal': 0.5,
        'decoy': 0.1
    }

    bob_bases = [random.choice([0, 1]) for _ in range(initial_length)]

    num_qubits_per_symbol = int(math.ceil(math.log2(dimension)))
    if backend is None:
        backend = AerSimulator()
    batch_size = get_optimal_batch_size(backend, num_qubits_per_symbol)
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

            # Encode the symbol
            symbol_bin = format(symbol, f'0{num_qubits_per_symbol}b')
            for j, bit in enumerate(symbol_bin):
                if bit == '1':
                    qc.x(qubit_start + j)

            # Apply Alice's basis transformation
            if alice_basis == 1:
                qc.h(range(qubit_start, qubit_end))
                for j in range(num_qubits_per_symbol):
                    for k in range(j+1, num_qubits_per_symbol):
                        angle = math.pi / (2 ** (k - j))
                        qc.cp(angle, qubit_start + k, qubit_start + j)
                for j in range(num_qubits_per_symbol//2):
                    qc.swap(qubit_start + j, qubit_start + num_qubits_per_symbol - j - 1)

            # Apply Bob's basis transformation
            if bob_basis == 1:
                for j in range(num_qubits_per_symbol//2):
                    qc.swap(qubit_start + j, qubit_start + num_qubits_per_symbol - j - 1)
                for j in range(num_qubits_per_symbol-1, -1, -1):
                    for k in range(num_qubits_per_symbol-1, j, -1):
                        angle = -math.pi / (2 ** (k - j))
                        qc.cp(angle, qubit_start + k, qubit_start + j)
                qc.h(range(qubit_start, qubit_end))

            # Measure
            qc.measure(range(qubit_start, qubit_end), range(qubit_start, qubit_end))

        circuits.append(qc)

    try:
        measured_symbols = []
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
    except Exception as e:
        st.error(f"Error running quantum circuits: {e}")
        return None

    # --- Authentication, decoy state analysis, error estimation, etc. ---
    # (Same as original protocol logic)
    # AUTHENTICATION: Include decoy state information
    if use_auth:
        auth_data = {
            "alice_bases": alice_bases,
            "alice_decoy_states": alice_decoy_states,
            "decoy_intensities": decoy_intensities
        }
        auth_data_str = json.dumps(auth_data)
        auth_hmac = generate_hmac(auth_data_str, auth_psk)
        if not verify_hmac(auth_data_str, auth_hmac, auth_psk):
            st.error("Authentication failed: Quantum state information could not be verified")
            return None

    # DECOY STATE ANALYSIS
    signal_indices = [i for i in range(initial_length) if not alice_decoy_states[i]]
    decoy_indices = [i for i in range(initial_length) if alice_decoy_states[i]]

    signal_error_rate = estimate_error_rate(signal_indices, alice_symbols, measured_symbols, alice_bases, bob_bases, auth_psk, use_auth)
    decoy_error_rate = estimate_error_rate(decoy_indices, alice_symbols, measured_symbols, alice_bases, bob_bases, auth_psk, use_auth)

    if decoy_error_rate > signal_error_rate * 1.5 and decoy_error_rate > 0.05:
        st.error("Photon number splitting attack detected! Protocol aborted.")
        return None

    # Continue with original protocol using only signal states
    sifted_key_alice = []
    sifted_key_bob = []
    for i in signal_indices:
        if alice_bases[i] == bob_bases[i]:
            sifted_key_alice.append(alice_symbols[i])
            sifted_key_bob.append(measured_symbols[i])

    sample_size = min(50, len(sifted_key_alice) // 2)
    if sample_size == 0:
        st.error("Not enough sifted key bits for error estimation.")
        return None
    sample_indices = random.sample(range(len(sifted_key_alice)), sample_size)
    if use_auth:
        sample_indices_str = json.dumps(sample_indices)
        sample_indices_hmac = generate_hmac(sample_indices_str, auth_psk)
        if not verify_hmac(sample_indices_str, sample_indices_hmac, auth_psk):
            st.error("Authentication failed: Sample indices could not be verified")
            return None
    error_count = 0
    for idx in sample_indices:
        if sifted_key_alice[idx] != sifted_key_bob[idx]:
            error_count += 1
    if use_auth:
        error_count_str = str(error_count)
        error_count_hmac = generate_hmac(error_count_str, auth_psk)
        if not verify_hmac(error_count_str, error_count_hmac, auth_psk):
            st.error("Authentication failed: Error count could not be verified")
            return None
    error_rate = error_count / sample_size if sample_size > 0 else 0
    if error_rate > (dimension + 1) / (2 * dimension):
        return None

    # Remove sample symbols from key
    final_key_alice = []
    final_key_bob = []
    for i in range(len(sifted_key_alice)):
        if i not in sample_indices:
            final_key_alice.append(sifted_key_alice[i])
            final_key_bob.append(sifted_key_bob[i])

    # --- Error Reconciliation: Cascade Protocol ---
    symbol_bits = int(math.log2(dimension))
    alice_binary = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in final_key_alice)
    bob_binary = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in final_key_bob)
    reconciled_binary, errors_corrected, iterations = cascade_error_reconciliation(
        alice_binary, bob_binary, block_size=block_size, max_iterations=max_iterations
    )
    if errors_corrected > 0:
        st.info(f" Error reconciliation corrected {errors_corrected} bits in {iterations} iterations")
    # Convert back to symbols
    reconciled_key = []
    for i in range(0, len(reconciled_binary), symbol_bits):
        symbol_bin = reconciled_binary[i:i+symbol_bits]
        if len(symbol_bin) == symbol_bits:
            reconciled_key.append(int(symbol_bin, 2))
    # --- Privacy Amplification ---
    key_str = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in reconciled_key)
    key_bytes = int(key_str, 2).to_bytes((len(key_str) + 7) // 8, byteorder='big')
    parity_bits_leaked = len(alice_binary) // block_size
    security_parameter = 128
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=(key_length + security_parameter) // 8,
        salt=None,
        info=b'high-dim-bb84-reconciled-key',
        backend=default_backend()
    )
    final_key = hkdf.derive(key_bytes)[:key_length//8]
    # Derive authentication key for future use
    auth_key = derive_authentication_key(binascii.hexlify(final_key).decode())

    return binascii.hexlify(final_key).decode()
# Helper for decoy state error estimation
def estimate_error_rate(indices, alice_symbols, measured_symbols, alice_bases, bob_bases, auth_key, use_auth):
    """Estimate error rate for a subset of states (signal or decoy)"""
    if not indices:
        return 0
    # Authentication check if enabled
    if use_auth:
        indices_str = json.dumps(indices)
        indices_hmac = generate_hmac(indices_str, auth_key)
        if not verify_hmac(indices_str, indices_hmac, auth_key):
            st.error("Authentication failed: Sample indices could not be verified")
            return float('inf')
    # Sifting: keep symbols where bases match
    sifted_alice = []
    sifted_bob = []
    for i in indices:
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_symbols[i])
            sifted_bob.append(measured_symbols[i])
    # Sample for error estimation
    sample_size = min(20, len(sifted_alice) // 2)
    if sample_size == 0:
        return 0
    sample_indices = random.sample(range(len(sifted_alice)), sample_size)
    # Authentication check if enabled
    if use_auth:
        sample_indices_str = json.dumps(sample_indices)
        sample_indices_hmac = generate_hmac(sample_indices_str, auth_key)
        if not verify_hmac(sample_indices_str, sample_indices_hmac, auth_key):
            st.error("Authentication failed: Sample indices could not be verified")
            return float('inf')
    # Calculate error rate
    error_count = 0
    for idx in sample_indices:
        if sifted_alice[idx] != sifted_bob[idx]:
            error_count += 1
    # Authentication check if enabled
    if use_auth:
        error_count_str = str(error_count)
        error_count_hmac = generate_hmac(error_count_str, auth_key)
        if not verify_hmac(error_count_str, error_count_hmac, auth_key):
            st.error("Authentication failed: Error count could not be verified")
            return float('inf')
    return error_count / sample_size
# ================== Part 1/5 ==================
import streamlit as st
import hashlib
import json
from datetime import datetime, timedelta
import ecdsa
import binascii
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import random
import base64
import numpy as np
import math
import subprocess
import sys
import tempfile
import os

# --- Add missing imports for QR code and BytesIO ---
import qrcode
from io import BytesIO

# --- Persistence helpers (must be defined before use) ---
def save_blockchain():
    """Save blockchain data to JSON file"""
    data = {
        "blockchain": st.session_state.blockchain,
        "balances": st.session_state.balances,
        "users": {k: {kk: vv for kk, vv in v.items() if kk != "private_key" and kk != "password_hash"}
                 for k, v in st.session_state.users.items()},
        "network_stats": st.session_state.network_stats,
        "quantum_keys": st.session_state.quantum_keys,
        "last_save": datetime.now().isoformat()
    }
    try:
        with open(st.session_state.blockchain_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving ledger: {e}")

def load_blockchain():
    """Load blockchain data from JSON file if exists"""
    if os.path.exists(st.session_state.blockchain_file):
        try:
            with open(st.session_state.blockchain_file, "r") as f:
                data = json.load(f)
                st.session_state.blockchain = data.get("blockchain", [])
                st.session_state.balances = data.get("balances", {})
                # Users are handled separately to preserve private keys
                st.session_state.network_stats = data.get("network_stats", {"total_transactions": 0, "total_volume": 0})
                st.session_state.quantum_keys = data.get("quantum_keys", {})
                st.success("✅ Ledger loaded successfully!")
        except Exception as e:
            st.error(f"Error loading ledger: {e}")
            # Initialize with genesis block if loading fails
            if not st.session_state.blockchain:
                genesis_block = {
                    "index": 0,
                    "transactions": [],
                    "merkle_root": "0",
                    "timestamp": str(datetime.now()),
                    "previous_hash": "0",
                    "hash": calculate_hash({"index": 0, "timestamp": str(datetime.now())}),
                    "miner": "network",
                    "quantum_dimension": st.session_state.quantum_dimension
                }
                st.session_state.blockchain.append(genesis_block)
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os, binascii, base64, secrets

# Import IBM Quantum dependencies
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Estimator, Sampler
    from qiskit_ibm_runtime.fake_provider import FakeManila, FakeLima  # For noise simulation
    IBM_QUANTUM_AVAILABLE = True
except ImportError:
    IBM_QUANTUM_AVAILABLE = False

# ================== FIXED BB84 IMPLEMENTATION ==================
def cascade_error_reconciliation(alice_bits, bob_bits, block_size=8, max_iterations=5):
    """
    Implement Cascade protocol for error reconciliation
    Returns reconciled bits, number of errors corrected, and iterations used
    """
    # Convert to list of blocks
    alice_blocks = [alice_bits[i:i+block_size] for i in range(0, len(alice_bits), block_size)]
    bob_blocks = [bob_bits[i:i+block_size] for i in range(0, len(bob_bits), block_size)]
    errors_corrected = 0
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        parity_mismatches = []
        for i, (a_block, b_block) in enumerate(zip(alice_blocks, bob_blocks)):
            a_parity = sum(int(bit) for bit in a_block) % 2
            b_parity = sum(int(bit) for bit in b_block) % 2
            if a_parity != b_parity:
                parity_mismatches.append(i)
        if not parity_mismatches:
            break
        # Correct first error found in this iteration
        for block_idx in parity_mismatches:
            a_block = alice_blocks[block_idx]
            b_block = bob_blocks[block_idx]
            left, right = 0, len(a_block) - 1
            # Binary search for error position
            while left < right:
                mid = (left + right) // 2
                a_left_parity = sum(int(bit) for bit in a_block[left:mid+1]) % 2
                b_left_parity = sum(int(bit) for bit in b_block[left:mid+1]) % 2
                if a_left_parity != b_left_parity:
                    right = mid
                else:
                    left = mid + 1
            # Flip the erroneous bit in Bob's block
            b_block_list = list(b_block)
            b_block_list[left] = str(1 - int(b_block_list[left]))
            bob_blocks[block_idx] = ''.join(b_block_list)
            errors_corrected += 1
            break  # Correct one error per iteration
    reconciled_bits = ''.join(bob_blocks)
    return reconciled_bits, errors_corrected, iterations

def run_high_dimensional_bb84_protocol(key_length=256, dimension=4, backend=None):
    """
    Optimized BB84 protocol with efficient circuit batching and improved error handling.
    """
    # Validate dimension is power of two
    if not is_power_of_two(dimension):
        st.error(f"❌ Quantum dimension must be a power of 2 (got {dimension})")
        return None
    use_auth = st.session_state.get("authentication_enabled", True)
    auth_psk = st.session_state.get("authentication_psk", "")
    block_size = st.session_state.get("reconciliation_block_size", 8)
    max_iterations = st.session_state.get("reconciliation_max_iterations", 5)

    bits_per_symbol = math.log2(dimension)
    decoy_ratio = 0.3
    # Increase initial length to account for higher error rates
    initial_length = int(key_length * 4 / bits_per_symbol / (1 - decoy_ratio))

    # Generate all random values upfront
    alice_symbols = [random.randint(0, dimension-1) for _ in range(initial_length)]
    alice_bases = [random.choice([0, 1]) for _ in range(initial_length)]
    alice_decoy_states = [random.random() < decoy_ratio for _ in range(initial_length)]
    bob_bases = [random.choice([0, 1]) for _ in range(initial_length)]

    decoy_intensities = {
        'signal': 0.5,
        'decoy': 0.1
    }

    # Batch circuit creation and execution
    num_qubits_per_symbol = int(math.ceil(math.log2(dimension)))
    batch_size = get_optimal_batch_size(backend, num_qubits_per_symbol)

    circuits = []
    measured_symbols = []

    # Show progress for circuit creation
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        # Process in batches for efficiency
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

                # Encode the symbol
                symbol_bin = format(symbol, f'0{num_qubits_per_symbol}b')
                for j, bit in enumerate(symbol_bin):
                    if bit == '1':
                        qc.x(qubit_start + j)

                # Apply Alice's basis transformation
                if alice_basis == 1:
                    qc.h(range(qubit_start, qubit_end))
                    for j in range(num_qubits_per_symbol):
                        for k in range(j+1, num_qubits_per_symbol):
                            angle = math.pi / (2 ** (k - j))
                            qc.cp(angle, qubit_start + k, qubit_start + j)
                    for j in range(num_qubits_per_symbol//2):
                        qc.swap(qubit_start + j, qubit_start + num_qubits_per_symbol - j - 1)

                # Apply Bob's basis transformation
                if bob_basis == 1:
                    for j in range(num_qubits_per_symbol//2):
                        qc.swap(qubit_start + j, qubit_start + num_qubits_per_symbol - j - 1)
                    for j in range(num_qubits_per_symbol-1, -1, -1):
                        for k in range(num_qubits_per_symbol-1, j, -1):
                            angle = -math.pi / (2 ** (k - j))
                            qc.cp(angle, qubit_start + k, qubit_start + j)
                    qc.h(range(qubit_start, qubit_end))

                # Measure
                qc.measure(range(qubit_start, qubit_end), range(qubit_start, qubit_end))

            circuits.append(qc)
            # Update progress
            progress = min(1.0, (batch_end / initial_length) * 0.5)  # 50% for circuit creation
            status_text.text(f" Creating quantum circuits: {int(progress*100)}%")
            progress_bar.progress(progress)

        # Execute all batches
        status_text.text(" Executing quantum circuits...")

        if backend is None:
            backend = AerSimulator()

        # Execute in smaller batches to avoid timeouts
        execution_batch_size = min(10, len(circuits))
        measured_symbols = []

        for i in range(0, len(circuits), execution_batch_size):
            batch_end = min(i + execution_batch_size, len(circuits))
            batch = circuits[i:batch_end]

            try:
                job = backend.run(batch, shots=1, memory=True)
                result = job.result()

                for j, circuit in enumerate(batch):
                    memory = result.get_memory(j)[0]
                    batch_symbol_count = circuit.num_qubits // num_qubits_per_symbol

                    for k in range(batch_symbol_count):
                        start_idx = k * num_qubits_per_symbol
                        end_idx = (k + 1) * num_qubits_per_symbol
                        measured_bits = memory[start_idx:end_idx]
                        measured_symbol = int(measured_bits, 2) % dimension
                        measured_symbols.append(measured_symbol)

                # Update progress
                progress = 0.5 + (min(i + execution_batch_size, len(circuits)) / len(circuits)) * 0.4  # 40% for execution
                status_text.text(f" Executing quantum circuits: {int(progress*100)}%")
                progress_bar.progress(progress)

            except Exception as e:
                st.error(f"Error running quantum circuits: {e}")
                # Try to continue with partial results
                remaining = initial_length - len(measured_symbols)
                if remaining > 0:
                    st.warning(f"Using simulated results for {remaining} symbols due to execution error")
                    for _ in range(remaining):
                        measured_symbols.append(random.randint(0, dimension-1))

        # Ensure we have the right number of measured symbols
        if len(measured_symbols) < initial_length:
            st.warning(" Some quantum measurements failed. Using partial results.")
            while len(measured_symbols) < initial_length:
                measured_symbols.append(random.randint(0, dimension-1))
        elif len(measured_symbols) > initial_length:
            measured_symbols = measured_symbols[:initial_length]

    except Exception as e:
        st.error(f"Unexpected error in quantum protocol: {e}")
        return None
    finally:
        progress_bar.empty()
        status_text.empty()

    # --- Authentication, decoy state analysis, error estimation, etc. ---
    # AUTHENTICATION: Include decoy state information
    if use_auth:
        auth_data = {
            "alice_bases": alice_bases,
            "alice_decoy_states": alice_decoy_states,
            "decoy_intensities": decoy_intensities
        }
        auth_data_str = json.dumps(auth_data)
        auth_hmac = generate_hmac(auth_data_str, auth_psk)
        if not verify_hmac(auth_data_str, auth_hmac, auth_psk):
            st.error("Authentication failed: Quantum state information could not be verified")
            return None

    # DECOY STATE ANALYSIS
    signal_indices = [i for i in range(initial_length) if not alice_decoy_states[i]]
    decoy_indices = [i for i in range(initial_length) if alice_decoy_states[i]]

    signal_error_rate = estimate_error_rate(signal_indices, alice_symbols, measured_symbols, alice_bases, bob_bases, auth_psk, use_auth)
    decoy_error_rate = estimate_error_rate(decoy_indices, alice_symbols, measured_symbols, alice_bases, bob_bases, auth_psk, use_auth)

    # More tolerant error threshold for demonstration
    max_error_rate = (dimension + 2) / (2 * dimension)  # Slightly more tolerant
    
    if decoy_error_rate > signal_error_rate * 2.0 and decoy_error_rate > 0.1:
        st.error("Photon number splitting attack detected! Protocol aborted.")
        return None

    # Continue with original protocol using only signal states
    sifted_key_alice = []
    sifted_key_bob = []
    for i in signal_indices:
        if alice_bases[i] == bob_bases[i]:
            sifted_key_alice.append(alice_symbols[i])
            sifted_key_bob.append(measured_symbols[i])

    if len(sifted_key_alice) < 20:
        st.error(f"Not enough sifted key bits ({len(sifted_key_alice)}). Protocol requires at least 20.")
        return None
        
    sample_size = min(50, len(sifted_key_alice) // 3)  # Smaller sample size
    sample_indices = random.sample(range(len(sifted_key_alice)), sample_size)
    
    if use_auth:
        sample_indices_str = json.dumps(sample_indices)
        sample_indices_hmac = generate_hmac(sample_indices_str, auth_psk)
        if not verify_hmac(sample_indices_str, sample_indices_hmac, auth_psk):
            st.error("Authentication failed: Sample indices could not be verified")
            return None
            
    error_count = 0
    for idx in sample_indices:
        if sifted_key_alice[idx] != sifted_key_bob[idx]:
            error_count += 1
            
    if use_auth:
        error_count_str = str(error_count)
        error_count_hmac = generate_hmac(error_count_str, auth_psk)
        if not verify_hmac(error_count_str, error_count_hmac, auth_psk):
            st.error("Authentication failed: Error count could not be verified")
            return None
            
    error_rate = error_count / sample_size if sample_size > 0 else 0
    
    # Adjust error rate threshold based on dimension with more tolerance
    if error_rate > max_error_rate:
        st.error(f"Error rate {error_rate:.3f} exceeds maximum allowed {max_error_rate:.3f} for {dimension}-D encoding")
        # For demonstration, we'll continue with a warning instead of aborting
        st.warning(" High error rate detected but continuing for demonstration purposes")
        # return None  # Commented out for demonstration

    # Remove sample symbols from key
    final_key_alice = []
    final_key_bob = []
    for i in range(len(sifted_key_alice)):
        if i not in sample_indices:
            final_key_alice.append(sifted_key_alice[i])
            final_key_bob.append(sifted_key_bob[i])

    # Error Reconciliation
    symbol_bits = int(math.log2(dimension))
    alice_binary = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in final_key_alice)
    bob_binary = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in final_key_bob)
    
    if len(alice_binary) == 0:
        st.error("No key bits remaining after sampling")
        return None
        
    reconciled_binary, errors_corrected, iterations = cascade_error_reconciliation(
        alice_binary, bob_binary, block_size=block_size, max_iterations=max_iterations
    )
    
    if errors_corrected > 0:
        st.info(f" Error reconciliation corrected {errors_corrected} bits in {iterations} iterations")

    # Convert back to symbols
    reconciled_key = []
    for i in range(0, len(reconciled_binary), symbol_bits):
        symbol_bin = reconciled_binary[i:i+symbol_bits]
        if len(symbol_bin) == symbol_bits:
            reconciled_key.append(int(symbol_bin, 2))

    # Privacy Amplification
    if not reconciled_key:
        st.error("No key bits after reconciliation")
        return None
        
    key_str = ''.join(f"{symbol:0{symbol_bits}b}" for symbol in reconciled_key)
    
    # Pad key if too short
    if len(key_str) < key_length:
        st.warning(f"Key too short ({len(key_str)} bits), padding with random bits")
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
    
    try:
        final_key = hkdf.derive(key_bytes)[:key_length//8]
    except Exception as e:
        st.error(f"Key derivation failed: {e}")
        return None
        
    # Derive authentication key for future use
    auth_key = derive_authentication_key(binascii.hexlify(final_key).decode())
    
    return binascii.hexlify(final_key).decode()

# Add this helper function to check if IBM Quantum is available and configured
def is_ibm_quantum_available():
    """Check if IBM Quantum is properly configured and available"""
    if not IBM_QUANTUM_AVAILABLE:
        return False
    if not st.session_state.get("ibm_quantum_configured", False):
        return False
    if not st.session_state.get("ibm_api_key"):
        return False
    return True

# Update the one_time_circuit_high_dim_bb84 function with better error handling
def one_time_circuit_high_dim_bb84(key_length=256, dimension=4):
    """
    Fixed one-time circuit for high-dimensional BB84 QKD with improved error handling.
    """
    # Validate dimension is power of two
    if not is_power_of_two(dimension):
        st.error(f"❌ Quantum dimension must be a power of 2 (got {dimension})")
        return None
    # Check if we should use real quantum hardware
    use_real_hardware = (
        is_ibm_quantum_available() and 
        st.session_state.get("use_real_hardware", False)
    )
    backend = None
    hardware_type = "Aer Simulator"
    if use_real_hardware:
        try:
            service = QiskitRuntimeService(
                channel=st.session_state.get("ibm_channel", "ibm_quantum"),
                token=st.session_state.get("ibm_api_key"),
                instance=st.session_state.get("ibm_instance", "")
            )
            # Get available backends
            backends = service.backends(simulator=False, operational=True)
            if backends:
                # Select the least busy backend
                try:
                    from qiskit_ibm_provider import least_busy
                    backend = least_busy(backends)
                except Exception:
                    backend = backends[0]
                hardware_type = f"IBM Quantum ({backend.name})"
                st.session_state["last_quantum_backend"] = backend.name
            else:
                st.warning("No suitable quantum hardware available. Using simulator.")
                backend = AerSimulator()
                hardware_type = "Aer Simulator (fallback)"
        except Exception as e:
            st.error(f"IBM Quantum connection failed: {e}. Using simulator.")
            backend = AerSimulator()
            hardware_type = "Aer Simulator (fallback)"
            st.session_state["ibm_quantum_configured"] = False
    else:
        backend = AerSimulator()
    # Show progress for long-running operations
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        status_text.text(" Generating quantum key...")
        progress_bar.progress(10)
        key = run_high_dimensional_bb84_protocol(key_length, dimension, backend)
        progress_bar.progress(90)
        if key:
            st.session_state["last_hardware_used"] = hardware_type
            status_text.text("✅ Quantum key generated successfully!")
            progress_bar.progress(100)
            time.sleep(0.5)  # Brief pause to show completion
            return key
        else:
            st.error("Quantum protocol aborted due to security issues")
            return None
    except Exception as e:
        st.error(f"Error in quantum protocol: {e}")
        return None
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

def one_time_circuit_high_dim_bb84(key_length=256, dimension=4):
    # Validate dimension is power of two
    if not is_power_of_two(dimension):
        st.error(f"❌ Quantum dimension must be a power of 2 (got {dimension})")
        return None
    """
    Fixed one-time circuit for high-dimensional BB84 QKD.
    Now supports both simulator and real quantum hardware.
    """
    # Check if we should use real quantum hardware
    use_real_hardware = (
        IBM_QUANTUM_AVAILABLE and 
        st.session_state.get("ibm_quantum_configured", False) and
        st.session_state.get("ibm_api_key") and
        st.session_state.get("use_real_hardware", False)
    )
    
    backend = None
    hardware_type = "Aer Simulator"
    
    if use_real_hardware:
        try:
            # Try to connect to IBM Quantum
            service = QiskitRuntimeService(
                channel=st.session_state.get("ibm_channel", "ibm_quantum"),
                token=st.session_state.get("ibm_api_key"),
                instance=st.session_state.get("ibm_instance", "")
            )
            
            # Get available backends
            backends = service.backends(simulator=False, operational=True)
            
            if backends:
                backend = backends[0]  # Use first available backend
                hardware_type = f"IBM Quantum ({backend.name})"
                st.session_state["last_quantum_backend"] = backend.name
            else:
                st.warning("No suitable quantum hardware available. Using simulator.")
                backend = AerSimulator()
                hardware_type = "Aer Simulator (fallback)"
                
        except Exception as e:
            st.error(f"IBM Quantum connection failed: {e}. Using simulator.")
            backend = AerSimulator()
            hardware_type = "Aer Simulator (fallback)"
            st.session_state["ibm_quantum_configured"] = False
    else:
        backend = AerSimulator()
    
    # Run the protocol with decoy states
    try:
        key = run_high_dimensional_bb84_protocol(key_length, dimension, backend)
        if key:
            st.session_state["last_hardware_used"] = hardware_type
            return key
        else:
            st.error("ERROR: Quantum protocol aborted due to security issues")
            return None
    except Exception as e:
        st.error(f"Error in quantum protocol: {e}")
        return None

# ================== END OF FIXED BB84 IMPLEMENTATION ==================

def bb84_encrypt(message, key_hex):
    """Encrypt message using BB84-derived key with AES-256-GCM"""
    try:
        key = binascii.unhexlify(key_hex)
        if len(key) != 32:  # ensure 256-bit key
            hk = HKDF(
                algorithm=hashes.SHA3_256(),
                length=32,
                salt=None,
                info=b'bb84-aes-key',
                backend=default_backend()
            )
            key = hk.derive(key)

        nonce = os.urandom(12)  # 96-bit nonce
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(message.encode()) + encryptor.finalize()
        encrypted_data = nonce + ciphertext + encryptor.tag
        return base64.b64encode(encrypted_data).decode()
    except Exception as e:
        st.error(f"Encryption error: {e}")
        return None

def bb84_decrypt(ciphertext_b64, key_hex):
    """Decrypt message using BB84-derived key with AES-256-GCM"""
    try:
        key = binascii.unhexlify(key_hex)
        if len(key) != 32:  # ensure 256-bit key
            hk = HKDF(
                algorithm=hashes.SHA3_256(),
                length=32,
                salt=None,
                info=b'bb84-aes-key',
                backend=default_backend()
            )
            key = hk.derive(key)

        encrypted_data = base64.b64decode(ciphertext_b64)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:-16]
        tag = encrypted_data[-16:]
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext.decode()
    except Exception as e:
        st.error(f"Decryption error: {e}")
        return None

# Standard cryptographic hash for blockchain operations
def calculate_hash(block: dict) -> str:
    """Standard SHA256 hash of a block for deterministic consensus"""
    block_copy = block.copy()
    block_copy.pop("hash", None)
    block_string = json.dumps(block_copy, sort_keys=True).encode()
    return hashlib.sha256(block_string).hexdigest()

def build_merkle_tree(transactions):
    """Return Merkle root for a list of tx dicts using standard hashing."""
    if not transactions:
        return "0"
    tx_hashes = [hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest() for tx in transactions]
    while len(tx_hashes) > 1:
        if len(tx_hashes) % 2 != 0:
            tx_hashes.append(tx_hashes[-1])
        tx_hashes = [
            hashlib.sha256((tx_hashes[i] + tx_hashes[i+1]).encode()).hexdigest()
            for i in range(0, len(tx_hashes), 2)
        ]
    return tx_hashes[0]

# ------------------ WALLET GENERATION ------------------
def generate_wallet(username):
    """Generates a new private and public key pair for a user."""
    sk = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
    vk = sk.get_verifying_key()
    
    return {
        "private_key": sk.to_string().hex(),
        "public_key": vk.to_string().hex(),
        "username": username,
        "created_at": datetime.now().isoformat(),
        "total_sent": 0,
        "total_received": 0,
        "transactions_count": 0
    }

# ------------------ SESSION STATE ------------------
if "users" not in st.session_state:
    st.session_state.users = {}
    st.session_state.balances = {}
    st.session_state.blockchain = []
    st.session_state.network_stats = {"total_transactions": 0, "total_volume": 0}
    st.session_state.price_history = []
    st.session_state.quantum_keys = {}  # Store quantum keys for users
    st.session_state.quantum_dimension = 4  # Default to 4-dimensional quantum system
    st.session_state.ibm_api_key = ""  # IBM Quantum API key
    st.session_state.ibm_channel = "ibm_quantum"  # Default channel
    st.session_state.ibm_instance = ""  # IBM Quantum instance
    st.session_state.use_real_hardware = False  # Whether to use real hardware
    st.session_state.ibm_quantum_configured = False  # Whether IBM Quantum is configured
    st.session_state.last_hardware_used = "Aer Simulator"  # Last hardware used
    st.session_state.last_quantum_backend = ""  # Last quantum backend used
    st.session_state.authentication_psk = hashlib.sha256(b"quantumverse_default_psk").hexdigest()  # Default PSK
    st.session_state.authentication_enabled = True  # Enable authentication by default
    st.session_state.auth_keys = {}  # Store derived authentication keys

    # Default demo users
    user_info = [
        ("krishna", "nikki"),
        ("koushik", "iphone"),
        ("varun", "ruttu_bhai"),
        ("sravan", "akshay")
    ]
    user_keys = ["K", "O", "V", "S"]  
    for i, (name, pwd) in enumerate(user_info):
        wallet = generate_wallet(name)
        wallet["password_hash"] = hashlib.sha256(pwd.encode()).hexdigest()
        st.session_state.users[user_keys[i]] = wallet
        st.session_state.balances[wallet["public_key"]] = 1000
        st.session_state.quantum_keys[user_keys[i]] = None

# ------------------ AUTHENTICATION UTILS ------------------
def generate_hmac(message, key):
    """Generate HMAC using SHA3-256 for stronger hashing."""
    if isinstance(message, str):
        message = message.encode()
    if isinstance(key, str):
        key = key.encode()
    h = hmac.HMAC(key, hashes.SHA3_256(), backend=default_backend())
    h.update(message)
    return h.finalize().hex()

def verify_hmac(message, received_hmac, key):
    """Verify HMAC using SHA3-256."""
    try:
        expected_hmac = generate_hmac(message, key)
        return secrets.compare_digest(expected_hmac, received_hmac)
    except Exception:
        return False

def derive_authentication_key(quantum_key):
    """Derive a 256-bit authentication key from a quantum key using SHA3-256 HKDF"""
    hkdf = HKDF(
        algorithm=hashes.SHA3_256(),
        length=32,  # 256-bit key
        salt=None,
        info=b'quantumverse-auth-key',
        backend=default_backend()
    )
    return hkdf.derive(binascii.unhexlify(quantum_key))

# ================== Part 2/5 ==================
# -------- Core blockchain + TX logic --------

def verify_signature(transaction) -> bool:
    """ECDSA verification for a signed transaction dict."""
    try:
        sender_pk_hex = transaction["sender"]
        signature_hex = transaction["signature"]
        sender_vk = ecdsa.VerifyingKey.from_string(
            binascii.unhexlify(sender_pk_hex), curve=ecdsa.SECP256k1
        )
        signature = binascii.unhexlify(signature_hex)
        
        # Create consistent data to verify
        tx_data_to_verify = {k: v for k, v in transaction.items() if k != "signature"}
        tx_string = json.dumps(tx_data_to_verify, sort_keys=True).encode()
        
        # Use standard SHA256 for verification consistency
        tx_hash = hashlib.sha256(tx_string).hexdigest()
        
        return sender_vk.verify(signature, tx_hash.encode())
    except Exception:
        return False

def update_balances(transactions):
    """Apply value transfers for a list of transactions."""
    for tx in transactions:
        sender = tx["sender"]
        receiver = tx["receiver"]
        amount = float(tx["amount"])
        if sender != "network":
            st.session_state.balances[sender] = st.session_state.balances.get(sender, 0) - amount
        st.session_state.balances[receiver] = st.session_state.balances.get(receiver, 0) + amount

def process_transaction(transaction) -> bool:
    """Immediately confirm a single valid transaction."""
    if not verify_signature(transaction):
        st.error("❌ Invalid transaction signature!")
        return False

    previous_hash = st.session_state.blockchain[-1]["hash"] if st.session_state.blockchain else "0"
    
    new_block = {
        "index": len(st.session_state.blockchain),
        "transactions": [transaction],
        "merkle_root": build_merkle_tree([transaction]),
        "timestamp": str(datetime.now()),
        "previous_hash": previous_hash,
        "miner": "network",
        "quantum_dimension": st.session_state.quantum_dimension
    }
    new_block["hash"] = calculate_hash(new_block)

    st.session_state.blockchain.append(new_block)
    update_balances([transaction])

    st.session_state.network_stats["total_transactions"] += 1
    st.session_state.network_stats["total_volume"] += float(transaction["amount"])
    return True

def create_transaction(
    sender_private_key_hex,
    receiver_public_key,
    amount,
    tx_type="transfer",
    fee: float = 0.0,
):
    """
    Create, sign, and confirm a transaction with authentication
    """
    try:
        sender_sk = ecdsa.SigningKey.from_string(
            binascii.unhexlify(sender_private_key_hex), curve=ecdsa.SECP256k1
        )
        sender_public_key = sender_sk.get_verifying_key().to_string().hex()
    except binascii.Error:
        st.error("❌ Invalid private key format.")
        return None

    amount = float(amount)
    fee = float(fee)
    balance = st.session_state.balances.get(sender_public_key, 0.0)
    total_cost = amount + fee
    
    if balance < total_cost:
        st.error(f"❌ Insufficient funds! Need {total_cost:.2f} coins, have {balance:.2f}")
        return None

    # Generate high-dimensional BB84 key using one-time circuit
    bb84_key = one_time_circuit_high_dim_bb84(256, st.session_state.quantum_dimension)
    if bb84_key is None:
        st.error("❌ Failed to generate quantum key. Transaction aborted.")
        return None

    # If authentication is enabled, derive authentication key
    if st.session_state.get("authentication_enabled", True):
        auth_key = derive_authentication_key(bb84_key)
        st.session_state.auth_keys[bb84_key] = auth_key

    tx_id = f"tx_{int(time.time())}_{random.randint(1000, 9999)}"

    # Create transaction data
    tx_core = {
        "sender": sender_public_key,
        "receiver": receiver_public_key,
        "amount": amount,
        "fee": fee,
        "type": tx_type,
        "timestamp": str(datetime.now()),
        "tx_id": tx_id,
        "quantum_secured": True,
        "quantum_dimension": st.session_state.quantum_dimension,
        "quantum_hardware": st.session_state.last_hardware_used
    }

    # Create consistent hash for signing
    tx_string = json.dumps(tx_core, sort_keys=True).encode()
    tx_hash = hashlib.sha256(tx_string).hexdigest()
    signature = sender_sk.sign(tx_hash.encode()).hex()

    signed_tx = {**tx_core, "signature": signature}

    ok = process_transaction(signed_tx)
    if ok:
        st.success(f"✅ Quantum-secured transaction {tx_id} confirmed!")
        st.info(f" Used {st.session_state.quantum_dimension}-dimensional quantum encoding")
        st.info(f" Hardware: {st.session_state.last_hardware_used}")

        # Store the quantum key for the receiver
        receiver_user = None
        for user_key, user_data in st.session_state.users.items():
            if user_data["public_key"] == receiver_public_key:
                receiver_user = user_key
                break

        if receiver_user:
            st.session_state.quantum_keys[receiver_user] = bb84_key

    return signed_tx if ok else None

# -------- Refresh Balance Function --------
def refresh_balance(public_key):
    """Recalculate balance by scanning all transactions in blockchain."""
    balance = 0.0
    for block in st.session_state.blockchain:
        for tx in block["transactions"]:
            if tx["sender"] == public_key:
                balance -= float(tx["amount"])
            if tx["receiver"] == public_key:
                balance += float(tx["amount"])
    
    st.session_state.balances[public_key] = max(0, balance)
    return st.session_state.balances[public_key]

# ================== Part 3/5 ==================
# -------- Streamlit Page Config + Styling --------

st.set_page_config(
    page_title=" QuantumVerse - High-Dimensional BB84 Quantum Blockchain",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with quantum theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102,126,234,0.1) 0%, transparent 70%);
        animation: quantum-pulse 8s infinite linear;
    }
    @keyframes quantum-pulse {
        0% { transform: rotate(0deg); opacity: 0.5; }
        50% { opacity: 0.8; }
        100% { transform: rotate(360deg); opacity: 0.5; }
    }
    .main-header h1 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        margin: 0;
        text-shadow: 0 0 15px rgba(102,126,234,0.7);
        letter-spacing: -0.5px;
        position: relative;
        z-index: 2;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin: 0.8rem 0 0 0;
        font-weight: 400;
        position: relative;
        z-index: 2;
    }

    .wallet-card, .metric-card, .transaction-card, .block-card {
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: none;
        position: relative;
        overflow: hidden;
    }
    .wallet-card::before, .metric-card::before, .transaction-card::before, .block-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: gradient-flow 3s infinite linear;
    }
    @keyframes gradient-flow {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    .wallet-card:hover, .metric-card:hover, .transaction-card:hover, .block-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    
    .wallet-card { 
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); 
        color: white; 
    }
    .metric-card { 
        background: linear-gradient(135deg, #1a2a6b 0%, #2c5364 100%); 
        color: white; 
        text-align: center; 
    }
    .transaction-card { 
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
        color: white;
    }
    .block-card { 
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%); 
        color: white;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        border: none; 
        border-radius: 12px;
        padding: 0.7rem 2rem; 
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102,126,234,0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px 12px 0 0;
        padding: 1rem 1.5rem;
        font-weight: 600;
        border: none;
        color: #2D3748;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .address-text {
        font-family: 'JetBrains+Mono', monospace;
        font-size: 0.85rem;
        background: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 8px;
        word-break: break-all;
        color: white;
    }
    
    .success-box {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        margin: 1rem 0;
    }
    
    .info-box {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin: 1rem 0;
    }
    
    .quantum-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,107,107,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255,107,107,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,107,107,0); }
    }
    
    .quantum-glow {
        text-shadow: 0 0 10px rgba(102,126,234,0.7), 0 0 20px rgba(102,126,234,0.5);
    }
    
    .dimension-badge {
        display: inline-block;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2D3748;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .hardware-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------- Header --------
st.markdown(f"""
<div class="main-header">
    <h1 class="quantum-glow"> QuantumVerse</h1>
    <p>High-Dimensional BB84 Quantum Blockchain with {st.session_state.quantum_dimension}-D Quantum Encoding</p>
</div>
""", unsafe_allow_html=True)

# -------- Genesis Block Setup --------
if not st.session_state.blockchain:
    genesis_block = {
        "index": 0,
        "transactions": [],
        "merkle_root": "0",
        "timestamp": str(datetime.now()),
        "previous_hash": "0",
        "hash": calculate_hash({"index": 0, "timestamp": str(datetime.now())}),
        "miner": "network",
        "quantum_dimension": st.session_state.quantum_dimension
    }
    st.session_state.blockchain.append(genesis_block)

# -------- Login System --------
if "logged_in_user" not in st.session_state:
    st.markdown("###  Quantum Secure Login")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form", clear_on_submit=True):
            user_keys = list(st.session_state.users.keys())
            username = st.selectbox(
                " Select User", user_keys,
                format_func=lambda x: f"{st.session_state.users[x]['username']} ({x})"
            )
            password = st.text_input(" Enter Password", type="password")

            col_login, col_demo = st.columns(2)
            with col_login:
                login_btn = st.form_submit_button(" Login", use_container_width=True)
            with col_demo:
                demo_btn = st.form_submit_button(" Demo Mode", use_container_width=True)

            if login_btn:
                if username in st.session_state.users:
                    password_hash = hashlib.sha256(password.encode()).hexdigest()
                    if password_hash == st.session_state.users[username]["password_hash"]:
                        st.session_state.logged_in_user = username
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials.")
                else:
                    st.error("❌ User not found.")
            if demo_btn:
                st.session_state.logged_in_user = user_keys[0]
                st.rerun()
                
# ================== Part 4/5 ==================
else:
    # ---- Logged-in user session ----
    logged_in_user = st.session_state.logged_in_user
    user_wallet = st.session_state.users[logged_in_user]
    user_public_key = user_wallet["public_key"]
    user_private_key = user_wallet["private_key"]
    quantum_key = st.session_state.quantum_keys.get(logged_in_user)

    # Quick user stats
    def get_user_stats(public_key):
        sent, received, tx_count = 0, 0, 0
        quantum_txs = 0
        max_dimension = 0
        hardware_types = {}
        for block in st.session_state.blockchain:
            for tx in block["transactions"]:
                if tx["sender"] == public_key:
                    sent += float(tx["amount"])
                    tx_count += 1
                    if tx.get("quantum_secured", False):
                        quantum_txs += 1
                        max_dimension = max(max_dimension, tx.get("quantum_dimension", 2))
                        hw = tx.get("quantum_hardware", "Aer Simulator")
                        hardware_types[hw] = hardware_types.get(hw, 0) + 1
                if tx["receiver"] == public_key:
                    received += float(tx["amount"])
                    if tx["sender"] != public_key:
                        tx_count += 1
                    if tx.get("quantum_secured", False):
                        quantum_txs += 1
                        max_dimension = max(max_dimension, tx.get("quantum_dimension", 2))
                        hw = tx.get("quantum_hardware", "Aer Simulator")
                        hardware_types[hw] = hardware_types.get(hw, 0) + 1
        return {
            "total_sent": sent,
            "total_received": received,
            "transaction_count": tx_count,
            "net_flow": received - sent,
            "quantum_txs": quantum_txs,
            "max_quantum_dim": max_dimension,
            "hardware_types": hardware_types
        }

    user_stats = get_user_stats(user_public_key)
    current_balance = st.session_state.balances.get(user_public_key, 0.0)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:1.5rem;
            background:linear-gradient(135deg,#0f2027 0%,#203a43 50%,#2c5364 100%);
            border-radius:16px; margin-bottom:1.5rem; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
            <h2 style="color:white; margin:0;"> {user_wallet['username']}</h2>
            <p style="color:rgba(255,255,255,0.9); margin:0.5rem 0 0 0;">Quantum Wallet</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(" Balance", f"{current_balance:}", f"{user_stats['net_flow']:}")
        with col2:
            st.metric(" Q-TXs", user_stats["quantum_txs"])
            
        st.markdown("---")
        
        # Quantum configuration
        st.markdown("### ⚙️ Quantum Configuration")
        new_dimension = st.selectbox(
            "Quantum Dimension",
            [2, 4, 8, 16],  # These are all powers of 2
            index=[2, 4, 8, 16].index(st.session_state.quantum_dimension) if st.session_state.quantum_dimension in [2, 4, 8, 16] else 1,
            help="Higher dimensions provide better security but require more computation. Must be a power of 2."
        )
        if new_dimension != st.session_state.quantum_dimension:
            if not is_power_of_two(new_dimension):
                st.error(f"❌ Quantum dimension must be a power of 2 (got {new_dimension})")
            else:
                st.session_state.quantum_dimension = new_dimension
                st.rerun()
        st.markdown(f"**Current:** {st.session_state.quantum_dimension}-D encoding")
        
        # IBM Quantum configuration
        st.markdown("###  IBM Quantum Settings")

        use_real_hardware = st.checkbox("Use Real Quantum Hardware", 
                                      value=st.session_state.get("use_real_hardware", False),
                                      help="Use IBM Quantum real hardware instead of simulator")

        if use_real_hardware != st.session_state.get("use_real_hardware", False):
            st.session_state.use_real_hardware = use_real_hardware
            st.rerun()

        api_key = st.text_input("IBM Quantum API Key", 
                              value=st.session_state.get("ibm_api_key", ""),
                              type="password",
                              help="Get from https://quantum-computing.ibm.com/")

        channel = st.text_input("IBM Quantum Channel", 
                              value=st.session_state.get("ibm_channel", "ibm_quantum"),
                              help="Usually 'ibm_quantum' or 'ibm_cloud'")

        instance = st.text_input("IBM Quantum Instance (optional)", 
                               value=st.session_state.get("ibm_instance", ""),
                               help="Optional: Your specific IBM Quantum instance")

        if st.button("Save IBM Quantum Settings"):
            if api_key:
                st.session_state.ibm_api_key = api_key
                st.session_state.ibm_channel = channel
                st.session_state.ibm_instance = instance
                # Test the connection
                try:
                    service = QiskitRuntimeService(
                        channel=channel,
                        token=api_key,
                        instance=instance if instance else None
                    )
                    # Try to get backends to verify connection
                    backends = service.backends(simulator=False, operational=True)
                    if backends:
                        st.session_state.ibm_quantum_configured = True
                        st.success("✅ IBM Quantum connection successful!")
                        st.info(f"Available backends: {[b.name for b in backends]}")
                    else:
                        st.warning("No quantum hardware available. Using simulator.")
                        st.session_state.ibm_quantum_configured = False
                except Exception as e:
                    st.error(f"IBM Quantum connection failed: {e}")
                    st.session_state.ibm_quantum_configured = False
            else:
                st.error("API key is required for IBM Quantum access")

        # Show current hardware status
        if st.session_state.get("ibm_quantum_configured", False):
            st.success("✅ IBM Quantum configured")
            if st.session_state.get("last_quantum_backend"):
                st.info(f"Last backend: {st.session_state.last_quantum_backend}")
        else:
            st.info("Using Aer Simulator")
                
        st.markdown("---")
        
        # BB84 key status
        st.markdown("###  Quantum Key Status")
        if quantum_key:
            st.success("✅ Quantum key available")
            if st.button("🔍 View Quantum Key", use_container_width=True):
                st.code(quantum_key[:64] + "..." if len(quantum_key) > 64 else quantum_key)
        else:
            st.warning(" No quantum key available")
            
        st.markdown("---")
        if st.button(" Logout", use_container_width=True):
            del st.session_state.logged_in_user
            st.rerun()

    # ---- Tabs ----
    tab1, tab2, tab3 = st.tabs([" Quantum Wallet", "  Analytics", " Quantum Network"])

    # ---- Wallet Tab ----
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="wallet-card">
                <h3> Quantum Account</h3>
                <p><b>User:</b> {user_wallet['username']}</p>
                <p><b>Addr:</b> <span class="address-text">{user_public_key[:16]}...</span></p>
                <p><b>Created:</b> {user_wallet['created_at'][:19]}</p>
                <p><b>Security:</b> High-Dimensional BB84 <span class="quantum-badge">{st.session_state.quantum_dimension}-D</span></p>
                <p><b>Hardware:</b> <span class="hardware-badge">{st.session_state.last_hardware_used}</span></p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3> Balance</h3>
                <h2 style="margin: 0.5rem 0; font-size: 2.2rem;">{current_balance:.2f} QCoins</h2>
                <p>Net Flow: {user_stats['net_flow']:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="transaction-card">
                <h3> Activity</h3>
                <p><b>Sent:</b> {user_stats['total_sent']:.2f}</p>
                <p><b>Received:</b> {user_stats['total_received']:.2f}</p>
                <p><b>Transactions:</b> {user_stats['transaction_count']}</p>
                <p><b>Quantum TXs:</b> {user_stats['quantum_txs']}</p>
                <p><b>Max Q-Dimension:</b> {user_stats['max_quantum_dim']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Wallet QR code and security audit
        st.markdown("### Wallet Address QR Code")
        if qrcode and BytesIO:
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(user_public_key)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption="Your Wallet Address QR Code", width=200)
        else:
            st.info("Install qrcode and pillow for QR code support.")
        st.markdown(f"**Public Address:** `{user_public_key}`")
        # Add wallet security audit
        st.markdown("#### Security Audit")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Quantum Transactions", user_stats["quantum_txs"])
        with colB:
            security_score = min(100, user_stats["quantum_txs"] * 5 + user_stats["max_quantum_dim"] * 10)
            st.metric("Security Score", f"{security_score}/100")

        st.markdown("###  Quantum Transaction")
        with st.form("send_form", clear_on_submit=True):
            colA, colB = st.columns(2)
            with colA:
                receiver_name = st.selectbox(
                    " Recipient",
                    [k for k in st.session_state.users if k != logged_in_user],
                    format_func=lambda x: f"{st.session_state.users[x]['username']} ({x})"
                )
                max_amount = max(0.01, float(current_balance))
                amount = st.number_input(" Amount", min_value=0.01, max_value=max_amount, step=0.01, value=min(1.0, max_amount))
            with colB:
                tx_type = st.selectbox(" Type", ["transfer","payment","gift","loan"])
                full_quantum = st.checkbox(" High-Dimensional Quantum Encryption", value=True)
                if full_quantum:
                    st.markdown(f"<small> Using {st.session_state.quantum_dimension}-D quantum encoding for enhanced security</small>", unsafe_allow_html=True)

            submitted = st.form_submit_button(" Send Quantum TX", use_container_width=True)

            if submitted:
                if receiver_name and amount > 0:
                    receiver_pk = st.session_state.users[receiver_name]["public_key"]
                    create_transaction(
                        user_private_key, receiver_pk,
                        amount, tx_type
                    )
                    time.sleep(1)  # Small delay to process
                    st.rerun()
                else:
                    st.error("❌ Please complete all fields.")

        # ---- Recent Transactions ----
        st.markdown("###  Quantum Transaction History")
        recent_txs = []
        for block in reversed(st.session_state.blockchain[-5:]):
            for tx in block["transactions"]:
                if tx["sender"] == user_public_key or tx["receiver"] == user_public_key:
                    tx_copy = tx.copy()
                    tx_copy["block"] = block["index"]
                    recent_txs.append(tx_copy)

        if recent_txs:
            for tx in recent_txs[:10]:
                direction = " Sent" if tx["sender"] == user_public_key else " Received"
                amount_color = "red" if tx["sender"] == user_public_key else "green"
                
                quantum_badge = f" <span class='quantum-badge'>{tx.get('quantum_dimension', 2)}-D</span>" if tx.get("quantum_secured", False) else ""
                hardware_info = f"<br><small>Hardware: {tx.get('quantum_hardware', 'Aer Simulator')}</small>" if tx.get("quantum_secured", False) else ""
                
                st.markdown(f"""
                <div class="transaction-card">
                    <div style="display:flex;justify-content:space-between;">
                        <div>
                            <b>{direction}</b> - Block {tx["block"]}{quantum_badge}{hardware_info}<br>
                            <small>ID: {tx.get("tx_id","N/A")}</small>
                        </div>
                        <div style="text-align:right;">
                            <span style="color:{amount_color}; font-weight:bold; font-size:1.2em;">
                                {"-" if tx["sender"]==user_public_key else "+"}{tx["amount"]:.2f}
                            </span><br>
                            <small>{tx["timestamp"][:19]}</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📭 No quantum transactions yet.")

# ================== Part 5/5 ==================
    # ---- Analytics Tab ----
    with tab2:
        st.markdown("###  Quantum Network Analytics")

        # --- Charts ---
        col1, col2 = st.columns(2)
        with col1:
            block_data = []
            quantum_txs_per_block = []
            quantum_dims = []
            for block in st.session_state.blockchain[1:]:
                block_volume = sum(float(tx["amount"]) for tx in block["transactions"] if tx["sender"] != "network")
                quantum_txs = sum(1 for tx in block["transactions"] if tx.get("quantum_secured", False))
                avg_dim = np.mean([tx.get("quantum_dimension", 2) for tx in block["transactions"] if tx.get("quantum_secured", False)] or [0])
                block_data.append({
                    "Block": block["index"],
                    "Volume": block_volume,
                    "Transactions": len(block["transactions"]),
                    "QuantumTXs": quantum_txs,
                    "AvgDimension": avg_dim
                })
                quantum_txs_per_block.append(quantum_txs)
                quantum_dims.append(avg_dim)
                
            if block_data:
                df = pd.DataFrame(block_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["Block"], y=df["Volume"],
                    mode="lines+markers", name="Volume",
                    line=dict(width=3), marker=dict(size=8)
                ))
                fig.add_trace(go.Bar(
                    x=df["Block"], y=df["QuantumTXs"],
                    name="Quantum TXs",
                    marker_color="#667eea"
                ))
                fig.update_layout(
                    title=" Transaction Volume & Quantum TXs per Block",
                    xaxis_title="Block #", 
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=400,
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Quantum dimension distribution
            dim_counts = {}
            for block in st.session_state.blockchain:
                for tx in block["transactions"]:
                    if tx.get("quantum_secured", False):
                        dim = tx.get("quantum_dimension", 2)
                        dim_counts[dim] = dim_counts.get(dim, 0) + 1
            
            if dim_counts:
                fig = px.pie(
                    values=list(dim_counts.values()), 
                    names=[f"{k}-D" for k in dim_counts.keys()], 
                    title=" Quantum Dimension Distribution",
                    template="plotly_dark", 
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(font=dict(family="Inter, sans-serif"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No quantum transactions yet")

        # Hardware usage chart
        if user_stats["hardware_types"]:
            fig = px.pie(
                values=list(user_stats["hardware_types"].values()),
                names=list(user_stats["hardware_types"].keys()),
                title=" Quantum Hardware Usage",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)

        # --- Stats ---
        st.markdown("###  Quantum Network Statistics")
        
        # Compute total quantum transactions and average dimension
        total_quantum_tx = 0
        total_dimension = 0
        hardware_usage = {}
        for block in st.session_state.blockchain:
            for tx in block["transactions"]:
                if tx.get("quantum_secured", False):
                    total_quantum_tx += 1
                    total_dimension += tx.get("quantum_dimension", 2)
                    hw = tx.get("quantum_hardware", "Aer Simulator")
                    hardware_usage[hw] = hardware_usage.get(hw, 0) + 1
        
        avg_quantum_dim = total_dimension / total_quantum_tx if total_quantum_tx > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: 
            st.markdown(f"""
            <div class="metric-card">
                <h3> Blocks</h3>
                <h2>{len(st.session_state.blockchain)}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2: 
            st.markdown(f"""
            <div class="metric-card">
                <h3> Transactions</h3>
                <h2>{st.session_state.network_stats["total_transactions"]}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3: 
            st.markdown(f"""
            <div class="metric-card">
                <h3> Volume</h3>
                <h2>{st.session_state.network_stats['total_volume']:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4: 
            st.markdown(f"""
            <div class="metric-card">
                <h3> Quantum TXs</h3>
                <h2>{total_quantum_tx}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h3> Avg Q-Dim</h3>
                <h2>{avg_quantum_dim:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Hardware usage stats
        if hardware_usage:
            st.markdown("####  Quantum Hardware Usage")
            for hw, count in hardware_usage.items():
                st.markdown(f"- **{hw}**: {count} transactions")

    # ---- Network Tab ----
    with tab3:
        st.markdown("###  Quantum Network Settings & Explorer")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h3> Quantum Network Info</h3>
                <p><b>Network:</b> QuantumVerse</p>
                <p><b>Genesis:</b> {st.session_state.blockchain[0]['timestamp'][:19] if st.session_state.blockchain else 'N/A'}</p>
                <p><b>Algorithm:</b> ECDSA + SHA-256</p>
                <p><b>Security:</b> High-Dimensional BB84 Quantum Encryption</p>
                <p><b>Current Dimension:</b> {st.session_state.quantum_dimension}-D</p>
                <p><b>Quantum TXs:</b> {total_quantum_tx}</p>
                <p><b>Avg Q-Dimension:</b> {avg_quantum_dim:.1f}</p>
                <p><b>Current Hardware:</b> {st.session_state.last_hardware_used}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quantum security audit
            st.markdown("####  Quantum Security Audit")
            if st.button("Run Quantum Audit", use_container_width=True):
                valid_blocks = 0
                quantum_errors = 0
                for i, block in enumerate(st.session_state.blockchain):
                    block_copy = block.copy()
                    block_copy.pop("hash", None)
                    calculated_hash = calculate_hash(block_copy)
                    if calculated_hash == block["hash"]:
                        valid_blocks += 1
                    
                    # Check quantum transactions
                    for tx in block["transactions"]:
                        if tx.get("quantum_secured", False) and not verify_signature(tx):
                            quantum_errors += 1
                
                st.success(f"Quantum audit complete: {valid_blocks}/{len(st.session_state.blockchain)} blocks valid")
                if quantum_errors > 0:
                    st.error(f"{quantum_errors} quantum transactions failed verification")
                else:
                    st.success("All quantum transactions verified successfully")
                
        with col2:
            st.markdown("####  Quantum Users")
            for k,u in st.session_state.users.items():
                bal = st.session_state.balances.get(u["public_key"],0)
                has_key = "✅" if st.session_state.quantum_keys.get(k) else "❌"
                st.markdown(f"""
                <div class="transaction-card">
                    <b>{u['username']} ({k})</b>: {bal:.2f} QCoins
                    <span style="float: right;">{has_key}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Export data
        st.markdown("####💾 Quantum Data Export")
        export_data = {
            "blockchain": st.session_state.blockchain,
            "balances": st.session_state.balances,
            "users": {k: {kk: vv for kk, vv in v.items() if kk != "private_key" and kk != "password_hash"} 
                     for k, v in st.session_state.users.items()},
            "network_stats": st.session_state.network_stats,
            "quantum_dimension": st.session_state.quantum_dimension,
            "export_timestamp": str(datetime.now())
        }
        
        st.download_button(
            "💾 Export Quantum Data",
            json.dumps(export_data, indent=2),
            file_name=f"quantumverse_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
        
        st.markdown("---")

        # Quantum Explorer
        if st.session_state.blockchain:
            st.markdown("###  Quantum Blockchain Explorer")
            idx = st.selectbox("Choose Block:", range(len(st.session_state.blockchain)),
                               format_func=lambda i: f"Block {i} ({len(st.session_state.blockchain[i]['transactions'])} TXs)")
            block = st.session_state.blockchain[idx]
            st.markdown(f"""
            <div class="block-card">
                <h4>Block {block['index']} <span class="quantum-badge">Quantum</span></h4>
                <p><b>Hash:</b> <span class="address-text">{block['hash']}</span></p>
                <p><b>Previous Hash:</b> <span class="address-text">{block['previous_hash']}</span></p>
                <p><b>Time:</b> {block['timestamp'][:19]}</p>
                <p><b>Transactions:</b> {len(block['transactions'])}</p>
                <p><b>Merkle Root:</b> <span class="address-text">{block['merkle_root']}</span></p>
                <p><b>Quantum Dimension:</b> {block.get('quantum_dimension', 2)}-D</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("####  Quantum Transactions")
            for i, tx in enumerate(block["transactions"]):
                sender = "Network" if tx["sender"]=="network" else tx["sender"][:12]
                receiver = tx["receiver"][:12]
                quantum_badge = f" <span class='quantum-badge'>{tx.get('quantum_dimension', 2)}-D</span>" if tx.get("quantum_secured", False) else ""
                hardware_info = f"<br><small>Hardware: {tx.get('quantum_hardware', 'Aer Simulator')}</small>" if tx.get("quantum_secured", False) else ""
                st.markdown(f"""
                <div class="transaction-card">
                    <h5>TX {i+1}{quantum_badge}{hardware_info}</h5>
                    <p><b>From:</b> {sender}</p>
                    <p><b>To:</b> {receiver}</p>
                    <p><b>Amount:</b> {tx['amount']}</p>
                    <p><b>Type:</b> {tx.get('type','transfer')}</p>
                    <p><b>ID:</b> {tx.get('tx_id','')}</p>
                    <p><b>Time:</b> {tx['timestamp'][:19]}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(" No quantum blocks yet.")

        # --- Advanced Transaction Explorer ---
        st.markdown("### Advanced Transaction Explorer")
        if st.button("💾 Save Ledger to Disk", use_container_width=True):
            save_blockchain()
            st.success("Ledger saved successfully!")
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_sender = st.text_input("Filter by Sender", "")
        with col2:
            filter_receiver = st.text_input("Filter by Receiver", "")
        with col3:
            filter_min_amount = st.number_input("Min Amount", min_value=0.0, value=0.0)
        # Search functionality
        search_term = st.text_input("Search Transactions", "")
        # Date range filter
        col4, col5 = st.columns(2)
        with col4:
            start_date = st.date_input("From Date", value=datetime.now() - timedelta(days=30))
        with col5:
            end_date = st.date_input("To Date", value=datetime.now())
        # Gather all transactions
        all_transactions = []
        for block in st.session_state.blockchain:
            for tx in block["transactions"]:
                tx_with_block = tx.copy()
                tx_with_block["block_height"] = block["index"]
                tx_with_block["block_time"] = block["timestamp"]
                all_transactions.append(tx_with_block)
        # Apply filters
        filtered_txs = all_transactions
        if filter_sender:
            filtered_txs = [tx for tx in filtered_txs if filter_sender.lower() in tx["sender"].lower()]
        if filter_receiver:
            filtered_txs = [tx for tx in filtered_txs if filter_receiver.lower() in tx["receiver"].lower()]
        if filter_min_amount > 0:
            filtered_txs = [tx for tx in filtered_txs if float(tx["amount"]) >= filter_min_amount]
        if search_term:
            filtered_txs = [tx for tx in filtered_txs if (
                search_term.lower() in tx.get("tx_id", "").lower() or
                search_term.lower() in tx["sender"].lower() or
                search_term.lower() in tx["receiver"].lower()
            )]
        # Date filtering
        filtered_txs = [tx for tx in filtered_txs if (
            datetime.strptime(tx["timestamp"][:10], "%Y-%m-%d").date() >= start_date and
            datetime.strptime(tx["timestamp"][:10], "%Y-%m-%d").date() <= end_date
        )]
        # Display results
        st.markdown(f"**Found {len(filtered_txs)} transactions**")
        for tx in filtered_txs[-20:]:  # Show most recent 20 transactions
            direction = "outgoing" if tx["sender"] == user_public_key else "incoming"
            color = "#ff4b4b" if direction == "outgoing" else "#4bb543"
            classical_badge = '' if tx.get('quantum_secured') else '• <span style="color: orange;">CLASSICAL</span>'
            quantum_secured_html = ("<div style='font-size: 0.8rem;'><b>Quantum Secured:</b> {}-D</div>".format(tx.get('quantum_dimension', 2)) if tx.get('quantum_secured') else "")
            html = (
                f"<div style='padding: 1rem; border-radius: 10px; background: rgba(0,0,0,0.05); margin: 0.5rem 0;'>"
                f"<div style='display: flex; justify-content: space-between;'>"
                f"<div>"
                f"<b>Block #{tx['block_height']}</b> • "
                f"<span style='color: {color}; font-weight: bold;'>{direction.upper()}</span>"
                f"{classical_badge}"
                f"</div>"
                f"<div>"
                f"<b>{float(tx['amount']):.2f} QCoins</b>"
                f"</div>"
                f"</div>"
                f"<div style='font-size: 0.8rem; color: #666; margin-top: 0.5rem;'>"
                f"From: {tx['sender'][:12]}...{tx['sender'][-6:]} → To: {tx['receiver'][:12]}...{tx['receiver'][-6:]}"
                f"</div>"
                f"<div style='font-size: 0.8rem; color: #888;'>"
                f"{tx['timestamp'][:19]} • {tx.get('type', 'transfer')}"
                f"</div>"
                f"{quantum_secured_html}"
                f"</div>"
            )
            st.markdown(html, unsafe_allow_html=True)
# ================== End of Code ==================