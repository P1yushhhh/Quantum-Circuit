import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def build_parity_circuit(bits):
    if len(bits) != 8 or any(b not in '01' for b in bits):
        raise ValueError("bits must be an 8-char string of 0/1, e.g. '10100110'")

    q_in = QuantumRegister(8, 'in')     # a..h
    q_out = QuantumRegister(8, 'out')   # y1..y8
    c_out = ClassicalRegister(8, 'c')   # classical bits to read outputs
    qc = QuantumCircuit(q_in, q_out, c_out)

    for i, ch in enumerate(bits):
        if ch == '1':
            qc.x(q_in[i])

    deps = [
        [0,1,2,5,6],   # y1 = NOT(a+b+c+f+g)
        [1,2,3,6,7],   # y2 = b+c+d+g+h
        [0,2,3,4,7],   # y3 = NOT(a+c+d+e+h)
        [0,1,3,4,5],   # y4 = a+b+d+e+f
        [1,2,4,5,6],   # y5 = b+c+e+f+g
        [2,3,5,6,7],   # y6 = NOT(c+d+f+g+h)
        [0,3,4,6,7],   # y7 = a+d+e+g+h
        [0,1,4,5,7],   # y8 = NOT(a+b+e+f+h)
    ]

    for out_idx, inputs in enumerate(deps):
        for in_idx in inputs:
            qc.cx(q_in[in_idx], q_out[out_idx])
        if out_idx in (0, 2, 5, 7):  # y1,y3,y6,y8 negated
            qc.x(q_out[out_idx])

    qc.measure(q_out, c_out)
    return qc

def run_on_ibm_device(bits, shots=8192, backend_name="ibm_brisbane"):
    print(f"Building circuit for input: {bits}")
    qc = build_parity_circuit(bits)
    
    # Measure original circuit properties
    print(f"Original circuit depth: {qc.depth()}")
    print(f"Original circuit gate count: {qc.count_ops()}")
    print(f"Original circuit qubits: {qc.num_qubits}")

    print("Connecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    print(f"Transpiling circuit for {backend.name}...")
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=1)
    isa_qc = pm.run(qc)
    
    # Measure transpiled circuit properties
    print(f"Transpiled circuit depth: {isa_qc.depth()}")
    print(f"Transpiled circuit gate count: {isa_qc.count_ops()}")
    print(f"Transpiled circuit qubits: {isa_qc.num_qubits}")

    print(f"Submitting job to {backend.name} with {shots} shots...")
    start_time = time.time()
    
    sampler = Sampler(backend)
    job = sampler.run([isa_qc], shots=shots)
    
    print(f"Job ID: {job.job_id()}")
    print("Waiting for results... (this may take several minutes)")
    
    res = job.result()
    elapsed = time.time() - start_time

    # Get counts from the result
    quasi = res[0].data.c.get_counts()
    
    # Reverse bitstrings so they read y1..y8 (c[0]..c[7])
    decoded = {k[::-1]: v for k, v in quasi.items()}
    most_common = max(decoded.items(), key=lambda x: x[1])[0]
    
    return most_common, decoded, qc, isa_qc, backend, elapsed

if __name__ == "__main__":
    user_input = input("Enter 8-bit input a..h (e.g. 10110011): ").strip().replace(" ", "")
    
    # Choose which IBM device to use
    print("\nAvailable backends: ibm_brisbane, ibm_torino")
    backend_choice = input("Choose backend (press Enter for ibm_brisbane): ").strip()
    if not backend_choice:
        backend_choice = "ibm_brisbane"
    
    shots = 8192  # Increased shots for better statistics on noisy hardware

    print(f"\n{'='*50}")
    print("RUNNING ON IBM QUANTUM HARDWARE")
    print(f"{'='*50}")
    
    ystring, all_counts, qc, isa_qc, backend, elapsed = run_on_ibm_device(
        user_input, shots=shots, backend_name=backend_choice
    )

    print(f"\n{'='*50}")
    print("RESULTS FROM IBM QUANTUM HARDWARE")
    print(f"{'='*50}")
    print("Backend:", backend.name)
    print("Total execution time:", f"{elapsed:.1f} seconds")
    print("Inputs (a..h):", user_input)
    print("Most likely outputs (y1..y8):", ystring)
    
    print(f"\nTop 15 measurement outcomes:")
    for i, (k, v) in enumerate(sorted(all_counts.items(), key=lambda kv: kv[1], reverse=True)[:15], 1):
        percentage = (v / shots) * 100
        print(f"{i:2d}. {k} - {v:4d} counts ({percentage:5.1f}%)")
    
    total_unique = len(all_counts)
    print(f"\nTotal unique outcomes: {total_unique}/256")
    
    print(f"\n{'='*50}")
    print("QUANTUM CIRCUITS")
    print(f"{'='*50}")
    print("Original circuit:")
    print(qc.draw(fold=120))
    print(f"\nTranspiled circuit for {backend.name}:")
    print(isa_qc.draw(fold=120))
    print(f"Transpiled depth: {isa_qc.depth()}")
    print(f"Transpiled gate counts: {isa_qc.count_ops()}")
