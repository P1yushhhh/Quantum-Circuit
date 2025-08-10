from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import time

def build_parity_circuit(bits):
    if len(bits) != 8 or any(b not in '01' for b in bits):
        raise ValueError("bits must be 8 characters of '0'/'1'.")

    q_in = QuantumRegister(8, 'in')
    q_out = QuantumRegister(8, 'out')
    c_out = ClassicalRegister(8, 'c')
    qc = QuantumCircuit(q_in, q_out, c_out)

    for i, ch in enumerate(bits):
        if ch == '1':
            qc.x(q_in[i])

    deps = [
        [0,1,2,5,6], [1,2,3,6,7], [0,2,3,4,7], [0,1,3,4,5],
        [1,2,4,5,6], [2,3,5,6,7], [0,3,4,6,7], [0,1,4,5,7]
    ]

    for out_idx, inputs in enumerate(deps):
        for in_idx in inputs:
            qc.cx(q_in[in_idx], q_out[out_idx])
        if out_idx in (0, 2, 5, 7):
            qc.x(q_out[out_idx])

    qc.measure(q_out, c_out)
    return qc

def run_simulator(bits, shots=1024):
    qc = build_parity_circuit(bits)
    sim = AerSimulator()
    t_qc = transpile(qc, sim)

    start_time = time.time()
    job = sim.run(t_qc, shots=shots)
    result = job.result()
    elapsed = time.time() - start_time

    counts = {k[::-1]: v for k, v in result.get_counts().items()}
    most_common = max(counts.items(), key=lambda x: x[1])[0]

    print("\n--- SIMULATOR RESULTS (IDEAL/NOISELESS) ---")
    print("Backend:", sim.name)
    print("Execution time (s):", elapsed)
    print("Inputs (a..h):", bits)
    print("Correct outputs (y1..y8):", most_common)
    
    # Show fidelity info
    correct_count = counts.get(most_common, 0)
    fidelity = (correct_count / shots) * 100
    print(f"Fidelity: {fidelity:.1f}% ({correct_count}/{shots} correct)")
    
    print("All counts (should show one dominant result):")
    for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (v / shots) * 100
        print(f"  {k}: {v:4d} ({percentage:5.1f}%)")
    
    print(f"\nCircuit complexity:")
    print(f"  Depth: {qc.depth()}")
    print(f"  Size: {qc.size()}")
    print(f"  CNOT gates: {qc.count_ops().get('cx', 0)}")
    
    print(f"\nCircuit diagram:")
    print(qc.draw(fold=120))
    
    return most_common, counts, qc

if __name__ == "__main__":  # Fixed this line
    user_input = input("Enter 8-bit input a..h: ").replace(" ", "")
    correct_answer, sim_counts, circuit = run_simulator(user_input, shots=1024)
    
    print(f"\n BASELINE ESTABLISHED")
    print(f"Correct answer for input '{user_input}': {correct_answer}")
    print(f"Use this to compare against IBM quantum hardware noise!")
