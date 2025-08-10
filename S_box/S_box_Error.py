import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class QuantumErrorAnalyzer:
    def __init__(self, simulator_counts: Dict[str, int], hardware_counts: Dict[str, int], 
                 correct_answer: str, total_shots: int):
        """
        Initialize error analyzer with results from both simulator and hardware.
        
        Args:
            simulator_counts: Counts from ideal AerSimulator 
            hardware_counts: Counts from IBM quantum hardware
            correct_answer: The correct y1..y8 bitstring from simulator
            total_shots: Total number of shots used in hardware run
        """
        self.sim_counts = simulator_counts
        self.hw_counts = hardware_counts
        self.correct_answer = correct_answer
        self.total_shots = total_shots
        
    def compute_fidelity(self) -> float:
        """Compute fidelity: fraction of times hardware got the correct answer."""
        correct_hw_counts = self.hw_counts.get(self.correct_answer, 0)
        return (correct_hw_counts / self.total_shots) * 100
    
    def compute_bit_error_rates(self) -> Dict[str, float]:
        """Compute error rate for each output bit position (y1..y8)."""
        bit_errors = {f'y{i+1}': 0 for i in range(8)}
        
        for outcome, count in self.hw_counts.items():
            for bit_pos in range(8):
                if outcome[bit_pos] != self.correct_answer[bit_pos]:
                    bit_errors[f'y{bit_pos+1}'] += count
        
        # Convert to error rates
        for bit in bit_errors:
            bit_errors[bit] = (bit_errors[bit] / self.total_shots) * 100
            
        return bit_errors
    
    def compute_hamming_distance_distribution(self) -> Dict[int, float]:
        """Compute distribution of Hamming distances from correct answer."""
        hamming_dist = {}
        
        for outcome, count in self.hw_counts.items():
            # Calculate Hamming distance
            dist = sum(c1 != c2 for c1, c2 in zip(outcome, self.correct_answer))
            hamming_dist[dist] = hamming_dist.get(dist, 0) + count
            
        # Convert to percentages
        for dist in hamming_dist:
            hamming_dist[dist] = (hamming_dist[dist] / self.total_shots) * 100
            
        return hamming_dist
    
    def compute_entropy(self) -> Tuple[float, float]:
        """Compute Shannon entropy for both simulator and hardware results."""
        def shannon_entropy(counts, total):
            entropy = 0
            for count in counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            return entropy
        
        sim_total = sum(self.sim_counts.values())
        hw_entropy = shannon_entropy(self.hw_counts, self.total_shots)
        sim_entropy = shannon_entropy(self.sim_counts, sim_total)
        
        return sim_entropy, hw_entropy
    
    def compute_tvd(self) -> float:
        """Compute Total Variation Distance between distributions."""
        # Normalize distributions
        sim_total = sum(self.sim_counts.values())
        
        tvd = 0
        all_outcomes = set(self.sim_counts.keys()) | set(self.hw_counts.keys())
        
        for outcome in all_outcomes:
            sim_prob = self.sim_counts.get(outcome, 0) / sim_total
            hw_prob = self.hw_counts.get(outcome, 0) / self.total_shots
            tvd += abs(sim_prob - hw_prob)
            
        return tvd / 2  # TVD is half the L1 distance
    
    def generate_error_report(self) -> str:
        """Generate comprehensive error analysis report."""
        fidelity = self.compute_fidelity()
        bit_errors = self.compute_bit_error_rates()
        hamming_dist = self.compute_hamming_distance_distribution()
        sim_entropy, hw_entropy = self.compute_entropy()
        tvd = self.compute_tvd()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    QUANTUM ERROR ANALYSIS                    ║
╚══════════════════════════════════════════════════════════════╝

 OVERALL FIDELITY
   • Correct answer: {self.correct_answer}
   • Hardware fidelity: {fidelity:.2f}% ({self.hw_counts.get(self.correct_answer, 0)}/{self.total_shots} shots)
   • Error rate: {100-fidelity:.2f}%

 BIT-WISE ERROR RATES
"""
        for bit, error_rate in bit_errors.items():
            report += f"   • {bit}: {error_rate:.2f}% error rate\n"
        
        avg_bit_error = np.mean(list(bit_errors.values()))
        report += f"   • Average bit error: {avg_bit_error:.2f}%\n"
        
        report += f"""
 HAMMING DISTANCE FROM CORRECT ANSWER
"""
        for dist in sorted(hamming_dist.keys()):
            report += f"   • {dist} bits wrong: {hamming_dist[dist]:.2f}% of shots\n"
        
        report += f"""
 DISTRIBUTION ANALYSIS  
   • Simulator entropy: {sim_entropy:.3f} bits
   • Hardware entropy: {hw_entropy:.3f} bits
   • Entropy increase: {hw_entropy - sim_entropy:.3f} bits (higher = more noise)
   • Total Variation Distance: {tvd:.3f} (0=identical, 1=completely different)

 OUTCOME STATISTICS
   • Simulator unique outcomes: {len(self.sim_counts)}
   • Hardware unique outcomes: {len(self.hw_counts)}
   • Most frequent hardware result: {max(self.hw_counts, key=self.hw_counts.get)} ({max(self.hw_counts.values())} counts)
"""
        return report


def compare_results(sim_counts=None, hw_counts=None, correct_answer=None, total_shots=None):
    """
    Compare simulator and hardware results for error analysis.
    """
    
    if sim_counts is None or hw_counts is None:
        print("Please provide simulator and hardware count dictionaries")
        return
    
    if correct_answer is None:
        # Assume most frequent simulator result is correct
        correct_answer = max(sim_counts, key=sim_counts.get)
    
    if total_shots is None:
        total_shots = sum(hw_counts.values())
    
    # Create analyzer and generate report
    analyzer = QuantumErrorAnalyzer(sim_counts, hw_counts, correct_answer, total_shots)
    
    print(analyzer.generate_error_report())
    
    # Plot Hamming distance distribution
    hamming_dist = analyzer.compute_hamming_distance_distribution()
    
    plt.figure(figsize=(10, 6))
    distances = list(hamming_dist.keys())
    percentages = list(hamming_dist.values())
    
    plt.bar(distances, percentages, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.xlabel('Hamming Distance from Correct Answer')
    plt.ylabel('Percentage of Shots (%)')
    plt.title('Distribution of Errors in IBM Quantum Hardware')
    plt.xticks(range(9))  # 0 to 8 bit errors possible
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, v in enumerate(percentages):
        plt.text(distances[i], v + 0.1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return analyzer


def run_complete_error_analysis():
    """
    Complete workflow: get user input, run simulator, then hardware, then analyze errors.
    """
    # Get user input
    user_input = input("Enter 8-bit input a..h (e.g. 10110011): ").strip().replace(" ", "")
    
    # Validate input
    if len(user_input) != 8 or not all(bit in '01' for bit in user_input):
        print("Error: Please enter exactly 8 bits (0s and 1s)")
        return None
    
    print(f"Running complete analysis for input: {user_input}")
    
    try:
        print("\n Step 1: Running simulator for baseline...")
        # Import and run simulator
        from S_box_Sim import run_simulator
        correct_answer, sim_counts, circuit = run_simulator(user_input, shots=1024)
        print(f" Simulator complete. Correct answer: {correct_answer}")
        
    except ImportError:
        print(" Error: Could not import S_box_Sim.py")
        print("Please ensure S_box_Sim.py is in the same directory")
        return None
    
    try:
        print("\n Step 2: Running on IBM quantum hardware...")  
        # Import and run hardware
        from S_box import run_on_ibm_device
        ystring, hw_counts, qc, isa_qc, backend, elapsed = run_on_ibm_device(user_input, shots=8192)
        print(f" Hardware run complete on {backend.name}")
        
        # Display hardware output results
        print(f"\n HARDWARE OUTPUT RESULTS:")
        print(f"   • Backend used: {backend.name}")
        print(f"   • Input (a..h): {user_input}")
        print(f"   • Most likely output (y1..y8): {ystring}")
        print(f"   • Total unique outcomes: {len(hw_counts)}")
        print(f"   • Execution time: {elapsed:.1f} seconds")
        
        print(f"\n   Top 10 hardware results:")
        for i, (bitstring, count) in enumerate(sorted(hw_counts.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            percentage = (count / 8192) * 100
            print(f"   {i:2d}. {bitstring} - {count:4d} counts ({percentage:5.2f}%)")
        
    except ImportError:
        print(" Error: Could not import S_box.py")
        print("Please ensure S_box.py is in the same directory")
        return None
    
    print("\n Step 3: Analyzing quantum errors...")
    analyzer = compare_results(
        sim_counts=sim_counts,
        hw_counts=hw_counts, 
        correct_answer=correct_answer,
        total_shots=8192
    )
    
    print(" Error analysis complete!")
    return analyzer

if __name__ == "__main__":
    print("Quantum Error Analysis Tool")
    print("This will run simulator → IBM hardware → error analysis")
    print("=" * 50)
    
    analyzer = run_complete_error_analysis()
    
    if analyzer:
        print("\n" + "=" * 50)
        print("Analysis complete! Check the error report above and the generated plot.")
    else:
        print("\n Analysis failed. Please check your code files and try again.")
