import subprocess
import re
import matplotlib.pyplot as plt

# --- Configuration ---
EXEC = "./wireroute"
INPUT_FILE = "inputs/timeinput/medium_wires.txt"
SA_PROB = 0.1
SA_ITERS = 5
PROCESSORS = [1,2, 4, 8, 16, 32, 64, 128]
BATCH_SIZES = [4, 16]

def run_experiment(num_procs, batch_size):
    """Runs the MPI program and returns the computation time."""
    cmd = [
        "mpirun", "-np", str(num_procs),
        EXEC,
        "-f", INPUT_FILE,
        "-p", str(SA_PROB),
        "-i", str(SA_ITERS),
        "-b", str(batch_size)
    ]
    
    print(f"Running: mpirun -np {num_procs} {EXEC} -b {batch_size} ... ", end="", flush=True)
    
    # Run the command and capture both stdout and stderr
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    # Use regex to find the computation time in the output
    match = re.search(r"Computation time \(sec\):\s+([\d\.]+)", result.stdout)
    
    if match:
        time_sec = float(match.group(1))
        print(f"{time_sec:.4f} seconds")
        return time_sec
    else:
        print("ERROR: Could not parse time!")
        print("--- Program Output ---")
        print(result.stdout)
        print(result.stderr)
        return None

def plot_speedup(batch_size, baseline_time, parallel_times):
    """Generates and saves a speedup graph for a specific batch size."""
    # Calculate speedup: S = T_baseline / T_parallel
    speedups = [baseline_time / t if t else 0 for t in parallel_times]
    ideal_speedup = PROCESSORS

    plt.figure(figsize=(8, 6))
    plt.plot(PROCESSORS, speedups, marker='o', linestyle='-', color='b', label=f'Actual Speedup (Batch={batch_size})')
    plt.plot(PROCESSORS, ideal_speedup, linestyle='--', color='k', label='Ideal Linear Speedup')

    plt.title(f'Speedup vs. Processors (Batch Size = {batch_size})')
    plt.xlabel('Number of Processors')
    plt.ylabel('Speedup')
    plt.xticks(PROCESSORS)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    filename = f'speedup_batch_{batch_size}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved graph to {filename}")

def main():
    print("Starting automated data collection...\n")
    
    # 1. Get Baseline (1 processor, batch size 1)
    print("--- Collecting Baseline ---")
    baseline_time = run_experiment(1, 1)
    if not baseline_time:
        print("Failed to get baseline. Exiting.")
        return

    # 2. Run experiments for each batch size
    for b in BATCH_SIZES:
        print(f"\n--- Running Experiments for Batch Size {b} ---")
        times = []
        for p in PROCESSORS:
            t = run_experiment(p, b)
            times.append(t)
            
        # 3. Graph the results for this batch size
        print("\nGenerating graph...")
        plot_speedup(b, baseline_time, times)
        
    print("\nAll experiments complete!")

if __name__ == "__main__":
    main()
