import subprocess
import re
import matplotlib.pyplot as plt
import os

# --- Configuration ---
EXEC = "./wireroute"
INPUT_FILES = [
    "inputs/timeinput/few_wires.txt",
    "inputs/timeinput/medium_wires.txt",
    "inputs/timeinput/abundant_wires.txt"
]
SA_PROB = 0.1
SA_ITERS = 5
PROCESSORS = [1, 2, 4, 8]
BATCH_SIZES = [4, 16]

def run_experiment(num_procs, batch_size, input_file):
    """Runs the MPI program and returns the computation time."""
    cmd = [
        "mpirun", "-np", str(num_procs),
        EXEC,
        "-f", input_file,
        "-p", str(SA_PROB),
        "-i", str(SA_ITERS),
        "-b", str(batch_size)
    ]
    
    filename = os.path.basename(input_file)
    print(f"Running: mpirun -np {num_procs} {EXEC} -f {filename} -b {batch_size} ... ", end="", flush=True)
    
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

def plot_speedup(batch_size, baselines, parallel_times_dict):
    """Generates and saves a speedup graph for a specific batch size with multiple lines."""
    plt.figure(figsize=(10, 6))

    # Plot ideal linear speedup line
    plt.plot(PROCESSORS, PROCESSORS, linestyle='--', color='k', label='Ideal Linear Speedup')

    # Markers and colors for the different files
    markers = ['o', 's', '^']
    colors = ['b', 'g', 'r']

    # Plot a line for each input file
    for i, input_file in enumerate(INPUT_FILES):
        filename = os.path.basename(input_file)
        baseline_time = baselines[input_file]
        parallel_times = parallel_times_dict[input_file]
        
        # Calculate speedup: S = T_baseline / T_parallel
        speedups = [baseline_time / t if t else 0 for t in parallel_times]
        
        plt.plot(PROCESSORS, speedups, marker=markers[i % len(markers)], 
                 linestyle='-', color=colors[i % len(colors)], 
                 label=f'{filename}')

    plt.title(f'Speedup vs. Processors (Batch Size = {batch_size})')
    plt.xlabel('Number of Processors')
    plt.ylabel('Speedup')
    plt.xticks(PROCESSORS)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    output_filename = f'speedup_batch_{batch_size}.png'
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved graph to {output_filename}")

def main():
    print("Starting automated data collection...\n")
    
    # 1. Get Baseline for EVERY file (1 processor, batch size 1)
    print("--- Collecting Baselines (1 proc, batch 1) ---")
    baselines = {}
    for f in INPUT_FILES:
        b_time = run_experiment(1, 1, f)
        if not b_time:
            print(f"Failed to get baseline for {os.path.basename(f)}. Exiting.")
            return
        baselines[f] = b_time

    # 2. Run experiments for each batch size
    for b in BATCH_SIZES:
        print(f"\n--- Running Experiments for Batch Size {b} ---")
        
        # Dictionary to store the list of times for each file
        parallel_times_dict = {f: [] for f in INPUT_FILES}
        
        for f in INPUT_FILES:
            print(f"\nTesting {os.path.basename(f)}:")
            for p in PROCESSORS:
                t = run_experiment(p, b, f)
                parallel_times_dict[f].append(t)
                
        # 3. Graph the results for this batch size
        print(f"\nGenerating graph for Batch Size {b}...")
        plot_speedup(b, baselines, parallel_times_dict)
        
    print("\nAll experiments complete!")

if __name__ == "__main__":
    main()
