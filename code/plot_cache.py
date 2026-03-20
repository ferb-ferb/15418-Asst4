import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load the data
csv_filename = 'cache_data.csv'
if not os.path.exists(csv_filename):
    print(f"Error: {csv_filename} not found. Run the bash script first!")
    exit(1)

df = pd.read_csv(csv_filename)

# 2. Configuration
files = ['few_wires.txt', 'medium_wires.txt', 'abundant_wires.txt']
batches = [4, 16]
x_ticks = [1, 2, 4, 8]
colors = {'few_wires.txt': 'red', 'medium_wires.txt': 'green', 'abundant_wires.txt': 'blue'}
markers = {'few_wires.txt': 'o', 'medium_wires.txt': 's', 'abundant_wires.txt': '^'}

print("Generating Cache Miss Graphs...")

# ==========================================
# GRAPH SET 1: Total Cache Misses
# ==========================================
for b in batches:
    plt.figure(figsize=(8, 6))
    batch_data = df[df['batch_size'] == b]
    
    for f in files:
        file_data = batch_data[batch_data['input_file'] == f]
        if not file_data.empty:
            plt.plot(file_data['threads'], file_data['total_misses'], 
                     marker=markers[f], color=colors[f], linewidth=2, label=f)
    
    # Styling
    plt.title(f'Total Cache Misses vs Number of Processors (B = {b})', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Processors', fontsize=12)
    plt.ylabel('Total Cache Misses', fontsize=12)
    plt.xticks(x_ticks)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Input Files", loc='upper left')
    
    # Save the plot
    out_name = f'total_misses_B{b}.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {out_name}")

# ==========================================
# GRAPH SET 2: Per-Thread Mean Cache Misses
# ==========================================
for b in batches:
    plt.figure(figsize=(8, 6))
    batch_data = df[df['batch_size'] == b]
    
    for f in files:
        file_data = batch_data[batch_data['input_file'] == f]
        if not file_data.empty:
            plt.plot(file_data['threads'], file_data['mean_misses'], 
                     marker=markers[f], color=colors[f], linewidth=2, label=f)
    
    # Styling
    plt.title(f'Mean Per-Thread Cache Misses vs Number of Processors (B = {b})', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Processors', fontsize=12)
    plt.ylabel('Mean Cache Misses', fontsize=12)
    plt.xticks(x_ticks)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Input Files", loc='upper right')
    
    # Save the plot
    out_name = f'mean_misses_B{b}.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {out_name}")

print("All graphs generated successfully!")
