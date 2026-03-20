#!/bin/bash

# Configuration Arrays
PROCS=(1 2 4 8)
BATCHES=(4 16)
FILES=("few_wires.txt" "medium_wires.txt" "abundant_wires.txt")

# Ensure the executable exists
if [ ! -f "./wireroute" ]; then
  echo "Error: ./wireroute executable not found. Please 'make' your code first."
  exit 1
fi

# Initialize the CSV file with headers
CSV_FILE="cache_data.csv"
echo "threads,batch_size,input_file,total_misses,mean_misses" >$CSV_FILE

echo "Starting Cache Miss Data Collection..."
echo "======================================"

for b in "${BATCHES[@]}"; do
  for f in "${FILES[@]}"; do
    for p in "${PROCS[@]}"; do

      echo "Running: Threads=$p | Batch=$b | File=$f"

      # Run perf stat on the mpirun command.
      # 2> redirects perf's output (stderr) to a temp file.
      # > /dev/null silences your program's normal stdout so your terminal stays clean.
      # NOTE: adjust "inputs/$f" if your text files are stored in a different folder!
      perf stat -e cache-misses mpirun -n $p ./wireroute -f inputs/$f -b $b 2>perf_tmp.txt >/dev/null

      # Parse the total misses from the perf output
      # 1. grep finds the line with "cache-misses"
      # 2. awk grabs the first column (the number)
      # 3. sed removes the commas so we can do math on it
      TOTAL_MISSES=$(grep "cache-misses" perf_tmp.txt | awk '{print $1}' | sed 's/,//g')

      # Fallback in case perf fails or is denied permission
      if [ -z "$TOTAL_MISSES" ]; then
        echo "  -> Error: Could not read cache misses. (Check perf permissions)"
        TOTAL_MISSES=0
        MEAN_MISSES=0
      else
        # Calculate the per-thread mean using integer division
        MEAN_MISSES=$((TOTAL_MISSES / p))
      fi

      # Append the results to the CSV
      echo "$p,$b,$f,$TOTAL_MISSES,$MEAN_MISSES" >>$CSV_FILE

    done
  done
done

# Clean up the temporary file
rm -f perf_tmp.txt

echo "======================================"
echo "Data collection complete! Results saved to $CSV_FILE"
