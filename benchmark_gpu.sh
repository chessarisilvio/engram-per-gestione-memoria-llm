#!/usr/bin/env bash
# benchmark_gpu.sh
# Manual benchmark script for Engram memory operations on GPU-free system.
# Measures insert, search, and update performance.
# Usage: ./benchmark_gpu.sh [options]
# Options (can also be set via environment variables):
#   -n NUM      Number of operations per test (default: 1000)
#   -d DIM      Embedding dimension (default: 128)
#   -o FILE     Output file for results (default: benchmark_results.txt)
#   -g          Enable GPU memory logging via nvidia-smi (if available)
#   -h          Show this help

set -euo pipefail

# Default values
NUM_OPS=${NUM_OPS:-1000}
EMBED_DIM=${EMBED_DIM:-128}
OUTFILE=${OUTFILE:-benchmark_results.txt}
LOG_GPU=${LOG_GPU:-0}

# Helper function to show usage
usage() {
    grep '^#' "$0" | cut -c4-
    exit 0
}

# Parse command line arguments
while getopts ":n:d:o:gh" opt; do
    case $opt in
        n) NUM_OPS="$OPTARG" ;;
        d) EMBED_DIM="$OPTARG" ;;
        o) OUTFILE="$OPTARG" ;;
        g) LOG_GPU=1 ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :)  echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# Log GPU memory if requested and nvidia-smi is available
log_gpu_mem() {
    if [[ $LOG_GPU -eq 1 && -x "$(command -v nvidia-smi)" ]]; then
        echo "=== GPU Memory Usage ===" >> "$OUTFILE"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader >> "$OUTFILE"
        echo "" >> "$OUTFILE"
    fi
}

# Start benchmark
echo "Starting Engram benchmark..." | tee -a "$OUTFILE"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$OUTFILE"
echo "Operations per test: $NUM_OPS" | tee -a "$OUTFILE"
echo "Embedding dimension: $EMBED_DIM" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

log_gpu_mem

# Create a temporary Python script for benchmarking
cat > /tmp/benchmark_engram.py << 'EOF'
#!/usr/bin/env python3
import time
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engram_manager import EngramStore, EngramIndex

def benchmark_insert(store, index, num_ops, embed_dim):
    start = time.perf_counter()
    for i in range(num_ops):
        engram = {
            "id": "",
            "timestamp": "",
            "embedding": np.random.randn(embed_dim).astype(float).tolist(),
            "metadata": {"source": "benchmark", "iteration": i}
        }
        engram_id = store.add_engram(engram)
        index.add_engram(engram_id, engram["embedding"])
    end = time.perf_counter()
    return end - start

def benchmark_search(index, num_ops, embed_dim):
    # Generate a random query vector
    query = np.random.randn(embed_dim).astype(float).tolist()
    start = time.perf_counter()
    for _ in range(num_ops):
        index.search(query, k=5)
    end = time.perf_counter()
    return end - start

def benchmark_update(store, index, num_ops, embed_dim):
    # First, add some engrams to update
    ids = []
    for i in range(min(100, num_ops)):  # ensure we have some to update
        engram = {
            "id": "",
            "timestamp": "",
            "embedding": np.random.randn(embed_dim).astype(float).tolist(),
            "metadata": {"source": "benchmark", "iteration": i}
        }
        engram_id = store.add_engram(engram)
        index.add_engram(engram_id, engram["embedding"])
        ids.append(engram_id)

    if not ids:
        ids = ["dummy"]  # fallback, though update will fail; but we assume num_ops>0

    start = time.perf_counter()
    for i in range(num_ops):
        engram_id = ids[i % len(ids)]
        engram = {
            "id": engram_id,
            "timestamp": "",
            "embedding": np.random.randn(embed_dim).astype(float).tolist(),
            "metadata": {"source": "benchmark", "iteration": i, "updated": True}
        }
        store.update_engram(engram_id, engram)
        # Update index: remove old and add new? For simplicity, we just add new (duplicate)
        # In a real scenario, you'd need to update the index; we'll just add new vector
        index.add_engram(engram_id + "_update", engram["embedding"])  # approximate
    end = time.perf_counter()
    return end - start

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-ops", type=int, default=1000, help="Number of operations")
    parser.add_argument("-d", "--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("-o", "--output", type=str, default="benchmark_results.txt", help="Output file")
    args = parser.parse_args()

    store = EngramStore()
    index = EngramIndex(embedding_dim=args.embed_dim)

    insert_time = benchmark_insert(store, index, args.num_ops, args.embed_dim)
    search_time = benchmark_search(index, args.num_ops, args.embed_dim)
    update_time = benchmark_update(store, index, args.num_ops, args.embed_dim)

    with open(args.output, 'w') as f:
        f.write(f"Insert {args.num_ops} ops: {insert_time:.4f} sec\\n")
        f.write(f"Search {args.num_ops} ops: {search_time:.4f} sec\\n")
        f.write(f"Update {args.num_ops} ops: {update_time:.4f} sec\\n")
        f.write(f"Insert ops/sec: {args.num_ops/insert_time:.2f}\\n")
        f.write(f"Search ops/sec: {args.num_ops/search_time:.2f}\\n")
        f.write(f"Update ops/sec: {args.num_ops/update_time:.2f}\\n")

    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()
EOF

chmod +x /tmp/benchmark_engram.py

# Run the benchmark
/tmp/benchmark_engram.py -n "$NUM_OPS" -d "$EMBED_DIM" -o "$OUTFILE" 2>&1 | tee -a "$OUTFILE"

# Log GPU memory after if requested
log_gpu_mem

echo "Benchmark completed. Results in $OUTFILE" | tee -a "$OUTFILE"

# Cleanup
rm -f /tmp/benchmark_engram.py