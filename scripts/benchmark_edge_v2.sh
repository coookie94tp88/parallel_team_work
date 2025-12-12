#!/bin/bash
# Updated benchmark script for Edge-Aware Version
echo "=== Edge-Aware Mosaic Performance Benchmark ==="
echo ""
echo "Running automated benchmark for video_mosaic_edge..."
echo ""

# Function to run test with timeout
run_test() {
    threads=$1
    echo "Testing with $threads threads..."
    
    # Run in background
    ./bin/video_mosaic_edge --ultra -f -j $threads -d data/cifar_tiles > edge_bench_${threads}.log 2>&1 &
    PID=$!
    
    # Let it run for 10 seconds
    sleep 10
    
    # Kill it
    kill -SIGINT $PID 2>/dev/null
    wait $PID 2>/dev/null
    
    # Parse log for FPS
    avg_fps=$(grep "Avg FPS:" edge_bench_${threads}.log | tail -n 1 | awk '{print $NF}')
    echo "  -> Result: $avg_fps FPS"
    echo ""
}

# Run tests
run_test 1
run_test 4
run_test 8

echo "=== Results Summary ==="
grep "Avg FPS:" edge_bench_*.log | tail -n 3
rm edge_bench_*.log
