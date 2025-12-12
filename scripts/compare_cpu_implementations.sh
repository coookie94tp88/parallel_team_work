#!/bin/bash
# Compare Original vs Optimized CPU implementations

echo "=== CPU Optimization Benchmark ==="
echo ""

run_test() {
    binary=$1
    threads=$2
    name=$3
    
    echo "Testing $name with $threads threads..."
    
    # Run in background with timeout
    ./$binary --ultra -f -j $threads -d data/cifar_tiles > bench_${name}_${threads}.log 2>&1 &
    PID=$!
    
    sleep 8
    
    kill -SIGINT $PID 2>/dev/null
    wait $PID 2>/dev/null
    
    avg_fps=$(grep "Avg FPS:" bench_${name}_${threads}.log | tail -n 1 | awk '{print $NF}')
    if [ -z "$avg_fps" ]; then avg_fps="N/A"; fi
    echo "  -> Result: $avg_fps FPS"
}

# 1. Baseline
echo "--- Baseline (Original) ---"
run_test "bin/video_mosaic_edge" 1 "baseline"
run_test "bin/video_mosaic_edge" 8 "baseline"

# 2. Optimized
echo ""
echo "--- Optimized (Full-Frame Edge Detection) ---"
run_test "bin/video_mosaic_edge_optimized" 1 "optimized"
run_test "bin/video_mosaic_edge_optimized" 8 "optimized"

echo ""
echo "=== Summary ==="
echo "Threads | Baseline FPS | Optimized FPS | Speedup"
echo "--------|--------------|---------------|--------"
fps_base_1=$(grep "Avg FPS:" bench_baseline_1.log | tail -n 1 | awk '{print $NF}')
fps_base_8=$(grep "Avg FPS:" bench_baseline_8.log | tail -n 1 | awk '{print $NF}')
fps_opt_1=$(grep "Avg FPS:" bench_optimized_1.log | tail -n 1 | awk '{print $NF}')
fps_opt_8=$(grep "Avg FPS:" bench_optimized_8.log | tail -n 1 | awk '{print $NF}')

if [ ! -z "$fps_base_1" ] && [ ! -z "$fps_opt_1" ]; then
    speedup_1=$(echo "scale=2; $fps_opt_1 / $fps_base_1" | bc)
    echo "   1    |      $fps_base_1      |       $fps_opt_1       |  ${speedup_1}x"
fi

if [ ! -z "$fps_base_8" ] && [ ! -z "$fps_opt_8" ]; then
    speedup_8=$(echo "scale=2; $fps_opt_8 / $fps_base_8" | bc)
    echo "   8    |      $fps_base_8      |       $fps_opt_8       |  ${speedup_8}x"
fi

rm bench_*.log
