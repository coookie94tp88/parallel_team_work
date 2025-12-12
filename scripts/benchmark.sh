#!/bin/bash
# Benchmark script to measure FPS with different thread counts

echo "=== Video Mosaic Performance Benchmark ==="
echo ""
echo "This will run the mosaic for 10 seconds with different thread counts"
echo "Watch the FPS counter in the window (green text at top)"
echo ""

CONFIGS=(
    "1:Serial (1 thread)"
    "2:2 threads"
    "4:4 threads"
    "8:8 threads (full parallel)"
)

for config in "${CONFIGS[@]}"; do
    threads="${config%%:*}"
    desc="${config#*:}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $desc"
    echo "Command: ./video_mosaic --ultra -f -j $threads"
    echo ""
    echo "Watch the FPS counter in the window!"
    echo "Press 'q' to continue to next test..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    ./video_mosaic --ultra -f -j $threads
    
    echo ""
    echo "Test completed. Moving to next configuration..."
    echo ""
    sleep 1
done

echo ""
echo "=== Benchmark Complete ==="
echo ""
echo "Summary:"
echo "- 1 thread should show lowest FPS"
echo "- 8 threads should show highest FPS"
echo "- Speedup = FPS(8 threads) / FPS(1 thread)"
