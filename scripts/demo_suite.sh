#!/bin/bash

# Demo Suite for Parallel Video Mosaic
# Usage: ./scripts/demo_suite.sh

TILES_DIR="data/cifar_tiles"
if [ ! -d "$TILES_DIR" ]; then
    TILES_DIR="data/pokemon_tiles"
fi

function print_header() {
    clear
    echo "========================================================"
    echo "   Parallel Video Mosaic - Final Project Demo"
    echo "========================================================"
    echo " Tiles Directory: $TILES_DIR"
    echo "========================================================"
}

function check_build() {
    echo "Checking build status..."
    make all > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Build failed! Attempting to fix and rebuild..."
        make clean && make all
    else
        echo "Build OK."
    fi
}

while true; do
    print_header
    echo "1. [Baseline]  CPU Optimized (1 Thread) - Medium Grid"
    echo "   -> Shows single-threaded performance (approx 0.5 - 1 FPS)"
    echo ""
    echo "2. [Parallel]  CPU Optimized (8 Threads) - Medium Grid"
    echo "   -> Shows OpenMP scaling (approx 2 - 3 FPS)"
    echo ""
    echo "3. [Struggle]  CPU Optimized (8 Threads) - Ultra Grid (100x75)"
    echo "   -> Shows CPU hitting the wall at high resolution (< 1 FPS)"
    echo ""
    echo "4. [Solution]  Metal GPU Acceleration - Ultra Grid (100x75)"
    echo "   -> The 'Wow' factor. Real-time >10 FPS at high res."
    echo ""
    echo "5. [Benchmark] Run Full Automated Benchmark Experiment"
    echo "   -> Generates the data used for the report."
    echo ""
    echo "q. Quit"
    echo "========================================================"
    read -p "Select a demo (1-5 or q): " choice

    case $choice in
        1)
            echo "Running CPU (1 Thread)... Press 'q' in window to stop."
            ./bin/video_mosaic_edge_optimized --medium -j 1 -d "$TILES_DIR" --show-fps
            ;;
        2)
            echo "Running CPU (8 Threads)... Press 'q' in window to stop."
            ./bin/video_mosaic_edge_optimized --medium -j 8 -d "$TILES_DIR" --show-fps
            ;;
        3)
            echo "Running CPU Ultra (8 Threads)... Press 'q' in window to stop."
            ./bin/video_mosaic_edge_optimized --ultra -j 8 -d "$TILES_DIR" --show-fps
            ;;
        4)
            echo "Running Metal GPU (Ultra)... Press 'q' in window to stop."
            ./bin/video_mosaic_edge_metal --ultra -d "$TILES_DIR"
            ;;
        5)
            echo "Running Benchmarks... This will take about 2 minutes."
            python3 scripts/run_experiments.py
            read -p "Press Enter to continue..."
            ;;
        q)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo "Invalid option."
            sleep 1
            ;;
    esac
done
