# Parallel Video Mosaic Generation

A high-performance video mosaic generator that reconstructs a live webcam feed using a database of 60,000 tile images (CIFAR-10) in real-time. This project demonstrates massive parallelism using **OpenMP (CPU)** and **Metal (GPU)** on Apple Silicon.

**Quick Start (Metal GPU):**
```bash
./video_mosaic_metal --ultra -d data/cifar_tiles
```

---

## Key Features

*   ✅ **Real-time 1080p Processing**: Achieves ~10 FPS on "Ultra" grid ($100 \times 75$) using Metal, compared to <1 FPS on CPU.
*   ✅ **Edge-Aware Matching**: Uses a custom metric (RGB + Sobel Gradients) to preserve structural details like edges and shapes.
*   ✅ **GPU Acceleration**: Custom Metal compute kernels leveraging Apple Silicon's Unified Memory Architecture.
*   ✅ **CPU Optimization**: OpenMP multi-threading and global feature extraction for non-Metal platforms.
*   ✅ **Massive Search**: Performs ~450 million comparisons per frame (7,500 cells $\times$ 60,000 tiles).

---

## Installation

### Prerequisites

*   **macOS (Recommended)**: Xcode Command Line Tools (for Metal support).
*   **Dependencies**: CMake, OpenCV, OpenMP, Boost.

**Install via Homebrew:**
```bash
brew install cmake opencv libomp boost
```

### Build

We use **CMake** for the build system.

1.  **Generate Build Files**:
    ```bash
    cmake .
    ```

2.  **Compile**:
    ```bash
    make
    ```

This will produce two executables:
*   `video_mosaic_metal`: GPU-accelerated version (macOS only).
*   `video_mosaic_cpu`: CPU-optimized version (Cross-platform).

---

## Usage

### 1. Download Tile Database (Required)
First, download the CIFAR-10 dataset to use as source tiles:
```bash
python3 download_cifar10.py
```
*This creates a `data/cifar_tiles` directory.*

### 2. Run Video Mosaic

**Run GPU Version (Recommended):**
```bash
# Ultra Quality (100x75 grid) - ~10 FPS
./video_mosaic_metal --ultra -d data/cifar_tiles

# Medium Quality (60x45 grid) - ~10 FPS
./video_mosaic_metal --medium -d data/cifar_tiles
```

**Run CPU Version (Benchmark):**
```bash
# Optimized CPU (8 threads)
./video_mosaic_cpu -j 8 --medium -d data/cifar_tiles
```

### Command-Line Options
*   `-d, --tiles DIR`: Directory containing tile images.
*   `--medium`: Use medium grid ($60 \times 45$).
*   `--ultra`: Use ultra grid ($100 \times 75$).
*   `-j N`: Number of CPU threads (CPU version only).

---

## Performance Benchmarks

Tested on MacBook Pro (M3 Pro):

| Workload | CPU (1 Thread) | CPU (8 Threads) | **Metal GPU** | **Speedup (vs CPU 1T)** |
| :--- | :--- | :--- | :--- | :--- |
| **Medium Grid ($60 \times 45$)** | 0.78 FPS | 3.09 FPS | **9.95 FPS** | **12.8x** |
| **Ultra Grid ($100 \times 75$)** | ~0.5 FPS | 1.21 FPS | **9.59 FPS** | **~19x** |

---

## Project Structure

*   `src/apps/`: Application entry points (`video_mosaic_metal.cpp`, `video_mosaic_cpu.cpp`).
*   `src/common/`: Shared logic (`VideoMosaicGenerator`, `TileDatabase`).
*   `src/gpu/`: Metal kernels (`kernels.metal`) and bridge code (`metal_compute.mm`).
*   `include/`: Header files.
*   `report/`: Project report and documentation.

## Authors

*   **B12902046 廖昀陽**: GPU Implementation (Metal), Build System.
*   **B12902131 陳柏宇**: CPU Optimization, Benchmarking.
*   **B13902055 薛閔澤**: Report, Coordination.
