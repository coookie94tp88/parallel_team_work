# Video Mosaic - Complete Documentation

A high-performance parallel video mosaic generator using OpenMP and SIMD optimization.

**Quick Start:** `./video_mosaic_optimal --small -j 8 -d cifar_tiles`

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Algorithms](#algorithms)
7. [Performance](#performance)
8. [Tile Datasets](#tile-datasets)
9. [Optimization Details](#optimization-details)
10. [Cross-Platform Support](#cross-platform-support)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This project creates real-time video mosaics from webcam input, where each frame is reconstructed using thousands of small tile images. Two implementations are provided:

- **Greedy** (`video_mosaic`) - Fast, real-time performance
- **Optimal** (`video_mosaic_optimal`) - Higher quality, no tile repetition

### Key Features

✅ **Real-time processing** - Up to 30 FPS with greedy algorithm  
✅ **Parallel processing** - OpenMP multi-threading  
✅ **SIMD optimization** - ARM NEON / x86 SSE vectorization  
✅ **60K+ tiles** - CIFAR-10 natural image dataset  
✅ **Temporal coherence** - Stable output for static scenes  
✅ **Flexible configuration** - Command-line options for everything  

---

## Quick Start

### 1. Install Dependencies

**macOS:**
```bash
brew install opencv libomp boost
```

**Linux:**
```bash
sudo apt install libopencv-dev libomp-dev libboost-filesystem-dev
```

### 2. Download Tiles

```bash
python3 download_cifar10.py
```

### 3. Compile

```bash
make all
```

### 4. Run

**Fast (Greedy):**
```bash
./video_mosaic --ultra -f -j 8 -d cifar_tiles
```

**Quality (Optimal):**
```bash
./video_mosaic_optimal --small -j 8 -d cifar_tiles -c
```

---

## Features

### Two Algorithms

| Feature | Greedy | Optimal |
|---------|--------|---------|
| **Speed** | Very fast (~5ms) | Moderate (~165ms) |
| **Quality** | Good | Excellent |
| **Tile Repetition** | Allowed | Minimized (max 1x) |
| **Use Case** | Real-time demo | Quality screenshots |
| **FPS (80×60)** | ~25 FPS | ~6 FPS |

### Optimizations

- **OpenMP Parallelization** - Multi-threaded processing
- **SIMD Vectorization** - ARM NEON / x86 SSE
- **Dynamic Scheduling** - Better load balancing
- **Temporal Coherence** - Stable output when still
- **Parallel Tile Loading** - 6x faster startup

---

## Installation

### Prerequisites

- C++11 compiler (GCC 8+, Clang 7+, MSVC 2017+)
- OpenCV 4.x
- OpenMP
- Boost Filesystem (or C++17 std::filesystem)

### Build

```bash
# Compile both versions
make all

# Compile individually
make video_mosaic          # Greedy version
make video_mosaic_optimal  # Optimal version

# Clean
make clean
```

---

## Usage

### Command-Line Options

**Common Options:**
```
-d, --tiles DIR      Tile directory (default: pokemon_tiles)
-j, --threads N      Number of threads (default: 4)
-t, --tile-size N    Tile size in pixels (default: 32)
-f, --show-fps       Show FPS counter
```

**Quality Presets:**
```
--small    40×30 grid  (1,200 cells)  - Fast
--medium   60×45 grid  (2,700 cells)  - Balanced
--large    80×60 grid  (4,800 cells)  - High quality
--ultra    100×75 grid (7,500 cells)  - Maximum quality
```

**Optimal-Only Options:**
```
-c, --coherence      Enable temporal coherence (stable when still)
-r, --reuse N        Max tile reuse (default: 1)
```

### Examples

**Real-time demo (greedy):**
```bash
./video_mosaic --ultra -f -j 8 -d cifar_tiles
```

**High-quality stable output (optimal):**
```bash
./video_mosaic_optimal --large -j 8 -d cifar_tiles -c
```

**Small grid for testing:**
```bash
./video_mosaic_optimal --small -j 4 -d cifar_tiles
```

**Save a frame:**
Press `s` while running to save current frame as `mosaic_YYYYMMDD_HHMMSS.png`

**Quit:**
Press `q` or `Ctrl+C`

---

## Algorithms

### Greedy Algorithm

**How it works:**
1. Resize input frame to grid size
2. For each cell, find best matching tile (parallel)
3. Place tiles to create mosaic

**Matching:** Euclidean distance in RGB space

**Time Complexity:** O(cells × tiles) with parallelization

**Performance:** ~5ms for 2,700 cells with 8 threads

### Optimal Algorithm

**How it works:**
1. Build cost matrix (all cell-tile distances) - **Parallelized**
2. Solve assignment problem with constraints - **Parallelized**
3. Place tiles to create mosaic - **Parallelized**

**Matching:** Greedy approximation with tile reuse limits

**Time Complexity:** O(cells × tiles) with parallelization

**Performance:** ~165ms for 4,800 cells with 8 threads

**Key Difference:** Optimal minimizes tile repetition globally

---

## Performance

### Benchmarks (80×60 grid, 60K tiles, 8 threads)

**Greedy:**
```
Mosaic generation: 5ms
FPS: ~25 FPS (limited by webcam)
```

**Optimal (with SIMD):**
```
Cost matrix:  120ms  (1.46x speedup from NEON)
Assignment:    44ms
Placement:      1ms
Total:        165ms  (~6 FPS)
```

### Optimization Results

**Before optimization:** 217ms per frame  
**After SIMD + dynamic scheduling:** 165ms per frame  
**Speedup:** 1.32x (32% faster)

### Scalability

| Threads | Cost Matrix | Assignment | Total | Speedup |
|---------|-------------|------------|-------|---------|
| 1       | 175ms       | 100ms      | 275ms | 1.0x    |
| 2       | 95ms        | 55ms       | 150ms | 1.8x    |
| 4       | 55ms        | 30ms       | 85ms  | 3.2x    |
| 8       | 120ms       | 44ms       | 165ms | 1.7x    |

*Note: 8 threads show memory bandwidth saturation*

---

## Tile Datasets

### CIFAR-10 (Recommended)

**Size:** 60,000 images  
**Resolution:** 32×32 pixels  
**Content:** Natural photos (animals, vehicles, etc.)  
**Advantages:** Consistent backgrounds, high variety

**Download:**
```bash
python3 download_cifar10.py
```

**Use:**
```bash
./video_mosaic_optimal -d cifar_tiles -j 8
```

### Pokemon Sprites

**Size:** 3,862 images  
**Resolution:** 96×96 pixels (resized to 32×32)  
**Content:** Pokemon character sprites  
**Advantages:** Colorful, recognizable

**Download:**
```bash
python3 download_pokemon.py
```

**Use:**
```bash
./video_mosaic_optimal -d pokemon_tiles -j 8
```

### Switching Datasets

```bash
./switch_tiles.sh cifar    # Use CIFAR-10
./switch_tiles.sh pokemon  # Use Pokemon
```

---

## Optimization Details

### SIMD Vectorization

**ARM NEON (Apple Silicon):**
```cpp
// Process 4 color distances simultaneously
float32x4_t target_b = vdupq_n_f32(target[0]);
float32x4_t diff_b = vsubq_f32(target_b, tile_b);
float32x4_t sq_b = vmulq_f32(diff_b, diff_b);
```

**x86 SSE (Intel/AMD):**
```cpp
// Process 4 color distances simultaneously
__m128 target_b = _mm_set1_ps(target[0]);
__m128 diff_b = _mm_sub_ps(target_b, tile_b);
__m128 sq_b = _mm_mul_ps(diff_b, diff_b);
```

**Speedup:** 1.46x on cost matrix computation

### Dynamic Scheduling

```cpp
#pragma omp parallel for schedule(dynamic, 64)
```

**Benefit:** Better load balancing, reduces thread idle time

### Temporal Coherence

**Feature:** Reuses previous frame's assignment if input unchanged

**Benefit:** Stable output when camera is still, faster processing

**Enable:** Use `-c` or `--coherence` flag

---

## Cross-Platform Support

### Supported Platforms

✅ **macOS ARM64** (Apple Silicon) - NEON SIMD  
✅ **macOS x86-64** (Intel) - SSE SIMD  
✅ **Linux x86-64** - SSE SIMD  
✅ **Windows x86-64** - SSE SIMD (requires CMake)  
⚠️ **Other architectures** - Scalar fallback

### Porting to Other Platforms

**Windows:**
1. Install OpenCV (vcpkg)
2. Use CMake instead of Makefile
3. Replace `boost::filesystem` with `std::filesystem`

**RISC-V / PowerPC:**
1. Code runs with scalar fallback
2. Optional: Add architecture-specific SIMD

**WebAssembly:**
1. Use Emscripten
2. Limited threading support
3. Use WASM SIMD128

---

## Troubleshooting

### Webcam Not Found

**Error:** `Cannot open webcam`

**Solution:**
- Check camera permissions in System Preferences (macOS)
- Try different camera index: modify `VideoCapture(0)` in code
- Test with `ls /dev/video*` (Linux)

### Slow Performance

**Issue:** Low FPS

**Solutions:**
1. Use smaller grid: `--small` instead of `--ultra`
2. Increase threads: `-j 8` (match CPU cores)
3. Use greedy version for real-time: `./video_mosaic`
4. Enable coherence: `-c` (optimal only)

### Compilation Errors

**Error:** `opencv not found`

**Solution:**
```bash
# macOS
brew install opencv

# Linux
sudo apt install libopencv-dev
```

**Error:** `omp.h not found`

**Solution:**
```bash
# macOS
brew install libomp

# Linux
sudo apt install libomp-dev
```

### Memory Issues

**Error:** `std::bad_alloc` or crash

**Solution:**
- Use fewer tiles (smaller dataset)
- Reduce grid size (`--small`)
- Close other applications

---

## Project Structure

```
.
├── video_mosaic.cpp              # Greedy implementation
├── video_mosaic_optimal.cpp      # Optimal implementation
├── Makefile                      # Build configuration
├── download_cifar10.py           # CIFAR-10 downloader
├── download_pokemon.py           # Pokemon downloader
├── switch_tiles.sh               # Dataset switcher
├── benchmark.sh                  # Performance benchmarking
├── cifar_tiles/                  # CIFAR-10 dataset (60K images)
├── pokemon_tiles/                # Pokemon dataset (3.8K images)
└── README.md                     # This file
```

---

## Performance Tips

### For Maximum FPS

1. **Use greedy algorithm** - 5x faster than optimal
2. **Use small grid** - `--small` (40×30)
3. **Maximize threads** - `-j 8` (or your CPU core count)
4. **Use fast matching** - Mean color (default in greedy)

### For Maximum Quality

1. **Use optimal algorithm** - No tile repetition
2. **Use large grid** - `--large` or `--ultra`
3. **Enable coherence** - `-c` for stable output
4. **Use CIFAR-10** - 60K diverse natural images

### For Balanced Performance

1. **Use optimal with medium grid** - `--medium`
2. **8 threads** - `-j 8`
3. **Enable coherence** - `-c`
4. **CIFAR-10 dataset** - `-d cifar_tiles`

**Recommended command:**
```bash
./video_mosaic_optimal --medium -j 8 -d cifar_tiles -c
```

---

## Technical Details

### SIMD Architecture Detection

```cpp
#if defined(__ARM_NEON) || defined(__aarch64__)
    // ARM NEON (Apple Silicon)
#elif defined(__SSE__)
    // x86 SSE (Intel/AMD)
#else
    // Scalar fallback
#endif
```

### Parallelization Strategy

1. **Tile Loading** - Parallel file I/O and preprocessing
2. **Cost Matrix** - Parallel distance computation
3. **Assignment** - Parallel greedy selection with atomics
4. **Tile Placement** - Parallel image copying

### Memory Layout

**Current:** Array of Structs (AoS)
```cpp
struct Tile {
    Mat image;
    Vec3f mean_color;
    Mat histogram;
};
vector<Tile> tiles;
```

**Future:** Struct of Arrays (SoA) for better cache locality

---

## Contributing

This is a course project for Parallel Programming. Contributions are welcome!

**Areas for improvement:**
- GPU acceleration (CUDA/Metal)
- Video file input/output
- More SIMD architectures (AVX2, RVV)
- Better assignment algorithms
- UI improvements

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **CIFAR-10 Dataset** - Alex Krizhevsky, Vinod Nair, Geoffrey Hinton
- **Pokemon Sprites** - PokeAPI (https://pokeapi.co/)
- **OpenCV** - Open Source Computer Vision Library
- **OpenMP** - Multi-platform shared-memory parallel programming

---

## Contact

For questions or issues, please open an issue on GitHub.

**Performance Stats:**
- Greedy: ~25 FPS (real-time)
- Optimal: ~6 FPS (high quality)
- Tiles: 60,000 (CIFAR-10)
- Parallelization: OpenMP + SIMD
- Speedup: 1.32x with optimizations
