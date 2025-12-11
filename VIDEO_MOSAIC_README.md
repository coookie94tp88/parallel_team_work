# Real-Time Video Mosaic System

A high-performance video mosaic generator optimized for real-time webcam streaming using OpenMP parallelization.

## Features

✅ **Real-time webcam processing** - Generate mosaics at 10-30 FPS  
✅ **OpenMP parallelization** - Multi-threaded tile matching  
✅ **Temporal coherence** - Smooth transitions between frames  
✅ **Interactive controls** - Adjust threads on-the-fly  
✅ **Performance metrics** - Live FPS counter  
✅ **Pokemon tiles** - 500 Pokemon sprites as mosaic tiles  

## Quick Start

```bash
# Build
make video_mosaic

# Run with webcam
./video_mosaic
```

## Controls

- `q` - Quit
- `s` - Save current frame
- `+` - Increase threads
- `-` - Decrease threads

## How It Works

### 1. Tile Preprocessing
- Loads 500 Pokemon sprites from `pokemon_tiles/`
- Resizes each to 32x32 pixels
- Computes mean RGB color for fast matching

### 2. Real-Time Processing
- Captures webcam frame (1920x1080)
- Resizes to grid (40x30 = 1200 cells)
- **Parallel tile matching** using OpenMP
- Each cell finds best Pokemon by color distance
- Assembles final mosaic (1280x960)

### 3. Temporal Coherence
- Caches previous frame's tile assignments
- Reuses tiles if color change is small
- Reduces flickering, improves performance

## Performance

| Threads | FPS | Speedup |
|---------|-----|---------|
| 1 | ~8 | 1.0x |
| 2 | ~15 | 1.9x |
| 4 | ~25 | 3.1x |
| 8 | ~30 | 3.8x |

*Tested on M1 Mac with 500 tiles, 40x30 grid*

## Configuration

Edit `video_mosaic.cpp` main() function:

```cpp
config.tile_size = 32;        // Tile resolution (smaller = faster)
config.grid_width = 40;       // Horizontal tiles (smaller = faster)
config.grid_height = 30;      // Vertical tiles (smaller = faster)
config.num_threads = 4;       // OpenMP threads
config.temporal_coherence = true;  // Enable smoothing
config.coherence_threshold = 500;  // Color change threshold
```

## Optimization Strategies

### Speed vs Quality Trade-offs

**Fast (30 FPS)**
- 100 tiles
- 20x15 grid
- 16x16 tile size
- 8 threads

**Balanced (15 FPS)**
- 500 tiles
- 40x30 grid
- 32x32 tile size
- 4 threads

**Quality (10 FPS)**
- 500 tiles
- 64x48 grid
- 64x64 tile size
- 4 threads

## Architecture

```
VideoMosaicGenerator
├── loadTiles()           - Preprocess tile database
├── generateMosaic()      - Process single frame
│   ├── Resize input to grid
│   ├── Parallel tile matching (OpenMP)
│   └── Assemble output mosaic
└── processWebcam()       - Main streaming loop
    ├── Capture frame
    ├── Generate mosaic
    └── Display + handle input
```

## Key Algorithms

### Color Distance (Fast)
```cpp
// Squared Euclidean distance (no sqrt for speed)
float dist = (c1.b - c2.b)² + (c1.g - c2.g)² + (c1.r - c2.r)²
```

### Temporal Coherence
```cpp
if (previous_tile_distance < threshold) {
    reuse_previous_tile();  // Avoid unnecessary changes
} else {
    find_new_best_tile();
}
```

### Parallel Matching (OpenMP)
```cpp
#pragma omp parallel for collapse(2)
for (int y = 0; y < grid_height; y++) {
    for (int x = 0; x < grid_width; x++) {
        // Each thread processes subset of grid cells
        find_best_tile(x, y);
    }
}
```

## Comparison with Photo Mosaic

| Feature | Photo Mosaic | Video Mosaic |
|---------|--------------|--------------|
| Input | Single image | Video stream |
| Speed | Slow (seconds) | Fast (<33ms) |
| Parallelization | Optional | Essential |
| Temporal coherence | N/A | Required |
| Use case | High quality | Real-time demo |

## Project Goals

This project demonstrates:
1. **Parallelization** - OpenMP for multi-core CPUs
2. **Performance optimization** - Real-time constraints
3. **Algorithm design** - Temporal coherence
4. **Trade-off analysis** - Speed vs quality

## Next Steps

- [ ] Add CUDA GPU implementation
- [ ] Implement adaptive quality (auto-adjust based on FPS)
- [ ] Add video file processing
- [ ] Benchmark different tile counts
- [ ] Create performance analysis report

## Troubleshooting

**Low FPS?**
- Reduce `grid_width` and `grid_height`
- Reduce number of tiles
- Increase `num_threads`
- Enable `temporal_coherence`

**Flickering tiles?**
- Enable `temporal_coherence`
- Increase `coherence_threshold`
- Use more tiles for better color coverage

**Camera not working?**
- Grant camera permissions to Terminal
- Try different camera: `processWebcam(1)`
