# Using Different Tile Datasets

## Available Datasets

### 1. Pokemon Sprites (3,862 tiles)
- **Directory:** `pokemon_tiles`
- **Content:** Pokemon character sprites
- **Issue:** Some have inconsistent backgrounds (black/white)
- **Best for:** Fun demos, recognizable characters

### 2. CIFAR-10 (60,000 tiles) ⭐ RECOMMENDED
- **Directory:** `cifar_tiles`
- **Content:** Natural photos (animals, vehicles, objects, scenes)
- **Advantage:** Consistent backgrounds, diverse colors
- **Best for:** High-quality mosaics, professional results

## Usage

### Specify Tile Directory

```bash
# Use Pokemon tiles
./video_mosaic --ultra -f -j 8 --tiles pokemon_tiles
./video_mosaic_optimal --small -j 8 --tiles pokemon_tiles

# Use CIFAR-10 tiles (recommended)
./video_mosaic --ultra -f -j 8 --tiles cifar_tiles
./video_mosaic_optimal --small -j 8 --tiles cifar_tiles

# Short form
./video_mosaic -d cifar_tiles --ultra -f -j 8
```

### Current Tile Counts

```bash
Pokemon:  3,862 tiles
CIFAR-10: 60,000 tiles
```

## Recommendations

### For Real-Time Demo (Greedy)
```bash
./video_mosaic --ultra -f -j 8 -d cifar_tiles
```
- Fast (5-10ms processing)
- Smooth video
- Good quality with CIFAR-10

### For Quality Demo (Optimal)
```bash
./video_mosaic_optimal --small -j 8 -d cifar_tiles
```
- Slower (~3ms processing)
- No tile repetition
- Excellent quality
- Clear parallelization benefits

### For Benchmarking
```bash
# Test with different thread counts
for threads in 1 2 4 8; do
    echo "Testing with $threads threads:"
    ./video_mosaic_optimal --small -j $threads -d cifar_tiles
done
```

## Why CIFAR-10 is Better

✅ **Consistent backgrounds** - No black/white fluctuation  
✅ **Natural photos** - Real-world scenes  
✅ **60K variety** - Minimal repetition  
✅ **Diverse colors** - Better color matching  
✅ **Professional look** - Suitable for presentations  

## Quick Start

```bash
# Download CIFAR-10 (if not already done)
python3 download_cifar10.py

# Test with CIFAR-10
./video_mosaic_optimal --small -j 8 -d cifar_tiles

# Save a frame (press 's' during execution)
# Compare with Pokemon version
```
