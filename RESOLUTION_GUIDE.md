# Video Mosaic - Resolution & Quality Guide

## Quick Start

```bash
# Default (medium quality)
./video_mosaic

# High quality
./video_mosaic --high

# Ultra quality (slower but best)
./video_mosaic --ultra

# Fast mode for testing
./video_mosaic --low
```

## Preset Quality Modes

| Mode | Grid | Tile Size | Output | FPS | Quality |
|------|------|-----------|--------|-----|---------|
| `--low` | 20x15 | 16px | 320x240 | ~30 | Low (fast mode) |
| `--medium` | 40x30 | 32px | 1280x960 | ~15 | Good (default) |
| `--high` | 60x45 | 48px | 2880x2160 | ~8 | Excellent |
| `--ultra` | 80x60 | 64px | 5120x3840 | ~4 | Best |

## Custom Settings

```bash
# Custom resolution
./video_mosaic -w 60 -h 45

# Custom tile size
./video_mosaic -t 64

# More threads for speed
./video_mosaic -j 8

# Fast matching mode
./video_mosaic -f

# Combine options
./video_mosaic -w 80 -h 60 -t 48 -j 8 -a
```

## All Options

```
-w, --width N       Grid width (tiles horizontally)
-h, --height N      Grid height (tiles vertically)
-t, --tile-size N   Size of each tile in pixels
-j, --threads N     Number of OpenMP threads
-f, --fast          Fast mode (mean color matching)
-a, --accurate      Accurate mode (histogram, default)
--help              Show help message
```

## Performance Tips

**For maximum FPS:**
```bash
./video_mosaic --low -f -j 8
```

**For best quality:**
```bash
./video_mosaic --ultra -a -j 4
```

**Balanced (recommended):**
```bash
./video_mosaic --high -j 8
```

## Understanding the Trade-offs

### Grid Resolution
- **Smaller** (20x15): Faster, less detail
- **Larger** (80x60): Slower, more detail

### Tile Size
- **Smaller** (16px): Faster, Pokemon less visible
- **Larger** (64px): Slower, Pokemon more recognizable

### Matching Mode
- **Fast** (-f): Mean color, ~2x faster, basic accuracy
- **Accurate** (-a): Histogram, better quality

### Threads
- More threads = faster (up to CPU core count)
- Diminishing returns after 8 threads

## Recommended Configurations

### For Presentation Demo
```bash
./video_mosaic --high -j 8
# Good balance of quality and speed
```

### For Benchmarking
```bash
# Test different thread counts
./video_mosaic -j 1  # Serial baseline
./video_mosaic -j 2
./video_mosaic -j 4
./video_mosaic -j 8
```

### For Screenshots
```bash
./video_mosaic --ultra
# Best quality for saving frames (press 's')
```

## Output Resolution

Output size = Grid × Tile Size

Examples:
- 40x30 grid × 32px tiles = 1280x960 output
- 60x45 grid × 48px tiles = 2880x2160 output
- 80x60 grid × 64px tiles = 5120x3840 output (4K+!)
