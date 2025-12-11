# Unified Command-Line Interface

Both `video_mosaic` and `video_mosaic_optimal` now share a consistent interface!

## Common Options

```bash
-d, --tiles DIR     # Tile directory (default: pokemon_tiles)
-j, --threads N     # Number of threads (default: 4)
-t, --tile-size N   # Tile size in pixels (default: 32)
-f, --show-fps      # Show FPS counter
--help              # Show help
```

## Preset Quality Modes

Both versions support the same presets:

| Preset | Grid Size | Cells | Description |
|--------|-----------|-------|-------------|
| `--small` | 40×30 | 1,200 | Fast, lower quality |
| `--medium` | 60×45 | 2,700 | **Default**, balanced |
| `--large` | 80×60 | 4,800 | High quality |
| `--ultra` | 100×75 | 7,500 | Maximum quality |

## Optimal-Only Options

```bash
-r, --reuse N       # Max tile reuse (default: 1)
-c, --coherence     # Enable temporal coherence
```

## Examples

```bash
# Default settings (both versions)
./video_mosaic -d cifar_tiles
./video_mosaic_optimal -d cifar_tiles

# Small grid, 8 threads (both versions)
./video_mosaic --small -j 8 -d cifar_tiles
./video_mosaic_optimal --small -j 8 -d cifar_tiles

# Large grid with coherence (optimal only)
./video_mosaic_optimal --large -j 8 -d cifar_tiles -c

# Ultra quality (both versions)
./video_mosaic --ultra -j 8 -d cifar_tiles
./video_mosaic_optimal --ultra -j 8 -d cifar_tiles
```

## Quick Comparison

| Command | Speed | Quality | Tile Reuse |
|---------|-------|---------|------------|
| `video_mosaic` | ⚡ Very Fast (~20ms) | ✓ Good | Unlimited |
| `video_mosaic_optimal` | 🐢 Moderate (~120ms) | ✓✓ Excellent | Limited |

Both now use **3x3 region-based matching** for better quality!
