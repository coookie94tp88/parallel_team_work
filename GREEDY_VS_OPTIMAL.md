# Greedy vs Optimal Comparison Guide

## Two Implementations

### 1. Greedy (Fast) - `video_mosaic`
**Algorithm:** Each cell independently picks best tile
- **Speed:** Very fast (5ms processing)
- **Quality:** Good
- **Repetition:** High (same Pokemon appears many times)
- **Parallelization:** Easy, but limited speedup (2-3x)

### 2. Optimal (Quality) - `video_mosaic_optimal`
**Algorithm:** Global optimization with tile reuse constraints
- **Speed:** Slower (500-2000ms processing)
- **Quality:** Excellent
- **Repetition:** Controlled (max 1-5 uses per tile)
- **Parallelization:** Clear speedup (8-10x expected)

---

## Quick Comparison

```bash
# Greedy - Fast but repetitive
./video_mosaic --medium -f -j 8
# Processing: ~5ms | Pikachu appears 20+ times

# Optimal - Slower but better quality
./video_mosaic_optimal --medium -j 8
# Processing: ~500-1000ms | Each Pokemon max 1-2 times
```

---

## Recommended Test Configurations

### For Demo (Best Balance)
```bash
# Greedy
./video_mosaic --medium -f -j 8
# 40x30 grid, fast, good for real-time

# Optimal  
./video_mosaic_optimal --small -j 8
# 40x30 grid, shows clear speedup
```

### For Benchmarking
```bash
# Test different thread counts with optimal
./video_mosaic_optimal --small -j 1  # Baseline
./video_mosaic_optimal --small -j 2
./video_mosaic_optimal --small -j 4
./video_mosaic_optimal --small -j 8

# Watch the timing breakdown in terminal!
```

### For Screenshots
```bash
# Greedy - save a frame
./video_mosaic --medium -f -j 8
# Press 's' to save

# Optimal - save a frame  
./video_mosaic_optimal --medium -j 8
# Press 's' to save

# Compare the two images side-by-side!
```

---

## What to Observe

### Greedy Mode
- ✅ Very fast (real-time)
- ⚠️ Repetitive tiles
- ⚠️ Limited parallelization benefit

### Optimal Mode
- ✅ Better tile distribution
- ✅ Clear parallelization speedup
- ✅ No/minimal repetition
- ⚠️ Slower (not real-time)

---

## Performance Expectations

### Greedy (40×30 = 1200 cells)
```
1 thread:  ~8ms
8 threads: ~3ms
Speedup:   2.7x
```

### Optimal (40×30 = 1200 cells)
```
Cost matrix:    ~200ms (parallelized)
Assignment:     ~300ms (parallelized)  
Placement:      ~5ms (parallelized)
Total:          ~500ms

1 thread:  ~2000ms
8 threads: ~500ms
Speedup:   4x
```

---

## For Your Presentation

**Show both approaches:**

1. **Greedy** - "Fast real-time processing"
   - Demo live webcam
   - Show FPS counter
   - Explain trade-offs

2. **Optimal** - "Quality-focused with parallelization"
   - Process single frames
   - Show timing breakdown
   - Demonstrate speedup with different thread counts
   - Compare visual quality

**Key message:** Different algorithms for different goals!

---

## Command Reference

```bash
# Compile
make video_mosaic          # Greedy version
make video_mosaic_optimal  # Optimal version

# Run greedy
./video_mosaic [options]

# Run optimal
./video_mosaic_optimal [options]

# Common options
-w N, --width N       Grid width
-h N, --height N      Grid height  
-j N, --threads N     Thread count
-r N, --reuse N       Max tile reuse (optimal only)

# Presets
--small    40×30
--medium   60×45
--large    80×60
```
