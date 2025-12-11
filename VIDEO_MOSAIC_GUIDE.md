# Real-Time Video Mosaic Implementation Guide

## Quick Test: Webcam Capture

First, verify your webcam works:

```bash
# Compile the test
g++ -std=c++11 webcam_test.cpp -o webcam_test \
    $(pkg-config --cflags --libs opencv4)

# Run it
./webcam_test
```

**Controls:**
- Press `q` to quit
- Press `s` to save a frame

---

## Integrating Webcam with Mosaic

### Approach 1: Simple Frame-by-Frame (Start Here)

Modify your `main.cpp` to accept webcam input:

```cpp
// Instead of:
Mat3b src = imread(p.input_image);

// Use:
VideoCapture cap(0);  // 0 = default webcam
Mat3b frame;
while (cap.read(frame)) {
    // Process frame as mosaic
    // Display result
}
```

### Approach 2: Real-Time Pipeline (Advanced)

Use threading for parallel capture/process/display:

```
Thread 1: Capture frames from webcam
Thread 2: Generate mosaic
Thread 3: Display output
```

---

## Performance Targets

For real-time video mosaic:

| Target FPS | Frame Time | Difficulty |
|------------|------------|------------|
| 10 FPS | 100ms | Easy |
| 15 FPS | 67ms | Moderate |
| 24 FPS | 42ms | Hard |
| 30 FPS | 33ms | Very Hard |

**Start with 10 FPS**, then optimize!

---

## Optimization Strategies

### 1. Reduce Tile Count
- Use 100-200 tiles instead of 500
- Faster matching, still looks good

### 2. Reduce Resolution
- Resize webcam frame to 320x240 or 640x480
- Smaller grid = fewer matches needed

### 3. Reuse Tiles Between Frames
- Cache previous frame's tile assignments
- Only update tiles where image changed significantly
- **Temporal coherence** - reduces flickering

### 4. Parallelize Tile Matching
- Use OpenMP on the matching loop
- Each thread handles subset of grid cells

### 5. Precompute Everything
- Tile features already computed (index file)
- Webcam frames are the only new input

---

## Implementation Steps

### Step 1: Static Video File (Easier)

Process a pre-recorded video first:

```cpp
VideoCapture cap("input_video.mp4");
VideoWriter out("output_mosaic.mp4", 
                VideoWriter::fourcc('M','J','P','G'),
                30, Size(width, height));

while (cap.read(frame)) {
    Mat mosaic = generateMosaic(frame, tiles);
    out.write(mosaic);
}
```

### Step 2: Live Webcam (Harder)

```cpp
VideoCapture cap(0);
Mat frame, mosaic;

while (cap.read(frame)) {
    auto start = chrono::high_resolution_clock::now();
    
    mosaic = generateMosaic(frame, tiles);
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    // Display FPS
    double fps = 1000.0 / duration.count();
    putText(mosaic, "FPS: " + to_string(fps), ...);
    
    imshow("Mosaic", mosaic);
    if (waitKey(1) == 'q') break;
}
```

### Step 3: Add OpenMP Parallelization

```cpp
// In your matching loop (main.cpp line 366-380)
#pragma omp parallel for collapse(2)
for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
        Vec3b color = src(i, j);
        ImageMean best_match = nearestImage(index, color, forbidden);
        // ... rest of matching code
    }
}
```

---

## Benchmarking Plan

Test with different configurations:

| Config | Tiles | Resolution | Grid Size | Target FPS |
|--------|-------|------------|-----------|------------|
| Fast | 100 | 320x240 | 20x15 | 30 |
| Balanced | 200 | 640x480 | 40x30 | 15 |
| Quality | 500 | 640x480 | 64x48 | 10 |

Measure:
- Serial baseline FPS
- OpenMP FPS (1, 2, 4, 8 threads)
- Speedup and efficiency

---

## Next Steps

1. **Test webcam** - Run `webcam_test` to verify camera works
2. **Benchmark static mosaic** - How fast is one frame?
3. **Add OpenMP** - Parallelize tile matching
4. **Extend to video** - Process video file frame-by-frame
5. **Add webcam input** - Real-time capture
6. **Optimize** - Hit target FPS

---

## Troubleshooting

**"Could not open webcam"**
- Check camera permissions (System Preferences → Security & Privacy → Camera)
- Try different camera index: `VideoCapture cap(1);`
- On Mac, you might need to grant terminal camera access

**"Too slow for real-time"**
- Reduce tile count
- Reduce resolution
- Add more parallelization
- Use temporal coherence (reuse tiles)

**"Flickering tiles"**
- Implement temporal coherence
- Add hysteresis (only change tile if new one is much better)
- Smooth transitions between frames
