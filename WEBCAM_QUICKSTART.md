# Webcam Capture - Quick Reference

## Test Your Webcam

```bash
# Compile
g++ -std=c++11 webcam_test.cpp -o webcam_test \
    $(pkg-config --cflags --libs opencv4)

# Run
./webcam_test
```

**Note**: On Mac, you may need to grant camera permissions to Terminal.

---

## Basic Webcam Code

```cpp
#include <opencv2/opencv.hpp>
using namespace cv;

int main() {
    VideoCapture cap(0);  // 0 = default camera
    
    if (!cap.isOpened()) {
        return -1;  // Camera failed
    }
    
    Mat frame;
    while (cap.read(frame)) {
        // Process frame here
        imshow("Webcam", frame);
        if (waitKey(1) == 'q') break;
    }
    
    return 0;
}
```

---

## Key OpenCV Functions

| Function | Purpose |
|----------|---------|
| `VideoCapture cap(0)` | Open camera (0 = first camera) |
| `cap.read(frame)` | Capture one frame |
| `cap.get(CAP_PROP_FPS)` | Get camera FPS |
| `cap.get(CAP_PROP_FRAME_WIDTH)` | Get frame width |
| `imshow("Window", frame)` | Display frame |
| `waitKey(1)` | Wait 1ms for key press |

---

## Performance Tips

1. **Reduce resolution**: `cap.set(CAP_PROP_FRAME_WIDTH, 640);`
2. **Measure FPS**: Use `chrono` to time each frame
3. **Skip frames**: Process every Nth frame if too slow
4. **Parallel processing**: Use OpenMP on mosaic generation

---

## Next: Integrate with Mosaic

See `VIDEO_MOSAIC_GUIDE.md` for full implementation plan.
