# Makefile for Video Mosaic Project
# Cross-platform compatible (macOS, Linux)

CXX = g++
CXXFLAGS = -std=c++11 -O3
LDFLAGS = 

# Platform detection
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS (Homebrew)
    CXXFLAGS += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
    OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
    BOOST_FLAGS = -lboost_filesystem
else ifeq ($(UNAME_S),Linux)
    # Linux
    CXXFLAGS += -fopenmp
    LDFLAGS += -fopenmp
    OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
    BOOST_FLAGS = -lboost_filesystem -lboost_system
endif

# Targets
# Targets
# Targets
all: prep_dirs bin/video_mosaic bin/video_mosaic_optimal bin/video_mosaic_edge bin/video_mosaic_edge_optimized bin/video_mosaic_edge_metal bin/photomosaic_mac bin/webcam_test

# Preparation
prep_dirs:
	mkdir -p bin

# Real-time video mosaic (greedy, fast)
bin/video_mosaic: src/video_mosaic.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Optimal assignment mosaic (global optimization)
bin/video_mosaic_optimal: src/video_mosaic_optimal.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Edge-aware mosaic (structure-preserving matching)
bin/video_mosaic_edge: src/video_mosaic_edge.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Edge-aware mosaic (CPU OPTIMIZED)
bin/video_mosaic_edge_optimized: src/video_mosaic_edge_optimized.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# --- METAL GPU ACCELERATION ---

# 1. Compile Metal Kernel to .air
src/gpu/kernels.air: src/gpu/kernels.metal
	xcrun -sdk macosx metal -c $< -o $@

# 2. Link .air to .metallib
default.metallib: src/gpu/kernels.air
	xcrun -sdk macosx metallib $< -o $@

# 3. Compile Obj-C++ Wrapper
bin/metal_compute.o: src/gpu/metal_compute.mm src/gpu/metal_compute.h
	clang++ -c -std=c++17 -fobjc-arc -O3 $< -o $@

# 4. Main Executable (Links C++ + Obj-C++)
bin/video_mosaic_edge_metal: src/video_mosaic_edge_metal.cpp bin/metal_compute.o default.metallib
	$(CXX) $(CXXFLAGS) -fobjc-arc src/video_mosaic_edge_metal.cpp bin/metal_compute.o -o $@ \
	$(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS) \
	-framework Metal -framework Foundation -framework CoreGraphics

# -----------------

# Original photo mosaic
bin/photomosaic_mac: src/legacy/main.cpp src/legacy/exif.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Webcam test
bin/webcam_test: src/tests/webcam_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(LDFLAGS)

# Clean
clean:
	rm -rf bin src/gpu/*.air default.metallib bin/*.o

# Run video mosaic
run: bin/video_mosaic_edge
	./bin/video_mosaic_edge

# Help
help:
	@echo "Available targets:"
	@echo "  make all                 - Build all programs"
	@echo "  make bin/video_mosaic        - Build greedy video mosaic (fast)"
	@echo "  make bin/video_mosaic_optimal - Build optimal assignment mosaic"
	@echo "  make bin/video_mosaic_edge   - Build edge-aware mosaic"
	@echo "  make bin/photomosaic_mac     - Build original photo mosaic"
	@echo "  make bin/webcam_test         - Build webcam test"
	@echo "  make run                 - Build and run edge-aware video mosaic"
	@echo "  make clean               - Remove bin directory"
	@echo ""
	@echo "Platform: $(UNAME_S)"

.PHONY: all clean run help
