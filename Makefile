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
all: prep_dirs bin/video_mosaic bin/video_mosaic_optimal bin/video_mosaic_edge bin/photomosaic_mac bin/webcam_test

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

# Original photo mosaic
bin/photomosaic_mac: src/legacy/main.cpp src/legacy/exif.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Webcam test
bin/webcam_test: src/tests/webcam_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(LDFLAGS)

# Clean
clean:
	rm -rf bin

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
