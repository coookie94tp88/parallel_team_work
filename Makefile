# Makefile for Video Mosaic Project
# Cross-platform compatible (macOS, Linux)

CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp
LDFLAGS = -fopenmp

# Platform detection
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS (Homebrew)
    CXXFLAGS += -Xpreprocessor -I/opt/homebrew/opt/libomp/include
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
    OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
    BOOST_FLAGS = -lboost_filesystem
else ifeq ($(UNAME_S),Linux)
    # Linux
    OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
    BOOST_FLAGS = -lboost_filesystem -lboost_system
endif

# Targets
all: video_mosaic video_mosaic_optimal video_mosaic_edge photomosaic_mac webcam_test

# Real-time video mosaic (greedy, fast)
video_mosaic: video_mosaic.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Optimal assignment mosaic (global optimization)
video_mosaic_optimal: video_mosaic_optimal.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Edge-aware mosaic (structure-preserving matching)
video_mosaic_edge: video_mosaic_edge.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Original photo mosaic
photomosaic_mac: main.cpp exif.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Webcam test
webcam_test: webcam_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(LDFLAGS)

# Clean
clean:
	rm -f video_mosaic video_mosaic_optimal video_mosaic_edge photomosaic_mac webcam_test

# Run video mosaic
run: video_mosaic
	./video_mosaic

# Help
help:
	@echo "Available targets:"
	@echo "  make all                 - Build all programs"
	@echo "  make video_mosaic        - Build greedy video mosaic (fast)"
	@echo "  make video_mosaic_optimal - Build optimal assignment mosaic"
	@echo "  make video_mosaic_edge   - Build edge-aware mosaic"
	@echo "  make photomosaic_mac     - Build original photo mosaic"
	@echo "  make webcam_test         - Build webcam test"
	@echo "  make run                 - Build and run video mosaic"
	@echo "  make clean               - Remove all binaries"
	@echo ""
	@echo "Platform: $(UNAME_S)"

.PHONY: all clean run help
