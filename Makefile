# Makefile for Video Mosaic Project

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
	# macOS (Homebrew) settings
	CXX = g++
	CXXFLAGS = -std=c++11 -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
	LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp
	OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
else
	# Linux / WSL / other Unix-like systems
	CXX = g++
	CXXFLAGS = -std=c++11 -O3 -fopenmp
	LDFLAGS = -fopenmp
	OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
endif

BOOST_FLAGS = -lboost_filesystem

# Targets
all: video_mosaic photomosaic_mac webcam_test

# Real-time video mosaic (NEW - optimized for streaming)
video_mosaic: video_mosaic.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Optimal assignment mosaic (NEW - global optimization)
video_mosaic_optimal: video_mosaic_optimal.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Original photo mosaic
photomosaic_mac: main.cpp exif.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_FLAGS) $(BOOST_FLAGS) $(LDFLAGS)

# Webcam test
webcam_test: webcam_test.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS) $(LDFLAGS)

# Clean
clean:
	rm -f video_mosaic photomosaic_mac webcam_test

# Run video mosaic
run: video_mosaic
	./video_mosaic

# Help
help:
	@echo "Available targets:"
	@echo "  make all          - Build all programs"
	@echo "  make video_mosaic - Build real-time video mosaic"
	@echo "  make photomosaic_mac - Build original photo mosaic"
	@echo "  make webcam_test  - Build webcam test"
	@echo "  make run          - Build and run video mosaic"
	@echo "  make clean        - Remove all binaries"

.PHONY: all clean run help
