# Cross-Platform Compatibility Guide

## Current Status

### ✅ **macOS (Apple Silicon & Intel)**
- **Fully Supported**
- Uses Homebrew OpenMP and OpenCV
- SIMD: ARM NEON (M1/M2) or x86 SSE (Intel)

### ✅ **Linux (Ubuntu/Debian)**
- **Fully Compatible**
- Requires: `g++`, `libomp-dev`, `libopencv-dev`, `libboost-filesystem-dev`
- SIMD: x86 SSE/AVX

### ⚠️ **Windows**
- **Needs Adjustments**
- Use Visual Studio or MinGW
- Replace Boost.Filesystem with `std::filesystem` (C++17)

---

## Platform-Specific Setup

### macOS

```bash
# Install dependencies
brew install opencv boost libomp

# Build
make all

# Run
./video_mosaic_edge --medium -j 8 -d pokemon_tiles
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    g++ \
    libomp-dev \
    libopencv-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    pkg-config

# Build
make all

# Run
./video_mosaic_edge --medium -j 8 -d pokemon_tiles
```

### Linux (Fedora/RHEL)

```bash
# Install dependencies
sudo dnf install -y \
    gcc-c++ \
    libomp-devel \
    opencv-devel \
    boost-filesystem \
    boost-system \
    pkgconfig

# Build
make all
```

---

## Code Portability

### ✅ **Portable Components**

1. **C++11 Standard** - Works everywhere
2. **OpenMP** - Available on all major platforms
3. **OpenCV** - Cross-platform library
4. **SIMD** - Conditional compilation for ARM/x86

### ⚠️ **Platform-Specific Code**

#### Boost.Filesystem (Currently Used)
```cpp
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
```

**Issue**: Requires Boost library installation

**Solution for C++17**: Replace with `std::filesystem`
```cpp
#include <filesystem>
namespace fs = std::filesystem;
```

#### OpenMP Flags
- **macOS**: Requires `-Xpreprocessor -fopenmp` and explicit library paths
- **Linux**: Just `-fopenmp` works out of the box

---

## Making Code More Portable

### Option 1: Use C++17 (Recommended)

Replace Boost.Filesystem with std::filesystem:

```cpp
// Change this:
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

// To this:
#include <filesystem>
namespace fs = std::filesystem;
```

**Benefits**:
- No external Boost dependency
- Standard C++ (works everywhere)
- Simpler build process

**Drawbacks**:
- Requires C++17 compiler
- Change Makefile: `-std=c++11` → `-std=c++17`

### Option 2: Keep Current Setup

**Pros**:
- C++11 compatible (older compilers)
- Already working

**Cons**:
- Requires Boost installation
- Extra dependency

---

## Testing on Linux

If you want to test on Linux, here's a quick Docker test:

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    g++ make pkg-config \
    libomp-dev libopencv-dev \
    libboost-filesystem-dev libboost-system-dev
WORKDIR /app
COPY . .
RUN make video_mosaic_edge
CMD ["./video_mosaic_edge", "--help"]
EOF

# Build and test
docker build -t video-mosaic-test .
docker run video-mosaic-test
```

---

## Recommendation

**For maximum portability**, I suggest:

1. **Upgrade to C++17** and use `std::filesystem`
2. **Keep OpenMP** (widely available)
3. **Keep OpenCV** (cross-platform)
4. **Keep SIMD** (conditional compilation already handles ARM/x86)

This would eliminate the Boost dependency and make the code work on any platform with just:
- C++17 compiler
- OpenMP
- OpenCV

Want me to make these changes?
