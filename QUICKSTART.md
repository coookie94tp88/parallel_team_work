# Quick Start: Pokemon Mosaic Demo

## ✅ What's Ready
- ✓ 500 Pokemon sprites downloaded (`pokemon_tiles/`)
- ✓ Configuration file created (`settings_pokemon.ini`)
- ✓ Source code ready (`main.cpp`, `exif.cpp`)

## ❌ What You Need to Install

### 1. Install Dependencies (Mac)
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenCV and Boost
brew install opencv boost pkg-config
```

### 2. Compile the Program
```bash
# Navigate to project folder
cd /Users/liaoyunyang/NTU_CSIE/2025_2/parallel_programming/final_project/parallel_team_work

# Compile (this will take ~30 seconds)
g++ -std=c++11 main.cpp exif.cpp -o photomosaic_mac \
    $(pkg-config --cflags --libs opencv4) \
    -lboost_filesystem -lboost_system
```

### 3. Get a Test Image
```bash
# Option A: Use your own photo
cp ~/Pictures/your_photo.jpg test_input.jpg

# Option B: Use your webcam (Mac)
# Take a photo and save as test_input.jpg

# Option C: Use a Pokemon as test (for quick demo)
cp pokemon_tiles/pokemon_0025.png test_input.jpg
```

### 4. Run the Demo
```bash
./photomosaic_mac settings_pokemon.ini
```

**First run will:**
1. Preprocess all 500 Pokemon (calculate average colors) - ~30 seconds
2. Generate mosaic from your input image
3. Save result as `output.png`

**View the result:**
```bash
open output.png
```

## 🎯 What You'll See

Your input image recreated using Pokemon sprites! Each pixel becomes a Pokemon that matches that color:
- Red areas → Charizard, Flareon
- Blue areas → Blastoise, Squirtle  
- Green areas → Bulbasaur, Sceptile
- Yellow areas → Pikachu, Jolteon

## ⚙️ Customize Settings

Edit `settings_pokemon.ini`:

```ini
#PIXEL SIZE (bigger = see Pokemon details, smaller = more mosaic-like)
64 64          # Try: 32 32 or 128 128

#RESIZE (smaller = more detail, bigger file)
0.1 0.1        # Try: 0.05 0.05 for more detail

#SKIP INTERVAL (higher = less repetition)
100            # Try: 50 or 200
```

## 🚀 Next Steps (After Basic Demo Works)

1. **Add OpenMP Parallelization** - Make it 10x faster
2. **Real-time Webcam Input** - Live Pokemon filter
3. **Performance Benchmarking** - Measure speedup
4. **Presentation Demo** - Show live to audience

## 📝 Troubleshooting

**"opencv4 not found"**
```bash
# Check if OpenCV is installed
brew list opencv

# If not, install it
brew install opencv
```

**"boost not found"**
```bash
brew install boost
```

**Compilation errors**
```bash
# Try using clang++ instead of g++
clang++ -std=c++11 main.cpp exif.cpp -o photomosaic_mac \
    $(pkg-config --cflags --libs opencv4) \
    -lboost_filesystem -lboost_system
```

**Output is too big/small**
- Adjust RESIZE in settings_pokemon.ini
- Smaller resize = bigger output file

---

## 💡 TL;DR - Absolute Minimum to Run Demo

```bash
# 1. Install dependencies (one time only)
brew install opencv boost

# 2. Compile
g++ -std=c++11 main.cpp exif.cpp -o photomosaic_mac \
    $(pkg-config --cflags --libs opencv4) \
    -lboost_filesystem -lboost_system

# 3. Get test image
cp pokemon_tiles/pokemon_0025.png test_input.jpg

# 4. Run
./photomosaic_mac settings_pokemon.ini

# 5. View result
open output.png
```

**Estimated time**: 5 minutes (if dependencies install quickly)
