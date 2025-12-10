# Pokemon Mosaic Demo Guide

## Quick Start

### 1. Compile the program
```bash
# You need OpenCV and Boost installed
g++ -std=c++11 main.cpp exif.cpp -o photomosaic \
    `pkg-config --cflags --libs opencv4` \
    -lboost_filesystem -lboost_system
```

### 2. Get a test image
You need an input image to convert into a mosaic. Options:

**Option A: Use your own photo**
```bash
# Copy any JPG/PNG to the project folder
cp ~/Pictures/your_photo.jpg test_input.jpg
```

**Option B: Download a test image**
```bash
# Download a sample image
curl -o test_input.jpg https://picsum.photos/800/600
```

**Option C: Take a webcam photo**
```bash
# On Mac, use built-in camera
# (We'll add webcam support later for real-time demo)
```

### 3. Run the mosaic generator
```bash
./photomosaic settings_pokemon.ini
```

This will:
1. **First run**: Preprocess all 500 Pokemon sprites (calculate average colors) - takes ~30 seconds
2. Generate the mosaic by matching each pixel to the best Pokemon sprite
3. Save result as `output.png`

### 4. View the result
```bash
open output.png
```

## What Each Setting Does

- **DATASET FOLDER**: `./pokemon_tiles` - where your 500 Pokemon sprites are
- **PIXELS FOLDER**: `pokemon_pixels/` - preprocessed tiles (created automatically)
- **PIXEL SIZE**: `64 64` - each Pokemon will be 64x64 pixels in final image
- **INPUT IMAGE**: `test_input.jpg` - the photo you want to "pokemonify"
- **RESIZE**: `0.1 0.1` - shrink input to 10% (controls mosaic grid size)
- **CALCULATE PIXELS**: `1` - preprocess tiles (set to `0` after first run to skip)
- **CALCULATE MOSAIC**: `1` - generate the mosaic
- **SKIP INTERVAL**: `100` - avoid repeating same Pokemon too close together

## Expected Output

If your input is 800x600 and resize is 0.1:
- Grid will be 80x60 pixels
- Each pixel becomes a 64x64 Pokemon sprite
- Final mosaic: 5120x3840 pixels (huge!)
- You'll see Pikachu, Charizard, Bulbasaur, etc. forming your image

## Troubleshooting

**"Command not found: pkg-config"**
```bash
brew install pkg-config
```

**"opencv4 not found"**
```bash
brew install opencv
```

**"boost not found"**
```bash
brew install boost
```

**Mosaic is too big/small**
- Adjust RESIZE (smaller = more detail, bigger file)
- Adjust PIXEL SIZE (smaller = more Pokemon visible, bigger file)

## Next Steps

Once this works, we'll add:
- OpenMP parallelization (make it 10x faster)
- Real-time webcam input (live Pokemon filter)
- Performance benchmarking
