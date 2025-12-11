# Understanding Mosaic Quality Issues

## The Reality of Photomosaics

Photomosaics **inherently look blocky** - that's the artistic style. However, there are ways to make them look better.

## Potential Issues

### 1. **Not Enough Tile Variety** ✓ Likely
- You have 1000 Pokemon
- Ultra mode uses 4800 tiles (80×60)
- **Each Pokemon used ~5 times per frame**
- This creates visible repetition

### 2. **Pokemon Backgrounds**
- Many Pokemon have **white/transparent backgrounds**
- Backgrounds dominate the histogram
- Makes different Pokemon look similar

### 3. **Histogram Bins Too Coarse**
- Currently using 8×8×8 = 512 bins
- Might not capture enough detail
- Could try 16×16×16 = 4096 bins

### 4. **No Color Adjustment**
- Tiles are used "as-is"
- No blending with target region
- Makes transitions harsh

## Solutions to Try

### Option 1: Color Blending (Best for appearance)
Blend each tile's color toward the target region:
```
final_tile = 0.7 * pokemon_tile + 0.3 * target_color
```

### Option 2: More Tiles
Download all Pokemon (currently ~1000, could get more variants):
- Different forms
- Shiny versions
- Regional variants

### Option 3: Finer Histogram
Use 16×16×16 bins instead of 8×8×8

### Option 4: Hybrid Approach
- Use histogram for coarse matching
- Use mean color for fine-tuning
- Blend result with target

## What Professional Photomosaics Do

They typically:
1. **Huge tile databases** (10,000+ images)
2. **Color adjustment** (tint tiles toward target)
3. **Tile reuse limits** (don't repeat same tile nearby)
4. **Alpha blending** (subtle overlay of original image)

## Recommendation

Add **color blending** - this will make the biggest visual difference without needing more tiles.

Would you like me to implement color blending?
