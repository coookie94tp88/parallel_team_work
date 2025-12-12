#!/usr/bin/env python3
"""
Fix Pokemon tile backgrounds - make them all consistent (transparent or neutral gray).
"""

import os
from PIL import Image

OUTPUT_DIR = "pokemon_tiles"

def normalize_backgrounds():
    """Convert all tiles to have consistent backgrounds."""
    print("Normalizing Pokemon tile backgrounds...")
    
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    processed = 0
    
    for filename in files:
        filepath = f"{OUTPUT_DIR}/{filename}"
        
        try:
            img = Image.open(filepath).convert("RGBA")
            
            # Create a neutral gray background
            background = Image.new("RGBA", img.size, (128, 128, 128, 255))
            
            # Composite the Pokemon on gray background
            result = Image.alpha_composite(background, img)
            
            # Convert back to RGB
            result = result.convert("RGB")
            
            # Save
            result.save(filepath.replace('.png', '_fixed.png'))
            os.remove(filepath)
            os.rename(filepath.replace('.png', '_fixed.png'), filepath)
            
            processed += 1
            
            if processed % 1000 == 0:
                print(f"  Processed {processed} tiles...")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"\n✅ Normalized {processed} tiles")
    print("All tiles now have consistent gray backgrounds")

if __name__ == "__main__":
    normalize_backgrounds()
