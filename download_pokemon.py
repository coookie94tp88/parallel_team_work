#!/usr/bin/env python3
"""
Download ONLY basic Pokemon PNG sprites (simple game style).
Much smaller download - only the essential sprites.
"""

import urllib.request
import os
from pathlib import Path

# Configuration
OUTPUT_DIR = "pokemon_tiles"
NUM_POKEMON = 500  # Adjust as needed (151 for Gen 1, 500 for variety)

# Using basic sprite URL - just simple PNG, no fancy variants
BASE_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"

def download_pokemon_sprites(num_pokemon=500):
    """Download basic Pokemon sprites only."""
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"Downloading {num_pokemon} basic Pokemon sprites...")
    print(f"Style: Simple game sprites (smallest files)")
    print(f"Output: {OUTPUT_DIR}/\n")
    
    successful = 0
    failed = []
    
    for i in range(1, num_pokemon + 1):
        url = f"{BASE_URL}/{i}.png"
        output_path = f"{OUTPUT_DIR}/pokemon_{i:04d}.png"
        
        try:
            urllib.request.urlretrieve(url, output_path)
            successful += 1
            
            if i % 50 == 0:
                print(f"  ✓ {i}/{num_pokemon} downloaded...")
                
        except Exception as e:
            failed.append(i)
            continue
    
    print(f"\n✅ Complete!")
    print(f"   Downloaded: {successful} sprites")
    print(f"   Failed: {len(failed)}")
    
    # Show file size
    if successful > 0:
        sample_file = f"{OUTPUT_DIR}/pokemon_0001.png"
        if os.path.exists(sample_file):
            sample_size = os.path.getsize(sample_file)
            total_mb = (sample_size * successful) / (1024 * 1024)
            print(f"   Total size: ~{total_mb:.1f} MB")
            print(f"   Avg per sprite: ~{sample_size/1024:.1f} KB")

if __name__ == "__main__":
    # Download first 500 Pokemon (good variety, not too large)
    download_pokemon_sprites(500)
