#!/usr/bin/env python3
"""
Download Pokemon sprites, skipping ones already downloaded.
Continues from where you left off.
"""

import urllib.request
import os
from pathlib import Path

# Configuration
OUTPUT_DIR = "pokemon_tiles"
MAX_POKEMON = 1000  # Download up to Pokemon #1000

# Using basic sprite URL - just simple PNG, no fancy variants
BASE_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"

def download_pokemon_sprites(max_pokemon=1000):
    """Download Pokemon sprites, skipping existing ones."""
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Check which Pokemon we already have
    existing = set()
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            if filename.startswith("pokemon_") and filename.endswith(".png"):
                # Extract number from pokemon_0001.png
                try:
                    num = int(filename.split("_")[1].split(".")[0])
                    existing.add(num)
                except:
                    continue
    
    print(f"Found {len(existing)} existing Pokemon")
    print(f"Downloading up to Pokemon #{max_pokemon}")
    print(f"Output: {OUTPUT_DIR}/\n")
    
    successful = 0
    skipped = 0
    failed = []
    
    for i in range(1, max_pokemon + 1):
        output_path = f"{OUTPUT_DIR}/pokemon_{i:04d}.png"
        
        # Skip if already exists
        if i in existing:
            skipped += 1
            continue
        
        url = f"{BASE_URL}/{i}.png"
        
        try:
            urllib.request.urlretrieve(url, output_path)
            successful += 1
            
            if successful % 50 == 0:
                print(f"  ✓ Downloaded {successful} new Pokemon...")
                
        except Exception as e:
            failed.append(i)
            continue
    
    total = len(existing) + successful
    print(f"\n✅ Complete!")
    print(f"   Already had: {skipped} sprites")
    print(f"   Downloaded: {successful} new sprites")
    print(f"   Failed: {len(failed)}")
    print(f"   Total Pokemon: {total}")
    
    # Show file size
    if total > 0:
        sample_file = f"{OUTPUT_DIR}/pokemon_0001.png"
        if os.path.exists(sample_file):
            sample_size = os.path.getsize(sample_file)
            total_mb = (sample_size * total) / (1024 * 1024)
            print(f"   Total size: ~{total_mb:.1f} MB")
            print(f"   Avg per sprite: ~{sample_size/1024:.1f} KB")

if __name__ == "__main__":
    # Download up to 1000 Pokemon (skips existing)
    download_pokemon_sprites(1000)
