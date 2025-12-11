#!/usr/bin/env python3
"""
Download ALL Pokemon sprite variants for maximum diversity.
Gets front, back, shiny front, shiny back for each Pokemon.
"""

import urllib.request
import os
from pathlib import Path
import time

# Configuration
OUTPUT_DIR = "pokemon_tiles"
MAX_POKEMON = 1000

# PokeAPI sprite URLs
BASE_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"

SPRITE_TYPES = {
    'front': '',
    'back': '/back',
    'shiny': '/shiny',
    'shiny_back': '/shiny/back',
}

def download_all_variants(max_pokemon=1000):
    """Download all sprite variants for maximum diversity."""
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Check existing files
    existing = set()
    if os.path.exists(OUTPUT_DIR):
        existing = set(os.listdir(OUTPUT_DIR))
    
    print(f"Downloading sprite variants for {max_pokemon} Pokemon...")
    print(f"Types: front, back, shiny, shiny_back")
    print(f"Already have: {len(existing)} files")
    print(f"Output: {OUTPUT_DIR}/\n")
    
    successful = 0
    skipped = 0
    failed = []
    
    for i in range(1, max_pokemon + 1):
        for variant_name, variant_path in SPRITE_TYPES.items():
            filename = f"pokemon_{i:04d}_{variant_name}.png"
            
            # Skip if exists
            if filename in existing:
                skipped += 1
                continue
            
            url = f"{BASE_URL}{variant_path}/{i}.png"
            output_path = f"{OUTPUT_DIR}/{filename}"
            
            try:
                urllib.request.urlretrieve(url, output_path)
                successful += 1
                
                if successful % 100 == 0:
                    print(f"  ✓ Downloaded {successful} new sprites...")
                    
            except Exception as e:
                failed.append((i, variant_name))
                # Clean up failed download
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue
            
            # Small delay to be nice to the server
            if successful % 50 == 0:
                time.sleep(0.1)
    
    total = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    print(f"\n✅ Complete!")
    print(f"   Already had: {skipped} sprites")
    print(f"   Downloaded: {successful} new sprites")
    print(f"   Failed: {len(failed)}")
    print(f"   Total unique sprites: {total}")
    
    # Estimate size
    if total > 0:
        sample_file = None
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.png'):
                sample_file = f"{OUTPUT_DIR}/{f}"
                break
        
        if sample_file and os.path.exists(sample_file):
            sample_size = os.path.getsize(sample_file)
            total_mb = (sample_size * total) / (1024 * 1024)
            print(f"   Total size: ~{total_mb:.1f} MB")

if __name__ == "__main__":
    download_all_variants(1000)
