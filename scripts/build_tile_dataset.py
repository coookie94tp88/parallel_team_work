#!/usr/bin/env python3
"""
Download 5000+ Pokemon tiles by getting all variants and using image augmentation.
"""

import urllib.request
import os
from pathlib import Path
from PIL import Image
import time

OUTPUT_DIR = "pokemon_tiles"
BASE_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"

def download_pokemon_variants(max_pokemon=1000):
    """Download all sprite variants."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    variants = {
        'front': '',
        'back': '/back',
        'shiny': '/shiny',
        'shiny_back': '/shiny/back',
    }
    
    print(f"Downloading Pokemon sprites (up to #{max_pokemon})...")
    print(f"Variants: {len(variants)} per Pokemon")
    
    successful = 0
    
    for i in range(1, max_pokemon + 1):
        for variant_name, variant_path in variants.items():
            filename = f"pokemon_{i:04d}_{variant_name}.png"
            filepath = f"{OUTPUT_DIR}/{filename}"
            
            # Skip if exists
            if os.path.exists(filepath):
                continue
            
            url = f"{BASE_URL}{variant_path}/{i}.png"
            
            try:
                urllib.request.urlretrieve(url, filepath)
                successful += 1
                
                if successful % 100 == 0:
                    print(f"  Downloaded {successful} sprites...")
                    
            except Exception:
                # Clean up failed download
                if os.path.exists(filepath):
                    os.remove(filepath)
                continue
            
            # Small delay
            if successful % 50 == 0:
                time.sleep(0.05)
    
    return successful

def augment_images():
    """Create augmented versions (flip, rotate) to increase tile count."""
    print("\nCreating augmented versions...")
    
    augmented = 0
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png') and not '_aug_' in f]
    
    for filename in files:
        filepath = f"{OUTPUT_DIR}/{filename}"
        base_name = filename.replace('.png', '')
        
        try:
            img = Image.open(filepath)
            
            # Horizontal flip
            flip_path = f"{OUTPUT_DIR}/{base_name}_aug_flip.png"
            if not os.path.exists(flip_path):
                img.transpose(Image.FLIP_LEFT_RIGHT).save(flip_path)
                augmented += 1
            
            # Rotate 180
            rot_path = f"{OUTPUT_DIR}/{base_name}_aug_rot180.png"
            if not os.path.exists(rot_path):
                img.rotate(180).save(rot_path)
                augmented += 1
                
            if augmented % 100 == 0:
                print(f"  Created {augmented} augmented images...")
                
        except Exception:
            continue
    
    return augmented

if __name__ == "__main__":
    print("=== Pokemon Tile Dataset Builder ===\n")
    
    # Download variants
    downloaded = download_pokemon_variants(1000)
    print(f"\nDownloaded: {downloaded} new sprites")
    
    # Create augmented versions
    augmented = augment_images()
    print(f"Augmented: {augmented} new images")
    
    # Count total
    total = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    print(f"\n✅ Total tiles: {total}")
    
    # Estimate size
    if total > 0:
        sample = f"{OUTPUT_DIR}/{os.listdir(OUTPUT_DIR)[0]}"
        size_mb = (os.path.getsize(sample) * total) / (1024 * 1024)
        print(f"   Total size: ~{size_mb:.1f} MB")
