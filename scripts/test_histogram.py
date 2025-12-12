#!/usr/bin/env python3
"""
Test if histogram matching is working by comparing a few Pokemon tiles.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def compute_histogram(img):
    """Compute 8x8x8 color histogram."""
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return hist

def compare_histograms(hist1, hist2):
    """Compare using Bhattacharyya distance."""
    return 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def test_histogram_matching():
    tile_dir = "pokemon_tiles"
    
    # Load a few test tiles
    tiles = []
    for i in range(1, 21):  # First 20 Pokemon
        path = f"{tile_dir}/pokemon_{i:04d}.png"
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                tiles.append((i, img, compute_histogram(img)))
    
    print(f"Loaded {len(tiles)} test tiles\n")
    
    # Test: Compare Pikachu (#25) with others
    pikachu_path = f"{tile_dir}/pokemon_0025.png"
    if os.path.exists(pikachu_path):
        pikachu = cv2.imread(pikachu_path)
        pikachu_hist = compute_histogram(pikachu)
        
        print("Comparing Pikachu (#25) with other Pokemon:")
        print("=" * 60)
        
        similarities = []
        for idx, img, hist in tiles:
            similarity = compare_histograms(pikachu_hist, hist)
            similarities.append((idx, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 most similar to Pikachu:")
        for idx, sim in similarities[:10]:
            print(f"  Pokemon #{idx:3d}: similarity = {sim:.4f}")
        
        print("\nLeast similar:")
        for idx, sim in similarities[-5:]:
            print(f"  Pokemon #{idx:3d}: similarity = {sim:.4f}")
    
    # Test color diversity
    print("\n" + "=" * 60)
    print("Analyzing color diversity of tiles...")
    
    mean_colors = []
    for idx, img, hist in tiles:
        mean_color = cv2.mean(img)[:3]
        mean_colors.append((idx, mean_color))
    
    print(f"\nSample mean colors:")
    for idx, (b, g, r) in mean_colors[:10]:
        print(f"  Pokemon #{idx:3d}: RGB({r:.0f}, {g:.0f}, {b:.0f})")

if __name__ == "__main__":
    test_histogram_matching()
