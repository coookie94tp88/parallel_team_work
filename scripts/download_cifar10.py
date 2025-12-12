#!/usr/bin/env python3
"""
Download CIFAR-10 dataset and extract as individual tile images.
60,000 32x32 color images in 10 classes.
"""

import urllib.request
import tarfile
import pickle
import os
import numpy as np
from PIL import Image

OUTPUT_DIR = "cifar_tiles"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_FILE = "cifar-10-python.tar.gz"

def download_cifar10():
    """Download CIFAR-10 dataset."""
    if os.path.exists(CIFAR_FILE):
        print(f"{CIFAR_FILE} already exists, skipping download")
        return
    
    print(f"Downloading CIFAR-10 from {CIFAR_URL}...")
    print("This is ~170 MB, may take a few minutes...")
    
    urllib.request.urlretrieve(CIFAR_URL, CIFAR_FILE)
    print("✓ Download complete!")

def extract_images():
    """Extract CIFAR-10 images to individual files."""
    print("\nExtracting images...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extract tar file
    with tarfile.open(CIFAR_FILE, 'r:gz') as tar:
        tar.extractall()
    
    # Load batches and save images
    batch_files = [
        'cifar-10-batches-py/data_batch_1',
        'cifar-10-batches-py/data_batch_2',
        'cifar-10-batches-py/data_batch_3',
        'cifar-10-batches-py/data_batch_4',
        'cifar-10-batches-py/data_batch_5',
        'cifar-10-batches-py/test_batch'
    ]
    
    image_count = 0
    
    for batch_file in batch_files:
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        
        images = batch[b'data']
        labels = batch[b'labels']
        
        # Reshape images from (10000, 3072) to (10000, 32, 32, 3)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        for i, (img_data, label) in enumerate(zip(images, labels)):
            # Convert to PIL Image
            img = Image.fromarray(img_data)
            
            # Save
            filename = f"{OUTPUT_DIR}/cifar_{image_count:05d}_class{label}.png"
            img.save(filename)
            
            image_count += 1
            
            if image_count % 5000 == 0:
                print(f"  Extracted {image_count} images...")
    
    print(f"\n✓ Extracted {image_count} images to {OUTPUT_DIR}/")
    
    # Cleanup
    import shutil
    shutil.rmtree('cifar-10-batches-py')
    os.remove(CIFAR_FILE)
    print("✓ Cleaned up temporary files")

def main():
    print("=== CIFAR-10 Dataset Downloader ===\n")
    
    download_cifar10()
    extract_images()
    
    print("\n=== Complete! ===")
    print(f"60,000 images ready in {OUTPUT_DIR}/")
    print("\nTo use with mosaic:")
    print(f"  ./video_mosaic --ultra -f -j 8")
    print(f"  (Update tile_dir to '{OUTPUT_DIR}' in code)")

if __name__ == "__main__":
    main()
