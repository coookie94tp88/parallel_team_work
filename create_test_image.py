#!/usr/bin/env python3
"""Create a simple test image for mosaic demo."""
import cv2
import numpy as np

# Create a 600x400 image with a gradient and some shapes
img = np.zeros((400, 600, 3), dtype=np.uint8)

# Create colorful gradient background
for y in range(400):
    for x in range(600):
        img[y, x] = [int(255 * x / 600), int(255 * y / 400), 128]

# Add a circle
cv2.circle(img, (300, 200), 100, (255, 100, 100), -1)

# Add a rectangle
cv2.rectangle(img, (100, 100), (250, 300), (100, 255, 100), -1)

# Save
cv2.imwrite('test_input.jpg', img)
print("✓ Created test_input.jpg (600x400)")
