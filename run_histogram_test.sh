#!/bin/bash
# Quick test to see if histogram matching is working

echo "Installing opencv-python for testing..."
pip install opencv-python

echo ""
echo "Running histogram test..."
python3 test_histogram.py
