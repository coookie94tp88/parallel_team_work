#pragma once
#include <array>
#include <opencv2/opencv.hpp>

struct EdgeFeatures {
    float edge_strength;              // Average gradient magnitude
    std::array<float, 4> edge_histogram;   // Edge directions: 0°, 45°, 90°, 135°
};

struct TileFeatures {
    std::array<cv::Vec3f, 9> region_colors;  // 3x3 grid of colors
    cv::Vec3f mean_color;                // Overall mean (for backward compat)
    EdgeFeatures edges;              // Edge/gradient information
};
