#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "EdgeFeatures.h"

// Struct of Arrays layout for better cache locality
struct TileDatabase {
    // Hot data (accessed every frame) - tightly packed
    std::vector<TileFeatures> features;
    
    // Cold data (accessed only during display)
    std::vector<cv::Mat> images;
    std::vector<std::string> paths;
    
    size_t size() const { return features.size(); }
    bool empty() const { return features.empty(); }
    
    void resize(size_t n) {
        features.resize(n);
        images.resize(n);
        paths.resize(n);
    }
};
