#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "VideoMosaicConfig.h"
#include "TileDatabase.h"

class VideoMosaicGenerator {
protected:
    TileDatabase tiles;
    VideoMosaicConfig config;
    std::vector<int> previous_assignments;
    int frame_count;
    
    // Global feature maps for optimization
    cv::Mat global_gray, global_grad_x, global_grad_y, global_magnitude, global_direction;

public:
    VideoMosaicGenerator(const VideoMosaicConfig& cfg);
    virtual ~VideoMosaicGenerator() = default;

    // Core functionality
    bool loadTiles();
    virtual cv::Mat generateMosaic(const cv::Mat& input_frame);
    void processWebcam(int camera_id = 0, int benchmark_frames = 0);
    
protected:
    // Helper methods
    TileFeatures computeRegionColors(const cv::Mat& img);
    EdgeFeatures computeEdgeFeatures(const cv::Mat& img);
    EdgeFeatures extractEdgeFeaturesFromGlobal(const cv::Mat& magnitude, const cv::Mat& direction);
    float regionDistance(const TileFeatures& f1, const TileFeatures& f2);
    float combinedDistance(const TileFeatures& f1, const TileFeatures& f2);
    int findBestTile(const TileFeatures& target_features);
};
