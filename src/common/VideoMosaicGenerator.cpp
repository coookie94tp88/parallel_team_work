#include "VideoMosaicGenerator.h"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using namespace cv;
using namespace std;
using namespace chrono;

VideoMosaicGenerator::VideoMosaicGenerator(const VideoMosaicConfig& cfg) 
    : config(cfg), frame_count(0) {
    omp_set_num_threads(config.num_threads);
}

TileFeatures VideoMosaicGenerator::computeRegionColors(const Mat& img) {
    TileFeatures features;
    
    // Create mask to ignore black backgrounds
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    Mat mask = gray > 30;
    
    // Divide tile into 3x3 grid
    int region_h = img.rows / 3;
    int region_w = img.cols / 3;
    
    for (int ry = 0; ry < 3; ry++) {
        for (int rx = 0; rx < 3; rx++) {
            int y_start = ry * region_h;
            int x_start = rx * region_w;
            int y_end = (ry == 2) ? img.rows : (ry + 1) * region_h;
            int x_end = (rx == 2) ? img.cols : (rx + 1) * region_w;
            
            Rect region_rect(x_start, y_start, x_end - x_start, y_end - y_start);
            Mat region = img(region_rect);
            Mat region_mask = mask(region_rect);
            
            Scalar mean = cv::mean(region, region_mask);
            
            if (countNonZero(region_mask) == 0) {
                mean = cv::mean(region);
            }
            
            int idx = ry * 3 + rx;
            features.region_colors[idx] = Vec3f(mean[0], mean[1], mean[2]);
        }
    }
    
    Scalar overall_mean = cv::mean(img, mask);
    if (countNonZero(mask) == 0) {
        overall_mean = cv::mean(img);
    }
    features.mean_color = Vec3f(overall_mean[0], overall_mean[1], overall_mean[2]);
    
    return features;
}

EdgeFeatures VideoMosaicGenerator::computeEdgeFeatures(const Mat& img) {
    EdgeFeatures features;
    features.edge_histogram.fill(0.0f);
    
    Mat gray;
    if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
    else gray = img.clone();
    
    Mat grad_x, grad_y;
    Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    
    Mat magnitude, direction;
    cartToPolar(grad_x, grad_y, magnitude, direction, true);
    
    features.edge_strength = mean(magnitude)[0];
    
    for (int y = 0; y < magnitude.rows; y++) {
        const float* mag_ptr = magnitude.ptr<float>(y);
        const float* dir_ptr = direction.ptr<float>(y);
        for (int x = 0; x < magnitude.cols; x++) {
            float mag = mag_ptr[x];
            float dir = dir_ptr[x];
            
            if (mag > 10.0f) {
                int bin = int((dir + 22.5f) / 45.0f) % 4;
                features.edge_histogram[bin] += mag;
            }
        }
    }
    
    float total = 0.0f;
    for (float v : features.edge_histogram) total += v;
    if (total > 0.001f) {
        for (float& v : features.edge_histogram) v /= total;
    }
    
    return features;
}

EdgeFeatures VideoMosaicGenerator::extractEdgeFeaturesFromGlobal(const Mat& magnitude, const Mat& direction) {
    EdgeFeatures features;
    features.edge_histogram.fill(0.0f);
    
    Scalar m = mean(magnitude);
    features.edge_strength = m[0];
    
    for (int y = 0; y < magnitude.rows; y++) {
        const float* mag_ptr = magnitude.ptr<float>(y);
        const float* dir_ptr = direction.ptr<float>(y);
        for (int x = 0; x < magnitude.cols; x++) {
            float mag = mag_ptr[x];
            float dir = dir_ptr[x];
            
            if (mag > 10.0f) {
                int bin = int((dir + 22.5f) / 45.0f) % 4;
                features.edge_histogram[bin] += mag;
            }
        }
    }
    
    float total = 0.0f;
    for (float v : features.edge_histogram) total += v;
    if (total > 0.001f) {
        for (float& v : features.edge_histogram) v /= total;
    }
    
    return features;
}

float VideoMosaicGenerator::regionDistance(const TileFeatures& f1, const TileFeatures& f2) {
    const float weights[9] = {
        0.5f, 1.0f, 0.5f,
        1.0f, 2.0f, 1.0f,
        0.5f, 1.0f, 0.5f
    };
    
    float total_dist = 0.0f;
    float total_weight = 0.0f;
    
    for (int i = 0; i < 9; i++) {
        const Vec3f& c1 = f1.region_colors[i];
        const Vec3f& c2 = f2.region_colors[i];
        
        float db = c1[0] - c2[0];
        float dg = c1[1] - c2[1];
        float dr = c1[2] - c2[2];
        float dist = db*db + dg*dg + dr*dr;
        
        total_dist += dist * weights[i];
        total_weight += weights[i];
    }
    
    return total_dist / total_weight;
}

float VideoMosaicGenerator::combinedDistance(const TileFeatures& f1, const TileFeatures& f2) {
    float color_dist = regionDistance(f1, f2);
    float strength_diff = abs(f1.edges.edge_strength - f2.edges.edge_strength);
    
    float direction_dist = 0.0f;
    int valid_bins = 0;
    for (int i = 0; i < 4; i++) {
        float sum = f1.edges.edge_histogram[i] + f2.edges.edge_histogram[i];
        if (sum > 0.01f) {
            float diff = f1.edges.edge_histogram[i] - f2.edges.edge_histogram[i];
            direction_dist += (diff * diff) / sum;
            valid_bins++;
        }
    }
    
    if (valid_bins > 0) direction_dist /= valid_bins;
    
    const float COLOR_WEIGHT = 0.7f;
    const float EDGE_DIR_WEIGHT = 0.2f;
    const float EDGE_STR_WEIGHT = 0.1f;
    
    float edge_component = 0.0f;
    if (f1.edges.edge_strength > 1.0f || f2.edges.edge_strength > 1.0f) {
        edge_component = EDGE_DIR_WEIGHT * direction_dist * 100.0f +
                       EDGE_STR_WEIGHT * strength_diff;
    }
    
    return COLOR_WEIGHT * color_dist + edge_component;
}

int VideoMosaicGenerator::findBestTile(const TileFeatures& target_features) {
    int best_idx = 0;
    float best_dist = __FLT_MAX__;
    
    // Naive linear search (O(N)) - parallelized in caller usually
    // But here just sequential for helper usage
    for (size_t i = 0; i < tiles.size(); i++) {
        float dist = combinedDistance(target_features, tiles.features[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    
    return best_idx;
}

bool VideoMosaicGenerator::loadTiles() {
    cout << "Loading tiles from: " << config.tile_dir << endl;
    
    if (!fs::exists(config.tile_dir) || !fs::is_directory(config.tile_dir)) {
        cerr << "Error: Directory " << config.tile_dir << " does not exist!" << endl;
        return false;
    }
    
    vector<string> filenames;
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(config.tile_dir); iter != end_iter; ++iter) {
        if (fs::is_regular_file(iter->status())) {
            string ext = iter->path().extension().string();
            if (ext == ".png" || ext == ".PNG" || 
                ext == ".jpg" || ext == ".JPG" || 
                ext == ".jpeg" || ext == ".JPEG") {
                filenames.push_back(iter->path().string());
            }
        }
    }
    
    if (filenames.empty()) {
        cerr << "No PNG/JPG files found in " << config.tile_dir << endl;
        return false;
    }
    
    tiles.resize(filenames.size());
    
    int loaded = 0;
    #pragma omp parallel for schedule(dynamic, 100)
    for (size_t i = 0; i < filenames.size(); i++) {
        Mat img = imread(filenames[i]);
        if (img.empty()) continue;
        
        Mat resized;
        resize(img, resized, Size(config.tile_size, config.tile_size));
        
        tiles.paths[i] = filenames[i];
        tiles.images[i] = resized;
        tiles.features[i] = computeRegionColors(resized);
        tiles.features[i].edges = computeEdgeFeatures(resized);
        
        #pragma omp atomic
        loaded++;
        
        if (loaded % 5000 == 0) {
            #pragma omp critical
            cout << "  Loaded " << loaded << " tiles..." << endl;
        }
    }
    
    cout << "Successfully loaded " << tiles.size() << " tiles" << endl;
    return !tiles.empty();
}

Mat VideoMosaicGenerator::generateMosaic(const Mat& input_frame) {
    auto start_time = high_resolution_clock::now();
    
    // Optimized generation using global Sobel
    Mat resized_frame;
    int target_w = config.grid_width * config.tile_size;
    int target_h = config.grid_height * config.tile_size;
    resize(input_frame, resized_frame, Size(target_w, target_h), 0, 0, INTER_LINEAR);
    
    cvtColor(resized_frame, global_gray, COLOR_BGR2GRAY);
    Sobel(global_gray, global_grad_x, CV_32F, 1, 0, 3);
    Sobel(global_gray, global_grad_y, CV_32F, 0, 1, 3);
    cartToPolar(global_grad_x, global_grad_y, global_magnitude, global_direction, true);
    
    Mat mosaic(target_h, target_w, CV_8UC3);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < config.grid_height; y++) {
        for (int x = 0; x < config.grid_width; x++) {
            // Extract from global maps
            Rect cell_rect(x * config.tile_size, y * config.tile_size, 
                          config.tile_size, config.tile_size);
            
            Mat cell_magnitude = global_magnitude(cell_rect);
            Mat cell_direction = global_direction(cell_rect);
            
            // Get Colors directly from resized frame
            Mat cell_img = resized_frame(cell_rect);
            TileFeatures cell_features = computeRegionColors(cell_img);
            
            // Get Edges from pre-computed maps
            cell_features.edges = extractEdgeFeaturesFromGlobal(cell_magnitude, cell_direction);
            
            // Find Match
            int best_idx = 0;
            float best_dist = __FLT_MAX__;
            
            for(size_t i=0; i<tiles.size(); i++) {
                float d = combinedDistance(cell_features, tiles.features[i]);
                if(d < best_dist) { best_dist = d; best_idx = i; }
            }
            
            // Place Tile
            tiles.images[best_idx].copyTo(mosaic(cell_rect));
        }
    }
    
    frame_count++;
    if (config.show_fps) {
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        double fps = 1000.0 / max(1, (int)duration.count());
        putText(mosaic, "FPS: " + to_string((int)fps), Point(20, 60), 
               FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 255, 0), 4);
    }
    
    return mosaic;
}

void VideoMosaicGenerator::processWebcam(int camera_id, int benchmark_frames) {
    VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam" << endl;
        return;
    }
    
    Mat frame, mosaic;
    int processed = 0;
    
    auto bench_start = high_resolution_clock::now();
    
    while (true) {
        if (!cap.read(frame)) break;
        mosaic = generateMosaic(frame);
        
        if (benchmark_frames > 0) {
            processed++;
            if (processed % 10 == 0) cout << "." << flush;
            if (processed >= benchmark_frames) break;
            continue; 
        }
        
        imshow("Mosaic", mosaic);
        if (waitKey(1) == 'q') break;
    }
    
    auto bench_end = high_resolution_clock::now();
    if (processed > 0) {
        auto duration = duration_cast<milliseconds>(bench_end - bench_start);
        double fps = 1000.0 * processed / duration.count();
        cout << endl << "Benchmark Result: " << fps << " FPS (" 
             << processed << " frames in " << duration.count() << "ms)" << endl;
    }
}
