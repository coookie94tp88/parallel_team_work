#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using namespace cv;
using namespace std;
using namespace chrono;

// Edge features for structure-aware matching
struct EdgeFeatures {
    float edge_strength;              // Average gradient magnitude
    array<float, 4> edge_histogram;   // Edge directions: 0°, 45°, 90°, 135°
};

// Multi-region tile features (3x3 grid for spatial detail + edges)
struct TileFeatures {
    array<Vec3f, 9> region_colors;  // 3x3 grid of colors
    Vec3f mean_color;                // Overall mean (for backward compat)
    EdgeFeatures edges;              // Edge/gradient information
};

// Struct of Arrays layout for better cache locality
struct TileDatabase {
    // Hot data (accessed every frame) - tightly packed
    vector<TileFeatures> features;
    
    // Cold data (accessed only during display)
    vector<Mat> images;
    vector<string> paths;
    
    size_t size() const { return features.size(); }
    bool empty() const { return features.empty(); }
    
    void resize(size_t n) {
        features.resize(n);
        images.resize(n);
        paths.resize(n);
    }
};

// Video mosaic configuration
struct VideoMosaicConfig {
    string tile_dir;
    int tile_size;           // Size of each tile (e.g., 32x32)
    int grid_width;          // Number of tiles horizontally
    int grid_height;         // Number of tiles vertically
    int num_threads;         // OpenMP threads
    bool temporal_coherence; // Reuse tiles between frames
    int coherence_threshold; // Color difference threshold for reuse
    bool show_fps;           // Display FPS counter
    bool use_histogram;      // true = histogram matching (accurate), false = mean color (fast)
    float color_blend;       // 0.0-1.0: blend tile color toward target (0=no blend, 1=full blend)
};

class VideoMosaicGenerator {
private:
    TileDatabase tiles;
    VideoMosaicConfig config;
    vector<int> previous_assignments;  // Cache previous frame's tile assignments
    int frame_count;
    
    // Global feature maps (reused per frame to avoid allocation)
    Mat global_gray, global_grad_x, global_grad_y, global_magnitude, global_direction;
    
    // Compute 3x3 grid of region colors to capture spatial detail
    TileFeatures computeRegionColors(const Mat& img) {
        TileFeatures features;
        
        // Create mask to ignore black backgrounds
        Mat gray;
        if (img.channels() == 3) {
            cvtColor(img, gray, COLOR_BGR2GRAY);
        } else {
            gray = img;
        }
        
        Mat mask = gray > 30;
        
        // Divide tile into 3x3 grid
        int region_h = img.rows / 3;
        int region_w = img.cols / 3;
        
        for (int ry = 0; ry < 3; ry++) {
            for (int rx = 0; rx < 3; rx++) {
                int y_start = ry * region_h;
                int x_start = rx * region_w;
                // Safe slicing
                int h = (ry == 2) ? (img.rows - y_start) : region_h;
                int w = (rx == 2) ? (img.cols - x_start) : region_w;
                
                Rect region_rect(x_start, y_start, w, h);
                Mat region = img(region_rect);
                Mat region_mask = mask(region_rect);
                
                Scalar mean;
                if (countNonZero(region_mask) == 0) {
                     mean = cv::mean(region);
                } else {
                     mean = cv::mean(region, region_mask);
                }
                
                int idx = ry * 3 + rx;
                features.region_colors[idx] = Vec3f(mean[0], mean[1], mean[2]);
            }
        }
        
        // Compute overall mean color
        Scalar overall_mean;
        if (countNonZero(mask) == 0) {
            overall_mean = cv::mean(img);
        } else {
            overall_mean = cv::mean(img, mask);
        }
        features.mean_color = Vec3f(overall_mean[0], overall_mean[1], overall_mean[2]);
        
        return features;
    }
    
    // Compute edge features using Sobel operators (Standard version for tiles)
    EdgeFeatures computeEdgeFeatures(const Mat& img) {
        EdgeFeatures features;
        features.edge_histogram.fill(0.0f);
        
        // Convert to grayscale
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        
        // Compute gradients using Sobel
        Mat grad_x, grad_y;
        Sobel(gray, grad_x, CV_32F, 1, 0, 3);  // Horizontal gradient
        Sobel(gray, grad_y, CV_32F, 0, 1, 3);  // Vertical gradient
        
        // Compute magnitude and direction
        Mat magnitude, direction;
        cartToPolar(grad_x, grad_y, magnitude, direction, true);  // true = degrees
        
        // Compute average edge strength
        features.edge_strength = mean(magnitude)[0];
        
        // Build histogram of edge directions
        for (int y = 0; y < magnitude.rows; y++) {
            float* mag_ptr = magnitude.ptr<float>(y);
            float* dir_ptr = direction.ptr<float>(y);
            for (int x = 0; x < magnitude.cols; x++) {
                float mag = mag_ptr[x];
                float dir = dir_ptr[x];
                
                if (mag > 10.0f) {
                    int bin = int((dir + 22.5f) / 45.0f) % 4;
                    features.edge_histogram[bin] += mag;
                }
            }
        }
        
        // Normalize histogram
        float total = 0.0f;
        for (float v : features.edge_histogram) total += v;
        if (total > 0.001f) {
            for (float& v : features.edge_histogram) v /= total;
        }
        
        return features;
    }
    
    // OPTIMIZED: Extract edge features from pre-computed global maps
    EdgeFeatures extractEdgeFeaturesFromGlobal(const Mat& magnitude, const Mat& direction) {
        EdgeFeatures features;
        features.edge_histogram.fill(0.0f);
        
        // Compute average edge strength directly from magnitude region
        Scalar m = mean(magnitude);
        features.edge_strength = m[0];
        
        // Build histogram
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
    
    // Multi-region distance: compare 3x3 grids with weighted importance
    inline float regionDistance(const TileFeatures& f1, const TileFeatures& f2) {
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
    
    // Combined distance: color + edge features
    inline float combinedDistance(const TileFeatures& f1, const TileFeatures& f2) {
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
        
        if (valid_bins > 0) {
            direction_dist /= valid_bins;
        }
        
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
    
    // Find best matching tile using combined features (greedy)
    int findBestTile(const TileFeatures& target_features) {
        int best_idx = 0;
        float best_dist = FLT_MAX;
        
        for (size_t i = 0; i < tiles.size(); i++) {
            float dist = combinedDistance(target_features, tiles.features[i]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
        
        return best_idx;
    }

public:
    VideoMosaicGenerator(const VideoMosaicConfig& cfg) : config(cfg), frame_count(0) {
        omp_set_num_threads(config.num_threads);
    }
    
    // Load tiles from directory
    bool loadTiles() {
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
        
        cout << "Found " << filenames.size() << " image files" << endl;
        
        if (filenames.empty()) {
            cerr << "No PNG/JPG files found in " << config.tile_dir << endl;
            return false;
        }
        
        tiles.resize(filenames.size());
        
        int loaded = 0;
        #pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < filenames.size(); i++) {
            Mat img = imread(filenames[i]);
            if (img.empty()) {
                continue;
            }
            
            Mat resized;
            resize(img, resized, Size(config.tile_size, config.tile_size));
            
            tiles.paths[i] = filenames[i];
            tiles.images[i] = resized;
            tiles.features[i] = computeRegionColors(resized);
            tiles.features[i].edges = computeEdgeFeatures(resized);
            
            #pragma omp atomic
            loaded++;
            
            if (loaded % 1000 == 0) {
                #pragma omp critical
                cout << "  Loaded " << loaded << " tiles..." << endl;
            }
        }
        
        cout << "Successfully loaded " << tiles.size() << " tiles" << endl;
        cout << "Using OPTIMAL CPU EDGE-AWARE matching (Global Pre-computation)" << endl;
        return !tiles.empty();
    }
    
    // Generate mosaic from a single frame (OPTIMIZED)
    Mat generateMosaic(const Mat& input_frame) {
        auto start_time = high_resolution_clock::now();
        
        // 1. Resize entire frame once to grid dimensions
        // Note: Using INTER_AREA for downsampling (better quality) or INTER_LINEAR for speed
        Mat resized_frame;
        int target_w = config.grid_width * config.tile_size;
        int target_h = config.grid_height * config.tile_size;
        resize(input_frame, resized_frame, Size(target_w, target_h), 0, 0, INTER_LINEAR);
        
        // 2. Compute Global Feature Maps
        cvtColor(resized_frame, global_gray, COLOR_BGR2GRAY);
        Sobel(global_gray, global_grad_x, CV_32F, 1, 0, 3);
        Sobel(global_gray, global_grad_y, CV_32F, 0, 1, 3);
        cartToPolar(global_grad_x, global_grad_y, global_magnitude, global_direction, true);
        
        // Output mosaic
        Mat mosaic(target_h, target_w, CV_8UC3);
        
        int total_cells = config.grid_width * config.grid_height;
        vector<int> current_assignments(total_cells);
        
        // 3. Parallel Processing of Cells
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < config.grid_height; y++) {
            for (int x = 0; x < config.grid_width; x++) {
                int grid_idx = y * config.grid_width + x;
                
                // Define ROI for this cell (no resizing needed!)
                Rect cell_roi(x * config.tile_size, y * config.tile_size, 
                              config.tile_size, config.tile_size);
                
                // Extract features efficiently using shared global maps
                TileFeatures cell_features;
                
                // Color features (slicing from resized_frame)
                Mat cell_color = resized_frame(cell_roi);
                cell_features = computeRegionColors(cell_color);
                
                // Edge features (slicing from global maps)
                Mat cell_mag = global_magnitude(cell_roi);
                Mat cell_dir = global_direction(cell_roi);
                cell_features.edges = extractEdgeFeaturesFromGlobal(cell_mag, cell_dir);
                
                // Find best match
                int tile_idx = findBestTile(cell_features);
                current_assignments[grid_idx] = tile_idx;
                
                // Place tile (direct copy)
                tiles.images[tile_idx].copyTo(mosaic(cell_roi));
            }
        }
        
        previous_assignments = current_assignments;
        frame_count++;
        
        if (config.show_fps) {
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end_time - start_time);
            double fps = 1000.0 / max(1, (int)duration.count());
            
            string fps_text = "FPS: " + to_string((int)fps) + 
                            " | Frame: " + to_string(frame_count) +
                            " | Threads: " + to_string(config.num_threads) + " (CPU OPT)";
            putText(mosaic, fps_text, Point(20, 60), 
                   FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 255, 0), 4);
        }
        
        return mosaic;
    }
    
    // Process webcam stream
    void processWebcam(int camera_id = 0, int benchmark_frames = 0) {
        VideoCapture cap(camera_id);
        
        if (!cap.isOpened()) {
            cerr << "Error: Could not open webcam" << endl;
            return;
        }

        if (benchmark_frames > 0) {
            cout << "Running benchmark for " << benchmark_frames << " frames..." << endl;
            // Warmup
             Mat frame;
             for(int i=0; i<3; i++) { cap.read(frame); generateMosaic(frame); }
        } else {
            cout << "\n=== Video Mosaic Started (CPU Optimized) ===" << endl;
            // ... (rest of print statements omitted for brevity if needed, or keep them)
        }
        
        Mat frame, mosaic;
        int saved_count = 0;
        
        auto last_frame_time = high_resolution_clock::now();
        double fps_avg = 0.0;
        int fps_count = 0;
        
        while (cap.read(frame)) {
            auto frame_start = high_resolution_clock::now();
            
            mosaic = generateMosaic(frame);
            
            auto frame_end = high_resolution_clock::now();
            auto total_duration = duration_cast<milliseconds>(frame_end - last_frame_time);
            last_frame_time = frame_end;
            
            double current_fps = 1000.0 / max(1, (int)total_duration.count());
            fps_avg = (fps_avg * fps_count + current_fps) / (fps_count + 1);
            fps_count++;
            
            if (benchmark_frames > 0) {
                if (fps_count >= benchmark_frames) {
                    cout << "RESULT_FPS: " << fps_avg << endl;
                    break;
                }
                continue; // Skip imshow/waitKey
            }
            
            if (fps_count % 30 == 0) {
                 cout << "FPS: " << (int)current_fps << " | Avg: " << (int)fps_avg << endl;
            }
            
            imshow("Mosaic CPU Optimized", mosaic);
            
            int key = waitKey(1);
            if (key == 'q') break;
            else if (key == 's') { /* save */ }
            else if (key == '+') { /* threads++ */ }
            else if (key == '-') { /* threads-- */ }
        }
        cap.release();
        destroyAllWindows();
    }
};

void printUsage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "\nOptions:" << endl;
    cout << "  -d, --tiles DIR     Tile directory (default: data/pokemon_tiles)" << endl;
    cout << "  -j, --threads N     Number of threads (default: 4)" << endl;
    cout << "  -t, --tile-size N   Tile size in pixels (default: 32)" << endl;
    cout << "  -f, --show-fps      Show FPS counter" << endl;
    cout << "  --help              Show this help" << endl;
    cout << "\nPreset Quality Modes:" << endl;
    cout << "  --small             Small grid (40x30)" << endl;
    cout << "  --medium            Medium grid (60x45, default)" << endl;
    cout << "  --large             Large grid (80x60)" << endl;
    cout << "  --ultra             Ultra grid (100x75)" << endl;
}

int main(int argc, char** argv) {
    VideoMosaicConfig config;
    config.tile_dir = "data/pokemon_tiles";
    config.tile_size = 32;
    config.grid_width = 60;
    config.grid_height = 45;
    config.num_threads = 4;
    config.temporal_coherence = true;
    config.coherence_threshold = 500.0;
    config.show_fps = true;
    config.use_histogram = false;
    config.color_blend = 0.0f;
    
    int benchmark_frames = 0;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help") { printUsage(argv[0]); return 0; }
        else if (arg == "-d" || arg == "--tiles") { if (i + 1 < argc) config.tile_dir = argv[++i]; }
        else if (arg == "-w" || arg == "--width") { if (i + 1 < argc) config.grid_width = atoi(argv[++i]); }
        else if (arg == "-h" || arg == "--height") { if (i + 1 < argc) config.grid_height = atoi(argv[++i]); }
        else if (arg == "-t" || arg == "--tile-size") { if (i + 1 < argc) config.tile_size = atoi(argv[++i]); }
        else if (arg == "-j" || arg == "--threads") { if (i + 1 < argc) config.num_threads = atoi(argv[++i]); }
        else if (arg == "--benchmark") { if (i + 1 < argc) benchmark_frames = atoi(argv[++i]); }
        else if (arg == "--small") { config.grid_width = 40; config.grid_height = 30; }
        else if (arg == "--medium") { config.grid_width = 60; config.grid_height = 45; }
        else if (arg == "--large") { config.grid_width = 80; config.grid_height = 60; }
        else if (arg == "--ultra") { config.grid_width = 100; config.grid_height = 75; }
        else if (arg == "-f" || arg == "--show-fps") { config.show_fps = true; }
        else { cerr << "Unknown option: " << arg << endl; printUsage(argv[0]); return 1; }
    }
    
    cout << "=== Configuration (CPU Optimized) ===" << endl;
    cout << "Grid: " << config.grid_width << "x" << config.grid_height 
         << " (" << (config.grid_width * config.grid_height) << " cells)" << endl;
    cout << "Tile size: " << config.tile_size << "x" << config.tile_size << endl;
    cout << "Threads: " << config.num_threads << endl;
    cout << "Benchmark: " << (benchmark_frames > 0 ? to_string(benchmark_frames) + " frames" : "Disabled") << endl;
    cout << "=====================================\n" << endl;
    
    VideoMosaicGenerator generator(config);
    if (!generator.loadTiles()) return -1;
    generator.processWebcam(0, benchmark_frames);
    return 0;
}
