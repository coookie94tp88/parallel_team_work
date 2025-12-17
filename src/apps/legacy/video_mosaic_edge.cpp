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
    
    // Compute 3x3 grid of region colors to capture spatial detail
    TileFeatures computeRegionColors(const Mat& img) {
        TileFeatures features;
        
        // Create mask to ignore black backgrounds
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
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
        
        // Compute overall mean color
        Scalar overall_mean = cv::mean(img, mask);
        if (countNonZero(mask) == 0) {
            overall_mean = cv::mean(img);
        }
        features.mean_color = Vec3f(overall_mean[0], overall_mean[1], overall_mean[2]);
        
        return features;
    }
    
    // Compute edge features using Sobel operators
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
        
        // Build histogram of edge directions (4 bins: 0°, 45°, 90°, 135°)
        for (int y = 0; y < magnitude.rows; y++) {
            for (int x = 0; x < magnitude.cols; x++) {
                float mag = magnitude.at<float>(y, x);
                float dir = direction.at<float>(y, x);
                
                // Only count strong edges (threshold = 10)
                if (mag > 10.0f) {
                    // Quantize direction to 4 bins
                    // 0°: 337.5-22.5, 45°: 22.5-67.5, 90°: 67.5-112.5, 135°: 112.5-157.5
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
    
    // Multi-region distance: compare 3x3 grids with weighted importance
    inline float regionDistance(const TileFeatures& f1, const TileFeatures& f2) {
        // Weight matrix: center region more important than edges
        const float weights[9] = {
            0.5f, 1.0f, 0.5f,  // top row
            1.0f, 2.0f, 1.0f,  // middle row (center = 2x weight)
            0.5f, 1.0f, 0.5f   // bottom row
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
        // 1. Color distance (existing multi-region matching)
        float color_dist = regionDistance(f1, f2);
        
        // 2. Edge strength difference (normalized)
        float strength_diff = abs(f1.edges.edge_strength - f2.edges.edge_strength);
        
        // 3. Edge direction histogram distance (chi-square)
        float direction_dist = 0.0f;
        int valid_bins = 0;
        for (int i = 0; i < 4; i++) {
            float sum = f1.edges.edge_histogram[i] + f2.edges.edge_histogram[i];
            if (sum > 0.01f) {  // Only count bins with significant values
                float diff = f1.edges.edge_histogram[i] - f2.edges.edge_histogram[i];
                direction_dist += (diff * diff) / sum;
                valid_bins++;
            }
        }
        
        // Average over valid bins to normalize
        if (valid_bins > 0) {
            direction_dist /= valid_bins;
        }
        
        // Weighted combination - if no edges detected, fall back to color only
        const float COLOR_WEIGHT = 0.7f;      // Increase color importance
        const float EDGE_DIR_WEIGHT = 0.2f;   // Reduce edge direction weight
        const float EDGE_STR_WEIGHT = 0.1f;   // Edge strength
        
        // Scale edge distances to match color distance range
        float edge_component = 0.0f;
        if (f1.edges.edge_strength > 1.0f || f2.edges.edge_strength > 1.0f) {
            // Only use edge features if at least one tile has edges
            edge_component = EDGE_DIR_WEIGHT * direction_dist * 100.0f +
                           EDGE_STR_WEIGHT * strength_diff;
        }
        
        return COLOR_WEIGHT * color_dist + edge_component;
    }
    
    // Find best matching tile using combined features (greedy)
    int findBestTile(const TileFeatures& target_features) {
        int best_idx = 0;
        float best_dist = FLT_MAX;
        
        // Find best matching tile using combined distance
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
        
        // Iterate through directory and find PNG/JPG files
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
        // Load and preprocess tiles in parallel
        #pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < filenames.size(); i++) {
            Mat img = imread(filenames[i]);
            if (img.empty()) {
                continue;
            }
            
            // Resize to tile size
            Mat resized;
            resize(img, resized, Size(config.tile_size, config.tile_size));
            
            // Store in SoA layout
            tiles.paths[i] = filenames[i];
            tiles.images[i] = resized;
            
            // Compute region colors
            tiles.features[i] = computeRegionColors(resized);
            
            // Compute edge features
            tiles.features[i].edges = computeEdgeFeatures(resized);
            
            // Thread-safe progress counter
            #pragma omp atomic
            loaded++;
            
            // Progress indicator
            if (loaded % 1000 == 0) {
                #pragma omp critical
                cout << "  Loaded " << loaded << " tiles..." << endl;
            }
        }
        
        cout << "Successfully loaded " << tiles.size() << " tiles" << endl;
        cout << "Using edge-aware matching (3x3 regions + 4-direction edges)" << endl;
        return !tiles.empty();
    }
    
    // Generate mosaic from a single frame
    Mat generateMosaic(const Mat& input_frame) {
        auto start_time = high_resolution_clock::now();
        
        // Resize input to grid size
        Mat resized_input;
        resize(input_frame, resized_input, 
               Size(config.grid_width, config.grid_height));
        
        // Create output mosaic
        int output_width = config.grid_width * config.tile_size;
        int output_height = config.grid_height * config.tile_size;
        Mat mosaic(output_height, output_width, CV_8UC3);
        
        // Prepare assignment cache
        int total_cells = config.grid_width * config.grid_height;
        vector<int> current_assignments(total_cells);
        
        // Calculate tile dimensions in the input frame
        int input_tile_width = input_frame.cols / config.grid_width;
        int input_tile_height = input_frame.rows / config.grid_height;
        
        // Parallel tile matching and placement
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < config.grid_height; y++) {
            for (int x = 0; x < config.grid_width; x++) {
                int grid_idx = y * config.grid_width + x;
                
                // Extract region from input frame for this cell
                int y_start = y * input_tile_height;
                int x_start = x * input_tile_width;
                int y_end = min((y + 1) * input_tile_height, input_frame.rows);
                int x_end = min((x + 1) * input_tile_width, input_frame.cols);
                
                Rect cell_rect(x_start, y_start, x_end - x_start, y_end - y_start);
                Mat cell_img = input_frame(cell_rect);
                
                // Resize to tile size for fair comparison
                Mat cell_resized;
                resize(cell_img, cell_resized, Size(config.tile_size, config.tile_size));
                
                // Extract region colors from this cell
                TileFeatures cell_features = computeRegionColors(cell_resized);
                
                // Extract edge features from this cell
                cell_features.edges = computeEdgeFeatures(cell_resized);
                
                // Find best matching tile
                int tile_idx = findBestTile(cell_features);
                current_assignments[grid_idx] = tile_idx;
                
                // Place tile in mosaic
                Rect roi(x * config.tile_size, y * config.tile_size, 
                        config.tile_size, config.tile_size);
                tiles.images[tile_idx].copyTo(mosaic(roi));
            }
        }
        
        // Update cache for next frame
        previous_assignments = current_assignments;
        frame_count++;
        
        // Add FPS counter if enabled
        if (config.show_fps) {
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end_time - start_time);
            double fps = 1000.0 / duration.count();
            
            string fps_text = "FPS: " + to_string((int)fps) + 
                            " | Frame: " + to_string(frame_count) +
                            " | Threads: " + to_string(config.num_threads);
            putText(mosaic, fps_text, Point(20, 60), 
                   FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 255, 0), 4);
        }
        
        return mosaic;
    }
    
    // Process webcam stream
    void processWebcam(int camera_id = 0) {
        VideoCapture cap(camera_id);
        
        if (!cap.isOpened()) {
            cerr << "Error: Could not open webcam" << endl;
            return;
        }
        
        cout << "\n=== Video Mosaic Started ===" << endl;
        cout << "Camera: " << cap.get(CAP_PROP_FRAME_WIDTH) << "x" 
             << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        cout << "Grid: " << config.grid_width << "x" << config.grid_height << endl;
        cout << "Tiles: " << tiles.size() << endl;
        cout << "Threads: " << config.num_threads << endl;
        cout << "\nPress 'q' to quit, 's' to save frame, '+/-' to adjust threads" << endl;
        
        Mat frame, mosaic;
        int saved_count = 0;
        
        // For accurate FPS measurement
        auto last_frame_time = high_resolution_clock::now();
        double fps_avg = 0.0;
        int fps_count = 0;
        
        while (cap.read(frame)) {
            auto frame_start = high_resolution_clock::now();
            
            mosaic = generateMosaic(frame);
            
            auto frame_end = high_resolution_clock::now();
            auto frame_duration = duration_cast<milliseconds>(frame_end - frame_start);
            
            // Calculate actual FPS (full frame time)
            auto total_duration = duration_cast<milliseconds>(frame_end - last_frame_time);
            last_frame_time = frame_end;
            
            double current_fps = 1000.0 / max(1, (int)total_duration.count());
            fps_avg = (fps_avg * fps_count + current_fps) / (fps_count + 1);
            fps_count++;
            
            // Print timing every 30 frames
            if (fps_count % 30 == 0) {
                cout << "Processing: " << frame_duration.count() << "ms | "
                     << "Total: " << total_duration.count() << "ms | "
                     << "FPS: " << (int)current_fps << " | "
                     << "Avg FPS: " << (int)fps_avg << endl;
            }
            
            imshow("Mosaic", mosaic);
            
            int key = waitKey(1);
            if (key == 'q') break;
            else if (key == 's') {
                string filename = "mosaic_" + to_string(saved_count++) + ".png";
                imwrite(filename, mosaic);
                cout << "Saved: " << filename << endl;
            }
            else if (key == '+' && config.num_threads < 16) {
                config.num_threads++;
                omp_set_num_threads(config.num_threads);
                cout << "Threads: " << config.num_threads << endl;
            }
            else if (key == '-' && config.num_threads > 1) {
                config.num_threads--;
                omp_set_num_threads(config.num_threads);
                cout << "Threads: " << config.num_threads << endl;
            }
        }
        
        cap.release();
        destroyAllWindows();
        
        cout << "\nFinal average FPS: " << (int)fps_avg << endl;
        cout << "\nTotal frames processed: " << frame_count << endl;
    }
    
    // Process video file
    void processVideoFile(const string& input_path, const string& output_path = "") {
        VideoCapture cap(input_path);
        
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video file" << endl;
            return;
        }
        
        VideoWriter writer;
        if (!output_path.empty()) {
            int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
            double fps = cap.get(CAP_PROP_FPS);
            Size frame_size(config.grid_width * config.tile_size, 
                          config.grid_height * config.tile_size);
            writer.open(output_path, fourcc, fps, frame_size);
        }
        
        Mat frame, mosaic;
        
        while (cap.read(frame)) {
            mosaic = generateMosaic(frame);
            
            if (writer.isOpened()) {
                writer.write(mosaic);
            }
            
            imshow("Mosaic", mosaic);
            if (waitKey(1) == 'q') break;
        }
        
        cap.release();
        if (writer.isOpened()) writer.release();
        destroyAllWindows();
    }
};

void printUsage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "\nOptions:" << endl;
    cout << "  -d, --tiles DIR     Tile directory (default: pokemon_tiles)" << endl;
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
    // Default configuration
    VideoMosaicConfig config;
    config.tile_dir = "data/pokemon_tiles";
    config.tile_size = 32;
    config.grid_width = 60;   // Match optimal default
    config.grid_height = 45;
    config.num_threads = 4;
    config.temporal_coherence = true;
    config.coherence_threshold = 500.0;
    config.show_fps = true;
    config.use_histogram = false;  // Not used anymore (multi-region instead)
    config.color_blend = 0.0f;    // Default no blending (user can enable with -b)
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-d" || arg == "--tiles") {
            if (i + 1 < argc) config.tile_dir = argv[++i];
        }
        else if (arg == "-w" || arg == "--width") {
            if (i + 1 < argc) config.grid_width = atoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--height") {
            if (i + 1 < argc) config.grid_height = atoi(argv[++i]);
        }
        else if (arg == "-t" || arg == "--tile-size") {
            if (i + 1 < argc) config.tile_size = atoi(argv[++i]);
        }
        else if (arg == "-j" || arg == "--threads") {
            if (i + 1 < argc) config.num_threads = atoi(argv[++i]);
        }
        // Preset quality modes
        else if (arg == "--small") {
            config.grid_width = 40;
            config.grid_height = 30;
        }
        else if (arg == "--medium") {
            config.grid_width = 60;
            config.grid_height = 45;
        }
        else if (arg == "--large") {
            config.grid_width = 80;
            config.grid_height = 60;
        }
        else if (arg == "--ultra") {
            config.grid_width = 100;
            config.grid_height = 75;
        }
        else if (arg == "-f" || arg == "--show-fps") {
            config.show_fps = true;
        }
        else {
            cerr << "Unknown option: " << arg << endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Print configuration
    cout << "=== Configuration ===" << endl;
    cout << "Grid: " << config.grid_width << "x" << config.grid_height 
         << " (" << (config.grid_width * config.grid_height) << " cells)" << endl;
    cout << "Tile size: " << config.tile_size << "x" << config.tile_size << endl;
    cout << "Threads: " << config.num_threads << endl;
    cout << "=====================\n" << endl;
    
    // Create generator
    VideoMosaicGenerator generator(config);
    
    // Load tiles
    if (!generator.loadTiles()) {
        return -1;
    }
    
    // Process webcam
    generator.processWebcam(0);
    
    return 0;
}
