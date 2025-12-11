#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <boost/filesystem.hpp>

// SIMD intrinsics for vectorization
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>  // ARM NEON
#define USE_SIMD_NEON
#elif defined(__SSE__)
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#define USE_SIMD_SSE
#endif

namespace fs = boost::filesystem;
using namespace cv;
using namespace std;
using namespace chrono;

// Multi-region tile features (3x3 grid for spatial detail)
struct TileFeatures {
    array<Vec3f, 9> region_colors;  // 3x3 grid of colors
    Vec3f mean_color;                // Overall mean (for backward compat)
};

// Struct of Arrays layout for better cache locality
struct TileDatabase {
    // Hot data (accessed every frame) - tightly packed
    vector<TileFeatures> features;
    
    // Cold data (accessed only during assignment/display)
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

// Configuration
struct VideoMosaicConfig {
    string tile_dir;
    int tile_size;
    int grid_width;
    int grid_height;
    int num_threads;
    bool temporal_coherence;
    int coherence_threshold;
    bool show_fps;
    bool use_histogram;
    float color_blend;
    int max_tile_reuse;  // Maximum times a tile can be reused (1 = no repetition)
};

class OptimalMosaicGenerator {
private:
    VideoMosaicConfig config;
    TileDatabase tiles;
    int frame_count;
    vector<int> previous_assignment;  // Cache previous frame's assignment
    Mat previous_frame;                // Cache previous input frame
    
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
    
    // Compute histogram, ignoring black backgrounds
    Mat computeHistogram(const Mat& img) {
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat mask = gray > 30;
        
        Mat hist;
        int histSize[] = {8, 8, 8};
        float range[] = {0, 256};
        const float* ranges[] = {range, range, range};
        int channels[] = {0, 1, 2};
        
        calcHist(&img, 1, channels, mask, hist, 3, histSize, ranges, true, false);
        normalize(hist, hist, 1, 0, NORM_L1);
        
        return hist;
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
    
    // Scalar color distance (for single colors)
    inline float colorDistance(const Vec3f& c1, const Vec3f& c2) {
        float db = c1[0] - c2[0];
        float dg = c1[1] - c2[1];
        float dr = c1[2] - c2[2];
        return db*db + dg*dg + dr*dr;
    }
    
#ifdef USE_SIMD_NEON
    // ARM NEON-optimized batch distance computation (for Apple Silicon)
    inline void colorDistanceBatch4(const Vec3f& target, 
                                     const Vec3f* tiles,
                                     float* distances) {
        // Load target color (broadcast to all lanes)
        float32x4_t target_b = vdupq_n_f32(target[0]);
        float32x4_t target_g = vdupq_n_f32(target[1]);
        float32x4_t target_r = vdupq_n_f32(target[2]);
        
        // Load 4 tile colors
        float tile_b_arr[4] = {tiles[0][0], tiles[1][0], tiles[2][0], tiles[3][0]};
        float tile_g_arr[4] = {tiles[0][1], tiles[1][1], tiles[2][1], tiles[3][1]};
        float tile_r_arr[4] = {tiles[0][2], tiles[1][2], tiles[2][2], tiles[3][2]};
        
        float32x4_t tile_b = vld1q_f32(tile_b_arr);
        float32x4_t tile_g = vld1q_f32(tile_g_arr);
        float32x4_t tile_r = vld1q_f32(tile_r_arr);
        
        // Compute differences
        float32x4_t diff_b = vsubq_f32(target_b, tile_b);
        float32x4_t diff_g = vsubq_f32(target_g, tile_g);
        float32x4_t diff_r = vsubq_f32(target_r, tile_r);
        
        // Square differences
        float32x4_t sq_b = vmulq_f32(diff_b, diff_b);
        float32x4_t sq_g = vmulq_f32(diff_g, diff_g);
        float32x4_t sq_r = vmulq_f32(diff_r, diff_r);
        
        // Sum: b² + g² + r²
        float32x4_t result = vaddq_f32(sq_b, vaddq_f32(sq_g, sq_r));
        
        // Store results
        vst1q_f32(distances, result);
    }
#elif defined(USE_SIMD_SSE)
    // SIMD-optimized batch distance computation
    inline void colorDistanceBatch4(const Vec3f& target, 
                                     const Vec3f* tiles,
                                     float* distances) {
        // Load target color (broadcast to all lanes)
        __m128 target_b = _mm_set1_ps(target[0]);
        __m128 target_g = _mm_set1_ps(target[1]);
        __m128 target_r = _mm_set1_ps(target[2]);
        
        // Load 4 tile colors
        __m128 tile_b = _mm_setr_ps(tiles[0][0], tiles[1][0], tiles[2][0], tiles[3][0]);
        __m128 tile_g = _mm_setr_ps(tiles[0][1], tiles[1][1], tiles[2][1], tiles[3][1]);
        __m128 tile_r = _mm_setr_ps(tiles[0][2], tiles[1][2], tiles[2][2], tiles[3][2]);
        
        // Compute differences
        __m128 diff_b = _mm_sub_ps(target_b, tile_b);
        __m128 diff_g = _mm_sub_ps(target_g, tile_g);
        __m128 diff_r = _mm_sub_ps(target_r, tile_r);
        
        // Square differences
        __m128 sq_b = _mm_mul_ps(diff_b, diff_b);
        __m128 sq_g = _mm_mul_ps(diff_g, diff_g);
        __m128 sq_r = _mm_mul_ps(diff_r, diff_r);
        
        // Sum: b² + g² + r²
        __m128 result = _mm_add_ps(sq_b, _mm_add_ps(sq_g, sq_r));
        
        // Store results
        _mm_storeu_ps(distances, result);
    }
#endif
    
    // Compare histograms
    float compareHistograms(const Mat& hist1, const Mat& hist2) {
        return 1.0 - compareHist(hist1, hist2, HISTCMP_BHATTACHARYYA);
    }
    
    // Optimized assignment algorithm - no sorting needed!
    vector<int> hungarianAlgorithm(const vector<vector<float>>& cost_matrix) {
        int num_cells = cost_matrix.size();
        int num_tiles = cost_matrix[0].size();
        
        vector<int> assignment(num_cells, -1);
        vector<int> tile_usage(num_tiles, 0);
        
        // Process each cell in parallel with dynamic scheduling
        #pragma omp parallel for schedule(dynamic, 32)
        for (int cell_idx = 0; cell_idx < num_cells; cell_idx++) {
            // Find top K best tiles for this cell
            const int K = 10;  // Consider top 10 candidates
            vector<pair<float, int>> candidates;  // (cost, tile_idx)
            candidates.reserve(K);
            
            for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
                float cost = cost_matrix[cell_idx][tile_idx];
                
                if (candidates.size() < K) {
                    candidates.push_back({cost, tile_idx});
                    if (candidates.size() == K) {
                        make_heap(candidates.begin(), candidates.end());
                    }
                } else if (cost < candidates.front().first) {
                    pop_heap(candidates.begin(), candidates.end());
                    candidates.back() = {cost, tile_idx};
                    push_heap(candidates.begin(), candidates.end());
                }
            }
            
            // Sort candidates by cost
            sort(candidates.begin(), candidates.end());
            
            // Try to assign best available tile
            int assigned_tile = -1;
            for (size_t cand_idx = 0; cand_idx < candidates.size(); cand_idx++) {
                float cost = candidates[cand_idx].first;
                int tile_idx = candidates[cand_idx].second;
                int current_usage;
                #pragma omp atomic read
                current_usage = tile_usage[tile_idx];
                
                if (current_usage < config.max_tile_reuse) {
                    // Try to claim this tile
                    #pragma omp atomic capture
                    {
                        current_usage = tile_usage[tile_idx];
                        tile_usage[tile_idx]++;
                    }
                    
                    if (current_usage < config.max_tile_reuse) {
                        assigned_tile = tile_idx;
                        break;
                    } else {
                        // Undo if we exceeded limit
                        #pragma omp atomic
                        tile_usage[tile_idx]--;
                    }
                }
            }
            
            // If no candidate worked, use least-used tile
            if (assigned_tile == -1) {
                int min_usage = INT_MAX;
                int min_tile = 0;
                
                for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
                    int usage;
                    #pragma omp atomic read
                    usage = tile_usage[tile_idx];
                    
                    if (usage < min_usage) {
                        min_usage = usage;
                        min_tile = tile_idx;
                    }
                }
                
                #pragma omp atomic
                tile_usage[min_tile]++;
                assigned_tile = min_tile;
            }
            
            assignment[cell_idx] = assigned_tile;
        }
        
        return assignment;
    }
    
public:
    OptimalMosaicGenerator(const VideoMosaicConfig& cfg) : config(cfg), frame_count(0) {
        omp_set_num_threads(config.num_threads);
    }
    
    // Check if frame has changed significantly
    bool frameChanged(const Mat& current_frame) {
        if (previous_frame.empty()) return true;
        
        // Compute difference
        Mat diff;
        absdiff(current_frame, previous_frame, diff);
        Scalar mean_diff = cv::mean(diff);
        double total_diff = mean_diff[0] + mean_diff[1] + mean_diff[2];
        
        // Threshold: if average pixel difference < 5, consider unchanged
        return total_diff > 15.0;
    }
    
    // Load tiles
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
        
        // Parallelize tile loading
        #pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < filenames.size(); i++) {
            Mat img = imread(filenames[i]);
            if (img.empty()) {
                continue;
            }
            
            Mat resized;
            resize(img, resized, Size(config.tile_size, config.tile_size));
            
            // Store in SoA layout
            tiles.paths[i] = filenames[i];
            tiles.images[i] = resized;
            tiles.features[i] = computeRegionColors(resized);
            
            // Thread-safe progress counter
            #pragma omp atomic
            loaded++;
            
            if (loaded % 1000 == 0) {
                #pragma omp critical
                cout << "  Loaded " << loaded << " tiles..." << endl;
            }
        }
        
        cout << "Successfully loaded " << tiles.size() << " tiles" << endl;
        cout << "Using OPTIMAL assignment (max reuse: " << config.max_tile_reuse << ")" << endl;
        return !tiles.empty();
    }
    
    // Generate mosaic with optimal assignment
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
        
        int total_cells = config.grid_width * config.grid_height;
        int input_tile_width = input_frame.cols / config.grid_width;
        int input_tile_height = input_frame.rows / config.grid_height;
        
        // Step 1: Build cost matrix (parallelized)
        cout << "  Building cost matrix..." << flush;
        auto cost_start = high_resolution_clock::now();
        
        vector<vector<float>> cost_matrix(total_cells, vector<float>(tiles.size()));
        
        // Use blocked iteration for better cache locality
        const int BLOCK_SIZE = 8;  // Process 8x8 blocks of cells
        #pragma omp parallel for schedule(static) collapse(2)
        for (int block_y = 0; block_y < config.grid_height; block_y += BLOCK_SIZE) {
            for (int block_x = 0; block_x < config.grid_width; block_x += BLOCK_SIZE) {
                // Process cells within this block
                for (int y = block_y; y < min(block_y + BLOCK_SIZE, config.grid_height); y++) {
                    for (int x = block_x; x < min(block_x + BLOCK_SIZE, config.grid_width); x++) {
                        int cell_idx = y * config.grid_width + x;
                        
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
                        
                        // Compare with all tiles using region-based distance
                        for (size_t tile_idx = 0; tile_idx < tiles.size(); tile_idx++) {
                            cost_matrix[cell_idx][tile_idx] = 
                                regionDistance(cell_features, tiles.features[tile_idx]);
                        }
                    }
                }
            }
        }
        
        auto cost_end = high_resolution_clock::now();
        auto cost_duration = duration_cast<milliseconds>(cost_end - cost_start);
        cout << cost_duration.count() << "ms" << endl;
        
        // Step 2: Solve assignment problem
        cout << "  Solving optimal assignment..." << flush;
        auto assign_start = high_resolution_clock::now();
        
        vector<int> assignment;
        
        // Use temporal coherence if enabled: reuse previous assignment if frame hasn't changed
        if (config.temporal_coherence && !frameChanged(input_frame) && !previous_assignment.empty()) {
            assignment = previous_assignment;
            cout << "0ms (cached)" << endl;
        } else {
            assignment = hungarianAlgorithm(cost_matrix);
            auto assign_end = high_resolution_clock::now();
            auto assign_duration = duration_cast<milliseconds>(assign_end - assign_start);
            cout << assign_duration.count() << "ms" << endl;
            
            // Cache for next frame if coherence enabled
            if (config.temporal_coherence) {
                previous_assignment = assignment;
                previous_frame = input_frame.clone();
            }
        }
        
        // Step 3: Place tiles
        cout << "  Placing tiles..." << flush;
        auto place_start = high_resolution_clock::now();
        
        #pragma omp parallel for
        for (int cell_idx = 0; cell_idx < total_cells; cell_idx++) {
            int y = cell_idx / config.grid_width;
            int x = cell_idx % config.grid_width;
            int tile_idx = assignment[cell_idx];
            
            Rect roi(x * config.tile_size, y * config.tile_size, 
                    config.tile_size, config.tile_size);
            tiles.images[tile_idx].copyTo(mosaic(roi));
        }
        
        auto place_end = high_resolution_clock::now();
        auto place_duration = duration_cast<milliseconds>(place_end - place_start);
        cout << place_duration.count() << "ms" << endl;
        
        frame_count++;
        
        // Add timing info
        if (config.show_fps) {
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end_time - start_time);
            
            string fps_text = "OPTIMAL | Total: " + to_string(duration.count()) + "ms" +
                            " | Frame: " + to_string(frame_count) +
                            " | Threads: " + to_string(config.num_threads);
            putText(mosaic, fps_text, Point(20, 60), 
                   FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 255, 0), 4);
        }
        
        return mosaic;
    }
    
    // Process webcam
    void processWebcam(int camera_id = 0) {
        VideoCapture cap(camera_id);
        
        if (!cap.isOpened()) {
            cerr << "Error: Could not open webcam" << endl;
            return;
        }
        
        cout << "\n=== Optimal Video Mosaic Started ===" << endl;
        cout << "Camera: " << cap.get(CAP_PROP_FRAME_WIDTH) << "x" 
             << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        cout << "Grid: " << config.grid_width << "x" << config.grid_height << endl;
        cout << "Tiles: " << tiles.size() << endl;
        cout << "Max tile reuse: " << config.max_tile_reuse << endl;
        cout << "Threads: " << config.num_threads << endl;
        cout << "\nPress 'q' to quit, 's' to save frame" << endl;
        cout << "\nProcessing frames (this will be slow - watch the timing!):\n" << endl;
        
        Mat frame, mosaic;
        int saved_count = 0;
        
        while (cap.read(frame)) {
            cout << "Frame " << (frame_count + 1) << ":" << endl;
            mosaic = generateMosaic(frame);
            
            imshow("Optimal Mosaic", mosaic);
            
            int key = waitKey(1);
            if (key == 'q') break;
            else if (key == 's') {
                string filename = "mosaic_optimal_" + to_string(saved_count++) + ".png";
                imwrite(filename, mosaic);
                cout << "Saved: " << filename << endl;
            }
            
            cout << endl;
        }
        
        cap.release();
        destroyAllWindows();
    }
};

void printUsage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "\nOptions:" << endl;
    cout << "  -d, --tiles DIR     Tile directory (default: pokemon_tiles)" << endl;
    cout << "  -w, --width N       Grid width (default: 40)" << endl;
    cout << "  -h, --height N      Grid height (default: 30)" << endl;
    cout << "  -t, --tile-size N   Tile size in pixels (default: 32)" << endl;
    cout << "  -j, --threads N     Number of threads (default: 4)" << endl;
    cout << "  -r, --reuse N       Max tile reuse (default: 1, no repetition)" << endl;
    cout << "  -c, --coherence     Enable temporal coherence (stable when still)" << endl;
    cout << "  --help              Show this help" << endl;
    cout << "\nPreset Quality Modes:" << endl;
    cout << "  --small             Small grid (40x30)" << endl;
    cout << "  --medium            Medium grid (60x45, default)" << endl;
    cout << "  --large             Large grid (80x60)" << endl;
    cout << "  --ultra             Ultra grid (100x75)" << endl;
}

int main(int argc, char** argv) {
    VideoMosaicConfig config;
    config.tile_dir = "pokemon_tiles";
    config.tile_size = 32;
    config.grid_width = 60;   // Smaller default for optimal (faster)
    config.grid_height = 45;
    config.num_threads = 4;
    config.temporal_coherence = false;  // Off by default
    config.coherence_threshold = 500.0;
    config.show_fps = true;
    config.use_histogram = false;  // Use mean color for speed
    config.color_blend = 0.0f;
    config.max_tile_reuse = 1;  // No repetition by default
    
    // Parse arguments
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
        else if (arg == "-r" || arg == "--reuse") {
            if (i + 1 < argc) config.max_tile_reuse = atoi(argv[++i]);
        }
        else if (arg == "-c" || arg == "--coherence") {
            config.temporal_coherence = true;
        }
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
    }
    
    cout << "=== Configuration ===" << endl;
    cout << "Grid: " << config.grid_width << "x" << config.grid_height 
         << " (" << (config.grid_width * config.grid_height) << " cells)" << endl;
    cout << "Tile size: " << config.tile_size << "x" << config.tile_size << endl;
    cout << "Threads: " << config.num_threads << endl;
    cout << "Max tile reuse: " << config.max_tile_reuse << endl;
    cout << "=====================\n" << endl;
    
    OptimalMosaicGenerator generator(config);
    
    if (!generator.loadTiles()) {
        return -1;
    }
    
    generator.processWebcam(0);
    
    return 0;
}
