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

// Tile information structure
struct Tile {
    string path;
    Mat image;
    Vec3f mean_color;      // Fast matching: average color
    Mat histogram;         // Better matching: color histogram (8x8x8 bins)
    int last_used_frame;   // For temporal coherence
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
    vector<Tile> tiles;
    VideoMosaicConfig config;
    vector<int> previous_assignments;  // Cache previous frame's tile assignments
    int frame_count;
    
    // Compute mean color of an image, ignoring black/transparent backgrounds
    Vec3f computeMeanColor(const Mat& img) {
        // Create mask for non-black pixels (brightness > 30)
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat mask = gray > 30;  // Ignore very dark pixels (transparent backgrounds)
        
        Scalar mean = cv::mean(img, mask);
        
        // If all pixels are black, fall back to regular mean
        if (countNonZero(mask) == 0) {
            mean = cv::mean(img);
        }
        
        return Vec3f(mean[0], mean[1], mean[2]);
    }
    
    // Compute color histogram (8x8x8 bins = 512 bins), ignoring black backgrounds
    Mat computeHistogram(const Mat& img) {
        // Create mask for non-black pixels
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Mat mask = gray > 30;  // Ignore very dark pixels
        
        Mat hist;
        int histSize[] = {8, 8, 8};  // 8 bins per channel
        float range[] = {0, 256};
        const float* ranges[] = {range, range, range};
        int channels[] = {0, 1, 2};
        
        calcHist(&img, 1, channels, mask, hist, 3, histSize, ranges, true, false);
        normalize(hist, hist, 1, 0, NORM_L1);  // Normalize to sum=1
        
        return hist;
    }
    
    // Compare two histograms using Bhattacharyya distance (lower = more similar)
    float compareHistograms(const Mat& hist1, const Mat& hist2) {
        return 1.0 - compareHist(hist1, hist2, HISTCMP_BHATTACHARYYA);
    }
    
    // Fast color distance (squared Euclidean, no sqrt)
    inline float colorDistance(const Vec3f& c1, const Vec3f& c2) {
        float db = c1[0] - c2[0];
        float dg = c1[1] - c2[1];
        float dr = c1[2] - c2[2];
        return db*db + dg*dg + dr*dr;
    }
    
    // Find best matching tile for a color (mean color matching - fast)
    int findBestTile(const Vec3f& target_color, int grid_idx) {
        int best_idx = 0;
        float best_dist = FLT_MAX;
        
        // If temporal coherence enabled, check if previous tile is still good
        if (config.temporal_coherence && frame_count > 0 && 
            grid_idx < previous_assignments.size()) {
            int prev_idx = previous_assignments[grid_idx];
            float prev_dist = colorDistance(target_color, tiles[prev_idx].mean_color);
            
            if (prev_dist < config.coherence_threshold) {
                return prev_idx;  // Reuse previous tile
            }
        }
        
        // Find best matching tile using mean color
        for (size_t i = 0; i < tiles.size(); i++) {
            float dist = colorDistance(target_color, tiles[i].mean_color);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
        
        return best_idx;
    }
    
    // Find best matching tile using histogram (more accurate but slower)
    int findBestTileHistogram(const Mat& target_hist, int grid_idx) {
        int best_idx = 0;
        float best_similarity = -FLT_MAX;
        
        // If temporal coherence enabled, check if previous tile is still good
        if (config.temporal_coherence && frame_count > 0 && 
            grid_idx < previous_assignments.size()) {
            int prev_idx = previous_assignments[grid_idx];
            float prev_sim = compareHistograms(target_hist, tiles[prev_idx].histogram);
            
            if (prev_sim > 0.8) {  // High similarity threshold
                return prev_idx;
            }
        }
        
        // Find best matching tile using histogram comparison
        for (size_t i = 0; i < tiles.size(); i++) {
            float similarity = compareHistograms(target_hist, tiles[i].histogram);
            if (similarity > best_similarity) {
                best_similarity = similarity;
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
        
        tiles.reserve(filenames.size());
        
        int loaded = 0;
        // Load and preprocess tiles
        for (const auto& filename : filenames) {
            Mat img = imread(filename);
            if (img.empty()) {
                cerr << "Warning: Could not load " << filename << endl;
                continue;
            }
            
            // Resize to tile size
            Mat resized;
            resize(img, resized, Size(config.tile_size, config.tile_size));
            
            Tile tile;
            tile.path = filename;
            tile.image = resized;
            tile.mean_color = computeMeanColor(resized);
            
            // Compute histogram if enabled
            if (config.use_histogram) {
                tile.histogram = computeHistogram(resized);
            }
            
            tile.last_used_frame = -1;
            
            tiles.push_back(tile);
            loaded++;
            
            // Progress indicator
            if (loaded % 100 == 0) {
                cout << "  Loaded " << loaded << " tiles..." << endl;
            }
        }
        
        cout << "Successfully loaded " << tiles.size() << " tiles" << endl;
        if (config.use_histogram) {
            cout << "Using histogram matching (accurate mode)" << endl;
        } else {
            cout << "Using mean color matching (fast mode)" << endl;
        }
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
                int tile_idx;
                
                if (config.use_histogram) {
                    // Histogram matching - use actual tile-sized region from input
                    int x_start = x * input_tile_width;
                    int y_start = y * input_tile_height;
                    int width = min(input_tile_width, input_frame.cols - x_start);
                    int height = min(input_tile_height, input_frame.rows - y_start);
                    
                    Rect roi(x_start, y_start, width, height);
                    Mat cell = input_frame(roi);
                    Mat target_hist = computeHistogram(cell);
                    tile_idx = findBestTileHistogram(target_hist, grid_idx);
                } else {
                    // Mean color matching - use resized input for speed
                    Vec3b pixel = resized_input.at<Vec3b>(y, x);
                    Vec3f target_color(pixel[0], pixel[1], pixel[2]);
                    tile_idx = findBestTile(target_color, grid_idx);
                }
                
                current_assignments[grid_idx] = tile_idx;
                
                // Place tile in mosaic with optional color blending
                Rect roi(x * config.tile_size, y * config.tile_size, 
                        config.tile_size, config.tile_size);
                
                if (config.color_blend > 0.0f) {
                    // Apply color blending: tint tile toward target color
                    // Get target color for this region
                    int x_start = x * input_tile_width;
                    int y_start = y * input_tile_height;
                    int width = min(input_tile_width, input_frame.cols - x_start);
                    int height = min(input_tile_height, input_frame.rows - y_start);
                    Rect target_roi(x_start, y_start, width, height);
                    Scalar target_mean = cv::mean(input_frame(target_roi));
                    
                    // Blend tile with target color
                    Mat blended_tile = tiles[tile_idx].image.clone();
                    Scalar tile_mean = cv::mean(blended_tile);
                    
                    // Adjust each pixel
                    for (int ty = 0; ty < blended_tile.rows; ty++) {
                        for (int tx = 0; tx < blended_tile.cols; tx++) {
                            Vec3b& pixel = blended_tile.at<Vec3b>(ty, tx);
                            for (int c = 0; c < 3; c++) {
                                float tile_val = pixel[c];
                                float target_val = target_mean[c];
                                float tile_avg = tile_mean[c];
                                
                                // Preserve relative brightness while shifting hue
                                float diff = tile_val - tile_avg;
                                float new_val = target_val + diff * (1.0f - config.color_blend);
                                pixel[c] = saturate_cast<uchar>(new_val);
                            }
                        }
                    }
                    
                    blended_tile.copyTo(mosaic(roi));
                } else {
                    // No blending, use tile as-is
                    tiles[tile_idx].image.copyTo(mosaic(roi));
                }
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
    cout << "  -w, --width N       Grid width (default: 40)" << endl;
    cout << "  -h, --height N      Grid height (default: 30)" << endl;
    cout << "  -t, --tile-size N   Tile size in pixels (default: 32)" << endl;
    cout << "  -j, --threads N     Number of threads (default: 4)" << endl;
    cout << "  -b, --blend N       Color blend 0.0-1.0 (default: 0.3, 0=off)" << endl;
    cout << "  -f, --fast          Use fast mode (mean color)" << endl;
    cout << "  -a, --accurate      Use accurate mode (histogram, default)" << endl;
    cout << "  --help              Show this help" << endl;
    cout << "\nPreset Quality Modes:" << endl;
    cout << "  --low               Low quality, high FPS (20x15, 16px tiles)" << endl;
    cout << "  --medium            Balanced (40x30, 32px tiles, default)" << endl;
    cout << "  --high              High quality (60x45, 48px tiles)" << endl;
    cout << "  --ultra             Ultra quality (80x60, 64px tiles)" << endl;
    cout << "\nExamples:" << endl;
    cout << "  " << program_name << "                    # Default settings" << endl;
    cout << "  " << program_name << " --high             # High quality preset" << endl;
    cout << "  " << program_name << " -w 60 -h 45        # Custom resolution" << endl;
    cout << "  " << program_name << " -f -j 8            # Fast mode, 8 threads" << endl;
}

int main(int argc, char** argv) {
    // Default configuration
    VideoMosaicConfig config;
    config.tile_dir = "pokemon_tiles";
    config.tile_size = 32;
    config.grid_width = 40;
    config.grid_height = 30;
    config.num_threads = 4;
    config.temporal_coherence = true;
    config.coherence_threshold = 500.0;
    config.show_fps = true;
    config.use_histogram = true;  // Default to accurate mode
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
        else if (arg == "-b" || arg == "--blend") {
            if (i + 1 < argc) config.color_blend = atof(argv[++i]);
        }
        else if (arg == "-f" || arg == "--fast") {
            config.use_histogram = false;
        }
        else if (arg == "-a" || arg == "--accurate") {
            config.use_histogram = true;
        }
        // Preset quality modes
        else if (arg == "--low") {
            config.grid_width = 20;
            config.grid_height = 15;
            config.tile_size = 16;
            config.use_histogram = false;  // Fast mode for low quality
        }
        else if (arg == "--medium") {
            config.grid_width = 40;
            config.grid_height = 30;
            config.tile_size = 32;
        }
        else if (arg == "--high") {
            config.grid_width = 60;
            config.grid_height = 45;
            config.tile_size = 48;
        }
        else if (arg == "--ultra") {
            config.grid_width = 80;
            config.grid_height = 60;
            config.tile_size = 64;
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
         << " (" << (config.grid_width * config.grid_height) << " tiles/frame)" << endl;
    cout << "Tile size: " << config.tile_size << "x" << config.tile_size << " pixels" << endl;
    cout << "Output: " << (config.grid_width * config.tile_size) << "x" 
         << (config.grid_height * config.tile_size) << " pixels" << endl;
    cout << "Threads: " << config.num_threads << endl;
    cout << "Mode: " << (config.use_histogram ? "Accurate (histogram)" : "Fast (mean color)") << endl;
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
