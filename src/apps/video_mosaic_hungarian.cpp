#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using namespace cv;
using namespace std;
using namespace chrono;

/**
 * Standard Hungarian Algorithm (Kuhn-Munkres) implementation.
 * Adapted for rectangular matrices (N rows, M columns, N <= M).
 * Complexity: O(N^2 * M)
 */
class HungarianAlgorithm {
public:
    void Solve(const vector<vector<float>>& costMatrix, vector<int>& assignment) {
        int n = costMatrix.size();
        if (n == 0) return;
        int m = costMatrix[0].size();
        
        assignment.assign(n, -1);
        
        // Potential for rows and columns
        vector<float> u(n + 1, 0.0f);
        vector<float> v(m + 1, 0.0f);
        vector<int> p(m + 1, 0);      // matching for columns
        vector<int> way(m + 1, 0);    // used to reconstruct path
        
        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            int j0 = 0;
            vector<float> minv(m + 1, numeric_limits<float>::max());
            vector<bool> used(m + 1, false);
            
            do {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                float delta = numeric_limits<float>::max();
                for (int j = 1; j <= m; ++j) {
                    if (!used[j]) {
                        float cur = costMatrix[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }
                for (int j = 0; j <= m; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);
            
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }
        
        for (int j = 1; j <= m; ++j) {
            if (p[j] != 0) {
                assignment[p[j] - 1] = j - 1;
            }
        }
    }
};

struct Tile {
    Mat image;
    Vec3f mean_color;
};

struct PhaseTiming {
    double preprocess_ms = 0;
    double matching_ms = 0;
    double render_ms = 0;
};

vector<Tile> loadTiles(const string& tile_dir, int tile_size) {
    vector<Tile> tiles;
    cout << "Loading tiles from: " << tile_dir << "..." << endl;
    
    if (!fs::exists(tile_dir) || !fs::is_directory(tile_dir)) {
        cerr << "Error: Tile directory not found!" << endl;
        return tiles;
    }
    
    vector<string> filenames;
    for (auto& entry : fs::directory_iterator(tile_dir)) {
        if (fs::is_regular_file(entry.status())) {
            string ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                filenames.push_back(entry.path().string());
            }
        }
    }
    
    tiles.resize(filenames.size());
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)filenames.size(); i++) {
        Mat img = imread(filenames[i]);
        if (img.empty()) continue;
        
        Mat resized;
        resize(img, resized, Size(tile_size, tile_size));
        
        Tile t;
        t.image = resized;
        Scalar m = mean(resized);
        t.mean_color = Vec3f(m[0], m[1], m[2]);
        tiles[i] = t;
    }
    
    cout << "Loaded " << tiles.size() << " tiles." << endl;
    return tiles;
}

void processPart(int thread_id, int num_splits, const Mat& frame, const vector<Tile>& tiles, 
                 int grid_w, int grid_h, int tile_size, Mat& output, PhaseTiming& timing) {
    
    auto t_start = high_resolution_clock::now();
    
    int rows_per_split = grid_h / num_splits;
    int start_y = thread_id * rows_per_split;
    int end_y = (thread_id == num_splits - 1) ? grid_h : (thread_id + 1) * rows_per_split;
    
    int num_cells = (end_y - start_y) * grid_w;
    int num_tiles = tiles.size();
    
    // 1. Preprocess: Build cost matrix
    vector<vector<float>> cost_matrix(num_cells, vector<float>(num_tiles));
    
    int cell_ptr = 0;
    for (int gy = start_y; gy < end_y; gy++) {
        for (int gx = 0; gx < grid_w; gx++) {
            Rect roi(gx * (frame.cols / grid_w), gy * (frame.rows / grid_h), 
                    frame.cols / grid_w, frame.rows / grid_h);
            Mat cell = frame(roi);
            Scalar m = mean(cell);
            Vec3f cell_color(m[0], m[1], m[2]);
            
            for (int t = 0; t < num_tiles; t++) {
                Vec3f diff = cell_color - tiles[t].mean_color;
                cost_matrix[cell_ptr][t] = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
            }
            cell_ptr++;
        }
    }
    
    auto t_match_start = high_resolution_clock::now();
    timing.preprocess_ms = duration_cast<microseconds>(t_match_start - t_start).count() / 1000.0;
    
    // 2. Matching: Solve Hungarian
    HungarianAlgorithm hungarian;
    vector<int> assignment;
    hungarian.Solve(cost_matrix, assignment);
    
    auto t_render_start = high_resolution_clock::now();
    timing.matching_ms = duration_cast<microseconds>(t_render_start - t_match_start).count() / 1000.0;
    
    // 3. Render: Place tiles
    cell_ptr = 0;
    for (int gy = start_y; gy < end_y; gy++) {
        for (int gx = 0; gx < grid_w; gx++) {
            int tile_idx = assignment[cell_ptr++];
            if (tile_idx != -1) {
                Rect roi(gx * tile_size, gy * tile_size, tile_size, tile_size);
                tiles[tile_idx].image.copyTo(output(roi));
            }
        }
    }
    
    auto t_end = high_resolution_clock::now();
    timing.render_ms = duration_cast<microseconds>(t_end - t_render_start).count() / 1000.0;
}

int main(int argc, char** argv) {
    string video_path = "data/test.mp4";
    string tile_dir = "data/cifar_tiles";
    int tile_size = 32;
    // Use 64x16 to make it easily splittable up to 64 parts
    int grid_w = 16;
    int grid_h = 64; 
    
    vector<Tile> tiles = loadTiles(tile_dir, tile_size);
    if (tiles.empty()) return -1;
    
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video " << video_path << endl;
        return -1;
    }
    Mat frame;
    cap >> frame;
    if (frame.empty()) return -1;
    
    vector<int> sweep_splits = {1, 2, 4, 8, 16, 32, 64};
    
    cout << "\n=== Hungarian Video Mosaic Profiling Sweep ===" << endl;
    cout << "Grid: " << grid_w << "x" << grid_h << " (" << (grid_w * grid_h) << " cells)" << endl;
    cout << "Tiles: " << tiles.size() << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << setw(8) << "Splits" << " | " 
         << setw(12) << "Preproc (ms)" << " | " 
         << setw(12) << "Match (ms)" << " | " 
         << setw(12) << "Render (ms)" << " | " 
         << setw(10) << "Total (ms)" << endl;
    cout << "------------------------------------------------------------" << endl;
    
    for (int num_splits : sweep_splits) {
        Mat output(grid_h * tile_size, grid_w * tile_size, CV_8UC3, Scalar(0,0,0));
        vector<PhaseTiming> thread_timings(num_splits);
        vector<thread> threads;
        
        auto start_all = high_resolution_clock::now();
        
        for (int i = 0; i < num_splits; i++) {
            threads.emplace_back(processPart, i, num_splits, ref(frame), ref(tiles), 
                                grid_w, grid_h, tile_size, ref(output), ref(thread_timings[i]));
        }
        
        for (auto& t : threads) t.join();
        
        auto end_all = high_resolution_clock::now();
        double total_ms = duration_cast<microseconds>(end_all - start_all).count() / 1000.0;
        
        // Find max timing across threads for each phase (bottleneck)
        double max_pre = 0, max_match = 0, max_render = 0;
        for (auto& t : thread_timings) {
            max_pre = max(max_pre, t.preprocess_ms);
            max_match = max(max_match, t.matching_ms);
            max_render = max(max_render, t.render_ms);
        }
        
        cout << setw(8) << num_splits << " | " 
             << setw(12) << fixed << setprecision(2) << max_pre << " | " 
             << setw(12) << max_match << " | " 
             << setw(12) << max_render << " | " 
             << setw(10) << total_ms << endl;
             
        if (num_splits == 4) {
            imwrite("output_mosaic_4.png", output);
        }
    }
    
    cout << "------------------------------------------------------------" << endl;
    cout << "Sweep complete." << endl;
    
    return 0;
}
