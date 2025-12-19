#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <limits>
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
        
        // Reconstruct assignment: assignment[i-1] = j-1 means row i-1 is matched to column j-1
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

// Function performed by each thread
void processPart(int thread_id, const Mat& frame, const vector<Tile>& tiles, 
                 int grid_w, int grid_h, int tile_size, Mat& output) {
    int start_y = (grid_h * thread_id) / 4;
    int end_y = (grid_h * (thread_id + 1)) / 4;
    
    int num_cells = (end_y - start_y) * grid_w;
    int num_tiles = tiles.size();
    
    cout << "Thread " << thread_id << " processing " << num_cells << " cells..." << endl;
    
    // 1. Build cost matrix for this part
    // cost_matrix[cell_idx][tile_idx]
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
    
    // 2. Solve Hungarian
    HungarianAlgorithm hungarian;
    vector<int> assignment;
    hungarian.Solve(cost_matrix, assignment);
    
    // 3. Render tiles to output
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
    
    cout << "Thread " << thread_id << " finished." << endl;
}

int main(int argc, char** argv) {
    string video_path = "data/test.mp4";
    string tile_dir = "data/cifar_tiles";
    int tile_size = 32;
    int grid_w = 40;
    int grid_h = 40; // Total 1600 cells. 4 parts of 400 cells each.
    
    // Load tiles
    vector<Tile> tiles = loadTiles(tile_dir, tile_size);
    if (tiles.empty()) return -1;
    
    // Since Hungarian is O(N^2 * M), 60k tiles and 400 cells is feasible but slow.
    // 400^2 * 60,000 = 160,000 * 60,000 = 9.6 billion operations.
    // This should take about 10-30 seconds per thread.
    
    // Read one frame
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video " << video_path << endl;
        return -1;
    }
    
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: Could not read frame from " << video_path << endl;
        return -1;
    }
    
    cout << "Original frame size: " << frame.cols << "x" << frame.rows << endl;
    
    // Create output mosaic image
    Mat output(grid_h * tile_size, grid_w * tile_size, CV_8UC3, Scalar(0,0,0));
    
    // Launch 4 threads
    cout << "Starting 4 threads for Hungarian assignment..." << endl;
    auto start_time = high_resolution_clock::now();
    
    vector<thread> threads;
    for (int i = 0; i < 4; i++) {
        threads.emplace_back(processPart, i, ref(frame), ref(tiles), 
                            grid_w, grid_h, tile_size, ref(output));
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);
    cout << "Total time: " << duration.count() << " seconds." << endl;
    
    // Save output
    string output_path = "output_mosaic.png";
    imwrite(output_path, output);
    cout << "Saved mosaic to: " << output_path << endl;
    
    return 0;
}
