#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <boost/filesystem.hpp>
#include "gpu/metal_compute.h"

namespace fs = boost::filesystem;
using namespace cv;
using namespace std;
using namespace chrono;

struct EdgeFeatures {
    float edge_strength;
    array<float, 4> edge_histogram;
};

struct TileFeatures {
    array<Vec3f, 9> region_colors;
    Vec3f mean_color;
    EdgeFeatures edges;
};

// Simplified Database for Host (Metal has its own copy)
struct TileDatabase {
    vector<Mat> images; // For display copy
    vector<TileFeatures> features; // For initialization upload
    size_t size() const { return images.size(); }
    void resize(size_t n) { images.resize(n); features.resize(n); }
};

struct VideoMosaicConfig {
    string tile_dir;
    int tile_size;
    int grid_width;
    int grid_height;
    bool show_fps;
};

class VideoMosaicGeneratorMetal {
private:
    TileDatabase tiles;
    VideoMosaicConfig config;
    int frame_count;
    
    // Metal Engine
    MetalComputeEngine metalEngine;
    
    // Global Feature Maps (Same as Optimized CPU)
    Mat global_gray, global_grad_x, global_grad_y, global_magnitude, global_direction;
    
    // Buffers for frame features (reused)
    vector<float> frame_colors;    // 9 * num_cells
    vector<float> frame_strengths; // num_cells
    vector<float> frame_hists;     // 4 * num_cells
    vector<int>   best_indices;    // num_cells
    
    // Helpers (Same as CPU, used for loading/preprocessing)
    TileFeatures computeFeatures(const Mat& img) {
        TileFeatures f;
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        
        // Colors
        int rh = img.rows/3; int rw=img.cols/3;
        for(int i=0; i<9; i++) {
            int y = (i/3)*rh; int x = (i%3)*rw;
            Scalar m = mean(img(Rect(x,y,rw,rh)));
            f.region_colors[i] = Vec3f(m[0], m[1], m[2]);
        }
        
        // Edges
        Mat gx, gy, mag, dir;
        Sobel(gray, gx, CV_32F, 1, 0, 3);
        Sobel(gray, gy, CV_32F, 0, 1, 3);
        cartToPolar(gx, gy, mag, dir, true);
        
        f.edges.edge_strength = mean(mag)[0];
        f.edges.edge_histogram.fill(0);
        
        for(int r=0; r<mag.rows; r++) {
            float* mptr = mag.ptr<float>(r);
            float* dptr = dir.ptr<float>(r);
            for(int c=0; c<mag.cols; c++) {
                if(mptr[c] > 10.0f) {
                    int bin = int((dptr[c]+22.5f)/45.0f)%4;
                    f.edges.edge_histogram[bin] += mptr[c];
                }
            }
        }
        float total = 0; for(float v: f.edges.edge_histogram) total+=v;
        if(total>0.001f) for(float& v: f.edges.edge_histogram) v/=total;
        
        return f;
    }

public:
    VideoMosaicGeneratorMetal(const VideoMosaicConfig& cfg) : config(cfg), frame_count(0) {}
    
    bool init() {
        if (!metalEngine.init()) return false;
        
        // Load Tiles
        if (!fs::exists(config.tile_dir)) return false;
        
        vector<string> files;
        for(auto& p: fs::directory_iterator(config.tile_dir)) {
             string ext = p.path().extension().string();
             if(ext==".png" || ext==".jpg" || ext==".jpeg" || ext==".PNG") 
                 files.push_back(p.path().string());
        }
        
        cout << "Loading " << files.size() << " tiles..." << endl;
        tiles.resize(files.size());
        
        // Parallel Load
        #pragma omp parallel for
        for(size_t i=0; i<files.size(); i++) {
            Mat m = imread(files[i]);
            resize(m, tiles.images[i], Size(config.tile_size, config.tile_size));
            tiles.features[i] = computeFeatures(tiles.images[i]);
        }
        
        // Prepare Metal Data Flattened
        MetalTileData mData;
        mData.count = tiles.size();
        mData.region_colors.resize(9 * 4 * tiles.size()); // 4 floats (RGBA) per color
        mData.edge_strengths.resize(tiles.size());
        mData.edge_hists.resize(4 * tiles.size());
        
        for(size_t i=0; i<tiles.size(); i++) {
            for(int r=0; r<9; r++) {
                // Layout: R, G, B, Pad, R, G, B, Pad... (Align to 16 bytes for float3/float4 on GPU)
                int base_idx = i*9*4 + r*4;
                mData.region_colors[base_idx + 0] = tiles.features[i].region_colors[r][0];
                mData.region_colors[base_idx + 1] = tiles.features[i].region_colors[r][1];
                mData.region_colors[base_idx + 2] = tiles.features[i].region_colors[r][2];
                mData.region_colors[base_idx + 3] = 0.0f; // Padding
            }
            mData.edge_strengths[i] = tiles.features[i].edges.edge_strength;
            for(int h=0; h<4; h++) {
                mData.edge_hists[i*4 + h] = tiles.features[i].edges.edge_histogram[h];
            }
        }
        
        // NOTE: Wait, vector<float> for float3 array?
        // mData.region_colors is vector<float>. So it has size 9 * 3 * N.
        // My struct definition in header said `vector<float> region_colors`.
        // Kernel expects `float3*`.
        // Metal treats `float3*` buffer as `struct { float x,y,z; }`.
        
        metalEngine.uploadTileData(mData);
        
        // Pre-allocate frame buffers
        int num_cells = config.grid_width * config.grid_height;
        frame_colors.resize(num_cells * 9 * 4); // 4 floats per color
        frame_strengths.resize(num_cells);
        frame_hists.resize(num_cells * 4);
        best_indices.resize(num_cells);
        
        return true;
    }
    
    // Main Processing Loop
    Mat generateMosaic(const Mat& input) {
        auto t1 = high_resolution_clock::now();
        
        // 1. CPU Pre-processing (Optimized)
        Mat resized;
        int gw = config.grid_width; int gh = config.grid_height;
        resize(input, resized, Size(gw*config.tile_size, gh*config.tile_size));
        
        Mat gray; cvtColor(resized, gray, COLOR_BGR2GRAY);
        Mat gx, gy; 
        Sobel(gray, gx, CV_32F, 1, 0, 3);
        Sobel(gray, gy, CV_32F, 0, 1, 3);
        Mat mag, dir;
        cartToPolar(gx, gy, mag, dir, true);
        
        // 2. Extract Grid Features -> Flat Buffers
        // Parallelizing this CPU part is still important
        #pragma omp parallel for collapse(2)
        for(int y=0; y<gh; y++) {
            for(int x=0; x<gw; x++) {
                int idx = y*gw + x;
                
                // Color features (9 regions)
                int ts = config.tile_size;
                int rts = ts/3;
                
                // Offset in resized image
                int img_y0 = y*ts; int img_x0 = x*ts;
                
                for(int r=0; r<9; r++) {
                    int ry = (r/3)*rts; int rx = (r%3)*rts;
                    
                    // Simple mean calculation (could be optimized further)
                    Rect roi(img_x0+rx, img_y0+ry, rts, rts);
                    Scalar m = mean(resized(roi)); // This is the slow part on CPU now!
                    
                    // Write to flattened buffer (x,y,z,pad, x,y,z,pad...)
                    int buf_idx = (idx * 9 * 4) + (r * 4);
                    frame_colors[buf_idx+0] = m[0];
                    frame_colors[buf_idx+1] = m[1];
                    frame_colors[buf_idx+2] = m[2];
                    frame_colors[buf_idx+3] = 0.0f; // Padding
                }
                
                // Edge features
                Rect roi(img_x0, img_y0, ts, ts);
                Scalar m_str = mean(mag(roi));
                frame_strengths[idx] = m_str[0];
                
                // Histogram (simplified for speed)
                float hist[4] = {0};
                // We'll skip the full histogram loop per cell and rely on GPU being fast enough
                // OR we have to implement it. Let's implement a fast version.
                // Actually, accessing `mag` and `dir` per pixel here on CPU is slow.
                // But for 100x75 grid, it's manageable.
                
                // FIXME: For max performance, move Histogram extraction to a Compute Shader too? 
                // For now, keep on CPU to stick to verified logic.
                // Using 8 threads, this loop is fast enough.
                Mat c_mag = mag(roi);
                Mat c_dir = dir(roi);
                for(int r=0; r<ts; r++) {
                    float* mptr = c_mag.ptr<float>(r);
                    float* dptr = c_dir.ptr<float>(r);
                    for(int c=0; c<ts; c++) {
                        if(mptr[c]>10) {
                            int bin = int((dptr[c]+22.5f)/45.0f)%4;
                            hist[bin]+=mptr[c];
                        }
                    }
                }
                float sum=0; for(float v:hist) sum+=v;
                if(sum>0.001) for(float& v:hist) v/=sum;
                
                frame_hists[idx*4+0] = hist[0];
                frame_hists[idx*4+1] = hist[1];
                frame_hists[idx*4+2] = hist[2];
                frame_hists[idx*4+3] = hist[3];
            }
        }
        
        // 3. Metal Compute (The heavy lifting)
        // Check "best_indices.size()" to confirm
        metalEngine.findBestMatches(frame_colors, frame_strengths, frame_hists, best_indices);
        
        // 4. Construct Mosaic
        Mat mosaic(gh*config.tile_size, gw*config.tile_size, CV_8UC3);
        
        #pragma omp parallel for
        for(int i=0; i<best_indices.size(); i++) {
            int y = i / gw;
            int x = i % gw;
            int tile_idx = best_indices[i];
            
            tiles.images[tile_idx].copyTo(mosaic(Rect(x*config.tile_size, y*config.tile_size, 
                                                     config.tile_size, config.tile_size)));
        }
        
        // FPS
        if(config.show_fps) {
            auto t2 = high_resolution_clock::now();
            double fps = 1000.0 / duration_cast<milliseconds>(t2-t1).count();
            string text = "METAL FPS: " + to_string((int)fps);
            putText(mosaic, text, Point(30,60), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0,255,0), 3);
        }
        
        return mosaic;
    }
    
    void run(int camId, int benchmark_frames = 0) {
        VideoCapture cap(camId);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open camera." << endl;
            return;
        }

        Mat frame;
        int frames_processed = 0;
        double total_fps = 0.0;
        
        cout << "Warming up..." << endl;
        
        // Explicitly create window to help with multi-monitor context placement
        namedWindow("Metal Mosaic", WINDOW_NORMAL);
        resizeWindow("Metal Mosaic", 1280, 720); // Set a reasonable default size
        
        // Warmup
        for(int i=0; i<5; i++) {
            if(!cap.read(frame)) break;
            generateMosaic(frame);
        }

        cout << "Starting loop..." << endl;
        auto start_total = high_resolution_clock::now();

        while(cap.read(frame)) {
            auto t1 = high_resolution_clock::now();
            Mat result = generateMosaic(frame);
            auto t2 = high_resolution_clock::now();
            
            double fps = 1000.0 / std::max(1.0, (double)duration_cast<milliseconds>(t2-t1).count());
            total_fps += fps;
            frames_processed++;

            if(benchmark_frames > 0) {
                if (frames_processed % 10 == 0) 
                    cout << "Benchmark: " << frames_processed << "/" << benchmark_frames << " frames..." << endl;
                
                if(frames_processed >= benchmark_frames) {
                    double avg = total_fps / frames_processed;
                    cout << "RESULT_FPS: " << avg << endl;
                    break;
                }
            } else {
                imshow("Metal Mosaic", result);
                if(waitKey(1)=='q') break;
            }
        }
    }
};

int main(int argc, char** argv) {
    VideoMosaicConfig config;
    config.tile_dir = "data/pokemon_tiles";
    config.tile_size = 32;
    config.grid_width = 80;
    config.grid_height = 60;
    config.show_fps = true;
    
    int benchmark_frames = 0;
    
    for(int i=1; i<argc; i++) {
        string arg = argv[i];
        if(arg=="-d") config.tile_dir = argv[++i];
        if(arg=="--small") { config.grid_width=40; config.grid_height=30; }
        if(arg=="--medium") { config.grid_width=60; config.grid_height=45; }
        if(arg=="--large") { config.grid_width=80; config.grid_height=60; }
        if(arg=="--ultra") { config.grid_width=100; config.grid_height=75; }
        if(arg=="--benchmark") benchmark_frames = atoi(argv[++i]);
    }
    
    cout << "Initializing Metal..." << endl;
    VideoMosaicGeneratorMetal app(config);
    if(!app.init()) return -1;
    
    app.run(0, benchmark_frames);
    return 0;
}
