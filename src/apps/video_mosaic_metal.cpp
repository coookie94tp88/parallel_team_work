#include <iostream>
#include <string>
#include "VideoMosaicGenerator.h"
#include "../gpu/metal_compute.h"

using namespace std;
using namespace cv;
using namespace chrono;

// Metal-accelerated Generator
class VideoMosaicGeneratorMetal : public VideoMosaicGenerator {
private:
    MetalComputeEngine metal_engine;

public:
    VideoMosaicGeneratorMetal(const VideoMosaicConfig& cfg) : VideoMosaicGenerator(cfg) {
        if(!metal_engine.init()) {
            cerr << "Failed to init Metal!" << endl;
            exit(1);
        }
    }

    bool loadTilesAndUpload() {
        if (!loadTiles()) return false;
        
        // Convert to Metal Layout (Struct of Arrays)
        MetalTileData data;
        data.count = tiles.size();
        
        // Kernel expects: float4* tile_region_colors (9 per tile)
        // So we need 9 * 4 floats per tile.
        data.region_colors.resize(data.count * 9 * 4);
        data.edge_strengths.resize(data.count);
        data.edge_hists.resize(data.count * 4);
        
        for(size_t i=0; i<tiles.size(); i++) {
            // Colors
            for(int k=0; k<9; k++) {
                Vec3f c = tiles.features[i].region_colors[k];
                // Pack as float4 (R, G, B, 0) - Kernel uses .xyz
                // Note: OpenCV uses BGR usually, but our computeRegionColors
                // stores as Vec3f. Let's ensure we send RGB if kernel treats it as color.
                // Actually distance is commutative so BGR vs RGB doesn't matter 
                // as long as consistent source vs tile.
                // We'll just copy the 3 components.
                size_t base_idx = (i * 9 + k) * 4;
                data.region_colors[base_idx + 0] = c[0];
                data.region_colors[base_idx + 1] = c[1];
                data.region_colors[base_idx + 2] = c[2];
                data.region_colors[base_idx + 3] = 0.0f; // Padding
            }
            
            data.edge_strengths[i] = tiles.features[i].edges.edge_strength;
            
            // Histogram (already 4 floats, maps naturally to float4)
            for(int k=0; k<4; k++) {
                data.edge_hists[i*4 + k] = tiles.features[i].edges.edge_histogram[k];
            }
        }
        
        cout << "Uploading " << data.count << " tiles to Metal GPU..." << endl;
        metal_engine.uploadTileData(data);
        return true;
    }

    Mat generateMosaic(const Mat& input_frame) override {
        auto start_time = high_resolution_clock::now();
        
        // 1. Resize & Features (CPU optimized)
        Mat resized_frame;
        int target_w = config.grid_width * config.tile_size;
        int target_h = config.grid_height * config.tile_size;
        resize(input_frame, resized_frame, Size(target_w, target_h), 0, 0, INTER_LINEAR);
        
        Mat gray, gx, gy, mag, dir;
        cvtColor(resized_frame, gray, COLOR_BGR2GRAY);
        Sobel(gray, gx, CV_32F, 1, 0, 3);
        Sobel(gray, gy, CV_32F, 0, 1, 3);
        cartToPolar(gx, gy, mag, dir, true);
        
        // 2. Prepare Data for Metal
        int num_cells = config.grid_width * config.grid_height;
        
        // Match Kernel Expectations:
        // cell_region_colors: float4 array -> 9 * 4 * num_cells
        vector<float> cell_colors(num_cells * 9 * 4);
        vector<float> cell_strengths(num_cells);
        vector<float> cell_hists(num_cells * 4);
        vector<int> best_indices(num_cells);
        
        #pragma omp parallel for collapse(2)
        for(int y=0; y<config.grid_height; y++) {
            for(int x=0; x<config.grid_width; x++) {
                int idx = y * config.grid_width + x;
                
                // Get Region Colors
                Rect cell_rect(x * config.tile_size, y * config.tile_size, config.tile_size, config.tile_size);
                Mat cell_img = resized_frame(cell_rect);
                
                // Fast 3x3 grid averaging
                int rw = cell_img.cols/3; int rh = cell_img.rows/3;
                for(int r=0; r<3; r++) {
                    for(int c=0; c<3; c++) {
                        Scalar m = mean(cell_img(Rect(c*rw, r*rh, rw, rh)));
                        // Pack float4 (R, G, B, 0)
                        size_t base_idx = (idx * 9 + (r*3+c)) * 4;
                        cell_colors[base_idx + 0] = (float)m[0];
                        cell_colors[base_idx + 1] = (float)m[1];
                        cell_colors[base_idx + 2] = (float)m[2];
                        cell_colors[base_idx + 3] = 0.0f;
                    }
                }
                
                // Get Edge Features
                Mat cell_mag = mag(cell_rect);
                Mat cell_dir = dir(cell_rect);
                EdgeFeatures ef = extractEdgeFeaturesFromGlobal(cell_mag, cell_dir);
                
                cell_strengths[idx] = ef.edge_strength;
                for(int k=0; k<4; k++) cell_hists[idx*4 + k] = ef.edge_histogram[k];
            }
        }
        
        // 3. Run Metal
        metal_engine.findBestMatches(cell_colors, cell_strengths, cell_hists, best_indices);
        
        // 4. Construct Image
        Mat mosaic(target_h, target_w, CV_8UC3);
        #pragma omp parallel for collapse(2)
        for(int y=0; y<config.grid_height; y++) {
            for(int x=0; x<config.grid_width; x++) {
                int idx = best_indices[y * config.grid_width + x];
                Rect roi(x*config.tile_size, y*config.tile_size, config.tile_size, config.tile_size);
                tiles.images[idx].copyTo(mosaic(roi));
            }
        }
        
        frame_count++;
        if (config.show_fps) {
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end_time - start_time);
            double fps = 1000.0 / max(1, (int)duration.count());
            putText(mosaic, "Metal FPS: " + to_string((int)fps), Point(20, 60), 
                   FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 255, 0), 4);
        }
        return mosaic;
    }
};

void printUsage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "  -d, --tiles DIR     Tile directory" << endl;
    cout << "  --medium            Medium grid (60x45)" << endl;
    cout << "  --ultra             Ultra grid (100x75)" << endl;
}

int main(int argc, char** argv) {
    VideoMosaicConfig config;
    int benchmark_frames = 0;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help") { printUsage(argv[0]); return 0; }
        else if (arg == "-d") { if (i + 1 < argc) config.tile_dir = argv[++i]; }
        else if (arg == "--frames") { if (i + 1 < argc) benchmark_frames = atoi(argv[++i]); }
        else if (arg == "--medium") { config.grid_width = 60; config.grid_height = 45; }
        else if (arg == "--ultra") { config.grid_width = 100; config.grid_height = 75; }
    }
    
    cout << "=== Metal GPU Video Mosaic ===" << endl;
    
    VideoMosaicGeneratorMetal generator(config);
    if (!generator.loadTilesAndUpload()) return -1;
    
    generator.processWebcam(0, benchmark_frames);
    return 0;
}
