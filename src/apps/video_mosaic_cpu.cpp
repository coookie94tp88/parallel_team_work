#include <iostream>
#include <string>
#include "VideoMosaicGenerator.h"

using namespace std;

void printUsage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "\nOptions:" << endl;
    cout << "  -d, --tiles DIR     Tile directory (default: data/pokemon_tiles)" << endl;
    cout << "  -j, --threads N     Number of threads (default: 4)" << endl;
    cout << "  --small             Small grid (40x30)" << endl;
    cout << "  --medium            Medium grid (60x45, default)" << endl;
    cout << "  --large             Large grid (80x60)" << endl;

    cout << "  --ultra             Ultra grid (100x75)" << endl;
    cout << "  -i, --input FILE    Input video file" << endl;
}

int main(int argc, char** argv) {
    VideoMosaicConfig config;
    int benchmark_frames = 0;
    string input_video = "";

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help") { printUsage(argv[0]); return 0; }
        else if (arg == "-d" || arg == "--tiles") { if (i + 1 < argc) config.tile_dir = argv[++i]; }
        else if (arg == "-j" || arg == "--threads") { if (i + 1 < argc) config.num_threads = atoi(argv[++i]); }
        else if (arg == "--frames") { if (i + 1 < argc) benchmark_frames = atoi(argv[++i]); }
        else if (arg == "--small") { config.grid_width = 40; config.grid_height = 30; }
        else if (arg == "--medium") { config.grid_width = 60; config.grid_height = 45; }
        else if (arg == "--large") { config.grid_width = 80; config.grid_height = 60; }
        else if (arg == "--ultra") { config.grid_width = 100; config.grid_height = 75; }
        else if (arg == "-i" || arg == "--input") { if (i + 1 < argc) input_video = argv[++i]; }
    }

    cout << "=== CPU Video Mosaic ===" << endl;
    cout << "Threads: " << config.num_threads << endl;
    cout << "Grid: " << config.grid_width << "x" << config.grid_height << endl;

    VideoMosaicGenerator generator(config);
    if (!generator.loadTiles()) return -1;
    
    if (!input_video.empty()) {
        generator.processVideo(input_video, benchmark_frames > 0 ? benchmark_frames : 1000000);
    } else {
        generator.processWebcam(0, benchmark_frames);
    }
    return 0;
}
