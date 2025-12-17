#pragma once
#include <string>

struct VideoMosaicConfig {
    std::string tile_dir = "data/pokemon_tiles";
    int tile_size = 32;
    int grid_width = 60;
    int grid_height = 45;
    int num_threads = 4;
    bool temporal_coherence = true;
    int coherence_threshold = 500;
    bool show_fps = true;
    bool use_histogram = false;
    float color_blend = 0.0f;
};
