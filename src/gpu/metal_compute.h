#pragma once
#include <vector>
#include <array>
#include <memory>

// Forward declare structs to avoid including Metal headers in pure C++
struct MetalContext;

// Data structures matching the Metal kernel layout
// We use flattened arrays (Still Struct of Arrays) for easy transfer
struct MetalTileData {
    // 9 floats (RGB * 3 regions) per tile
    std::vector<float> region_colors; 
    
    // 1 float per tile
    std::vector<float> edge_strengths;
    
    // 4 floats per tile
    std::vector<float> edge_hists;
    
    size_t count;
};

class MetalComputeEngine {
public:
    MetalComputeEngine();
    ~MetalComputeEngine();
    
    bool init();
    
    // Upload the static tile database to GPU (done once)
    void uploadTileData(const MetalTileData& data);
    
    // Find best matches for a frame of grid cells
    // Inputs are the flattened feature arrays for the current frame
    // Output is written to best_indices vector
    void findBestMatches(
        const std::vector<float>& cell_colors,     // 9 * num_cells floats
        const std::vector<float>& cell_strengths,  // num_cells floats
        const std::vector<float>& cell_hists,      // 4 * num_cells floats
        std::vector<int>& best_indices             // Output: num_cells ints
    );

private:
    void* device;           // MTLDevice
    void* commandQueue;     // MTLCommandQueue
    void* computeState;     // MTLComputePipelineState
    
    // GPU Buffers
    void* bufferTileColors;
    void* bufferTileStrengths;
    void* bufferTileHists;
    
    // Cached sizes
    size_t numTiles;
    
    // Persistent Buffers for input cells (reused per frame)
    void* bufferCellColors;
    void* bufferCellStrengths;
    void* bufferCellHists;
    void* bufferOutput;
    
    size_t bufferCapacityCells; // Current capacity (number of cells)
};
