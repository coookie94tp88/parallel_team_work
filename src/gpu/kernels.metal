
#include <metal_stdlib>
using namespace metal;

// Must match C++ struct layout EXACTLY
struct EdgeFeatures {
    float edge_strength;
    float edge_histogram[4]; 
};

struct TileFeatures {
    packed_float3 region_colors[9];
    float pad1[3]; // Padding to align 16-byte boundary if needed, but packed_float3 is 12 bytes. 
                   // Let's use float4 for colors on C++ side to be safe/aligned, or carefully pack.
                   // SIMPLER STRATEGY: Use arrays of floats in separate buffers (Struct of Arrays) to avoid alignment headdaches.
                   // Actually, let's keep it simple first: Struct of Arrays is best for GPU coalescing.
};

// BETTER DATA LAYOUT for GPU:
// Features are passed as separate buffer pointers.
// This ensures perfect alignment and read coalescing.

kernel void find_best_match(
    // Input: Tile Database (Read Only, Constant for all threads)
    device const float4* tile_region_colors [[ buffer(0) ]], // 9 colors per tile * num_tiles (float4 aligned)
    device const float* tile_edge_strength  [[ buffer(1) ]], // 1 float per tile
    device const float4* tile_edge_hist     [[ buffer(2) ]], // 1 float4 per tile
    
    // Input: Grid Cells (Read Only, Changes per frame)
    device const float4* cell_region_colors [[ buffer(3) ]], // 9 colors per cell * num_cells
    device const float* cell_edge_strength  [[ buffer(4) ]],
    device const float4* cell_edge_hist     [[ buffer(5) ]],
    
    // Output: Best Tile Index for each cell
    device int* best_indices [[ buffer(6) ]],
    
    // Constants
    constant uint& num_tiles [[ buffer(7) ]],
    
    // Thread Index
    uint id [[ thread_position_in_grid ]]
) {
    // 1. Load this cell's features into registers
    float3 my_colors[9];
    for(int i=0; i<9; i++) {
        my_colors[i] = cell_region_colors[id * 9 + i].xyz; // Load xyz, ignore w
    }
    float my_strength = cell_edge_strength[id];
    float4 my_hist = cell_edge_hist[id];
    
    // Weights (hardcoded for now, same as CPU)
    // 0.5, 1.0, 0.5
    // 1.0, 2.0, 1.0
    // 0.5, 1.0, 0.5
    const float region_weights[9] = {
        0.5f, 1.0f, 0.5f,
        1.0f, 2.0f, 1.0f,
        0.5f, 1.0f, 0.5f
    };
    
    const float COLOR_WEIGHT = 0.7f;
    const float EDGE_DIR_WEIGHT = 0.2f;   
    const float EDGE_STR_WEIGHT = 0.1f;
    
    // 2. Iterate over all tiles to find best match
    float min_dist = 1e30f; // float max
    int best_index = 0;
    
    for (uint i = 0; i < num_tiles; i++) {
        // --- Region Color Distance ---
        float total_color_dist = 0.0f;
        float total_weight = 0.0f;
        
        for (int r = 0; r < 9; r++) {
            float3 t_col = tile_region_colors[i * 9 + r].xyz; // Load xyz, ignore w
            float3 diff_v = my_colors[r] - t_col;
            float dist_sq = dot(diff_v, diff_v); // r^2 + g^2 + b^2
            
            total_color_dist += dist_sq * region_weights[r];
            total_weight += region_weights[r];
        }
        float color_dist = total_color_dist / total_weight;
        
        // --- Combined Distance Logic ---
        // (Copying CPU logic: only use edges if strength > 1.0)
        float t_strength = tile_edge_strength[i];
        float total_dist = 0.0f;
        
        if (my_strength > 1.0f || t_strength > 1.0f) {
            float strength_diff = abs(my_strength - t_strength);
            
            // Histogram Chi-Square
            float4 t_hist = tile_edge_hist[i];
            float4 sum = my_hist + t_hist;
            float4 diff = my_hist - t_hist;
            float4 diff_sq = diff * diff;
            
            float dir_dist = 0.0f;
            int valid_bins = 0;
            
            // Unrolled loop for float4
            if (sum[0] > 0.01f) { dir_dist += diff_sq[0] / sum[0]; valid_bins++; }
            if (sum[1] > 0.01f) { dir_dist += diff_sq[1] / sum[1]; valid_bins++; }
            if (sum[2] > 0.01f) { dir_dist += diff_sq[2] / sum[2]; valid_bins++; }
            if (sum[3] > 0.01f) { dir_dist += diff_sq[3] / sum[3]; valid_bins++; }
            
            if (valid_bins > 0) dir_dist /= valid_bins;
            
            float edge_component = EDGE_DIR_WEIGHT * dir_dist * 100.0f + 
                                  EDGE_STR_WEIGHT * strength_diff;
            
            total_dist = COLOR_WEIGHT * color_dist + edge_component;
        } else {
            // Fallback to pure color
            total_dist = COLOR_WEIGHT * color_dist;
        }
        
        // Update Min
        if (total_dist < min_dist) {
            min_dist = total_dist;
            best_index = i;
        }
    }
    
    // 3. Write Best Index
    best_indices[id] = best_index;
}
