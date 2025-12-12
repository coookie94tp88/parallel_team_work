#include "metal_compute.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <fstream>

using namespace std;

MetalComputeEngine::MetalComputeEngine() 
    : device(nullptr), commandQueue(nullptr), computeState(nullptr),
      bufferTileColors(nullptr), bufferTileStrengths(nullptr), bufferTileHists(nullptr),
      numTiles(0) {}

MetalComputeEngine::~MetalComputeEngine() {
    // ARC (Automatic Reference Counting) handles Objective-C cleanup 
    // when compiled with -fobjc-arc. If not, we'd need [obj release].
    // Assuming modern setup, we rely on smart pointers or ARC.
    // But for raw void* in C++ class, actually we should explicit release if not using ARC.
    // For simplicity in this hybrid setup, we'll assume the process termination cleans up GPU resources.
}

bool MetalComputeEngine::init() {
    // 1. Get default device
    id<MTLDevice> mtlDevice = MTLCreateSystemDefaultDevice();
    if (!mtlDevice) {
        cerr << "Error: No Metal device found!" << endl;
        return false;
    }
    device = (__bridge_retained void*)mtlDevice;
    cout << "Metal Device: " << [mtlDevice.name UTF8String] << endl;
    
    // 2. Create Load Library
    NSError* error = nil;
    
    // Load default library (searches for .metallib in bundle/executable)
    // NOTE: We need to compile kernels.metal to default.metallib manually in Makefile
    // and place it next to executable
    
    // Try to load from local file "./default.metallib" if bundle fails
    id<MTLLibrary> library = nil;
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *currentPath = [fileManager currentDirectoryPath];
    NSString *libPath = [currentPath stringByAppendingPathComponent:@"default.metallib"];
        
    if ([fileManager fileExistsAtPath:libPath]) {
        // Load independent library
        NSURL *url = [NSURL fileURLWithPath:libPath];
        library = [mtlDevice newLibraryWithURL:url error:&error];
    } else {
        // Fallback to default (if embedded)
        library = [mtlDevice newDefaultLibrary];
    }

    if (!library) {
        cerr << "Error: Could not load Metal library!" << endl;
        if (error) cerr << [[error localizedDescription] UTF8String] << endl;
        return false;
    }
    
    // 3. Get Kernel Function
    id<MTLFunction> kernelFunc = [library newFunctionWithName:@"find_best_match"];
    if (!kernelFunc) {
        cerr << "Error: Could not find kernel function 'find_best_match'" << endl;
        return false;
    }
    
    // 4. Create Compute Pipeline
    id<MTLComputePipelineState> pso = [mtlDevice newComputePipelineStateWithFunction:kernelFunc error:&error];
    if (!pso) {
        cerr << "Error: Could not create pipeline state!" << endl;
        if (error) cerr << [[error localizedDescription] UTF8String] << endl;
        return false;
    }
    computeState = (__bridge_retained void*)pso;
    
    // 5. Create Command Queue
    id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
    commandQueue = (__bridge_retained void*)queue;
    
    return true;
}

void MetalComputeEngine::uploadTileData(const MetalTileData& data) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    numTiles = data.count;
    
    // Allocate private buffers (we only write once, GPU reads many times)
    // Actually, managed is easier for unified memory (Apple Silicon)
    MTLResourceOptions options = MTLResourceStorageModeManaged;
    
    // Colors (9 floats per tile)
    size_t sizeColors = data.region_colors.size() * sizeof(float);
    id<MTLBuffer> bufColors = [mtlDevice newBufferWithBytes:data.region_colors.data() 
                                                     length:sizeColors 
                                                    options:options];
    bufferTileColors = (__bridge_retained void*)bufColors;
    
    // Strengths (1 float per tile)
    size_t sizeStr = data.edge_strengths.size() * sizeof(float);
    id<MTLBuffer> bufStr = [mtlDevice newBufferWithBytes:data.edge_strengths.data() 
                                                  length:sizeStr 
                                                 options:options];
    bufferTileStrengths = (__bridge_retained void*)bufStr;
    
    // Hists (4 floats per tile)
    size_t sizeHist = data.edge_hists.size() * sizeof(float);
    id<MTLBuffer> bufHist = [mtlDevice newBufferWithBytes:data.edge_hists.data() 
                                                   length:sizeHist 
                                                  options:options];
    bufferTileHists = (__bridge_retained void*)bufHist;
    
    cout << "Uploaded " << numTiles << " tiles to GPU memory." << endl;
}

void MetalComputeEngine::findBestMatches(
    const std::vector<float>& cell_colors,
    const std::vector<float>& cell_strengths,
    const std::vector<float>& cell_hists,
    std::vector<int>& best_indices
) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)computeState;
    
    size_t numCells = best_indices.size();
    
    // Create per-frame buffers for Cells
    // NOTE: In a continuous loop, it's better to stick to one reusable buffer
    // and just memcpy/didModifyRange. But buffer creation on Metal is fast.
    MTLResourceOptions options = MTLResourceStorageModeManaged;
    
    id<MTLBuffer> bufCellColors = [mtlDevice newBufferWithBytes:cell_colors.data() length:cell_colors.size()*sizeof(float) options:options];
    id<MTLBuffer> bufCellStr = [mtlDevice newBufferWithBytes:cell_strengths.data() length:cell_strengths.size()*sizeof(float) options:options];
    id<MTLBuffer> bufCellHist = [mtlDevice newBufferWithBytes:cell_hists.data() length:cell_hists.size()*sizeof(float) options:options];
    
    // Output Buffer
    id<MTLBuffer> bufOutput = [mtlDevice newBufferWithLength:numCells * sizeof(int) options:options];
    
    // Constant Buffer for numTiles
    uint32_t nTiles = (uint32_t)numTiles;
    
    // Create Command Buffer
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pso];
    
    // Bind Buffers (Indices match kernels.metal)
    // Tile Data (0, 1, 2)
    [encoder setBuffer:(__bridge id<MTLBuffer>)bufferTileColors offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)bufferTileStrengths offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)bufferTileHists offset:0 atIndex:2];
    
    // Cell Data (3, 4, 5)
    [encoder setBuffer:bufCellColors offset:0 atIndex:3];
    [encoder setBuffer:bufCellStr offset:0 atIndex:4];
    [encoder setBuffer:bufCellHist offset:0 atIndex:5];
    
    // Output (6)
    [encoder setBuffer:bufOutput offset:0 atIndex:6];
    
    // Constants (7)
    [encoder setBytes:&nTiles length:sizeof(uint32_t) atIndex:7];
    
    // Dispatch Threads
    MTLSize threadsPerGrid = MTLSizeMake(numCells, 1, 1);
    
    // Threads per threadgroup (Metal max is usually 512 or 1024)
    // We just need efficient occupancy.
    NSUInteger w = pso.threadExecutionWidth;
    NSUInteger h = pso.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadsPerGroup = MTLSizeMake(w * h, 1, 1);
    if (threadsPerGroup.width > numCells) threadsPerGroup.width = numCells;
    
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    
    // Commit and Wait
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
    
    // Read Results
    int* ptr = (int*)[bufOutput contents];
    memcpy(best_indices.data(), ptr, numCells * sizeof(int));
    
    // Buffer release handled by ARC (autorelease pool usually, but here scope end)
}
