cmake_minimum_required(VERSION 3.16)
project(VideoMosaic VERSION 1.0.0 LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization Flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-O3 -Wall -Wextra)
endif()

# Dependencies
find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(Boost REQUIRED COMPONENTS filesystem)

# Include Directories
include_directories(include)

# Common Library (Shared Logic)
add_library(video_mosaic_common
    src/common/TileDatabase.cpp
    src/common/VideoMosaicGenerator.cpp
)
target_link_libraries(video_mosaic_common 
    ${OpenCV_LIBS} 
    Boost::filesystem 
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(video_mosaic_common OpenMP::OpenMP_CXX)
endif()

# --- Applications ---

# Optimized CPU Version
add_executable(video_mosaic_cpu src/apps/video_mosaic_cpu.cpp)
target_link_libraries(video_mosaic_cpu video_mosaic_common)

# Metal GPU Version (macOS Only)
if(APPLE)
    enable_language(OBJCXX)
    find_library(METAL_LIB Metal)
    find_library(FOUNDATION_LIB Foundation)
    find_library(COREGRAPHICS_LIB CoreGraphics)
    
    # Compile Metal Shaders
    set(METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/kernels.metal)
    set(METAL_LIB_OUT ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)
    
    add_custom_command(
        OUTPUT ${METAL_LIB_OUT}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SRC} -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/kernels.air -o ${METAL_LIB_OUT}
        DEPENDS ${METAL_SRC}
        COMMENT "Compiling Metal Shaders"
    )
    add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIB_OUT})

    # Metal App
    add_executable(video_mosaic_metal 
        src/apps/video_mosaic_metal.mm 
        src/gpu/metal_compute.mm
    )
    target_link_libraries(video_mosaic_metal 
        video_mosaic_common 
        ${METAL_LIB} 
        ${FOUNDATION_LIB} 
        ${COREGRAPHICS_LIB}
    )
    target_compile_options(video_mosaic_metal PRIVATE -fobjc-arc)
    add_dependencies(video_mosaic_metal metal_shaders)
endif()

# --- Legacy Targets (for reference) ---
# Wrapping legacy single-file implementations
# add_executable(legacy_greedy src/apps/legacy/video_mosaic.cpp)
# target_link_libraries(legacy_greedy ${OpenCV_LIBS} Boost::filesystem Boost::system)
