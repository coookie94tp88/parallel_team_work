import subprocess
import re
import time
import sys

# Configuration
TILES_DIR = "data/cifar_tiles"
FRAMES = 50
BIN_CPU = "bin/video_mosaic_edge_optimized"
BIN_METAL = "bin/video_mosaic_edge_metal"

configs = [
    # CPU Scaling
    {"name": "CPU_Opt_1_Thread",  "cmd": [BIN_CPU, "--medium", "-j", "1", "-d", TILES_DIR, "--benchmark", str(FRAMES)]},
    {"name": "CPU_Opt_4_Threads", "cmd": [BIN_CPU, "--medium", "-j", "4", "-d", TILES_DIR, "--benchmark", str(FRAMES)]},
    {"name": "CPU_Opt_8_Threads", "cmd": [BIN_CPU, "--medium", "-j", "8", "-d", TILES_DIR, "--benchmark", str(FRAMES)]},
    
    # Grid Size Scaling (CPU vs Metal)
    {"name": "CPU_Opt_Ultra_8T",  "cmd": [BIN_CPU, "--ultra",  "-j", "8", "-d", TILES_DIR, "--benchmark", str(FRAMES)]},
    {"name": "Metal_Medium",      "cmd": [BIN_METAL, "--medium", "-d", TILES_DIR, "--benchmark", str(FRAMES)]},
    {"name": "Metal_Ultra",       "cmd": [BIN_METAL, "--ultra",  "-d", TILES_DIR, "--benchmark", str(FRAMES)]},
]

results = {}

print(f"=== Starting Experiments ({FRAMES} frames each) ===")

for config in configs:
    name = config["name"]
    cmd = config["cmd"]
    print(f"Running {name}...", end="", flush=True)
    
    try:
        # Run command
        start_t = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        end_t = time.time()
        
        # Parse output
        fps = 0.0
        for line in result.stdout.split('\n'):
            if "RESULT_FPS:" in line:
                fps = float(line.split(":")[1].strip())
        
        if fps == 0.0:
            print(f" FAILED (No FPS output)")
            print(result.stderr)
        else:
            print(f" Done. {fps:.2f} FPS")
            results[name] = fps
            
    except Exception as e:
        print(f" ERROR: {e}")
        results[name] = 0.0

print("\n=== Results Summary ===")
print("| Configuration | FPS | Speedup (vs CPU 1T) |")
print("|---|---|---|")

base_fps = results.get("CPU_Opt_1_Thread", 1.0)

for config in configs:
    name = config["name"]
    fps = results.get(name, 0.0)
    speedup = fps / base_fps if base_fps > 0 else 0
    print(f"| {name} | {fps:.2f} | {speedup:.2f}x |")
