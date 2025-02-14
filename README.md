# Kinect Fusion

## Deps
```
sudo apt-get install libx11-dev libxcb1-dev libxcb-keysyms1-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev libxxf86vm-dev libvulkan-dev libglm-dev libfreeimage-dev
```

## Building
```
cmake --preset release
cmake --build --preset release
# OR
cmake --workflow --preset  release
```

## Running

### Volumetric integration
```
./bin_x64/Release/vk_mini_fusion_exe_app <path_to_dataset>
```
If not dataset is specified, it will default to "../Data/rgbd_dataset_freiburg1_xyz/"

This binary will run the volumetric integration on the provided dataset and write a file called sdf_values_0000.bin containing the final TSDF.

The different variant are:
```
# Compute shader
./bin_x64/Release/vk_mini_fusion_exe_app

# CUDA
./bin_x64/Release/vk_mini_fusion_exe_app --cuda

# CPU
./bin_x64/Release/vk_mini_fusion_exe_app --cpu  
```

### Rendering

```
./bin_x64/Release/vk_mini_viewer_app
```

Will open the sdf_values_0000.bin file and raytrace it to show the rendering code.

## Benchmarks
Install the python requirements:
```
python -m pip install pandas seaborn PyQt6
```

After building a release build, the benchmark script can be executed. This will create a file called "vol_run_times.json" containing the results.
```
python plots/vol_run_time.py
```

To plot the results call the plotting script:
```
python plots/plot_vol.py
```

## Important Source Files and Folder 

### kinect_fusion/helper/vulkan_helper.h
Contains a wrapper class for vulkan to easily manage the required resources using RAII.

### kinect_fusion/shader/perlin.slang
Contains the compute shader for the volumetric integration. This is both compiled to CUDA and to SPIR-V for usage with Vulkan.

### kinect_fusion/shader/raster.slang
Rendering code using rayMarching. Is only compiled to SPIR-V.

### kinect_fusion/fusion.cu
CUDA wrapper that calls the generated CUDA code.

### kinect_fusion/fusion_main.cpp
Main cpp file for the integration_exe, contains the Vulkan wrapper and the CPU version of the volumetric integration.

### kinect_fusion/viewer.cpp
Code for the viewer application to interactively render the scene in a GUI (calls the raster.slang shader)

# Source
The wrapper nvpro_core files, the project structure and the rendering code are based on the NVIDIA [vk_mini_samples](https://github.com/nvpro-samples/vk_mini_samples/tree/main) 