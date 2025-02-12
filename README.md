# Kinect Fusion

## Deps
Not sure if all of them are needed
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

### Rendering

```
./bin_x64/Release/vk_mini_viewer_app
```

Will open the sdf_values_0000.bin file and raytrace it to show the rendering code.