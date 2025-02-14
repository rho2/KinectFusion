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

```
./bin_x64/Release/vk_mini_fusion_icp_exe
```

This runs the icp algorithm for one frame of the data.