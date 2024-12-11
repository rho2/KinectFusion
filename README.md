# Kinect Fusion

## Deps
Not sure if all of them are needed
```
sudo apt-get install libx11-dev libxcb1-dev libxcb-keysyms1-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev libxxf86vm-dev libvulkan-dev libglm-dev libfreeimage-dev
```

## Building
```
cmake --preset debug
cmake --build --preset debug
# OR
cmake --workflow --preset  debug
```