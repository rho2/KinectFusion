# Kinect Fusion
The code is split into multiple branches at the moment. The build instructions cam be found in the README's on the individual branches.

## vol_integration
Contains the code for volumetric integration (without color) which writes it's results to a file and the viewer application which loads these files and renders them using raymarching.

It contains all variants of the volumetric integration (CPU, CUDA, compute shader) and the python scripts to run the benchmarks shown in the video and the report.

## vol_integr_fused
Contains the inital implementation for the color integration.
The application runs the integration and rendering in a single binary, only supporting compute shaders. 

## vol_integration_color
Port of the color integration for the other variant (CUDA, CPU) for benchmarking. Basically the *vol_integration* with color support.

## icp_dev
Contains the application running our ICP implementation