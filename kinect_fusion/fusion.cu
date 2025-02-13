#include <iostream>
#include <cuda_runtime.h>
#include <fstream>

#include <memory>
#include <vector>
#include <glm/glm.hpp>

// #include "VirtualSensor.h"

namespace DH {
    using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
} // namespace DH

#define SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
#include "fusion_cuda.cu"


struct Setting {
    std::string filename;
    uint32_t size;
    bool every_step;
    bool cpu;
    bool cuda;
};

#include <chrono>

class Stopwatch {
    std::chrono::high_resolution_clock::time_point start_time;

public:
    Stopwatch() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double end() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;
        return duration.count();
    }
};

void DumpToFileCuda(const float *DataGpu, const uint32_t voxel_size, const uint32_t size, const std::string &prefix, const uint32_t index) {
    auto Data = std::make_unique<float[]>(size);
    cudaMemcpy(Data.get(), DataGpu, size, cudaMemcpyDeviceToHost);

    std::ostringstream filename;
    filename << prefix << "_" << std::setw(4) << std::setfill('0') << index << ".bin";

    std::ofstream file(filename.str(), std::ios::binary);
    file.write(reinterpret_cast<const char *>(&voxel_size), sizeof(voxel_size));
    file.write(reinterpret_cast<const char *>(Data.get()), size);
    file.close();

    std::cout << "Wrote: " << filename.str() << " with size " << voxel_size << std::endl;
}


struct VirtualSensor;
bool hack_ProcessNextFrame(VirtualSensor *sensor);
glm::mat4 hack_GetTransform(VirtualSensor *sensor);
float* hack_GetDepth(VirtualSensor *sensor);

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        // Get the error string
        const char* errorString = cudaGetErrorString(err);

        // Throw a runtime_error with the error string
        throw std::runtime_error("CUDA error: " + std::string(errorString));
    }
}

__host__ void run_cuda(VirtualSensor *sensor, const Setting& settings)
{
    const size_t size = settings.size;
    const size_t byte_size = size * size * size * sizeof(float);

    std::vector<float> voxel_grid_cpu(size * size * size);
    std::fill(voxel_grid_cpu.begin(), voxel_grid_cpu.end(), TRUNC_DISTANCE);

    std::vector<float> voxel_grid_weights_cpu(size * size * size);
    std::fill(voxel_grid_weights_cpu.begin(), voxel_grid_weights_cpu.end(), 0.0f);

    float *BufferVoxelGrid;
    cudaMalloc(&BufferVoxelGrid, byte_size);
    cudaMemcpy(BufferVoxelGrid, voxel_grid_cpu.data(), byte_size, cudaMemcpyHostToDevice);

    float *BufferVoxelGridWeights;
    cudaMalloc(&BufferVoxelGridWeights, byte_size);
    cudaMemcpy(BufferVoxelGridWeights, voxel_grid_weights_cpu.data(), byte_size, cudaMemcpyHostToDevice);

    float *BufferDepthMap;
    cudaMalloc(&BufferDepthMap, 640 * 480 * sizeof(float));

    PerlinSettings_natural_0 *BufferSettings;
    cudaMalloc(&BufferSettings, sizeof(PerlinSettings_natural_0));


    GlobalParams_0 params{};
    params.voxel_grid_0.data = BufferVoxelGrid;
    params.voxel_grid_weights_0.data = BufferVoxelGridWeights;
    params.depth_map_0.data = BufferDepthMap;
    params.perlinSettings_0 = BufferSettings;

    checkCudaError(cudaMemcpyToSymbol(SLANG_globalParams, &params, sizeof(GlobalParams_0)));

    PerlinSettings_natural_0 settings_cpu{};
    settings_cpu.inv_scale_0 = 2.0f / float(size);
    settings_cpu.size_0 = size;
    settings_cpu.size2_0 = size * size;
    settings_cpu.size3_0 = size * size * size;

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x,
                  (size + blockSize.y - 1) / blockSize.y,
                  (size + blockSize.z - 1) / blockSize.z);

    double sum = 0.0;
    int count = 0;

    while (hack_ProcessNextFrame(sensor)) {
        Stopwatch watch{};

        auto transform = hack_GetTransform(sensor);

        settings_cpu.transform_0.data_0.m_data[0] = make_float4(transform[0].x, transform[0].y, transform[0].z, transform[0].w);
        settings_cpu.transform_0.data_0.m_data[1] = make_float4(transform[1].x, transform[1].y, transform[1].z, transform[1].w);
        settings_cpu.transform_0.data_0.m_data[2] = make_float4(transform[2].x, transform[2].y, transform[2].z, transform[2].w);
        settings_cpu.transform_0.data_0.m_data[3] = make_float4(transform[3].x, transform[3].y, transform[3].z, transform[3].w);

        cudaMemcpy(BufferSettings, &settings_cpu, sizeof(settings_cpu), cudaMemcpyHostToDevice);
        cudaMemcpy(BufferDepthMap, hack_GetDepth(sensor), 640 * 480 * sizeof(float), cudaMemcpyHostToDevice);

        computeMain<<<gridSize, blockSize>>>();

        checkCudaError(cudaGetLastError());
        cudaDeviceSynchronize();

        sum += watch.end();
        count++;

        if (settings.every_step) {
            DumpToFileCuda(BufferVoxelGrid, settings.size, byte_size, "sdf_values", count);
        }
    }

    std::cout << "Average runtime: " << sum/count << std::endl;
    DumpToFileCuda(BufferVoxelGrid, settings.size, byte_size, "sdf_values", 0);

    cudaFree(BufferVoxelGrid);
    cudaFree(BufferVoxelGridWeights);
    cudaFree(BufferDepthMap);
}