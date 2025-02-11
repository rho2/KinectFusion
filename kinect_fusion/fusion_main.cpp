#if defined(_WIN32)
#include "windows.h"
#endif

#include "VirtualSensor.h"
#include "vulkan_helper.h"

#include <cstdio>
#include <iomanip>
#include <glm/ext/matrix_transform.hpp>

namespace DH {
    using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
} // namespace DH

#include "_autogen/perlin_slang.h"
const auto &comp_shd = std::vector<uint32_t>{std::begin(perlinSlang), std::end(perlinSlang)};

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

void DumpToFile(const void *Data, const uint32_t size, const std::string &prefix, const uint32_t index) {
    std::ostringstream filename;
    filename << prefix << "_" << std::setw(4) << std::setfill('0') << index << ".bin";

    std::ofstream file(filename.str(), std::ios::binary);
    file.write(reinterpret_cast<const char *>(Data), size);
    file.close();

    std::cout << "Wrote: " << filename.str() << std::endl;
}

int main(int argc, char **argv) {
    VirtualSensor sensor;

    const char *file = "../Data/rgbd_dataset_freiburg1_xyz/";
    if (argc > 2) {
        file = argv[1];
    }

    if (!sensor.Init(file)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        exit(1);
    }

    const size_t size = 256;
    const size_t byte_size = size * size * size * sizeof(float);

    VulkanWrapper vulkanWrapper{"VolumetricFusion"};

    auto &BufferVoxelGrid = vulkanWrapper.addBuffer(byte_size);
    auto &BufferVoxelGridWeights = vulkanWrapper.addBuffer(byte_size);

    {
        auto P = BufferVoxelGrid.map<float>();
        for (auto i = 0; i < size * size * size; ++i) { P[i] = 0.1f; }
    }

    {
        auto P = BufferVoxelGridWeights.map<float>();
        for (auto i = 0; i < size * size * size; ++i) { P[i] = 0.0f; }
    }

    auto &BufferDepthMap = vulkanWrapper.addBuffer(640 * 480 * sizeof(float));

    vulkanWrapper.createPipeline(sizeof(DH::PerlinSettings), comp_shd, "computeMain");

    std::cout << "Start to process\n";
    std::cout << "================================================================================================\n";

    DH::PerlinSettings perlinSettings{};
    perlinSettings.size = size;


    double sum = 0.0;
    int count = 0;

    while (sensor.ProcessNextFrame()) {
        Stopwatch watch{};

        BufferDepthMap.fillWith(sensor.GetDepth());

        auto &CmdBuffer = vulkanWrapper.startCommandBuffer();

        Matrix4f transform = sensor.GetTrajectory() * sensor.GetDepthExtrinsics();

        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                perlinSettings.transform[i][j] = transform(j, i);
;

        vulkanWrapper.addCommandPushConstants(perlinSettings);
        vulkanWrapper.submitAndWait(size, size, size);

        sum += watch.end();
        count++;
    }

    std::cout << sum/count << std::endl;

    {
        auto P = BufferVoxelGrid.map<float>();
        DumpToFile(P.data, byte_size, "sdf_values", 0);
    }
}
