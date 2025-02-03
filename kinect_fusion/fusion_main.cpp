#include "VirtualSensor.h"
#include "vulkan_helper.h"

#include <cstdio>
#include <iomanip>

namespace DH {
    using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#include "_autogen/perlin_slang.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(perlinSlang), std::end(perlinSlang)};


void DumpToFile(const void* Data, const uint32_t size, const std::string& prefix, const uint32_t index) {
    std::ostringstream filename;
    filename << prefix << "_" << std::setw(4) << std::setfill('0') << index << ".bin";

    std::ofstream file(filename.str(), std::ios::binary);
    file.write(reinterpret_cast<const char*>(Data), size);
    file.close();

    std::cout << "Wrote: " << filename.str() << std::endl;
}

int main() {
    VirtualSensor sensor;
    if (!sensor.Init("../Data/rgbd_dataset_freiburg1_xyz/")) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        exit(1);
    }

    const size_t size = 8;
    const size_t byte_size = size * size * size * sizeof(float);

    VulkanWrapper vulkanWrapper{"VolumetricFusion"};

    auto& BufferVoxelGrid = vulkanWrapper.addBuffer(byte_size);
    auto& BufferVoxelGridWeights = vulkanWrapper.addBuffer(byte_size);

    auto& BufferDepthMap = vulkanWrapper.addBuffer(640 * 480 * sizeof(float));

    vulkanWrapper.createPipeline(sizeof(DH::PerlinSettings), comp_shd, "computeMain");

    std::cout << "Start to process\n";
    std::cout << "================================================================================================\n";

    DH::PerlinSettings perlinSettings{};
    perlinSettings.size = size;

    while (sensor.ProcessNextFrame()) {
        // BufferDepthMap.fillWith(sensor.GetDepth());
        {
            auto P = BufferDepthMap.map<float>();
            P[0] = 98.76;
            P[1] = 12.34;
        }

        auto& CmdBuffer = vulkanWrapper.startCommandBuffer();

        if (sensor.m_increment == 0) {
            CmdBuffer.fillBuffer(*BufferVoxelGrid._buffer, 0, byte_size, 0);
            CmdBuffer.fillBuffer(*BufferVoxelGridWeights._buffer, 0, byte_size, 0);
        }

        Matrix4f foo = sensor.GetTrajectory() * sensor.GetDepthExtrinsics();
        for (int i = 0; i < 4; ++i) {
            perlinSettings.transform[i].x = foo.row(i).x();
            perlinSettings.transform[i].y = foo.row(i).y();
            perlinSettings.transform[i].z = foo.row(i).z();
            perlinSettings.transform[i].w = foo.row(i).w();
        }
        vulkanWrapper.addCommandPushConstants(perlinSettings);
        vulkanWrapper.submitAndWait(size, size, size);
        {
            auto P = BufferVoxelGrid.map<float>();
            DumpToFile(P.data, byte_size, "sdf_values", sensor.m_currentIdx);
        }

        {
            auto P = BufferVoxelGridWeights.map<float>();
            DumpToFile(P.data, byte_size, "sdf_weights", sensor.m_currentIdx);
        }

        break;
    }
}
