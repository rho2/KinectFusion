
#if defined(_WIN32)
#include "windows.h"
#endif

#include "VirtualSensor.h"
#include "vulkan_helper.h"

#include <cstdio>
#include <iomanip>

namespace DH {
    using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#include "_autogen/icp_slang.h"
const auto& icp_shd = std::vector<uint32_t>{std::begin(icpSlang), std::end(icpSlang)};

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
    if (!sensor.Init("../../../Data/rgbd_dataset_freiburg1_xyz/")) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        exit(1);
    }

    const size_t size = 8;
    const size_t byte_size = size * size * size * sizeof(float);

    VulkanWrapper vulkanWrapper{"ICP"};

    auto& BufferCurrentVertexMap = vulkanWrapper.addBuffer(640 * 480 * sizeof(float));
    auto& BuffferLastVertexMap = vulkanWrapper.addBuffer(640 * 480 * sizeof(float));

    auto& BufferNormalMap = vulkanWrapper.addBuffer(640 * 480 * sizeof(float));
    auto& BufferLastNormalMap = vulkanWrapper.addBuffer(640 * 480 * sizeof(float));
    auto& BufferAta = vulkanWrapper.addBuffer(6 * 6 * sizeof(float));
    auto& BufferAtb = vulkanWrapper.addBuffer(6 * sizeof(float));

    vulkanWrapper.createPipeline(sizeof(DH::ICPSettings), icp_shd, "ICPCalcMain");

    std::cout << "Start to process\n";
    std::cout << "================================================================================================\n";

    DH::PerlinSettings perlinSettings{};
    perlinSettings.size = size;

    while (sensor.ProcessNextFrame()) {
        // BufferDepthMap.fillWith(sensor.GetDepth());
        {
            auto P = BufferNormalMap.map<float>();
            P[0] = 98.76;
            P[1] = 12.34;
        }

        auto& CmdBuffer = vulkanWrapper.startCommandBuffer();

        if (sensor.m_increment == 0) {
            CmdBuffer.fillBuffer(*BufferCurrentVertexMap._buffer, 0, byte_size, 0);
            CmdBuffer.fillBuffer(*BuffferLastVertexMap._buffer, 0, byte_size, 0);
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
            auto P = BufferCurrentVertexMap.map<float>();
            DumpToFile(P.data, byte_size, "sdf_values", sensor.m_currentIdx);
        }

        {
            auto P = BuffferLastVertexMap.map<float>();
            DumpToFile(P.data, byte_size, "sdf_weights", sensor.m_currentIdx);
        }

        break;
    }
}

