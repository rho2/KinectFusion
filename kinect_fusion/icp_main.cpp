
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

void FillVertexNormalMap(std::vector<Vector3f>& targetMap, std::vector<Vector3f>& targetNormalMap, const Matrix4f& depthExtrinsics, const Matrix3f& depthIntrinsics, const float* depthMap, const int width, const int height) {
    const float maxDistance = 1000;
    float fovX = depthIntrinsics(0, 0);
    float fovY = depthIntrinsics(1, 1);
    float cX = depthIntrinsics(0, 2);
    float cY = depthIntrinsics(1, 2);
    const float maxDistanceHalved = maxDistance / 2.f;

    // Compute inverse depth extrinsics.
    Matrix4f depthExtrinsicsInv = depthExtrinsics.inverse();
    Matrix3f rotationInv = depthExtrinsicsInv.block(0, 0, 3, 3);
    Vector3f translationInv = depthExtrinsicsInv.block(0, 3, 3, 1);


    // For every pixel row.
#pragma omp parallel for
    for (int v = 0; v < height; ++v) {
        // For every pixel in a row.
        for (int u = 0; u < width; ++u) {
            unsigned int idx = v * width + u; // linearized index
            float depth = depthMap[idx];
            if (depth == MINF) {
                targetMap[idx] = Vector3f(MINF, MINF, MINF);
            }
            else {
                // Back-projection to camera space.
                targetMap[idx] = rotationInv * Vector3f((u - cX) / fovX * depth, (v - cY) / fovY * depth, depth) + translationInv;
            }
        }
    }

    #pragma omp parallel for
        for (int v = 1; v < height - 1; ++v) {
            for (int u = 1; u < width - 1; ++u) {
                unsigned int idx = v * width + u; // linearized index

                const float du = 0.5f * (depthMap[idx + 1] - depthMap[idx - 1]);
                const float dv = 0.5f * (depthMap[idx + width] - depthMap[idx - width]);
                if (!std::isfinite(du) || !std::isfinite(dv) || abs(du) > maxDistanceHalved || abs(dv) > maxDistanceHalved) {
                    targetNormalMap[idx] = Vector3f(MINF, MINF, MINF);
                    continue;
                }

                targetNormalMap[idx] = (targetMap[idx + 1] - targetMap[idx - 1]).cross(targetMap[idx + width] - targetMap[idx - width]);
                targetNormalMap[idx].normalize();
            }
        }

        // We set invalid normals for border regions.
        for (int u = 0; u < width; ++u) {
            targetNormalMap[u] = Vector3f(MINF, MINF, MINF);
            targetNormalMap[u + (height - 1) * width] = Vector3f(MINF, MINF, MINF);
        }
        for (int v = 0; v < height; ++v) {
            targetNormalMap[v * width] = Vector3f(MINF, MINF, MINF);
            targetNormalMap[(width - 1) + v * width] = Vector3f(MINF, MINF, MINF);
        }

}

int main() {
    VirtualSensor sensor;
    if (!sensor.Init("../../../Data/rgbd_dataset_freiburg1_xyz/")) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        exit(1);
    }

    const size_t size = 8;
    const size_t byte_size = size * size * size * sizeof(float);
    const size_t width = 640;
    const size_t height = 480;

    VulkanWrapper vulkanWrapper{"ICP"};


    const size_t mapSize = width * height * 3 * sizeof(float);
    auto& BufferCurrentVertexMap = vulkanWrapper.addBuffer(mapSize);
    auto& BuffferLastVertexMap = vulkanWrapper.addBuffer(mapSize);
    auto& BufferNormalMap = vulkanWrapper.addBuffer(mapSize);
    auto& BufferLastNormalMap = vulkanWrapper.addBuffer(mapSize);
    auto& BufferAta = vulkanWrapper.addBuffer(6 * 6 * sizeof(float));
    auto& BufferAtb = vulkanWrapper.addBuffer(6 * sizeof(float));

    vulkanWrapper.createPipeline(sizeof(DH::ICPSettings), icp_shd, "ICPCalcMain");

    std::cout << "Start to process\n";
    std::cout << "================================================================================================\n";

    DH::ICPSettings icpSettings{};
    // perlinSettings.size = size;
    sensor.ProcessNextFrame();
    Matrix4f lastFrame {sensor.GetTrajectory()};
    for (int i = 0 ; i < 4 ; i ++) {
        icpSettings.lastFramePose[i].x = sensor.GetTrajectory().row(i).x();
        icpSettings.lastFramePose[i].y = sensor.GetTrajectory().row(i).y();
        icpSettings.lastFramePose[i].z = sensor.GetTrajectory().row(i).z();
        icpSettings.lastFramePose[i].w = sensor.GetTrajectory().row(i).w();
    }

    std::vector<Vector3f> initialVertexMap(width * height);
    std::vector<Vector3f> initialNormalMap(width * height);
    FillVertexNormalMap(initialVertexMap, initialNormalMap, sensor.GetDepthExtrinsics(), sensor.GetDepthIntrinsics(), sensor.GetDepth(), width, height);

    while (sensor.ProcessNextFrame()) {

        std::vector<Vector3f> vertexMap(width * height);
        std::vector<Vector3f> normalMap(width * height);
        FillVertexNormalMap(vertexMap, normalMap, sensor.GetDepthExtrinsics(), sensor.GetDepthIntrinsics(), sensor.GetDepth(), width, height);
        BufferCurrentVertexMap.fillWith(vertexMap.data());
        BuffferLastVertexMap.fillWith(initialVertexMap.data());
        BufferNormalMap.fillWith(normalMap.data());
        BufferLastNormalMap.fillWith(initialNormalMap.data());

        auto& CmdBuffer = vulkanWrapper.startCommandBuffer();

        // if (sensor.m_increment == 0) {
        //     CmdBuffer.fillBuffer(*BufferCurrentVertexMap._buffer, 0, byte_size, 0);
        //     CmdBuffer.fillBuffer(*BuffferLastVertexMap._buffer, 0, byte_size, 0);
        // }

        Matrix4f foo = lastFrame;
        for (int i = 0; i < 4; ++i) {
            icpSettings.pose[i].x = lastFrame.row(i).x();
            icpSettings.pose[i].y = lastFrame.row(i).y();
            icpSettings.pose[i].z = lastFrame.row(i).z();
            icpSettings.pose[i].w = lastFrame.row(i).w();
        }
        Matrix3f intrinsics = sensor.GetDepthIntrinsics();
        for (int i = 0; i < 3; ++i) {
            icpSettings.cameraProjection[i].x = intrinsics.row(i).x();
            icpSettings.cameraProjection[i].y = intrinsics.row(i).y();
            icpSettings.cameraProjection[i].z = intrinsics.row(i).z();
        }
        icpSettings.width = 640;
        icpSettings.distanceThreshold = 0.1f;
        icpSettings.angleThreshold = 0.1f;


        vulkanWrapper.addCommandPushConstants(icpSettings);
        vulkanWrapper.submitAndWait(size, size, size);
        {
            auto P = BufferAta.map<float>();
            DumpToFile(P.data, byte_size, "sdf_values", sensor.m_currentIdx);
        }

        {
            auto P = BufferAtb.map<float>();
            DumpToFile(P.data, byte_size, "sdf_weights", sensor.m_currentIdx);
        }

        break;
    }
}

