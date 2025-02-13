
#if defined(_WIN32)
#include "windows.h"
#endif

#include "VirtualSensor.h"
#include "vulkan_helper.h"

#include <cstdio>
#include <iomanip>

namespace DH
{
    using namespace glm;
#include "shaders/device_host.h" // Shared between host and device
} // namespace DH

#include "_autogen/icp_slang.h"
const auto &icp_shd = std::vector<uint32_t>{std::begin(icpSlang), std::end(icpSlang)};

void DumpToFile(const void *Data, const uint32_t size, const std::string &prefix, const uint32_t index)
{
    std::ostringstream filename;
    filename << prefix << "_" << std::setw(4) << std::setfill('0') << index << ".bin";

    std::ofstream file(filename.str(), std::ios::binary);
    file.write(reinterpret_cast<const char *>(Data), size);
    file.close();

    std::cout << "Wrote: " << filename.str() << std::endl;
}

void FillVertexNormalMap(std::vector<Vector4f> &targetMap, std::vector<Vector4f> &targetNormalMap, const Matrix4f &depthExtrinsics, const Matrix3f &depthIntrinsics, const float *depthMap, const int width, const int height)
{
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
    for (int v = 0; v < height; ++v)
    {
        // For every pixel in a row.
        for (int u = 0; u < width; ++u)
        {
            unsigned int idx = v * width + u; // linearized index
            float depth = depthMap[idx];
            if (depth == MINF)
            {
                targetMap[idx] = Vector4f(MINF, MINF, MINF, MINF);
            }
            else
            {
                // Back-projection to camera space.
                targetMap[idx] = (rotationInv * Vector3f((u - cX) / fovX * depth, (v - cY) / fovY * depth, depth) + translationInv).homogeneous();
            }
        }
    }

#pragma omp parallel for
    for (int v = 1; v < height - 1; ++v)
    {
        for (int u = 1; u < width - 1; ++u)
        {
            unsigned int idx = v * width + u; // linearized index

            const float du = 0.5f * (depthMap[idx + 1] - depthMap[idx - 1]);
            const float dv = 0.5f * (depthMap[idx + width] - depthMap[idx - width]);
            if (!std::isfinite(du) || !std::isfinite(dv) || abs(du) > maxDistanceHalved || abs(dv) > maxDistanceHalved)
            {
                targetNormalMap[idx] = Vector4f(MINF, MINF, MINF, MINF);
                continue;
            }

            Vector3f normal{(targetMap[idx + 1].head<3>() - targetMap[idx - 1].head<3>()).cross(targetMap[idx + width].head<3>() - targetMap[idx - width].head<3>())};
            normal.normalize();
            targetNormalMap[idx] = Vector4f(normal.x(), normal.y(), normal.z(), 0.f);
        }
    }

    // We set invalid normals for border regions.
    for (int u = 0; u < width; ++u)
    {
        targetNormalMap[u] = Vector4f(MINF, MINF, MINF, MINF);
        targetNormalMap[u + (height - 1) * width] = Vector4f(MINF, MINF, MINF, MINF);
    }
    for (int v = 0; v < height; ++v)
    {
        targetNormalMap[v * width] = Vector4f(MINF, MINF, MINF, MINF);
        targetNormalMap[(width - 1) + v * width] = Vector4f(MINF, MINF, MINF, MINF);
    }
}

void printVector(std::vector<Vector4f> vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i].x() << " " << vec[i].y() << " " << vec[i].z() << " " << vec[i].w() << std::endl;
    }
}

int main()
{
    VirtualSensor sensor;
    if (!sensor.Init("../../../Data/rgbd_dataset_freiburg1_xyz/"))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        exit(1);
    }

    const size_t size = 8;
    const size_t byte_size = size * size * size * sizeof(float);
    const size_t width = 640;
    const size_t height = 480;
    const size_t groupCount = 640 * 480 / 16;

    VulkanWrapper vulkanWrapper{"ICP"};

    const size_t mapSize = width * height * 4 * sizeof(float);
    auto &BufferCurrentVertexMap = vulkanWrapper.addBuffer(mapSize);
    auto &BuffferLastVertexMap = vulkanWrapper.addBuffer(mapSize);
    auto &BufferNormalMap = vulkanWrapper.addBuffer(mapSize);
    auto &BufferLastNormalMap = vulkanWrapper.addBuffer(mapSize);
    auto &BufferAta = vulkanWrapper.addBuffer(groupCount * 6 * 6 * sizeof(float));
    auto &BufferAtb = vulkanWrapper.addBuffer(groupCount * 6 * sizeof(float));

    vulkanWrapper.createPipeline(sizeof(DH::ICPSettings), icp_shd, "ICPCalcMain");

    std::cout << "Start to process\n";
    std::cout << "================================================================================================\n";

    DH::ICPSettings icpSettings{};
    // perlinSettings.size = size;
    sensor.ProcessNextFrame();
    Matrix4f lastFrame{sensor.GetTrajectory()};
    for (int i = 0; i < 4; i++)
    {
        icpSettings.lastFramePose[i].x = sensor.GetTrajectory().row(i).x();
        icpSettings.lastFramePose[i].y = sensor.GetTrajectory().row(i).y();
        icpSettings.lastFramePose[i].z = sensor.GetTrajectory().row(i).z();
        icpSettings.lastFramePose[i].w = sensor.GetTrajectory().row(i).w();
    }

    

    std::vector<Vector4f> initialVertexMap(width * height);
    std::vector<Vector4f> initialNormalMap(width * height);
    FillVertexNormalMap(initialVertexMap, initialNormalMap, sensor.GetDepthExtrinsics(), sensor.GetDepthIntrinsics(), sensor.GetDepth(), width, height);

    while (sensor.ProcessNextFrame())
    {

        std::vector<Vector4f> vertexMap(width * height);
        std::vector<Vector4f> normalMap(width * height);
        FillVertexNormalMap(vertexMap, normalMap, sensor.GetDepthExtrinsics(), sensor.GetDepthIntrinsics(), sensor.GetDepth(), width, height);
        BufferCurrentVertexMap.fillWith(vertexMap.data());
        BuffferLastVertexMap.fillWith(initialVertexMap.data());
        BufferNormalMap.fillWith(normalMap.data());
        BufferLastNormalMap.fillWith(initialNormalMap.data());
        const int icp_turns = 30;
        Matrix4f foo = lastFrame;
        for (int i = 0; i < 4; ++i)
        {
            icpSettings.pose[i].x = foo.row(i).x();
            icpSettings.pose[i].y = foo.row(i).y();
            icpSettings.pose[i].z = foo.row(i).z();
            icpSettings.pose[i].w = foo.row(i).w();
        }

        for (int turn = 0; turn < icp_turns; turn++)
        {

            {
                auto P = BufferAta.map<float>();
                for (auto i = 0; i < groupCount * 6 * 6; ++i)
                {
                    P[i] = 0.0f;
                }
            }
        
            {
                auto P = BufferAtb.map<float>();
                for (auto i = 0; i < groupCount * 6; ++i)
                {
                    P[i] = 0.0f;
                }
            }

            auto &CmdBuffer = vulkanWrapper.startCommandBuffer();
            Matrix3f intrinsics = sensor.GetDepthIntrinsics();

            icpSettings.width = 640;
            icpSettings.distanceThreshold = 0.1f;
            icpSettings.angleThreshold = 1.05f;

            vulkanWrapper.addCommandPushConstants(icpSettings);
            vulkanWrapper.submitAndWait(width / 16, height, 1);
            Eigen::Matrix<float, 6, 6> ATA;
            {
                auto P = BufferAta.map<float>();
                for (int i = 0; i < groupCount * 6 * 6; i++)
                {
                    ATA((i % 36) / 6, (i % 36) % 6) += P[i];
                }
            }

            Eigen::Vector<float, 6> ATB;
            {
                auto P = BufferAtb.map<float>();
                for (int i = 0; i < groupCount * 6; i++)
                {
                    ATB(i % 6) += P[i];
                }
            }

            JacobiSVD<MatrixXf> svd(ATA, ComputeThinU | ComputeThinV);
            VectorXf x = svd.solve(ATB);
            std::cout << x << std::endl;

            float alpha = x(0), beta = x(1), gamma = x(2);

            Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                                AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
                                AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

            Vector3f translation = x.tail(3);

            // Build the pose matrix using the rotation and translation matrices
            Matrix4f estimatedPose2 = Matrix4f::Identity();
            estimatedPose2.block(0, 0, 3, 3) = rotation;
            estimatedPose2.block(0, 3, 3, 1) = translation;

            foo = estimatedPose2 * foo;
            for (int i = 0; i < 4; ++i)
            {
                icpSettings.pose[i].x = foo.row(i).x();
                icpSettings.pose[i].y = foo.row(i).y();
                icpSettings.pose[i].z = foo.row(i).z();
                icpSettings.pose[i].w = foo.row(i).w();
            }
            std::cout << foo << std::endl;
        }

        break;
    }
}
