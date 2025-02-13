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

struct Setting {
    std::string filename = "../Data/rgbd_dataset_freiburg1_xyz/";
    uint32_t size = 256;
    bool every_step = false;
    bool cpu = false;
};

void ParseCommandLine(int argc, char** argv, Setting& settings) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-f" || arg == "--filename") {
            if (i + 1 < argc) {
                settings.filename = argv[++i];
            } else {
                std::cerr << "Error: No filename provided for -f/--filename option." << std::endl;
                exit(EXIT_FAILURE);
            }
        } else if (arg == "-s" || arg == "--size") {
            if (i + 1 < argc) {
                try {
                    settings.size = std::stoi(argv[++i]);
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Invalid value for size." << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                std::cerr << "Error: No size provided for -s/--size option." << std::endl;
                exit(EXIT_FAILURE);
            }
        } else if (arg == "-e" || arg == "--every-step") {
            settings.every_step = true;
        } else if (arg == "-c" || arg == "--cpu") {
            settings.cpu = true;
        }else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [-f filename] [-s size] [-e]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

void DumpToFile(const void *Data, const uint32_t voxel_size, const uint32_t size, const std::string &prefix, const uint32_t index) {
    std::ostringstream filename;
    filename << prefix << "_" << std::setw(4) << std::setfill('0') << index << ".bin";

    std::ofstream file(filename.str(), std::ios::binary);
    file.write(reinterpret_cast<const char *>(&voxel_size), sizeof(voxel_size));

    file.write(reinterpret_cast<const char *>(Data), size);
    file.close();

    std::cout << "Wrote: " << filename.str() << " with size " << voxel_size << std::endl;
}

void run_gpu(VirtualSensor &sensor, const Setting& settings) {
    const size_t size = settings.size;
    const size_t byte_size = size * size * size * sizeof(float);

    VulkanWrapper vulkanWrapper{"VolumetricFusion"};

    auto &BufferVoxelGrid = vulkanWrapper.addBuffer(byte_size);
    auto &BufferVoxelGridWeights = vulkanWrapper.addBuffer(byte_size);

    {
        auto P = BufferVoxelGrid.map<float>();
        for (auto i = 0; i < size * size * size; ++i) { P[i] = TRUNC_DISTANCE + 1; }
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
    perlinSettings.size2 = size * size;
    perlinSettings.size3 = size * size * size;
    perlinSettings.inv_scale = 2.0f / size;

    double sum = 0.0;
    int count = 0;

    while (sensor.ProcessNextFrame()) {
        Stopwatch watch{};

        Matrix4f transform = sensor.GetTrajectory() * sensor.GetDepthExtrinsics();

        BufferDepthMap.fillWith(sensor.GetDepth());

        auto &CmdBuffer = vulkanWrapper.startCommandBuffer();

        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                perlinSettings.transform[i][j] = transform(j, i);
        perlinSettings.transform[0] *= -1;

        vulkanWrapper.addCommandPushConstants(perlinSettings);
        vulkanWrapper.submitAndWait(size, size, size);

        sum += watch.end();
        count++;

        if (settings.every_step) {
            auto P = BufferVoxelGrid.map<float>();
            DumpToFile(P.data, size, byte_size, "sdf_values", count);
        }
    }

    std::cout << "Average runtime: " << sum/count << std::endl;

    {
        auto P = BufferVoxelGrid.map<float>();
        DumpToFile(P.data, size, byte_size, "sdf_values", 0);
    }
}

void run_cpu(VirtualSensor &sensor, const Setting& settings) {
    const size_t size = settings.size;
    const size_t full_size = size * size * size;
    const size_t byte_size = full_size  * sizeof(float);

    auto voxel_grid = std::make_unique<float[]>(full_size);
    auto voxel_grid_weights = std::make_unique<float[]>(full_size);

    double sum = 0.0;
    int count = 0;

    while (sensor.ProcessNextFrame()) {
        Stopwatch watch{};

        float* depthMap = sensor.GetDepth();
        Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
        Matrix4f transform = sensor.GetTrajectory() * sensor.GetDepthExtrinsics();

        float scale = size / 2.0f;
        int offset  = size / 2;

        for (int z = 0; z < size; ++z) {
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    unsigned int index = z * size * size + y * size + x;

                    Vector4f coord = {
                        -1.0f * (x - offset) / scale,
                        z / scale,
                        y / scale,
                        1.0f
                    };

                    Vector4f camera_coord = transform * coord;
                    Vector3f pix_coord = depthIntrinsics * camera_coord.head(3);

                    if (pix_coord.z() <= 0.0) continue;

                    auto pix_x = static_cast<int>(pix_coord.x() / pix_coord.z());
                    auto pix_y = static_cast<int>(pix_coord.y() / pix_coord.z());

                    if (pix_x < 0 || pix_x >= 640  || pix_y < 0 || pix_y >= 480) {
                        continue;
                    }

                    const unsigned int depth_index = pix_y * 640 + pix_x;
                    float d = depthMap[depth_index];

                    if (!std::isfinite(d)) {continue;}

                    const float sdf = d - camera_coord.z();

                    if (sdf > -TRUNC_DISTANCE) {
                        const float old_weight = voxel_grid_weights[index];
                        const float new_weight = old_weight + 1;

                        const float old_value = voxel_grid[index];
                        const float new_value = (old_value * 1 + sdf) / 2;

                        voxel_grid[index] = (new_value > TRUNC_DISTANCE)? TRUNC_DISTANCE : (new_value < -TRUNC_DISTANCE)? -TRUNC_DISTANCE: new_value;
                        voxel_grid_weights[index] = new_weight;
                    }
                }
            }
        }

        sum += watch.end();
        count++;

        if (settings.every_step) {
            DumpToFile(voxel_grid.get(), size, byte_size, "sdf_values", count);
        }
    }

    std::cout << "Average runtime: " << sum/count << std::endl;
    DumpToFile(voxel_grid.get(), size, byte_size, "sdf_values", 0);
}

int main(int argc, char **argv) {
    VirtualSensor sensor;

    Setting settings;
    ParseCommandLine(argc, argv, settings);

    std::cout << "Running on " << settings.filename << std::endl;

    if (!sensor.Init(settings.filename)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        exit(1);
    }

    if (settings.cpu) {
        std::cout << "Running on CPU" << std::endl;
        run_cpu(sensor, settings);
    } else {
        std::cout << "Running on GPU" << std::endl;
        run_gpu(sensor, settings);
    }

}
