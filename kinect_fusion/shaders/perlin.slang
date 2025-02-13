#version 460

#include "device_host.h"

#define WORKGROUP_SIZE 16
RWStructuredBuffer<float> voxel_grid;
RWStructuredBuffer<float> voxel_grid_weights;
StructuredBuffer<float> depth_map;

[[vk::push_constant]] ConstantBuffer<PerlinSettings> perlinSettings;

const mat3 depthIntrinsics = mat3(
    vec3(525.0f, 0.0f, 319.5f),
    vec3(0.0f, 525.0f, 239.5f),
    vec3(0.0f, 0.0f, 1.0f)
);

[shader("compute")]
[numthreads(32, 4,  1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
    if (threadIdx.x >= perlinSettings.size || threadIdx.y >= perlinSettings.size || threadIdx.z >= perlinSettings.size)
        return;

    uint index = threadIdx.z * perlinSettings.size * perlinSettings.size + threadIdx.y * perlinSettings.size + threadIdx.x;

    float scale = perlinSettings.size/2.0;
    int offset = perlinSettings.size/2;

    float4 coord;
    coord.x =  -1.0f * (threadIdx.x - offset) / scale;
    coord.y = (threadIdx.z) / scale;
    coord.z = (threadIdx.y) / scale;
    coord.w = 1.0;

    float4 camera_coord = mul(perlinSettings.transform, coord);
    float3 pix_coord = mul(depthIntrinsics, camera_coord.xyz);

    if (pix_coord.z <= 0.0)
        return;

    uint2 pix = uint2(
        int(pix_coord.x / pix_coord.z),
        int(pix_coord.y / pix_coord.z)
    );

    if (pix.x < 0 || pix.x >= 640 || pix.y < 0 || pix.y >= 480)
        return;

    int pix_index = pix.y * 640 + pix.x;
    float d = depth_map[pix_index];

    if (isnan(d))
        return;

    float sdf = d - camera_coord.z;

    if (sdf > -TRUNC_DISTANCE) {
        float old_weight = voxel_grid_weights[index];
        float new_weight = old_weight + 1;

        float old_value = voxel_grid[index];
        float new_value = (old_value * 1 + sdf) / 2;

        voxel_grid[index] = clamp(new_value, -TRUNC_DISTANCE, TRUNC_DISTANCE);
        voxel_grid_weights[index] = new_weight;
    }
}
