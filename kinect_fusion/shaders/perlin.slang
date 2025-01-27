#version 460

#include "device_host.h"

#define WORKGROUP_SIZE 16
RWTexture3D<float> voxel_grid;
RWTexture3D<float> voxel_grid_weights;
RWTexture2D<float> depth_map;

[[vk::push_constant]] ConstantBuffer<PerlinSettings> perlinSettings;

const mat3 depthIntrinsics = mat3(
    vec3(525.0f, 0.0f, 319.5f),
    vec3(0.0f, 525.0f, 239.5f),
    vec3(0.0f, 0.0f, 1.0f)
);

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
    if (!perlinSettings.is_ini) {
        voxel_grid[threadIdx] = 10000;
        voxel_grid_weights[threadIdx] = 0;
    }

    float4 coord;
    coord.xyz = (threadIdx.xyz - 128.) / 100.;
    coord.w = 1.0;

    float3 camera_coord = (perlinSettings.transform * coord).xyz;
    float3 pix_coord = depthIntrinsics * camera_coord;

    if (pix_coord.z <= 0.0)
        return;

    uint2 pix = uint2(
        int(pix_coord.x / pix_coord.z),
        int(pix_coord.y / pix_coord.z)
    );

    if (pix.x < 0 || pix.x >= 640 || pix.y < 0 || pix.y >= 480)
        return;

    float d = depth_map[pix];

    if (isnan(d))
        return;

    float sdf = d - camera_coord.z;
    float trunc_distance = 0.1f;

    if (sdf < -trunc_distance)
        return;

    sdf = min(1.0, abs(sdf) / trunc_distance) * copysign(1.0f, sdf);

    float old_weight = voxel_grid_weights[threadIdx.xyz];
    float new_weight = old_weight + 1;

    float old_value = voxel_grid[threadIdx.xyz];
    float new_value = (old_value * old_weight + sdf) / new_weight;

    voxel_grid[threadIdx.xyz] = new_value;
    voxel_grid_weights[threadIdx.xyz] = new_weight;
}
