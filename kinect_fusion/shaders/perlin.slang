/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

#include "device_host.h"

#define WORKGROUP_SIZE 16
RWTexture3D<float> voxel_grid;
RWTexture3D<float> voxel_grid_weights;
RWTexture2D<float> depth_map;
ConstantBuffer<DepthInfo> dInfo;

[[vk::push_constant]] ConstantBuffer<PerlinSettings> perlinSettings;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
    if (!dInfo.is_ini) {
        voxel_grid[threadIdx] = 10000;
        voxel_grid_weights[threadIdx] = 0;
    }

    voxel_grid[threadIdx.xyz] = 10000;

    float4 coord;
    coord.xyz = (threadIdx.xyz - 128.) / 100.;
    coord.z = 1.0;

    float3 camera_coord = (dInfo.transform * coord).xyz;
    float3 pix_coord = dInfo.depthIntrinsics * camera_coord;

    if (pix_coord.z <= 0.0)
        return;

    uint2 pix = uint2(float2(pix_coord.x / pix_coord.z, pix_coord.y / pix_coord.z));

    if (pix.x < 0 || pix.x >= dInfo.width || pix.y < 0 || pix.y >= dInfo.height)
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

    voxel_grid[threadIdx.xyz] = voxel_grid[threadIdx] - 5001;
    voxel_grid_weights[threadIdx.xyz] = new_weight;
}
