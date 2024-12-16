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
RWTexture2D<float> depth_map;
ConstantBuffer<DepthInfo> dInfo;

[[vk::push_constant]] ConstantBuffer<PerlinSettings> perlinSettings;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
    if (isnan(depth_map[threadIdx.xy]))
        return;

    float3 coord = float3(threadIdx.x, threadIdx.y, 1.0);
    float3 coord_cam_space = dInfo.depthIntrinsicsInv * (depth_map[threadIdx.xy] * coord);

    float4 coord_cam_space_homo;
    coord_cam_space_homo.xyz = coord_cam_space;
    coord_cam_space_homo.w = 1.0;

    float4 pos = dInfo.transform * coord_cam_space_homo;

    int vx = int(floor(100.0 * pos.x) + (perlinSettings.size / 2.0f));
    int vy = int(floor(100.0 * pos.z) + (perlinSettings.size / 2.0f));
    int vz = int(floor(100.0 * pos.y) + (perlinSettings.size / 2.0f));

    if (vx >= perlinSettings.size || vy >= perlinSettings.size || vz >= perlinSettings.size || vx < 0 || vy < 0 || vz < 0) {

    } else {
        voxel_grid[uint3(vx, vy, vz)] = 1;
    }
}
