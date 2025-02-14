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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 460

#include "common/shaders/functions.hlsli"
#include "device_host.h"

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  float3 position : POSITION;
};

// Output of the vertex shader, and input to the fragment shader.
struct PSin
{
  float3 position : POSITION;
};

// Output of the vertex shader
struct VSout
{
  PSin stage;
  float4 sv_position : SV_Position;
};

// Output of the fragment shader
struct PSout
{
  float4 color : SV_Target;
};


[[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
[[vk::binding(0, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(1)]] StructuredBuffer<float> voxel_grid;

// Ray structure
struct Ray
{
  float3 origin;
  float3 direction;
};

struct Bbox
{
  float3 bMin;
  float3 bMax;
};

// Determines whether a ray intersects with a bounding box (bbox) representing a cube.
// If an intersection occurs, it calculates the two points of intersection and returns
// them as output.
// The function returns a boolean value indicating whether an intersection took place.
bool intersectCube(in Ray ray, in Bbox bbox, out float3 p1, out float3 p2)
{
  p1 = float3(0.0, 0.0, 0.0);
  p2 = float3(0.0, 0.0, 0.0);

  float3 invDir = 1.0 / ray.direction;

  float3 tMin = (bbox.bMin - ray.origin) * invDir;
  float3 tMax = (bbox.bMax - ray.origin) * invDir;

  float3 t1 = min(tMin, tMax);
  float3 t2 = max(tMin, tMax);

  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar = min(min(t2.x, t2.y), t2.z);

  
  if (tNear > tFar || tFar < 0.0) {
    return false;
  }

  p1 = ray.origin + ray.direction * max(tNear, 0.0);
  p2 = ray.origin + ray.direction * tFar;

  return true;
}

float getValueAt(float3 p) {
    int3 scaled = clamp(int3(p * (pushConst.size - 1)), 0, pushConst.size - 1);
    uint index = scaled.z * pushConst.size * pushConst.size  + scaled.y * pushConst.size  + scaled.x;
    return voxel_grid[index];
}

float3 computeVolumeGradient(float3 p, float voxelSize)
{
  float inc = voxelSize * 0.5F;
  float dx = (getValueAt(p - float3(inc, 0, 0)) - getValueAt(p + float3(inc, 0, 0))) / voxelSize;
  float dy = (getValueAt(p - float3(0, inc, 0)) - getValueAt(p + float3(0, inc, 0))) / voxelSize;
  float dz = (getValueAt(p - float3(0, 0, inc)) - getValueAt(p + float3(0, 0, inc))) / voxelSize;

  return normalize(float3(dx, dy, dz));
}

bool rayMarchingLERP(const float3 p1, const float3 p2, const int numSteps, out float3 hitPoint) {
    const float3 stepSize = (p2 - p1) / float(numSteps);
    hitPoint = p1;
    float prevValue = getValueAt(hitPoint);

    for (int i = 0; i < numSteps; ++i) {
        float3 newHitPoint = hitPoint + stepSize;
        float currentValue = getValueAt(newHitPoint);

        if (currentValue < 0) {
            float t = clamp(prevValue / (prevValue - currentValue), 0.0f, 1.0f);
            hitPoint = lerp(hitPoint, newHitPoint, t);
            return true;
        }

        prevValue = currentValue;
        hitPoint = newHitPoint;
    }

    return false;
}

bool rayMarching(const float3 p1, const float3 p2, const int numSteps, out float3 hitPoint)
{
  const float3 stepSize = (p2 - p1) / float(numSteps);
  hitPoint = p1;

  for (int i = 0; i < numSteps; ++i) {
    float value = getValueAt(hitPoint);

    if (value < 0) {
      return true;
    }

    hitPoint += stepSize;
  }

  return false;
}

// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  float4 pos = mul(pushConst.transfo, float4(input.position.xyz, 1.0));

  VSout output;
  output.sv_position = mul(frameInfo.proj, mul(frameInfo.view, float4(pos)));
  output.stage.position = pos.xyz;

  return output;
}


// Fragment Shader
[shader("pixel")]
PSout fragmentMain(PSin stage, bool isFrontFacing : SV_IsFrontFace)
{
  PSout output;
  
  Ray ray;
  ray.origin = frameInfo.camPos;
  ray.direction = normalize(stage.position - frameInfo.camPos);

  // Intersection against the cube
  float3 p1, p2;
  Bbox bbox = { float3(-0.5, -0.5, -0.5), float3(0.5, 0.5, 0.5) };
  bool hit = intersectCube(ray, bbox, p1, p2);
  if (!hit)
    discard;

  // Avoid drawing triangle that can't be seen
  bool isInside = all(p1 == ray.origin);
  if ((isInside && isFrontFacing) || (!isInside && !isFrontFacing))
    discard;
  
  // Uniform position [0,1] for sampling in the volume
  p1 = p1 - bbox.bMin / (bbox.bMax - bbox.bMin);
  p2 = p2 - bbox.bMin / (bbox.bMax - bbox.bMin);

  // Ray-marching
  float3 hitPoint;

  if (pushConst.render_lerp)
    hit = rayMarchingLERP(p1, p2, pushConst.steps, hitPoint);
  else
    hit = rayMarching(p1, p2, pushConst.steps, hitPoint);
  if (!hit)
    discard;

  float3 normal = computeVolumeGradient(hitPoint, 1.0 / pushConst.size);
  if (dot(ray.direction, normal) > 0)
    normal *= -1;

  if (pushConst.render_normals) {
    normal.x *= -1;
    normal.z *= -1;

    output.color.xyz = normal.xyz;
  } else {
    float3 lightDir = (frameInfo.headlight == 1) ? -ray.direction : normalize(frameInfo.toLight);
    float3 V = -ray.direction;
    output.color.xyz =  simpleShading(V, lightDir, normal, pushConst.color.xyz);
  }

  output.color.w = 1;
  return output;
}
