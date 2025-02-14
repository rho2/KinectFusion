#version 460

#include "device_host.h"

[[vk::binding(0, 0)]] StructuredBuffer<float> voxel_grid;         
[[vk::binding(1, 0)]] StructuredBuffer<float> voxel_grid_weights;   //TODO: Implement weighted ICP
[[vk::binding(2, 0)]] StructuredBuffer<float4> voxel_color;         //color matching
[[vk::binding(3, 0)]] StructuredBuffer<float4> current_points;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4x4> icp_update;

[[vk::push_constant]] ConstantBuffer<ICPSettings> icpSettings;

shared float3 sharedSumCurrent[256];
shared float3 sharedSumTarget[256];
shared uint   sharedValidCount[256];
shared float3 sharedCov0[256]; 
shared float3 sharedCov1[256]; 
shared float3 sharedCov2[256]; 

float determinant3x3(float3x3 m)
{
    return dot(m[0], cross(m[1], m[2]));
}

float3x3 inverse3x3(float3x3 m)
{
    float3 c0 = cross(m[1], m[2]);
    float3 c1 = cross(m[2], m[0]);
    float3 c2 = cross(m[0], m[1]);
    float det = dot(m[0], c0);
    return float3x3(c0, c1, c2) / det;
}

float3x3 transpose3x3(float3x3 m)
{
    return float3x3(
        float3(m[0].x, m[1].x, m[2].x),
        float3(m[0].y, m[1].y, m[2].y),
        float3(m[0].z, m[1].z, m[2].z)
    );
}

float3x3 computeRotation(float3x3 cov)
{
    float3x3 R = cov;
    for (int i = 0; i < 10; i++)
    {
        float3x3 invTransR = transpose3x3(inverse3x3(R));
        R = 0.5 * (R + invTransR);
    }
    return R;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void icpCompute(uint3 threadIdx : SV_DispatchThreadID)
{
    uint idx = threadIdx.x;
    //local stuff
    float3 localSumCurrent = float3(0.0, 0.0, 0.0);
    float3 localSumTarget = float3(0.0, 0.0, 0.0);
    uint localValid = 0;
    float3 localCov0 = float3(0.0, 0.0, 0.0); 
    float3 localCov1 = float3(0.0, 0.0, 0.0); 
    float3 localCov2 = float3(0.0, 0.0, 0.0); 

    if (idx < icpSettings.numPoints)
    {
        float4 curPoint = current_points[idx];
        if (curPoint.w > 0.0)
        {
            uint base = idx * 3;
            float3 targetPoint = float3(
                voxel_grid[base],
                voxel_grid[base+1],
                voxel_grid[base+2]
            );
            localSumCurrent = curPoint.xyz;
            localSumTarget  = targetPoint;
            localValid = 1;
            

            localCov0 = targetPoint * curPoint.x;  
            localCov1 = targetPoint * curPoint.y; 
            localCov2 = targetPoint * curPoint.z;
        }
    }

    sharedSumCurrent[idx] = localSumCurrent;
    sharedSumTarget[idx]  = localSumTarget;
    sharedValidCount[idx] = localValid;
    sharedCov0[idx] = localCov0;
    sharedCov1[idx] = localCov1;
    sharedCov2[idx] = localCov2;

    GroupMemoryBarrierWithGroupSync();

    // Reduction
    for (uint stride = 128; stride > 0; stride /= 2)
    {
        if (idx < stride)
        {
            sharedSumCurrent[idx] += sharedSumCurrent[idx + stride];
            sharedSumTarget[idx]  += sharedSumTarget[idx + stride];
            sharedValidCount[idx] += sharedValidCount[idx + stride];
            sharedCov0[idx] += sharedCov0[idx + stride];
            sharedCov1[idx] += sharedCov1[idx + stride];
            sharedCov2[idx] += sharedCov2[idx + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    //Thread 0 computes the final transformation.
    if (idx == 0)
    {
        uint validCount = sharedValidCount[0];
        if (validCount == 0)
        {
            icp_update[0] = float4x4(
                float4(1, 0, 0, 0),
                float4(0, 1, 0, 0),
                float4(0, 0, 1, 0),
                float4(0, 0, 0, 1)
            );
            return;
        }
        
        float3 centroidCurrent = sharedSumCurrent[0] / float(validCount);
        float3 centroidTarget  = sharedSumTarget[0] / float(validCount);
        
        float3x3 M = float3x3(
            sharedCov0[0],
            sharedCov1[0],
            sharedCov2[0]
        );
        float3x3 centroidOuter = float3x3(
            centroidTarget * centroidCurrent.x,
            centroidTarget * centroidCurrent.y,
            centroidTarget * centroidCurrent.z
        );
        M = M - float3x3(
            centroidOuter[0] * float(validCount),
            centroidOuter[1] * float(validCount),
            centroidOuter[2] * float(validCount)
        );
        
        float3x3 R = computeRotation(M);
        
        float3 t = centroidTarget - mul(R, centroidCurrent);
        
        float4x4 transformation = float4x4(
            float4(R[0], 0),
            float4(R[1], 0),
            float4(R[2], 0),
            float4(t,    1)
        );
        
        icp_update[0] = transformation;
    }
}