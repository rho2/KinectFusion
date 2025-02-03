#version 460

#define GROUPDIMENSION 16
#define THREADS_PER_GROUP (GROUPDIMENSION * GROUPDIMENSION)

#include "device_host.h"

// Input buffers for the current and predicted frames
[[vk::push_constant]] ConstantBuffer<ICPSettings> icpSettings;

Texture2D<float3>  currentVertexMap;   // Input current vertexMap V_(u)
Texture2D<float3> lastVertexMap;      // Input last vertexMap Vhat
Texture2D<float3> currentNormalMap;  // Input current normalMap N_(u)
Texture2D<float3> lastNormalMap;     // Input past normalMap Nhat
RWStructuredBuffer<float[6][6]> ata;       // Output ata
RWStructuredBuffer<float[6]> atb;

groupshared float[6][6] localATA[THREADS_PER_GROUP];
groupshared float[6] localATB[THREADS_PER_GROUP];

float4x4 transformInverse(float4x4 transform) {
    float3x3 rotation = (float3x3)transform; // Extract the 3x3 rotation part
    float3 translation = transform[3].xyz;   // Extract the translation part

    float3x3 invRotation = transpose(rotation);             // Inverse of rotation matrix is its transpose
    float3 invTranslation = -mul(invRotation, translation); // Adjust the translation

    float4x4 inverseTransform = float4x4(
        float4(invRotation[0], 0),
        float4(invRotation[1], 0),
        float4(invRotation[2], 0),
        float4(invTranslation, 1)
    );

    return inverseTransform;
}


[shader("compute")]
[numthreads(GROUPDIMENSION, 1, 1)]
void ICPCalcMain(uint3 threadId: SV_DispatchThreadID, uint3 gtid: SV_GroupThreadID, uint3 GId: SV_GroupID) {
    uint index = threadId.x + threadId.y * icpSettings.width;
    uint2 pixelCoord = threadId.xy;

    float3 v = currentVertexMap.Load(int3(pixelCoord, 0));
    float3 n = currentNormalMap.Load(int3(pixelCoord, 0));
    // VertexNormal u_hat = predictedFrame[index];
    // compute v_hat calculate t_(k-1,k). bring in to camera image space. apply perspective. we now have u_hat. check the value of u_hat from the past vertex map
    float4x4 lastFrameToCurrentFrame = mul(transformInverse(icpSettings.lastFramePose), icpSettings.pose);
    float3 currentVertexInLasFrame = mul(lastFrameToCurrentFrame, float4(v, 1.0)).xyz;
    float3 homogenousCoord = float3(currentVertexInLasFrame.x / currentVertexInLasFrame.z, currentVertexInLasFrame.y / currentVertexInLasFrame.z, 1.0);
    float3 pixelPos = float3(mul(icpSettings.cameraProjection, homogenousCoord).xy, 0);
    float3 v_hat = lastVertexMap.Load((int3)floor(pixelPos));
    float3 n_hat = lastNormalMap.Load((int3)floor(pixelPos));

    float3 transformedNormal = mul(icpSettings.lastFramePose , float4(n,0)).xyz;

    if (length(v_hat - v) > icpSettings.distanceThreshold || dot(normalize(n), normalize(n_hat)) > icpSettings.angleThreshold) {
        return;
    }

    float3x3 skewSymmetricCurrentV = float3x3(0, -v.z, v.y,
                                              v.z, 0, -v.x,
                                              -v.y, v.x, 0);

    float3 at_0 = mul(skewSymmetricCurrentV, n_hat);
    float at_array[6] = { at_0.x, at_0.y, at_0.z, n_hat.x, n_hat.y, n_hat.z };


    float partialAta[6][6];
    [unroll]
    for (int i = 0; i < 6; i++) {
        [unroll]
        for (int j = 0; j < 6; j++) {
            partialAta[i][j] = at_array[i] * at_array[j];
        }
    }
    

    // TODO: check this matrix( 6?)
    matrix<float, 6, 1> at;
    
    float partialAtb[6];

    float b = dot(n_hat, v_hat - v);
    [unroll]
    for (int i = 0; i < 6; i++) {
        partialAtb[i] = b * at_array[i];
    }

    localATA[gtid.x] = partialAta;
    localATB[gtid.x] = partialAtb;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = GROUPDIMENSION / 2; s > 0; s >>= 1) {
        if (gtid.x < s) {
            [unroll]
            for (int i = 0; i < 6; i++) {
                [unroll]
                for (int j = 0; j < 6; j++) {
                    localATA[gtid.x][i][j] += localATA[gtid.x + s][i][j];
                }
                localATB[gtid.x][i] += localATB[gtid.x + s][i];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gtid.x == 0)
    {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                // InterlockedAdd(ata[i * 6 + j], localATA[0][i][j]);
                ata[GId.x][i][j] += localATA[0][i][j];
                // atomicAddAta(i, j, localATA[0][i][j]);
            }
            // InterlockedAdd(atb[i], localATB[0][i]);
            atb[GId.x][i] += localATB[0][i];
            // atomicAddAtb(i, localATB[0][i]);
        }
    }

}