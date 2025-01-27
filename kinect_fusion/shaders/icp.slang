
#define GROUPDIMENSION 16


// Input buffers for the current and predicted frames
cbuffer FrameData : register(b0) {
    float4x4 pose;
    float4x4 lastFramePose;
    float3x3 cameraProjection;
    int width;
    float distanceThreshold;      // Distance threshold for correspondence
    float angleThreshold;         // Cosine of the angle threshold
};

Texture2D<float3>  currentVertexMap: register(t0);   // Input current vertexMap V_(u)
Texture2D<float3> lastVertexMap : register(t1);      // Input last vertexMap Vhat
Texture2D<float3> currentNormalMap : register(t2);  // Input current normalMap N_(u)
Texture2D<float3> lastNormalMap : register(t3);     // Input past normalMap Nhat
RWStructuredBuffer<float[6]> ata : register(u0);       // Output ata
RWStructuredBuffer<float[6]> atb : register(u1);

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
[numthreads(GROUPDIMENSION, GROUPDIMENSION, 1)]
void ICPCalcMain(uint3 threadId: SV_DispatchThreadID) {
    uint index = threadId.x + threadId.y * width;
    uint2 pixelCoord = threadId.xy;

    float3 v = currentVertexMap.Load(int3(pixelCoord, 0));
    float3 n = currentNormalMap.Load(int3(pixelCoord, 0));
    // VertexNormal u_hat = predictedFrame[index];
    // compute v_hat calculate t_(k-1,k). bring in to camera image space. apply perspective. we now have u_hat. check the value of u_hat from the past vertex map
    float4x4 lastFrameToCurrentFrame = mul(transformInverse(lastFramePose), pose);
    float3 currentVertexInLasFrame = mul(lastFrameToCurrentFrame, float4(v, 1.0)).xyz;
    float3 homogenousCoord = float3(currentVertexInLasFrame.x / currentVertexInLasFrame.z, currentVertexInLasFrame.y / currentVertexInLasFrame.z, 1.0);
    float3 pixelPos = float3(mul(cameraProjection, homogenousCoord).xy, 0);
    float3 v_hat = lastVertexMap.Load((int3)floor(pixelPos));
    float3 n_hat = lastNormalMap.Load((int3)floor(pixelPos));

    float3 transformedNormal = mul(lastFramePose , float4(n,0)).xyz;

    if (length(v_hat - v) > distanceThreshold || dot(normalize(n), normalize(n_hat)) > angleThreshold) {
        return;
    }

    float3x3 skewSymmetricCurrentV = float3x3(0, -v.z, v.y,
                                              v.z, 0, -v.x,
                                              -v.y, v.x, 0);

    float3 at_0 = mul(skewSymmetricCurrentV, n_hat);
    float at_array[6] = { at_0.x, at_0.y, at_0.z, n_hat.x, n_hat.y, n_hat.z };

    [unroll]
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            ata[index][i * 6 + j] = at_array[i] * at_array[j];
        }
    }
    

    // TODO: check this matrix( 6?)
    matrix<float, 6, 1> at;
    
    float b = dot(n_hat, v_hat - v);

    [unroll]
    for (int i = 0; i < 6; i++) {
        atb[index][i] = b * at[i];
    }

    
    // float3 transformedNormal = mul(float4(v.normal, 0), pose).xyz;

    // float distance = length(transformedVertex - u_hat.vertex);
    // float angle = dot(normalize(transformedNormal), normalize(u_hat.normal));

    //Check if omega is not nulll
    // if (distance < distanceThreshold && angle > angleThreshold) {

    //     // populate AtA and AtB
    // }





}