
#define GROUPDIMENSION 16

struct VertexNormal {
    float3 vertex;
    float3 normal;
};

// Input buffers for the current and predicted frames
cbuffer FrameData : register(b0) {
    float4x4 pose;
    float4x4 lastFramePose;
    float distanceThreshold;      // Distance threshold for correspondence
    float angleThreshold;         // Cosine of the angle threshold
};

StructuredBuffer<VertexNormal> currentFrame : register(t0);   // Input current frame
StructuredBuffer<VertexNormal> predictedFrame : register(t1); // Input predicted frame
RWStructuredBuffer<float> residuals : register(u0);           // Output residuals for optimization


[shader("compute")]
[numthreads(GROUPDIMENSION, 1, 1)]
void ICPCalcMain(uint3 threadId: SV_DispatchThreadID) {
    uint index = threadId.x;
    if (index >= currentFrame.getCount() || index >= predictedFrame.getCount()) {
        return;
    }

    VertexNormal v = currentFrame[index];
    // VertexNormal u_hat = predictedFrame[index];
    //compute v_hat calculate t_(k-1,k). bring in to camera image space. apply perspective. we now have u_hat. check the value of u_hat from the past vertex map




    float3 transformedVertex = mul(float4(v.vertex, 1.0), pose).xyz;
    float3 transformedNormal = mul(float4(v.normal, 0), pose).xyz;

    // float distance = length(transformedVertex - u_hat.vertex);
    // float angle = dot(normalize(transformedNormal), normalize(u_hat.normal));

    //Check if omega is not nulll
    // if (distance < distanceThreshold && angle > angleThreshold) {

    //     // populate AtA and AtB
    // }





}