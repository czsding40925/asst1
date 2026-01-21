import math;
import texture;
import material;
import model;
import primitive;

public struct RendererUniform
{
    // Camera parameters.
    public Camera camera;
    // Ambient color, used when no ray hits a primitive
    public float4 ambientColor;
    // Ray samples per pixel
    public uint sqrtSpp;
    // Primitives.
    public StructuredBuffer<Triangle> triangleBuf;
    public uint triangleCount;
    public SharedTexture3DBuffer<float4> volumeTexBuf;
    public StructuredBuffer<Volume> volumeBuf;
    public uint volumeCount;
    public SDFBuffer sdfBuf;
    // Acceleration structure.
    public bool useBVH;
    public BVH<Triangle> bvh;

    public bool renderDepth;
    public bool renderNormal;
}

static uint MAX_STEPS = 100;
static float EPSILON = 1e-6;

/**
 * Sample the scene at the given uv coordinate.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @return The color of the ray sample at the given uv coordinate.
 */
public float4 sample(float2 uv, RendererUniform uniforms)
{
    // We generate a camera ray for the given uv coordinate.
    // You need to implement the `generateRay` function in `Camera` struct in `src/cs248a_renderer/slang_shaders/model/camera.slang`
    Ray ray = uniforms.camera.generateRay(uv);
    RayHitResult result;
    bool hitAnything = false;

    // TODO: Student implementation starts here.
    RayHitResult curr_hit; 
    float curr_smallest_t = float.maxValue; 
    StructuredBuffer<Triangle> triangles = uniforms.triangleBuf;
    for (int i = 0; i < uniforms.triangleCount; i++){
        // Comput t 
        Triangle triangle = triangles[i]; 
        float3 v0 = triangle.vertices[0]; 
        float3 v1 = triangle.vertices[1]; 
        float3 v2 = triangle.vertices[2]; 
        float3 e1 = v2 - v0; 
        float3 e2 = v1 - v0; 
        float3 n = normalize(cross(e2, e1)); 
        float d = -dot(n, triangle.vertices[0]); 
        if (abs(dot(n, ray.direction)) <= 1e-12){
            continue; 
        }
        float t = -(dot(n, ray.origin) + d) / dot(n, ray.direction); 
        if (t <= 0.0) continue;

        // Determine whether that ray is in the triangle
        float3 ray_endpoint = ray.origin + t * ray.direction; 
        
        float3 n0 = cross(v1 - v0, ray_endpoint - v0); 
        float3 n1 = cross(v2 - v1, ray_endpoint - v1); 
        float3 n2 = cross(v0 - v2, ray_endpoint - v2); 

        
        float d0 = dot(n0, n); 
        float d1 = dot(n1, n); 
        float d2 = dot(n2, n); 
        bool all_pos = d0 > 0 && d1 > 0 && d2 > 0; 
        bool all_neg = d0 < 0 && d1 < 0 && d2 < 0; 

        if ((all_pos || all_neg) && t < curr_smallest_t)
        {
            curr_smallest_t = t;
            curr_hit.t = t;
            curr_hit.normal = n;
            hitAnything = true;
        }else{
            continue; 
        }
    }
    if (!hitAnything)
        return uniforms.ambientColor;
    result = curr_hit; 
    // TODO: Student implementation ends here.

    if (uniforms.renderDepth)
    {
        // Map depth to [0, 1] range for visualization
        return float4(float3(result.t), 1.0);
    }

    if (uniforms.renderNormal)
    {
        // Map normal from [-1, 1] to [0, 1] range for visualization
        return float4(result.normal * 0.5 + 0.5, 1.0);
    }

    // Once you have the intersection result, use the following code to shade (calculate color) for the ray.
    // The shading function needs the camera ray and the intersection result.
    SimpleMaterial simpleMaterial;
    float3 color = simpleMaterial.shade(ray, result);
    return float4(color, 1.0);
}

/**
 * The entry point of the renderer.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @return The color of the pixel at the given thread coordinate.
 */
public float4 render(uint2 tid, RendererUniform uniforms)
{
    // TODO: Student implementation starts here.

    // Basic version: no supersampling (remove this implementation once you implement supersampling)
    // float2 uv = (float2(tid) + 0.5) / float2(uniforms.camera.canvasSize);
    // return sample(uv, uniforms);
    float4 final_color = float4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < uniforms.sqrtSpp; i++){
        for (int j = 0; j < uniforms.sqrtSpp; j++){
            float2 uv = float2(tid.x + float(i) / uniforms.sqrtSpp, tid.y + float(j) / uniforms.sqrtSpp) / float2(uniforms.camera.canvasSize);
            final_color += sample(uv, uniforms);
        }
    }
    return final_color / float(uniforms.sqrtSpp * uniforms.sqrtSpp);
    // TODO: Student implementation ends here.
}
