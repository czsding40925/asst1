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

// static uint MAX_STEPS = 100;
// static float EPSILON = 1e-6;
static uint MAX_STEPS = 256;
static float EPSILON = 1e-6;
static float T_MIN = 1e-4;          // small positive
static float T_MAX = 1e6;           // or scene-dependent

/**
 * Sample the scene at the given uv coordinate.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @return The color of the ray sample at the given uv coordinate.
 */
public float4 sample(float2 uv, RendererUniform uniforms)
{
    Ray ray = uniforms.camera.generateRay(uv);
    
    bool hitAnything_triangle = false;
    bool hitAnything_volume = false; 
    bool hitAnything_sdf = false; 
    SimpleMaterial simpleMaterial;
    float3 color; //= simpleMaterial.shade(ray, bestHit);
    float bestT_triangle = float.maxValue;
    float bestT_volume = float.maxValue; 
    float bestT_sdf = float.maxValue; 
    RayHitResult bestHit_triangle;
    RayHitResult bestHit_volume;
    RayHitResult BestHit_sdf;

    // ---------- TRIANGLES ----------
     if (uniforms.useBVH){
        BVH<Triangle> bvh = uniforms.bvh;

        Optional<bvh.HitResult> bvh_hit = bvh.hit(ray);
        if (bvh_hit.hasValue){
            hitAnything_triangle = true;
            BVH<Triangle>.HitResult hr = bvh_hit.value;
            bestHit_triangle = hr.rayHitResult;
        }
    }
    else{
        StructuredBuffer<Triangle> triangles = uniforms.triangleBuf;
        for (uint i = 0; i < uniforms.triangleCount; i++){
       
            Triangle tri = triangles[i];
            Optional<RayHitResult> tri_hit = tri.hit(ray);
            if (!tri_hit.hasValue) continue;

            RayHitResult h = tri_hit.value; 
            if (h.t >= bestT_triangle) continue;

            bestHit_triangle.t = h.t;
            bestHit_triangle.normal = h.normal; 
            bestT_triangle = h.t;
            hitAnything_triangle = true;
        }
    }
    if (hitAnything_triangle){
        if (uniforms.renderDepth)
            return float4(float3(bestHit_triangle.t), 1.0);

        if (uniforms.renderNormal)
            return float4(bestHit_triangle.normal * 0.5 + 0.5, 1.0);  
        color = simpleMaterial.shade(ray, bestHit_triangle);
        return float4(color, 1.0);   
    }
    
    // ---------- VOLUMES ----------
    {
        StructuredBuffer<Volume> volumes = uniforms.volumeBuf;
        for (uint i = 0; i < uniforms.volumeCount; i++)
        {
            Volume vol = volumes[i];
            Optional<RayHitResult> ohit = vol.hit(ray, uniforms.volumeTexBuf);
            if (!ohit.hasValue) continue;

            RayHitResult h = ohit.value;
            if (h.t <= T_MIN || h.t >= bestT_volume) continue;

            bestT_volume = h.t;
            bestHit_volume = h;
            hitAnything_volume = true;
        }
        if (hitAnything_volume){
            if (uniforms.renderDepth)
                return float4(float3(bestHit_volume.t), 1.0);

            if (uniforms.renderNormal)
                return float4(bestHit_volume.normal * 0.5 + 0.5, 1.0);  
            color = simpleMaterial.shade(ray, bestHit_volume);
            return float4(color, 1.0);
            
        }
    }
    

     //---------- SDF ----------
    {
        float t = 0.0;
        for (uint step = 0; step < MAX_STEPS; step++)
        {
            if (t >= min(bestT_sdf, T_MAX)) break;  

            float3 p = ray.origin + t * ray.direction;
            Tuple<float, float3> dn = uniforms.sdfBuf.sample(p);
            float dist = dn._0;

            if (dist < EPSILON)
            {
                if (t > T_MIN && t < bestT_sdf)
                {
                    bestT_sdf = t;
                    BestHit_sdf.t = t;
                    BestHit_sdf.normal = normalize(dn._1);
                    hitAnything_sdf = true;
                }
                break;
            }

            t += max(dist, 1e-4);
        }
    if (hitAnything_sdf){
        if (uniforms.renderDepth)
            return float4(float3(BestHit_sdf.t), 1.0);

        if (uniforms.renderNormal)
            return float4(BestHit_sdf.normal * 0.5 + 0.5, 1.0);  

        color = simpleMaterial.shade(ray, BestHit_sdf);
        return float4(color, 1.0);
        
    }
}    
    
    
    return uniforms.ambientColor;

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
