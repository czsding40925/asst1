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

public Optional<RayHitResult> sample_triangle(Ray ray, RendererUniform uniforms){
    
    Optional<RayHitResult> bestHit_triangle;
    bool hitAnything_triangle = false;
    float bestT_triangle = float.maxValue;
    if (uniforms.useBVH){
        BVH<Triangle> bvh = uniforms.bvh;
        Optional<bvh.HitResult> bvh_hit = bvh.hit(ray);
        if (bvh_hit.hasValue){
            BVH<Triangle>.HitResult hr = bvh_hit.value;
            bestHit_triangle = hr.rayHitResult;
        }
    }
    else{
        StructuredBuffer<Triangle> triangles = uniforms.triangleBuf;
        for (uint i = 0; i < uniforms.triangleCount; i++){
       
            Triangle tri = triangles[i];
            float3 v0 = tri.vertices[0];
            float3 v1 = tri.vertices[1];
            float3 v2 = tri.vertices[2];

            // plane intersection
            float3 e1 = v1 - v0;
            float3 e2 = v2 - v0;
            float3 n  = normalize(cross(e1, e2)); 
            float denom = dot(n, ray.direction);
            if (abs(denom) < 1e-12) continue;

            float t = dot(n, (v0 - ray.origin)) / denom;
            if (t <= T_MIN || t >= bestT_triangle) continue;

            float3 p = ray.origin + t * ray.direction;

            // inside test (same-side)
            float3 c0 = cross(v1 - v0, p - v0);
            float3 c1 = cross(v2 - v1, p - v1);
            float3 c2 = cross(v0 - v2, p - v2);

            float d0 = dot(c0, n);
            float d1 = dot(c1, n);
            float d2 = dot(c2, n);

            bool inside = (d0 >= 0 && d1 >= 0 && d2 >= 0) || (d0 <= 0 && d1 <= 0 && d2 <= 0);
            if (!inside) continue;
            
            RayHitResult h;
            h.t = t;
            h.normal = n; 
            bestT_triangle = t;
            bestHit_triangle = h; 
            hitAnything_triangle = true;
        } 
    }
    return bestHit_triangle;
}

public Optional<RayHitResult> sample_volume(Ray ray, RendererUniform uniforms){
    float bestT_volume = float.maxValue; 

    StructuredBuffer<Volume> volumes = uniforms.volumeBuf;
    Optional<RayHitResult> bestHit_volume;
    bool hitAnything_volume = false;
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
    return bestHit_volume;
}

public Optional<RayHitResult> sample_sdf(Ray ray, RendererUniform uniforms){
    float bestT_sdf = float.maxValue; 
    Optional<RayHitResult> BestHit_sdf;
    bool hitAnything_sdf = false;
    float t = 0.0;

    for (uint step = 0; step < MAX_STEPS; step++)
    {
        if (t >= min(bestT_sdf, T_MAX)) break;  // don't march past already-found hit

        float3 p = ray.origin + t * ray.direction;
        Tuple<float, float3> dn = uniforms.sdfBuf.sample(p);
        float dist = dn._0;

        if (dist < EPSILON)
        {
            if (t > T_MIN && t < bestT_sdf)
            {
                RayHitResult h;
                bestT_sdf = t;
                h.t = t;
                h.normal = normalize(dn._1);
                hitAnything_sdf = true;
            }
            break;
        }

        t += max(dist, 1e-4);
    }
    return BestHit_sdf;
}
/**
 * Sample the scene at the given uv coordinate.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @return The color of the ray sample at the given uv coordinate.
 */
public float4 sample(float2 uv, RendererUniform uniforms)
{
    Ray ray = uniforms.camera.generateRay(uv);
    
    // Optional<RayHitResult> triangle_hit = sample_triangle(ray, uniforms);
    // Optional<RayHitResult> volume_hit = sample_volume(ray, uniforms);
    // Optional<RayHitResult> sdf_hit = sample_sdf(ray, uniforms);

    // float3 color;

    // if (triangle_hit.hasValue){
    //     if (uniforms.renderDepth)
    //         return float4(float3(triangle_hit.value.t), 1.0);

    //     if (uniforms.renderNormal)
    //         return float4(triangle_hit.value.normal * 0.5 + 0.5, 1.0);  
    //     SimpleMaterial simpleMaterial;
    //     color = simpleMaterial.shade(ray, triangle_hit.value);
    //     return float4(color, 1.0);
    // }
    // else{
    //     return uniforms.ambientColor;
    // }

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
        // float2 rootBox = bvh.nodes[0].hit(ray);
        // if (rootBox.x > rootBox.y) return float4(1,0,1,1); // magenta = ray misses root box

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
            float3 v0 = tri.vertices[0];
            float3 v1 = tri.vertices[1];
            float3 v2 = tri.vertices[2];

            // plane intersection
            float3 e1 = v1 - v0;
            float3 e2 = v2 - v0;
            float3 n  = normalize(cross(e1, e2)); 
            float denom = dot(n, ray.direction);
            if (abs(denom) < 1e-12) continue;

            float t = dot(n, (v0 - ray.origin)) / denom;
            if (t <= T_MIN || t >= bestT_triangle) continue;

            float3 p = ray.origin + t * ray.direction;

            // inside test (same-side)
            float3 c0 = cross(v1 - v0, p - v0);
            float3 c1 = cross(v2 - v1, p - v1);
            float3 c2 = cross(v0 - v2, p - v2);

            float d0 = dot(c0, n);
            float d1 = dot(c1, n);
            float d2 = dot(c2, n);

            bool inside = (d0 >= 0 && d1 >= 0 && d2 >= 0) || (d0 <= 0 && d1 <= 0 && d2 <= 0);
            if (!inside) continue;

            bestHit_triangle.t = t;
            bestHit_triangle.normal = n; 
            bestT_triangle = t;
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
        // return float4(normalize(bestHit_triangle.normal) * 0.5 + 0.5, 1.0);
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
    

    

    // //---------- SDF ----------
    {
        float t = 0.0;
        for (uint step = 0; step < MAX_STEPS; step++)
        {
            if (t >= min(bestT_sdf, T_MAX)) break;  // don't march past already-found hit

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
