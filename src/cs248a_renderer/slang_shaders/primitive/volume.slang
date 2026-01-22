implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.005;
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {
        // TODO: Student implementation starts here.
        // Ray bounding box intersection (bound) 
        float3 origin = ray.origin; 
        float3 direction = ray.direction; 
        float t = 0;
        for (int i = 0; i < 10000; i++){
            float3 curr_ray_point = origin + t * direction; 
            float3 p_uvw = getUVW(curr_ray_point); 
            float4 sampled_point = volumeTexBuf.trilinearSample(tex, p_uvw);
            if (sampled_point.w >= DENSITY_THRESHOLD){
                RayHitResult hit; 
                hit.t = t; 
                // compute normal
                float du = 1.0 /tex.size.x; 
                float dv = 1.0 /tex.size.y;
                float dw = 1.0 /tex.size.z; 

                float4 gx = volumeTexBuf.trilinearSample(tex, p_uvw + float3(du, 0.0, 0.0)) - volumeTexBuf.trilinearSample(tex, p_uvw - float3(du, 0.0, 0.0));
                float4 gy = volumeTexBuf.trilinearSample(tex, p_uvw + float3(0.0, dv, 0.0)) - volumeTexBuf.trilinearSample(tex, p_uvw - float3(0.0, dv, 0.0));
                float4 gz = volumeTexBuf.trilinearSample(tex, p_uvw + float3(0.0, 0.0, dw)) - volumeTexBuf.trilinearSample(tex, p_uvw - float3(0.0, 0.0, dw));

                float3 n = -normalize(float3(gx.w, gy.w, gz.w));
                hit.normal = n;
                
                return Optional<RayHitResult>(hit);
            }
            t += STEP_SIZE; 
        }
        // TODO: Student implementation ends here.

        return Optional<RayHitResult>();
    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in world space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }

    /** 
    * Helper function to compute the surface normals 
    * @param p The point in the world space 
    * @return the surface normal 
    */
    public float3 getNormal(float3 p){
        float du = 1/tex.size.x; 
        float dv = 1/tex.size.y;
        float dw = 1/tex.size.z; 
        float3 p_uvw = getUVW(p); 

        return float3(0.0); 
    }
};
