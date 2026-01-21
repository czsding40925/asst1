implementing "../primitive.slang";
import math;

public struct Triangle : IRayCastable
{
    public float3[3] vertices;
    public float3[3] normals;
    public float2[3] uvs;
    public float3[3] colors;
    public uint materialId;

    /* Returns the average color of the triangle. */
    public float3 averageColor()
    {
        return (this.colors[0] + this.colors[1] + this.colors[2]) / 3.0;
    }

    /**
     * Test ray-triangle intersection.
     * @param ray The ray to test against the triangle.
     * @return An Optional containing RayHitResult if the ray hits the triangle, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray)
    {
        // float3 e1 = vertices[2] - vertices[0]; 
        // float3 e2 = vertices[1] - vertices[0]; 
        // float3 n = normalize(cross(e1, e2)); 

        // float3 ds[3]; 
        // for (int i = 0; i < 3; i ++){
        //     ds[i] = -dot(n, vertices[i]); 
        // }



        // TODO: Student implementation starts here.
        float2 v0 = float2(vertices[0].x, vertices[0].y);
        float2 v1 = float2(vertices[1].x, vertices[1].y);
        float2 v2 = float2(vertices[2].x, vertices[2].y);

        float norm_x = ray.origin.x;
        float norm_y = ray.origin.y;

        float A_0 = v1.y - v0.y; 
        float A_1 = v2.y - v1.y; 
        float A_2 = v0.y - v2.y; 

        float B_0 = v0.x - v1.x; 
        float B_1 = v1.x - v2.x; 
        float B_2 = v2.x - v0.x; 

        float C_0 = v0.y*(v1.x - v0.x) - v0.x*(v1.y - v0.y);
        float C_1 = v1.y*(v2.x - v1.x) - v1.x*(v2.y - v1.y);
        float C_2 = v2.y*(v0.x - v2.x) - v2.x*(v0.y - v2.y);

        float L_0 = A_0 * norm_x + B_0 * norm_y + C_0; 
        float L_1 = A_1 * norm_x + B_1 * norm_y + C_1; 
        float L_2 = A_2 * norm_x + B_2 * norm_y + C_2; 
 
        RayHitResult outHit;
        outHit.t = vertices[0].z; 
        float3 n = float3(0.0, 0.0, 1.0);
        outHit.normal = n; 

        if (L_0 <= 0 && L_1 <= 0 && L_2 <= 0){  
            return Optional<RayHitResult>(outHit);
        }
        else{
            return Optional<RayHitResult>();
        }
        return Optional<RayHitResult>();

        // TODO: Student implementation ends here.
    }
}
