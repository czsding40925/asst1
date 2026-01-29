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
        float3 v0 = vertices[0];
        float3 v1 = vertices[1];
        float3 v2 = vertices[2];

        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;

        float3 pvec = cross(ray.direction, e2);
        float det = dot(e1, pvec);

        const float eps = 1e-8;
        if (abs(det) < eps)
            return none;

        float invDet = 1.0 / det;

        float3 tvec = ray.origin - v0;
        float u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0)
            return none;

        float3 qvec = cross(tvec, e1);
        float v = dot(ray.direction, qvec) * invDet;
        if (v < 0.0 || (u + v) > 1.0)
            return none;

        float t = dot(e2, qvec) * invDet;
        if (t < ray.tRange.x || t > ray.tRange.y)
            return none;

        RayHitResult outHit;
        outHit.t = t;

        float3 n = normalize(cross(e1, e2));

        if (dot(n, ray.direction) > 0.0)
            n = -n;

        outHit.normal = n;

        return outHit; // implicit Optional wrapping
    }
}
