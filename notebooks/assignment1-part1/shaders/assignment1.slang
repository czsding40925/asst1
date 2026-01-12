import math;
import model;
import primitive;

/**
 * Fill a solid red color.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @return The color red as a float4 vector.
 */
public float4 fillRed(uint2 tid)
{
    // Return the color as a float4 vector (red, green, blue, alpha).
    // In this example, we set alpha to 1 to indicate a fully opaque pixel.
    // We'll cover alpha, transparency later in the class.
    return float4(1.0, 0.0, 0.0, 1.0);
}

/**
 * Fill a uv gradient pattern.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @return The color corresponding to the uv(xy) coordinate of the pixel.
 */
public float4 fillGradient(uint2 tid, uint2 canvasSize)
{
    // TODO: Student implementation starts here.
    float red_value = float(tid.x) / float(canvasSize.x - 1);
    float green_value = float(tid.y) / float(canvasSize.y - 1);

    return float4(red_value, green_value , 0.0, 1.0);

    // TODO: Student implementation ends here.
}

/**
 * Fill a checkerboard pattern.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param numSquares The number of squares along each axis.
 * @return The color white or black depending on the checkerboard pattern.
 */
public float4 fillCheckerboard(uint2 tid, uint2 canvasSize, uint numSquares)
{
    // TODO: Student implementation starts here.

    
    float norm_x = float(tid.x) / float(canvasSize.x - 1);
    float norm_y = float(tid.y) / float(canvasSize.y - 1);

    int bin_x = int(norm_x * numSquares);
    int bin_y = int(norm_y * numSquares); 

    if (bin_x % 2 != bin_y % 2){
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    else{
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    

    // TODO: Student implementation ends here.
}

/**
 * Rasterize a single triangle.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param v0 The first vertex of the triangle in normalized coordinates.
 * @param v1 The second vertex of the triangle in normalized coordinates.
 * @param v2 The third vertex of the triangle in normalized coordinates.
 * @return The color white if the pixel is inside the triangle, otherwise black.
 */
public float4 triangleRasterizer(uint2 tid, uint2 canvasSize, float2 v0, float2 v1, float2 v2)
{
    // TODO: Student implementation starts here.

    float norm_x = float(tid.x) / float(canvasSize.x - 1);
    float norm_y = float(tid.y) / float(canvasSize.y - 1);

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

    if (L_0 <= 0 && L_1 <= 0 && L_2 <= 0){  
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    else{
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    

    // TODO: Student implementation ends here.
}

/**
 * Test ray-triangle intersection for a single triangle.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param triangles A buffer containing triangles to test against. Only the first triangle is used in this function.
 * @return The color of the triangle if hit, otherwise black.
 */
public float4 rayTriangleTest(uint2 tid, uint2 canvasSize, StructuredBuffer<Triangle> triangles)
{
    // Get the first triangle in the buffer.
    Triangle triangle = triangles[0];
    // We generate rays with origin from (0, 0, 0) to (1, 1, 0) and direction towards the negative z axis (0, 0, -1).
    float2 uv = float2(tid) / float2(canvasSize);
    Ray ray = Ray(float3(uv, 0.0), float3(0.0, 0.0, -1.0), float2(0.0, float.maxValue));
    // How can we make sure the rays are set up correctly? Just return the value as color.
    // return float4(ray.origin.xy, 0.0, 1.0);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    if (hitResult.hasValue)
    {
        return float4(triangle.averageColor(), 1.0);
    }
    else
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
}

/**
 * Test ray-triangle intersection for multiple triangles.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param triangles A buffer containing triangles to test against.
 * @param triangleCount The number of triangles in the buffer.
 * @return The color of the closest triangle if hit, otherwise black.
 */
public float4 rayTrianglesTest(uint2 tid, uint2 canvasSize, StructuredBuffer<Triangle> triangles, uint triangleCount)
{
    // We generate rays with origin from (0, 0, 0) to (1, 1, 0) and direction towards the negative z axis (0, 0, -1).
    float2 uv = float2(tid) / float2(canvasSize);
    Ray ray = Ray(float3(uv, 0.0), float3(0.0, 0.0, -1.0), float2(0.0, float.maxValue));

    // TODO: Student implementation starts here.

    return float4(1.0, 0.0, 1.0, 1.0);

    // TODO: Student implementation starts here.
}
