implementing "../texture.slang";

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        uint3 coords; 
        coords.x = min(uint(floor(uvw.x * tex.size.x + 0.5)), tex.size.x - 1);
        coords.y = min(uint(floor(uvw.y * tex.size.y + 0.5)), tex.size.y - 1);
        coords.z = min(uint(floor(uvw.z * tex.size.z + 0.5)), tex.size.z - 1);

        return getBufferValue(tex, coords);

        // TODO: Student implementation ends here.
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
    **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        float3 coords; 

        uint sx = tex.size.x;
        uint sy = tex.size.y;
        uint sz = tex.size.z;

        coords.x = uvw.x * (sx - 1);
        coords.y = uvw.y * (sy - 1);
        coords.z = uvw.z * (sz - 1);
        
        // uint x_0 = uint(floor(coords.x));
        // uint y_0 = uint(floor(coords.y));
        // uint z_0 = uint(floor(coords.z));

        // uint x_1 = x_0 + 1;
        // uint y_1 = y_0 + 1;
        // uint z_1 = z_0 + 1;

        // Address out of bounds error 
        uint x_0 = min(uint(floor(coords.x)), sx - 1);
        uint y_0 = min(uint(floor(coords.y)), sy - 1);
        uint z_0 = min(uint(floor(coords.z)), sz - 1);

        uint x_1 = min(x_0 + 1, sx - 1);
        uint y_1 = min(y_0 + 1, sy - 1);
        uint z_1 = min(z_0 + 1, sz - 1);

        float f_x = coords.x - float(x_0);
        float f_y = coords.y - float(y_0);
        float f_z = coords.z - float(z_0);


        // Cubes 
        T Cube_000 = getBufferValue(tex, uint3(x_0, y_0, z_0)); 
        T Cube_100 = getBufferValue(tex, uint3(x_1, y_0, z_0)); 
        T Cube_010 = getBufferValue(tex, uint3(x_0, y_1, z_0)); 
        T Cube_011 = getBufferValue(tex, uint3(x_0, y_1, z_1)); 
        T Cube_001 = getBufferValue(tex, uint3(x_0, y_0, z_1)); 
        T Cube_101 = getBufferValue(tex, uint3(x_1, y_0, z_1)); 
        T Cube_110 = getBufferValue(tex, uint3(x_1, y_1, z_0)); 
        T Cube_111 = getBufferValue(tex, uint3(x_1, y_1, z_1)); 

        // Along x 
        T c00 = Cube_000 * T((1 - f_x)) + Cube_100 * T(f_x); 
        T c01 = Cube_001 * T((1 - f_x)) + Cube_101 * T(f_x);
        T c10 = Cube_010 * T((1 - f_x)) + Cube_110 * T(f_x); 
        T c11 = Cube_011 * T((1 - f_x)) + Cube_111 * T(f_x);
        

        // Along y
        T c0 = c00 * T(1 - f_y) + c10 * T(f_y); 
        T c1 = c01 * T(1 - f_y) + c11 * T(f_y); 

        // Along z 
        T final = c0 * T(1 - f_z) + c1 * T(f_z); 
 
        //return T(final);
        return final;

        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}

       