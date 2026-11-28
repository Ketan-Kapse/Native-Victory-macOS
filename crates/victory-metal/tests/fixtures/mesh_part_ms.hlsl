// One pass of a mesh program drawn in two passes: the left half of the target, or with RIGHT
// defined (mesh_part_right_ms.dxil: this file with `#define RIGHT` before it) the right half.
// No resources, root reads or user varyings. No game code or data.
struct Vertex { float4 position : SV_Position; };
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void main(uint lane : SV_GroupIndex,
          out vertices Vertex verts[4], out indices uint3 tris[2]) {
    SetMeshOutputCounts(4, 2);
#ifdef RIGHT
    float left = 0, right = 1;
#else
    float left = -1, right = 0;
#endif
    if (lane < 4) verts[lane].position = float4((lane & 1) ? right : left, (lane & 2) ? 1.0 : -1.0, 0.5, 1);
    if (lane == 0) tris[0] = uint3(0, 1, 2);
    if (lane == 1) tris[1] = uint3(1, 3, 2);
}
