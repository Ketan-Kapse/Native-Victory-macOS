// Mesh outputs over Metal's limit: 128 vertices of 296 bytes (the position, fifteen float4, then
// float3 float3 float float in this order: 12 + 12 + 4 + 4 after 8 bytes of padding), 37,888 bytes
// a thread group. Metal counts the outputs in this declared order, not in the order of the
// registers the compiler packs them into (float3 float, float3 float: 304 bytes).
// A full-screen triangle whose vertices carry fifteen float4 outputs with known values. No game
// code or data.
struct Vertex {
    float4 position : SV_Position;
    float4 a0 : TEXCOORD0;
    float4 a1 : TEXCOORD1;
    float4 a2 : TEXCOORD2;
    float4 a3 : TEXCOORD3;
    float4 a4 : TEXCOORD4;
    float4 a5 : TEXCOORD5;
    float4 a6 : TEXCOORD6;
    float4 a7 : TEXCOORD7;
    float4 a8 : TEXCOORD8;
    float4 a9 : TEXCOORD9;
    float4 a10 : TEXCOORD10;
    float4 a11 : TEXCOORD11;
    float4 a12 : TEXCOORD12;
    float4 a13 : TEXCOORD13;
    float4 a14 : TEXCOORD14;
    float3 b0 : COLOR0;
    float3 b1 : COLOR1;
    float b2 : COLOR2;
    float b3 : COLOR3;
};
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void main(uint lane : SV_GroupIndex,
          out vertices Vertex verts[128], out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);
    if (lane < 3) {
        float2 positions[3] = { float2(-1,-1), float2(3,-1), float2(-1,3) };
        Vertex v;
        v.position = float4(positions[lane], 0.5, 1);
        v.a0 = float4(0, 0 + 0.25, 0 + 0.5, 0 + 0.75);
        v.a1 = float4(1, 1 + 0.25, 1 + 0.5, 1 + 0.75);
        v.a2 = float4(2, 2 + 0.25, 2 + 0.5, 2 + 0.75);
        v.a3 = float4(3, 3 + 0.25, 3 + 0.5, 3 + 0.75);
        v.a4 = float4(4, 4 + 0.25, 4 + 0.5, 4 + 0.75);
        v.a5 = float4(5, 5 + 0.25, 5 + 0.5, 5 + 0.75);
        v.a6 = float4(6, 6 + 0.25, 6 + 0.5, 6 + 0.75);
        v.a7 = float4(7, 7 + 0.25, 7 + 0.5, 7 + 0.75);
        v.a8 = float4(8, 8 + 0.25, 8 + 0.5, 8 + 0.75);
        v.a9 = float4(9, 9 + 0.25, 9 + 0.5, 9 + 0.75);
        v.a10 = float4(10, 10 + 0.25, 10 + 0.5, 10 + 0.75);
        v.a11 = float4(11, 11 + 0.25, 11 + 0.5, 11 + 0.75);
        v.a12 = float4(12, 12 + 0.25, 12 + 0.5, 12 + 0.75);
        v.a13 = float4(13, 13 + 0.25, 13 + 0.5, 13 + 0.75);
        v.a14 = float4(14, 14 + 0.25, 14 + 0.5, 14 + 0.75);
        v.b0 = 0;
        v.b1 = 1;
        v.b2 = 2;
        v.b3 = 3;
        verts[lane] = v;
    }
    if (lane == 0) tris[0] = uint3(0,1,2);
}
