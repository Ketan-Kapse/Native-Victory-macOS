// Synthetic GFX10 programs, shared by the lowering tests and coordinator GPU tests.
// Each body leaves its result in v2 and preserves the output pointer s[2:3].
pub const READ_LANE: &[u32] = &[
    0xBE89_03FF,
    77,          // s9 = 77
    0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
    0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    0xBF88_0002, // s_cbranch_execz over readlane
    0xD760_0009,
    0x0001_5100, // s9 = lane 40's v0, even with that lane off
    0x88FE_0A7E, // restore EXEC
    0x7E04_0209, // v2 = s9 (scalar join, including the inactive half)
];
pub const FIRST_LANE: &[u32] = &[
    0xBE89_03FF,
    77,
    0xBE8C_0380, // s12 = 0
    0x7D82_00B2, // v_cmp_lt_u32 vcc, 50, v0 (lanes 51 and up)
    0xBE8A_246A,
    0xBF88_0004,
    0xD760_000E,
    0x0001_1100, // s14 = lane 8 (keeps the fallback required)
    0xBE8C_107E, // s12 = population of the complete EXEC
    0x7E12_0500, // v_readfirstlane_b32 s9, v0
    0x88FE_0A7E,
    0x8009_0C09, // s9 += s12 (keeps the cross-half count live)
    0x8009_0E09, // s9 += s14
    0x7E04_0209,
];
pub const MASK_SCAN: &[u32] = &[
    0xBE89_03FF,
    77,
    0x7D82_00B2, // v0 > 50, spanning halves only in the first console wave
    0xBE8A_246A,
    0xBF88_0006,
    0xD760_000E,
    0x0001_1100,
    0xBE89_147E, // s_ff1_i32_b64 s9, exec (51 in wave 0, 0 in wave 1)
    0xBE8C_107E, // s_bcnt1_i32_b64 s12, exec (13 in wave 0, 64 in wave 1)
    0x8009_0C09, // s_add_u32 s9, s9, s12
    0x8009_0E09, // s9 += s14
    0x88FE_0A7E,
    0x7E04_0209,
];
pub const LOOP: &[u32] = &[
    0xBE89_0380,
    0x7E04_0280, // v2 = 0
    0x7D8A_0080, // vcc = (v0 != 0)
    0xBE8A_246A,
    0xBF88_0006,
    0xD760_0009,
    0x0001_1100,
    0x4A04_0481, // ++v2
    0x7D82_0102, // vcc = (v2 < v0)
    0x87FE_6A7E,
    0xBF89_FFFA,
    0x88FE_0A7E,
    0x4A04_0409, // v2 += s9
];

pub const MASK_JOIN: &[u32] = &[
    0xBE89_03FF,
    77,
    0xBE8C_0480, // s[12:13] = 0 on the skip edge
    0x7D82_00B2, // v0 > 50
    0xBE8A_246A,
    0xBF88_0003,
    0xD760_0009,
    0x0001_1100,
    0xBE8C_047E, // s[12:13] = exec on the taken edge
    0x88FE_0A7E,
    0xBE8E_100C, // s14 = popcount of the joined 64-bit mask
    0x8009_0E09,
    0x7E04_0209,
];
