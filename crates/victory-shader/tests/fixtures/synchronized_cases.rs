// Hand-assembled synthetic GFX10. No title words. v2 is the output; s[2:3]
// remains available to the GPU fixture. Each wave enables only its upper half.
fn vop2(op: u32, dst: u32, src: u32, vector: u32) -> u32 {
    op << 25 | dst << 17 | vector << 9 | src
}

fn prepare(first: u32, second: u32) -> Vec<u32> {
    vec![
        vop2(0x16, 1, 0x86, 0), // v1 = thread id >> 6 (console wave id)
        0x7D84_0280,            // vcc = (v1 == 0)
        0x7E08_02FF,
        first, // v4 = first
        vop2(1, 1, 0xff, 4),
        second,                 // v1 = wave 0 ? first : second
        0x7E04_0280,            // v2 = 0
        0xBE89_0380,            // s9 = 0
        vop2(0x1b, 3, 0xbf, 0), // v3 = console lane
        0x7D82_069F,            // vcc = (31 < v3)
        vop2(1, 1, 0xa5, 1),    // inactive lower half holds 37, not the bound/mask
        0xBE8A_246A,            // save EXEC, enable upper half only
        0x7E18_0501,            // s12 = first active lane's v1 (upper half)
    ]
}

pub fn nested(first: u32, second: u32) -> Vec<u32> {
    let mut w = prepare(first, second);
    w.extend([
        0xBF0A_0C09,
        0xBF84_000B,
        0xBE90_0380, // inner count = 0
        0x8011_8109, // inner bound = outer count + 1
        0xD760_000E,
        0x0001_1100,
        vop2(0x25, 2, 14, 2),
        0x8010_8110,
        0xBF0A_1110,
        0xBF85_FFFA,
        0x8009_8109,
        0xBF0A_0C09,
        0xBF85_FFF5,
        0x88FE_0A7E,
        vop2(0x25, 2, 9, 2),
    ]);
    w
}

pub fn counter(first: u32, second: u32, fixed: bool) -> Vec<u32> {
    let mut w = prepare(first, second);
    if fixed {
        w.pop();
        w.extend([0xBE8C_03FF, 12]); // literal bound, class C
    }
    w.extend([
        0xBF0A_0C09, // s_cmp_lt_u32 s9, s12
        0xBF84_0006, // zero-trip skip
        0xD760_000E,
        0x0001_1100,          // s14 = v0 of named lane 8 (inactive)
        vop2(0x25, 2, 14, 2), // v2 += s14
        0x8009_8109,          // s9 += 1
        0xBF0A_0C09,
        0xBF85_FFFA,         // iterate while s9 < bound
        0x88FE_0A7E,         // restore EXEC
        vop2(0x25, 2, 9, 2), // every lane sees final scalar count
    ]);
    w
}

// A value computed before EXEC narrows is read from an inactive named lane in
// every iteration. Its exchange can move once, including when a wave takes no
// iterations and its neighbour takes twelve. The result must survive slot reuse.
pub fn hoisted_counter(first: u32, second: u32) -> Vec<u32> {
    let mut w = counter(first, second, false);
    w.insert(0, vop2(0x25, 7, 0x87, 0)); // v7 = thread id + 7, with full EXEC
    let read = w.iter().position(|&v| v == 0x0001_1100).unwrap();
    w[read] = 0x0001_1107; // named lane 8 of v7
    w
}

// Wave 0 has no active EXEC bits on entry to the loop; wave 1 still needs the
// whole-wave permute/read and 64..127 EXEC-shrinking iterations.
pub fn idle_wave() -> Vec<u32> {
    let mut w = EXEC_PERMUTE.to_vec();
    let at = w.iter().position(|&v| v == 0x7D8A_0080).unwrap();
    w[at] = 0x7D82_00BF; // vcc = 63 < thread id
    w
}

pub fn mask_scan(first: u32, second: u32) -> Vec<u32> {
    let mut w = prepare(first, second);
    w.extend([
        0xBF07_800C, // mask != 0
        0xBF84_0009,
        0xBE8D_130C, // s13 = ff1(s12)
        0x8F0E_0D81, // s14 = 1 << s13
        0x8A0C_0E0C, // s12 &= ~s14
        0xD760_000F,
        0x0000_1B00, // s15 = v0 of named lane s13
        vop2(0x25, 2, 15, 2),
        0x8009_8109,
        0xBF07_800C,
        0xBF85_FFF7,
        0x88FE_0A7E,
        vop2(0x25, 2, 9, 2),
    ]);
    w
}

// Two scalar-buffer bounds, then a scalar load in each active iteration. The GPU
// fixture's `buffer=false` form reads the same three words from title address s[0:1].
pub fn range(buffer: bool) -> Vec<u32> {
    let mut w = prepare(0, 0);
    w.pop();
    let load = |dst: u32| {
        if buffer {
            0xF420_0002 | dst << 6
        } else {
            0xF400_0000 | dst << 6
        }
    };
    w.extend([
        0x7E1A_0500, // s13 = first active thread id
        0x900D_860D, // s13 >>= 6
        0x8F0D_820D, // byte offset = 4 * wave id
        load(12),
        0x1A00_0000, // bound = input[wave]
        0xBF0A_0C09,
        0xBF84_0009,
        0xD760_000E,
        0x0001_1100,
        load(15),
        0xFA00_0008, // input[2], only for active loop iterations
        vop2(0x25, 2, 14, 2),
        vop2(0x25, 2, 15, 2),
        0x8009_8109,
        0xBF0A_0C09,
        0xBF85_FFF7,
        0x88FE_0A7E,
        vop2(0x25, 2, 9, 2),
    ]);
    w
}

pub fn exec_mask_scan(threshold: u32) -> Vec<u32> {
    vec![
        0x7E04_0280,
        0xBE89_0380,
        0x7D82_00FF,
        threshold, // EXEC's lanes above threshold seed both words
        0xBE8A_246A,
        0xBE8C_047E, // s[12:13] = EXEC
        0xBF13_800C,
        0xBF84_0008,
        0xBE90_140C, // ff1 of the full 64-bit mask
        0xBE8C_1C10, // s_bitset0_b64 s[12:13], s16: remove that bit
        0xD760_0011,
        0x0000_2100,
        vop2(0x25, 2, 17, 2),
        0x8009_8109,
        0xBF13_800C,
        0xBF85_FFF8,
        0x88FE_0A7E,
        vop2(0x25, 2, 9, 2),
    ]
}

pub const EXEC_PERMUTE: &[u32] = &[
    0xBE90_03FF,
    0x7654_3210,
    0xBE91_03FF,
    0xFEDC_BA98, // identity selectors in each 16-lane row
    0xBE89_0380,
    0x7E04_0280,
    0x7D8A_0080,
    0xBE8A_246A,
    0xBF88_000B,
    0xBE8C_047E, // save this iteration's EXEC
    0xBEFE_04C1, // whole-wave EXEC
    0xD778_0004,
    0x0044_2100, // swap the two 16-lane rows of each host half
    0xBEFE_040C,
    0xD760_0009,
    0x0001_1104, // read v4 of console lane 8 -> v0 of lane 24
    0x4A04_0481,
    0x7D82_0102,
    0x87FE_6A7E,
    0xBF89_FFF5,
    0x88FE_0A7E,
    0x4A04_0409,
];

// One elected lane (32) per console wave increments a separate counter at
// input[128 + wave]. Its old value is read with full console first-lane semantics.
pub fn atomic(first: u32, second: u32, buffer: bool) -> Vec<u32> {
    let mut w = prepare(first, second);
    w.extend([
        0x7E1A_0500,
        0x900D_860D,
        0x8F0D_820D,
        0xBF0A_0C09,
        0xBF84_000F,
        0xD760_000E,
        0x0001_1100,
        0x7E0A_0281, // v5 = increment
        0x7E0C_020D, // v6 = wave * 4
        0x7D84_06A0, // elect local lane 32
        0xBE92_246A,
        if buffer { 0xE0C8_4200 } else { 0xDCC9_8200 },
        if buffer { 0x0D01_0500 } else { 0x0500_0506 },
        0xBEFE_0412,
        0x7E1E_0505, // s15 = old counter returned to lane 32
        vop2(0x25, 2, 14, 2),
        vop2(0x25, 2, 15, 2),
        0x8009_8109,
        0xBF0A_0C09,
        0xBF85_FFF1,
        0x88FE_0A7E,
        vop2(0x25, 2, 9, 2),
    ]);
    w
}
