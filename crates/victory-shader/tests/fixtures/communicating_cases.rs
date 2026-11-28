// Hand-assembled synthetic GFX10 only. v2 is the result; s[2:3] is
// reserved for the GPU output address. No title instructions or data.
fn vop2(op: u32, dst: u32, src: u32, vector: u32) -> u32 {
    op << 25 | dst << 17 | vector << 9 | src
}

#[derive(Default)]
struct Asm {
    words: Vec<u32>,
    labels: std::collections::BTreeMap<&'static str, usize>,
    branches: Vec<(usize, &'static str)>,
}
impl Asm {
    fn emit(&mut self, words: &[u32]) {
        self.words.extend(words);
    }
    fn label(&mut self, name: &'static str) {
        self.labels.insert(name, self.words.len());
    }
    fn branch(&mut self, op: u32, label: &'static str) {
        self.branches.push((self.words.len(), label));
        self.words.push(op);
    }
    fn finish(mut self) -> Vec<u32> {
        for (at, label) in self.branches {
            let offset = self.labels[label] as isize - at as isize - 1;
            self.words[at] |= i16::try_from(offset).unwrap() as u16 as u32;
        }
        self.words
    }
}

// Scalar loop counts vary by console wave (0,1,2,3). Only the upper half
// produces. Every consumer reads the corresponding lane in the OTHER wave.
// A second barrier protects those reads from the subsequent overwrite.
pub fn producer(loops: bool) -> Vec<u32> {
    let mut a = Asm::default();
    a.emit(&[
        vop2(0x1a, 1, 0x82, 0), // byte address = thread id * 4
        vop2(0x1b, 3, 0xbf, 0), // console lane
        vop2(0x16, 6, 0x86, 0), // console wave
        0x7E04_0280,
        0xD834_0000,
        0x0000_0201, // initialize every word
        0x7E18_0506,
        0x870C_830C, // scalar count = wave & 3
        0xBE89_0380, // counter = 0
        0x7D82_069F,
        0xBE8A_246A, // upper half EXEC; save full EXEC
    ]);
    if loops {
        a.emit(&[0xBF0A_0C09]);
        a.branch(0xBF84_0000, "join");
    } else {
        a.branch(0xBF88_0000, "join");
    }
    a.label("loop");
    a.emit(&[
        0xD760_000E,
        0x0001_5100, // named lane 40, exchanged across halves
        vop2(0x25, 2, 14, 2),
        vop2(0x25, 2, 0x81, 2),
        0xD834_0000,
        0x0000_0201,
    ]);
    if loops {
        a.emit(&[0x8009_8109, 0xBF0A_0C09]);
        a.branch(0xBF85_0000, "loop");
    }
    a.label("join");
    a.emit(&[
        0xBEFE_040A,
        0xBF8A_0000, // restore EXEC; title barrier
        vop2(0x1d, 1, 0xff, 0),
        64, // thread id XOR 64
        vop2(0x1a, 1, 0x82, 1),
        0xD8D8_0000,
        0x0200_0001, // read other console wave's LDS word
        0xBF8A_0000,
        0x7E0A_0280,
        0xD834_0000,
        0x0000_0501, // overwrite AFTER readers
    ]);
    a.finish()
}

// The first active lane is 33 (the lower host half has no EXEC lanes).
// Each console wave elects once per iteration using ff1 of the full EXEC,
// atomically allocates from a common LDS counter, and exchanges its old value.
// Output: final counter in bits 16+, ticket 0 in bits 8..15, ticket 1 in 0..7.
pub fn elected(iterations: u32) -> Vec<u32> {
    assert!((1..=2).contains(&iterations));
    let mut a = Asm::default();
    a.emit(&[
        vop2(0x1b, 3, 0xbf, 0),
        0x7E0A_0280,
        0x7E0C_0280, // zero data and address
        0x7D84_0080,
        0xBE8A_246A, // only thread 0 initializes counter
        0xD834_0000,
        0x0000_0506,
        0xBEFE_040A,
        0xBF8A_0000,
        0x7D82_06A0,
        0xBE8A_246A, // lanes 33..63
        0xBE91_0380,
        0xBE92_0380, // accumulator and loop count
    ]);
    a.label("loop");
    a.emit(&[
        0xBE8C_147E, // ff1_i32_b64 s12, EXEC (both halves)
        0x7E0A_0281,
        0x7D84_060C,
        0xBE8E_246A, // elect lane s12, save upper EXEC
        0xD880_0000,
        0x0500_0506, // ds_add_rtn_u32 v5, v6, v5
        0x7E20_0505, // s16 = first active lane's returned value
        0xBEFE_040E,
        0x8F11_8811,
        0x8011_1011, // pack tickets in scalar s17
        0x8012_8112,
        0xBF0A_0012 | (0x80 + iterations) << 8,
    ]);
    a.branch(0xBF85_0000, "loop");
    a.emit(&[
        0xBEFE_040A,
        0xBF8A_0000,
        0xD8D8_0000,
        0x0700_0006,
        vop2(0x1a, 7, 0x90, 7),
        vop2(0x25, 2, 17, 7),
    ]);
    a.finish()
}

// Every lane leaves at its own iteration (0..7). The lower host half of a
// wave finishes before the upper. The final cross-half LDS read uses the
// other half's word; title barriers separate initialization, loop and read.
pub fn lane_loop() -> Vec<u32> {
    let mut a = Asm::default();
    a.emit(&[
        vop2(0x1b, 3, 0xbf, 0),
        vop2(0x16, 3, 0x83, 3), // bound = console lane / 8
        vop2(0x1a, 1, 0x82, 0),
        0x7E04_0280,
        0xD834_0000,
        0x0000_0201,
        0xBF8A_0000,
        0x7D8A_0680,
        0xBE8A_246A,
    ]);
    a.branch(0xBF88_0000, "join");
    a.label("loop");
    a.emit(&[
        0xD760_000E,
        0x0001_5100, // consume named lane in output after join
        0x4A04_0481,
        0xD834_0000,
        0x0000_0201,
        0x7D82_0702,
        0x87FE_6A7E,
    ]);
    a.branch(0xBF89_0000, "loop");
    a.label("join");
    a.emit(&[
        0xBEFE_040A,
        0xBF8A_0000,
        vop2(0x1d, 1, 0xa0, 0),
        vop2(0x1a, 1, 0x82, 1),
        0xD8D8_0000,
        0x0200_0001,
        vop2(0x25, 2, 14, 2),
    ]);
    a.finish()
}

// Per-lane (not elected) add and unsigned-min at input[512 + thread id].
// v_readfirstlane of the add result supplies a scalar to BOTH host halves.
pub fn buffer_atomics(buffer: bool) -> Vec<u32> {
    let mut a = Asm::default();
    a.emit(&[
        vop2(0x1b, 3, 0xbf, 0),
        vop2(0x1a, 6, 0x82, 0),
        0x7E0A_0281,
        0x7E0E_0280,
        0xBE90_0380,
        0x7D82_06A0,
        0xBE8A_246A,
    ]);
    // GLOBAL's immediate is signed 12-bit: 0x800 is -2048, unlike
    // BUFFER's unsigned offset. Add +2048 to the VGPR address explicitly.
    if !buffer {
        a.emit(&[address_add(), 2048]);
    }
    a.branch(0xBF88_0000, "join");
    a.emit(&[
        0xD760_000E,
        0x0001_1100, // live named read in the divergent block
        if buffer { 0xE0C8_5800 } else { 0xDCC9_8000 },
        if buffer { 0x8001_0506 } else { 0x0500_0506 },
        0x7E20_0505, // first add's old value from active lane 33
        0x7E0E_0287,
        if buffer { 0xE0D8_5800 } else { 0xDCD9_8000 },
        if buffer { 0x8001_0706 } else { 0x0700_0706 },
        vop2(0x25, 7, 14, 7),
    ]);
    a.label("join");
    a.emit(&[0xBEFE_040A, vop2(0x25, 2, 16, 7)]);
    a.finish()
}

// Isolate one per-lane atomic, including its returned old value. The root
// form uses s[0:1] as the address; the descriptor form loads s[12:15] via s4.
pub fn single_buffer_atomic(descriptor: bool, minimum: bool) -> Vec<u32> {
    let mut w = Vec::new();
    if descriptor {
        w.extend([0xBE8A_0304, 0xBE8B_0380, 0xF408_0305, 0xFA00_0000, 0xBF8C_C07F]);
    }
    w.extend([vop2(0x1a, 6, 0x82, 0), 0x7E04_0200 | if minimum { 0x87 } else { 0x81 }]);
    if descriptor {
        w.extend([if minimum { 0xE0D8_5800 } else { 0xE0C8_5800 }, 0x8003_0206]);
        // Output at byte 8192 of the same raw UAV, disjoint from every input.
        w.extend([vop2(0x25, 6, 0xff, 6), 8192, 0xE070_1000, 0x8003_0206, 0xBF81_0000]);
    } else {
        w.extend([address_add(), 2048]);
        w.extend([if minimum { 0xDCD9_8000 } else { 0xDCC9_8000 }, 0x0200_0206]);
        w.extend([0x3402_0082, 0xDC70_8000, 0x0002_0201, 0xBF81_0000]);
    }
    w
}

// Shared encoding for the decoder/address regression check.
pub fn address_add() -> u32 { vop2(0x25, 6, 0xff, 6) }
