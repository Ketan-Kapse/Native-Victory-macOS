//! Shared-memory ordering in the emitted console-PC schedule. Synthetic only.
use super::console_tests::lower_body_with_lds;
mod cases {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/communicating_cases.rs"
    ));
}

#[test]
fn lds_consumers_wait_for_other_waves_scalar_loops_and_partial_exec() {
    for loops in [false, true] {
        for threads in [128, 256, 1024] {
            let l = lower_body_with_lds(&cases::producer(loops), threads, threads * 4).unwrap();
            assert!(l.synchronized && l.exchanges);
            for reverse in [false, true] {
                let got = l
                    .values
                    .as_ref()
                    .unwrap()
                    .run(threads as usize, &[], reverse);
                assert_eq!(got.writes, vec![1; threads as usize]);
                assert_eq!(got.lds_reads.len(), threads as usize);
                let writes: usize = (0..threads)
                    .map(|id| {
                        2 + if id % 64 < 32 {
                            0
                        } else if loops {
                            (id / 64 % 4) as usize
                        } else {
                            1
                        }
                    })
                    .sum();
                assert_eq!(
                    got.lds_writes.len(),
                    writes,
                    "no replayed or speculative writes"
                );
                for (id, &x) in got.output.iter().enumerate() {
                    let other = id as u32 ^ 64;
                    let trips = if loops { other / 64 % 4 } else { 1 };
                    let want = if other % 64 < 32 {
                        0
                    } else {
                        trips * (other / 64 * 64 + 41)
                    };
                    assert_eq!(x, want, "threads={threads}, loops={loops}, lane={id}");
                }
            }
        }
    }
}

#[test]
fn an_elected_lds_atomic_runs_once_per_console_wave_and_broadcasts_to_empty_halves() {
    for threads in [128, 256, 1024] {
        for iterations in [1, 2] {
            let l = lower_body_with_lds(&cases::elected(iterations), threads, 512).unwrap();
            assert!(l.synchronized && l.exchanges);
            for reverse in [false, true] {
                let got = l
                    .values
                    .as_ref()
                    .unwrap()
                    .run(threads as usize, &[], reverse);
                let count = threads / 64 * iterations;
                assert_eq!(got.lds_atomics.len(), count as usize);
                for (n, &(lane, addr, old, new)) in got.lds_atomics.iter().enumerate() {
                    assert_eq!((lane % 64, addr, old, new), (33, 0, n as u32, n as u32 + 1));
                }
                assert_eq!(got.writes, vec![1; threads as usize]);
                let mut tickets = Vec::new();
                for wave in got.output.chunks(64) {
                    assert!(
                        wave.iter().all(|&x| x == wave[0]),
                        "including empty lower EXEC half"
                    );
                    assert_eq!(wave[0] >> 16, count);
                    if iterations == 2 {
                        assert!((wave[0] >> 8 & 255) < (wave[0] & 255));
                        tickets.push(wave[0] >> 8 & 255);
                    }
                    tickets.push(wave[0] & 255);
                }
                tickets.sort_unstable();
                assert_eq!(tickets, (0..count).collect::<Vec<_>>());
            }
        }
    }
}

#[test]
fn exec_shrinking_loops_keep_lds_and_exit_decisions_in_step_across_both_halves() {
    for threads in [128, 1024] {
        let l = lower_body_with_lds(&cases::lane_loop(), threads, threads * 4).unwrap();
        assert!(l.synchronized && l.exchanges);
        for reverse in [false, true] {
            let got = l
                .values
                .as_ref()
                .unwrap()
                .run(threads as usize, &[], reverse);
            assert_eq!(got.writes, vec![1; threads as usize]);
            assert_eq!(got.lds_writes.len(), threads as usize * 9 / 2);
            for (id, &x) in got.output.iter().enumerate() {
                assert_eq!(
                    x,
                    ((id ^ 32) % 64 / 8 + id / 64 * 64 + 40) as u32,
                    "lane={id}"
                );
            }
        }
    }
}

#[test]
fn cyclic_and_bypassed_title_barriers_keep_the_precise_refusal() {
    for cyclic in [false, true] {
        let mut body = cases::producer(true);
        if cyclic {
            let at = body
                .iter()
                .position(|&w| w & 0xffff0000 == 0xBF85_0000)
                .unwrap();
            body.insert(at, 0xBF8A_0000);
            body[at + 1] -= 1; // backedge still points to the original loop head
        } else {
            // A lane-dependent early exit bypasses all title phases. The direct
            // lowering has not proved that these waves can safely be retained.
            body.splice(
                0..0,
                [
                    0x7D84_0080,
                    0xBE94_246A,
                    0xBF88_0001,
                    0xBF81_0000,
                    0xBEFE_0414,
                ],
            );
        }
        let e = lower_body_with_lds(&body, 128, 512)
            .err()
            .expect("unproved title barrier");
        assert!(
            e.to_string()
                .contains("a title barrier not proved exactly once on every group path"),
            "{e}"
        );
    }
}

#[test]
fn lds_compare_exchange_and_signed_unsigned_min_max_return_the_atomic_old_value() {
    for (op, initial, data, replacement) in [
        (0x30, 0, 0, 7), // compare-exchange: first wins, second fails
        (0x27, 10, 3, 0),
        (0x28, 2, 9, 0), // unsigned min/max
        (0x25, 2, u32::MAX, 0),
        (0x26, u32::MAX, 9, 0), // signed min/max
    ] {
        let mut body = cases::elected(1);
        let init = body.iter().position(|&w| w == 0x7E0A_0280).unwrap();
        body.splice(init..init + 1, [0x7E0A_02FF, initial]);
        let value = body.iter().position(|&w| w == 0x7E0A_0281).unwrap();
        body.splice(
            value..value + 1,
            [0x7E0A_02FF, data, 0x7E08_02FF, replacement],
        );
        let at = body.iter().position(|&w| w == 0xD880_0000).unwrap();
        body[at] = 0xD800_0000 | op << 18;
        if op == 0x30 {
            body[at + 1] |= 4 << 16;
        }
        // Added three words inside the loop (the initializer is before it).
        let branch = body
            .iter()
            .position(|&w| w & 0xffff0000 == 0xBF85_0000)
            .unwrap();
        body[branch] -= 3;
        let l = lower_body_with_lds(&body, 128, 512).unwrap();
        for reverse in [false, true] {
            let got = l.values.as_ref().unwrap().run(128, &[], reverse);
            assert_eq!(got.lds_atomics.len(), 2);
            let mut expected = initial;
            for &(lane, addr, old, new) in &got.lds_atomics {
                let next = match op {
                    0x30 => {
                        if expected == data {
                            replacement
                        } else {
                            expected
                        }
                    }
                    0x27 => expected.min(data),
                    0x28 => expected.max(data),
                    0x25 => (expected as i32).min(data as i32) as u32,
                    0x26 => (expected as i32).max(data as i32) as u32,
                    _ => unreachable!(),
                };
                assert_eq!((addr, old, new), (0, expected, next));
                assert_eq!(lane % 64, 33);
                expected = next;
            }
            for &(lane, _, old, _) in &got.lds_atomics {
                let want = expected.wrapping_shl(16).wrapping_add(old);
                assert!(
                    got.output[lane / 64 * 64..lane / 64 * 64 + 64]
                        .iter()
                        .all(|&v| v == want)
                );
            }
        }
    }
}

#[test]
fn acyclic_buffer_add_min_and_compare_exchange_are_per_lane_with_console_wide_results() {
    for threads in [128, 1024] {
        for compare_exchange in [false, true] {
            let mut body = cases::buffer_atomics(true);
            if compare_exchange {
                let at = body.iter().position(|&w| w == 0xE0D8_5800).unwrap();
                body[at] = 0xE0C4_5800; // buffer_atomic_cmpswap
                body.splice(at..at, [0x4A10_00FF, 1001]); // v8 compare = input[thread] + 1
                let branch = body
                    .iter()
                    .position(|&w| w & 0xffff0000 == 0xBF88_0000)
                    .unwrap();
                body[branch] += 2;
            }
            let l = lower_body_with_lds(&body, threads, 0).unwrap();
            assert!(l.synchronized && l.exchanges);
            let mut input = vec![0; 512 + threads as usize];
            for (id, v) in input[512..].iter_mut().enumerate() {
                *v = 1000 + id as u32;
            }
            for reverse in [false, true] {
                let got = l
                    .values
                    .as_ref()
                    .unwrap()
                    .run(threads as usize, &input, reverse);
                assert_eq!(got.atomics.len(), threads as usize / 64 * 31 * 2);
                for id in 0..threads as usize {
                    let base = 1000 + id as u32 / 64 * 64 + 33;
                    let want = base
                        + if id % 64 > 32 {
                            1001 + id as u32 + id as u32 / 64 * 64 + 8
                        } else {
                            0
                        };
                    assert_eq!(got.output[id], want, "lane {id}");
                    assert_eq!(
                        got.memory[&(512 + id)],
                        if id % 64 > 32 { 7 } else { 1000 + id as u32 }
                    );
                    let old: Vec<_> = got
                        .atomics
                        .iter()
                        .filter(|x| x.0 == 512 + id)
                        .map(|x| x.1)
                        .collect();
                    assert_eq!(
                        old,
                        if id % 64 > 32 {
                            vec![1000 + id as u32, 1001 + id as u32]
                        } else {
                            vec![]
                        }
                    );
                }
            }
        }
    }
}

#[test]
fn a_loop_limit_escape_cancels_peers_waiting_at_a_title_barrier() {
    let mut l = lower_body_with_lds(&cases::producer(true), 256, 1024).unwrap();
    let program = l.values.as_mut().unwrap();
    program.replace_i32_constant(super::loop_limit(), 1);
    for reverse in [false, true] {
        let got = program.run(256, &[], reverse);
        assert_eq!(
            got.writes,
            vec![0; 256],
            "no wave may consume an incomplete LDS phase"
        );
        assert!(got.lds_reads.is_empty());
    }
}

#[test]
fn inactive_lds_readers_never_dereference_their_invalid_addresses() {
    let mut body = cases::producer(false);
    let at = body.iter().position(|&w| w == 0x7D82_069F).unwrap();
    body.splice(at + 1..at + 1, [0x0202_02FF, 0xFFFF_FFF0]); // v1 = upper ? v1 : invalid
    let at = body.iter().position(|&w| w == 0xD760_000E).unwrap();
    body.splice(at..at, [0xD8D8_0000, 0x0200_0001]); // read own initialized LDS into v2
    let at = body
        .iter()
        .position(|&w| w & 0xffff0000 == 0xBF88_0000)
        .unwrap();
    body[at] += 2;
    let l = lower_body_with_lds(&body, 128, 512).unwrap();
    for reverse in [false, true] {
        let got = l.values.as_ref().unwrap().run(128, &[], reverse);
        assert_eq!(got.lds_reads.len(), 192);
        assert!(got.lds_reads.iter().all(|&(_, address)| address < 128));
        for (id, &value) in got.output.iter().enumerate() {
            assert_eq!(
                value,
                if id % 64 < 32 {
                    0
                } else {
                    ((id ^ 64) / 64 * 64 + 41) as u32
                }
            );
        }
    }
}

#[test]
fn coordinator_shared_memory_fixtures_lower_with_native_bindings() {
    for (body, lds_bytes) in [
        (cases::producer(false), 4096u32),
        (cases::producer(true), 4096),
        (cases::elected(1), 512),
        (cases::elected(2), 512),
        (cases::lane_loop(), 4096),
        (cases::buffer_atomics(false), 0),
    ] {
        let mut words = body;
        words.extend([0x3402_0082, 0xDC70_8000, 0x0002_0201, 0xBF81_0000]);
        let native = native_fixture(&words, 1024, lds_bytes);
        let l = super::console_tests::lower_native(&native).unwrap();
        assert!(l.synchronized && l.exchanges);
    }
}

fn native_fixture(words: &[u32], threads: u32, lds_bytes: u32) -> crate::native::NativeProgram {
    use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 5,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1 | lds_bytes.div_ceil(512) << 15,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

#[test]
fn retaining_an_empty_console_wave_makes_title_barriers_mandatory_without_replaying_its_effects() {
    let mut words = cases::producer(true);
    let at = words.iter().position(|&w| w == 0x7E18_0506).unwrap();
    // Wave zero exits with empty EXEC after initialization. The existing retry
    // keeps it participating, with all subsequent memory effects still off.
    words.splice(at..at, [0x7D8A_0C80, 0x87FE_6A7E, 0xBF88_0000]);
    words.extend([0x3402_0082, 0xDC70_8000, 0x0002_0201, 0xBF81_0000]);
    words[at + 2] |= (words.len() - (at + 2) - 2) as u32;
    let native = native_fixture(&words, 256, 1024);
    let p = crate::ir::build(&native.code, true, 16).unwrap();
    let q = crate::ir::keep_lanes_to_end(&p).expect("existing empty-wave retry");
    let l = super::console_tests::lower_native_program(&native, &q).unwrap();
    assert!(l.synchronized);
    for reverse in [false, true] {
        let got = l.values.as_ref().unwrap().run(256, &[], reverse);
        assert!(got.writes[..64].iter().all(|&n| n == 0));
        assert!(got.writes[64..].iter().all(|&n| n == 1));
        assert_eq!(got.lds_reads.len(), 192);
        assert_eq!(
            got.lds_writes
                .iter()
                .filter(|&&(lane, _)| lane < 64)
                .count(),
            64
        );
        for id in 64..256 {
            let other = id ^ 64;
            let want = if other % 64 < 32 {
                0
            } else {
                (other / 64 % 4) * (other / 64 * 64 + 41)
            };
            assert_eq!(got.output[id], want as u32, "lane {id}");
        }
    }
}

#[test]
fn root_atomic_fixtures_use_a_positive_address_not_a_negative_immediate() {
    use crate::gfx10::Fields;
    for body in [cases::buffer_atomics(false), cases::single_buffer_atomic(false, false), cases::single_buffer_atomic(false, true)] {
        let bytes: Vec<_> = body.iter().copied().chain([0xBF81_0000]).flat_map(u32::to_le_bytes).collect();
        let decoded = crate::gfx10::decode(&bytes).unwrap();
        let atomics: Vec<_> = decoded.iter().filter(|i| i.name().unwrap().starts_with("global_atomic_")).collect();
        assert!(!atomics.is_empty());
        for i in atomics {
            let Fields::Flat { offset, .. } = i.fields(true) else { panic!("global atomic") };
            assert_eq!(offset, 0, "+2048 belongs in v6: the signed immediate would encode -2048");
        }
        assert!(body.windows(2).any(|w| w == [cases::address_add(), 2048]));
    }
    // The table/descriptor form really has an unsigned immediate.
    for minimum in [false, true] {
        let body = cases::single_buffer_atomic(true, minimum);
        let bytes: Vec<_> = body.iter().copied().chain([0xBF81_0000]).flat_map(u32::to_le_bytes).collect();
        let decoded = crate::gfx10::decode(&bytes).unwrap();
        let atomic = decoded.iter().find(|i| i.name().unwrap().starts_with("buffer_atomic_")).unwrap();
        let Fields::Buffer { offset, .. } = atomic.fields(true) else { panic!("buffer atomic") };
        assert_eq!(offset, 2048);
    }
}

#[test]
fn identical_lds_initializers_share_their_phase_fences() {
    let original = cases::producer(false);
    let baseline = lower_body_with_lds(&original, 128, 512).unwrap();
    let mut batched = original;
    let at = batched.windows(2).position(|w| w == [0xD834_0000, 0x0000_0201]).unwrap();
    batched.splice(at..at, [0xD834_0000, 0x0000_0201].repeat(40));
    let optimized = lower_body_with_lds(&batched, 128, 512).unwrap();
    assert_eq!(baseline.barriers, optimized.barriers, "identical overlapping writes need no extra fences");
    for reverse in [false, true] {
        let before = baseline.values.as_ref().unwrap().run(128, &[], reverse);
        let after = optimized.values.as_ref().unwrap().run(128, &[], reverse);
        assert_eq!(before.output, after.output);
        assert_eq!(before.barriers, after.barriers);
        assert_eq!(after.lds_writes.len() - before.lds_writes.len(), 40 * 128);
    }
}

#[test]
fn a_masked_communicating_dispatch_returns_before_any_exchange_or_effect() {
    let l = super::console_tests::lower_body_options(&cases::producer(true), 128, 512,
        super::Options { thread_counts: true, ..Default::default() }).unwrap();
    let got = l.values.as_ref().unwrap().run_with_user_data(128, &[], false, &[0; 4]);
    assert_eq!(got.barriers, 0);
    assert_eq!(got.exchange_writes, 0);
    assert!(got.lds_writes.is_empty() && got.lds_reads.is_empty() && got.lds_atomics.is_empty());
    assert!(got.writes.iter().all(|&n| n == 0));
}
