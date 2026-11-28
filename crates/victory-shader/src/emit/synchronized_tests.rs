use super::console_tests::lower_body;
mod cases {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/synchronized_cases.rs"
    ));
}

#[test]
fn scalar_counter_values_survive_zero_one_and_twelve_trips_in_different_waves() {
    for (a, b, fixed) in [
        (0, 12, false),
        (1, 0, false),
        (12, 1, false),
        (12, 12, true),
    ] {
        for threads in [112, 128, 256] {
            let l = lower_body(&cases::counter(a, b, fixed), threads, false).unwrap();
            assert!(l.synchronized && l.exchanges);
            for reverse in [false, true] {
                let got = l
                    .values
                    .as_ref()
                    .unwrap()
                    .run(threads as usize, &[], reverse);
                assert!(got.barriers >= 2);
                assert_eq!(got.writes, vec![1; threads as usize]);
                for (i, &value) in got.output.iter().enumerate() {
                    let wave = i as u32 / 64;
                    let count = if fixed {
                        12
                    } else if wave == 0 {
                        a
                    } else {
                        b
                    };
                    let want = count * if i % 64 < 32 { 1 } else { wave * 64 + 9 };
                    assert_eq!(value, want, "threads={threads} a={a} b={b} lane={i}");
                }
            }
        }
    }
}

#[test]
fn an_invariant_exchange_runs_once_even_with_different_wave_trip_counts() {
    for threads in [112, 128, 256] {
        let mut cost = None;
        for (a, b) in [(0, 0), (0, 12), (1, 0), (12, 1)] {
            let l = lower_body(&cases::hoisted_counter(a, b), threads, false).unwrap();
            for reverse in [false, true] {
                let got = l
                    .values
                    .as_ref()
                    .unwrap()
                    .run(threads as usize, &[], reverse);
                let this = (got.barriers, got.exchange_writes, got.exchange_reads);
                assert_eq!(
                    *cost.get_or_insert(this),
                    this,
                    "no exchange or vote in the loop"
                );
                assert_eq!(got.writes, vec![1; threads as usize]);
                for (id, &x) in got.output.iter().enumerate() {
                    let count = if id < 64 { a } else { b };
                    assert_eq!(
                        x,
                        count
                            * if id % 64 < 32 {
                                1
                            } else {
                                id as u32 / 64 * 64 + 16
                            }
                    );
                }
                if a == 0 && b == 12 {
                    assert!(
                        got.steps[32] < got.steps[96] / 2,
                        "idle wave skips the body"
                    );
                }
            }
        }
    }
}

#[test]
fn an_empty_console_wave_still_attends_its_neighbours_required_exchanges() {
    let l = lower_body(&cases::idle_wave(), 128, false).unwrap();
    for reverse in [false, true] {
        let got = l.values.as_ref().unwrap().run(128, &[], reverse);
        assert_eq!(got.writes, vec![1; 128]);
        for (id, &x) in got.output.iter().enumerate() {
            assert_eq!(x, if id < 64 { 0 } else { (id + 88) as u32 });
        }
    }
}

#[test]
fn a_wholly_unlaunched_group_returns_before_any_exchange_or_body_load() {
    let l = super::console_tests::lower_body_options(
        &cases::range(true),
        128,
        0,
        super::Options {
            thread_counts: true,
            ..Default::default()
        },
    )
    .unwrap();
    // The integer harness supplies zero dispatch dimensions in user data.
    for reverse in [false, true] {
        let got = l.values.as_ref().unwrap().run(128, &[12, 12, 7], reverse);
        assert_eq!(got.writes, vec![0; 128]);
        assert_eq!(
            (got.barriers, got.exchange_reads, got.exchange_writes),
            (0, 0, 0)
        );
        assert!(got.reads.is_empty());
        let live = l.values.as_ref().unwrap().run_with_user_data(
            128,
            &[0, 12, 7],
            reverse,
            &[0, 128, 1, 1],
        );
        let ordinary = lower_body(&cases::range(true), 128, false).unwrap();
        let ordinary = ordinary
            .values
            .as_ref()
            .unwrap()
            .run(128, &[0, 12, 7], reverse);
        assert_eq!(
            live.output, ordinary.output,
            "covered groups keep their effects"
        );
        assert_eq!(live.writes, ordinary.writes);
    }
}

#[test]
fn round_switches_are_independent_and_part_of_the_options_cache_key() {
    #[allow(dead_code)]
    mod predicated {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/predicated_cases.rs"
        ));
    }
    #[allow(dead_code)]
    mod communicating {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/communicating_cases.rs"
        ));
    }
    let bodies = [
        predicated::READ_LANE.to_vec(),
        cases::counter(0, 12, false),
        communicating::producer(true),
    ];
    let mut keys = std::collections::HashSet::new();
    for flags in 0..8 {
        let o = super::Options {
            no_if_convert: flags & 1 != 0,
            no_sync_loops: flags & 2 != 0,
            no_comm_waves: flags & 4 != 0,
            ..Default::default()
        };
        assert!(
            keys.insert(format!("{o:?}")),
            "the native cache uses this exact Options representation"
        );
        for (round, body) in bodies.iter().enumerate() {
            let result = super::console_tests::lower_body_options(
                body,
                128,
                if round == 2 { 512 } else { 0 },
                o,
            );
            if flags & (1 << round) == 0 {
                let baseline = lower_body(body, 128, round == 2).unwrap();
                assert_eq!(
                    result.unwrap().dxil,
                    baseline.dxil,
                    "unrelated switch: {flags}, {round}"
                );
            } else {
                assert!(
                    matches!(
                        result,
                        Err(crate::ShaderError::Unsupported {
                            what: super::lower::DIVERGENT_EXCHANGE,
                            ..
                        })
                    ),
                    "{flags}, {round}"
                );
            }
        }
    }
}

#[test]
fn mask_scans_exchange_the_seed_and_remove_each_bit_once() {
    for (a, b) in [
        (0, u32::MAX),
        (1, 0),
        (u32::MAX, 1),
        (0x80000001, 0x55555555),
    ] {
        let l = lower_body(&cases::mask_scan(a, b), 128, false).unwrap();
        assert!(l.synchronized && l.exchanges);
        for reverse in [false, true] {
            let got = l.values.as_ref().unwrap().run(128, &[], reverse);
            assert_eq!(got.writes, vec![1; 128]);
            for (i, &value) in got.output.iter().enumerate() {
                let mask = if i < 64 { a } else { b };
                let sum: u32 = (0..32)
                    .filter(|bit| mask & (1 << bit) != 0)
                    .map(|bit| i as u32 / 64 * 64 + bit)
                    .sum();
                let want = mask.count_ones() + if i % 64 < 32 { 0 } else { sum };
                assert_eq!(value, want, "a={a:x} b={b:x} lane={i}");
            }
        }
    }
}

#[test]
fn scalar_buffer_ranges_load_only_for_the_waves_live_iterations() {
    let l = lower_body(&cases::range(true), 128, false).unwrap();
    assert!(l.synchronized && l.guarded_buffer_reads);
    for (a, b) in [(0, 0), (0, 12), (1, 0), (12, 1)] {
        for reverse in [false, true] {
            let got = l.values.as_ref().unwrap().run(128, &[a, b, 7], reverse);
            assert_eq!(got.writes, vec![1; 128]);
            // Each cbuffer load fetches one 16-byte row: one bound per lane,
            // then one row per live iteration, including inactive EXEC lanes.
            assert_eq!(got.reads.len(), 128 + 64 * (a + b) as usize);
            for (i, &value) in got.output.iter().enumerate() {
                let count = if i < 64 { a } else { b };
                let want = count
                    * if i % 64 < 32 {
                        1
                    } else {
                        i as u32 / 64 * 64 + 16
                    };
                assert_eq!(value, want, "a={a} b={b} lane={i}");
            }
        }
    }
}

#[test]
fn a_lane_varying_readlane_selector_cannot_choose_a_loop_exit() {
    let mut body = cases::counter(1, 12, false);
    let at = body.iter().position(|&w| w == 0x7E18_0501).unwrap();
    body.splice(at..at + 1, [0xD760_000C, 0x0002_0101]); // s12 = v_readlane(v1, v0)
    let e = lower_body(&body, 128, false)
        .err()
        .expect("no uniform scalar bound");
    assert!(e.to_string().contains(super::lockstep::LANE_SOURCE), "{e}");
}

#[test]
fn nested_loops_reset_the_inner_counter_and_keep_finished_waves_frozen() {
    for (a, b) in [(0, 12), (1, 0), (12, 1)] {
        let l = lower_body(&cases::nested(a, b), 128, false).unwrap();
        assert!(l.synchronized);
        for reverse in [false, true] {
            let got = l.values.as_ref().unwrap().run(128, &[], reverse);
            assert_eq!(got.writes, vec![1; 128]);
            for (i, &value) in got.output.iter().enumerate() {
                let count = if i < 64 { a } else { b };
                let sum = count * (count + 1) / 2 * (i as u32 / 64 * 64 + 8);
                assert_eq!(value, count + if i % 64 < 32 { 0 } else { sum }, "lane {i}");
            }
        }
    }
}

#[test]
fn mixed_numeric_and_lane_mask_backedges_keep_the_exact_refusal() {
    for threshold in [127, 126, 63, 50, 0] {
        let e = lower_body(&cases::exec_mask_scan(threshold), 128, false)
            .err()
            .expect("mask representation not proved");
        assert!(
            e.to_string()
                .contains("a register joining a lane mask and data, read as data"),
            "{e}"
        );
    }
}

#[test]
fn exec_loops_keep_whole_wave_permutes_outside_the_lane_guards() {
    let l = lower_body(cases::EXEC_PERMUTE, 128, false).unwrap();
    assert!(l.synchronized);
    for reverse in [false, true] {
        let got = l.values.as_ref().unwrap().run(128, &[], reverse);
        assert_eq!(got.writes, vec![1; 128]);
        for (i, &value) in got.output.iter().enumerate() {
            assert_eq!(value, (i + i / 64 * 64 + 24) as u32, "lane {i}");
        }
    }
}

#[test]
fn elected_atomics_and_their_first_lane_results_run_only_on_live_iterations() {
    for (a, b) in [(0, 12), (1, 0), (12, 1)] {
        let l = lower_body(&cases::atomic(a, b, true), 128, false).unwrap();
        assert!(l.synchronized);
        for reverse in [false, true] {
            let got = l.values.as_ref().unwrap().run(128, &[], reverse);
            assert_eq!(got.writes, vec![1; 128]);
            for (wave, count) in [a, b].into_iter().enumerate() {
                let old: Vec<_> = got
                    .atomics
                    .iter()
                    .filter(|x| x.0 == 128 + wave)
                    .map(|x| x.1)
                    .collect();
                assert_eq!(old, (0..count).collect::<Vec<_>>());
            }
            assert_eq!(got.atomics.len(), (a + b) as usize);
            for (i, &value) in got.output.iter().enumerate() {
                let count = if i < 64 { a } else { b };
                let sum = count * (i as u32 / 64 * 64 + 8) + count * count.saturating_sub(1) / 2;
                assert_eq!(value, count + if i % 64 < 32 { 0 } else { sum }, "lane {i}");
            }
        }
    }
}

#[test]
fn a_decoder_proven_far_jump_keeps_its_target_in_the_schedule() {
    let body = cases::counter(0, 12, false);
    let at = body.iter().position(|&w| w == 0xBF0A_0C09).unwrap();
    let mut jumped = body.clone();
    jumped.splice(
        at..at,
        [
            0xBE92_1F00, // getpc s[18:19]
            0x8012_FF12,
            20,
            0x8213_8013,
            0xBE80_2012, // setpc s[18:19], proved to target the next real instruction
            0xFFFF_FFFF, // skipped non-code
        ],
    );
    let l = lower_body(&jumped, 128, false).unwrap();
    let reference = lower_body(&body, 128, false).unwrap();
    assert!(l.synchronized);
    assert_eq!(
        l.values.as_ref().unwrap().run(128, &[], true).output,
        reference
            .values
            .as_ref()
            .unwrap()
            .run(128, &[], true)
            .output
    );
}

#[test]
fn coordinator_fixtures_lower_with_the_native_root_constants_and_memory_layout() {
    use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
    for body in [
        cases::counter(0, 12, false),
        cases::counter(12, 12, true),
        cases::hoisted_counter(0, 12),
        cases::idle_wave(),
        cases::mask_scan(1, u32::MAX),
        cases::nested(0, 12),
        cases::EXEC_PERMUTE.to_vec(),
        cases::range(false),
        cases::atomic(0, 12, false),
    ] {
        let mut words = body;
        words.extend([0x3402_0082, 0xDC70_8000, 0x0002_0201, 0xBF81_0000]);
        let native = NativeProgram {
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
                rsrc2: 5 << 1,
                threads: [128, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let l = super::console_tests::lower_native(&native).unwrap();
        assert!(l.synchronized && l.exchanges);
    }
}

#[test]
fn mutually_dependent_loop_phis_use_the_previous_iteration_values() {
    for (a, b) in [(0, 12), (1, 2), (12, 1)] {
        let mut body = cases::counter(a, b, false);
        let cmp = body.iter().position(|&w| w == 0xBF0A_0C09).unwrap();
        body.splice(cmp..cmp, [0xBE94_0381, 0xBE95_0382]); // s20=1, s21=2
        let add = body.iter().position(|&w| w == 0x8009_8109).unwrap();
        body.splice(add..add, [0xBE90_0314, 0xBE94_0315, 0xBE95_0310]); // simultaneous swap
        for w in &mut body {
            if *w == 0xBF84_0006 {
                *w = 0xBF84_0009;
            }
            if *w == 0xBF85_FFFA {
                *w = 0xBF85_FFF7;
            }
        }
        body.extend([0x9316_8A14, 0x8016_1516, 0x4A04_0416]); // v2 += 10*s20+s21
        let l = lower_body(&body, 128, false).unwrap();
        let got = l.values.as_ref().unwrap().run(128, &[], true);
        for (i, &value) in got.output.iter().enumerate() {
            let count = if i < 64 { a } else { b };
            let sum = if i % 64 < 32 {
                0
            } else {
                count * (i as u32 / 64 * 64 + 8)
            };
            assert_eq!(
                value,
                count + sum + if count % 2 == 0 { 12 } else { 21 },
                "lane {i}"
            );
        }
    }
}
