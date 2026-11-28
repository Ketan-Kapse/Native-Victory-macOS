// Coordinator: append this test-only include to native/split_wave_tests.rs:
// include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../victory-shader/tests/fixtures/synchronized_gpu.rs"));
// Run on arm64: cargo test -j4 --offline --release -p victory-gpu synchronized_multi_wave -- --nocapture
mod synchronized_multi_wave {
    use super::{Fixture, VA_A, dispatch, halves, run_io, shaders, title, translate_io};
    #[allow(dead_code)] // exec_mask_scan is an explicit CPU lowering refusal test.
    mod cases {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../victory-shader/tests/fixtures/synchronized_cases.rs"
        ));
    }

    fn words(body: &[u32]) -> Vec<u32> {
        let mut w = body.to_vec();
        w.extend([0x3402_0082, 0xDC70_8000, 0x0002_0201, 0xBF81_0000]);
        w
    }

    fn check(got: &[u32], a: u32, b: u32, sum: impl Fn(u32, u32) -> u32) {
        for (i, &x) in got.iter().enumerate() {
            let wave = i as u32 / 64;
            let count = if wave == 0 { a } else { b };
            let want = count + if i % 64 < 32 { 0 } else { sum(wave, count) };
            assert_eq!(x, want, "a={a} b={b} lane={i}");
        }
    }

    #[test]
    fn zero_one_and_twelve_trip_counters_with_an_inactive_lower_half() {
        for (a, b, fixed) in [
            (0, 12, false),
            (1, 0, false),
            (12, 1, false),
            (12, 12, true),
        ] {
            for threads in [112, 128, 256] {
                let got = run_io(&words(&cases::counter(a, b, fixed)), 0, threads, 0);
                if got.is_empty() {
                    eprintln!("SKIPPED: Metal/MSC unavailable");
                    return;
                }
                check(&got, a, b, |wave, count| count * (wave * 64 + 8));
            }
        }
    }

    #[test]
    fn hoisted_exchange_survives_different_trip_counts_and_scratch_reuse() {
        for (a, b) in [(0, 0), (0, 12), (1, 0), (12, 1)] {
            for threads in [112, 128, 256] {
                let got = run_io(&words(&cases::hoisted_counter(a, b)), 0, threads, 0);
                if got.is_empty() {
                    eprintln!("SKIPPED: Metal/MSC unavailable");
                    return;
                }
                check(&got, a, b, |wave, count| count * (wave * 64 + 15));
            }
        }
    }

    #[test]
    fn an_idle_console_wave_waits_for_its_exec_shrinking_neighbour() {
        let got = run_io(&words(&cases::idle_wave()), 0, 128, 0);
        if got.is_empty() {
            eprintln!("SKIPPED: Metal/MSC unavailable");
            return;
        }
        for (id, &x) in got.iter().enumerate() {
            assert_eq!(x, if id < 64 { 0 } else { (id + 88) as u32 });
        }
    }

    #[test]
    fn mask_scans_have_zero_one_and_thirty_two_trips() {
        for (a, b) in [
            (0u32, u32::MAX),
            (1, 0),
            (u32::MAX, 1),
            (0x80000001, 0x55555555),
        ] {
            let got = run_io(&words(&cases::mask_scan(a, b)), 0, 128, 0);
            if got.is_empty() {
                eprintln!("SKIPPED: Metal/MSC unavailable");
                return;
            }
            check(&got, a.count_ones(), b.count_ones(), |wave, _| {
                let mask = if wave == 0 { a } else { b };
                (0..32)
                    .filter(|bit| mask & (1 << bit) != 0)
                    .map(|bit| wave * 64 + bit)
                    .sum()
            });
        }
    }

    #[test]
    fn nested_ranges_reset_the_inner_count() {
        for (a, b) in [(0, 12), (1, 0), (12, 1)] {
            let got = run_io(&words(&cases::nested(a, b)), 0, 128, 0);
            if got.is_empty() {
                eprintln!("SKIPPED: Metal/MSC unavailable");
                return;
            }
            check(&got, a, b, |wave, count| {
                count * (count + 1) / 2 * (wave * 64 + 8)
            });
        }
    }

    #[test]
    fn whole_wave_permutes_and_lane_reads_inside_an_exec_loop() {
        let got = run_io(&words(cases::EXEC_PERMUTE), 0, 128, 0);
        if got.is_empty() {
            eprintln!("SKIPPED: Metal/MSC unavailable");
            return;
        }
        for (i, &x) in got.iter().enumerate() {
            assert_eq!(x, (i + i / 64 * 64 + 24) as u32, "lane {i}");
        }
    }

    #[test]
    fn scalar_memory_bounds_and_loads_in_live_iterations() {
        let Some(mut t) = title() else {
            eprintln!("SKIPPED: Metal/MSC unavailable");
            return;
        };
        let f = Fixture::new(&t);
        let cs = translate_io(&words(&cases::range(false)), 5, 0, 128, shaders::options());
        let [input_lo, input_hi] = halves(VA_A);
        let [output_lo, output_hi] = halves(VA_A + 0x4_0000);
        for (a, b) in [(0, 0), (0, 12), (1, 0), (12, 1)] {
            f.put(0x2_0000, &[a, b, 7]);
            dispatch(
                &mut t,
                &cs,
                &[input_lo, input_hi, output_lo, output_hi, 0],
                false,
            );
            assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
            check(&f.get(0x6_0000, 128), a, b, |wave, count| {
                count * (wave * 64 + 15)
            });
        }
    }

    #[test]
    fn elected_atomic_effects_and_returned_values_are_not_replayed() {
        let Some(mut t) = title() else {
            eprintln!("SKIPPED: Metal/MSC unavailable");
            return;
        };
        let f = Fixture::new(&t);
        let [input_lo, input_hi] = halves(VA_A);
        let [output_lo, output_hi] = halves(VA_A + 0x4_0000);
        for (a, b) in [(0, 12), (1, 0), (12, 1)] {
            f.put(0x2_0200, &[0, 0]);
            let cs = translate_io(
                &words(&cases::atomic(a, b, false)),
                5,
                0,
                128,
                shaders::options(),
            );
            dispatch(
                &mut t,
                &cs,
                &[input_lo, input_hi, output_lo, output_hi, 0],
                false,
            );
            assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
            assert_eq!(f.get(0x2_0200, 2), [a, b]);
            check(&f.get(0x6_0000, 128), a, b, |wave, count| {
                count * (wave * 64 + 8) + count * count.saturating_sub(1) / 2
            });
        }
    }
}
