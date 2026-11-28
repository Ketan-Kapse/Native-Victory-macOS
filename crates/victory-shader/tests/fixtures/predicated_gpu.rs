// Coordinator: include this file at the end of native/split_wave_tests.rs with:
// include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../victory-shader/tests/fixtures/predicated_gpu.rs"));
// Uses the existing test-only run_io harness; no production GPU changes are needed.
mod predicated_multi_wave {
    use super::run_io;
    #[allow(dead_code)] // LOOP is a lowering-only refusal test.
    mod cases {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../victory-shader/tests/fixtures/predicated_cases.rs"
        ));
    }
    fn run(body: &[u32], threads: u32) -> Vec<u32> {
        let mut words = body.to_vec();
        words.extend([
            0x3402_0082, // v1 = 4 * v0
            0xDC70_8000,
            0x0002_0201, // global_store_dword v1, v2, s[2:3]
            0xBF81_0000,
        ]);
        run_io(&words, 0, threads, 0)
    }
    #[test]
    fn a_named_lane_from_an_inactive_half_survives_the_scalar_join() {
        let got = run(cases::READ_LANE, 128);
        if got.is_empty() {
            return;
        }
        assert_eq!(&got[..64], &[40; 64]);
        assert_eq!(&got[64..], &[77; 64]);
    }
    #[test]
    fn first_lane_and_count_use_the_full_exec_of_each_console_wave() {
        for threads in [96, 128, 256] {
            let got = run(cases::FIRST_LANE, threads);
            if got.is_empty() {
                return;
            }
            for (i, &x) in got.iter().enumerate() {
                let wave = i as u32 / 64;
                let want = if wave == 0 {
                    51 + 13
                } else {
                    (wave * 64 + 64).min(threads)
                } + wave * 64
                    + 8;
                assert_eq!(x, want, "threads={threads}, lane={i}");
            }
        }
    }
    #[test]
    fn scans_and_counts_include_an_empty_lower_half_and_a_partial_group() {
        for threads in [96, 128] {
            let got = run(cases::MASK_SCAN, threads);
            if got.is_empty() {
                return;
            }
            for (i, &x) in got.iter().enumerate() {
                let want = if i < 64 {
                    51 + 13 + 8
                } else {
                    threads - 64 + 72
                };
                assert_eq!(x, want, "threads={threads}, lane={i}");
            }
        }
    }
    #[test]
    fn a_mask_join_keeps_the_high_word_and_the_waves_edge() {
        for threads in [96, 128] {
            let got = run(cases::MASK_JOIN, threads);
            if got.is_empty() {
                return;
            }
            for (i, &x) in got.iter().enumerate() {
                let want = if i < 64 { 8 + 13 } else { 72 + threads - 64 };
                assert_eq!(x, want, "threads={threads}, lane={i}");
            }
        }
    }
}
