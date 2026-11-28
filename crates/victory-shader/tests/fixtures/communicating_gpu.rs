// Coordinator: include this from victory-gpu/src/native/split_wave_tests.rs.
// Run on arm64: cargo test -j4 --offline --release -p victory-gpu communicating_multi_wave -- --nocapture
mod communicating_multi_wave {
    use super::{Fixture, VA_A, dispatch, halves, run_io, shaders, title, translate_io};
    mod cases {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../victory-shader/tests/fixtures/communicating_cases.rs"
        ));
    }
    fn words(body: &[u32]) -> Vec<u32> {
        let mut words = body.to_vec();
        words.extend([0x3402_0082, 0xDC70_8000, 0x0002_0201, 0xBF81_0000]);
        words
    }
    fn run(body: &[u32], threads: u32, bytes: u32) -> Vec<u32> {
        let got = run_io(&words(body), bytes.div_ceil(512) << 15, threads, 0);
        if got.is_empty() {
            eprintln!("SKIPPED: Metal/MSC unavailable");
        }
        got
    }

    #[test]
    fn partial_exec_producers_and_cross_wave_consumers_obey_title_phases() {
        for loops in [false, true] {
            for threads in [128, 256, 1024] {
                let got = run(&cases::producer(loops), threads, threads * 4);
                if got.is_empty() {
                    return;
                }
                for (id, &value) in got.iter().enumerate() {
                    let other = id as u32 ^ 64;
                    let trips = if loops { other / 64 % 4 } else { 1 };
                    let want = if other % 64 < 32 {
                        0
                    } else {
                        trips * (other / 64 * 64 + 41)
                    };
                    assert_eq!(value, want, "threads={threads}, loops={loops}, lane={id}");
                }
            }
        }
    }

    fn check_tickets(threads: u32, iterations: u32) -> bool {
        let got = run(&cases::elected(iterations), threads, 512);
        if got.is_empty() {
            return false;
        }
        let count = threads / 64 * iterations;
        let mut tickets = Vec::new();
        for wave in got.chunks(64) {
            assert!(
                wave.iter().all(|&x| x == wave[0]),
                "empty EXEC half receives the broadcast"
            );
            assert_eq!(wave[0] >> 16, count, "exact atomic effect count");
            if iterations == 2 {
                assert!((wave[0] >> 8 & 255) < (wave[0] & 255), "per-wave order");
                tickets.push(wave[0] >> 8 & 255);
            }
            tickets.push(wave[0] & 255);
        }
        tickets.sort_unstable();
        assert_eq!(
            tickets,
            (0..count).collect::<Vec<_>>(),
            "all allocations distinct"
        );
        true
    }

    #[test]
    fn full_exec_election_executes_once_per_console_wave() {
        for threads in [128, 256, 1024] {
            if !check_tickets(threads, 1) {
                return;
            }
        }
    }

    #[test]
    fn two_console_waves_interleave_allocations_without_replaying_them() {
        // Accept any inter-wave atomic order, but each wave's two tickets must
        // increase and the counter must contain exactly four allocations.
        check_tickets(128, 2);
    }

    #[test]
    fn each_lane_leaves_its_loop_at_a_different_iteration_in_both_halves() {
        for threads in [128, 1024] {
            let got = run(&cases::lane_loop(), threads, threads * 4);
            if got.is_empty() {
                return;
            }
            for (id, &value) in got.iter().enumerate() {
                assert_eq!(
                    value,
                    ((id ^ 32) % 64 / 8 + id / 64 * 64 + 40) as u32,
                    "lane={id}"
                );
            }
        }
    }

    #[test]
    fn per_lane_buffer_add_and_unsigned_min_keep_all_results() {
        let Some(mut t) = title() else {
            eprintln!("SKIPPED: Metal/MSC unavailable");
            return;
        };
        let f = Fixture::new(&t);
        let [input_lo, input_hi] = halves(VA_A);
        let [output_lo, output_hi] = halves(VA_A + 0x4_0000);
        for threads in [128, 1024] {
            let input: Vec<_> = (0..threads).map(|id| 1000 + id).collect();
            f.put(0x2_0800, &input);
            // The CPU fixture also checks descriptor-form buffer_atomic_*.
            // Here a root address uses global_atomic_*; both lower to the same
            // per-thread DXIL buffer atomics.
            let cs = translate_io(
                &words(&cases::buffer_atomics(false)),
                5,
                0,
                threads,
                shaders::options(),
            );
            dispatch(
                &mut t,
                &cs,
                &[input_lo, input_hi, output_lo, output_hi, 0],
                false,
            );
            assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
            let got = f.get(0x6_0000, threads as usize);
            let memory = f.get(0x2_0800, threads as usize);
            for (id, &value) in got.iter().enumerate() {
                let base = 1000 + id as u32 / 64 * 64 + 33;
                let want = base
                    + if id % 64 > 32 {
                        1001 + id as u32 + id as u32 / 64 * 64 + 8
                    } else {
                        0
                    };
                assert_eq!(value, want, "lane {id}");
                assert_eq!(memory[id], if id % 64 > 32 { 7 } else { 1000 + id as u32 });
            }
        }
    }
    fn single_atomic(descriptor: bool, minimum: bool) {
        use crate::native::{list::NativeList, title::Pipe};
        use victory_backend::resources::HeapKind;
        let Some(mut t) = title() else {
            eprintln!("SKIPPED: Metal/MSC unavailable");
            return;
        };
        let f = Fixture::new(&t);
        // One heap entry for both sizes: a raw buffer UAV of 4096 dwords at
        // `VA_A`. The records outlive every dispatch that reads them.
        let records = [0u32; 8];
        let slot = records.as_ptr() as u64;
        let heap = t.host.heap(HeapKind::Resource, slot, 1).unwrap();
        if descriptor {
            let mut resource = [0u32; 14];
            resource[0] = 1;
            let mut view = [0u32; 16];
            (view[1], view[4], view[8]) = (1, 4096, 1);
            [view[10], view[11]] = halves(VA_A);
            crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, slot);
            assert!(t.host.mirror(slot).unwrap());
        }
        for threads in [128, 1024] {
            let seed = if minimum { 0x8000_0000u32 } else { 1000 };
            let input: Vec<_> = (0..threads).map(|id| seed + id).collect();
            f.put(0x2_0800, &input);
            let words = cases::single_buffer_atomic(descriptor, minimum);
            let output = if descriptor {
                // Same raw UAV binding and DXIL atomic form as heap_tests'
                // passing buffer_atomic_sub fixture.
                let mut uav = [0u8; 32];
                uav[0] = 0x41;
                uav[4..6].copy_from_slice(&[3, 3]);
                uav[16] = 1;
                uav[20..23].copy_from_slice(&[4, 4, 0]);
                let container = crate::pipeline::test_compute_container_of(&words, &[uav], 7, threads, true);
                let cs = shaders::translate(&container).unwrap();
                let Pipe::Compute(pso) = t.compute_pipe(&cs, "single descriptor atomic") else {
                    panic!("{:?}", t.shaders);
                };
                let mut l = NativeList::default();
                l.set_heaps(Some(&heap), None);
                l.set_compute_pipeline(&pso);
                l.set_compute_root_table(0, &heap, 0);
                l.dispatch(1, 1, 1).unwrap();
                super::super::run(&t.host, &[&l]);
                0x2_2000
            } else {
                let cs = translate_io(&words, 5, 0, threads, shaders::options());
                let [a, b] = halves(VA_A);
                let [c, d] = halves(VA_A + 0x4_0000);
                dispatch(&mut t, &cs, &[a, b, c, d, 0], false);
                0x6_0000
            };
            assert_eq!(f.get(output, threads as usize), input, "returned old values");
            let want: Vec<_> = input.iter().map(|&x| if minimum { 7 } else { x + 1 }).collect();
            assert_eq!(f.get(0x2_0800, threads as usize), want, "atomic effects");
        }
    }

    #[test]
    fn single_root_atomic_add() { single_atomic(false, false); }
    #[test]
    fn single_root_atomic_unsigned_min() { single_atomic(false, true); }
    #[test]
    fn single_descriptor_atomic_add() { single_atomic(true, false); }
    #[test]
    fn single_descriptor_atomic_unsigned_min() { single_atomic(true, true); }

}
