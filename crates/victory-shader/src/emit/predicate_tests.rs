//! Multi-wave if-conversion: all exchange barriers must postdominate entry, even though
//! the console waves take different branches. Value checks live in the coordinator fixture.
use super::console_tests::lower_body;
mod cases {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/predicated_cases.rs"
    ));
}

#[test]
fn named_lane_reads_and_scalar_joins_are_predicated_across_console_waves() {
    let l = lower_body(cases::READ_LANE, 128, false).unwrap();
    assert!(l.predicated && l.exchanges && l.uniform_barriers);
    assert!(l.barriers >= 4);
    assert_eq!(l.wave_size, Some(32));
}

#[test]
fn first_lane_reads_use_full_exec_including_an_empty_lower_half() {
    for threads in [96, 128, 256] {
        let l = lower_body(cases::FIRST_LANE, threads, false).unwrap();
        assert!(l.predicated && l.exchanges && l.uniform_barriers);
        assert!(l.barriers >= 4);
    }
}

#[test]
fn mask_scans_and_counts_exchange_both_halves_under_partial_exec() {
    let l = lower_body(cases::MASK_SCAN, 128, false).unwrap();
    assert!(l.predicated && l.exchanges && l.uniform_barriers);
}

#[test]
fn unused_lds_allocation_does_not_change_the_emitted_program() {
    for body in [cases::LOOP, cases::READ_LANE] {
        let allocated = lower_body(body, 128, true).unwrap();
        let plain = lower_body(body, 128, false).unwrap();
        assert_eq!(allocated.dxil, plain.dxil);
    }
}

#[test]
fn exec_loops_use_the_group_schedule() {
    let l = lower_body(cases::LOOP, 128, false).unwrap();
    assert!(l.synchronized && l.exchanges);
    for reverse in [false, true] {
        let got = l.values.as_ref().unwrap().run(128, &[], reverse);
        assert!(got.barriers > 128);
        assert!(got.reads.is_empty());
        assert_eq!(got.writes, vec![1; 128]);
        for (i, &value) in got.output.iter().enumerate() {
            assert_eq!(value, (i + (i / 64) * 64 + 8) as u32, "lane {i}");
        }
    }
}

#[test]
fn joined_masks_select_both_words_on_the_console_edge() {
    let l = lower_body(cases::MASK_JOIN, 128, false).unwrap();
    assert!(l.predicated && l.exchanges && l.uniform_barriers);
}

#[test]
fn scalar_memory_is_not_read_by_a_wave_that_skips_the_arm() {
    let mut body = cases::READ_LANE.to_vec();
    body[4] = 0xBF88_0005;
    body.splice(
        7..7,
        [
            0xF420_0302,
            0xFA00_000C, // s_buffer_load_dword s12, s[4:7], 12
            0x8009_0C09, // s9 += s12
        ],
    );
    let l = lower_body(&body, 128, false).unwrap();
    assert!(l.predicated, "if-conversion must be used");
    assert!(l.uniform_barriers, "exchanges must remain outside resource guards");
    assert!(l.guarded_buffer_reads, "the scalar load must stay guarded");
}

#[test]
fn a_guarded_value_must_be_joined_before_use_after_the_guard() {
    use super::bitcode::{Attr, Bin, Term};
    use super::shader::Shader;
    for joined in [false, true] {
        let mut s = Shader::new(false);
        let (yes, join) = (s.m.new_block(), s.m.new_block());
        let cond = s.op(s.entry, "waveIsFirstLane", Attr::NoUnwind, s.t.i1, 110, &[]);
        s.m.terminate(s.entry, Term::CondBr(cond, yes, join));
        s.m.terminate(yes, Term::Br(join));
        let (one, two) = (s.m.i32(1), s.m.i32(2));
        let x = s.m.bin(yes, Bin::Add, one, two);
        let value = if joined {
            let phi = s.m.phi(join, s.t.i32);
            s.m.set_phi(phi, vec![(x, yes), (one, s.entry)]);
            phi
        } else {
            x
        };
        let z = s.m.i32(0);
        let col = s.m.i8(0);
        s.op(
            join,
            "storeOutput.i32",
            Attr::NoUnwind,
            s.t.void,
            5,
            &[z, z, col, value],
        );
        s.m.terminate(join, Term::Ret);
        assert_eq!(s.m.values_dominate_uses(), joined);
    }
}

#[test]
fn conditional_mode_changes_and_large_groups_keep_the_refusal() {
    let mut mode = cases::READ_LANE.to_vec();
    mode[4] = 0xBF88_0004; // also skip the mode write
    mode.splice(5..5, [0xBA80_0881, 3]); // s_setreg_imm32_b32: FP rounding
    for (body, threads) in [(mode.as_slice(), 128), (cases::READ_LANE, 512)] {
        let e = lower_body(body, threads, false)
            .err()
            .expect("outside the proof");
        assert!(
            e.to_string().contains(super::lower::DIVERGENT_EXCHANGE),
            "{e}"
        );
    }
}

#[test]
fn a_barrier_inside_a_guard_fails_the_final_control_flow_proof() {
    use super::bitcode::{Attr, Term};
    use super::shader::Shader;
    for guarded in [false, true] {
        let mut s = Shader::new(false);
        let (yes, no, join) = (s.m.new_block(), s.m.new_block(), s.m.new_block());
        let cond = s.op(s.entry, "waveIsFirstLane", Attr::NoUnwind, s.t.i1, 110, &[]);
        s.m.terminate(s.entry, Term::CondBr(cond, yes, no));
        s.m.terminate(yes, Term::Br(join));
        s.m.terminate(no, Term::Br(join));
        let mode = s.m.i32(9);
        s.op(
            if guarded { yes } else { join },
            "barrier",
            Attr::NoDuplicate,
            s.t.void,
            80,
            &[mode],
        );
        s.m.terminate(join, Term::Ret);
        assert_eq!(s.m.calls_on_all_paths("dx.op.barrier"), !guarded);
        s.m.terminate(join, Term::Br(s.entry));
        assert!(
            !s.m.calls_on_all_paths("dx.op.barrier"),
            "cycles need a separate proof"
        );
    }
}
