//! Mesh programs over Metal's mesh output limit, linked to their pixel programs and made to fit
//! (`translate_fitted`). Our own programs (`native::synthetic`); the pictures they draw are
//! compared on Metal in `victory-gpu` (`native::mesh_tests`).

use super::{Fitted, MESH_SPLIT_PASSES, METAL_MESH_OUTPUT_BYTES, Options, mesh_footprint, metal_mesh_bytes};
use crate::native::synthetic;
use crate::native::HwStage;

fn options() -> Options {
    Options { wave32_host: true, zero_copy_memory: true, ..Default::default() }
}

/// A program that fits is the one `translate_linked` makes, byte for byte.
#[test]
fn a_linked_mesh_within_the_limit_is_left_as_it_is() {
    for split in [false, true] {
        let (ms, ps) = synthetic::grid_pair(11, split, 64);
        let (linked, pixel) = super::translate_linked(&ms, Some(&ps), options()).unwrap();
        let fitted = super::translate_fitted(&ms, Some(&ps), options()).unwrap();
        assert_eq!(fitted.fit, Fitted::AsLinked);
        assert_eq!(fitted.passes.len(), 1);
        assert_eq!(fitted.passes[0].dxil, linked.dxil);
        assert_eq!(fitted.pixel.unwrap().dxil, pixel.unwrap().dxil);
        assert!(metal_mesh_bytes(&linked.dxil).unwrap() <= METAL_MESH_OUTPUT_BYTES);
    }
}

/// Eleven rows of a `float3` and a `float` are 280 bytes a vertex as declared (each `float3`
/// starts at a multiple of 12, so every pair takes 24 bytes) and 192 in the tightest order (the
/// position, the floats, the float3s: no padding at all).
#[test]
fn outputs_over_the_limit_through_padding_are_reordered() {
    let (ms, ps) = synthetic::grid_pair(11, true, 128);
    let (linked, pixel) = super::translate_linked(&ms, Some(&ps), options()).unwrap();
    assert_eq!(metal_mesh_bytes(&linked.dxil), Some(128 * 280));
    let fitted = super::translate_fitted(&ms, Some(&ps), options()).unwrap();
    assert_eq!(fitted.fit, Fitted::Reordered { from: 128 * 280, bytes: 128 * 192 });
    assert_eq!(fitted.passes.len(), 1);
    assert_eq!(fitted.passes[0].stage, HwStage::Ms);
    assert_eq!(metal_mesh_bytes(&fitted.passes[0].dxil), Some(128 * 192));
    // The stages link by semantic: the pixel program is the linked one.
    assert_eq!(fitted.pixel.unwrap().dxil, pixel.unwrap().dxil);
    // Every output is still there, each once: the position, then the floats, then the float3s.
    let outputs = |dxil: &[u8]| -> Vec<String> {
        let text = super::mesh_outputs_text(dxil);
        text.split(" | ").nth(1).unwrap().split(' ').map(str::to_owned).collect()
    };
    let (was, now) = (outputs(&linked.dxil), outputs(&fitted.passes[0].dxil));
    assert_eq!(now.len(), 23, "{now:?}");
    assert_eq!(now[0], "SV_Position0.xyzw");
    assert!(now[1..12].iter().all(|n| n.ends_with(".w")), "{now:?}");
    assert!(now[12..].iter().all(|n| n.ends_with(".xyz")), "{now:?}");
    let sorted = |mut names: Vec<String>| {
        names.sort();
        names
    };
    assert_eq!(sorted(was), sorted(now));
}

/// Sixteen `float4` rows are 272 bytes a vertex in any order: the program is drawn in four
/// passes of 120 vertices. Sixteen rows of a `float3` and a `float` are 400 bytes as declared and
/// 272 reordered: four passes of 120 as well.
#[test]
fn outputs_over_the_limit_in_any_order_are_split_in_passes() {
    for (split, from, excluded, vertex) in [(false, 128 * 272, 8, 272), (true, 128 * 400, 8, 272)] {
        let (ms, ps) = synthetic::grid_pair(16, split, 128);
        let fitted = super::translate_fitted(&ms, Some(&ps), options()).unwrap();
        let bytes = (128 - u32::from(excluded)) * vertex;
        assert_eq!(fitted.fit, Fitted::Split { from, excluded, bytes });
        assert_eq!(fitted.passes.len(), usize::from(MESH_SPLIT_PASSES));
        for pass in &fitted.passes {
            let f = mesh_footprint(&pass.dxil).unwrap();
            assert_eq!((f.vertices, f.primitives, f.vertex_bytes), (128 - u32::from(excluded), 64, vertex));
            assert!(f.bytes().unwrap() <= METAL_MESH_OUTPUT_BYTES);
        }
        // The passes are different programs with the same outputs.
        let text = |n: usize| super::mesh_outputs_text(&fitted.passes[n].dxil);
        for n in 1..fitted.passes.len() {
            assert_eq!(text(n), text(0));
            assert_ne!(fitted.passes[n].dxil, fitted.passes[0].dxil);
        }
    }
}

/// A mesh program keeps the outputs its pixel program reads, so its translation, its order and
/// its passes are per pixel partner: here one partner reads every row (split in passes) and
/// one reads four rows (fits as linked).
#[test]
fn a_mesh_program_is_fitted_for_each_pixel_program_it_is_linked_to() {
    let (ms, all) = synthetic::grid_pair(16, false, 128);
    let few = synthetic::pixel(&synthetic::weighted_pixel_code(4), 4, &synthetic::row_elements(4, false, false));
    let with_all = super::translate_fitted(&ms, Some(&all), options()).unwrap();
    let with_few = super::translate_fitted(&ms, Some(&few), options()).unwrap();
    assert!(matches!(with_all.fit, Fitted::Split { .. }));
    assert_eq!(with_few.fit, Fitted::AsLinked);
    assert_eq!(metal_mesh_bytes(&with_few.passes[0].dxil), Some(128 * (16 + 4 * 16)));
    assert_ne!(with_few.passes[0].dxil, with_all.passes[0].dxil);
    // Without a pixel program (depth only) nothing but the position is kept.
    let depth_only = super::translate_fitted(&ms, None, options()).unwrap();
    assert_eq!(depth_only.fit, Fitted::AsLinked);
    assert_eq!(metal_mesh_bytes(&depth_only.passes[0].dxil), Some(128 * 16));
}

/// Outputs too large for four passes of three quarters of the vertices stay over the limit,
/// and say so: the host refuses the pipeline.
#[test]
fn outputs_too_large_for_passes_are_reported() {
    // 22 float4 rows: 368 bytes a vertex, and 96 vertices of them are 35,328 bytes.
    let rows = 22;
    let (ms, ps) = synthetic::grid_pair(rows, false, 128);
    let fitted = super::translate_fitted(&ms, Some(&ps), options()).unwrap();
    assert_eq!(fitted.fit, Fitted::TooLarge { from: 128 * 368, bytes: 128 * 368 });
    assert_eq!(fitted.passes.len(), 1);
    assert!(metal_mesh_bytes(&fitted.passes[0].dxil).unwrap() > METAL_MESH_OUTPUT_BYTES);
}

/// The triangles of the synthetic grid: each is drawn by exactly one pass of a split (the first
/// whose block of vertices holds none of its corners), and with the blocks of 8 vertices of the
/// programs above every pass draws some: the pictures compared on Metal exercise all four.
#[test]
fn every_triangle_of_the_grid_is_drawn_by_one_pass_and_every_pass_draws() {
    for excluded in [8u32] {
        let mut drawn = [0; MESH_SPLIT_PASSES as usize];
        for n in 0..synthetic::GRID_TRIANGLES {
            let corners = synthetic::grid_triangle(n);
            assert!(corners.iter().all(|&c| c < 64));
            let free: Vec<u32> = (0..u32::from(MESH_SPLIT_PASSES))
                .filter(|pass| corners.iter().all(|c| c / excluded != *pass))
                .collect();
            assert!(!free.is_empty(), "three corners cannot be in four blocks");
            drawn[free[0] as usize] += 1;
        }
        assert!(drawn.iter().all(|&n| n > 0), "{excluded}: {drawn:?}");
        assert_eq!(drawn.iter().sum::<u32>(), synthetic::GRID_TRIANGLES);
    }
}
