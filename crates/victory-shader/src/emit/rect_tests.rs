//! Vertex-ID-only rectangles must complete outputs after evaluating the program.
use super::{Options, lower};
use crate::{Container, FourCC, io, ir, resource, structure};

mod program {
    use crate as victory_shader;
    include!("../../tests/programs/rect.rs");
}

#[test]
fn vertex_id_rectangles_complete_position_and_interpolants() {
    let bytes = program::vertex_container();
    let c = Container::parse(&bytes).unwrap();
    assert!(!super::graphics::is_guessed(&c));
    let native = crate::native_programs(c.part(FourCC::DXIL).unwrap().data)
        .unwrap()
        .remove(0);
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let parts = lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: super::signature(c.part(FourCC::OSG1).unwrap().data).unwrap(),
        prims: vec![],
        mesh_stride: 0, mesh_fit: Default::default(),
        view_marks: false,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        rect_list: true,
        copy_params: [0; 32],
        bvh_no_hit: false,
    };
    let expanded = lower::lower(&parts).unwrap();
    // Both xy position and uv are completed, as are the remaining float components. This
    // inspects the live graph feeding the final DXIL stores, not just an unused arithmetic op.
    assert_eq!(
        expanded.completed_outputs,
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3)
        ]
    );
    let ordinary = lower::lower(&lower::Parts {
        rect_list: false,
        ..parts
    })
    .unwrap();
    assert_ne!(ordinary.dxil, expanded.dxil);
    let inputs = super::signature(
        Container::parse(&expanded.dxil)
            .unwrap()
            .part(FourCC::ISG1)
            .unwrap()
            .data,
    )
    .unwrap();
    assert!(inputs.iter().any(|e| e.kind == 1), "vertex ID");
    assert!(inputs.iter().any(|e| e.kind == 2), "instance ID");
    assert!(inputs.iter().all(|e| e.kind != 0), "no vertex buffers");
    // Exercise the public container path used by described native pipelines as well.
    let translated = super::translate_with(
        &bytes,
        Options {
            rect_list: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(translated.stage, crate::HwStage::Vs);
    super::translate(&program::pixel_container()).unwrap();
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut translated.dxil.clone()).unwrap();
    }
}

/// A described vertex program with stage-in attributes is not refused as a rectangle list: it
/// declares each attribute it reads for the rectangle's three corners (the corner in the
/// semantic index's top bits), reads the vertex and the instance ID for the strip's corner and
/// the rectangle, and completes its outputs. Its ordinary translation declares them once.
#[test]
fn stage_in_rectangles_declare_three_corners_and_complete_their_outputs() {
    let bytes = crate::native::synthetic::rect_vertex();
    assert!(!super::graphics::is_guessed(&Container::parse(&bytes).unwrap()));
    let inputs = |dxil: &[u8]| {
        super::signature(Container::parse(dxil).unwrap().part(FourCC::ISG1).unwrap().data).unwrap()
    };
    let ordinary = super::translate(&bytes).unwrap();
    let names = |dxil: &[u8]| -> Vec<(String, u32, u8)> {
        inputs(dxil).iter().map(|e| (e.name.clone(), e.index, e.kind)).collect()
    };
    assert_eq!(names(&ordinary.dxil), [("POSITION".into(), 0, 0), ("TEXCOORD".into(), 0, 0), ("SV_VertexID".into(), 0, 1)]);
    let rect = super::translate_with(&bytes, Options { rect_list: true, ..Default::default() }).unwrap();
    let corner = |c: u32| c << super::RECT_CORNER_SHIFT;
    assert_eq!(names(&rect.dxil), [
        ("POSITION".into(), 0, 0), ("TEXCOORD".into(), 0, 0), ("SV_VertexID".into(), 0, 1), ("SV_InstanceID".into(), 0, 2),
        ("POSITION".into(), corner(1), 0), ("POSITION".into(), corner(2), 0),
        ("TEXCOORD".into(), corner(1), 0), ("TEXCOORD".into(), corner(2), 0),
    ]);
    // No two elements share a register.
    let mut rows: Vec<_> = inputs(&rect.dxil).iter().map(|e| e.row).collect();
    rows.sort();
    rows.dedup();
    assert_eq!(rows.len(), 8);
    super::translate(&crate::native::synthetic::rect_pixel()).unwrap();
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut rect.dxil.clone()).unwrap();
    }
}

/// A described vertex and a described pixel program that read title memory at an address from
/// their root constants both find it through the page table (`HostShader::memory`), with nothing
/// to mark under the native host's zero-copy option.
#[test]
fn described_graphics_programs_read_title_memory_through_the_page_table() {
    let (vertex, pixel) = crate::native::synthetic::memory_pair();
    let options = super::Options { wave32_host: true, zero_copy_memory: true, ..Default::default() };
    for (container, stage) in [(&vertex, crate::HwStage::Vs), (&pixel, crate::HwStage::Ps)] {
        assert!(!super::graphics::is_guessed(&Container::parse(container).unwrap()));
        let shader = super::translate_with(container, options).unwrap();
        assert_eq!(shader.stage, stage);
        assert!(shader.memory && !shader.memory_writes && !shader.view_marks, "{stage:?}");
        assert_eq!(shader.user_data_dwords(), 4, "{stage:?}: the address, rounded up to four dwords");
    }
}
