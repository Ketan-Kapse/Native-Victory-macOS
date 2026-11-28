//! Rectangle lists of described pipelines whose vertex shader takes stage-in attributes, on
//! Metal: hand-assembled programs (`victory_shader::native::synthetic::rect_vertex`), the
//! title's input layout, vertices in a pool, and the pictures against the same rectangles drawn
//! as two triangles each by the ordinary pipeline.

use super::addr_tests::{Fixture, VA_A, title};
use crate::native::list::NativeList;
use crate::native::shaders;
use crate::native::state::Graphics;
use crate::native::title::{Pipe, Title};
use victory_backend::resources::*;
use victory_metal::metal::Metal;
use victory_shader::native::synthetic;

const W: usize = 16;
/// Where `VA_A` is in the fixture's pool.
const AT: usize = 0x2_0000;

/// A described state: `POSITION` and `TEXCOORD`, two floats each, 16 bytes a vertex in slot 0;
/// one RGBA8 target; D3D12's defaults (back faces culled, clockwise is front).
fn state() -> Graphics {
    let names = [c"POSITION", c"TEXCOORD"];
    let mut elements = Vec::new();
    for (n, offset) in [(0, 0u32), (1, 8)] {
        let mut e = [0u8; 32];
        e[..8].copy_from_slice(&(names[n].as_ptr() as u64).to_le_bytes());
        // Semantic index 0, R32G32_FLOAT, slot 0, the offset, per vertex.
        for (k, w) in [0, 16, 0, offset, 0, 0].into_iter().enumerate() {
            e[8 + 4 * k..12 + 4 * k].copy_from_slice(&u32::to_le_bytes(w));
        }
        elements.extend(e);
    }
    let mut s = 12u32.to_le_bytes().to_vec();
    s.extend([0; 4]);
    s.extend((elements.as_ptr() as u64).to_le_bytes());
    s.extend(2u32.to_le_bytes());
    s.extend([0; 4]);
    let mut g = Graphics::default();
    // SAFETY: a whole input layout subobject over live elements and names.
    unsafe { g.read(12, s.as_ptr()) };
    g.with_targets(vec![Format::Rgba8Unorm], None)
}

type Vertex = [f32; 4];

/// The corners of rectangle `n` (0: target columns 2 to 5, 1: columns 10 to 13; rows 4 to 11):
/// top left, top right, bottom right, bottom left, each with its texture coordinate.
fn corners(n: usize) -> [Vertex; 4] {
    let (x0, x1) = if n == 0 { (-0.75, -0.25) } else { (0.25, 0.75) };
    [[x0, 0.5, 0.0, 0.0], [x1, 0.5, 1.0, 0.0], [x1, -0.5, 1.0, 1.0], [x0, -0.5, 0.0, 1.0]]
}

/// What the rectangles' pixels show: red and green the texture coordinate at the pixel's centre.
fn expected(x: usize, y: usize, rects: &[usize], blue: u8) -> [u8; 4] {
    for &n in rects {
        let left = if n == 0 { 2 } else { 10 };
        if (left..left + 4).contains(&x) && (4..12).contains(&y) {
            let u = (x - left) as f32 + 0.5;
            let v = (y - 4) as f32 + 0.5;
            return [(u / 4.0 * 255.0).round() as u8, (v / 8.0 * 255.0).round() as u8, blue, 255];
        }
    }
    [0; 4]
}

struct Rig {
    t: Title<Metal>,
    f: Fixture,
    strip: (std::rc::Rc<<Metal as GraphicsBackend>::RenderPipeline>, Option<std::rc::Rc<crate::native::list::Binding>>),
    plain: (std::rc::Rc<<Metal as GraphicsBackend>::RenderPipeline>, Option<std::rc::Rc<crate::native::list::Binding>>),
    color: std::rc::Rc<<Metal as ResourceBackend>::Texture>,
}

impl Rig {
    fn new() -> Option<Rig> {
        let mut t = title()?;
        let f = Fixture::new(&t);
        let vs = shaders::translate(&synthetic::rect_vertex()).unwrap();
        let rect = shaders::translate_rect(&synthetic::rect_vertex()).unwrap();
        let ps = shaders::translate(&synthetic::rect_pixel()).unwrap();
        let mut pipe = |vs, name: &str| match t.render_pipe(vs, Some(&ps), &state(), name) {
            Pipe::Render(pso, binding) => (pso, binding),
            _ => panic!("{name}: {:?}", t.shaders),
        };
        let (strip, plain) = (pipe(&rect, "rectangles"), pipe(&vs, "triangles"));
        assert!(strip.1.as_ref().is_some_and(|b| b.rect_strip), "the rectangle variant is drawn as strips");
        assert!(!plain.1.as_ref().is_some_and(|b| b.rect_strip));
        let mut desc = TextureDesc::d2(Format::Rgba8Unorm, W, W);
        desc.render_target = true;
        let color = t.host.texture(&desc).unwrap();
        Some(Rig { t, f, strip, plain, color })
    }

    /// The picture of `draw` over a cleared target, with the vertex buffer at `VA_A`.
    fn picture(&self, strip: bool, draw: impl FnOnce(&mut NativeList<Metal>)) -> Vec<u8> {
        let (pso, binding) = if strip { &self.strip } else { &self.plain };
        let mut l = NativeList::default();
        l.set_render_targets(std::slice::from_ref(&self.color), None);
        l.clear_render_target(&self.color, [0.0; 4]);
        l.set_graphics_pipeline(pso, if strip { Topology::TriangleStrip } else { Topology::Triangles });
        l.set_graphics_binding(binding.clone());
        l.set_vertex_buffer(0, VA_A);
        draw(&mut l);
        super::run(&self.t.host, &[&l]);
        let mut out = vec![0u8; W * W * 4];
        self.t.host.backend.read_texture(&self.color, 0, 0, &mut out, W * 4, 0);
        out
    }
}

fn px(out: &[u8], x: usize, y: usize) -> [u8; 4] {
    <[u8; 4]>::try_from(&out[(y * W + x) * 4..][..4]).unwrap()
}

fn near(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= 2)
}

/// Two rectangles of three stored corners each, in every order that keeps the list's clockwise
/// winding (the right angle first, second and last) and one counter-clockwise: the strips cover
/// exactly the rectangles, the texture coordinate is interpolated over the half the list left
/// out as over the half it gave, and the picture is the one the ordinary pipeline draws of the
/// same rectangles as two triangles each (so the faces culled are the same: none clockwise,
/// all counter-clockwise). Console vertex IDs count the list's vertices: the first rectangle's
/// are below 3 (blue 0), the second's are not (blue 255).
#[test]
fn stage_in_rectangle_lists_cover_their_rectangles_as_two_triangles_do() {
    let Some(r) = Rig::new() else { return };
    // Corner numbers ([`corners`]) of the three stored vertices, and whether they are clockwise.
    let orders = [([0, 1, 2], true), ([1, 2, 0], true), ([2, 0, 1], true), ([3, 0, 1], true), ([0, 2, 1], false)];
    for (order, front) in orders {
        // The rectangle list, then from vertex 6 the same rectangles as two triangles each: the
        // stored one, and the other two corners with the fourth, wound the same way.
        let mut vertices: Vec<Vertex> = Vec::new();
        for n in 0..2 {
            vertices.extend(order.map(|k| corners(n)[k]));
        }
        for n in 0..2 {
            let c = corners(n);
            let fourth = (0..4).find(|k| !order.contains(k)).unwrap();
            // The stored corner opposite the fourth (the right angle) is the one not next to it.
            let pivot = order.iter().position(|&k| k == (fourth + 2) % 4).unwrap();
            let (n1, n2) = (order[(pivot + 1) % 3], order[(pivot + 2) % 3]);
            vertices.extend(order.map(|k| c[k]));
            vertices.extend([c[n2], c[n1], c[fourth]]);
        }
        r.f.put(AT, &vertices);
        let rects = r.picture(true, |l| l.draw_instanced(4, 2, 0, 0).unwrap());
        let triangles = r.picture(false, |l| l.draw_instanced(12, 1, 6, 0).unwrap());
        let mut covered = 0;
        for y in 0..W {
            for x in 0..W {
                let (got, reference) = (px(&rects, x, y), px(&triangles, x, y));
                // Blue apart: the triangles' vertex IDs are not the list's.
                assert!(near([got[0], got[1], 0, got[3]], [reference[0], reference[1], 0, reference[3]]),
                    "{order:?} pixel ({x},{y}): {got:?}, two triangles draw {reference:?}");
                let blue = if x < 8 { 0 } else { 255 };
                let want = if front { expected(x, y, &[0, 1], blue) } else { [0; 4] };
                assert!(near(got, want), "{order:?} pixel ({x},{y}): {got:?}, expected {want:?}");
                covered += usize::from(got[3] != 0);
            }
        }
        assert_eq!(covered, if front { 64 } else { 0 }, "{order:?}");
    }
    assert_eq!(r.t.shaders.refusals(), 0, "{:?}", r.t.shaders.refused);
}

/// The draws the host records for such a list (`glue::draw`: four vertices, a rectangle an
/// instance, from rectangle `first / 3`; an indirect one as `DrawCall::Rects`, its record
/// rewritten on the GPU): a list that starts at its second rectangle draws that one alone, with
/// console vertex IDs counted from the draw's first vertex (blue 0), and an indirect list of
/// both draws both, directly and under a count that lets it run; a count that does not draws
/// nothing. The title's records are left as they were.
#[test]
fn stage_in_rectangle_lists_draw_from_a_first_vertex_and_from_gpu_records() {
    let Some(r) = Rig::new() else { return };
    let vertices: Vec<Vertex> = (0..2).flat_map(|n| [0, 1, 2].map(|k| corners(n)[k])).collect();
    r.f.put(AT, &vertices);
    let check = |out: &[u8], rects: &[usize], blue: [u8; 2], what: &str| {
        for y in 0..W {
            for x in 0..W {
                let want = expected(x, y, rects, blue[x / 8]);
                assert!(near(px(out, x, y), want), "{what} pixel ({x},{y}): {:?}, expected {want:?}", px(out, x, y));
            }
        }
    };
    // `Draw(3 vertices, first 3)`: the second rectangle, whose IDs are 0 to 2.
    let second = r.picture(true, |l| l.draw_instanced(4, 1, 0, 1).unwrap());
    check(&second, &[1], [0, 0], "first vertex 3");
    // Records: both rectangles; the second alone; then counts of 1 and 0.
    let (records, counts) = (AT + 0x1000, AT + 0x1100);
    r.f.put(records, &[6u32, 1, 0, 0, 3, 1, 3, 0]);
    r.f.put(counts, &[1u32, 0]);
    let va = |at: usize| VA_A + (at - AT) as u64;
    let both = r.picture(true, |l| l.draw_rects_indirect(va(records), None).unwrap());
    check(&both, &[0, 1], [0, 255], "indirect");
    let one = r.picture(true, |l| l.draw_rects_indirect(va(records + 16), None).unwrap());
    check(&one, &[1], [0, 0], "indirect from vertex 3");
    let counted = r.picture(true, |l| l.draw_rects_indirect(va(records), Some((va(counts), 0))).unwrap());
    check(&counted, &[0, 1], [0, 255], "counted");
    let denied = r.picture(true, |l| l.draw_rects_indirect(va(records), Some((va(counts + 4), 0))).unwrap());
    check(&denied, &[], [0, 0], "a count of 0");
    assert_eq!(r.f.get(records, 8), [6, 1, 0, 0, 3, 1, 3, 0], "the title's records are unchanged");
}
