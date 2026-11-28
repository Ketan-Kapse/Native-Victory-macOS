//! Metal mesh output storage, and how a mesh program is made to fit it.
//!
//! Metal refuses a mesh pipeline whose outputs take more than 32,768 bytes a thread group
//! ("Total mesh size (N) exceeds the maximum mesh size allowed (32768)"). What it counts, as
//! measured by building pipelines of our own shaders on Metal (`docs/native-host.md`, "Mesh
//! programs over Metal's output limit"):
//!
//! * a vertex is its clip and cull distances (4 bytes each), its position (16 bytes, at the next
//!   multiple of 16), then every other output **in the order of the signature's elements** (the
//!   order `PSV0` and the DXIL metadata list them in, not the register order `OSG1` is sorted
//!   by), each at the next multiple of its own size (a `float3` is 12 bytes at a multiple of
//!   12); the distances are counted again there, as the element that declares them;
//! * the declared vertex count is rounded up to a multiple of 4;
//! * a primitive is its per-primitive outputs laid out the same way, and its indices are free.

use super::container::SigElement;
use crate::dxbc::u32_at;
use crate::{Container, FourCC};

pub const METAL_MESH_OUTPUT_BYTES: u32 = 32_768;

/// The passes a split mesh program is drawn in ([`MeshSplit`]).
pub const MESH_SPLIT_PASSES: u8 = 4;

/// One pass of a mesh program whose vertices do not all fit Metal's output limit.
///
/// Each pass leaves out a different block of `excluded` vertices (pass `n` the vertices
/// `n * excluded ..= (n + 1) * excluded - 1`) and so declares `excluded` vertices fewer. A
/// triangle has three corners, so at least one of the four blocks holds none of them: it is
/// drawn in the first such pass and culled in the others. Every lane computes its vertex as
/// it always does, in every pass; only where the vertex is stored (the vertices after the
/// block move down) and which primitives are emitted differ. The four passes together draw
/// exactly the program's triangles, each once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshSplit {
    pub pass: u8,
    pub excluded: u16,
}

/// How a linked mesh program is changed to fit Metal's output limit; the default changes nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MeshFit {
    /// The vertex outputs are declared in the order that takes the least storage
    /// ([`tightest_order`]); the stages link by semantic, so the pixel program is as it was.
    pub reorder: bool,
    pub split: Option<MeshSplit>,
}

/// What Metal counts for a mesh program's outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshFootprint {
    /// Declared maxima.
    pub vertices: u32,
    pub primitives: u32,
    /// Bytes of one vertex and of one primitive.
    pub vertex_bytes: u32,
    pub primitive_bytes: u32,
}

impl MeshFootprint {
    /// The bytes of a thread group's outputs with `vertices` declared.
    fn bytes_with(&self, vertices: u32) -> Option<u32> {
        vertices
            .checked_next_multiple_of(4)?
            .checked_mul(self.vertex_bytes)?
            .checked_add(self.primitives.checked_mul(self.primitive_bytes)?)
    }

    /// The bytes of a thread group's outputs.
    pub fn bytes(&self) -> Option<u32> {
        self.bytes_with(self.vertices)
    }

    /// The fewest vertices each of [`MESH_SPLIT_PASSES`] passes must leave out for a pass to
    /// fit Metal's limit; `None` if four disjoint blocks of that many vertices do not exist
    /// (the outputs are too large even for three quarters of the vertices).
    pub fn split_excluded(&self) -> Option<u16> {
        (1..=self.vertices / u32::from(MESH_SPLIT_PASSES))
            .find(|&x| self.bytes_with(self.vertices - x).is_some_and(|b| b <= METAL_MESH_OUTPUT_BYTES))
            .and_then(|x| u16::try_from(x).ok())
    }
}

/// A signature element as `PSV0` lists it.
struct PsvElement {
    name: String,
    index: u32,
    rows: u8,
    col: u8,
    cols: u8,
    /// DXIL semantic kind: 0 arbitrary, 3 position, 6 clip distance, 7 cull distance.
    kind: u8,
}

/// A mesh program's `PSV0` runtime info, its vertex outputs and its per-primitive outputs,
/// each in element order. `None` for anything but a well-formed mesh program.
fn psv_outputs(dxil: &[u8]) -> Option<(Vec<u8>, Vec<PsvElement>, Vec<PsvElement>)> {
    let c = Container::parse(dxil).ok()?;
    let psv = c.part(FourCC::PSV0)?.data;
    let word = |at: usize| u32_at(psv, at).ok().map(|w| w as usize);
    let info_size = word(0)?;
    let info = psv.get(4..4usize.checked_add(info_size)?)?;
    // Runtime info version 1 on has the stage and the element counts; 13 is a mesh shader.
    if info_size < 36 || info[24] != 13 {
        return None;
    }
    let mut at = 4 + info_size;
    let resources = word(at)?;
    at += 4;
    if resources > 0 {
        let size = word(at)?;
        at = at.checked_add(4)?.checked_add(resources.checked_mul(size)?)?;
    }
    let strings_len = word(at)?;
    let strings = psv.get(at + 4..(at + 4).checked_add(strings_len)?)?;
    at += 4 + strings_len;
    let indices = word(at)?;
    let index_at = at + 4;
    at = index_at.checked_add(indices.checked_mul(4)?)?;
    let (inputs, outputs, prims) = (usize::from(info[28]), usize::from(info[29]), usize::from(info[30]));
    if inputs + outputs + prims == 0 {
        return Some((info.to_vec(), Vec::new(), Vec::new()));
    }
    let size = word(at)?;
    at += 4;
    if size < 16 {
        return None;
    }
    let element = |n: usize| -> Option<PsvElement> {
        let e = psv.get(at.checked_add(n.checked_mul(size)?)?..)?.get(..16)?;
        let name = strings
            .get(u32_at(e, 0).ok()? as usize..)?
            .split(|&b| b == 0)
            .next()
            .map(|s| String::from_utf8_lossy(s).into_owned())?;
        let index = word(index_at.checked_add(4usize.checked_mul(u32_at(e, 4).ok()? as usize)?)?).unwrap_or(0) as u32;
        let (cols, col) = (e[10] & 0xF, e[10] >> 4 & 3);
        if cols == 0 || cols > 4 {
            return None;
        }
        Some(PsvElement { name, index, rows: e[8], col, cols, kind: e[11] })
    };
    let outs = (inputs..inputs + outputs).map(element).collect::<Option<Vec<_>>>()?;
    let prs = (inputs + outputs..inputs + outputs + prims).map(element).collect::<Option<Vec<_>>>()?;
    Some((info.to_vec(), outs, prs))
}

/// The end of `sizes` (bytes each) placed one after the other from `start`, each at the next
/// multiple of its own size.
fn layout(start: u32, sizes: impl Iterator<Item = u32>) -> Option<u32> {
    sizes
        .filter(|&s| s > 0)
        .try_fold(start, |end, size| end.checked_next_multiple_of(size)?.checked_add(size))
}

/// The bytes before a vertex's outputs other than its position: `distances` clip and cull
/// distances, then the position.
fn vertex_base(distances: u32) -> Option<u32> {
    distances.checked_mul(4)?.checked_next_multiple_of(16)?.checked_add(16)
}

/// What Metal counts for a mesh program's outputs (the module's text has the rule). Counts are
/// the declared maxima, not what a dispatch happens to emit. Fails closed on anything but a
/// well-formed mesh program.
pub fn mesh_footprint(dxil: &[u8]) -> Option<MeshFootprint> {
    let (info, vertex, primitive) = psv_outputs(dxil)?;
    let half = |at: usize| Some(u32::from(u16::from_le_bytes(info.get(at..at + 2)?.try_into().ok()?)));
    let rows = |e: &PsvElement| std::iter::repeat_n(4 * u32::from(e.cols), usize::from(e.rows));
    let distances = vertex
        .iter()
        .filter(|e| matches!(e.kind, 6 | 7))
        .map(|e| u32::from(e.cols) * u32::from(e.rows))
        .sum();
    Some(MeshFootprint {
        vertices: half(12)?,
        primitives: half(14)?,
        vertex_bytes: layout(vertex_base(distances)?, vertex.iter().filter(|e| e.kind != 3).flat_map(rows))?,
        primitive_bytes: layout(0, primitive.iter().flat_map(rows))?,
    })
}

/// The bytes Metal counts for a mesh program's outputs ([`mesh_footprint`]).
pub fn metal_mesh_bytes(dxil: &[u8]) -> Option<u32> {
    mesh_footprint(dxil)?.bytes()
}

/// Declares a mesh program's vertex outputs in the order that takes the least storage: the
/// position, then the others as [`tightest_order`] places them. Before anything refers to an
/// output by its place.
pub(crate) fn reorder_outputs(outputs: &mut Vec<SigElement>) {
    let distances: u32 = outputs.iter().filter(|e| matches!(e.kind, 6 | 7)).map(|e| u32::from(e.cols)).sum();
    let Some(base) = vertex_base(distances) else { return };
    let (position, rest): (Vec<SigElement>, Vec<SigElement>) = outputs.drain(..).partition(|e| e.kind == 3);
    let sizes: Vec<u8> = rest.iter().map(|e| e.cols).collect();
    let (order, _) = tightest_order(base, &sizes);
    *outputs = position;
    outputs.extend(order.into_iter().map(|i| rest[i].clone()));
}

/// The order of a mesh program's vertex outputs after its position that takes the least
/// storage: `sizes` are the outputs' component counts (1 to 4) in their present order, `base`
/// the bytes before them (a multiple of 16: the distances and the position), and the result
/// lists indices into `sizes` with the bytes one vertex then takes. Metal places each output at
/// the next multiple of its own size, so `float3 float float3 float` takes 48 bytes and
/// `float3 float3 float float` 32. Exact: a search over how many outputs of each width are
/// placed and where the next one starts (offsets repeat every 48 bytes). Outputs of one width
/// keep their order.
pub(crate) fn tightest_order(base: u32, sizes: &[u8]) -> (Vec<usize>, u32) {
    use std::collections::HashMap;
    type Counts = [u8; 4];
    let mut total: Counts = [0; 4];
    for &n in sizes {
        total[usize::from(n.clamp(1, 4)) - 1] += 1;
    }
    // Least end per (counts placed, end modulo 48), with the width placed last and the state
    // before it.
    type Reached = [Option<(u32, u8, u8)>; 12];
    let mut reached: HashMap<Counts, Reached> = HashMap::new();
    let mut start: Reached = [None; 12];
    start[(base % 48 / 4) as usize] = Some((base, 0, 0));
    reached.insert([0; 4], start);
    let mut frontier = vec![[0u8; 4]];
    for _ in 0..sizes.len() {
        let mut next = Vec::new();
        for counts in frontier {
            let from = reached[&counts];
            for width in 1..=4u8 {
                let k = usize::from(width) - 1;
                if counts[k] == total[k] {
                    continue;
                }
                let mut to = counts;
                to[k] += 1;
                let bytes = 4 * u32::from(width);
                for (residue, state) in from.iter().enumerate() {
                    let Some((end, _, _)) = *state else { continue };
                    let end = end.next_multiple_of(bytes) + bytes;
                    let slot = &mut reached.entry(to).or_insert_with(|| {
                        next.push(to);
                        [None; 12]
                    })[(end % 48 / 4) as usize];
                    if slot.is_none_or(|s| end < s.0) {
                        *slot = Some((end, width, residue as u8));
                    }
                }
            }
        }
        frontier = next;
    }
    let done = reached[&total];
    let (mut residue, &(end, _, _)) = done
        .iter()
        .enumerate()
        .filter_map(|(r, s)| s.as_ref().map(|s| (r, s)))
        .min_by_key(|(_, s)| s.0)
        .expect("an order");
    let mut widths = Vec::with_capacity(sizes.len());
    let mut counts = total;
    while counts != [0; 4] {
        let (_, width, before) = reached[&counts][residue].expect("a reached state");
        widths.push(width);
        counts[usize::from(width) - 1] -= 1;
        residue = usize::from(before);
    }
    widths.reverse();
    let mut of_width: [std::collections::VecDeque<usize>; 4] = Default::default();
    for (i, &n) in sizes.iter().enumerate() {
        of_width[usize::from(n.clamp(1, 4)) - 1].push_back(i);
    }
    let order = widths
        .iter()
        .map(|&w| of_width[usize::from(w) - 1].pop_front().expect("an output of that width"))
        .collect();
    (order, end)
}

/// A mesh program's declared maxima and outputs in one line, in element order, for diagnostics:
/// `v128 p128 | SV_Position0.xyzw TEXCOORD3.xy ... | <per-primitive outputs>`.
pub fn mesh_outputs_text(dxil: &[u8]) -> String {
    let Some((info, vertex, primitive)) = psv_outputs(dxil) else { return "not a mesh program".into() };
    let half = |at: usize| u16::from_le_bytes([info[at], info[at + 1]]);
    let elements = |els: &[PsvElement]| -> String {
        els.iter()
            .map(|e| {
                let name = match e.kind {
                    0 => e.name.as_str(),
                    3 => "SV_Position",
                    6 => "SV_ClipDistance",
                    7 => "SV_CullDistance",
                    _ => "SV",
                };
                let channels: String = "xyzw".chars().skip(usize::from(e.col)).take(usize::from(e.cols)).collect();
                format!("{name}{}.{channels}", e.index)
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!("v{} p{} | {} | {}", half(12), half(14), elements(&vertex), elements(&primitive))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::container::{Psv, psv};
    use crate::emit::shader::Mesh;

    fn element(kind: u8, cols: u8) -> SigElement {
        let name = if kind == 0 { "TEXCOORD" } else { "SV" };
        SigElement { name: name.into(), index: 0, kind, comp: 9, interp: 0, row: Some(0), col: 0, cols, usage: 0, half: false }
    }

    /// A container with the `PSV0` of a mesh program of these outputs, as the emitter writes it.
    fn shader(vertices: u32, primitives: u32, vertex: &[(u8, u8)], primitive: &[(u8, u8)]) -> Vec<u8> {
        let elements = |els: &[(u8, u8)]| -> Vec<SigElement> {
            els.iter()
                .enumerate()
                .map(|(i, &(kind, cols))| SigElement { index: i as u32, row: Some(i as u32), ..element(kind, cols) })
                .collect()
        };
        let (outputs, prims) = (elements(vertex), elements(primitive));
        let part = psv(&Psv {
            stage: 13,
            flag: false,
            inputs: &[],
            outputs: &outputs,
            prims: &prims,
            mesh: Some((Mesh { max_vertices: vertices, max_primitives: primitives, topology: 2 }, 0)),
            gs: None,
            bindings: &[],
            threads: [128, 1, 1],
            wave_size: None,
        });
        crate::dxbc::build(&[(FourCC::PSV0, &part)])
    }

    /// The numbers are Metal's own, from building pipelines of these outputs (the error text
    /// names the size it counted).
    #[test]
    fn mesh_size_follows_element_order_alignment_and_rounded_counts() {
        // Position 16, scalar 4, float3 at 24, float4 at 48: 64 a vertex; a primitive's float 4.
        assert_eq!(
            metal_mesh_bytes(&shader(128, 64, &[(3, 4), (0, 1), (0, 3), (0, 4)], &[(0, 1)])),
            Some(128 * 64 + 64 * 4)
        );
        let mut outputs = vec![(3, 4)];
        outputs.extend([(0, 4); 15]);
        assert_eq!(metal_mesh_bytes(&shader(128, 0, &outputs, &[])), Some(METAL_MESH_OUTPUT_BYTES));
        // The position is first wherever it is declared.
        outputs.rotate_left(1);
        assert_eq!(metal_mesh_bytes(&shader(128, 0, &outputs, &[])), Some(METAL_MESH_OUTPUT_BYTES));
        outputs.rotate_right(1);
        // Vertices count in fours (129 as 132, 125 as 128); primitives as declared.
        assert_eq!(metal_mesh_bytes(&shader(129, 0, &outputs, &[])), Some(132 * 256));
        assert_eq!(metal_mesh_bytes(&shader(125, 0, &outputs, &[])), Some(128 * 256));
        assert_eq!(metal_mesh_bytes(&shader(128, 127, &outputs, &[(0, 4), (0, 1)])), Some(32_768 + 127 * 20));
        assert_eq!(metal_mesh_bytes(&shader(128, 128, &outputs, &[(0, 1), (0, 3)])), Some(32_768 + 128 * 24));
        // Element order decides the padding: 3 1 3 1 takes 48 bytes, 3 3 1 1 takes 40.
        let with = |tail: &[(u8, u8)]| metal_mesh_bytes(&shader(128, 0, &[&outputs[..], tail].concat(), &[])).unwrap();
        assert_eq!(with(&[(0, 3), (0, 1), (0, 3), (0, 1)]), 128 * 304);
        assert_eq!(with(&[(0, 3), (0, 3), (0, 1), (0, 1)]), 128 * 296);
        assert_eq!(with(&[(0, 2), (0, 1), (0, 2), (0, 1)]), 128 * 284);
        // A distance is counted before the position and again as its element: one clip distance
        // after fifteen float4 is 4 + 12 (padding) + 16 + 240 + 4, declared first 4 + 12 + 16 +
        // 4 + 12 + 240.
        assert_eq!(with(&[(6, 1)]), 128 * 276);
        let mut first = vec![(3, 4), (6, 1)];
        first.extend([(0, 4); 15]);
        assert_eq!(metal_mesh_bytes(&shader(128, 0, &first, &[])), Some(128 * 288));
        first[1] = (6, 3);
        assert_eq!(metal_mesh_bytes(&shader(128, 0, &first, &[])), Some(128 * 288));
        assert_eq!(metal_mesh_bytes(&shader(1, 0, &[(3, 4), (6, 2)], &[])), Some(4 * 40));
    }

    #[test]
    fn mesh_size_rejects_malformed_or_non_mesh() {
        let b = shader(3, 1, &[(3, 4), (0, 2)], &[]);
        assert_eq!(metal_mesh_bytes(&b), Some(4 * 24));
        for end in 0..b.len() {
            assert_eq!(metal_mesh_bytes(&b[..end]), None);
        }
        let mut b = b;
        // First part starts after the table; the runtime info's shader kind is at part + 8 + 28.
        let at = u32_at(&b, 32).unwrap() as usize;
        b[at + 36] = 1;
        assert_eq!(metal_mesh_bytes(&b), None);
    }

    /// The search finds the least storage: checked against every order of small sets.
    #[test]
    fn the_tightest_order_is_the_least_of_all_orders() {
        fn end(base: u32, sizes: &[u8], order: &[usize]) -> u32 {
            layout(base, order.iter().map(|&i| 4 * u32::from(sizes[i]))).unwrap()
        }
        fn least(base: u32, sizes: &[u8], order: &mut Vec<usize>, left: &mut Vec<usize>) -> u32 {
            if left.is_empty() {
                return end(base, sizes, order);
            }
            let mut best = u32::MAX;
            for k in 0..left.len() {
                let i = left.remove(k);
                order.push(i);
                best = best.min(least(base, sizes, order, left));
                order.pop();
                left.insert(k, i);
            }
            best
        }
        let mut seed = 0x2545_F491u32;
        for _ in 0..200 {
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed
            };
            let sizes: Vec<u8> = (0..next() % 8).map(|_| (next() % 4 + 1) as u8).collect();
            let base = 16 * (1 + next() % 3);
            let (order, bytes) = tightest_order(base, &sizes);
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..sizes.len()).collect::<Vec<_>>(), "every output once");
            assert_eq!(end(base, &sizes, &order), bytes);
            assert_eq!(bytes, least(base, &sizes, &mut Vec::new(), &mut (0..sizes.len()).collect()), "{base} {sizes:?}");
        }
        // Outputs of one width keep their order: here the floats fill the 8 bytes before the
        // first multiple of 12.
        assert_eq!(tightest_order(16, &[3, 1, 3, 1]), (vec![1, 3, 0, 2], 48));
        // One of the title's signatures: 308 bytes a vertex as declared, 248 reordered.
        let sizes = [4, 4, 4, 1, 3, 1, 3, 1, 3, 1, 4, 3, 1, 3, 1, 4, 4, 3, 1, 4, 1];
        assert_eq!(end(32, &sizes, &(0..sizes.len()).collect::<Vec<_>>()), 308);
        assert_eq!(tightest_order(32, &sizes).1, 248);
    }

    #[test]
    fn reordered_outputs_keep_the_position_first_and_every_output() {
        let mut outputs: Vec<SigElement> = [(0, 3), (0, 1), (3, 4), (0, 3), (6, 1), (0, 1)]
            .iter()
            .enumerate()
            .map(|(i, &(kind, cols))| SigElement { index: i as u32, ..element(kind, cols) })
            .collect();
        reorder_outputs(&mut outputs);
        assert_eq!(outputs[0].kind, 3);
        let mut indices: Vec<u32> = outputs.iter().map(|e| e.index).collect();
        // After 4 (distance) + 12 + 16 (position): a float at 32, the float3s at 36 and 48, the
        // other floats (one of them the distance's element): 68 bytes, where the declared order
        // took 76.
        assert_eq!(indices, [2, 1, 0, 3, 4, 5]);
        assert_eq!(layout(32, outputs[1..].iter().map(|e| 4 * u32::from(e.cols))), Some(68));
        indices.sort_unstable();
        assert_eq!(indices, [0, 1, 2, 3, 4, 5]);
    }

    /// A pass of a split leaves out the fewest vertices that fit, counted in Metal's fours.
    #[test]
    fn a_split_leaves_out_the_fewest_vertices_that_fit() {
        let f = |vertex_bytes| MeshFootprint { vertices: 128, primitives: 128, vertex_bytes, primitive_bytes: 0 };
        // 116 vertices of 280 bytes are 32,480 bytes; 120 are 33,600.
        assert_eq!(f(280).split_excluded(), Some(12));
        assert_eq!(f(260).split_excluded(), Some(4));
        // 96 vertices of 341 bytes fit; of 342 they do not, and no four blocks of more exist.
        assert_eq!(f(341).split_excluded(), Some(32));
        assert_eq!(f(342).split_excluded(), None);
        let per_primitive = MeshFootprint { vertices: 128, primitives: 128, vertex_bytes: 256, primitive_bytes: 16 };
        assert_eq!(per_primitive.split_excluded(), Some(8));
    }
}
