//! Descriptor heaps, textures, samplers, render pipelines and passes on Metal, plus the one encoder
//! loop (`submit_passes`) that compute `submit` also goes through.
//!
//! Descriptor encoding is `metal_irconverter_runtime.h` re-implemented from its documented layout
//! (`IRDescriptorTableSetBuffer` / `SetTexture` / `SetSampler`, 24-byte entries), so no Apple header
//! is needed at build time.

use crate::backend::{ComputeBackend, Heaps, RootArgs};
use crate::frame_time::{self, Metric, Timer};
use crate::metal::{Buffer, Metal, Queue};
use crate::resources::*;
use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::*;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;
use std::rc::Rc;

pub(crate) type Res = Retained<ProtocolObject<dyn MTLResource>>;
pub(crate) type TextureObject = Retained<ProtocolObject<dyn MTLTexture>>;
type HeapRef = (Res, Option<TextureObject>);
type TextureViewKey = (usize, usize, usize, usize, usize, usize);

const BUFFER_VIEWS: usize = 8192;
const TEXTURE_VIEWS: usize = 64;
const RETIRED_OBJECTS: usize = 512;

/// A fence chaining the encoders of one command buffer, and whether one has run: what a pass waits
/// for and updates when the queue's buffers are made resident by a set instead of named.
struct Order {
    fence: Option<Retained<ProtocolObject<dyn MTLFence>>>,
    started: Cell<bool>,
}

type Ordering<'a> = Option<(&'a ProtocolObject<dyn MTLFence>, &'a Cell<bool>)>;

/// `IRDescriptorTableEntry`: `{ gpuVA, textureViewID, metadata }`.
const ENTRY: usize = 24;

/// `kIRDescriptorHeapBindPoint`, `kIRSamplerHeapBindPoint`, `kIRArgumentBufferBindPoint`,
/// `kIRArgumentBufferDrawArgumentsBindPoint`, `kIRArgumentBufferUniformsBindPoint`,
/// `kIRVertexBufferBindPoint`, `kIRStageInAttributeStartIndex` (`metal_irconverter_runtime.h`).
const HEAP_SLOT: [usize; 2] = [0, 1];
const ARGUMENT_BUFFER: usize = 2;
const DRAW_ARGUMENTS: usize = 4;
const UNIFORMS: usize = 5;
const VERTEX_BUFFERS: usize = 6;
const STAGE_IN_ATTRIBUTES: usize = 11;

pub struct Heap {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    kind: HeapKind,
    count: usize,
    trace_mirrors: bool,
    fast_buffers: bool,
    raw_buffers: RefCell<HashMap<usize, std::rc::Weak<HeapRef>>>,
    /// What each entry names: `(resident_allocation, optional_buffer_backed_texture_view)`.
    /// A buffer-backed `TypeTextureBuffer` shares its parent `MTLBuffer`'s backing allocation, so
    /// retaining it here keeps its `gpuResourceID` live without duplicating the pool's pages in
    /// `alloc_counts` / `MTLResidencySet` / per-encoder `useResource`.
    refs: RefCell<std::collections::HashMap<usize, Rc<HeapRef>>>,
    /// Entries holding a typed buffer view.
    typed_views: Cell<usize>,
    /// The distinct allocations `refs` names, by raw `MTLResource` pointer, with how many entries
    /// name each: the desired membership of `set`, reconciled only at commit.
    /// Removed objects stay in `retired` until commit; without a set each is named once per encoder.
    alloc_counts: RefCell<std::collections::HashMap<usize, (usize, Res)>>,
    /// Residency set holding the heap buffer and everything its entries name.
    set: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    /// Committed allocations, excluding the heap buffer (which stays until Drop).
    in_set: RefCell<HashSet<usize>>,
    dirty: Cell<bool>,
    /// The set asked to stay resident (`VICTORY_METAL_REQUEST_RESIDENCY`): it ends that when dropped.
    requested: bool,
    /// MTLResidencySet does not retain allocations, even while a removal awaits commit.
    retired: RefCell<HashMap<usize, Res>>,
    retired_views: RefCell<Vec<TextureObject>>,
    /// The same ownership as arrays (`crate::dense`), which replaces `refs`, `alloc_counts`,
    /// `in_set`, `retired`, `retired_views` and `raw_buffers` (they stay empty) unless
    /// `VICTORY_DENSE_ENTRIES=0` keeps the maps as the reference.
    dense: Option<RefCell<crate::dense::Dense<Res, TextureObject>>>,
    /// Per-heap commits, including creation, for residency lifetime regression tests.
    set_commits: Cell<u64>,
}

/// Prepared Metal descriptor words and ownership, independent of any heap slot. Sampler IDs
/// remain owned by Metal's existing sampler cache for the device's lifetime.
#[derive(Clone)]
pub struct Descriptor {
    words: [u64; 3],
    keep: Option<Rc<HeapRef>>,
    kind: Metric,
}

impl Heap {
    pub fn capture_descriptor(&self, index: usize) -> Descriptor {
        assert!(index < self.count);
        // SAFETY: the checked slot is live; the descriptor writer excludes concurrent GPU use.
        let words = unsafe { self.buf.contents().as_ptr().cast::<[u64; 3]>().add(index).read() };
        let keep = match &self.dense {
            Some(dense) => dense.borrow().get(index).map(|(_, r, view)| Rc::new((r.clone(), view.cloned()))),
            None => self.refs.borrow().get(&index).cloned(),
        };
        let kind = if self.kind == HeapKind::Sampler { Metric::MirrorSampler }
            else if keep.as_ref().is_some_and(|r| r.1.is_some()) { Metric::MirrorTyped }
            else if words[1] != 0 { Metric::MirrorImage }
            else if words[2] == 0 { Metric::MirrorNull }
            else { Metric::MirrorBuffer };
        Descriptor { words, keep, kind }
    }

    /// Reconcile desired membership with the last commit: one add or one remove per allocation
    /// per commit. Deferred entry-level adds/removes preceded an IOGPUGroupMemory kernel panic.
    fn commit_set(&self, metal: &Metal) {
        if let Some(set) = &self.set {
            let Some(_changes) = RESIDENCY.changes() else { return };
            let counts = self.alloc_counts.borrow();
            let mut in_set = self.in_set.borrow_mut();
            let mut retired = self.retired.borrow_mut();
            let mut dense = self.dense.as_ref().map(|d| d.borrow_mut());
            let (mut adds, mut removes) = (0, 0);
            if self.set_commits.get() == 0 {
                set.addAllocation(ProtocolObject::from_ref(&*self.buf));
                adds += 1;
            }
            let add = |allocation: &Res| set.addAllocation(ProtocolObject::from_ref(&**allocation));
            let remove = |allocation: &Res| set.removeAllocation(ProtocolObject::from_ref(&**allocation));
            let (more_adds, more_removes) = match &mut dense {
                Some(dense) => dense.reconcile(add, remove),
                None => crate::residency::reconcile(&counts, &mut in_set, &retired, add, remove),
            };
            adds += more_adds;
            removes += more_removes;
            metal.count(|c| {
                c.set_adds += adds;
                c.set_removes += removes;
            });
            frame_time::count(Metric::SetAdds, adds);
            frame_time::count(Metric::SetRemoves, removes);
            if adds != 0 || removes != 0 {
                self.publish_set(set);
            }
            // With no difference there is nothing pending in Metal, so retirement is safe too.
            // Only here, after the set is published, do the removed allocations go.
            // They are kept until the work submitted before has finished: a command buffer in
            // flight may still read them through entries or table copies written before
            // (docs/hang-notes.md).
            let (mut gone, mut views) = match &mut dense {
                Some(dense) => dense.finish_commit(),
                None => (Vec::new(), Vec::new()),
            };
            views.append(&mut self.retired_views.borrow_mut());
            gone.extend(retired.drain().map(|(_, r)| r));
            metal.keep_until_done(gone, views);
            self.dirty.set(false);
        }
    }

    fn publish_set(&self, set: &ProtocolObject<dyn MTLResidencySet>) {
        let _commit = Timer::counted(Metric::SetCommit, Metric::SetCommits);
        crate::metal::residency_timed("heap_commit", Some(set), || set.commit());
        self.set_commits.set(self.set_commits.get() + 1);
    }

    /// Entries that name an allocation.
    fn named_entries(&self) -> usize {
        match &self.dense {
            Some(dense) => dense.borrow().entries(),
            None => self.refs.borrow().len(),
        }
    }

    /// Distinct allocations the entries name.
    fn named_allocations(&self) -> usize {
        match &self.dense {
            Some(dense) => dense.borrow().live(),
            None => self.alloc_counts.borrow().len(),
        }
    }

    /// Each distinct allocation the entries name, with its raw pointer.
    fn each_allocation(&self, mut f: impl FnMut(usize, &Res)) {
        match &self.dense {
            Some(dense) => dense.borrow().counts().for_each(|(id, _, r)| f(id, r)),
            None => self.alloc_counts.borrow().iter().for_each(|(&id, (_, r))| f(id, r)),
        }
    }

    /// Each entry that names an allocation, with the allocation's raw pointer.
    fn each_entry(&self, mut f: impl FnMut(usize, usize)) {
        match &self.dense {
            Some(dense) => dense.borrow().iter().for_each(|(index, id, ..)| f(index, id)),
            None => self.refs.borrow().iter().for_each(|(&index, r)| {
                f(index, (&*r.0 as *const ProtocolObject<dyn MTLResource>).cast::<()>() as usize)
            }),
        }
    }

    /// Removed objects (allocations, views) still owned for the next commit.
    #[cfg(test)]
    fn pending_retired(&self) -> (usize, usize) {
        match &self.dense {
            Some(dense) => {
                let dense = dense.borrow();
                (dense.retired(), dense.retired_views())
            }
            None => (self.retired.borrow().len(), self.retired_views.borrow().len()),
        }
    }

    #[cfg(test)]
    fn allocation_counts(&self) -> HashMap<usize, usize> {
        match &self.dense {
            Some(dense) => dense.borrow().counts().map(|(id, n, _)| (id, n)).collect(),
            None => self.alloc_counts.borrow().iter().map(|(&id, (n, _))| (id, *n)).collect(),
        }
    }

    /// The allocations the set holds as of the last commit, without the heap buffer.
    #[cfg(test)]
    fn committed(&self) -> HashSet<usize> {
        match &self.dense {
            Some(dense) => dense.borrow().membership(),
            None => self.in_set.borrow().clone(),
        }
    }

    /// The allocation (raw pointer) and typed view entry `index` names.
    #[cfg(test)]
    fn entry(&self, index: usize) -> Option<(usize, Option<TextureObject>)> {
        match &self.dense {
            Some(dense) => dense.borrow().get(index).map(|(id, _, view)| (id, view.cloned())),
            None => self.refs.borrow().get(&index).map(|r| {
                ((&*r.0 as *const ProtocolObject<dyn MTLResource>).cast::<()>() as usize, r.1.clone())
            }),
        }
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        // MTLResidencySet does not retain allocations: detach everything and commit before
        // `buf` and `refs` are dropped.
        if let Some(set) = &self.set {
            let Some(_changes) = RESIDENCY.changes() else { return };
            if self.requested {
                crate::metal::residency_timed("end", Some(set), || set.endResidency());
            }
            set.removeAllAllocations();
            self.publish_set(set);
            self.in_set.borrow_mut().clear();
        }
    }
}

pub struct Texture {
    tex: Retained<ProtocolObject<dyn MTLTexture>>,
    /// What the pixels are as Metal reads them: sizes the CPU transfers' bounds check.
    format: Format,
    /// Present reuses the display-value view instead of allocating an sRGB view each frame.
    present_view: std::cell::OnceCell<Retained<ProtocolObject<dyn MTLTexture>>>,
    /// Fitted format/type/mip/slice views. A view retains the Metal parent; that parent does
    /// not retain the view. Heap entries retain views independently when this cache is cleared.
    views: RefCell<std::collections::HashMap<TextureViewKey, TextureObject>>,
}

/// What a CPU transfer of a `w` x `h` x `d` mip needs of `format`: `(row bytes, total bytes)`, or the
/// reason the pitches or the `len` bytes cannot hold it. Metal asserts (and may write past the buffer)
/// when the row pitch is short, so a refused transfer is skipped.
fn transfer_need(format: Format, [w, h, d]: [usize; 3], row_pitch: usize, slice_pitch: usize, len: usize) -> Result<usize, (usize, usize)> {
    let row = format.row_bytes(w);
    let rows = h.div_ceil(format.block().0);
    let image = (rows - 1) * row_pitch + row;
    let total = if d > 1 { (d - 1) * slice_pitch + image } else { image };
    if row_pitch < row || (d > 1 && slice_pitch < image) || len < total {
        return Err((row, total));
    }
    Ok(total)
}

/// Whether the transfer fits; else logs it (the backtrace once) and says to skip it.
#[allow(clippy::too_many_arguments)]
fn transfer_fits(what: &str, t: &Texture, mip: usize, dims: [usize; 3], row_pitch: usize, slice_pitch: usize, len: usize) -> bool {
    static FIRST: std::sync::Once = std::sync::Once::new();
    let Err((row, total)) = transfer_need(t.format, dims, row_pitch, slice_pitch, len) else {
        return true;
    };
    tracing::warn!(
        "{what} skipped: {:?} {:?} mip {mip} region {}x{}x{} row_pitch {row_pitch} slice_pitch {slice_pitch} buffer {len} bytes, needs row {row} total {total}",
        t.format, t.tex.pixelFormat(), dims[0], dims[1], dims[2]
    );
    FIRST.call_once(|| tracing::warn!("first refused transfer, called from:\n{}", std::backtrace::Backtrace::force_capture()));
    false
}

impl Texture {
    /// The texture as the bytes it holds when it is sRGB (a view in the plain format), else itself.
    pub(crate) fn view_without_srgb(&self, _m: &Metal) -> Retained<ProtocolObject<dyn MTLTexture>> {
        let plain = match self.tex.pixelFormat() {
            MTLPixelFormat::RGBA8Unorm_sRGB => MTLPixelFormat::RGBA8Unorm,
            MTLPixelFormat::BGRA8Unorm_sRGB => MTLPixelFormat::BGRA8Unorm,
            _ => return self.tex.clone(),
        };
        self.present_view.get_or_init(|| {
            self.tex.newTextureViewWithPixelFormat(plain).unwrap_or_else(|| self.tex.clone())
        }).clone()
    }
}

pub struct Sampler(Retained<ProtocolObject<dyn MTLSamplerState>>);

pub struct RenderPipeline {
    pso: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth: Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    cull: MTLCullMode,
    raster: Raster,
    /// The pipeline's target has a stencil plane (the pass attaches it too).
    stencil: bool,
    /// Made for no colour and no depth target ([`targetless_samples`]): its draws run in a pass
    /// without attachments, and in no other.
    targetless: bool,
    mesh: Option<MeshDispatch>,
    object_static: Option<StaticSamplers>,
    /// Each stage's static samplers, which the stage's argument buffer points at.
    vs_static: Option<StaticSamplers>,
    ps_static: Option<StaticSamplers>,
}

/// One shader stage's static samplers, as Apple's converter reads them: not built into the
/// shader, but one table of `IRDescriptorTableEntry` (sampler, 0, LOD bias), `s0` first, whose
/// address the stage's top-level argument buffer holds at `offset`, right after the root
/// parameters. Nothing wrote that address before: the shader read a sampler through whatever
/// bytes followed the parameters, which sampled zeros on M1 and M2 (Apple7, Apple8) GPUs and
/// left the window black at the title's present pass (Native-Victory-macOS #26).
pub(crate) struct StaticSamplers {
    table: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
    /// The table holds only their resource IDs.
    _samplers: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>>,
}

impl StaticSamplers {
    /// `root` with the table's address at its offset.
    fn args<'a>(t: Option<&Self>, root: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
        let Some(t) = t else { return root.into() };
        let mut v = root.to_vec();
        if v.len() < t.offset + 8 {
            v.resize(t.offset + 8, 0);
        }
        v[t.offset..t.offset + 8].copy_from_slice(&t.table.gpuAddress().to_le_bytes());
        v.into()
    }

    fn table(t: Option<&Self>) -> Option<&ProtocolObject<dyn MTLResource>> {
        t.map(|t| ProtocolObject::from_ref(&*t.table))
    }
}

pub(crate) struct MeshDispatch {
    threads: MTLSize,
    object_threads: MTLSize,
    object: bool,
    /// The further passes of a split mesh program (`MeshPipelineDesc::more`): a draw runs the
    /// pipeline's own state and then each of these, with the same arguments.
    more: Vec<RenderState>,
}

fn pixel_format(f: Format) -> MTLPixelFormat {
    match f {
        Format::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        Format::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Format::Rgba16Float => MTLPixelFormat::RGBA16Float,
        Format::R32Float => MTLPixelFormat::R32Float,
        Format::D32Float => MTLPixelFormat::Depth32Float,
        Format::D32FloatS8 => MTLPixelFormat::Depth32Float_Stencil8,
        Format::Bc1 => MTLPixelFormat::BC1_RGBA,
        Format::Bc2 => MTLPixelFormat::BC2_RGBA,
        Format::Bc3 => MTLPixelFormat::BC3_RGBA,
        Format::Bc4 => MTLPixelFormat::BC4_RUnorm,
        Format::Bc5 => MTLPixelFormat::BC5_RGUnorm,
        // Metal's `RGBFloat` is the signed BC6H, `RGBUfloat` the unsigned one.
        Format::Bc6h => MTLPixelFormat::BC6H_RGBUfloat,
        Format::Bc6hSigned => MTLPixelFormat::BC6H_RGBFloat,
        Format::Bc7 => MTLPixelFormat::BC7_RGBAUnorm,
        Format::R8Unorm => MTLPixelFormat::R8Unorm,
        Format::A8Unorm => MTLPixelFormat::A8Unorm,
        Format::R16Unorm => MTLPixelFormat::R16Unorm,
        Format::Rgba32Float => MTLPixelFormat::RGBA32Float,
        Format::Rg32Float => MTLPixelFormat::RG32Float,
        Format::Rgb10a2Unorm => MTLPixelFormat::RGB10A2Unorm,
        Format::Rg11b10Float => MTLPixelFormat::RG11B10Float,
        Format::Rgba8Uint => MTLPixelFormat::RGBA8Uint,
        Format::Rg16Float => MTLPixelFormat::RG16Float,
        Format::Rg16Unorm => MTLPixelFormat::RG16Unorm,
        Format::R32Uint => MTLPixelFormat::R32Uint,
        Format::R8Uint => MTLPixelFormat::R8Uint,
        Format::Rgb9e5Float => MTLPixelFormat::RGB9E5Float,
        Format::Bc7Srgb => MTLPixelFormat::BC7_RGBAUnorm_sRGB,
        Format::Rgba32Uint => MTLPixelFormat::RGBA32Uint,
        Format::Rgba16Unorm => MTLPixelFormat::RGBA16Unorm,
        Format::Rgba16Snorm => MTLPixelFormat::RGBA16Snorm,
        Format::Rg32Uint => MTLPixelFormat::RG32Uint,
        Format::Rg16Uint => MTLPixelFormat::RG16Uint,
        Format::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        Format::Rg8Uint => MTLPixelFormat::RG8Uint,
        Format::Rg8Snorm => MTLPixelFormat::RG8Snorm,
        Format::R16Float => MTLPixelFormat::R16Float,
        Format::R16Uint => MTLPixelFormat::R16Uint,
        Format::D16Unorm => MTLPixelFormat::Depth16Unorm,
        Format::Bc1Srgb => MTLPixelFormat::BC1_RGBA_sRGB,
        Format::Bc3Srgb => MTLPixelFormat::BC3_RGBA_sRGB,
        Format::Bgr5a1Unorm => MTLPixelFormat::BGR5A1Unorm,
        Format::Abgr4Unorm => MTLPixelFormat::ABGR4Unorm,
        Format::Rgba16Uint => MTLPixelFormat::RGBA16Uint,
        Format::Rgba8Snorm => MTLPixelFormat::RGBA8Snorm,
        Format::Rg16Snorm => MTLPixelFormat::RG16Snorm,
        Format::R16Snorm => MTLPixelFormat::R16Snorm,
        Format::R8Snorm => MTLPixelFormat::R8Snorm,
        Format::Rgb10a2Uint => MTLPixelFormat::RGB10A2Uint,
    }
}

/// Whether a texture of `from` may be viewed as `to`: uncompressed formats of one texel size, or
/// a block format and its sRGB or signed twin. Never wider than what Metal casts (it asserts on
/// the rest); a depth view of a colour texture is not one of them.
fn castable(from: Format, to: Format) -> bool {
    use Format::*;
    let twin = |f| match f {
        Bc1Srgb => Bc1,
        Bc3Srgb => Bc3,
        Bc7Srgb => Bc7,
        Bc6hSigned => Bc6h,
        f => f,
    };
    if matches!(to, D32Float | D32FloatS8 | D16Unorm) {
        return false;
    }
    if from.block().0 == 1 { from.block() == to.block() } else { twin(from) == twin(to) }
}

fn tex_type(d: TexDim) -> MTLTextureType {
    match d {
        TexDim::D2 => MTLTextureType::Type2D,
        TexDim::D2Array => MTLTextureType::Type2DArray,
        TexDim::D3 => MTLTextureType::Type3D,
    }
}

fn compare(c: Compare) -> MTLCompareFunction {
    match c {
        Compare::Never => MTLCompareFunction::Never,
        Compare::Less => MTLCompareFunction::Less,
        Compare::Equal => MTLCompareFunction::Equal,
        Compare::LessEqual => MTLCompareFunction::LessEqual,
        Compare::Greater => MTLCompareFunction::Greater,
        Compare::NotEqual => MTLCompareFunction::NotEqual,
        Compare::GreaterEqual => MTLCompareFunction::GreaterEqual,
        Compare::Always => MTLCompareFunction::Always,
    }
}

fn blend_factor(f: BlendFactor) -> MTLBlendFactor {
    match f {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::SrcColor => MTLBlendFactor::SourceColor,
        BlendFactor::InvSrcColor => MTLBlendFactor::OneMinusSourceColor,
        BlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::InvSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        BlendFactor::DestAlpha => MTLBlendFactor::DestinationAlpha,
        BlendFactor::InvDestAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        BlendFactor::DestColor => MTLBlendFactor::DestinationColor,
        BlendFactor::InvDestColor => MTLBlendFactor::OneMinusDestinationColor,
        BlendFactor::SrcAlphaSat => MTLBlendFactor::SourceAlphaSaturated,
        BlendFactor::Constant => MTLBlendFactor::BlendColor,
        BlendFactor::InvConstant => MTLBlendFactor::OneMinusBlendColor,
        BlendFactor::Src1Color => MTLBlendFactor::Source1Color,
        BlendFactor::InvSrc1Color => MTLBlendFactor::OneMinusSource1Color,
        BlendFactor::Src1Alpha => MTLBlendFactor::Source1Alpha,
        BlendFactor::InvSrc1Alpha => MTLBlendFactor::OneMinusSource1Alpha,
    }
}

fn blend_op(o: BlendOp) -> MTLBlendOperation {
    match o {
        BlendOp::Add => MTLBlendOperation::Add,
        BlendOp::Subtract => MTLBlendOperation::Subtract,
        BlendOp::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOp::Min => MTLBlendOperation::Min,
        BlendOp::Max => MTLBlendOperation::Max,
    }
}

fn stencil_op(o: StencilOp) -> MTLStencilOperation {
    match o {
        StencilOp::Keep => MTLStencilOperation::Keep,
        StencilOp::Zero => MTLStencilOperation::Zero,
        StencilOp::Replace => MTLStencilOperation::Replace,
        StencilOp::IncrementClamp => MTLStencilOperation::IncrementClamp,
        StencilOp::DecrementClamp => MTLStencilOperation::DecrementClamp,
        StencilOp::Invert => MTLStencilOperation::Invert,
        StencilOp::IncrementWrap => MTLStencilOperation::IncrementWrap,
        StencilOp::DecrementWrap => MTLStencilOperation::DecrementWrap,
    }
}

fn stencil_face(f: &StencilFace, s: &StencilState) -> Retained<MTLStencilDescriptor> {
    let d = MTLStencilDescriptor::new();
    d.setStencilFailureOperation(stencil_op(f.fail));
    d.setDepthFailureOperation(stencil_op(f.depth_fail));
    d.setDepthStencilPassOperation(stencil_op(f.pass));
    d.setStencilCompareFunction(compare(f.compare));
    d.setReadMask(u32::from(s.read_mask));
    d.setWriteMask(u32::from(s.write_mask));
    d
}

fn address(a: Address) -> MTLSamplerAddressMode {
    match a {
        Address::Wrap => MTLSamplerAddressMode::Repeat,
        Address::Mirror => MTLSamplerAddressMode::MirrorRepeat,
        Address::Clamp => MTLSamplerAddressMode::ClampToEdge,
        Address::Border => MTLSamplerAddressMode::ClampToBorderColor,
    }
}

fn filter(f: Filter) -> MTLSamplerMinMagFilter {
    match f {
        Filter::Nearest => MTLSamplerMinMagFilter::Nearest,
        Filter::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

fn dynamic_sampler_descriptor(d: &SamplerDesc) -> Retained<MTLSamplerDescriptor> {
    let sd = MTLSamplerDescriptor::new();
    sd.setMinFilter(filter(d.min));
    sd.setMagFilter(filter(d.mag));
    sd.setMipFilter(match d.mip {
        Filter::Nearest => MTLSamplerMipFilter::Nearest,
        Filter::Linear => MTLSamplerMipFilter::Linear,
    });
    sd.setSAddressMode(address(d.address[0]));
    sd.setTAddressMode(address(d.address[1]));
    sd.setRAddressMode(address(d.address[2]));
    sd.setBorderColor(match d.border_color {
        BorderColor::TransparentBlack => MTLSamplerBorderColor::TransparentBlack,
        BorderColor::OpaqueBlack => MTLSamplerBorderColor::OpaqueBlack,
        BorderColor::OpaqueWhite => MTLSamplerBorderColor::OpaqueWhite,
    });
    sd.setMaxAnisotropy(d.max_anisotropy.max(1));
    // Heap entries store resource IDs, which exist only for argument-buffer samplers.
    sd.setSupportArgumentBuffers(true);
    sd
}

/// A sampler for a static sampler's `D3D12_STATIC_SAMPLER_DESC` words up to `MaxLOD` (filter,
/// address U, V, W, LOD bias, anisotropy, comparison, border, minimum and maximum LOD). The LOD
/// bias is the table entry's. Minimum and maximum filtering (filter bits 0x100) have no
/// equivalent here and filter normally.
fn static_sampler(
    dev: &ProtocolObject<dyn MTLDevice>,
    w: &[u32; 10],
) -> Result<Retained<ProtocolObject<dyn MTLSamplerState>>, String> {
    let [
        filter,
        u,
        v,
        ww,
        _,
        anisotropy,
        comparison,
        border,
        min_lod,
        max_lod,
    ] = *w;
    let aniso = filter & 0x40 != 0;
    let linear = |bit: u32| {
        if aniso || filter & bit != 0 {
            MTLSamplerMinMagFilter::Linear
        } else {
            MTLSamplerMinMagFilter::Nearest
        }
    };
    let mode = |m: u32| match m {
        1 => MTLSamplerAddressMode::Repeat,
        2 => MTLSamplerAddressMode::MirrorRepeat,
        4 => MTLSamplerAddressMode::ClampToBorderColor,
        5 => MTLSamplerAddressMode::MirrorClampToEdge,
        _ => MTLSamplerAddressMode::ClampToEdge,
    };
    let sd = MTLSamplerDescriptor::new();
    sd.setMinFilter(linear(0x10));
    sd.setMagFilter(linear(0x4));
    sd.setMipFilter(if aniso || filter & 1 != 0 {
        MTLSamplerMipFilter::Linear
    } else {
        MTLSamplerMipFilter::Nearest
    });
    sd.setSAddressMode(mode(u));
    sd.setTAddressMode(mode(v));
    sd.setRAddressMode(mode(ww));
    sd.setMaxAnisotropy(if aniso {
        anisotropy.clamp(1, 16) as usize
    } else {
        1
    });
    if filter & 0x80 != 0 {
        // `D3D12_COMPARISON_FUNC`, 1 (never) to 8 (always).
        let c = [
            Compare::Never,
            Compare::Less,
            Compare::Equal,
            Compare::LessEqual,
            Compare::Greater,
            Compare::NotEqual,
            Compare::GreaterEqual,
            Compare::Always,
        ];
        sd.setCompareFunction(compare(c[(comparison.clamp(1, 8) - 1) as usize]));
    }
    sd.setBorderColor(match border {
        1 => MTLSamplerBorderColor::OpaqueBlack,
        2 => MTLSamplerBorderColor::OpaqueWhite,
        _ => MTLSamplerBorderColor::TransparentBlack,
    });
    let (lo, hi) = (f32::from_bits(min_lod), f32::from_bits(max_lod));
    let lo = if lo.is_nan() { 0.0 } else { lo.max(0.0) };
    sd.setLodMinClamp(lo);
    sd.setLodMaxClamp(if hi.is_nan() { f32::MAX } else { hi.max(lo) });
    // The table holds the sampler's resource ID, which exists only for argument-buffer samplers.
    sd.setSupportArgumentBuffers(true);
    dev.newSamplerStateWithDescriptor(&sd)
        .ok_or_else(|| "newSamplerState failed for a static sampler".into())
}

fn vertex_format(f: VertexFormat) -> MTLVertexFormat {
    match f {
        VertexFormat::Float => MTLVertexFormat::Float,
        VertexFormat::Float2 => MTLVertexFormat::Float2,
        VertexFormat::Float3 => MTLVertexFormat::Float3,
        VertexFormat::Float4 => MTLVertexFormat::Float4,
        VertexFormat::Uint => MTLVertexFormat::UInt,
        VertexFormat::Uint2 => MTLVertexFormat::UInt2,
        VertexFormat::Uint3 => MTLVertexFormat::UInt3,
        VertexFormat::Uint4 => MTLVertexFormat::UInt4,
        VertexFormat::Sint => MTLVertexFormat::Int,
        VertexFormat::Sint2 => MTLVertexFormat::Int2,
        VertexFormat::Sint3 => MTLVertexFormat::Int3,
        VertexFormat::Sint4 => MTLVertexFormat::Int4,
        VertexFormat::Half => MTLVertexFormat::Half,
        VertexFormat::Half2 => MTLVertexFormat::Half2,
        VertexFormat::Half4 => MTLVertexFormat::Half4,
        VertexFormat::Unorm16 => MTLVertexFormat::UShortNormalized,
        VertexFormat::Unorm16x2 => MTLVertexFormat::UShort2Normalized,
        VertexFormat::Unorm16x4 => MTLVertexFormat::UShort4Normalized,
        VertexFormat::Snorm16 => MTLVertexFormat::ShortNormalized,
        VertexFormat::Snorm16x2 => MTLVertexFormat::Short2Normalized,
        VertexFormat::Snorm16x4 => MTLVertexFormat::Short4Normalized,
        VertexFormat::Uint16 => MTLVertexFormat::UShort,
        VertexFormat::Uint16x2 => MTLVertexFormat::UShort2,
        VertexFormat::Uint16x4 => MTLVertexFormat::UShort4,
        VertexFormat::Sint16 => MTLVertexFormat::Short,
        VertexFormat::Sint16x2 => MTLVertexFormat::Short2,
        VertexFormat::Sint16x4 => MTLVertexFormat::Short4,
        VertexFormat::Unorm8 => MTLVertexFormat::UCharNormalized,
        VertexFormat::Unorm8x2 => MTLVertexFormat::UChar2Normalized,
        VertexFormat::Unorm8x4 => MTLVertexFormat::UChar4Normalized,
        VertexFormat::Snorm8 => MTLVertexFormat::CharNormalized,
        VertexFormat::Snorm8x2 => MTLVertexFormat::Char2Normalized,
        VertexFormat::Snorm8x4 => MTLVertexFormat::Char4Normalized,
        VertexFormat::Uint8 => MTLVertexFormat::UChar,
        VertexFormat::Uint8x2 => MTLVertexFormat::UChar2,
        VertexFormat::Uint8x4 => MTLVertexFormat::UChar4,
        VertexFormat::Sint8 => MTLVertexFormat::Char,
        VertexFormat::Sint8x2 => MTLVertexFormat::Char2,
        VertexFormat::Sint8x4 => MTLVertexFormat::Char4,
        VertexFormat::Unorm1010102 => MTLVertexFormat::UInt1010102Normalized,
        VertexFormat::Snorm1010102 => MTLVertexFormat::Int1010102Normalized,
        VertexFormat::FloatRG11B10 => MTLVertexFormat::FloatRG11B10,
    }
}

/// `MTLResourceID` is a `#[repr(transparent)]` u64 whose field is crate-private in `objc2-metal`.
fn resource_id(id: MTLResourceID) -> u64 {
    // SAFETY: MTLResourceID is one u64 (`Encode`d as such).
    unsafe { std::mem::transmute::<MTLResourceID, u64>(id) }
}

fn origin(a: [usize; 3]) -> MTLOrigin {
    MTLOrigin {
        x: a[0],
        y: a[1],
        z: a[2],
    }
}

fn size(a: [usize; 3]) -> MTLSize {
    MTLSize {
        width: a[0],
        height: a[1],
        depth: a[2],
    }
}

impl Metal {
    pub fn write_descriptor(&self, heap: &Heap, index: usize, entry: &Descriptor) {
        self.heap_entry(heap, index, entry.words, entry.keep.clone(), entry.kind);
    }

    fn heap_entry(
        &self,
        heap: &Heap,
        index: usize,
        words: [u64; 3],
        keep: Option<Rc<HeapRef>>,
        kind: Metric,
    ) {
        assert!(index < heap.count, "descriptor index {index} out of {}", heap.count);
        // SAFETY: index is inside the shared heap. The title must not change an entry while
        // the GPU reads it (D3D12 has the same rule).
        let p = unsafe { heap.buf.contents().as_ptr().cast::<u64>().add(index * ENTRY / 8) };
        let allocation = |r: &Rc<HeapRef>| (&*r.0 as *const ProtocolObject<dyn MTLResource>).cast::<()>() as usize;
        let new_id = keep.as_ref().map(allocation);
        if let Some(dense) = &heap.dense {
            let mut dense = dense.borrow_mut();
            let (old_id, old_view) = match dense.get(index) {
                Some((id, _, view)) => (Some(id), view),
                None => (None, None),
            };
            let same_view = match (old_view, keep.as_ref().and_then(|r| r.1.as_ref())) {
                (None, None) => true,
                (Some(a), Some(b)) => std::ptr::eq(&**a, &**b),
                _ => false,
            };
            // SAFETY: all three words belong to the checked slot.
            if old_id == new_id && same_view && unsafe { (0..3).all(|i| p.add(i).read_volatile() == words[i]) } {
                if heap.trace_mirrors { frame_time::count(Metric::EntrySame, 1); }
                return;
            }
            // SAFETY: same live slot, with descriptor-write exclusion as above.
            unsafe { for (i, w) in words.iter().enumerate() { p.add(i).write_volatile(*w); } }
            self.count(|c| c.entry_writes += 1);
            if heap.trace_mirrors { frame_time::count(kind, 1); }
            // The slab takes its own reference only for an allocation the heap does not hold;
            // a changed offset, size or view of the same allocation changes no count.
            let change = match &keep {
                Some(r) => dense.set(index, new_id, || r.0.clone(), r.1.clone()),
                None => dense.set(index, None, || unreachable!("nothing is named"), None),
            };
            heap.typed_views.set(dense.typed());
            let full = dense.retired() + dense.retired_views() > RETIRED_OBJECTS;
            drop(dense);
            if heap.set.is_some() && (change.added || change.retired) {
                heap.dirty.set(true);
                if change.retired && full { heap.commit_set(self); }
            }
            return;
        }
        let mut refs = heap.refs.borrow_mut();
        let old = refs.get(&index);
        let old_id = old.map(allocation);
        let same_allocation = old_id == new_id;
        let same_view = match (old.and_then(|r| r.1.as_ref()), keep.as_ref().and_then(|r| r.1.as_ref())) {
            (None, None) => true,
            (Some(a), Some(b)) => std::ptr::eq(&**a, &**b),
            _ => false,
        };
        // SAFETY: all three words belong to the checked slot.
        if same_allocation && same_view && unsafe { (0..3).all(|i| p.add(i).read_volatile() == words[i]) } {
            if heap.trace_mirrors { frame_time::count(Metric::EntrySame, 1); }
            return;
        }
        // SAFETY: same live slot, with descriptor-write exclusion as above.
        unsafe { for (i, w) in words.iter().enumerate() { p.add(i).write_volatile(*w); } }
        self.count(|c| c.entry_writes += 1);
        if heap.trace_mirrors { frame_time::count(kind, 1); }
        // Changing an offset, byte size, or texture-buffer view of the same allocation cannot
        // change residency. In particular raw -> typed -> raw never touches alloc_counts.
        let (added, retired) = if same_allocation { (false, None) } else {
            crate::residency::replace(&mut heap.alloc_counts.borrow_mut(), old_id,
                keep.as_ref().map(|r| (new_id.expect("allocation"), &r.0)))
        };
        if let Some((id, _)) = &retired { heap.raw_buffers.borrow_mut().remove(id); }
        let added_typed = keep.as_ref().is_some_and(|r| r.1.is_some());
        let old = match keep {
            Some(r) => refs.insert(index, r),
            None => refs.remove(&index),
        };
        drop(refs);
        let removed_typed = old.as_ref().is_some_and(|r| r.1.is_some());
        if added_typed != removed_typed {
            let n = heap.typed_views.get();
            heap.typed_views.set(if added_typed { n + 1 } else { n - 1 });
        }
        if heap.set.is_some() {
            if added { heap.dirty.set(true); }
            if let Some((id, allocation)) = retired {
                heap.dirty.set(true);
                let mut retired = heap.retired.borrow_mut();
                retired.insert(id, allocation);
                let mut views = heap.retired_views.borrow_mut();
                if let Some(view) = old.as_ref().and_then(|r| r.1.as_ref()) { views.push(view.clone()); }
                let full = retired.len() + views.len() > RETIRED_OBJECTS;
                drop(views);
                drop(retired);
                if full { heap.commit_set(self); }
            }
        }
    }

    /// A descriptor heap whose entry ownership is dense arrays (`dense`) or the reference maps.
    fn create_heap_as(&self, kind: HeapKind, count: usize, dense: bool) -> Result<Heap, String> {
        let buf = self
            .dev
            .newBufferWithLength_options(count * ENTRY, MTLResourceOptions::StorageModeShared)
            .ok_or("descriptor heap allocation failed")?;
        let set = if self.residency_sets {
            let d = MTLResidencySetDescriptor::new();
            let s = self
                .dev
                .newResidencySetWithDescriptor_error(&d)
                .map_err(|e| e.localizedDescription().to_string())?;
            Some(track_residency_set(s))
        } else {
            None
        };
        let heap = Heap {
            dense: dense.then(|| RefCell::new(crate::dense::Dense::new(count, set.is_some()))),
            buf,
            kind,
            count,
            trace_mirrors: frame_time::enabled(),
            fast_buffers: std::env::var("VICTORY_BUFFER_FAST").as_deref() != Ok("0"),
            raw_buffers: RefCell::default(),
            refs: RefCell::default(),
            typed_views: Cell::new(0),
            alloc_counts: RefCell::default(),
            requested: set.is_some() && self.request_residency,
            set,
            in_set: RefCell::default(),
            dirty: Cell::new(false),
            retired: RefCell::default(),
            retired_views: RefCell::default(),
            set_commits: Cell::new(0),
        };
        heap.commit_set(self);
        if let Some(set) = &heap.set { self.request(set); }
        Ok(heap)
    }

    /// A command buffer of `queue`, with encoder execution status when tracing failures.
    fn new_command_buffer(&self, queue: &Queue) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, String> {
        if self.trace_fails && self.encoder_status {
            let d = MTLCommandBufferDescriptor::new();
            d.setErrorOptions(MTLCommandBufferErrorOption::EncoderExecutionStatus);
            queue.0.commandBufferWithDescriptor(&d)
        } else {
            queue.0.commandBuffer()
        }
        .ok_or_else(|| "no command buffer".to_string())
    }

    /// The command buffer for the next piece of a split submission (`VICTORY_PASSES_PER_BUFFER`).
    /// A residency set keeps its allocations resident only for the command buffers it was given
    /// to, so every heap is attached again (`attached_heaps` is emptied): without that, pieces 2
    /// and later read heap entries with nothing keeping them resident, and the GPU faulted
    /// (`kIOGPUCommandBufferCallbackErrorPageFault`). The submission's waits are encoded again too,
    /// so no piece starts before what the submission waits for (values already reached cost
    /// nothing), and the piece waits for the one before it (`after`: the submission's piece event
    /// and the value that piece raises it to, by the GPU or, if the GPU fails it, by its recovery).
    fn next_piece(
        &self,
        queue: &Queue,
        waits: &[(&crate::metal::Fence, u64)],
        attached_heaps: &mut Vec<*const Heap>,
        after: Option<(&ProtocolObject<dyn MTLSharedEvent>, u64)>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, String> {
        let cb = self.new_command_buffer(queue)?;
        attached_heaps.clear();
        for (f, v) in waits {
            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*f.0), *v);
        }
        if let Some((event, v)) = after {
            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(event), v);
        }
        Ok(cb)
    }

    /// With `VICTORY_METAL_REQUEST_RESIDENCY`, asks that a heap's set, as last committed, stay
    /// resident between the command buffers that use it. Without it the system may make the
    /// set's allocations non-resident whenever no command buffer in flight uses the set, and
    /// make them resident again for the next one: in a run on a 48 GB Mac the wired memory fell
    /// from about 22 GB to 6 GB and came back within 10 s in the world, then command buffers
    /// timed out (#79). Measured since (`VICTORY_TRACE_RESIDENCY`, docs/native-host.md): the
    /// wired memory falls only after the host has committed no command buffer for a few seconds,
    /// this request does not stop that, and making the set resident again cost 0.1 to 0.3 s: the
    /// fall follows a stall, it does not cause it.
    fn request(&self, set: &ProtocolObject<dyn MTLResidencySet>) {
        if self.request_residency {
            let Some(_changes) = RESIDENCY.changes() else { return };
            crate::metal::residency_timed("request", Some(set), || set.requestResidency());
            self.count(|c| c.requests += 1);
        }
    }

    /// A set covers all passes in the command buffer. Later uses can still commit new entries.
    fn use_heap_cb(
        &self,
        heap: &Heap,
        attached_heaps: &mut Vec<*const Heap>,
        attach: impl FnOnce(&ProtocolObject<dyn MTLResidencySet>),
    ) {
        let _heap_sets = Timer::new(Metric::HeapSets);
        if let Some(set) = &heap.set {
            if heap.dirty.get() {
                heap.commit_set(self);
                self.request(set);
                self.count(|c| {
                    c.heap_entries = c.heap_entries.max(heap.named_entries() as u64);
                    c.heap_allocations = c.heap_allocations.max(heap.named_allocations() as u64);
                    c.heap_set_bytes = c.heap_set_bytes.max(set.allocatedSize());
                });
            }
            if !attached_heaps.contains(&std::ptr::from_ref(heap)) {
                attached_heaps.push(heap);
                attach(set);
            }
        }
    }

    /// Residency without sets: the heap buffer and each distinct allocation its entries name, once
    /// per encoder (`named` holds the heaps this encoder was given; residency lasts the encoder).
    /// Naming every entry at every draw cost a `useResource` per entry per draw, and the title fills
    /// more entries as the world streams in, so frame time grew with play time.
    fn use_heap_encoder(
        &self,
        heap: &Heap,
        named: &mut Vec<*const Heap>,
        mut use_res: impl FnMut(&ProtocolObject<dyn MTLResource>),
    ) {
        if heap.set.is_some() || named.contains(&(heap as *const Heap)) {
            return;
        }
        named.push(heap);
        use_res(ProtocolObject::from_ref(&*heap.buf));
        let refs = heap.named_entries() as u64;
        let allocations = heap.named_allocations() as u64;
        heap.each_allocation(|_, r| use_res(r));
        self.count(|c| {
            c.heap_resources += 1 + allocations;
            c.heap_entries = c.heap_entries.max(refs);
            c.heap_allocations = c.heap_allocations.max(allocations);
        });
    }

    /// The static samplers of one stage's `layout`, if it declares any.
    pub(crate) fn static_samplers(
        &self,
        layout: Option<&victory_backend::backend::RootLayout>,
    ) -> Result<Option<StaticSamplers>, String> {
        let Some(l) = layout.filter(|l| !l.static_samplers.is_empty()) else {
            return Ok(None);
        };
        // The converter's layout names the table as a table resource with no register.
        // With no table in the layout there is nothing to fill: the pipeline is still made.
        let Some(offset) = self
            .msc()?
            .locations(l)
            .map_err(|e| e.to_string())?
            .iter()
            .find(|&&(kind, space, slot, ..)| kind == 0 && space == u32::MAX && slot == u32::MAX)
            .map(|&(.., offset, _)| offset as usize)
        else {
            return Ok(None);
        };
        let table = self
            .dev
            .newBufferWithLength_options(
                l.static_samplers.len() * ENTRY,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or("no static sampler table")?;
        let mut samplers = Vec::new();
        for (n, w) in l.static_samplers.iter().enumerate() {
            let s = static_sampler(&self.dev, w)?;
            let bias = f32::from_bits(w[4]);
            let bias = if bias.is_nan() {
                0.0
            } else {
                bias.clamp(-16.0, 15.99)
            };
            let entry = [resource_id(s.gpuResourceID()), 0, u64::from(bias.to_bits())];
            // SAFETY: entry `n` is inside the table, which no GPU work reads yet.
            unsafe {
                let p = table.contents().as_ptr().cast::<u64>().add(n * ENTRY / 8);
                entry
                    .iter()
                    .enumerate()
                    .for_each(|(i, w)| p.add(i).write(*w));
            }
            samplers.push(s);
        }
        Ok(Some(StaticSamplers {
            table,
            offset,
            _samplers: samplers,
        }))
    }

    fn bind_root_compute(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        root: &RootArgs,
        heaps: &Heaps<'_, Metal>,
        statics: Option<&StaticSamplers>,
        named: &mut Vec<*const Heap>,
    ) {
        let root = StaticSamplers::args(statics, &root.0);
        if !root.is_empty() {
            // SAFETY: setBytes copies `root` (at most 4 KiB) before returning.
            unsafe {
                e.setBytes_length_atIndex(
                    NonNull::new(root.as_ptr().cast_mut().cast()).unwrap(),
                    root.len(),
                    ARGUMENT_BUFFER,
                )
            };
        }
        if let Some(t) = StaticSamplers::table(statics) {
            e.useResource_usage(t, MTLResourceUsage::Read);
        }
        for (h, slot) in [
            (heaps.resource, HEAP_SLOT[0]),
            (heaps.sampler, HEAP_SLOT[1]),
        ] {
            if let Some(h) = h {
                // SAFETY: a live buffer at an in-range slot.
                unsafe { e.setBuffer_offset_atIndex(Some(&h.buf), 0, slot) };
                self.use_heap_encoder(h, named, |r| {
                    e.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write)
                });
            }
        }
    }

    fn bind_root_render(
        &self,
        e: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        root: &RootArgs,
        fragment: Option<&RootArgs>,
        object: Option<&RootArgs>,
        heaps: &Heaps<'_, Metal>,
        pipeline: &RenderPipeline,
        named: &mut Vec<*const Heap>,
    ) {
        let mesh = pipeline.mesh.as_ref();
        let fragment = fragment.unwrap_or(root);
        let object = object.unwrap_or(root);
        let (vs, ps, os) = (pipeline.vs_static.as_ref(), pipeline.ps_static.as_ref(), pipeline.object_static.as_ref());
        let (root, fragment, object) = (
            StaticSamplers::args(vs, &root.0),
            StaticSamplers::args(ps, &fragment.0),
            StaticSamplers::args(os, &object.0),
        );
        // SAFETY: setBytes copies the bytes before returning.
        unsafe {
            if !root.is_empty() {
                let p = NonNull::new(root.as_ptr().cast_mut().cast()).unwrap();
                if mesh.is_some() {
                    e.setMeshBytes_length_atIndex(p, root.len(), ARGUMENT_BUFFER);
                } else {
                    e.setVertexBytes_length_atIndex(p, root.len(), ARGUMENT_BUFFER);
                }
            }
            if mesh.is_some_and(|m| m.object) && !object.is_empty() {
                e.setObjectBytes_length_atIndex(NonNull::new(object.as_ptr().cast_mut().cast()).unwrap(), object.len(), ARGUMENT_BUFFER);
            }
            if !fragment.is_empty() {
                let p = NonNull::new(fragment.as_ptr().cast_mut().cast()).unwrap();
                e.setFragmentBytes_length_atIndex(p, fragment.len(), ARGUMENT_BUFFER);
            }
        }
        for (t, stage) in [
            (vs, if mesh.is_some() { MTLRenderStages::Mesh } else { MTLRenderStages::Vertex }),
            (os, MTLRenderStages::Object),
            (ps, MTLRenderStages::Fragment),
        ] {
            if let Some(t) = StaticSamplers::table(t) {
                e.useResource_usage_stages(t, MTLResourceUsage::Read, stage);
            }
        }
        for (h, slot) in [
            (heaps.resource, HEAP_SLOT[0]),
            (heaps.sampler, HEAP_SLOT[1]),
        ] {
            if let Some(h) = h {
                // SAFETY: a live buffer at an in-range slot.
                unsafe {
                    if let Some(m) = mesh {
                        e.setMeshBuffer_offset_atIndex(Some(&h.buf), 0, slot);
                        if m.object { e.setObjectBuffer_offset_atIndex(Some(&h.buf), 0, slot); }
                    } else {
                        e.setVertexBuffer_offset_atIndex(Some(&h.buf), 0, slot);
                    }
                    e.setFragmentBuffer_offset_atIndex(Some(&h.buf), 0, slot);
                }
                self.use_heap_encoder(h, named, |r| {
                    e.useResource_usage_stages(
                        r,
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                        // A heap is named once per encoder, which may mix vertex and mesh draws.
                        MTLRenderStages::Vertex | MTLRenderStages::Mesh | MTLRenderStages::Object | MTLRenderStages::Fragment,
                    )
                });
            }
        }
    }
}

impl ResourceBackend for Metal {
    type Texture = Texture;
    type Sampler = Sampler;

    fn create_heap(&self, kind: HeapKind, count: usize) -> Result<Heap, String> {
        self.create_heap_as(kind, count, std::env::var("VICTORY_DENSE_ENTRIES").as_deref() != Ok("0"))
    }

    fn heap_address(&self, heap: &Heap, index: usize) -> u64 {
        heap.buf.gpuAddress() + (index * ENTRY) as u64
    }

    fn write_buffer(
        &self,
        heap: &Heap,
        index: usize,
        buffer: &Buffer,
        title: *const u8,
        size: u32,
    ) {
        debug_assert_eq!(heap.kind, HeapKind::Resource);
        let gpu = self.gpu_address(buffer, title);
        let kind = if size == 0 { Metric::MirrorNull } else { Metric::MirrorBuffer };
        let allocation: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&*buffer.buf);
        let id = (allocation as *const ProtocolObject<dyn MTLResource>).cast::<()>() as usize;
        let words = [gpu, 0, u64::from(size)];
        if let Some(dense) = heap.dense.as_ref().filter(|_| heap.fast_buffers) {
            assert!(index < heap.count);
            let mut dense = dense.borrow_mut();
            let same = dense.is_raw(index, id);
            // SAFETY: the slot is in bounds; the title excludes descriptor writes during use.
            let p = unsafe { heap.buf.contents().as_ptr().cast::<[u64; 3]>().add(index) };
            if same && unsafe { p.read_volatile() } == words {
                if heap.trace_mirrors { frame_time::count(Metric::EntrySame, 1); }
            } else {
                unsafe { p.write_volatile(words); }
                self.count(|c| c.entry_writes += 1);
                if heap.trace_mirrors { frame_time::count(kind, 1); }
            }
            if same {
                // Same allocation and raw view: only the three entry words can change.
                if heap.trace_mirrors { frame_time::count(Metric::BufferFast, 1); }
                return;
            }
            // A fresh or different slot: an index and an increment when the heap already holds
            // the buffer (the pool of every view into it); the Metal object is retained only the
            // first time. Counts still describe every slot, and commit_set reconciles from them.
            let made = Cell::new(false);
            let change = dense.set(index, Some(id), || { made.set(true); allocation.retain() }, None);
            heap.typed_views.set(dense.typed());
            let full = dense.retired() + dense.retired_views() > RETIRED_OBJECTS;
            drop(dense);
            if heap.trace_mirrors && !made.get() { frame_time::count(Metric::BufferRetained, 1); }
            if heap.set.is_some() && (change.added || change.retired) {
                heap.dirty.set(true);
                if change.retired && full { heap.commit_set(self); }
            }
            return;
        }
        if heap.fast_buffers {
            assert!(index < heap.count);
            let refs = heap.refs.borrow();
            if let Some(old) = refs.get(&index)
                && old.1.is_none() && std::ptr::eq(&*old.0, allocation)
            {
                // Same allocation and raw view: only the three entry words can change. In
                // particular no retain/release, Rc allocation, reference-count update or
                // residency-set mutation is needed. commit_set still reconciles every change.
                // SAFETY: the slot is in bounds; the title excludes descriptor writes during use.
                let p = unsafe { heap.buf.contents().as_ptr().cast::<[u64; 3]>().add(index) };
                if unsafe { p.read_volatile() } == words {
                    if heap.trace_mirrors { frame_time::count(Metric::EntrySame, 1); }
                } else {
                    unsafe { p.write_volatile(words); }
                    self.count(|c| c.entry_writes += 1);
                    if heap.trace_mirrors { frame_time::count(kind, 1); }
                }
                if heap.trace_mirrors { frame_time::count(Metric::BufferFast, 1); }
                return;
            }
        }
        // Fresh ring slots can share the heap's existing ownership of the pool. Weak entries
        // neither prolong allocations nor evade the desired-count/retirement invariant.
        let keep = if heap.fast_buffers {
            let mut raw = heap.raw_buffers.borrow_mut();
            if let Some(keep) = raw.get(&id).and_then(std::rc::Weak::upgrade) {
                if heap.trace_mirrors { frame_time::count(Metric::BufferRetained, 1); }
                keep
            } else {
                let keep = Rc::new((allocation.retain(), None));
                raw.insert(id, Rc::downgrade(&keep));
                keep
            }
        } else { Rc::new((allocation.retain(), None)) };
        // IRDescriptorTableSetBuffer(entry, gpu, metadata = size & 0xffffffff).
        self.heap_entry(heap, index, words, Some(keep), kind);
    }

    fn write_typed_buffer(
        &self,
        heap: &Heap,
        index: usize,
        buffer: &Buffer,
        title: *const u8,
        size: u32,
        format: Format,
        writable: bool,
    ) -> bool {
        debug_assert_eq!(heap.kind, HeapKind::Resource);
        let (bw, elem_bytes) = format.block();
        if bw != 1 || format.is_depth() || size == 0 {
            self.write_buffer(heap, index, buffer, title, size);
            return false;
        }
        let pf = pixel_format(format);
        let align = self
            .dev
            .minimumTextureBufferAlignmentForPixelFormat(pf)
            .max(elem_bytes);
        let offset = title as usize - buffer.base as usize;
        let max_elems = (buffer.buf.length().saturating_sub(offset)) / elem_bytes;
        let width = ((size as usize) / elem_bytes).min(max_elems).min(1 << 26);
        if width == 0 {
            self.write_buffer(heap, index, buffer, title, size);
            return false;
        }
        if !offset.is_multiple_of(align) {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    offset = format!("{offset:#x}"),
                    align,
                    ?format,
                    size,
                    "typed buffer view offset is not aligned for MTLTextureTypeTextureBuffer; falling back to raw buffer entry (first time)"
                );
            } else {
                tracing::debug!(
                    offset = format!("{offset:#x}"),
                    align,
                    ?format,
                    size,
                    "typed buffer view offset is not aligned for MTLTextureTypeTextureBuffer"
                );
            }
            self.write_buffer(heap, index, buffer, title, size);
            return false;
        }
        let key = (offset, width, pf.0, writable);
        let mut views = buffer.views.borrow_mut();
        let tex = if let Some(tex) = views.get(&key) {
            frame_time::count(Metric::TexbufCached, 1);
            tex.clone()
        } else {
            let td = MTLTextureDescriptor::new();
            td.setTextureType(MTLTextureType::TypeTextureBuffer);
            td.setPixelFormat(pf);
            // SAFETY: `width` is in `1..=(1 << 26)` and fits inside `buffer.buf`.
            unsafe {
                td.setWidth(width);
            }
            td.setStorageMode(MTLStorageMode::Shared);
            let mut usage = MTLTextureUsage::ShaderRead;
            if writable {
                usage |= MTLTextureUsage::ShaderWrite;
            }
            td.setUsage(usage);
            let Some(tex) = buffer
                .buf
                .newTextureWithDescriptor_offset_bytesPerRow(&td, offset, width * elem_bytes)
            else {
                self.write_buffer(heap, index, buffer, title, size);
                return false;
            };
            frame_time::count(Metric::TexbufMade, 1);
            if views.len() >= BUFFER_VIEWS {
                views.clear();
            }
            views.insert(key, tex.clone());
            tex
        };
        drop(views);
        let gpu = self.gpu_address(buffer, title);
        let tex_id = resource_id(tex.gpuResourceID());
        let keep_buf: Res = ProtocolObject::from_ref(&*buffer.buf).retain();
        self.heap_entry(
            heap,
            index,
            [gpu, tex_id, size as u64],
            Some(Rc::new((keep_buf, Some(tex)))),
            Metric::MirrorTyped,
        );
        true
    }

    fn write_texture(&self, heap: &Heap, index: usize, t: &Texture, min_lod: f32) {
        debug_assert_eq!(heap.kind, HeapKind::Resource);
        let id = resource_id(t.tex.gpuResourceID());
        let keep: Res = ProtocolObject::from_ref(&*t.tex).retain();
        // IRDescriptorTableSetTexture(entry, tex, minLODClamp, metadata = 0).
        self.heap_entry(heap, index, [0, id, min_lod.to_bits() as u64], Some(Rc::new((keep, None))), Metric::MirrorImage);
    }

    fn write_sampler(&self, heap: &Heap, index: usize, s: &Sampler, lod_bias: f32) {
        debug_assert_eq!(heap.kind, HeapKind::Sampler);
        let id = resource_id(s.0.gpuResourceID());
        // IRDescriptorTableSetSampler(entry, sampler, lodBias): samplers need no residency.
        self.heap_entry(heap, index, [id, 0, lod_bias.to_bits() as u64], None, Metric::MirrorSampler);
    }

    fn clear_entry(&self, heap: &Heap, index: usize) {
        self.heap_entry(heap, index, [0; 3], None, Metric::MirrorNull);
    }

    fn heap_typed_views(&self, heap: &Heap) -> usize {
        heap.typed_views.get()
    }

    fn heap_allocations(&self, heap: &Heap, last_used: &[u32]) -> Option<Vec<HeapAllocation>> {
        use objc2_foundation::NSObjectProtocol;
        heap.set.as_ref()?;
        // Each allocation's latest use over the entries naming it.
        let mut latest = std::collections::HashMap::<usize, u32>::new();
        heap.each_entry(|index, ptr| {
            let used = last_used.get(index).copied().unwrap_or(0);
            let e = latest.entry(ptr).or_insert(0);
            *e = (*e).max(used);
        });
        // A view's (or a texture over a buffer's) memory is its parent's: count the allocation under
        // it, once, with the latest use over every object made of it.
        let mut under = std::collections::HashMap::<usize, HeapAllocation>::new();
        heap.each_allocation(|id, r| {
            let used = latest.get(&id).copied().unwrap_or(0);
            let texture = r.respondsToSelector(objc2::sel!(textureType));
            let root: Res = if texture {
                // SAFETY: it answers `textureType`, so it is an MTLTexture.
                let mut t: Retained<ProtocolObject<dyn MTLTexture>> = unsafe { Retained::cast_unchecked(r.clone()) };
                while let Some(p) = t.parentTexture() {
                    t = p;
                }
                match t.buffer() {
                    Some(b) => ProtocolObject::from_retained(b),
                    None => ProtocolObject::from_retained(t),
                }
            } else {
                r.clone()
            };
            let rid = (&*root as *const ProtocolObject<dyn MTLResource>).cast::<()>() as usize;
            under
                .entry(rid)
                .and_modify(|a| a.latest = a.latest.max(used))
                .or_insert(HeapAllocation { id: rid, bytes: MTLResource::allocatedSize(&*root) as u64, latest: used, texture });
        });
        Some(under.into_values().collect())
    }

    fn texture_bytes(&self, t: &Texture) -> u64 {
        MTLResource::allocatedSize(&*t.tex) as u64
    }

    fn create_texture(&self, d: &TextureDesc) -> Result<Texture, String> {
        let td = MTLTextureDescriptor::new();
        td.setTextureType(tex_type(d.dim));
        td.setPixelFormat(pixel_format(d.format));
        // SAFETY: plain size properties of a fresh descriptor; Metal validates them at creation.
        unsafe {
            td.setWidth(d.width);
            td.setHeight(d.height);
            td.setDepth(if d.dim == TexDim::D3 {
                d.depth_or_layers
            } else {
                1
            });
            td.setArrayLength(if d.dim == TexDim::D2Array {
                d.depth_or_layers
            } else {
                1
            });
            td.setMipmapLevelCount(d.mips);
        }
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView;
        if d.unordered_access {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        if d.render_target {
            usage |= MTLTextureUsage::RenderTarget;
        }
        td.setUsage(usage);
        td.setStorageMode(MTLStorageMode::Shared);
        let tex = self
            .dev
            .newTextureWithDescriptor(&td)
            .ok_or("newTextureWithDescriptor failed")?;
        Ok(Texture { tex, format: d.format, present_view: Default::default(), views: Default::default() })
    }

    fn create_texture_view(&self, t: &Texture, v: &ViewDesc) -> Result<Texture, String> {
        let ty = t.tex.textureType();
        // A depth texture cannot be viewed as a color format (D3D's R32_FLOAT
        // over a depth buffer): it keeps its own, and shaders read it as the
        // float it holds.
        let source = t.tex.pixelFormat();
        let format = if is_depth(source) {
            source
        } else {
            pixel_format(v.format)
        };
        let title_format = if is_depth(source) { t.format } else { v.format };
        // Metal asserts on a format it cannot cast the texture to, too: a title view left in a
        // heap over a resource object the title has since made anew, with another format.
        if format != source && !castable(t.format, v.format) {
            return Err(format!("a {:?} view of a {:?} texture", v.format, t.format));
        }
        // Metal asserts (it does not return nil) on a range past the texture, which a
        // title view asks for over a texture we made with fewer levels or layers (one
        // over title memory has one level): the range is cut to what is there.
        let fit = |(start, len): (usize, usize), count: usize| {
            (start < count).then(|| NSRange::new(start, len.min(count - start)))
        };
        let slices = if ty == MTLTextureType::Type2DArray {
            t.tex.arrayLength()
        } else {
            1
        };
        let (Some(levels), Some(layers)) =
            (fit(v.mips, t.tex.mipmapLevelCount()), fit(v.slices, slices))
        else {
            return Err(format!(
                "view {:?} {:?} is past the texture",
                v.mips, v.slices
            ));
        };
        let key = (format.0, ty.0, levels.location, levels.length, layers.location, layers.length);
        let mut views = t.views.borrow_mut();
        let tex = if let Some(tex) = views.get(&key) {
            frame_time::count(Metric::ViewCached, 1);
            tex.clone()
        } else {
            // SAFETY: the ranges are inside the texture.
            let tex = unsafe {
                t.tex
                    .newTextureViewWithPixelFormat_textureType_levels_slices(format, ty, levels, layers)
            }
            .ok_or("newTextureView refused (format not view-compatible)")?;
            frame_time::count(Metric::ViewMade, 1);
            if views.len() >= TEXTURE_VIEWS {
                views.clear();
            }
            views.insert(key, tex.clone());
            tex
        };
        Ok(Texture { tex, format: title_format, present_view: Default::default(), views: Default::default() })
    }

    fn create_texture_over_memory(
        &self,
        buffer: &Buffer,
        title: *const u8,
        d: &TextureDesc,
        row_pitch: usize,
    ) -> Result<Texture, String> {
        // SAFETY: plain sizes; Metal validates them at creation.
        let td = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                pixel_format(d.format),
                d.width,
                d.height,
                false,
            )
        };
        td.setStorageMode(MTLStorageMode::Shared);
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView;
        if d.unordered_access {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        td.setUsage(usage);
        let offset = title as usize - buffer.base as usize;
        let tex = buffer
            .buf
            .newTextureWithDescriptor_offset_bytesPerRow(&td, offset, row_pitch)
            .ok_or("buffer-backed texture refused (offset, pitch or format)")?;
        Ok(Texture { tex, format: d.format, present_view: Default::default(), views: Default::default() })
    }

    fn linear_alignment(&self, f: Format) -> usize {
        self.dev
            .minimumLinearTextureAlignmentForPixelFormat(pixel_format(f))
    }

    fn upload_texture(
        &self,
        t: &Texture,
        mip: usize,
        slice: usize,
        data: &[u8],
        row_pitch: usize,
        slice_pitch: usize,
    ) {
        let (w, h, d) = (
            (t.tex.width() >> mip).max(1),
            (t.tex.height() >> mip).max(1),
            (t.tex.depth() >> mip).max(1),
        );
        if !transfer_fits("upload_texture", t, mip, [w, h, d], row_pitch, slice_pitch, data.len()) {
            return;
        }
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: size([w, h, d]),
        };
        // SAFETY: `data` covers the region at the given pitches (callers size it by `Format::row_bytes`).
        unsafe {
            t.tex
                .replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    mip,
                    slice,
                    NonNull::new(data.as_ptr().cast_mut().cast()).unwrap(),
                    row_pitch,
                    slice_pitch,
                )
        };
    }

    fn read_texture(
        &self,
        t: &Texture,
        mip: usize,
        slice: usize,
        out: &mut [u8],
        row_pitch: usize,
        slice_pitch: usize,
    ) {
        let (w, h, d) = (
            (t.tex.width() >> mip).max(1),
            (t.tex.height() >> mip).max(1),
            (t.tex.depth() >> mip).max(1),
        );
        if !transfer_fits("read_texture", t, mip, [w, h, d], row_pitch, slice_pitch, out.len()) {
            return;
        }
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: size([w, h, d]),
        };
        // SAFETY: `out` covers the region at the given pitches.
        unsafe {
            t.tex
                .getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                    NonNull::new(out.as_mut_ptr().cast()).unwrap(),
                    row_pitch,
                    slice_pitch,
                    region,
                    mip,
                    slice,
                )
        };
    }

    fn create_sampler(&self, d: &SamplerDesc) -> Result<Sampler, String> {
        if let Some((_, s)) = self.samplers.borrow().iter().find(|(k, _)| k == d) {
            return Ok(Sampler(s.clone()));
        }
        let sd = dynamic_sampler_descriptor(d);
        let s = self
            .dev
            .newSamplerStateWithDescriptor(&sd)
            .ok_or("newSamplerState failed")?;
        self.samplers.borrow_mut().push((*d, s.clone()));
        Ok(Sampler(s))
    }
}

/// Identity of all stages and state owned by an off-thread render compile.
fn render_key(d: &RenderPipelineDesc<'_>) -> [u8; 32] {
    let state = format!(
        "{} {:?}",
        d.vs.len(),
        (d.vs_layout, d.ps_layout, d.attributes, d.streams, d.color, d.depth, d.depth_test, d.cull, d.color_state, d.dual_source, d.stencil, d.raster)
    );
    let mut key = blake3::Hasher::new();
    key.update(d.vs).update(d.ps.unwrap_or_default()).update(state.as_bytes());
    if let Some(m) = &d.mesh {
        key.update(b"mesh").update(&m.mesh.len().to_le_bytes()).update(m.mesh);
        for pass in m.more {
            key.update(b"pass").update(&pass.len().to_le_bytes()).update(pass);
        }
        key.update(m.amplification.unwrap_or_default());
        key.update(format!("{:?} {:?} {:?}", m.layout, m.amplification.map(<[u8]>::len), m.amplification_layout).as_bytes());
    }
    *key.finalize().as_bytes()
}

/// What a compiler crash is remembered by: the two stages' DXIL (the crash is in compiling them,
/// whatever the targets and state).
fn crash_key(d: &RenderPipelineDesc<'_>) -> String {
    match &d.mesh {
        Some(m) => mesh_crash_key(d, m),
        None => crate::disk::key(&[d.vs, d.ps.unwrap_or_default()]),
    }
}

/// [`crash_key`] of one mesh stage: each pass of a split mesh program is compiled, and
/// remembered, by itself.
fn mesh_crash_key(d: &RenderPipelineDesc<'_>, m: &MeshPipelineDesc<'_>) -> String {
    crate::disk::key(&[m.mesh, m.amplification.unwrap_or_default(), d.ps.unwrap_or_default()])
}

impl GraphicsBackend for Metal {
    type RenderPipeline = RenderPipeline;

    fn create_render_pipeline(&self, d: &RenderPipelineDesc<'_>) -> Result<RenderPipeline, String> {
        let (pso, mesh) = self.compiler()?.render(d)?;
        self.finish_render(d, pso, mesh)
    }

    /// As `create_compute_pipeline_within`: converted and compiled on a `pipeline-compile`
    /// thread, and picked up by the same call later when `wait` was not enough.
    fn create_render_pipeline_within(
        &self,
        d: &RenderPipelineDesc<'_>,
        wait: std::time::Duration,
    ) -> Result<Option<RenderPipeline>, String> {
        let key = render_key(d);
        let made = crate::metal::within(&self.compiling_render, key, wait, || {
            let (compiler, owned) = (self.compiler()?, OwnedDesc::of(d));
            Ok(move || compiler.render(&owned.desc()))
        })?;
        made.map(|made| made.and_then(|(pso, mesh)| self.finish_render(d, pso, mesh))).transpose()
    }

    fn render_pipeline_targetless(&self, p: &RenderPipeline) -> bool {
        p.targetless
    }

    fn name_render_pipeline(&self, p: &RenderPipeline, name: &str) {
        if self.trace_fails || self.gate_stats.is_some() {
            let key = Retained::as_ptr(&p.pso) as *const u8 as usize;
            self.pipe_names.borrow_mut().insert(key, name.to_string());
        }
    }

    fn submit_passes(&self, queue: &Queue, work: &PassSubmission<'_, Self>) -> Result<(), SubmitFailure> {
        self.let_go_of_done();
        self.log_gate_stats();
        if self.trace_fails {
            report_in_flight();
            self.trace_enc.borrow_mut().clear();
            self.trace_args.borrow_mut().clear();
        }
        let mut cb = self.new_command_buffer(queue).map_err(SubmitFailure::uncommitted)?;
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut label = String::new();
        if self.trace_fails {
            label = format!(
                "q{:x} #{} t={:.3} waits={} signals={} passes={}",
                Retained::as_ptr(&queue.0) as *const u8 as usize,
                serial,
                crate::metal::since_start(),
                work.waits.len(),
                work.signals.len(),
                work.passes.len()
            );
            cb.setLabel(Some(&objc2_foundation::NSString::from_str(&label)));
        }
        for (f, v) in work.waits {
            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*f.0), *v);
        }
        // With residency sets the encoders do not name what they reach through a heap (nor, with
        // the pool's set on the queue, the title's buffers), so Metal cannot order them: this
        // command buffer starts after the previous one finished, and each encoder after the one
        // before.
        let set_on_queue = queue.1.get();
        let order = set_on_queue.then(|| {
            let n = queue.3.get();
            queue.3.set(n + 1);
            if n > 0 {
                cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*queue.2), n);
            }
            Order {
                fence: self.dev.newFence(),
                started: Cell::new(false),
            }
        });
        let order = order.as_ref().and_then(|o| o.fence.as_ref().map(|f| (&**f, &o.started)));
        // A pass that cannot be encoded must not take the submission's fence signals and
        // bottom-of-pipe value writes with it: the title's compute queues wait for those writes,
        // and a lost one is `Unsignaled CrossQueueBarrier` (the title then dies). The failing pass
        // is dropped, the command buffer still commits, and the caller is told which passes were
        // dropped, so it can pay only their lost work.
        let mut refused: Vec<usize> = Vec::new();
        let mut first_error: Option<String> = None;
        let mut attached_heaps = Vec::new();
        // A submission's passes can go in command buffers of at most `passes_per_buffer` passes
        // (`VICTORY_PASSES_PER_BUFFER`; default 0: all in one). macOS stops a command buffer that
        // runs too long on the GPU (`kIOGPUCommandBufferCallbackErrorTimeout`): a world
        // submission of 72 to 80 passes ran 6 to 21 s with the ray-traced passes on, while none of
        // its passes alone took more than 1.4 s. The earlier pieces are committed as they fill.
        // Each piece owes its own value writes, and raises the submission's piece event for the
        // next piece, which waits for it; the last piece owes the fence signals and the order
        // event. A piece the GPU fails pays its own share on the CPU, once ([`Owed`]), so the
        // later pieces still run and nothing is paid twice.
        let per = self.passes_per_buffer;
        let pieces = if per > 0 && work.passes.len() > per {
            Some(track_event(self.dev.newSharedEvent().ok_or_else(|| SubmitFailure::uncommitted("newSharedEvent failed"))?))
        } else {
            None
        };
        let queue_id = Retained::as_ptr(&queue.0) as *const u8 as usize;
        let mut first = 0;
        for (n, pass) in work.passes.iter().enumerate() {
            let encoded = match pass {
                Pass::Compute(dispatches) => {
                    let _encode = Timer::counted(Metric::EncCompute, Metric::PassesCompute);
                    self.encode_compute(&cb, dispatches, set_on_queue, order, &mut attached_heaps)
                }
                Pass::Render(rp) => {
                    let _encode = Timer::counted(Metric::EncRender, Metric::PassesRender);
                    self.encode_render(&cb, rp, set_on_queue, order, &mut attached_heaps)
                }
                Pass::Blit(ops) => {
                    let _encode = Timer::counted(Metric::EncBlit, Metric::PassesBlit);
                    self.encode_blit(&cb, ops, order)
                }
                Pass::ClearColorRects { target, color, rects } => {
                    // Include the scratch render and copies here, without timing them twice.
                    let _encode = Timer::counted(Metric::EncRender, Metric::PassesClear);
                    self.encode_color_clear_rects(&cb, target, *color, rects, set_on_queue, order, &mut attached_heaps)
                }
            };
            if let Err(e) = encoded {
                // A pass that cannot be encoded (a render target with no attachments, a piece
                // whose root arguments were never set) is dropped; the passes after it still run,
                // so one bad pass does not cost the frame's whole picture.
                refused.push(n);
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
            if let Some(event) = &pieces
                && (n + 1) % per == 0
                && n + 1 < work.passes.len()
            {
                let piece = ((n + 1) / per) as u64;
                cb.encodeSignalEvent_value(ProtocolObject::from_ref(&**event), piece);
                let owed = Owed::of(vec![(event.clone(), piece)], &work.passes[first..=n], 0);
                // The trace names what this piece holds, not the whole submission.
                let tinfo = self.trace_fails.then(|| {
                    let enc = std::mem::take(&mut *self.trace_enc.borrow_mut()).join(" | ");
                    (format!("{label} piece {piece} | {enc}"), std::mem::take(&mut *self.trace_args.borrow_mut()))
                });
                self.commit_piece(queue, &cb, owed, piece, tinfo);
                first = n + 1;
                cb = match self.next_piece(queue, work.waits, &mut attached_heaps, Some((&**event, piece))) {
                    Ok(cb) => cb,
                    Err(e) => {
                        // The current piece never started: every pass from `first` on was lost,
                        // and the earlier pieces did commit.
                        refused.extend(first..work.passes.len());
                        return Err(SubmitFailure {
                            committed: true,
                            refused,
                            error: e,
                        });
                    }
                };
                if self.trace_fails {
                    let piece = format!("{label} piece {}", piece + 1);
                    cb.setLabel(Some(&objc2_foundation::NSString::from_str(&piece)));
                }
            }
        }
        if set_on_queue {
            cb.encodeSignalEvent_value(ProtocolObject::from_ref(&*queue.2), queue.3.get());
        }
        for (f, v) in work.signals {
            cb.encodeSignalEvent_value(ProtocolObject::from_ref(&*f.0), *v);
        }
        let mut events = Vec::with_capacity(work.signals.len() + 1);
        if set_on_queue {
            events.push((queue.2.clone(), queue.3.get()));
        }
        events.extend(work.signals.iter().map(|(f, v)| (f.0.clone(), *v)));
        let owed = std::sync::Arc::new(Owed::of(events, &work.passes[first..], work.signals.len()));
        flights().push_back(Flight { serial, queue: queue_id, since: std::time::Instant::now(), owed: owed.clone() });
        // A command buffer the GPU fails is logged (the first few), not silently dropped.
        let trace = self.trace_fails;
        let tinfo = trace.then(|| (serial, label.clone(), self.trace_enc.borrow().join(" | ")));
        self.trace_commit(serial, queue_id, &label);
        if let Some((s, l, enc)) = &tinfo {
            IN_FLIGHT.lock().unwrap().insert(*s, (std::time::Instant::now(), format!("{l} | {enc}")));
            GATED.lock().unwrap().insert(*s, std::mem::take(&mut *self.trace_args.borrow_mut()));
        }
        // The handler is installed before commit, but latency must start at commit, not before
        // allocating/copying the block. This shared timestamp exists only with tracing enabled.
        let committed = frame_time::enabled().then(|| {
            std::sync::Arc::new(std::sync::OnceLock::<std::time::Instant>::new())
        });
        let completed_commit = committed.clone();
        let failed = block2::RcBlock::new(move |done: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            // SAFETY: Metal passes the command buffer that completed.
            let done = unsafe { done.as_ref() };
            if let Some(started) = completed_commit.as_ref().and_then(|t| t.get()) {
                frame_time::completed(
                    started.elapsed(),
                    done.GPUStartTime(),
                    done.GPUEndTime(),
                    done.kernelStartTime(),
                    done.kernelEndTime(),
                );
            }
            trace_done(serial, done);
            landed(serial, queue_id, done.error().is_some());
            if let Some((s, l, enc)) = &tinfo {
                let started = IN_FLIGHT.lock().unwrap().remove(s).map(|(t, _)| t);
                gated_args(*s, true);
                if done.error().is_none()
                    && let Some(t) = started
                    && t.elapsed().as_secs_f64() > 1.0
                    && TRACE_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 100
                {
                    tracing::warn!(
                        "metal: slow command buffer {l} took {:.3}s (gpu {:.3}s, kernel {:.3}s) | {enc}",
                        t.elapsed().as_secs_f64(),
                        done.GPUEndTime() - done.GPUStartTime(),
                        done.kernelEndTime() - done.kernelStartTime()
                    );
                }
            }
            if let Some(e) = done.error() {
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < if trace { 200 } else { 20 } {
                    if trace {
                        trace_failure(done, &e);
                    } else {
                        eprintln!("metal: command buffer failed: {e:?} ({})", e.localizedDescription());
                    }
                }
                owed.recover();
            }
        });
        // SAFETY: the block is retained by Metal until it has run.
        unsafe { cb.addCompletedHandler(&*failed as *const _ as *mut _) };
        let commit = Timer::counted(Metric::Commit, Metric::Cbs);
        if let (Some(stamp), Some(started)) = (&committed, commit.started()) {
            let _ = stamp.set(started);
        }
        crate::metal::residency_timed("cb_commit", None, || cb.commit());
        drop(commit);
        self.committed.set(self.committed.get() + 1);
        match first_error {
            Some(e) => Err(SubmitFailure {
                committed: true,
                refused,
                error: e,
            }),
            None => Ok(()),
        }
    }
}

impl RenderPipeline {
    /// A mesh pipeline's thread-group size (else `None`: a vertex pipeline).
    pub fn mesh_threads(&self) -> Option<[usize; 3]> {
        self.mesh.as_ref().map(|m| [m.threads.width, m.threads.height, m.threads.depth])
    }

    /// The passes a mesh draw of this pipeline runs: more than one for a split mesh program
    /// (`MeshPipelineDesc::more`); 0 for a vertex pipeline.
    pub fn mesh_passes(&self) -> usize {
        self.mesh.as_ref().map_or(0, |m| 1 + m.more.len())
    }
}

/// The raster sample count of a pipeline made for no target, and of the pass without attachments
/// its draws run in. The title draws such passes with conservative rasterization (light volumes
/// into a grid of screen tiles, 32 pixels a tile: a tile a volume touches anywhere takes the
/// light), which Metal does not have. Four sample positions a pixel stand in for it: the pixel
/// shader runs once for a pixel any of them covers.
// shortcut: a primitive that crosses a pixel between all four sample positions is missed, where
// conservative rasterization shades it; expand the primitives in a geometry step if a title's
// lights are seen to drop out at the edges of their volumes.
fn targetless_samples(dev: &ProtocolObject<dyn MTLDevice>) -> usize {
    if dev.supportsTextureSampleCount(4) { 4 } else { 1 }
}

/// Formats, write masks and blending of the colour targets.
fn color_attachments(arr: &MTLRenderPipelineColorAttachmentDescriptorArray, d: &RenderPipelineDesc<'_>) {
    for (i, f) in d.color.iter().enumerate() {
        // SAFETY: colour attachment index below 8.
        let a = unsafe { arr.objectAtIndexedSubscript(i) };
        a.setPixelFormat(pixel_format(*f));
        let state = d.color_state.get(i).copied().unwrap_or_default();
        let mut mask = MTLColorWriteMask::empty();
        for (bit, channel) in [
            MTLColorWriteMask::Red,
            MTLColorWriteMask::Green,
            MTLColorWriteMask::Blue,
            MTLColorWriteMask::Alpha,
        ]
        .into_iter()
        .enumerate()
        {
            if state.write_mask >> bit & 1 != 0 {
                mask |= channel;
            }
        }
        a.setWriteMask(mask);
        if let Some(b) = state.blend.filter(|_| !f.is_integer()) {
            a.setBlendingEnabled(true);
            a.setSourceRGBBlendFactor(blend_factor(b.src));
            a.setDestinationRGBBlendFactor(blend_factor(b.dst));
            a.setRgbBlendOperation(blend_op(b.op));
            a.setSourceAlphaBlendFactor(blend_factor(b.src_alpha));
            a.setDestinationAlphaBlendFactor(blend_factor(b.dst_alpha));
            a.setAlphaBlendOperation(blend_op(b.op_alpha));
        }
    }
}

impl Metal {
    /// Debugging: level 0 of a depth texture's depth plane as the GPU has it now (32-bit floats,
    /// or 16-bit normalised values), read through a command buffer of its own that this waits
    /// for. False if it was not read. A depth texture gives no bytes to `read_texture`.
    pub fn debug_read_depth(&self, t: &Texture, out: &mut [u8]) -> bool {
        let (texel, options) = match t.tex.pixelFormat() {
            MTLPixelFormat::Depth32Float_Stencil8 => (4, MTLBlitOption::DepthFromDepthStencil),
            MTLPixelFormat::Depth32Float => (4, MTLBlitOption::None),
            MTLPixelFormat::Depth16Unorm => (2, MTLBlitOption::None),
            _ => return false,
        };
        let (w, h) = (t.tex.width(), t.tex.height());
        let len = w * h * texel;
        if len == 0 || out.len() < len {
            return false;
        }
        let Some(buf) = self.dev.newBufferWithLength_options(len.next_multiple_of(4096), MTLResourceOptions::StorageModeShared) else { return false };
        let Some(queue) = self.dev.newCommandQueue() else { return false };
        let Some(cb) = queue.commandBuffer() else { return false };
        let Some(e) = cb.blitCommandEncoder() else { return false };
        // SAFETY: the region is level 0 of the texture, and its rows fit the buffer.
        unsafe {
            e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                &t.tex,
                0,
                0,
                origin([0, 0, 0]),
                size([w, h, 1]),
                &buf,
                0,
                w * texel,
                len,
                options,
            )
        };
        e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return false;
        }
        // SAFETY: the buffer holds `len` bytes the GPU has finished writing.
        out[..len].copy_from_slice(unsafe { std::slice::from_raw_parts(buf.contents().as_ptr().cast::<u8>(), len) });
        true
    }
}

impl Metal {
    /// Commits piece `piece` (from 1) of a submission, not its last ([`GraphicsBackend::submit_passes`]),
    /// with what it owes: its value writes and the piece event raised to `piece`, which the next
    /// piece waits for. One the GPU fails is logged as the last piece's failure is and pays that on
    /// the CPU, so the next piece runs. It is in flight as any command buffer is: one Metal never
    /// reports on is paid by [`recover_lost`].
    fn commit_piece(
        &self,
        queue: &Queue,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        owed: Owed,
        piece: u64,
        tinfo: Option<(String, Vec<(String, usize, usize)>)>,
    ) {
        let owed = std::sync::Arc::new(owed);
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let queue_id = Retained::as_ptr(&queue.0) as *const u8 as usize;
        // Tests: a piece the GPU drops, as one it fails (its passes and its event signal not run).
        #[cfg(test)]
        let dropped = DROP_PIECE.with(|d| d.get()) == Some(piece);
        #[cfg(not(test))]
        let dropped = {
            let _ = piece;
            false
        };
        let empty;
        let cb = if dropped {
            empty = self.new_command_buffer(queue).expect("a command buffer");
            &*empty
        } else {
            cb
        };
        flights().push_back(Flight { serial, queue: queue_id, since: std::time::Instant::now(), owed: owed.clone() });
        self.trace_commit(serial, queue_id, &format!("piece {piece}"));
        let trace = self.trace_fails;
        let tinfo = tinfo.map(|(d, args)| {
            IN_FLIGHT.lock().unwrap().insert(serial, (std::time::Instant::now(), d.clone()));
            GATED.lock().unwrap().insert(serial, args);
            d
        });
        let committed = frame_time::enabled().then(|| {
            std::sync::Arc::new(std::sync::OnceLock::<std::time::Instant>::new())
        });
        let completed_commit = committed.clone();
        let failed = block2::RcBlock::new(move |done: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            // SAFETY: Metal passes the command buffer that completed.
            let done = unsafe { done.as_ref() };
            if let Some(started) = completed_commit.as_ref().and_then(|t| t.get()) {
                frame_time::completed(
                    started.elapsed(),
                    done.GPUStartTime(),
                    done.GPUEndTime(),
                    done.kernelStartTime(),
                    done.kernelEndTime(),
                );
            }
            trace_done(serial, done);
            let error = done.error();
            landed(serial, queue_id, dropped || error.is_some());
            // A piece is traced as a whole command buffer is: slow ones, and what a failed one held
            // (a time-out carries no encoder states, so the label alone does not name the pass).
            if let Some(d) = &tinfo {
                let started = IN_FLIGHT.lock().unwrap().remove(&serial).map(|(t, _)| t);
                let secs = started.map_or(0.0, |t| t.elapsed().as_secs_f64());
                let args = gated_args(serial, true);
                if (error.is_some() || secs > 1.0)
                    && TRACE_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 100
                {
                    tracing::warn!(
                        "metal: {} piece {d} took {secs:.3}s (gpu {:.3}s, kernel {:.3}s){args}",
                        if error.is_some() { "failed" } else { "slow" },
                        done.GPUEndTime() - done.GPUStartTime(),
                        done.kernelEndTime() - done.kernelStartTime()
                    );
                }
            }
            if let Some(e) = &error
                && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < if trace { 200 } else { 20 }
            {
                if trace {
                    trace_failure(done, e);
                } else {
                    eprintln!("metal: command buffer failed: {e:?} ({})", e.localizedDescription());
                }
            }
            if dropped || error.is_some() {
                owed.recover();
            }
        });
        // SAFETY: the block is retained by Metal until it has run.
        unsafe { cb.addCompletedHandler(&*failed as *const _ as *mut _) };
        let commit = Timer::counted(Metric::Commit, Metric::Cbs);
        if let (Some(stamp), Some(started)) = (&committed, commit.started()) {
            let _ = stamp.set(started);
        }
        crate::metal::residency_timed("cb_commit", None, || cb.commit());
        drop(commit);
        self.committed.set(self.committed.get() + 1);
    }

    /// The pipeline of a compiled state `pso`: what the passes set beside it.
    fn finish_render(&self, d: &RenderPipelineDesc<'_>, pso: RenderState, mesh: Option<MeshDispatch>) -> Result<RenderPipeline, String> {
        let has_stencil = d.depth == Some(Format::D32FloatS8);
        let depth = (d.depth_test.is_some() || (d.stencil.is_some() && has_stencil)).then(|| {
            let dd = MTLDepthStencilDescriptor::new();
            let (write, cmp) = d.depth_test.unwrap_or((false, Compare::Always));
            dd.setDepthCompareFunction(compare(cmp));
            dd.setDepthWriteEnabled(write);
            if let Some(s) = d.stencil.filter(|_| has_stencil) {
                dd.setFrontFaceStencil(Some(&stencil_face(&s.front, &s)));
                dd.setBackFaceStencil(Some(&stencil_face(&s.back, &s)));
            }
            self.dev
                .newDepthStencilStateWithDescriptor(&dd)
                .expect("depth state")
        });
        let cull = match d.cull {
            Cull::None => MTLCullMode::None,
            Cull::Front => MTLCullMode::Front,
            Cull::Back => MTLCullMode::Back,
        };
        Ok(RenderPipeline {
            pso,
            depth,
            cull,
            raster: d.raster,
            stencil: has_stencil,
            // (A mesh pipeline is not given the pass's sample count: it stays a pipeline with targets.)
            targetless: d.mesh.is_none() && d.color.is_empty() && d.depth.is_none(),
            mesh,
            vs_static: self.static_samplers(d.mesh.as_ref().map_or(d.vs_layout, |m| m.layout))?,
            object_static: match d.mesh.as_ref().filter(|m| m.amplification.is_some()) {
                Some(m) => self.static_samplers(m.amplification_layout)?,
                None => None,
            },
            ps_static: if d.ps.is_some() {
                self.static_samplers(d.ps_layout)?
            } else {
                None
            },
        })
    }
}

pub(crate) type RenderState = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;

/// The mesh and object stages retained until their compile thread finishes.
struct OwnedMesh {
    mesh: Vec<u8>,
    more: Vec<Vec<u8>>,
    layout: Option<victory_backend::backend::RootLayout>,
    amplification: Option<Vec<u8>>,
    amplification_layout: Option<victory_backend::backend::RootLayout>,
}

impl OwnedMesh {
    fn of(d: &MeshPipelineDesc<'_>) -> Self {
        Self {
            mesh: d.mesh.to_vec(),
            more: d.more.to_vec(),
            layout: d.layout.cloned(),
            amplification: d.amplification.map(<[u8]>::to_vec),
            amplification_layout: d.amplification_layout.cloned(),
        }
    }

    fn desc(&self) -> MeshPipelineDesc<'_> {
        MeshPipelineDesc {
            mesh: &self.mesh,
            layout: self.layout.as_ref(),
            amplification: self.amplification.as_deref(),
            amplification_layout: self.amplification_layout.as_ref(),
            more: &self.more,
        }
    }
}

/// A `RenderPipelineDesc` that owns what it names, for the compile thread.
struct OwnedDesc {
    mesh: Option<OwnedMesh>,
    vs: Vec<u8>,
    ps: Option<Vec<u8>>,
    vs_layout: Option<victory_backend::backend::RootLayout>,
    ps_layout: Option<victory_backend::backend::RootLayout>,
    attributes: Vec<VertexAttr>,
    streams: Vec<VertexStream>,
    color: Vec<Format>,
    color_state: Vec<ColorState>,
    depth: Option<Format>,
    depth_test: Option<(bool, Compare)>,
    cull: Cull,
    dual_source: bool,
    stencil: Option<StencilState>,
    raster: Raster,
}

impl OwnedDesc {
    fn of(d: &RenderPipelineDesc<'_>) -> OwnedDesc {
        OwnedDesc {
            mesh: d.mesh.as_ref().map(OwnedMesh::of),
            vs: d.vs.to_vec(),
            ps: d.ps.map(<[u8]>::to_vec),
            vs_layout: d.vs_layout.cloned(),
            ps_layout: d.ps_layout.cloned(),
            attributes: d.attributes.to_vec(),
            streams: d.streams.to_vec(),
            color: d.color.to_vec(),
            color_state: d.color_state.to_vec(),
            depth: d.depth,
            depth_test: d.depth_test,
            cull: d.cull,
            dual_source: d.dual_source,
            stencil: d.stencil,
            raster: d.raster,
        }
    }

    fn desc(&self) -> RenderPipelineDesc<'_> {
        RenderPipelineDesc {
            mesh: self.mesh.as_ref().map(OwnedMesh::desc),
            vs: &self.vs,
            ps: self.ps.as_deref(),
            vs_layout: self.vs_layout.as_ref(),
            ps_layout: self.ps_layout.as_ref(),
            attributes: &self.attributes,
            streams: &self.streams,
            color: &self.color,
            color_state: &self.color_state,
            depth: self.depth,
            depth_test: self.depth_test,
            cull: self.cull,
            dual_source: self.dual_source,
            stencil: self.stencil,
            raster: self.raster,
        }
    }
}

/// The part of a render pipeline Metal compiles: all of it may run on another thread.
impl crate::metal::Compiler {
    /// The converted function of a stage and the converter's stage number (`IRShaderStage`).
    fn stage_function(
        &self,
        dxil: &[u8],
        layout: Option<&victory_backend::backend::RootLayout>,
        dual_source: bool,
    ) -> Result<(i32, Retained<ProtocolObject<dyn MTLFunction>>), String> {
        let (stage, lib) = self.msc.convert_with(dxil, None, layout, dual_source).map_err(|e| e.to_string())?;
        let l = crate::metal::load_library(&self.dev, &lib)?;
        let name = l
            .functionNames()
            .iter()
            .next()
            .ok_or("converted library is empty")?;
        Ok((stage, l.newFunctionWithName(&name).ok_or("no function")?))
    }

    fn mesh_function(&self, dxil: &[u8], layout: Option<&victory_backend::backend::RootLayout>, amplification: bool)
        -> Result<(Retained<ProtocolObject<dyn MTLFunction>>, crate::msc::MeshInfo), String>
    {
        let (lib, reflected) = self.msc.convert_mesh_with(dxil, layout, amplification).map_err(|e| e.to_string())?;
        let library = crate::metal::load_library(&self.dev, &lib)?;
        let name = objc2_foundation::NSString::from_str(&reflected.entry);
        let f = library.newFunctionWithName(&name).ok_or("reflected mesh entry point is missing")?;
        Ok((f, reflected.info))
    }

    /// Both described and guessed mesh stages use the reflected mesh path on this thread.
    pub(crate) fn render(&self, d: &RenderPipelineDesc<'_>) -> Result<(RenderState, Option<MeshDispatch>), String> {
        if crate::disk::crashed(crash_key(d).as_bytes()) {
            return Err("Metal's compiler crashed on it on an earlier run".into());
        }
        // A mesh producer comes described (`d.mesh`, with an optional amplification stage: a
        // stream pipeline's stages) or as the stage in `d.vs` (a guessed vertex program that
        // exports primitives, translated as a mesh shader). Both are made by `mesh_pipeline`.
        let guessed;
        let mut vertex = None;
        let mesh_desc = match &d.mesh {
            Some(ms) => Some(ms),
            None => {
                let (stage, vf) = self.stage_function(d.vs, d.vs_layout, false)?;
                if stage == crate::msc::STAGE_MESH {
                    guessed = MeshPipelineDesc {
                        mesh: d.vs,
                        layout: d.vs_layout,
                        amplification: None,
                        amplification_layout: None,
                        more: &[],
                    };
                    Some(&guessed)
                } else {
                    vertex = Some(vf);
                    None
                }
            }
        };
        let (pso, mesh) = match (mesh_desc, vertex) {
            (Some(ms), _) => {
                let (pso, mut dispatch) = self.mesh_pipeline(d, ms)?;
                // The further passes of a split mesh program: the same pipeline of another
                // mesh function. Any of them failing fails the pipeline: a share of the
                // primitives alone must never be drawn.
                for pass in ms.more {
                    let pass = MeshPipelineDesc { mesh: pass, more: &[], ..*ms };
                    let (state, other) = self.mesh_pipeline(d, &pass)?;
                    if (other.threads, other.object_threads) != (dispatch.threads, dispatch.object_threads) {
                        return Err("the passes of a split mesh program differ in their thread groups".into());
                    }
                    dispatch.more.push(state);
                }
                (pso, Some(dispatch))
            }
            (None, Some(vf)) => {
                let ff = d.ps.map(|ps| self.stage_function(ps, d.ps_layout, d.needs_dual_source()).map(|f| f.1)).transpose()?;
                (self.vertex_pipeline(d, &vf, ff.as_deref())?, None)
            }
            (None, None) => unreachable!("a vertex function or a mesh stage"),
        };
        Ok((pso, mesh))
    }

    /// The render pipeline state of a vertex and fragment function, with the stage-in layout.
    fn vertex_pipeline(
        &self,
        d: &RenderPipelineDesc<'_>,
        vf: &ProtocolObject<dyn MTLFunction>,
        ff: Option<&ProtocolObject<dyn MTLFunction>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
        let pd = MTLRenderPipelineDescriptor::new();
        pd.setVertexFunction(Some(vf));
        pd.setFragmentFunction(ff);
        // The converter's stage-in function takes its inputs at attribute indices from
        // `kIRStageInAttributeStartIndex`, in input-signature order; buffers from `kIRVertexBufferBindPoint`.
        let vd = MTLVertexDescriptor::new();
        // SAFETY: indices are below the attribute and buffer limits for sane layouts; plain sizes.
        unsafe {
            for (i, a) in d.attributes.iter().enumerate() {
                let at = vd
                    .attributes()
                    .objectAtIndexedSubscript(STAGE_IN_ATTRIBUTES + i);
                at.setFormat(vertex_format(a.format));
                at.setOffset(a.offset);
                at.setBufferIndex(VERTEX_BUFFERS + a.slot);
            }
            for s in d.streams {
                let l = vd
                    .layouts()
                    .objectAtIndexedSubscript(VERTEX_BUFFERS + s.slot);
                l.setStride(s.stride);
                l.setStepFunction(if s.per_instance {
                    MTLVertexStepFunction::PerInstance
                } else {
                    MTLVertexStepFunction::PerVertex
                });
            }
        }
        pd.setVertexDescriptor(Some(&vd));
        color_attachments(&pd.colorAttachments(), d);
        if let Some(f) = d.depth {
            pd.setDepthAttachmentPixelFormat(pixel_format(f));
            if d.depth == Some(Format::D32FloatS8) {
                pd.setStencilAttachmentPixelFormat(pixel_format(f));
            }
        }
        if d.color.is_empty() && d.depth.is_none() {
            pd.setRasterSampleCount(targetless_samples(&self.dev));
        }
        let crash_key = crash_key(d);
        match &self.archive {
            Some(a) => a.render(&self.dev, &pd, &crash_key),
            None => crate::archive::compile_render(&self.dev, &pd, &crash_key),
        }
    }
}


/// What a submission owes the title once it completes: the order event, the title's fence signals
/// and its value writes (fills). A command buffer the GPU fails does none of them, so the queue and
/// every fence and barrier word waiting on it would hang; `recover` pays them from the CPU.
///
/// The fills retain their buffer wrappers so the pool keeps the title memory mapped.
pub(crate) struct Owed {
    recovery: std::sync::Once,
    events: Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
    /// Buffer, offset, length, byte.
    #[allow(clippy::type_complexity)]
    fills: Vec<(Retained<ProtocolObject<dyn MTLBuffer>>, usize, usize, u8)>,
    signals: usize,
}

impl Owed {
    pub(crate) fn new(queue: &Queue, work: &PassSubmission<'_, Metal>, set_on_queue: bool) -> Owed {
        let mut events = Vec::with_capacity(work.signals.len() + 1);
        if set_on_queue {
            events.push((queue.2.clone(), queue.3.get()));
        }
        events.extend(work.signals.iter().map(|(f, v)| (f.0.clone(), *v)));
        Owed::of(events, work.passes, work.signals.len())
    }

    /// What `passes` owe if they fail: their value writes, then `events` raised (`signals` of them
    /// the submission's fence signals, for the log).
    #[allow(clippy::type_complexity)]
    pub(crate) fn of(
        events: Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
        passes: &[Pass<'_, Metal>],
        signals: usize,
    ) -> Owed {
        let mut fills = Vec::new();
        for pass in passes {
            if let Pass::Blit(ops) = pass {
                for op in ops {
                    // shortcut: copies are not replayed (one may have partly run and its source
                    // may have changed since), when a title needs a failed copy's data upgrade
                    // to a replay that checks the source is unchanged.
                    if let BlitOp::Fill { dst, len, value } = op {
                        fills.push((dst.0.buf.clone(), dst.1 as usize - dst.0.base as usize, *len, *value));
                    }
                }
            }
        }
        Owed { recovery: std::sync::Once::new(), events, fills, signals }
    }

    /// Writes each fill, in order, then raises each event (never lowers it): a waiter that sees
    /// its fence reached reads the values the submission wrote before it.
    pub(crate) fn recover(&self) {
        self.recovery.call_once(|| {
            static WARNED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            for (buf, off, len, value) in &self.fills {
                // SAFETY: the range lies inside the wrapped title memory (the caller resolved it).
                unsafe { std::ptr::write_bytes(buf.contents().as_ptr().cast::<u8>().add(*off), *value, *len) };
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            for (e, v) in &self.events {
                if e.signaledValue() < *v {
                    e.setSignaledValue(*v);
                }
            }
            if WARNED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                tracing::warn!(
                    "metal: a command buffer failed, was lost or was left at a stop; paid {} fence signals and {} value writes on the CPU",
                    self.signals,
                    self.fills.len()
                );
            }
        });
    }
}

// SAFETY: Metal allows the retained shared events and buffers to be used from either thread.
unsafe impl Send for Owed {}
// SAFETY: the fills and events are immutable lists. The Once lets only one recovery write the
// buffers and signal the events, even if the timeout and completion threads both call recover.
unsafe impl Sync for Owed {}

/// A command buffer committed and not yet reported on by Metal: its queue, when, what it owes.
struct Flight {
    serial: u64,
    queue: usize,
    since: std::time::Instant,
    owed: std::sync::Arc<Owed>,
}

/// Tests only. The list below is the process's, and a recovery with no age pays every command
/// buffer in it, other tests' too: their fences are signalled before their work has run. So the
/// test that recovers takes this for writing, and the tests that depend on when their own command
/// buffers complete take it for reading.
#[cfg(test)]
pub(crate) static FLIGHTS_IN_TESTS: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Committed command buffers Metal has not reported on, oldest first, and per queue the newest
/// serial Metal failed and when.
static FLIGHTS: std::sync::Mutex<std::collections::VecDeque<Flight>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
static FAILURES: std::sync::Mutex<Vec<(usize, u64, std::time::Instant)>> = std::sync::Mutex::new(Vec::new());

fn flights() -> std::sync::MutexGuard<'static, std::collections::VecDeque<Flight>> {
    FLIGHTS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Metal {
    /// Keeps `resources` and `views` that a heap let go until every command buffer made before
    /// now has finished ([`Self::let_go_of_done`]). Without that, a texture view or a span buffer
    /// reached only through a heap entry (and the residency set, which retains nothing) was freed
    /// while queued work could still read it: freeing such a texture in flight faults the GPU.
    pub(crate) fn keep_until_done(&self, resources: Vec<Res>, views: Vec<TextureObject>) {
        if resources.is_empty() && views.is_empty() {
            return;
        }
        let mark = work_mark();
        let mut kept = self.kept_until_done.borrow_mut();
        match kept.back_mut() {
            Some((m, r, v)) if *m == mark => {
                r.extend(resources);
                v.extend(views);
            }
            _ => kept.push_back((mark, resources, views)),
        }
    }

    /// Lets go of what [`Self::keep_until_done`] kept once the work made before it has finished.
    pub(crate) fn let_go_of_done(&self) {
        let mut kept = self.kept_until_done.borrow_mut();
        // Marks only grow: the oldest goes first.
        while kept.front().is_some_and(|(mark, ..)| work_done(*mark)) {
            kept.pop_front();
        }
    }
}

/// The serial the next command buffer takes: every command buffer made before this call has a
/// smaller one (`ComputeBackend::work_mark`).
pub(crate) fn work_mark() -> u64 {
    SERIAL.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether Metal has reported on every committed command buffer with a serial below `mark`
/// (completed, failed, or paid as lost). A command buffer is made and committed on the thread
/// that asks, so none below `mark` is still being encoded.
pub(crate) fn work_done(mark: u64) -> bool {
    flights().iter().all(|f| f.serial >= mark)
}

/// Metal reported on command buffer `serial` of `queue` (completed, or `failed`).
fn landed(serial: u64, queue: usize, failed: bool) {
    flights().retain(|f| f.serial != serial);
    if let Some(pipes) = CB_PIPES.lock().unwrap_or_else(|e| e.into_inner()).remove(&serial)
        && failed
    {
        FAILED_PIPES.lock().unwrap_or_else(|e| e.into_inner()).push((serial, pipes));
    }
    if let Some(fills) = CB_FILLS.lock().unwrap_or_else(|e| e.into_inner()).remove(&serial)
        && failed
    {
        FAILED_FILLS.lock().unwrap_or_else(|e| e.into_inner()).push((serial, fills));
    }
    if failed {
        let mut failures = FAILURES.lock().unwrap_or_else(|e| e.into_inner());
        match failures.iter_mut().find(|(q, ..)| *q == queue) {
            Some(f) if f.1 < serial => *f = (queue, serial, std::time::Instant::now()),
            Some(_) => {}
            None => failures.push((queue, serial, std::time::Instant::now())),
        }
    }
}

/// How long after Metal failed a command buffer the ones committed before it on its queue, and
/// still not reported on, count as lost; and how long any command buffer may go unreported.
const LOST_BEHIND_A_FAILURE: std::time::Duration = std::time::Duration::from_secs(5);
const LOST_ANYWAY: std::time::Duration = std::time::Duration::from_secs(30);

/// Pays what lost command buffers owe (fence signals, the order event, value writes) from the CPU
/// and returns how many were lost. The host calls it while work stays held.
///
/// Observed with memory running out (two games at once on a 48 GB Mac, 41 GB wired): Metal fails
/// some command buffers of a queue with `kIOGPUCommandBufferCallbackErrorTimeout` and never
/// reports on others committed before them, neither completed nor failed. Their value writes
/// never land, so every title queue stays held on its barrier word and the title spins for good.
///
/// Guess: a command buffer older than one Metal failed on the same queue and unreported 5 s
/// later, or unreported for 30 s, will never run. If one does run after all, its draws land late
/// and its signals and writes repeat what was paid. The console has no lost command buffers, so
/// this is our recovery, not console behaviour: the title goes on and the lost frames' draws are
/// missing.
pub fn recover_lost() -> usize {
    recover_lost_after(LOST_BEHIND_A_FAILURE, LOST_ANYWAY)
}

fn recover_lost_after(behind_a_failure: std::time::Duration, anyway: std::time::Duration) -> usize {
    let failures = FAILURES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let lost: Vec<Flight> = {
        let mut flying = flights();
        let is_lost = |f: &Flight| {
            f.since.elapsed() >= anyway
                || failures.iter().any(|(q, s, at)| *q == f.queue && f.serial < *s && at.elapsed() >= behind_a_failure)
        };
        let (lost, kept): (Vec<Flight>, Vec<Flight>) = flying.drain(..).partition(is_lost);
        *flying = kept.into();
        lost
    };
    for f in &lost {
        f.owed.recover();
    }
    if !lost.is_empty() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::error!(
                "metal: {} command buffers were never reported on by the GPU (the first unreported for {:.1}s); \
                 their fence signals and value writes are paid on the CPU and their draws are lost. \
                 This is what running out of memory looks like: close other apps, or other games. (First time; later ones are counted.)",
                lost.len(),
                lost[0].since.elapsed().as_secs_f64()
            );
        });
    }
    lost.len()
}

/// The process's shared events (queue order events, title fences, piece events), weakly, so a
/// stop can wake every command buffer waiting on one ([`stop_gpu`]).
struct TrackedEvents(Vec<objc2::rc::Weak<ProtocolObject<dyn MTLSharedEvent>>>);
// SAFETY: shared events may be used from any thread; the list is only touched under its lock.
unsafe impl Send for TrackedEvents {}
static EVENTS: std::sync::Mutex<TrackedEvents> = std::sync::Mutex::new(TrackedEvents(Vec::new()));

/// Remembers `event` for [`stop_gpu`] and returns it.
pub(crate) fn track_event(
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
) -> Retained<ProtocolObject<dyn MTLSharedEvent>> {
    let mut events = EVENTS.lock().unwrap_or_else(|e| e.into_inner());
    if events.0.len() % 1024 == 1023 {
        events.0.retain(|w| w.load().is_some());
    }
    events.0.push(objc2::rc::Weak::from_retained(&event));
    event
}

/// A residency mutation holds a read guard through its commit. Terminal cleanup holds the
/// write guard while emptying sets, then forbids changes against their obsolete membership.
/// The title lock still serializes ordinary changes to a given heap or pool set.
pub(crate) struct ResidencyGate(std::sync::RwLock<bool>);
pub(crate) static RESIDENCY: ResidencyGate = ResidencyGate(std::sync::RwLock::new(true));

impl ResidencyGate {
    pub(crate) fn changes(&self) -> Option<std::sync::RwLockReadGuard<'_, bool>> {
        let guard = self.0.read().unwrap_or_else(|e| e.into_inner());
        (*guard).then_some(guard)
    }

    fn stop(&self) -> std::sync::RwLockWriteGuard<'_, bool> {
        let mut guard = self.0.write().unwrap_or_else(|e| e.into_inner());
        *guard = false;
        guard
    }
}

#[cfg(test)]
mod residency_gate_tests {
    use super::ResidencyGate;
    use std::sync::{RwLock, mpsc};
    use std::time::Duration;

    #[test]
    fn shutdown_waits_for_the_pending_commit_and_rejects_later_changes() {
        let gate = ResidencyGate(RwLock::new(true));
        let pending = gate.changes().expect("a running set accepts changes");
        std::thread::scope(|scope| {
            let (started_tx, started_rx) = mpsc::channel();
            let (emptied_tx, emptied_rx) = mpsc::channel();
            let gate = &gate;
            scope.spawn(move || {
                started_tx.send(()).unwrap();
                let _stopped = gate.stop();
                emptied_tx.send(()).unwrap();
            });
            started_rx.recv().unwrap();
            assert_eq!(emptied_rx.recv_timeout(Duration::from_millis(20)), Err(mpsc::RecvTimeoutError::Timeout));
            drop(pending);
            emptied_rx.recv_timeout(Duration::from_secs(5)).expect("cleanup follows the commit");
        });
        assert!(gate.changes().is_none(), "emptied sets cannot use their old membership again");
    }
}

/// The process's residency sets (each heap's and the pool's), weakly, so a stop can give back what
/// they hold before the process exits ([`stop_gpu`]).
struct TrackedSets(Vec<objc2::rc::Weak<ProtocolObject<dyn MTLResidencySet>>>);
// SAFETY: residency sets may be used from any thread; the list is only touched under its lock.
unsafe impl Send for TrackedSets {}
static SETS: std::sync::Mutex<TrackedSets> = std::sync::Mutex::new(TrackedSets(Vec::new()));

/// Remembers `set` for [`stop_gpu`] and returns it.
pub(crate) fn track_residency_set(
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
) -> Retained<ProtocolObject<dyn MTLResidencySet>> {
    let mut sets = SETS.lock().unwrap_or_else(|e| e.into_inner());
    if sets.0.len() % 256 == 255 {
        sets.0.retain(|w| w.load().is_some());
    }
    sets.0.push(objc2::rc::Weak::from_retained(&set));
    set
}

/// Empties and commits `sets`: what they held is no longer kept resident. Returns the bytes they
/// held.
fn release_sets(sets: &[Retained<ProtocolObject<dyn MTLResidencySet>>]) -> u64 {
    let mut bytes = 0;
    for set in sets {
        bytes += set.allocatedSize();
        set.removeAllAllocations();
        set.commit();
    }
    bytes
}

/// A value above any a command buffer waits for: an event raised to it lets every wait on it pass.
const STOP_VALUE: u64 = 1 << 62;

/// Stops the process's GPU work cleanly before it exits: raises every shared event a command
/// buffer may wait on ([`track_event`]), waits up to `wait` for the command buffers in flight to
/// complete, then, for those still in flight (running on the GPU: no event wakes them), up to
/// `longest` in all, and pays what the ones that did not complete still owe on the CPU. Returns
/// how many were in flight and how many of them did not complete.
///
/// Guess (#357): a process that exits while its command buffers still run on the GPU is what
/// leaves their memory allocated; one long pass ran 25 s and then completed (res5), so a bounded
/// longer wait lets most finish first. Paying their debts on the CPU does not stop the GPU.
///
/// A process killed while its command buffers sleep in the kernel on events it never signals can
/// leave the GPU memory they hold allocated after it exits: on macOS 26.5 an M5 Max kept about
/// 20 GB "in use" by the GPU driver, owned by no process, until a reboot (#357). Woken first, they
/// finish and the process exits with nothing pending.
pub fn stop_gpu(wait: std::time::Duration, longest: std::time::Duration) -> (usize, usize) {
    let events: Vec<_> = EVENTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .0
        .iter()
        .filter_map(objc2::rc::Weak::load)
        .collect();
    let drained = wake_and_drain(&events, wait, longest);
    // Then what the residency sets keep resident is given back before the process goes, not left
    // to the driver's teardown: a run that hung the GPU at world entry with ~16 GB in its heaps'
    // sets left 19.7 GB "in use" after it exited (#357). Process exit skips `Heap`'s `Drop`.
    let _stopped = RESIDENCY.stop();
    let sets: Vec<_> = SETS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .0
        .iter()
        .filter_map(objc2::rc::Weak::load)
        .collect();
    let bytes = release_sets(&sets);
    tracing::warn!(sets = sets.len(), mb = bytes >> 20, "stop: residency sets emptied");
    drained
}

/// [`stop_gpu`] for `events` (the tests wake only their own).
fn wake_and_drain(
    events: &[Retained<ProtocolObject<dyn MTLSharedEvent>>],
    wait: std::time::Duration,
    longest: std::time::Duration,
) -> (usize, usize) {
    let in_flight = flights().len();
    for e in events {
        if e.signaledValue() < STOP_VALUE {
            e.setSignaledValue(STOP_VALUE);
        }
    }
    let start = std::time::Instant::now();
    let drain_until = |deadline: std::time::Instant| {
        while !flights().is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    };
    drain_until(start + wait);
    if !flights().is_empty() && longest > wait {
        let still: Vec<String> = flights()
            .iter()
            .map(|f| format!("q{:x} #{} for {:.1}s", f.queue, f.serial, f.since.elapsed().as_secs_f64()))
            .collect();
        tracing::warn!(
            "stopping: {} command buffers still in flight {:.1}s after their events were raised (running on the GPU); waiting up to {:.0}s in all: {still:?}",
            still.len(),
            wait.as_secs_f64(),
            longest.as_secs_f64()
        );
        drain_until(start + longest);
    }
    let left: Vec<Flight> = flights().drain(..).collect();
    for f in &left {
        f.owed.recover();
    }
    (in_flight, left.len())
}

/// `VICTORY_TRACE_COMMITS=1` (with `VICTORY_TRACE_GPU_FAILS` for the encoder names): a line for
/// each command buffer as it is committed (serial, queue, its encoders: pipelines by shader hash,
/// draw counts, dispatch sizes) and one as it completes, so a command buffer the GPU never
/// finishes is the one committed and never done (#79).
fn commits_traced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TRACE_COMMITS").is_ok_and(|v| v.trim() == "1"))
}

/// The completion line of [`commits_traced`].
fn trace_done(serial: u64, done: &ProtocolObject<dyn MTLCommandBuffer>) {
    if commits_traced() {
        tracing::info!(
            "cb done #{serial} t={:.3} gpu_ms={:.1} failed={}",
            crate::metal::since_start(),
            (done.GPUEndTime() - done.GPUStartTime()) * 1000.0,
            done.error().is_some()
        );
    }
}

static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
std::thread_local! {
    /// Tests: the piece of this thread's split submissions that the GPU drops, as one it fails.
    static DROP_PIECE: Cell<Option<u64>> = const { Cell::new(None) };
}
/// Tracing: command buffers committed and not yet completed, by serial: when, and what they carry.
static IN_FLIGHT: std::sync::Mutex<std::collections::BTreeMap<u64, (std::time::Instant, String)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
/// Tracing: the gate buffers of a command buffer's indirect dispatches and draws (pipeline name or
/// the draw's place in its pass, contents address, words), by serial. The command buffer keeps
/// the buffers alive while it is in flight.
static GATED: std::sync::Mutex<std::collections::BTreeMap<u64, Vec<(String, usize, usize)>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The group counts the GPU gave command buffer `serial`'s indirect dispatches, as its gates
/// wrote them (zeros for a gate that has not run); `forget` when the command buffer has landed.
fn gated_args(serial: u64, forget: bool) -> String {
    let mut gated = GATED.lock().unwrap();
    let args = if forget { gated.remove(&serial) } else { gated.get(&serial).cloned() };
    let mut out = String::new();
    for (name, at, words) in args.unwrap_or_default() {
        // SAFETY: a shared buffer of twice `words` words (what the gate ran, then what the record
        // asked for), alive while its command buffer is in flight; a landed one is read in its
        // completion handler, before Metal releases it.
        let w: Vec<u32> = (0..2 * words).map(|k| unsafe { (at as *const u32).add(k).read_unaligned() }).collect();
        let (ran, asked) = w.split_at(words);
        out.push_str(&format!(" | gpu read {name} {asked:?}"));
        // A record past what the gate runs, which ran nothing (not one its count buffer left
        // out, or a rectangle list's, rewritten): each pipeline's first few.
        let total = |n: &[u32]| n.iter().map(|g| u64::from(*g)).fold(1u128, |a, g| a * u128::from(g));
        let past = match words {
            3 => total(asked) > 1 << 22,
            _ => asked[1] > 1 << 20 || total(&asked[..2]) > 1 << 26,
        };
        if past && ran.iter().all(|x| *x == 0) {
            static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap();
            if seen.iter().filter(|n| **n == name).count() < 4 {
                seen.push(name.clone());
                tracing::warn!("metal: an indirect record past the most the gate runs ran nothing: {name} {asked:?}");
            }
        }
    }
    out
}
static REPORTED: std::sync::Mutex<std::collections::BTreeSet<u64>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());
/// Slow and in-flight lines printed so far (each capped at 100).
static TRACE_LINES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static FLIGHT_LINES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Logs each command buffer in flight for over 3 s, once (the locks are not held while logging),
/// and returns how many are in flight. Only `VICTORY_TRACE_GPU_FAILS=1` records them. Called at
/// each submission, and by the host when work stays held: nothing is submitted then, so a
/// command buffer the GPU never finishes would otherwise go unreported.
pub fn report_in_flight() -> usize {
    let (count, old): (usize, Vec<(u64, f64, String)>) = {
        let flying = IN_FLIGHT.lock().unwrap();
        let old = flying
            .iter()
            .filter(|(_, (t, _))| t.elapsed().as_secs_f64() > 3.0)
            .map(|(s, (t, d))| (*s, t.elapsed().as_secs_f64(), d.clone()))
            .collect();
        (flying.len(), old)
    };
    for (s, secs, d) in old {
        if REPORTED.lock().unwrap().insert(s)
            && FLIGHT_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 100
        {
            tracing::warn!("metal: command buffer in flight for {secs:.1}s: {d}{}", gated_args(s, false));
        }
    }
    count
}

/// One `tracing::warn!` line for a failed command buffer: its label and each encoder's state.
fn trace_failure(cb: &ProtocolObject<dyn MTLCommandBuffer>, e: &objc2_foundation::NSError) {
    let label = cb.label().map(|l| l.to_string()).unwrap_or_default();
    let mut encoders = String::new();
    // SAFETY: the key is a Metal constant; the value is an NSArray of encoder infos.
    let infos = e.userInfo().objectForKey(unsafe { MTLCommandBufferEncoderInfoErrorKey });
    if let Some(infos) = infos {
        // SAFETY: Metal documents the value as an array of `MTLCommandBufferEncoderInfo`.
        let infos: Retained<objc2_foundation::NSArray<ProtocolObject<dyn MTLCommandBufferEncoderInfo>>> =
            unsafe { Retained::cast_unchecked(infos) };
        for i in infos.iter() {
            let st = match i.errorState() {
                MTLCommandEncoderErrorState::Completed => "Completed",
                MTLCommandEncoderErrorState::Pending => "Pending",
                MTLCommandEncoderErrorState::Faulted => "Faulted",
                MTLCommandEncoderErrorState::Affected => "Affected",
                _ => "Unknown",
            };
            encoders.push_str(&format!(" | [{st}] {}", i.label()));
        }
    }
    tracing::warn!(
        "metal: command buffer failed: [{label}] now={:.3} ({}){encoders}",
        crate::metal::since_start(),
        e.localizedDescription()
    );
}

/// Caps an encoder label.
fn cap_label_str(mut s: String) -> String {
    if s.len() > 600 {
        let mut n = 600;
        while !s.is_char_boundary(n) {
            n -= 1;
        }
        s.truncate(n);
    }
    s
}

impl Metal {
    /// The capped label for an encoder, kept for the command buffer's description.
    /// The command buffer's description keeps the whole label when `VICTORY_TRACE_COMMITS` logs it
    /// (a capped one hid the end of a 17-dispatch pass, #79).
    fn cap_label(&self, s: String) -> Retained<objc2_foundation::NSString> {
        let capped = cap_label_str(s.clone());
        let n = objc2_foundation::NSString::from_str(&capped);
        self.trace_enc.borrow_mut().push(if commits_traced() { s } else { capped });
        n
    }

    /// The commit line of [`commits_traced`]: this command buffer's encoders, which are then
    /// forgotten so the next piece lists only its own.
    fn trace_commit(&self, serial: u64, queue_id: usize, what: &str) {
        if self.gate_stats.is_some() {
            let pipes = std::mem::take(&mut *self.cb_pipes.borrow_mut());
            CB_PIPES.lock().unwrap_or_else(|e| e.into_inner()).insert(serial, pipes);
            let fills = std::mem::take(&mut *self.cb_fills.borrow_mut());
            if !fills.is_empty() {
                CB_FILLS.lock().unwrap_or_else(|e| e.into_inner()).insert(serial, fills);
            }
        }
        if commits_traced() {
            let enc = std::mem::take(&mut *self.trace_enc.borrow_mut()).join(" | ");
            tracing::warn!("cb commit #{serial} q{queue_id:x} t={:.3} {what} | {enc}", crate::metal::since_start());
        }
    }

    fn encode_blit(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        ops: &[BlitOp<'_, Metal>],
        order: Ordering<'_>,
    ) -> Result<(), String> {
        let nonzero = |op: &BlitOp<'_, Metal>| match op {
            BlitOp::Copy { len, .. } | BlitOp::Fill { len, .. } => *len > 0,
            BlitOp::CopyTexture { size, .. } => !size.contains(&0),
            BlitOp::CopyBufferToTexture { len, size, .. }
            | BlitOp::CopyTextureToBuffer { len, size, .. } => *len > 0 && !size.contains(&0),
            BlitOp::CopyDepthPlane { size, row_pitch, .. } => !size.contains(&0) && *row_pitch > 0,
        };
        if !ops.iter().any(nonzero) {
            return Ok(());
        }
        let e = cb.blitCommandEncoder().ok_or("no blit encoder")?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence(f);
        }
        let off = |b: &Buffer, p: *const u8| p as usize - b.base as usize;
        let tex_dims = |t: &ProtocolObject<dyn MTLTexture>, mip: usize| {
            [(t.width() >> mip).max(1), (t.height() >> mip).max(1), (t.depth() >> mip).max(1)]
        };
        let tex_slices = |t: &ProtocolObject<dyn MTLTexture>| {
            if t.textureType() == MTLTextureType::Type2DArray { t.arrayLength() } else { 1 }
        };
        let fits_region = |t: &ProtocolObject<dyn MTLTexture>, mip: usize, slice: usize, at: [usize; 3], sz: [usize; 3]| {
            if sz.contains(&0) || mip >= t.mipmapLevelCount() || slice >= tex_slices(t) {
                return false;
            }
            let dims = tex_dims(t, mip);
            (0..3).all(|i| at[i].saturating_add(sz[i]) <= dims[i])
        };
        let fits_buffer_transfer = |what: &str, buf: (&Buffer, *const u8), t: &Texture, mip: usize, slice: usize, at: [usize; 3], sz: [usize; 3], row_pitch: usize, slice_pitch: usize, len: usize| -> Option<(usize, usize)> {
            // A depth and stencil texture's rows are its 4-byte depth plane (`DepthFromDepthStencil`).
            let plane = t.format == Format::D32FloatS8;
            let align = if plane { 4 } else { t.format.block().1 };
            let buf_off = off(buf.0, buf.1);
            let rows_fit = if plane {
                (sz[2] - 1) * slice_pitch + (sz[1] - 1) * row_pitch + sz[0] * 4 <= len && row_pitch >= sz[0] * 4
            } else {
                transfer_fits(what, t, mip, sz, row_pitch, slice_pitch, len)
            };
            if (is_depth(t.tex.pixelFormat()) && !matches!(t.format, Format::D32FloatS8 | Format::D32Float | Format::D16Unorm))
                || !fits_region(&t.tex, mip, slice, at, sz)
                || buf_off % align != 0
                || row_pitch % align != 0
                || buf_off.saturating_add(len) > buf.0.buf.length()
                || !rows_fit
            {
                return None;
            }
            let slice_bytes = if sz[2] > 1 || t.tex.textureType() == MTLTextureType::Type3D {
                slice_pitch
            } else {
                0
            };
            (slice_bytes % align == 0).then_some((buf_off, slice_bytes))
        };
        for op in ops.iter().filter(|op| nonzero(op)) {
            match op {
                BlitOp::Copy { src, dst, len } => {
                    // SAFETY: both ranges lie inside live buffers (the caller resolved them).
                    unsafe {
                        e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &src.0.buf,
                            off(src.0, src.1),
                            &dst.0.buf,
                            off(dst.0, dst.1),
                            *len,
                        )
                    };
                }
                BlitOp::Fill { dst, len, value } => {
                    let o = off(dst.0, dst.1);
                    if self.gate_stats.is_some() {
                        let note = self.fill_notes.borrow_mut().remove(&(dst.1 as usize)).unwrap_or_default();
                        self.cb_fills.borrow_mut().push(format!(
                            "[{note}; buffer {:#x} len {:#x} pool {} off {o:#x} n {len} value {value:#x}]",
                            dst.0.base as usize,
                            dst.0.buf.length(),
                            dst.0.pool,
                        ));
                    }
                    e.fillBuffer_range_value(&dst.0.buf, NSRange::new(o, *len), *value);
                }
                BlitOp::CopyTexture {
                    src,
                    src_mip,
                    src_slice,
                    src_origin,
                    size: sz,
                    dst,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                } => {
                    if !direct_copy(src.format, dst.format) {
                        // Block-compressed against plain texels of the block's size, or a depth
                        // plane against a plain 4-byte format: no view joins them, so rows go
                        // through a buffer of our own (the command buffer keeps it alive). `sz` is
                        // in source texels.
                        let ((sw, sb), (dw, db)) = (src.format.block(), dst.format.block());
                        let bytes = |f: Format, b: usize| if f == Format::D32FloatS8 { 4 } else { b };
                        if bytes(src.format, sb) != bytes(dst.format, db) || !fits_region(&src.tex, *src_mip, *src_slice, *src_origin, *sz) {
                            continue;
                        }
                        let (row, rows) = (sz[0].div_ceil(sw) * bytes(src.format, sb), sz[1].div_ceil(sw));
                        let dims = tex_dims(&dst.tex, *dst_mip);
                        let d: [usize; 3] = std::array::from_fn(|i| match i {
                            2 => sz[2],
                            _ if dw > sw => (sz[i] * dw).min(dims[i].saturating_sub(dst_origin[i])),
                            _ if sw > dw => sz[i].div_ceil(sw),
                            _ => sz[i],
                        });
                        if !fits_region(&dst.tex, *dst_mip, *dst_slice, *dst_origin, d) {
                            continue;
                        }
                        let Some(scratch) = self.dev.newBufferWithLength_options((row * rows * sz[2]).max(16), MTLResourceOptions::StorageModePrivate) else {
                            continue;
                        };
                        let pitch = if sz[2] > 1 { row * rows } else { 0 };
                        // SAFETY: the regions lie inside both subresources (checked above) and the
                        // buffer holds their rows.
                        unsafe {
                            e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                                &src.tex, *src_slice, *src_mip, origin(*src_origin), size(*sz), &scratch, 0, row, pitch, depth_plane(src),
                            );
                            e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                                &scratch, 0, row, pitch, size(d), &dst.tex, *dst_slice, *dst_mip, origin(*dst_origin), depth_plane(dst),
                            );
                        }
                        continue;
                    }
                    if !fits_region(&src.tex, *src_mip, *src_slice, *src_origin, *sz)
                        || !fits_region(&dst.tex, *dst_mip, *dst_slice, *dst_origin, *sz)
                    {
                        continue;
                    }
                    let view = if src.tex.pixelFormat() == dst.tex.pixelFormat() {
                        None
                    } else if !is_depth(src.tex.pixelFormat())
                        && !is_depth(dst.tex.pixelFormat())
                        && castable(src.format, dst.format)
                        && src.tex.textureType() == dst.tex.textureType()
                    {
                        src.tex.newTextureViewWithPixelFormat(dst.tex.pixelFormat())
                    } else {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| {
                            tracing::warn!(
                                src = ?src.format,
                                dst = ?dst.format,
                                "metal: skipping incompatible CopyTexture formats"
                            );
                        });
                        continue;
                    };
                    if src.tex.pixelFormat() != dst.tex.pixelFormat() && view.is_none() {
                        continue;
                    }
                    let src_tex: &ProtocolObject<dyn MTLTexture> = view.as_deref().unwrap_or(&*src.tex);
                    // SAFETY: bounds, mip/slice ranges, and equal pixel formats are checked above.
                    unsafe {
                        e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                            src_tex,
                            *src_slice,
                            *src_mip,
                            origin(*src_origin),
                            size(*sz),
                            &dst.tex,
                            *dst_slice,
                            *dst_mip,
                            origin(*dst_origin),
                        )
                    };
                }
                BlitOp::CopyDepthPlane { texture, stencil, to_texture, buffer, size: sz, row_pitch } => {
                    let format = texture.tex.pixelFormat();
                    let combined = format == MTLPixelFormat::Depth32Float_Stencil8;
                    let texel = match (format, *stencil) {
                        (MTLPixelFormat::Depth32Float_Stencil8, true) => 1,
                        (MTLPixelFormat::Depth32Float_Stencil8 | MTLPixelFormat::Depth32Float, false) => 4,
                        (MTLPixelFormat::Depth16Unorm, false) => 2,
                        _ => continue,
                    };
                    let at = off(buffer.0, buffer.1);
                    if sz[0] > texture.tex.width()
                        || sz[1] > texture.tex.height()
                        || *row_pitch < sz[0] * texel
                        || *row_pitch % texel != 0
                        || at % 4 != 0
                        || at.saturating_add(row_pitch * sz[1]) > buffer.0.buf.length()
                    {
                        continue;
                    }
                    let options = match (combined, *stencil) {
                        (false, _) => MTLBlitOption::None,
                        (true, false) => MTLBlitOption::DepthFromDepthStencil,
                        (true, true) => MTLBlitOption::StencilFromDepthStencil,
                    };
                    let whole = size([sz[0], sz[1], 1]);
                    // SAFETY: the region lies in level 0 of the texture and the rows in the buffer
                    // (checked above); the options name the plane of a combined format.
                    unsafe {
                        if *to_texture {
                            e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                                &buffer.0.buf,
                                at,
                                *row_pitch,
                                row_pitch * sz[1],
                                whole,
                                &texture.tex,
                                0,
                                0,
                                origin([0, 0, 0]),
                                options,
                            )
                        } else {
                            e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                                &texture.tex,
                                0,
                                0,
                                origin([0, 0, 0]),
                                whole,
                                &buffer.0.buf,
                                at,
                                *row_pitch,
                                row_pitch * sz[1],
                                options,
                            )
                        }
                    };
                }
                BlitOp::CopyBufferToTexture {
                    src,
                    len,
                    row_pitch,
                    slice_pitch,
                    size: sz,
                    dst,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                } => {
                    let Some((src_off, slice_bytes)) = fits_buffer_transfer(
                        "copy_buffer_to_texture",
                        *src,
                        dst,
                        *dst_mip,
                        *dst_slice,
                        *dst_origin,
                        *sz,
                        *row_pitch,
                        *slice_pitch,
                        *len,
                    ) else {
                        continue;
                    };
                    // SAFETY: buffer bounds, pitches, alignments, and texture region are validated above.
                    unsafe {
                        e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                            &src.0.buf,
                            src_off,
                            *row_pitch,
                            slice_bytes,
                            size(*sz),
                            &dst.tex,
                            *dst_slice,
                            *dst_mip,
                            origin(*dst_origin),
                            depth_plane(dst),
                        )
                    };
                }
                BlitOp::CopyTextureToBuffer {
                    src,
                    src_mip,
                    src_slice,
                    src_origin,
                    size: sz,
                    dst,
                    len,
                    row_pitch,
                    slice_pitch,
                } => {
                    let Some((dst_off, slice_bytes)) = fits_buffer_transfer(
                        "copy_texture_to_buffer",
                        *dst,
                        src,
                        *src_mip,
                        *src_slice,
                        *src_origin,
                        *sz,
                        *row_pitch,
                        *slice_pitch,
                        *len,
                    ) else {
                        continue;
                    };
                    // SAFETY: buffer bounds, pitches, alignments, and texture region are validated above.
                    unsafe {
                        e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                            &src.tex,
                            *src_slice,
                            *src_mip,
                            origin(*src_origin),
                            size(*sz),
                            &dst.0.buf,
                            dst_off,
                            *row_pitch,
                            slice_bytes,
                            depth_plane(src),
                        )
                    };
                }
            }
        }
        if let Some((f, started)) = order {
            e.updateFence(f);
            started.set(true);
        }
        if self.trace_fails {
            let fills = ops.iter().filter(|o| matches!(o, BlitOp::Fill { .. })).count();
            e.setLabel(Some(&self.cap_label(format!("blit: {} ops ({fills} fills)", ops.len()))));
        }
        e.endEncoding();
        Ok(())
    }
}

/// Words of the `VICTORY_TRACE_GATE` buffer: groups, pipeline and the three record words of the
/// largest dispatch record a gate let run, then the same of the largest it stopped.
pub(crate) const GATE_STATS_WORDS: usize = 10;

/// `VICTORY_TRACE_GATE=1`: the pipelines (low pointer bits) and group counts of the dispatches and
/// indirect dispatches (groups 0) of each command buffer in flight, by serial; a failed one's move
/// to [`FAILED_PIPES`] and are logged by name at the next submission.
static CB_PIPES: std::sync::Mutex<std::collections::BTreeMap<u64, Vec<CbDispatch>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
static FAILED_PIPES: std::sync::Mutex<Vec<(u64, Vec<CbDispatch>)>> = std::sync::Mutex::new(Vec::new());
/// `VICTORY_TRACE_GATE=1`: the fills of each command buffer in flight (what the host said they
/// write, and the buffer, offset and length encoded), by serial; a failed one's move to
/// [`FAILED_FILLS`] and are logged at the next submission.
static CB_FILLS: std::sync::Mutex<std::collections::BTreeMap<u64, Vec<String>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
static FAILED_FILLS: std::sync::Mutex<Vec<(u64, Vec<String>)>> = std::sync::Mutex::new(Vec::new());
/// A dispatch of [`CB_PIPES`]: the pipeline's low pointer bits and the group counts.
pub(crate) type CbDispatch = (u32, [u32; 3]);

impl Metal {
    /// `VICTORY_TRACE_GATE=1`: once a second, the largest dispatch record the gates let run and the
    /// largest they stopped since the last line (a dispatch or a mesh dispatch, by group count),
    /// with its pipeline; then both are cleared.
    fn log_gate_stats(&self) {
        let Some((stats, last)) = &self.gate_stats else { return };
        let failed = std::mem::take(&mut *FAILED_PIPES.lock().unwrap_or_else(|e| e.into_inner()));
        if !failed.is_empty() {
            let names = self.pipe_names.borrow();
            for (serial, pipes) in failed {
                let list: Vec<String> = pipes
                    .iter()
                    .map(|(id, g)| {
                        let n = names.iter().find(|(k, _)| **k as u32 == *id).map_or_else(|| format!("{id:#x}"), |(_, n)| n.clone());
                        format!("{n} {g:?}")
                    })
                    .collect();
                tracing::warn!("gate: failed command buffer #{serial} ran {} dispatches: {}", list.len(), list.join("; "));
            }
        }
        for (serial, fills) in std::mem::take(&mut *FAILED_FILLS.lock().unwrap_or_else(|e| e.into_inner())) {
            tracing::warn!("gate: failed command buffer #{serial} held {} fills: {}", fills.len(), fills.join(" "));
        }
        let now = crate::metal::since_start();
        if now - last.get() < 1.0 {
            return;
        }
        last.set(now);
        let p = stats.contents().as_ptr().cast::<u32>();
        // SAFETY: a shared buffer of `GATE_STATS_WORDS` words; the GPU's racing writes are atomic.
        let w: Vec<u32> = (0..GATE_STATS_WORDS).map(|k| unsafe { p.add(k).read_volatile() }).collect();
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(p, 0, GATE_STATS_WORDS) };
        let names = self.pipe_names.borrow();
        let name = |id: u32| {
            names
                .iter()
                .find(|(k, _)| **k as u32 == id)
                .map_or_else(|| format!("{id:#x}"), |(_, n)| n.clone())
        };
        if w[0] != 0 || w[5] != 0 {
            tracing::info!(
                "gate: largest run {} groups {} {:?}; largest stopped {} groups {} {:?}",
                w[0],
                name(w[1]),
                &w[2..5],
                w[5],
                if w[5] != 0 { name(w[6]) } else { String::new() },
                &w[7..10]
            );
        }
    }
}

/// The blit option a copy between `t` and buffer rows takes: a depth and stencil texture's rows
/// are its depth plane.
fn depth_plane(t: &Texture) -> MTLBlitOption {
    if t.format == Format::D32FloatS8 { MTLBlitOption::DepthFromDepthStencil } else { MTLBlitOption::None }
}

/// Whether a texture copies into another with one texture-to-texture blit: the same format, or
/// two plain formats of one texel size (through a view).
pub fn direct_copy(a: Format, b: Format) -> bool {
    let plain = |f: Format| f.block().0 == 1 && !f.is_depth();
    a == b || (plain(a) && plain(b) && a.block() == b.block())
}

/// The same mapping as GATE: starts align to a rectangle boundary. The shader divides the
/// converted SV_VertexID by six; the converter subtracts the doubled start from Metal's ID,
/// preserving ordinary draws' relative IDs. Keep instance count/base in their original units.
fn rectangle_arguments([vertices, instances, first, instance]: [u32; 4]) -> Option<[u32; 4]> {
    let vertices = (vertices / 3).checked_mul(6)?;
    let start = first.checked_mul(2)?;
    start.checked_add(vertices)?;
    (first.is_multiple_of(3)).then_some([vertices, instances, start, instance])
}

/// `ExecuteIndirect` with a count buffer runs record `index` only while `index < count`, both read
/// on the GPU: this one-thread kernel copies the record's `words`, or zeros (no groups, no
/// vertices or instances), into a buffer of its own, which the indirect command then reads.
/// A rectangle list's record (`DrawCall::Rects`) becomes the four-vertex strip instances a direct
/// one draws: (4, vertices / 3, 0, first vertex / 3).
/// A dispatch's record, or a mesh dispatch's, runs nothing if it asks for more than 2^22 groups in
/// all: such a record is what its writer left unfinished in title memory, and Metal would run it
/// (510 x 1,182,027,346 x 1,314,076,063 groups in GTA VI's world: that command buffer never
/// ended, and every one behind it on the queue timed out). A guard of ours, not a console rule:
/// the title's own records ask for far less (9,192 groups at most seen here), but for more than
/// D3D12's 65,535 in one dimension on larger targets, so the limit is on the product.
/// A draw's record draws nothing if it asks for more than 2^20 instances or 2^26 vertices in all,
/// for the same reason: one read as 3 vertices and 864,026,624 instances (stale float bits where
/// the count should be) kept the GPU at 100 % until the process was gone, and the driver then
/// kept the process's 19 GB. Guess: no draw of the title's comes near either limit.
/// Bound at 20 and up, apart from the converter's slots. `at`: the record's index, its words,
/// whether a count applies, 1 for a rectangle list's record or 2 for a dispatch's or a mesh
/// dispatch's (three words). After the words it runs, `out` holds the words the record asked for
/// (`VICTORY_TRACE_GPU_FAILS`).
const GATE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void gate(device const uint *count [[buffer(20)]],
                 device const uint *record [[buffer(21)]],
                 device uint *out [[buffer(22)]],
                 constant uint4 &at [[buffer(23)]],
                 constant uint4 &trace [[buffer(24)]],
                 device atomic_uint *stats [[buffer(25)]]) {
    bool run = at.z == 0u || at.x < count[0];
    if (at.w == 2u) {
        // Two steps: the product of three words does not fit 64 bits.
        ulong plane = ulong(record[0]) * record[1];
        run = run && plane <= 4194304ul && plane * record[2] <= 4194304ul;
        // `VICTORY_TRACE_GATE`: the largest record run (words 0-4: groups, pipeline, record) and
        // stopped (5-9), by groups capped at 2^32 - 1; racing writers may mix two records' words.
        if (trace.x != 0u && (at.z == 0u || at.x < count[0])) {
            ulong all = min(plane, 0xfffffffful) * record[2];
            uint groups = uint(min(all, 0xfffffffful));
            uint base = run ? 0u : 5u;
            if (atomic_fetch_max_explicit(&stats[base], groups, memory_order_relaxed) < groups) {
                atomic_store_explicit(&stats[base + 1u], trace.y, memory_order_relaxed);
                for (uint w = 0; w < 3u; w++) {
                    atomic_store_explicit(&stats[base + 2u + w], record[w], memory_order_relaxed);
                }
            }
        }
    } else if (at.w == 1u) {
        // A rectangle a vertex triple.
        run = run && record[0] <= (3u << 20);
    } else {
        run = run && record[1] <= (1u << 20) && ulong(record[0]) * record[1] <= (1ul << 26);
    }
    for (uint w = 0; w < at.y; w++) {
        out[w] = run ? record[w] : 0u;
        out[at.y + w] = record[w];
    }
    if (at.w == 1u) {
        // shortcut: an instanced rectangle list draws nothing (the rectangle translation takes the
        // instance for the rectangle); draw its instances when a title is seen to use them.
        out[0] = 4u;
        out[1] = run && record[1] == 1u ? record[0] / 3u : 0u;
        out[2] = 0u;
        out[3] = run ? record[2] / 3u : 0u;
    } else if (at.w == 3u) {
        ulong vertices = ulong(out[0] / 3) * 6;
        ulong first = ulong(out[2]) * 2;
        bool fits = vertices + first <= 0xfffffffful && out[2] % 3 == 0;
        out[0] = fits ? uint(vertices) : 0u;
        out[2] = fits ? uint(first) : 0u;
    }
}
"#;

impl Metal {
    fn gate_pipeline(&self) -> Result<&ProtocolObject<dyn MTLComputePipelineState>, String> {
        self.gate
            .get_or_init(|| {
                let lib = self
                    .dev
                    .newLibraryWithSource_options_error(
                        &objc2_foundation::NSString::from_str(GATE),
                        None,
                    )
                    .map_err(|e| e.localizedDescription().to_string())?;
                let f = lib
                    .newFunctionWithName(&objc2_foundation::NSString::from_str("gate"))
                    .ok_or("no gate function")?;
                self.dev
                    .newComputePipelineStateWithFunction_error(&f)
                    .map_err(|e| e.localizedDescription().to_string())
            })
            .as_deref()
            .map_err(Clone::clone)
    }

    /// Encodes the gate of one counted (`count`: buffer, address, index), rectangle-list (`kind` 1)
    /// or dispatch or mesh dispatch (`kind` 2) record (`words` long at `record`) into `e`; returns the
    /// buffer the indirect command reads instead of the record. Tracked by Metal (unlike the
    /// title's pools), so whatever reads it later in the command buffer waits for the write.
    /// shortcut: one gate dispatch and one small buffer per record; gate all the records of one
    /// `ExecuteIndirect` in a single dispatch if a large `MaxCommandCount` shows up in profiles.
    fn encode_gate(
        &self,
        e: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        count: Option<(&Buffer, *const u8, u32)>,
        (record, title): (&Buffer, *const u8),
        words: u32,
        kind: u32,
        set_on_queue: bool,
        pipe: u32,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let out = self
            .dev
            .newBufferWithLength_options(
                // The words it runs, then the words the record asked for.
                words as usize * 8,
                // Tracing reads the record the GPU used back (`GATED`).
                if self.trace_fails { MTLResourceOptions::StorageModeShared } else { MTLResourceOptions::StorageModePrivate },
            )
            .ok_or("no buffer for a gated indirect record")?;
        let counted = u32::from(count.is_some());
        // Without a count, slot 20 is bound to the record (not read).
        let (count, at, index) = count.unwrap_or((record, title, 0));
        for b in [count, record] {
            if self.names(b, set_on_queue) {
                e.useResource_usage(ProtocolObject::from_ref(&*b.buf), MTLResourceUsage::Read);
            }
        }
        if let Some((stats, _)) = &self.gate_stats {
            e.useResource_usage(ProtocolObject::from_ref(&**stats), MTLResourceUsage::Read | MTLResourceUsage::Write);
        }
        // What the work before wrote (the count, the record) is seen.
        e.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        e.setComputePipelineState(self.gate_pipeline()?);
        let args = [index, words, counted, kind];
        // SAFETY: both title ranges lie inside live buffers (the caller resolved them); the bytes
        // are copied before the call returns.
        unsafe {
            e.setBuffer_offset_atIndex(Some(&count.buf), at as usize - count.base as usize, 20);
            e.setBuffer_offset_atIndex(
                Some(&record.buf),
                title as usize - record.base as usize,
                21,
            );
            e.setBuffer_offset_atIndex(Some(&out), 0, 22);
            e.setBytes_length_atIndex(
                NonNull::new(args.as_ptr().cast_mut().cast()).unwrap(),
                16,
                23,
            );
            let trace = [u32::from(self.gate_stats.is_some()), pipe, 0, 0];
            e.setBytes_length_atIndex(NonNull::new(trace.as_ptr().cast_mut().cast()).unwrap(), 16, 24);
            // Without the trace the kernel never touches it: the gate's own buffer stands in.
            e.setBuffer_offset_atIndex(Some(self.gate_stats.as_ref().map_or(&*out, |(b, _)| &**b)), 0, 25);
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(size([1, 1, 1]), size([1, 1, 1]));
        frame_time::count(Metric::Dispatches, 1);
        e.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        Ok(out)
    }

    /// Whether encoder `use` calls are needed for `b`: not when the queue has the pool residency set
    /// and the buffer is in it.
    fn names(&self, b: &Buffer, set_on_queue: bool) -> bool {
        let needed = !(set_on_queue && b.is_resident());
        if needed {
            self.count(|c| c.use_resource += 1);
        }
        needed
    }

    pub(crate) fn encode_compute(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatches: &[crate::backend::Dispatch<'_, Metal>],
        set_on_queue: bool,
        order: Ordering<'_>,
        attached_heaps: &mut Vec<*const Heap>,
    ) -> Result<(), String> {
        let runs = |d: &crate::backend::Dispatch<'_, Metal>| {
            d.indirect.is_some() || !d.groups.contains(&0)
        };
        if !dispatches.iter().any(runs) {
            return Ok(());
        }
        let e = cb.computeCommandEncoder().ok_or("no compute encoder")?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence(f);
        }
        let mut named = Vec::new();
        for d in dispatches.iter().filter(|d| runs(d)) {
            // Every indirect record goes through the gate: the count buffer's limit, and the most
            // groups a record may ask for.
            let gated = match d.indirect {
                // An open encoder left behind makes Metal abort at the next pass's encoder.
                Some(record) => Some(
                    self.encode_gate(&e, d.count, record, 3, 2, set_on_queue, Retained::as_ptr(&d.pipeline.0) as *const u8 as usize as u32)
                        .inspect_err(|_| e.endEncoding())?,
                ),
                None => None,
            };
            if let (true, Some(g)) = (self.trace_fails, &gated) {
                let k = Retained::as_ptr(&d.pipeline.0) as *const u8 as usize;
                let name = self.pipe_names.borrow().get(&k).cloned().unwrap_or_else(|| format!("{k:#x}"));
                self.trace_args.borrow_mut().push((name, g.contents().as_ptr() as usize, 3));
            }
            if self.gate_stats.is_some() {
                let id = Retained::as_ptr(&d.pipeline.0) as *const u8 as usize as u32;
                self.cb_pipes.borrow_mut().push((id, if d.indirect.is_some() { [0; 3] } else { d.groups.map(|g| g as u32) }));
            }
            e.setComputePipelineState(&d.pipeline.0);
            for h in [d.heaps.resource, d.heaps.sampler].into_iter().flatten() {
                self.use_heap_cb(h, attached_heaps, |set| cb.useResidencySet(set));
            }
            self.bind_root_compute(&e, &d.root, &d.heaps, d.pipeline.1.as_ref(), &mut named);
            for b in d.resources.iter().filter(|b| self.names(b, set_on_queue)) {
                e.useResource_usage(
                    ProtocolObject::from_ref(&*b.buf),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
            match &gated {
                // SAFETY: the gate wrote three words to its own buffer.
                Some(g) => unsafe {
                    e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                        g,
                        0,
                        size(d.threads_per_group),
                    )
                },
                None => e.dispatchThreadgroups_threadsPerThreadgroup(size(d.groups), size(d.threads_per_group)),
            }
            frame_time::count(Metric::Dispatches, 1);
        }
        if let Some((f, started)) = order {
            e.updateFence(f);
            started.set(true);
        }
        if self.trace_fails {
            let names = self.pipe_names.borrow();
            let mut l = String::from("compute: ");
            for d in dispatches.iter().filter(|d| runs(d)) {
                let k = Retained::as_ptr(&d.pipeline.0) as *const u8 as usize;
                match names.get(&k) {
                    Some(n) => l.push_str(n),
                    None => l.push_str(&format!("{k:#x}")),
                }
                match d.indirect {
                    // The record as title memory holds it now: what earlier command buffers
                    // left there, not what this one's earlier passes will write.
                    Some((_, title)) => {
                        // SAFETY: the caller resolved three words at `title` inside the live buffer.
                        let w: [u32; 3] = unsafe { title.cast::<[u32; 3]>().read_unaligned() };
                        l.push_str(&format!(" indirect{w:?}; "));
                    }
                    None => l.push_str(&format!(" {}x{}x{}; ", d.groups[0], d.groups[1], d.groups[2])),
                }
            }
            e.setLabel(Some(&self.cap_label(l)));
        }
        e.endEncoding();
        Ok(())
    }

    fn encode_color_clear_rects(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        target: &Texture,
        color: [f32; 4],
        rects: &[ClearRect],
        set_on_queue: bool,
        order: Ordering<'_>,
        attached_heaps: &mut Vec<*const Heap>,
    ) -> Result<(), String> {
        let clip = |p, limit| usize::try_from(p).unwrap_or_default().min(limit);
        let (width, height) = (target.tex.width(), target.tex.height());
        let regions: Vec<_> = rects
            .iter()
            .filter_map(|r| {
                let (left, right) = (clip(r.left, width), clip(r.right, width));
                let (top, bottom) = (clip(r.top, height), clip(r.bottom, height));
                (left < right && top < bottom).then(|| ([left, top, 0], [right - left, bottom - top, 1]))
            })
            .collect();
        if regions.is_empty() {
            return Ok(());
        }
        if !matches!(target.tex.textureType(), MTLTextureType::Type2D | MTLTextureType::Type2DArray)
            || target.tex.sampleCount() != 1
        {
            return Err("rectangular color clear requires a single-sampled 2D color target".into());
        }
        if target.format.is_depth()
            || !target.tex.usage().contains(MTLTextureUsage::RenderTarget)
        {
            return Err("rectangular color clear requires a renderable color format".into());
        }

        // Use Metal's clear conversion in the view format, including sRGB and integer targets.
        // A retained command buffer keeps this private scratch alive until the blits finish.
        let (max_width, max_height) = regions.iter().fold((0, 0), |(w, h), (_, size)| {
            (w.max(size[0]), h.max(size[1]))
        });
        let td = MTLTextureDescriptor::new();
        td.setPixelFormat(target.tex.pixelFormat());
        td.setStorageMode(MTLStorageMode::Private);
        td.setUsage(MTLTextureUsage::RenderTarget);
        // SAFETY: these nonzero dimensions fit the selected view.
        unsafe {
            td.setWidth(max_width);
            td.setHeight(max_height);
        }
        let scratch = Texture {
            tex: self.dev.newTextureWithDescriptor(&td).ok_or("no rectangular clear texture")?,
            format: target.format,
            present_view: Default::default(),
            views: Default::default(),
        };
        self.encode_render(
            cb,
            &RenderPass {
                color: vec![ColorTarget { texture: &scratch, clear: Some(color) }],
                depth: None,
                draws: vec![],
            },
            set_on_queue,
            order,
            attached_heaps,
        )?;
        let copies: Vec<_> = regions
            .into_iter()
            .map(|(dst_origin, size)| BlitOp::CopyTexture {
                src: &scratch,
                src_mip: 0,
                src_slice: 0,
                src_origin: [0, 0, 0],
                size,
                dst: target,
                dst_mip: 0,
                dst_slice: 0,
                dst_origin,
            })
            .collect();
        self.encode_blit(cb, &copies, order)
    }

    fn encode_render(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        rp: &RenderPass<'_, Metal>,
        set_on_queue: bool,
        order: Ordering<'_>,
        attached_heaps: &mut Vec<*const Heap>,
    ) -> Result<(), String> {
        // Attachments must agree in sample count. Their sizes may differ (a 1920x1080 colour
        // target over a 3840x2160 depth target in GTA VI's story load): the console draws into
        // the area all of them cover, and so does the pass, constrained to that intersection.
        let mut samples = None;
        let mut rt_size = (usize::MAX, usize::MAX);
        for t in rp
            .color
            .iter()
            .map(|c| &c.texture.tex)
            .chain(rp.depth.as_ref().map(|d| &d.texture.tex))
        {
            let n = t.sampleCount();
            if let Some(want) = samples
                && want != n
            {
                return Err(format!("render targets of different sample counts: {want} and {n}"));
            }
            samples = Some(n);
            rt_size = (rt_size.0.min(t.width()), rt_size.1.min(t.height()));
        }
        // The title drew with no render targets: its pixel shaders write through unordered access
        // views alone (the lighting's light masks are rasterized so). Metal runs such a pass
        // given its extent and sample count in place of attachments; the extent is what the
        // draws' viewports cover (D3D12 has no other bound either).
        let targetless = rp.color.is_empty() && rp.depth.is_none();
        if targetless {
            const LARGEST: f32 = 16384.0;
            let extent = rp
                .draws
                .iter()
                .filter(|d| d.pipeline.targetless)
                .filter_map(|d| d.state.viewport)
                .filter(|v| v.width > 0.0 && v.height > 0.0 && v.x >= 0.0 && v.y >= 0.0)
                .map(|v| ((v.x + v.width).min(LARGEST).ceil() as usize, (v.y + v.height).min(LARGEST).ceil() as usize))
                .reduce(|a, b| (a.0.max(b.0), a.1.max(b.1)));
            let Some(extent) = extent else {
                // No draw of a pipeline made for such a pass, or none with a viewport: nothing
                // says where to draw. Skipped rather than refused: the submission's rest runs.
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!("metal: a pass with no render targets and no viewport is skipped ({} draws)", rp.draws.len());
                });
                return Ok(());
            };
            rt_size = extent;
            // Said at each power of two: whether a title's passes of this kind run, and how big.
            static RAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = RAN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                eprintln!(
                    "metal: passes without render targets so far: {n} (this one {}x{}, {} samples, {} draws)",
                    extent.0, extent.1, targetless_samples(&self.dev), rp.draws.len()
                );
            }
        }
        let desc = MTLRenderPassDescriptor::new();
        desc.setRenderTargetWidth(rt_size.0);
        desc.setRenderTargetHeight(rt_size.1);
        if targetless {
            desc.setDefaultRasterSampleCount(targetless_samples(&self.dev));
        }
        for (i, c) in rp.color.iter().enumerate() {
            // SAFETY: colour attachment index below 8.
            let a = unsafe { desc.colorAttachments().objectAtIndexedSubscript(i) };
            a.setTexture(Some(&c.texture.tex));
            a.setStoreAction(MTLStoreAction::Store);
            match c.clear {
                Some(v) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearColor(MTLClearColor {
                        red: v[0] as f64,
                        green: v[1] as f64,
                        blue: v[2] as f64,
                        alpha: v[3] as f64,
                    });
                }
                None => a.setLoadAction(MTLLoadAction::Load),
            }
        }
        // Indirect draws are gated first, in a compute pass of their own: the count buffer's
        // limit, a rectangle list's record, and the most a record may ask for.
        let gated = |d: &Draw<'_, Metal>| d.indirect.is_some();
        let mut gates = Vec::new();
        if rp.draws.iter().any(gated) {
            let c = cb.computeCommandEncoder().ok_or("no compute encoder")?;
            if let Some((f, started)) = order
                && started.get()
            {
                c.waitForFence(f);
            }
            for d in &rp.draws {
                let words = match d.call {
                    DrawCall::Indexed { .. } => 5,
                    DrawCall::Arrays { .. } | DrawCall::Rects => 4,
                    DrawCall::Mesh { .. } => 3,
                };
                let kind = match d.call {
                    DrawCall::Rects => 1,
                    DrawCall::Mesh { .. } => 2,
                    _ if d.topology == Topology::Rectangles => 3,
                    _ => 0,
                };
                gates.push(match d.indirect {
                    Some(record) if gated(d) => {
                        let g = self.encode_gate(&c, d.count, record, words, kind, set_on_queue, Retained::as_ptr(&d.pipeline.pso) as *const u8 as usize as u32).inspect_err(|_| c.endEncoding())?;
                        if self.trace_fails {
                            let name = format!("{} {}", if kind == 2 { "mesh draw" } else { "draw" }, gates.len());
                            self.trace_args.borrow_mut().push((name, g.contents().as_ptr() as usize, words as usize));
                        }
                        Some(g)
                    }
                    _ => None,
                });
            }
            if let Some((f, started)) = order {
                c.updateFence(f);
                started.set(true);
            }
            c.endEncoding();
        }
        if let Some(dp) = &rp.depth {
            let a = desc.depthAttachment();
            a.setTexture(Some(&dp.texture.tex));
            a.setStoreAction(MTLStoreAction::Store);
            match dp.clear {
                Some(v) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearDepth(v as f64);
                }
                None => a.setLoadAction(MTLLoadAction::Load),
            }
            // A combined depth and stencil texture is attached as both.
            if dp.texture.tex.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8 {
                let a = desc.stencilAttachment();
                a.setTexture(Some(&dp.texture.tex));
                a.setStoreAction(MTLStoreAction::Store);
                match dp.clear_stencil {
                    Some(v) => {
                        a.setLoadAction(MTLLoadAction::Clear);
                        a.setClearStencil(u32::from(v));
                    }
                    None => a.setLoadAction(MTLLoadAction::Load),
                }
            }
        }
        let e = cb.renderCommandEncoderWithDescriptor(&desc).ok_or_else(|| {
            // Metal refuses an encoder whose attachments disagree (size, sample count, format
            // kind); the caller's log then says which targets the pass named.
            let size = |t: &ProtocolObject<dyn MTLTexture>| {
                format!("{}x{} fmt {}", t.width(), t.height(), t.pixelFormat().0)
            };
            let color: Vec<String> = rp.color.iter().map(|c| size(&c.texture.tex)).collect();
            let depth = rp.depth.as_ref().map(|d| size(&d.texture.tex));
            format!(
                "no render encoder: {} colour {color:?}, depth {depth:?}, {} draws",
                rp.color.len(),
                rp.draws.len()
            )
        })?;
        if let Some((f, started)) = order
            && started.get()
        {
            e.waitForFence_beforeStages(f, MTLRenderStages::Vertex | MTLRenderStages::Object | MTLRenderStages::Mesh);
        }
        let mut named = Vec::new();
        let mut depth_set = false;
        let mut default_depth = None;
        for (k, d) in rp.draws.iter().enumerate() {
            // The stage that reads the draw's buffers before the pixel shader.
            let first = match (d.pipeline.mesh.is_some(), d.call) {
                (true, DrawCall::Mesh { .. }) => MTLRenderStages::Mesh,
                (false, DrawCall::Arrays { .. } | DrawCall::Indexed { .. } | DrawCall::Rects) => MTLRenderStages::Vertex,
                _ => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| eprintln!("metal: a mesh draw of a vertex pipeline, or the reverse, is dropped"));
                    continue;
                }
            };
            if d.indirect.is_none() && empty_draw(&d.call) {
                continue;
            }
            // A pipeline runs in the kind of pass it was made for: Metal faults a pipeline with
            // target formats in a pass without attachments, and the reverse.
            if d.pipeline.targetless != targetless {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| eprintln!("metal: a draw whose pipeline and pass disagree about having targets is dropped"));
                continue;
            }
            for h in [d.heaps.resource, d.heaps.sampler].into_iter().flatten() {
                self.use_heap_cb(h, attached_heaps, |set| cb.useResidencySet(set));
            }
            if d.topology == Topology::Rectangles && !matches!(d.call, DrawCall::Arrays { .. }) {
                e.endEncoding();
                return Err("rectangle output expansion requires a non-indexed draw".into());
            }
            e.setRenderPipelineState(&d.pipeline.pso);
            e.setCullMode(d.pipeline.cull);
            let r = &d.pipeline.raster;
            e.setFrontFacingWinding(if r.front_ccw {
                MTLWinding::CounterClockwise
            } else {
                MTLWinding::Clockwise
            });
            e.setTriangleFillMode(if r.wireframe {
                MTLTriangleFillMode::Lines
            } else {
                MTLTriangleFillMode::Fill
            });
            e.setDepthClipMode(if r.depth_clip {
                MTLDepthClipMode::Clip
            } else {
                MTLDepthClipMode::Clamp
            });
            e.setDepthBias_slopeScale_clamp(r.depth_bias.0, r.depth_bias.1, r.depth_bias.2);
            // Metal validation forbids passing nil to `setDepthStencilState:`: bind a default
            // state object when returning from a depth/stencil pipeline to one without.
            if let Some(depth) = d.pipeline.depth.as_deref() {
                e.setDepthStencilState(Some(depth));
                depth_set = true;
            } else if depth_set {
                let depth = default_depth.get_or_insert_with(|| {
                    self.dev
                        .newDepthStencilStateWithDescriptor(&MTLDepthStencilDescriptor::new())
                        .expect("default depth state")
                });
                e.setDepthStencilState(Some(depth));
                depth_set = false;
            }
            if d.pipeline.stencil {
                e.setStencilReferenceValue(d.state.stencil_ref);
            }
            let [r, g, b, a] = d.state.blend_factor;
            e.setBlendColorRed_green_blue_alpha(r, g, b, a);
            // The target's size bounds the scissor (Metal refuses a rectangle outside it).
            let (tw, th) = rt_size;
            let viewport = match d.state.viewport {
                Some(v) if v.width <= 0.0 || v.height <= 0.0 => continue,
                Some(v) => MTLViewport {
                    originX: v.x as f64,
                    originY: v.y as f64,
                    width: v.width as f64,
                    height: v.height as f64,
                    znear: v.min_depth as f64,
                    zfar: v.max_depth as f64,
                },
                None => MTLViewport {
                    originX: 0.0,
                    originY: 0.0,
                    width: tw as f64,
                    height: th as f64,
                    znear: 0.0,
                    zfar: 1.0,
                },
            };
            e.setViewport(viewport);
            let scissor = match d.state.scissor {
                Some([l, t, r, b]) => {
                    let (l, t) = (l.clamp(0, tw as i32) as usize, t.clamp(0, th as i32) as usize);
                    let (r, b) = (r.clamp(0, tw as i32) as usize, b.clamp(0, th as i32) as usize);
                    MTLScissorRect {
                        x: l,
                        y: t,
                        width: r.saturating_sub(l),
                        height: b.saturating_sub(t),
                    }
                }
                None => MTLScissorRect {
                    x: 0,
                    y: 0,
                    width: tw,
                    height: th,
                },
            };
            if scissor.width == 0 || scissor.height == 0 {
                // Nothing of the target is inside the rectangle: nothing is drawn.
                continue;
            }
            e.setScissorRect(scissor);
            self.bind_root_render(&e, &d.root, d.fragment_root.as_ref(), d.object_root.as_ref(), &d.heaps, d.pipeline, &mut named);
            let producer = producer_stages(d.pipeline.mesh.as_ref());
            for b in d.resources.iter().filter(|b| self.names(b, set_on_queue)) {
                e.useResource_usage_stages(
                    ProtocolObject::from_ref(&*b.buf),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                    producer | MTLRenderStages::Fragment,
                );
            }
            if let (DrawCall::Mesh { groups }, Some(m)) = (d.call, &d.pipeline.mesh) {
                // Native DX mesh/AS grids are supplied by these Metal commands. Runtime slots
                // 4/5 hold vertex draw / geometry-tessellation emulation parameters, not this
                // 12-byte dispatch grid. Root arguments for mesh/object/fragment stay at slot 2.
                let indirect = d.indirect.map(|(ab, title)| {
                    if self.names(ab, set_on_queue) {
                        e.useResource_usage_stages(ProtocolObject::from_ref(&*ab.buf), MTLResourceUsage::Read, producer);
                    }
                    match gates.get(k).and_then(Option::as_ref) {
                        Some(g) => (&**g, 0),
                        None => (&*ab.buf, title as usize - ab.base as usize),
                    }
                });
                // A split mesh program draws once per pass, the further passes with their own
                // state and everything else as it is set: the same groups, arguments and targets.
                for pass in std::iter::once(None).chain(m.more.iter().map(Some)) {
                    if let Some(state) = pass {
                        e.setRenderPipelineState(state);
                    }
                    match indirect {
                        // SAFETY: the caller resolved the complete 12-byte record in this live buffer.
                        Some((args, at)) => unsafe {
                            e.drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                                args, at, m.object_threads, m.threads);
                        },
                        None => e.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(size3(groups), m.object_threads, m.threads),
                    }
                    frame_time::count(Metric::Draws, 1);
                }
                continue;
            }
            for &(slot, b, title) in &d.vertex_buffers {
                let off = title as usize - b.base as usize;
                // SAFETY: the buffer is live and `off` is inside it.
                unsafe {
                    e.setVertexBuffer_offset_atIndex(Some(&b.buf), off, VERTEX_BUFFERS + slot)
                };
                if self.names(b, set_on_queue) {
                    e.useResource_usage_stages(
                        ProtocolObject::from_ref(&*b.buf),
                        MTLResourceUsage::Read,
                        MTLRenderStages::Vertex,
                    );
                }
            }
            let prim = match d.topology {
                Topology::Triangles | Topology::Rectangles => MTLPrimitiveType::Triangle,
                Topology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
                Topology::Lines => MTLPrimitiveType::Line,
                Topology::LineStrip => MTLPrimitiveType::LineStrip,
                Topology::Points => MTLPrimitiveType::Point,
            };
            // What `IRRuntimeDrawPrimitives` / `IRRuntimeDrawIndexedPrimitives` do: draw arguments
            // (`IRRuntimeDrawParams`, 5 words) at slot 4 and the index type at slot 5 (0 = not indexed,
            // else Metal's index type + 1), then the draw itself.
            if let Some((ab, title)) = d.indirect {
                // What the converter runtime's indirect `IRRuntimeDraw(Indexed)Primitives` do: the
                // argument record itself is the shader's draw arguments (D3D12's records are
                // `IRRuntimeDraw(Indexed)Argument`s and Metal's indirect ones, word for word).
                if self.names(ab, set_on_queue) {
                    e.useResource_usage_stages(
                        ProtocolObject::from_ref(&*ab.buf),
                        MTLResourceUsage::Read,
                        first,
                    );
                }
                // A counted record is read from its gate's buffer.
                let (args, at) = match gates.get(k).and_then(Option::as_ref) {
                    Some(g) => (&**g, 0),
                    None => (&*ab.buf, title as usize - ab.base as usize),
                };
                // SAFETY: the record is inside the live buffer (the caller resolved it).
                unsafe { e.setVertexBuffer_offset_atIndex(Some(args), at, DRAW_ARGUMENTS) };
                match (d.call, d.index_buffer) {
                    (DrawCall::Mesh { .. }, _) => unreachable!("mesh handled above"),
                    (DrawCall::Arrays { .. } | DrawCall::Rects, _) => {
                        set_index_type(&e, 0);
                        // SAFETY: as above.
                        unsafe { e.drawPrimitives_indirectBuffer_indirectBufferOffset(prim, args, at) };
                    }
                    (DrawCall::Indexed { wide, .. }, Some((b, title))) => {
                        set_index_type(&e, if wide { 2 } else { 1 });
                        if self.names(b, set_on_queue) {
                            e.useResource_usage_stages(
                                ProtocolObject::from_ref(&*b.buf),
                                MTLResourceUsage::Read,
                                MTLRenderStages::Vertex,
                            );
                        }
                        // SAFETY: both buffers are live; the record's start index counts from `title`.
                        unsafe {
                            e.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                                prim,
                                if wide { MTLIndexType::UInt32 } else { MTLIndexType::UInt16 },
                                &b.buf,
                                title as usize - b.base as usize,
                                args,
                                at,
                            )
                        };
                    }
                    (DrawCall::Indexed { .. }, None) => {
                        e.endEncoding();
                        return Err("indexed draw without an index buffer".into());
                    }
                }
                frame_time::count(Metric::Draws, 1);
                continue;
            }
            let mut dp = [0u32; 5];
            let ib: u16;
            match d.call {
                DrawCall::Mesh { .. } => unreachable!("mesh handled above"),
                DrawCall::Arrays {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    let args = [vertex_count, instance_count, first_vertex, first_instance];
                    let Some(args) = (if d.topology == Topology::Rectangles { rectangle_arguments(args) } else { Some(args) }) else {
                        continue;
                    };
                    let [vertex_count, instance_count, first_vertex, first_instance] = args;
                    if vertex_count == 0 { continue; }
                    dp[..4].copy_from_slice(&args);
                    ib = 0;
                    set_draw_bytes(&e, &dp, ib);
                    // SAFETY: arguments are plain counts.
                    unsafe {
                        e.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                            prim,
                            first_vertex as usize,
                            vertex_count as usize,
                            instance_count as usize,
                            first_instance as usize,
                        )
                    };
                }
                DrawCall::Rects => unreachable!("an indirect-only call, and so empty here"),
                DrawCall::Indexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                    wide,
                } => {
                    // A draw that names no index buffer is dropped, not the pass: returning here
                    // would leave the render encoder open and, now that a refused pass still
                    // commits (for its fence signals), Metal would assert on the open encoder.
                    let Some((b, title)) = d.index_buffer else {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| {
                            eprintln!("metal: an indexed draw without an index buffer is dropped")
                        });
                        continue;
                    };
                    let width = if wide { 4 } else { 2 };
                    let off = title as usize - b.base as usize + first_index as usize * width;
                    dp = [
                        index_count,
                        instance_count,
                        off as u32,
                        base_vertex as u32,
                        first_instance,
                    ];
                    ib = if wide { 2 } else { 1 };
                    set_draw_bytes(&e, &dp, ib);
                    if self.names(b, set_on_queue) {
                        e.useResource_usage_stages(
                            ProtocolObject::from_ref(&*b.buf),
                            MTLResourceUsage::Read,
                            MTLRenderStages::Vertex,
                        );
                    }
                    // SAFETY: the index buffer is live and holds `index_count` indices at `off`.
                    unsafe {
                        e.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                            prim,
                            index_count as usize,
                            if wide { MTLIndexType::UInt32 } else { MTLIndexType::UInt16 },
                            &b.buf,
                            off,
                            instance_count as usize,
                            base_vertex as isize,
                            first_instance as usize,
                        )
                    };
                }
            }
            frame_time::count(Metric::Draws, 1);
        }
        if let Some((f, started)) = order {
            e.updateFence_afterStages(f, MTLRenderStages::Fragment);
            started.set(true);
        }
        if self.trace_fails {
            // Each distinct pipeline once, with its draw count: names the pass's shaders.
            let names = self.pipe_names.borrow();
            let mut seen: Vec<(usize, usize)> = Vec::new();
            for d in &rp.draws {
                let k = Retained::as_ptr(&d.pipeline.pso) as *const u8 as usize;
                match seen.iter_mut().find(|(p, _)| *p == k) {
                    Some((_, n)) => *n += 1,
                    None => seen.push((k, 1)),
                }
            }
            let mut l = format!("render: {} draws", rp.draws.len());
            for (k, n) in seen {
                match names.get(&k) {
                    Some(name) => l.push_str(&format!("; {name} x{n}")),
                    None => l.push_str(&format!("; {k:#x} x{n}")),
                }
            }
            drop(names);
            e.setLabel(Some(&self.cap_label(l)));
        }
        e.endEncoding();
        Ok(())
    }
}

fn producer_stages(mesh: Option<&MeshDispatch>) -> MTLRenderStages {
    match mesh {
        None => MTLRenderStages::Vertex,
        Some(m) if m.object => MTLRenderStages::Object | MTLRenderStages::Mesh,
        Some(_) => MTLRenderStages::Mesh,
    }
}

fn size3(t: [u32; 3]) -> MTLSize {
    MTLSize { width: t[0] as usize, height: t[1] as usize, depth: t[2] as usize }
}

/// Whether a direct draw has a zero vertex, index, instance or mesh-group count.
fn empty_draw(call: &DrawCall) -> bool {
    match *call {
        DrawCall::Arrays { vertex_count, instance_count, .. } => {
            vertex_count == 0 || instance_count == 0
        }
        DrawCall::Indexed { index_count, instance_count, .. } => {
            index_count == 0 || instance_count == 0
        }
        DrawCall::Mesh { groups } => groups.contains(&0),
        // Only an indirect call.
        DrawCall::Rects => true,
    }
}

impl crate::metal::Compiler {
    /// The pipeline state and dispatch sizes of a mesh pipeline: a described mesh stage (with an
    /// optional amplification stage) or a guessed vertex program translated as a mesh shader.
    /// Every mesh pipeline uses upstream's archive and compile deadline.
    fn mesh_pipeline(
        &self,
        d: &RenderPipelineDesc<'_>,
        ms: &MeshPipelineDesc<'_>,
    ) -> Result<(Retained<ProtocolObject<dyn MTLRenderPipelineState>>, MeshDispatch), String> {
        // Debugging (`VICTORY_DEBUG_MESH_DEPTH_ONLY=1`): mesh pipelines with a pixel stage are
        // refused, so only depth-only mesh passes (shadows) draw, as before the mesh+pixel pairs
        // compiled. To tell whether the main view's draws are what empties the shadow maps.
        if d.ps.is_some() && std::env::var_os("VICTORY_DEBUG_MESH_DEPTH_ONLY").is_some() {
            return Err("debug: mesh pipelines with a pixel stage are off".into());
        }
        let has_stencil = d.depth == Some(Format::D32FloatS8);
        use victory_shader::emit::{metal_mesh_bytes, METAL_MESH_OUTPUT_BYTES};
        let bytes = metal_mesh_bytes(ms.mesh).ok_or("mesh output footprint is unknown")?;
        if bytes > METAL_MESH_OUTPUT_BYTES {
            return Err(format!("mesh outputs {bytes} exceed {METAL_MESH_OUTPUT_BYTES} bytes"));
        }
        let (mf, info) = self.mesh_function(ms.mesh, ms.layout, false)?;
        let pd = MTLMeshRenderPipelineDescriptor::new();
        // SAFETY: this is the reflected mesh entry point of a converted mesh library.
        unsafe { pd.setMeshFunction(Some(&mf)) };
        let count = |threads: [u32; 3]| -> Result<usize, String> {
            threads.into_iter().try_fold(1usize, |n, x| n.checked_mul(x as usize))
                .filter(|&n| n > 0 && n <= 1024).ok_or("invalid reflected mesh threadgroup size".into())
        };
        if info.max_vertices == 0 || info.max_vertices > 256 || info.max_primitives > 256 {
            return Err("invalid reflected mesh output counts".into());
        }
        pd.setMaxTotalThreadsPerMeshThreadgroup(count(info.threads)?);
        let mut dispatch = MeshDispatch { threads: size3(info.threads), object_threads: size3([1; 3]), object: false, more: Vec::new() };
        if let Some(dxil) = ms.amplification {
            let (af, ai) = self.mesh_function(dxil, ms.amplification_layout, true)?;
            // SAFETY: reflected amplification entry point (Metal object function).
            unsafe { pd.setObjectFunction(Some(&af)) };
            pd.setMaxTotalThreadsPerObjectThreadgroup(count(ai.threads)?);
            pd.setPayloadMemoryLength(ai.payload_bytes.max(info.payload_bytes) as usize);
            dispatch.object_threads = size3(ai.threads);
            dispatch.object = true;
        } else if info.payload_bytes != 0 {
            return Err("mesh payload without an amplification stage".into());
        }
        if let Some(ps) = d.ps {
            let pf = self.stage_function(ps, d.ps_layout, d.needs_dual_source())?.1;
            // SAFETY: converted fragment function.
            unsafe { pd.setFragmentFunction(Some(&pf)) };
        } else {
            // Metal requires a fragment function on every mesh pipeline, even
            // depth-only ones (D3D12 allows none): an empty one writes nothing.
            let lib = self
                .dev
                .newLibraryWithSource_options_error(
                    &objc2_foundation::NSString::from_str("fragment void victory_no_fragment() {}"),
                    None,
                )
                .map_err(|e| e.localizedDescription().to_string())?;
            let pf = lib
                .newFunctionWithName(&objc2_foundation::NSString::from_str("victory_no_fragment"))
                .ok_or("the empty fragment function did not compile")?;
            // SAFETY: a fragment function with no inputs or outputs.
            unsafe { pd.setFragmentFunction(Some(&pf)) };
        }
        color_attachments(&pd.colorAttachments(), d);
        if let Some(f) = d.depth {
            pd.setDepthAttachmentPixelFormat(pixel_format(f));
            if has_stencil { pd.setStencilAttachmentPixelFormat(pixel_format(f)); }
        }
        // The mesh output footprint goes with a failure: it says how near Metal's limit the
        // programs are that its compiler fails on.
        let key = mesh_crash_key(d, ms);
        let built = match &self.archive {
            Some(a) => a.mesh(&self.dev, &pd, &key),
            None => crate::archive::compile_mesh(&self.dev, &pd, &key),
        };
        // `VICTORY_DUMP_MESH_PAIRS=<dir>`: the converted libraries and the descriptor's
        // essentials of every mesh pipeline with a pixel stage and no amplification stage,
        // by outcome, so that a pair Metal's compiler fails on can be built outside the game.
        if let (Some(dir), Some(ps), None) = (std::env::var_os("VICTORY_DUMP_MESH_PAIRS"), d.ps, ms.amplification)
            && let (Ok((mesh_lib, stage)), Ok((_, pixel_lib))) = (
                self.msc.convert_mesh_with(ms.mesh, ms.layout, false),
                self.msc.convert_with(ps, None, d.ps_layout, d.needs_dual_source()),
            )
        {
            let id = &crate::disk::key(&[ms.mesh, ps])[..16];
            let name = format!("{}_{id}", if built.is_ok() { "ok" } else { "fail" });
            let dir = std::path::Path::new(&dir);
            let colors: Vec<String> = (0..d.color.len())
                // SAFETY: colour attachment index below 8.
                .map(|i| unsafe { pd.colorAttachments().objectAtIndexedSubscript(i) }.pixelFormat().0.to_string())
                .collect();
            let meta = format!(
                "entry {}\nthreads {}\ncolors {}\ndepth {}\nstencil {}\nbytes {bytes}\nvertices {}\nprimitives {}\nmesh_dxil {}\npixel_dxil {}\nresult {}\n",
                stage.entry,
                pd.maxTotalThreadsPerMeshThreadgroup(),
                colors.join(","),
                pd.depthAttachmentPixelFormat().0,
                pd.stencilAttachmentPixelFormat().0,
                info.max_vertices,
                info.max_primitives,
                ms.mesh.len(),
                ps.len(),
                built.as_ref().err().map_or("ok", String::as_str),
            );
            let _ = std::fs::write(dir.join(format!("{name}.mesh.metallib")), mesh_lib);
            let _ = std::fs::write(dir.join(format!("{name}.pixel.metallib")), pixel_lib);
            let _ = std::fs::write(dir.join(format!("{name}.mesh.dxil")), ms.mesh);
            let _ = std::fs::write(dir.join(format!("{name}.pixel.dxil")), ps);
            let _ = std::fs::write(dir.join(format!("{name}.txt")), meta);
        }
        let pso = built
            .map_err(|e| format!("{e} [mesh outputs {bytes} of {METAL_MESH_OUTPUT_BYTES} bytes, {} vertices, {} primitives, pixel stage {}]", info.max_vertices, info.max_primitives, d.ps.is_some()))?;
        Ok((pso, dispatch))
    }
}

fn set_draw_bytes(e: &ProtocolObject<dyn MTLRenderCommandEncoder>, dp: &[u32; 5], index_type: u16) {
    // SAFETY: setVertexBytes copies the bytes before returning.
    unsafe {
        e.setVertexBytes_length_atIndex(
            NonNull::new(dp.as_ptr().cast_mut().cast()).unwrap(),
            20,
            DRAW_ARGUMENTS,
        );
    }
    set_index_type(e, index_type);
}

/// The converter's index type word (0 = not indexed, else Metal's index type + 1).
fn set_index_type(e: &ProtocolObject<dyn MTLRenderCommandEncoder>, index_type: u16) {
    // SAFETY: setVertexBytes copies the bytes before returning.
    unsafe {
        e.setVertexBytes_length_atIndex(
            NonNull::new((&raw const index_type).cast_mut().cast()).unwrap(),
            2,
            UNIFORMS,
        );
    }
}

/// Whether a pixel format is a depth or stencil one.
fn is_depth(f: MTLPixelFormat) -> bool {
    [
        MTLPixelFormat::Depth16Unorm,
        MTLPixelFormat::Depth32Float,
        MTLPixelFormat::Stencil8,
        MTLPixelFormat::Depth24Unorm_Stencil8,
        MTLPixelFormat::Depth32Float_Stencil8,
    ]
    .contains(&f)
}

#[cfg(test)]
mod compile_tests {
    use super::*;
    use victory_backend::backend::{RootLayout, RootParam};

    #[test]
    fn pending_mesh_compile_owns_its_stages_and_layouts() {
        let (owned, key, crash) = {
            let mesh = vec![1, 2, 3];
            let object = vec![4, 5, 6];
            let pixel = vec![7, 8, 9];
            let mesh_layout = RootLayout { params: vec![RootParam::Srv { register: 3, space: 0 }], ..Default::default() };
            let object_layout = RootLayout { params: vec![RootParam::Uav { register: 4, space: 0 }], ..Default::default() };
            let d = RenderPipelineDesc {
                mesh: Some(MeshPipelineDesc {
                    mesh: &mesh,
                    layout: Some(&mesh_layout),
                    amplification: Some(&object),
                    amplification_layout: Some(&object_layout),
                    more: &[],
                }),
                ps: Some(&pixel),
                dual_source: true,
                ..Default::default()
            };
            (OwnedDesc::of(&d), render_key(&d), crash_key(&d))
        };
        let d = owned.desc();
        let m = d.mesh.as_ref().unwrap();
        assert_eq!(m.mesh, &[1, 2, 3]);
        assert_eq!(m.amplification, Some(&[4, 5, 6][..]));
        assert_eq!(m.layout.unwrap().params, [RootParam::Srv { register: 3, space: 0 }]);
        assert_eq!(m.amplification_layout.unwrap().params, [RootParam::Uav { register: 4, space: 0 }]);
        assert_eq!(d.ps, Some(&[7, 8, 9][..]));
        assert!(d.dual_source);
        assert_eq!(render_key(&d), key);
        assert_eq!(crash_key(&d), crash);
    }

    #[test]
    fn mesh_compile_identity_includes_object_stages_and_layouts() {
        let layout = RootLayout::default();
        let changed_layout = RootLayout { bounds_checked: true, ..Default::default() };
        let mut d = RenderPipelineDesc {
            mesh: Some(MeshPipelineDesc {
                mesh: &[1, 2],
                layout: Some(&layout),
                amplification: Some(&[3, 4]),
                amplification_layout: Some(&layout),
                more: &[],
            }),
            ..Default::default()
        };
        let (key, crash) = (render_key(&d), crash_key(&d));
        d.mesh.as_mut().unwrap().mesh = &[2, 1];
        assert_ne!(render_key(&d), key);
        assert_ne!(crash_key(&d), crash);
        d.mesh.as_mut().unwrap().mesh = &[1, 2];
        // A further pass is part of the pipeline; each pass is compiled, and a compiler crash
        // on it remembered, by itself.
        let passes = [vec![9u8]];
        d.mesh.as_mut().unwrap().more = &passes;
        assert_ne!(render_key(&d), key);
        assert_eq!(crash_key(&d), crash);
        assert_eq!(OwnedDesc::of(&d).desc().mesh.unwrap().more, &passes[..]);
        d.mesh.as_mut().unwrap().more = &[];
        d.mesh.as_mut().unwrap().amplification = Some(&[4, 3]);
        assert_ne!(render_key(&d), key);
        assert_ne!(crash_key(&d), crash);
        d.mesh.as_mut().unwrap().amplification = Some(&[3, 4]);
        d.mesh.as_mut().unwrap().layout = Some(&changed_layout);
        assert_ne!(render_key(&d), key);
        d.mesh.as_mut().unwrap().layout = Some(&layout);
        d.mesh.as_mut().unwrap().amplification_layout = Some(&changed_layout);
        assert_ne!(render_key(&d), key);
    }
}

#[cfg(test)]
mod transfer_tests {
    use super::*;

    #[test]
    fn transfers_are_checked_against_the_format() {
        // RGBA16Float 4x4: a row is 32 bytes.
        assert!(transfer_need(Format::Rgba16Float, [4, 4, 1], 16, 0, 1 << 20).is_err());
        assert_eq!(transfer_need(Format::Rgba16Float, [4, 4, 1], 32, 0, 128), Ok(128));
        // BC1 8x8: 2 block rows of 16 bytes; the last row needs no padding.
        assert_eq!(transfer_need(Format::Bc1, [8, 8, 1], 16, 0, 32), Ok(32));
        assert!(transfer_need(Format::Bc1, [8, 8, 1], 8, 0, 1 << 20).is_err());
        // Too short a buffer, and a 3D slice pitch below one image.
        assert!(transfer_need(Format::Rgba8Unorm, [4, 4, 1], 16, 0, 63).is_err());
        assert!(transfer_need(Format::Rgba8Unorm, [4, 4, 2], 16, 32, 1 << 20).is_err());
        assert_eq!(transfer_need(Format::Rgba8Unorm, [4, 4, 2], 16, 64, 128), Ok(128));
    }
}

#[cfg(test)]
mod render_target_tests {
    use super::*;
    use objc2_foundation::NSString;

    fn draw_green_triangles(metal: &Metal, target: &Texture, clear: Option<[f32; 4]>, viewports: &[Option<Viewport>]) {
        let _flight = FLIGHTS_IN_TESTS.read().unwrap_or_else(|e| e.into_inner());
        let library = metal.dev.newLibraryWithSource_options_error(&NSString::from_str(r#"
            #include <metal_stdlib>
            using namespace metal;
            vertex float4 test_vertex(uint i [[vertex_id]]) {
                return float4(i == 1 ? 3.0 : -1.0, i == 2 ? 3.0 : -1.0, 0.0, 1.0);
            }
            fragment float4 test_fragment() { return float4(0.0, 1.0, 0.0, 1.0); }
        "#), None).expect("compile generated Metal shaders");
        let pipeline_desc = MTLRenderPipelineDescriptor::new();
        let vertex = library.newFunctionWithName(&NSString::from_str("test_vertex")).unwrap();
        let fragment = library.newFunctionWithName(&NSString::from_str("test_fragment")).unwrap();
        pipeline_desc.setVertexFunction(Some(&vertex));
        pipeline_desc.setFragmentFunction(Some(&fragment));
        // SAFETY: the descriptor has a color attachment at index 0.
        unsafe { pipeline_desc.colorAttachments().objectAtIndexedSubscript(0) }
            .setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        let pipeline = RenderPipeline {
            pso: metal.dev.newRenderPipelineStateWithDescriptor_error(&pipeline_desc).unwrap(),
            depth: None,
            cull: MTLCullMode::None,
            raster: Raster::default(),
            stencil: false,
            targetless: false,
            mesh: None,
            object_static: None,
            vs_static: None,
            ps_static: None,
        };
        let queue = metal.create_queue().unwrap();
        let fence = metal.create_fence(0).unwrap();
        let passes = [Pass::Render(RenderPass {
            color: vec![ColorTarget { texture: target, clear }],
            depth: None,
            draws: viewports.iter().map(|&viewport| Draw {
                pipeline: &pipeline,
                topology: Topology::Triangles,
                state: DrawState { viewport, ..Default::default() },
                root: RootArgs::default(),
                fragment_root: None,
                object_root: None,
                heaps: Heaps::default(),
                resources: vec![],
                vertex_buffers: vec![],
                index_buffer: None,
                call: DrawCall::Arrays {
                    vertex_count: 3,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
                indirect: None,
                count: None,
            }).collect(),
        })];
        metal.submit_passes(&queue, &PassSubmission {
            passes: &passes,
            waits: &[],
            signals: &[(&fence, 1)],
        }).expect("draw green triangles");
        assert!(metal.wait(&fence, 1, 2000), "draw did not complete");
    }

    #[test]
    fn a_draw_into_a_narrowed_render_target_view_preserves_neighbors() {
        let metal = Metal::new().expect("Metal device");
        let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 8, 8);
        desc.dim = TexDim::D2Array;
        desc.depth_or_layers = 2;
        desc.mips = 2;
        desc.render_target = true;
        let texture = metal.create_texture(&desc).unwrap();
        let sentinel = |mip, slice| [17 + (mip * 40 + slice * 80) as u8, 31, 47, 255];
        for slice in 0..2 {
            for mip in 0..2 {
                let n = 8 >> mip;
                metal.upload_texture(&texture, mip, slice, &sentinel(mip, slice).repeat(n * n), n * 4, 0);
            }
        }
        let target = metal.create_texture_view(&texture, &ViewDesc {
            format: Format::Rgba8Unorm,
            mips: (1, 1),
            slices: (1, 1),
        }).expect("narrowed render target");
        assert_eq!((target.tex.width(), target.tex.height()), (4, 4));

        draw_green_triangles(&metal, &target, None, &[None]);
        for slice in 0..2 {
            for mip in 0..2 {
                let n = 8 >> mip;
                let mut data = vec![0; n * n * 4];
                metal.read_texture(&texture, mip, slice, &mut data, n * 4, 0);
                let expected = if (mip, slice) == (1, 1) { [0, 255, 0, 255] } else { sentinel(mip, slice) };
                assert!(data.as_chunks::<4>().0.iter().all(|p| *p == expected),
                        "mip {mip} slice {slice}: expected {expected:?}, got {:?}", &data[..4]);
            }
        }
    }

    #[test]
    fn default_viewport_restores_the_full_target_in_the_same_pass() {
        let metal = Metal::new().expect("Metal device");
        let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 16, 16);
        desc.render_target = true;
        let target = metal.create_texture(&desc).unwrap();
        let half = Viewport {
            x: 0.0,
            y: 0.0,
            width: 8.0,
            height: 16.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        draw_green_triangles(&metal, &target, Some([0.0; 4]), &[Some(half), None]);

        let mut pixels = [0; 16 * 16 * 4];
        metal.read_texture(&target, 0, 0, &mut pixels, 16 * 4, 0);
        assert_eq!(pixels.as_chunks::<4>().0[8 * 16 + 12], [0, 255, 0, 255],
                   "the second draw must reach (12, 8) outside the first viewport");
    }
}

#[cfg(test)]
mod owed_tests {
    use super::*;
    use crate::backend::ComputeBackend;
    use crate::mem::{Region, page_size};

    /// A submission of more passes than a command buffer takes goes in several, in order: every
    /// pass runs and the fence is signalled once the last has.
    #[test]
    fn a_long_submission_is_cut_into_command_buffers_of_at_most_n_passes() {
        let m = Metal::new().unwrap().with_passes_per_buffer(8);
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        // SAFETY: inside the region; the GPU is idle.
        let at = |o: usize| unsafe { region.ptr.add(o) as *const u8 };
        let q = m.create_queue().unwrap();
        let fence = m.create_fence(0).unwrap();
        // Each pass fills its own 8 bytes; the last one also overwrites the first pass's.
        let mut passes: Vec<_> = (0..20u8)
            .map(|n| Pass::Blit(vec![BlitOp::Fill { dst: (&buf, at(usize::from(n) * 8)), len: 8, value: n + 1 }]))
            .collect();
        passes.push(Pass::Blit(vec![BlitOp::Fill { dst: (&buf, at(0)), len: 8, value: 0xEE }]));
        let signals = [(&fence, 1)];
        let before = m.command_buffers();
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &signals }).unwrap();
        assert_eq!(m.command_buffers() - before, 3, "21 passes, 8 to a command buffer");
        assert!(m.wait(&fence, 1, 5000), "the last piece signals");
        // SAFETY: inside the region; the GPU is done with it.
        let bytes = unsafe { std::slice::from_raw_parts(region.ptr, 20 * 8) };
        assert_eq!(&bytes[..8], &[0xEE; 8], "the pieces ran in order");
        for n in 1..20 {
            assert_eq!(&bytes[n * 8..][..8], &[n as u8 + 1; 8], "pass {n}");
        }

        // 0: the whole submission in one command buffer.
        let m = m.with_passes_per_buffer(0);
        let before = m.command_buffers();
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &[(&fence, 2)] }).unwrap();
        assert_eq!(m.command_buffers() - before, 1);
        assert!(m.wait(&fence, 2, 5000));
    }

    /// A piece of a split submission that the GPU fails (here one the test drops: its passes and
    /// its release of the next piece do not run, as when the GPU fails it) pays its own value
    /// writes on the CPU and lets the next piece go; the later pieces still run on the GPU, and the
    /// fence is signalled by the last one. Nothing is paid twice: a write made after the fence was
    /// reached stays.
    #[test]
    fn a_failed_earlier_piece_pays_its_own_writes_and_the_later_pieces_still_run() {
        let _shared = FLIGHTS_IN_TESTS.read().unwrap_or_else(|e| e.into_inner());
        let m = Metal::new().unwrap().with_passes_per_buffer(4);
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        // SAFETY: inside the region; the GPU is idle.
        let at = |o: usize| unsafe { region.ptr.add(o) as *const u8 };
        let q = m.create_queue().unwrap();
        q.1.set(true); // as with the pool's residency set: the order event is in play
        let fence = m.create_fence(0).unwrap();
        // 12 passes, three pieces of 4: pass n fills its own 8 bytes with n + 1.
        let passes: Vec<_> = (0..12u8)
            .map(|n| Pass::Blit(vec![BlitOp::Fill { dst: (&buf, at(usize::from(n) * 8)), len: 8, value: n + 1 }]))
            .collect();
        let check = |first: u8, signal: u64| {
            DROP_PIECE.with(|d| d.set(Some(1)));
            let before = m.command_buffers();
            let signals = [(&fence, signal)];
            m.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &signals }).unwrap();
            DROP_PIECE.with(|d| d.set(None));
            assert_eq!(m.command_buffers() - before, 3);
            assert!(m.wait(&fence, signal, 5000), "the later pieces did not run");
            // SAFETY: inside the region; the GPU is done with it.
            let bytes = unsafe { std::slice::from_raw_parts(region.ptr, 12 * 8) };
            for n in 0..12 {
                assert_eq!(&bytes[n * 8..][..8], &[n as u8 + first; 8], "pass {n}");
            }
        };
        check(1, 1);
        // The title writes over the first piece's words once it has seen the fence: a second
        // payment of the failed piece would undo that.
        // SAFETY: inside the region; the GPU is done with it.
        unsafe { std::ptr::write_bytes(region.ptr, 0xCD, 8) };
        let after = [(&fence, 2)];
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &[], signals: &after }).unwrap();
        assert!(m.wait(&fence, 2, 5000));
        // SAFETY: inside the region.
        assert_eq!(unsafe { std::slice::from_raw_parts(region.ptr, 8) }, [0xCD; 8], "paid twice");
        assert_eq!(q.2.signaledValue(), 2, "the order event went past what the submissions owe");
        // Again from zeros: the failed piece's words come from the CPU, the rest from the GPU.
        // SAFETY: inside the region; the GPU is idle.
        unsafe { std::ptr::write_bytes(region.ptr, 0, 12 * 8) };
        check(1, 3);
    }

    #[test]
    fn a_failed_command_buffer_is_paid_on_the_cpu() {
        let m = Metal::new().unwrap();
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        // SAFETY: inside the region; the GPU is idle.
        let at = |o: usize| unsafe { region.ptr.add(o) };
        let q = m.create_queue().unwrap();
        q.1.set(true); // as with the pool's residency set: the order event is in play
        let fence = m.create_fence(0).unwrap();
        let passes = [Pass::Blit(vec![
            BlitOp::Fill { dst: (&buf, at(64) as *const u8), len: 8, value: 0xAB },
            BlitOp::Copy { src: (&buf, at(0) as *const u8), dst: (&buf, at(256) as *const u8), len: 16 },
        ])];
        let (s5, s7, s9) = ([(&fence, 5)], [(&fence, 7)], [(&fence, 9)]);
        let sub = |signals| PassSubmission { waits: &[], passes: &passes, signals };
        // The recovery a submission signalling the fence to 7 would owe, built as `submit_passes` does.
        q.3.set(3);
        let owed = Owed::new(&q, &sub(&s7), true);
        owed.recover();
        // SAFETY: bounds as above.
        let bytes = unsafe { std::slice::from_raw_parts(at(60), 16) };
        assert_eq!(bytes, [0, 0, 0, 0, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0xAB, 0, 0, 0, 0]);
        // SAFETY: inside the region.
        assert_eq!(unsafe { *at(256) }, 0, "copies are not replayed");
        assert_eq!(m.completed(&fence), 7);
        assert_eq!(q.2.signaledValue(), 3);
        // An older submission failing later does not lower anything.
        Owed::new(&q, &sub(&s5), false).recover();
        assert_eq!(m.completed(&fence), 7);
        // A normal submission after it still completes: the queue is not stuck behind the order value.
        m.submit_passes(&q, &sub(&s9)).unwrap();
        assert!(m.wait(&fence, 9, 5000), "queue stuck after recovery");
    }

    #[test]
    fn repeated_recovery_preserves_the_waiters_write() {
        let m = Metal::new().unwrap();
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        let q = m.create_queue().unwrap();
        let fence = m.create_fence(0).unwrap();
        let passes = [Pass::Blit(vec![BlitOp::Fill { dst: (&buf, region.ptr), len: 8, value: 0xAB }])];
        let signals = [(&fence, 7)];
        let owed = Owed::new(&q, &PassSubmission { waits: &[], passes: &passes, signals: &signals }, false);
        let word = region.ptr.cast::<[u8; 8]>();

        owed.recover();
        assert_eq!(m.completed(&fence), 7);
        // SAFETY: these eight bytes are inside the live region. Only this thread accesses them.
        unsafe {
            assert_eq!(word.read(), [0xAB; 8]);
            word.write([0xCD; 8]);
        }
        owed.recover();
        // SAFETY: same region and exclusive access as above.
        assert_eq!(unsafe { word.read() }, [0xCD; 8]);
    }

    /// A command buffer Metal never reports on (here one waiting for an event nobody signals) is
    /// found once it is old enough, and what it owes is paid: the fill and the fence signal the
    /// title's queues wait for.
    #[test]
    fn a_lost_command_buffer_pays_what_it_owes() {
        let _alone = FLIGHTS_IN_TESTS.write().unwrap_or_else(|e| e.into_inner());
        let m = Metal::new().unwrap();
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        let q = m.create_queue().unwrap();
        let (never, fence) = (m.create_fence(0).unwrap(), m.create_fence(0).unwrap());
        let passes = [Pass::Blit(vec![BlitOp::Fill { dst: (&buf, region.ptr as *const u8), len: 8, value: 0x5A }])];
        let (waits, signals) = ([(&never, 1)], [(&fence, 3)]);
        m.submit_passes(&q, &PassSubmission { waits: &waits, passes: &passes, signals: &signals }).unwrap();
        // Not old enough by either rule: nothing is lost.
        let (long, zero) = (std::time::Duration::from_secs(3600), std::time::Duration::ZERO);
        assert_eq!(recover_lost_after(long, long), 0);
        assert_eq!(m.completed(&fence), 0);
        // Other tests' command buffers may be in flight too (those that do not mind): at least ours.
        assert!(recover_lost_after(long, zero) >= 1);
        // SAFETY: inside the region.
        assert_eq!(unsafe { std::slice::from_raw_parts(region.ptr, 8) }, [0x5A; 8]);
        assert_eq!(m.completed(&fence), 3);
        // Let it finish: a late completion of a recovered command buffer changes nothing.
        never.0.setSignaledValue(1);
        let after = [(&fence, 4)];
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &[], signals: &after }).unwrap();
        assert!(m.wait(&fence, 4, 5000), "the queue runs on behind the recovered command buffer");
    }

    /// A stop wakes a command buffer asleep on an event nobody signals, as a title's queues are
    /// when the process is told to stop: it completes on the GPU (its fence is reached by the GPU,
    /// not paid), and nothing is left in flight.
    #[test]
    fn a_stop_wakes_command_buffers_waiting_on_events_nobody_signals() {
        let _alone = FLIGHTS_IN_TESTS.write().unwrap_or_else(|e| e.into_inner());
        let m = Metal::new().unwrap();
        let q = m.create_queue().unwrap();
        let (never, fence) = (m.create_fence(0).unwrap(), m.create_fence(0).unwrap());
        let (waits, signals) = ([(&never, 1)], [(&fence, 3)]);
        m.submit_passes(&q, &PassSubmission { waits: &waits, passes: &[], signals: &signals }).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(m.completed(&fence), 0, "the command buffer sleeps on its wait");
        let (in_flight, left) =
            wake_and_drain(&[never.0.clone(), q.2.clone()], std::time::Duration::from_secs(5), std::time::Duration::from_secs(5));
        assert!(in_flight >= 1);
        assert_eq!(left, 0, "every command buffer completed on the GPU");
        assert_eq!(m.completed(&fence), 3);
        // The events a stop wakes are tracked as they are made, and let go with them.
        let tracked = |e: &Retained<ProtocolObject<dyn MTLSharedEvent>>| {
            EVENTS.lock().unwrap().0.iter().any(|w| w.load().is_some_and(|t| Retained::as_ptr(&t) == Retained::as_ptr(e)))
        };
        assert!(tracked(&never.0) && tracked(&q.2));
    }

    /// A command buffer the stop's raised events do not wake (here asleep on an event the stop
    /// does not know; in a game, one still running on the GPU) gets the longer wait: one that
    /// completes within it is not left behind, and one that never does is paid and left after
    /// `longest`, not before and not much after.
    #[test]
    fn a_stop_waits_longer_for_command_buffers_its_events_do_not_wake() {
        let _alone = FLIGHTS_IN_TESTS.write().unwrap_or_else(|e| e.into_inner());
        let ms = std::time::Duration::from_millis;
        let m = Metal::new().unwrap();
        let q = m.create_queue().unwrap();
        // Finishes 300 ms into the stop: past the first wait, within the longer one.
        let (late, fence) = (m.create_fence(0).unwrap(), m.create_fence(0).unwrap());
        m.submit_passes(&q, &PassSubmission { waits: &[(&late, 1)], passes: &[], signals: &[(&fence, 2)] }).unwrap();
        let event = late.0.clone();
        struct Send<T>(T);
        // SAFETY: test only; Metal's shared events may be signalled from any thread.
        unsafe impl<T> std::marker::Send for Send<T> {}
        let event = Send(event);
        let finisher = std::thread::spawn(move || {
            let event = event;
            std::thread::sleep(ms(300));
            event.0.setSignaledValue(1);
        });
        let started = std::time::Instant::now();
        let (in_flight, left) = wake_and_drain(&[], ms(50), ms(3000));
        finisher.join().unwrap();
        assert!(in_flight >= 1);
        assert_eq!(left, 0, "it completed within the longer wait");
        assert_eq!(m.completed(&fence), 2, "on the GPU");
        assert!(started.elapsed() < ms(2500), "the stop goes on once nothing is in flight");
        // Never completes: left after the bound, its debt paid.
        let (never, owed) = (m.create_fence(0).unwrap(), m.create_fence(0).unwrap());
        m.submit_passes(&q, &PassSubmission { waits: &[(&never, 1)], passes: &[], signals: &[(&owed, 4)] }).unwrap();
        let started = std::time::Instant::now();
        let (_, left) = wake_and_drain(&[], ms(50), ms(400));
        let took = started.elapsed();
        assert_eq!(left, 1);
        assert!(took >= ms(400) && took < ms(1500), "bounded: {took:?}");
        assert_eq!(m.completed(&owed), 4, "what it owed is paid on the CPU");
        never.0.setSignaledValue(1);
    }


    /// A stop empties the residency sets it tracks, so nothing it made stays resident.
    #[test]
    fn a_stop_empties_the_residency_sets() {
        use objc2_foundation::NSObjectProtocol;
        let m = Metal::new().unwrap();
        if !m.dev.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
            return;
        }
        let set = track_residency_set(m.dev.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()).unwrap());
        let buf = m.dev.newBufferWithLength_options(1 << 20, MTLResourceOptions::StorageModePrivate).unwrap();
        set.addAllocation(ProtocolObject::from_ref(&*buf));
        set.commit();
        assert_eq!(set.allocationCount(), 1);
        assert!(SETS.lock().unwrap().0.iter().any(|w| w.load().is_some_and(|t| Retained::as_ptr(&t) == Retained::as_ptr(&set))));
        assert!(release_sets(std::slice::from_ref(&set)) >= 1 << 20);
        assert_eq!(set.allocationCount(), 0);
    }

    #[test]
    fn recovery_writes_the_fills_before_it_signals() {
        struct Send<T>(T);
        // SAFETY: test only; the recovery runs on one other thread while this one only reads.
        unsafe impl<T> std::marker::Send for Send<T> {}
        let m = Metal::new().unwrap();
        let len = 64 << 20;
        let region = Region::new(len, None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        let q = m.create_queue().unwrap();
        let fence = m.create_fence(0).unwrap();
        let passes = [Pass::Blit(vec![BlitOp::Fill { dst: (&buf, region.ptr as *const u8), len, value: 1 }])];
        let signals = [(&fence, 1)];
        let owed = Send(Owed::new(&q, &PassSubmission { waits: &[], passes: &passes, signals: &signals }, false));
        let t = std::thread::spawn(move || {
            let owed = owed;
            owed.0.recover();
        });
        while fence.0.signaledValue() < 1 {
            std::hint::spin_loop();
        }
        // SAFETY: inside the region; read as the title would once the fence is reached.
        let last = unsafe { std::ptr::read_volatile(region.ptr.add(len - 1)) };
        t.join().unwrap();
        assert_eq!(last, 1, "the fence was reached before the fill it covers was written");
    }
}

#[cfg(test)]
mod sampler_tests {
    use super::*;

    #[test]
    fn dynamic_samplers_apply_the_metal_border_descriptor() {
        for (color, expected) in [
            (BorderColor::TransparentBlack, MTLSamplerBorderColor::TransparentBlack),
            (BorderColor::OpaqueBlack, MTLSamplerBorderColor::OpaqueBlack),
            (BorderColor::OpaqueWhite, MTLSamplerBorderColor::OpaqueWhite),
        ] {
            let desc = dynamic_sampler_descriptor(&SamplerDesc {
                min: Filter::Nearest,
                mag: Filter::Nearest,
                mip: Filter::Nearest,
                address: [Address::Border, Address::Clamp, Address::Mirror],
                border_color: color,
                max_anisotropy: 1,
            });
            assert_eq!(desc.sAddressMode(), MTLSamplerAddressMode::ClampToBorderColor);
            assert_eq!(desc.tAddressMode(), MTLSamplerAddressMode::ClampToEdge);
            assert_eq!(desc.rAddressMode(), MTLSamplerAddressMode::MirrorRepeat);
            assert_eq!(desc.borderColor(), expected);
        }
    }

    #[test]
    fn dynamic_samplers_sample_their_border_color_and_keep_distinct_cache_keys() {
        use objc2_foundation::NSString;

        const GREEN: [u8; 4] = [0, 255, 0, 255];
        let metal = Metal::new().expect("Metal device");
        let source = metal.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        metal.upload_texture(&source, 0, 0, &GREEN.repeat(4), 8, 0);
        let mut output_desc = TextureDesc::d2(Format::Rgba8Unorm, 4, 1);
        output_desc.render_target = true;
        let output = metal.create_texture(&output_desc).unwrap();

        let library = metal.dev.newLibraryWithSource_options_error(&NSString::from_str(r#"
            #include <metal_stdlib>
            using namespace metal;

            vertex float4 test_vertex(uint i [[vertex_id]]) {
                return float4(i == 1 ? 3.0 : -1.0, i == 2 ? 3.0 : -1.0, 0.0, 1.0);
            }
            fragment float4 test_fragment(
                float4 position [[position]],
                texture2d<float> tex [[texture(0)]],
                sampler s [[sampler(0)]]
            ) {
                const float2 uv[] = {
                    float2(-0.25, 0.25), float2(0.25, 0.25),
                    float2(1.25, 0.25), float2(0.25, -0.25),
                };
                return tex.sample(s, uv[uint(position.x)]);
            }
        "#), None).expect("compile generated Metal shader source");

        let desc = MTLRenderPipelineDescriptor::new();
        let vertex = library.newFunctionWithName(&NSString::from_str("test_vertex")).unwrap();
        let fragment = library.newFunctionWithName(&NSString::from_str("test_fragment")).unwrap();
        desc.setVertexFunction(Some(&vertex));
        desc.setFragmentFunction(Some(&fragment));
        // SAFETY: color attachment 0 exists.
        unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) }
            .setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        let pipeline = metal.dev.newRenderPipelineStateWithDescriptor_error(&desc).unwrap();
        let queue = metal.dev.newCommandQueue().unwrap();

        let sample = |sampler: &ProtocolObject<dyn MTLSamplerState>| {
            let pass = MTLRenderPassDescriptor::new();
            // SAFETY: color attachment 0 exists.
            let a = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            a.setTexture(Some(&output.tex));
            a.setLoadAction(MTLLoadAction::Clear);
            a.setClearColor(MTLClearColor { red: 1.0, green: 0.0, blue: 1.0, alpha: 1.0 });
            a.setStoreAction(MTLStoreAction::Store);
            let cb = queue.commandBuffer().unwrap();
            let e = cb.renderCommandEncoderWithDescriptor(&pass).unwrap();
            e.setRenderPipelineState(&pipeline);
            // SAFETY: the generated fragment shader uses texture and sampler slot 0, and
            // the vertex shader draws a fullscreen triangle using vertex IDs 0 through 2.
            unsafe {
                e.setFragmentTexture_atIndex(Some(&source.tex), 0);
                e.setFragmentSamplerState_atIndex(Some(sampler), 0);
                e.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
            e.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed, "{:?}", cb.error());

            let mut bytes = [0; 16];
            metal.read_texture(&output, 0, 0, &mut bytes, 16, 0);
            *bytes.as_chunks::<4>().0.first_chunk::<4>().unwrap()
        };

        // Independently construct the edge sampler the old dynamic path used.
        let edge_desc = MTLSamplerDescriptor::new();
        edge_desc.setMinFilter(MTLSamplerMinMagFilter::Nearest);
        edge_desc.setMagFilter(MTLSamplerMinMagFilter::Nearest);
        edge_desc.setMipFilter(MTLSamplerMipFilter::Nearest);
        edge_desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        edge_desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        edge_desc.setRAddressMode(MTLSamplerAddressMode::ClampToEdge);
        edge_desc.setSupportArgumentBuffers(true);
        let legacy_edge = metal.dev.newSamplerStateWithDescriptor(&edge_desc).unwrap();
        assert_eq!(sample(&legacy_edge), [GREEN; 4]);

        let cases = [
            (Address::Border, BorderColor::OpaqueWhite, [255; 4]),
            (Address::Border, BorderColor::OpaqueBlack, [0, 0, 0, 255]),
            (Address::Border, BorderColor::TransparentBlack, [0; 4]),
            (Address::Clamp, BorderColor::TransparentBlack, GREEN),
        ];
        let cached_before = metal.samplers.borrow().len();
        let mut ids = std::collections::HashSet::new();
        for (mode, border_color, outside) in cases {
            let key = SamplerDesc {
                min: Filter::Nearest,
                mag: Filter::Nearest,
                mip: Filter::Nearest,
                address: [mode, Address::Clamp, Address::Clamp],
                border_color,
                max_anisotropy: 1,
            };
            let sampler = metal.create_sampler(&key).unwrap();
            let id = resource_id(sampler.0.gpuResourceID());
            assert!(ids.insert(id), "two distinct samplers shared an ID");
            let repeated = metal.create_sampler(&key).unwrap();
            assert_eq!(resource_id(repeated.0.gpuResourceID()), id, "repeat lookup changed cache identity");

            let pixels = sample(&sampler.0);
            assert_eq!(pixels, [outside, GREEN, outside, GREEN], "{mode:?}, {border_color:?}");
            println!("{mode:?} {border_color:?}: {pixels:?}");
        }
        assert_eq!(metal.samplers.borrow().len(), cached_before + cases.len());
    }

    #[test]
    fn a_sampler_in_a_heap_keeps_its_id() {
        let m = Metal::new().unwrap();
        let desc = |min_mag| SamplerDesc {
            min: min_mag,
            mag: min_mag,
            mip: Filter::Nearest,
            address: [Address::Clamp; 3],
            border_color: BorderColor::TransparentBlack,
            max_anisotropy: 1,
        };
        let id = |s: &Sampler| resource_id(s.0.gpuResourceID());
        // The host drops its sampler once the ID is in the heap; a later one must not take that ID.
        let first = id(&m.create_sampler(&desc(Filter::Nearest)).unwrap());
        let second = id(&m.create_sampler(&desc(Filter::Linear)).unwrap());
        assert_ne!(first, second, "a freed sampler's ID was reused while a heap still names it");
        assert_eq!(id(&m.create_sampler(&desc(Filter::Nearest)).unwrap()), first, "samplers are shared by descriptor");
    }

    #[test]
    fn heap_samplers_keep_distinct_ids_for_independent_min_and_mag_filters() {
        use MTLSamplerMinMagFilter as MetalFilter;

        let m = Metal::new().unwrap();
        let nearest = SamplerDesc {
            min: Filter::Nearest,
            mag: Filter::Nearest,
            mip: Filter::Nearest,
            address: [Address::Clamp; 3],
            border_color: BorderColor::TransparentBlack,
            max_anisotropy: 1,
        };
        let cases = [
            (nearest, (MetalFilter::Nearest, MetalFilter::Nearest)),
            (SamplerDesc { min: Filter::Linear, ..nearest }, (MetalFilter::Linear, MetalFilter::Nearest)),
            (SamplerDesc { mag: Filter::Linear, ..nearest }, (MetalFilter::Nearest, MetalFilter::Linear)),
        ];
        let id = |key: &SamplerDesc| resource_id(m.create_sampler(key).unwrap().0.gpuResourceID());
        // Like a heap entry, keep just the ID after creating each sampler.
        let mut ids = std::collections::HashSet::new();
        let first = id(&nearest);
        for (key, expected) in cases {
            let desc = dynamic_sampler_descriptor(&key);
            assert_eq!((desc.minFilter(), desc.magFilter()), expected);
            assert!(ids.insert(id(&key)), "distinct filter pairs shared a sampler ID");
        }
        assert_eq!(id(&nearest), first, "samplers are shared by descriptor");
    }
}

#[cfg(test)]
mod heap_set_tests {
    use super::*;

    /// Whether `gone` holds once the work made before has finished: other tests' command buffers
    /// may be in flight, so this waits for them (at most 10 s).
    fn let_go(m: &Metal, gone: impl Fn() -> bool) -> bool {
        let started = std::time::Instant::now();
        loop {
            m.let_go_of_done();
            if gone() {
                return true;
            }
            if started.elapsed() > std::time::Duration::from_secs(10) {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// A texture a heap let go is kept while a command buffer committed before is in flight (it
    /// may still read the texture through an entry or a table copy written before), and goes once
    /// Metal has reported on that command buffer.
    #[test]
    fn a_retired_texture_outlives_the_command_buffers_committed_before() {
        let _order = FLIGHTS_IN_TESTS.read().unwrap_or_else(|e| e.into_inner());
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        if heap.set.is_none() { return }
        let texture = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        let weak = objc2::rc::Weak::from_retained(&texture.tex);
        m.write_texture(&heap, 0, &texture, 0.0);
        m.use_heap_cb(&heap, &mut Vec::new(), |_| {});
        // A command buffer that may read the entry is committed and not yet done.
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let owed = std::sync::Arc::new(Owed::of(Vec::new(), &[], 0));
        flights().push_back(Flight { serial, queue: 0, since: std::time::Instant::now(), owed });
        drop(texture);
        m.clear_entry(&heap, 0);
        m.use_heap_cb(&heap, &mut Vec::new(), |_| {});
        m.let_go_of_done();
        assert!(weak.load().is_some(), "kept while the command buffer committed before runs");
        landed(serial, 0, false);
        assert!(let_go(&m, || weak.load().is_none()), "let go once it is done");
    }

    #[test]
    fn raw_buffer_fast_path_matches_reference_ownership_and_residency() {
        let m = Metal::new().unwrap().with_heap_sets();
        let mut fast = m.create_heap(HeapKind::Resource, 8).unwrap();
        let mut slow = m.create_heap(HeapKind::Resource, 8).unwrap();
        fast.fast_buffers = true;
        slow.fast_buffers = false;
        let (a, pa) = m.host_buffer(65536).unwrap();
        let (b, pb) = m.host_buffer(65536).unwrap();
        for n in 0..512 {
            let (buffer, ptr) = if n % 31 == 0 { (&b, pb) } else { (&a, pa) };
            let index = n % 8;
            for heap in [&fast, &slow] {
                m.write_buffer(heap, index, buffer, ptr.wrapping_add(n * 16), 64);
                if n % 19 == 0 { m.clear_entry(heap, index); }
            }
            for index in 0..8 {
                assert_eq!(fast.capture_descriptor(index).words, slow.capture_descriptor(index).words);
            }
            assert_eq!(fast.allocation_counts(), slow.allocation_counts());
            if n % 17 == 0 {
                fast.commit_set(&m); slow.commit_set(&m);
                assert_eq!(fast.committed(), slow.committed());
                assert_eq!(fast.set_commits.get(), slow.set_commits.get());
            }
        }
        // Many raw entries share one wrapper, as well as the allocation counted above.
        m.write_buffer(&fast, 0, &a, pa, 64);
        m.write_buffer(&fast, 1, &a, pa.wrapping_add(64), 64);
        if fast.dense.is_none() { assert!(Rc::ptr_eq(&fast.refs.borrow()[&0], &fast.refs.borrow()[&1])); }
        let id = (&*a.buf as *const ProtocolObject<dyn MTLBuffer>).cast::<()>() as usize;
        assert_eq!((fast.entry(0).unwrap().0, fast.entry(1).unwrap().0), (id, id));
    }

    /// `VICTORY_DENSE_ENTRIES`: the slab against the reference maps, on real Metal objects and
    /// residency sets. Same entry words, counts, typed views and dirty state after every write;
    /// the same adds, removes, commits and committed membership at every commit.
    #[test]
    fn dense_ownership_matches_the_reference_maps_through_random_writes_and_commits() {
        let m = Metal::new().unwrap().with_heap_sets();
        const SLOTS: [usize; 8] = [0, 1, 2, 4094, 4095, 4096, 4097, 4999];
        let heaps = [m.create_heap_as(HeapKind::Resource, 5000, true).unwrap(), m.create_heap_as(HeapKind::Resource, 5000, false).unwrap()];
        assert!(heaps[0].dense.is_some() && heaps[1].dense.is_none());
        let buffers: Vec<_> = (0..3).map(|_| m.host_buffer(65536).unwrap()).collect();
        let textures: Vec<_> = (0..3).map(|_| m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 4, 4)).unwrap()).collect();
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut removals = 0;
        for n in 0..12_000 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            let slot = SLOTS[seed as usize % SLOTS.len()];
            let other = SLOTS[(seed >> 8) as usize % SLOTS.len()];
            let (buffer, ptr) = &buffers[(seed >> 16) as usize % buffers.len()];
            let at = ptr.wrapping_add(((seed >> 20) as usize % 64) * 256);
            let op = (seed >> 32) % 9;
            // A texture only the heaps keep alive: its removal is the last reference.
            let own = (op == 8).then(|| m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap());
            for heap in &heaps {
                match op {
                    0 | 1 => m.clear_entry(heap, slot),
                    2 | 3 => m.write_buffer(heap, slot, buffer, at, 64 + (seed >> 40) as u32 % 2 * 64),
                    4 => { m.write_typed_buffer(heap, slot, buffer, at, 1024, Format::Rgba32Uint, (seed >> 41) & 1 == 0); }
                    5 => m.write_texture(heap, slot, &textures[(seed >> 44) as usize % textures.len()], 0.0),
                    6 | 7 => m.write_descriptor(heap, slot, &heap.capture_descriptor(other)),
                    _ => m.write_texture(heap, slot, own.as_ref().unwrap(), 0.0),
                }
            }
            drop(own);
            let [dense, maps] = &heaps;
            for i in [slot, other] {
                assert_eq!(dense.capture_descriptor(i).words, maps.capture_descriptor(i).words);
                let (a, b) = (dense.entry(i), maps.entry(i));
                assert_eq!(a.as_ref().map(|e| e.0), b.as_ref().map(|e| e.0));
                match (a.and_then(|e| e.1), b.and_then(|e| e.1)) {
                    (None, None) => {}
                    (Some(a), Some(b)) => assert!(std::ptr::eq(&*a, &*b)),
                    _ => panic!("typed views differ at {i}"),
                }
            }
            assert_eq!(dense.allocation_counts(), maps.allocation_counts());
            assert_eq!((dense.named_entries(), dense.named_allocations()), (maps.named_entries(), maps.named_allocations()));
            assert_eq!(dense.typed_views.get(), maps.typed_views.get());
            assert_eq!(dense.dirty.get(), maps.dirty.get(), "write {n}");
            assert_eq!(dense.set_commits.get(), maps.set_commits.get(), "write {n}");
            if n % 13 == 0 {
                let calls = heaps.each_ref().map(|heap| {
                    let before = set_calls(&m);
                    m.use_heap_cb(heap, &mut Vec::new(), |_| {});
                    let after = set_calls(&m);
                    assert_set_matches(heap);
                    (after.0 - before.0, after.1 - before.1)
                });
                assert_eq!(calls[0], calls[1], "write {n}");
                removals += calls[0].1;
                assert_eq!(dense.committed(), maps.committed());
                assert_eq!(dense.set_commits.get(), maps.set_commits.get());
            }
        }
        assert!(removals > 50, "the sequence must commit removals: {removals}");
    }

    /// Coordinator/GPU companion to residency::tests: the same transitions through actual
    /// descriptor snapshots, retained Metal objects, and MTLResidencySet membership.
    #[test]
    fn prepared_descriptors_preserve_words_and_random_residency() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 37).unwrap();
        let (a, pa) = m.host_buffer(65536).unwrap();
        let (b, pb) = m.host_buffer(65536).unwrap();
        let texture = m.create_texture(&TextureDesc { mips: 4, ..TextureDesc::d2(Format::Rgba8Unorm, 16, 16) }).unwrap();
        let view = m.create_texture_view(&texture, &ViewDesc { format: Format::Rgba8UnormSrgb, mips: (1, 2), slices: (0, 1) }).unwrap();
        m.write_buffer(&heap, 0, &a, pa, 64);
        m.write_buffer(&heap, 1, &a, pa.wrapping_add(32), 256);
        m.write_buffer(&heap, 2, &b, pb, 128);
        m.write_typed_buffer(&heap, 3, &a, pa, 256, Format::R32Uint, true);
        m.write_texture(&heap, 4, &texture, 0.0);
        m.write_texture(&heap, 5, &view, 1.0);
        m.clear_entry(&heap, 6);
        let entries: Vec<_> = (0..7).map(|i| heap.capture_descriptor(i)).collect();
        let mut seed = 0x243f6a8885a308d3u64;
        for n in 0..5000 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            let slot = seed as usize % heap.count;
            let entry = &entries[(seed >> 32) as usize % entries.len()];
            m.write_descriptor(&heap, slot, entry);
            assert_eq!(heap.capture_descriptor(slot).words, entry.words);
            let mut expected = HashMap::new();
            let mut typed = 0;
            for (id, view) in (0..heap.count).filter_map(|i| heap.entry(i)) {
                *expected.entry(id).or_insert(0) += 1;
                typed += usize::from(view.is_some());
            }
            assert_eq!(heap.allocation_counts(), expected);
            assert_eq!(heap.typed_views.get(), typed);
            if n % 17 == 0 {
                heap.commit_set(&m);
                assert_set_matches(&heap);
            }
        }
        heap.commit_set(&m);
        assert_set_matches(&heap);
        let samplers = m.create_heap(HeapKind::Sampler, 2).unwrap();
        let sampler = m.create_sampler(&SamplerDesc {
            min: Filter::Nearest, mag: Filter::Linear, mip: Filter::Nearest,
            address: [Address::Clamp; 3], border_color: BorderColor::OpaqueWhite, max_anisotropy: 1,
        }).unwrap();
        m.write_sampler(&samplers, 0, &sampler, 1.25);
        let entry = samplers.capture_descriptor(0);
        m.clear_entry(&samplers, 0);
        m.write_descriptor(&samplers, 1, &entry);
        assert_eq!(samplers.capture_descriptor(1).words, entry.words);
    }

    fn assert_set_matches(heap: &Heap) {
        let set = heap.set.as_ref().unwrap();
        let counts = heap.allocation_counts();
        assert_eq!(set.allocationCount(), counts.len() + 1, "includes the heap buffer");
        assert!(set.containsAllocation(ProtocolObject::from_ref(&*heap.buf)));
        assert_eq!(heap.committed(), counts.keys().copied().collect::<HashSet<_>>());
        heap.each_allocation(|_, allocation| {
            assert!(set.containsAllocation(ProtocolObject::from_ref(&**allocation)));
        });
        assert!(!heap.dirty.get());
        assert_eq!(heap.pending_retired(), (0, 0));
    }

    fn set_calls(m: &Metal) -> (u64, u64) {
        let counts = m.residency_counts();
        (counts.set_adds, counts.set_removes)
    }

    #[test]
    fn typed_buffer_views_are_reused_and_identical_entries_are_not_written() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 2).unwrap();
        let (buf, ptr) = m.host_buffer(65536).unwrap();
        m.write_typed_buffer(&heap, 0, &buf, ptr, 1024, Format::Rgba32Uint, true);
        let first = heap.entry(0).unwrap().1.expect("a texture-buffer view");
        let writes = m.residency_counts().entry_writes;
        m.write_typed_buffer(&heap, 0, &buf, ptr, 1024, Format::Rgba32Uint, true);
        assert_eq!(m.residency_counts().entry_writes, writes, "the second entry write is skipped");
        assert!(std::ptr::eq(&*first, &*heap.entry(0).unwrap().1.unwrap()));
        m.write_typed_buffer(&heap, 1, &buf, ptr, 1024, Format::Rgba32Uint, true);
        assert!(std::ptr::eq(&*first, &*heap.entry(1).unwrap().1.unwrap()), "shared across entries");

        // The fitted width is the same, but the descriptor's byte size is different.
        let writes = m.residency_counts().entry_writes;
        m.write_typed_buffer(&heap, 0, &buf, ptr, 1025, Format::Rgba32Uint, true);
        assert!(std::ptr::eq(&*first, &*heap.entry(0).unwrap().1.unwrap()));
        assert_eq!(m.residency_counts().entry_writes, writes + 1, "changed metadata is still written");
        m.write_typed_buffer(&heap, 0, &buf, ptr, 1024, Format::Rgba32Uint, false);
        assert!(!std::ptr::eq(&*first, &*heap.entry(0).unwrap().1.unwrap()), "writable usage is part of the key");
        m.write_typed_buffer(&heap, 0, &buf, ptr, 1024, Format::Rgba32Float, true);
        assert!(!std::ptr::eq(&*first, &*heap.entry(0).unwrap().1.unwrap()), "format is part of the key");
    }

    #[test]
    fn texture_views_are_cached_by_the_fitted_ranges() {
        let m = Metal::new().unwrap().with_heap_sets();
        let texture = m.create_texture(&TextureDesc {
            dim: TexDim::D2Array, depth_or_layers: 4, mips: 4,
            ..TextureDesc::d2(Format::Rgba8Unorm, 32, 32)
        }).unwrap();
        let all = ViewDesc { format: Format::Rgba8Unorm, mips: (1, usize::MAX), slices: (1, usize::MAX) };
        let fitted = ViewDesc { mips: (1, 3), slices: (1, 3), ..all };
        let first = m.create_texture_view(&texture, &all).unwrap();
        let second = m.create_texture_view(&texture, &fitted).unwrap();
        assert!(std::ptr::eq(&*first.tex, &*second.tex));
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        m.write_texture(&heap, 0, &first, 0.0);
        let writes = m.residency_counts().entry_writes;
        m.write_texture(&heap, 0, &second, 0.0);
        assert_eq!(m.residency_counts().entry_writes, writes);
        for other in [
            ViewDesc { mips: (2, 2), ..fitted },
            ViewDesc { slices: (0, 3), ..fitted },
            ViewDesc { format: Format::Rgba8UnormSrgb, ..fitted },
        ] {
            let view = m.create_texture_view(&texture, &other).unwrap();
            assert!(!std::ptr::eq(&*first.tex, &*view.tex), "distinct views must stay distinct");
        }
    }

    #[test]
    fn replaced_allocations_live_until_the_next_heap_use_commits_their_removal() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        let Some(set) = &heap.set else { return };
        assert_set_matches(&heap);
        let heap_only = set.allocatedSize();
        let old = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 1024, 1024)).unwrap();
        let old_weak = objc2::rc::Weak::from_retained(&old.tex);
        m.write_texture(&heap, 0, &old, 0.0);
        let mut attached = Vec::new();
        m.use_heap_cb(&heap, &mut attached, |_| {});
        assert_set_matches(&heap);
        let (bytes, commits) = (set.allocatedSize(), heap.set_commits.get());
        let (adds, removes) = set_calls(&m);
        let new = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 16, 16)).unwrap();
        m.write_texture(&heap, 0, &new, 0.0);
        drop(old);
        assert_eq!(heap.set_commits.get(), commits, "replacement must not commit");
        assert_eq!(set.allocatedSize(), bytes, "the set still reports the last commit");
        assert!(old_weak.load().is_some(), "the pending removal owns the last reference");
        assert_eq!(heap.pending_retired().0, 1);
        assert_eq!(set_calls(&m), (adds, removes), "entry edits issue no set calls");
        m.use_heap_cb(&heap, &mut attached, |_| panic!("already attached to this command buffer"));
        assert_eq!(heap.set_commits.get(), commits + 1);
        assert_eq!(set_calls(&m), (adds + 1, removes + 1), "one add and one remove in one commit");
        assert_set_matches(&heap);
        assert_eq!(heap.pending_retired().0, 0);
        assert!(let_go(&m, || old_weak.load().is_none()), "the committed removal releases the old texture once earlier work is done");
        // Metal reports sizes in its own granules: a 16x16 texture may add nothing it shows.
        assert!(set.allocatedSize() < bytes && set.allocatedSize() >= heap_only, "{} of {bytes}, heap {heap_only}", set.allocatedSize());
        assert_eq!(set.allocationCount(), 2);
    }

    #[test]
    fn an_allocation_removed_then_added_before_commit_stays_resident() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 2).unwrap();
        let Some(set) = &heap.set else { return };
        let texture = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 128, 128)).unwrap();
        let mut attached = Vec::new();
        m.write_texture(&heap, 0, &texture, 0.0);
        m.use_heap_cb(&heap, &mut attached, |_| {});
        assert_set_matches(&heap);
        let (bytes, commits) = (set.allocatedSize(), heap.set_commits.get());
        let calls = set_calls(&m);
        m.clear_entry(&heap, 0);
        m.write_texture(&heap, 1, &texture, 0.0);
        assert_eq!(heap.set_commits.get(), commits);
        assert_eq!(set_calls(&m), calls);
        assert!(set.containsAllocation(ProtocolObject::from_ref(&*texture.tex)), "Metal membership was never changed");
        m.use_heap_cb(&heap, &mut attached, |_| panic!("already attached"));
        assert_eq!(heap.set_commits.get(), commits, "unchanged membership needs no commit");
        assert_eq!(set_calls(&m), calls);
        assert_set_matches(&heap);
        assert_eq!(set.allocationCount(), 2);
        assert_eq!(set.allocatedSize(), bytes);
        assert_eq!(heap.pending_retired().0, 0);
        assert!(set.containsAllocation(ProtocolObject::from_ref(&*texture.tex)));
    }

    #[test]
    fn an_allocation_added_and_retired_before_commit_never_enters_the_set() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        let Some(set) = &heap.set else { return };
        assert_set_matches(&heap);
        let calls = set_calls(&m);
        let commits = heap.set_commits.get();
        let texture = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        let weak = objc2::rc::Weak::from_retained(&texture.tex);
        m.write_texture(&heap, 0, &texture, 0.0);
        assert!(!set.containsAllocation(ProtocolObject::from_ref(&*texture.tex)));
        m.clear_entry(&heap, 0);
        drop(texture);
        assert!(weak.load().is_some(), "retirement prevents address reuse");
        assert_eq!(set_calls(&m), calls);
        m.use_heap_cb(&heap, &mut Vec::new(), |_| {});
        assert_eq!(set_calls(&m), calls);
        assert_eq!(heap.set_commits.get(), commits);
        assert_set_matches(&heap);
        assert!(let_go(&m, || weak.load().is_none()), "a no-op reconciliation also releases retired objects");
    }

    #[test]
    fn alternating_entry_changes_issue_only_the_net_membership_change() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        let Some(set) = &heap.set else { return };
        let texture = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        let mut attached = Vec::new();
        assert_set_matches(&heap);
        // Add-remove-add, then remove-add-remove, each spanning a single commit.
        for present in [true, false] {
            let (adds, removes) = set_calls(&m);
            let commits = heap.set_commits.get();
            for name in [present, !present, present] {
                if name {
                    m.write_texture(&heap, 0, &texture, 0.0);
                } else {
                    m.clear_entry(&heap, 0);
                }
                assert_eq!(set_calls(&m), (adds, removes));
                assert_eq!(heap.set_commits.get(), commits);
            }
            m.use_heap_cb(&heap, &mut attached, |_| {});
            assert_eq!(set_calls(&m), (adds + u64::from(present), removes + u64::from(!present)));
            assert_eq!(heap.set_commits.get(), commits + 1);
            assert_eq!(set.containsAllocation(ProtocolObject::from_ref(&*texture.tex)), present);
            assert_set_matches(&heap);
        }
    }

    #[test]
    fn a_retired_typed_entry_keeps_both_the_buffer_and_its_view_until_commit() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        if heap.set.is_none() { return }
        let (buffer, ptr) = m.host_buffer(65536).unwrap();
        let buffer_weak = objc2::rc::Weak::from_retained(&buffer.buf);
        m.write_typed_buffer(&heap, 0, &buffer, ptr, 1024, Format::Rgba32Uint, true);
        let view_weak = objc2::rc::Weak::from_retained(&heap.entry(0).unwrap().1.unwrap());
        let mut attached = Vec::new();
        m.use_heap_cb(&heap, &mut attached, |_| {});
        let commits = heap.set_commits.get();
        drop(buffer); // Its cache goes too, but the entry still owns both objects.
        m.clear_entry(&heap, 0);
        assert_eq!(heap.pending_retired(), (1, 1));
        assert_eq!(heap.set_commits.get(), commits);
        assert!(buffer_weak.load().is_some() && view_weak.load().is_some());
        m.use_heap_cb(&heap, &mut attached, |_| panic!("already attached"));
        assert_eq!(heap.set_commits.get(), commits + 1);
        assert_set_matches(&heap);
        assert!(let_go(&m, || buffer_weak.load().is_none() && view_weak.load().is_none()));
    }

    #[test]
    fn a_heap_without_command_buffers_bounds_pending_removals() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        let Some(set) = &heap.set else { return };
        let textures: Vec<_> = (0..RETIRED_OBJECTS + 2)
            .map(|_| m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap())
            .collect();
        m.write_texture(&heap, 0, &textures[0], 0.0);
        let commits = heap.set_commits.get();
        for texture in &textures[1..=RETIRED_OBJECTS] {
            m.write_texture(&heap, 0, texture, 0.0);
        }
        assert_eq!(heap.set_commits.get(), commits);
        assert_eq!(heap.pending_retired().0, RETIRED_OBJECTS);
        m.write_texture(&heap, 0, &textures[RETIRED_OBJECTS + 1], 0.0);
        assert_eq!(heap.set_commits.get(), commits + 1);
        assert_eq!(heap.pending_retired().0, 0);
        assert!(!heap.dirty.get());
        assert_eq!(set.allocationCount(), 2);
        assert!(set.containsAllocation(ProtocolObject::from_ref(&*textures[RETIRED_OBJECTS + 1].tex)));
        assert_eq!(set_calls(&m), (2, 0), "only the heap buffer and final texture were added");
        assert_set_matches(&heap);
    }

    #[test]
    fn retired_views_also_force_reconciliation_at_the_bound() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        if heap.set.is_none() { return }
        let (buffer, ptr) = m.host_buffer(65536).unwrap();
        let commits = heap.set_commits.get();
        let calls = set_calls(&m);
        for _ in 0..RETIRED_OBJECTS - 1 {
            m.write_typed_buffer(&heap, 0, &buffer, ptr, 1024, Format::Rgba32Uint, true);
            m.clear_entry(&heap, 0);
        }
        assert_eq!(heap.pending_retired(), (1, RETIRED_OBJECTS - 1));
        assert!(heap.dirty.get());
        m.write_typed_buffer(&heap, 0, &buffer, ptr, 1024, Format::Rgba32Uint, true);
        m.clear_entry(&heap, 0);
        assert_set_matches(&heap);
        assert_eq!(heap.set_commits.get(), commits, "reconciliation has no net change");
        assert_eq!(set_calls(&m), calls);
    }

    #[test]
    fn multiple_removals_commit_once_and_drop_detaches_before_releasing_objects() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 4).unwrap();
        let Some(set) = heap.set.clone() else { return };
        let mut weak = Vec::new();
        for i in 0..4 {
            let texture = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
            weak.push(objc2::rc::Weak::from_retained(&texture.tex));
            m.write_texture(&heap, i, &texture, 0.0);
        }
        let mut attached = Vec::new();
        m.use_heap_cb(&heap, &mut attached, |_| {});
        assert_set_matches(&heap);
        let commits = heap.set_commits.get();
        let (adds, removes) = set_calls(&m);
        for i in 0..3 {
            m.clear_entry(&heap, i);
        }
        assert_eq!(set_calls(&m), (adds, removes));
        assert!(weak.iter().all(|w| w.load().is_some()));
        m.use_heap_cb(&heap, &mut attached, |_| panic!("already attached"));
        assert_eq!(heap.set_commits.get(), commits + 1);
        assert_eq!(set_calls(&m), (adds, removes + 3));
        assert_set_matches(&heap);
        assert!(let_go(&m, || weak[..3].iter().all(|w| w.load().is_none())));

        // Drop with a committed retirement and a desired addition that Metal never saw.
        m.clear_entry(&heap, 3);
        let new = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        let new_weak = objc2::rc::Weak::from_retained(&new.tex);
        m.write_texture(&heap, 0, &new, 0.0);
        drop(new);
        let calls = set_calls(&m);
        drop(heap);
        assert_eq!(set.allocationCount(), 0);
        assert_eq!(set_calls(&m), calls, "Drop uses removeAllAllocations without reconciliation");
        assert!(weak[3].load().is_none() && new_weak.load().is_none());
    }

    #[test]
    fn without_sets_removed_objects_are_released_at_once() {
        let m = Metal::new().unwrap().with_residency_sets(false);
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        let texture = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        let weak = objc2::rc::Weak::from_retained(&texture.tex);
        m.write_texture(&heap, 0, &texture, 0.0);
        drop(texture);
        assert!(weak.load().is_some());
        m.clear_entry(&heap, 0);
        assert!(weak.load().is_none());
        assert_eq!(heap.pending_retired(), (0, 0));
        assert!(heap.committed().is_empty());
        assert_eq!(set_calls(&m), (0, 0));
        assert_eq!(heap.set_commits.get(), 0);
    }

    #[test]
    fn each_command_buffer_attaches_each_heap_once_and_commits_later_changes() {
        let m = Metal::new().unwrap().with_heap_sets();
        let a = m.create_heap(HeapKind::Resource, 1).unwrap();
        let b = m.create_heap(HeapKind::Sampler, 1).unwrap();
        let Some(set_a) = &a.set else {
            eprintln!("skipped: no residency sets on this system");
            return;
        };
        let attaches = |heap: &Heap, attached_heaps: &mut Vec<*const Heap>| {
            let mut called = false;
            m.use_heap_cb(heap, attached_heaps, |_| called = true);
            assert_set_matches(heap);
            called
        };
        let mut first = Vec::new();
        assert!(attaches(&a, &mut first));
        assert!(attaches(&b, &mut first));

        // Metal reports allocatedSize at the last commit.
        let before = set_a.allocatedSize();
        let texture = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 128, 128)).unwrap();
        m.write_texture(&a, 0, &texture, 0.0);
        assert!(a.dirty.get());
        assert!(!attaches(&a, &mut first));
        assert!(!a.dirty.get());
        assert!(set_a.allocatedSize() > before);

        let mut second = Vec::new();
        assert!(attaches(&a, &mut second));
        assert!(!attaches(&a, &mut second));
    }

    /// Each piece of a split submission is a command buffer of its own, and a residency set keeps
    /// its allocations resident only for the command buffers it was given to: the next piece
    /// attaches every heap again.
    #[test]
    fn the_next_piece_of_a_split_submission_attaches_the_heaps_again() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 1).unwrap();
        if heap.set.is_none() {
            eprintln!("skipped: no residency sets on this system");
            return;
        }
        let q = m.create_queue().unwrap();
        let attaches = |attached: &mut Vec<*const Heap>| {
            let mut called = false;
            m.use_heap_cb(&heap, attached, |_| called = true);
            called
        };
        let mut attached = Vec::new();
        assert!(attaches(&mut attached), "the first piece");
        assert!(!attaches(&mut attached), "once per command buffer");
        let _next = m.next_piece(&q, &[], &mut attached, None).unwrap();
        assert!(attaches(&mut attached), "the next piece is given the heap's set too");
    }

    /// With `VICTORY_METAL_REQUEST_RESIDENCY` a heap's set asks to stay resident when it is made
    /// and after each commit a use makes; without it, never.
    #[test]
    fn a_heaps_set_asks_to_stay_resident_only_when_switched_on() {
        for on in [false, true] {
            let m = Metal::new().unwrap().with_heap_sets().with_requested_residency(on);
            let heap = m.create_heap(HeapKind::Resource, 4).unwrap();
            if heap.set.is_none() {
                eprintln!("skipped: no residency sets on this system");
                return;
            }
            let before = m.residency_counts().requests;
            let t = m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
            m.write_texture(&heap, 0, &t, 0.0);
            let mut attached = Vec::new();
            m.use_heap_cb(&heap, &mut attached, |_| {});
            m.use_heap_cb(&heap, &mut attached, |_| {});
            let made = u64::from(on);
            assert_eq!(before, made, "on={on}: when the set is made");
            assert_eq!(m.residency_counts().requests, 2 * made, "on={on}: again after the one commit");
            drop(heap); // ends the residency it asked for
        }
    }

    /// A heap's residency set holds the heap buffer and each allocation its entries name, once,
    /// and lets an allocation go at the next command-buffer use after its last entry is
    /// overwritten: the set is what a command buffer that uses the heap keeps resident.
    #[test]
    fn a_heaps_set_follows_its_entries() {
        let m = Metal::new().unwrap().with_heap_sets();
        let heap = m.create_heap(HeapKind::Resource, 4).unwrap();
        let Some(set) = &heap.set else {
            eprintln!("skipped: no residency sets on this system");
            return;
        };
        let held = || {
            // Match production use, including releasing objects retained for pending removals.
            m.use_heap_cb(&heap, &mut Vec::new(), |_| {});
            assert_set_matches(&heap);
            set.allocationCount()
        };
        let texture = || m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, 2, 2)).unwrap();
        let (t1, t2) = (texture(), texture());
        m.write_texture(&heap, 0, &t1, 0.0);
        m.write_texture(&heap, 1, &t1, 0.0);
        assert_eq!(held(), 2, "the heap buffer and the texture two entries name");
        m.write_texture(&heap, 0, &t2, 0.0);
        assert_eq!(held(), 3);
        m.write_texture(&heap, 1, &t2, 0.0);
        assert_eq!(held(), 2, "the first texture went with its last entry");
        m.clear_entry(&heap, 0);
        assert_eq!(held(), 2, "an entry still names the second texture");
        m.clear_entry(&heap, 1);
        assert_eq!(held(), 1, "cleared entries let their texture go");
        // SAFETY: the heap buffer holds four entries of `ENTRY` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(heap.buf.contents().as_ptr().cast::<u8>(), 2 * ENTRY) };
        assert!(bytes.iter().all(|b| *b == 0), "a cleared entry reads as one never written");

        let heap_only = set.allocatedSize();
        let arena = crate::mem::Alias::new(65536).unwrap();
        let buf = victory_backend::backend::ComputeBackend::wrap_title_memory(
            &m,
            arena.primary,
            arena.len,
        )
        .unwrap();
        let entry_words = |i: usize| -> [u64; 3] {
            // SAFETY: the heap buffer holds four entries of `ENTRY` (24) bytes.
            unsafe {
                heap.buf
                    .contents()
                    .as_ptr()
                    .cast::<[u64; 3]>()
                    .add(i)
                    .read_unaligned()
            }
        };
        assert!(m.write_typed_buffer(&heap, 0, &buf, arena.primary, 1024, Format::Rgba32Uint, true));
        assert_eq!(m.heap_typed_views(&heap), 1);
        assert_ne!(entry_words(0)[1], 0, "aligned typed buffer writes textureViewID");
        // SAFETY: `1024` and `4` are inside the 64 KiB `arena` allocation.
        let second = unsafe { arena.primary.add(1024) };
        let unaligned = unsafe { arena.primary.add(4) };
        assert!(m.write_typed_buffer(&heap, 1, &buf, second, 1024, Format::Rgba32Float, true));
        assert_eq!(m.heap_typed_views(&heap), 2);
        assert!(!m.write_typed_buffer(&heap, 2, &buf, unaligned, 1024, Format::Rgba32Float, true));
        assert_eq!(m.heap_typed_views(&heap), 2, "unaligned fallback holds no typed view");
        assert_ne!(entry_words(2)[0], 0, "unaligned fallback preserves gpuVA in word 0");
        assert_eq!(entry_words(2)[1], 0, "unaligned fallback leaves textureViewID null");
        assert_eq!(entry_words(2)[2], 1024);
        assert_eq!(held(), 2, "typed buffer views over one pool add only the pool allocation");
        assert_eq!(
            set.allocatedSize(),
            heap_only + arena.len as u64,
            "buffer-backed texture views do not multiply the pool's residency-set size"
        );
        m.write_buffer(&heap, 0, &buf, arena.primary, 1024);
        assert_eq!(m.heap_typed_views(&heap), 1, "overwriting with write_buffer drops the typed view");
        assert_eq!(entry_words(0)[1], 0, "write_buffer zeroes textureViewID");
        m.clear_entry(&heap, 0);
        m.clear_entry(&heap, 2);
        assert_eq!(held(), 2);
        assert_eq!(m.heap_typed_views(&heap), 1);
        m.clear_entry(&heap, 1);
        assert_eq!(m.heap_typed_views(&heap), 0, "clear_entry drops the last typed view");
        assert_eq!(held(), 1, "cleared typed buffer entries release the pool allocation");
    }

    #[test]
    #[ignore = "benchmark: cargo test -p victory-metal typed_buffer_view_cost_benchmark -- --ignored --nocapture"]
    fn typed_buffer_view_cost_benchmark() {
        fn phys_footprint() -> usize {
            const TASK_VM_INFO: i32 = 22;
            unsafe extern "C" {
                fn mach_task_self() -> u32;
                fn task_info(task: u32, flavor: i32, info: *mut u64, count: *mut u32) -> i32;
            }
            let mut info = [0u64; 64];
            let mut count = 128u32;
            // SAFETY: local buffer of `count` 32-bit words for `TASK_VM_INFO`.
            let kr = unsafe { task_info(mach_task_self(), TASK_VM_INFO, info.as_mut_ptr(), &mut count) };
            if kr == 0 && count >= 38 { info[18] as usize } else { 0 }
        }

        let m = Metal::new().unwrap().with_heap_sets();
        for fmt in [
            Format::R8Unorm,
            Format::R16Float,
            Format::R32Uint,
            Format::Rg32Float,
            Format::Rgba16Uint,
            Format::Rgba32Uint,
            Format::Rgba32Float,
        ] {
            let pf = pixel_format(fmt);
            let align = m.dev.minimumTextureBufferAlignmentForPixelFormat(pf);
            println!("align {fmt:?} (MTLPixelFormat {:?}): {align} B", pf);
        }

        const TOTAL: usize = 40_000;
        let arena_len = TOTAL * 64;
        let arena = crate::mem::Alias::new(arena_len).unwrap();
        // Touch pages upfront so pool page faults do not pollute descriptor measurements.
        // SAFETY: `arena.primary` is a writable `arena_len` allocation.
        unsafe { std::ptr::write_bytes(arena.primary, 0, arena_len) };
        let buf = victory_backend::backend::ComputeBackend::wrap_title_memory(
            &m,
            arena.primary,
            arena.len,
        )
        .unwrap();
        let heap = m.create_heap(HeapKind::Resource, TOTAL).unwrap();
        heap.commit_set(&m);

        let fp0 = phys_footprint();
        let t0 = std::time::Instant::now();
        for i in 0..TOTAL {
            // SAFETY: `i * 64 < arena_len`.
            let ptr = unsafe { arena.primary.add(i * 64) };
            m.write_buffer(&heap, i, &buf, ptr, 64);
        }
        let raw_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let fp_raw = phys_footprint();
        println!(
            "40,000 write_buffer (lazy mirror): {raw_ms:.2} ms, phys_footprint delta = {:.2} KiB",
            (fp_raw as isize - fp0 as isize) as f64 / 1024.0
        );

        for count in [64usize, 512, 1_000, 10_000, 40_000] {
            for i in 0..count {
                // SAFETY: `i * 64 < arena_len`.
                let ptr = unsafe { arena.primary.add(i * 64) };
                m.write_buffer(&heap, i, &buf, ptr, 64);
            }
            let before = phys_footprint();
            let start = std::time::Instant::now();
            for i in 0..count {
                // SAFETY: `i * 64 < arena_len`.
                let ptr = unsafe { arena.primary.add(i * 64) };
                let fmt = match i % 3 {
                    0 => Format::Rgba32Float,
                    1 => Format::Rg32Float,
                    _ => Format::R32Uint,
                };
                m.write_typed_buffer(&heap, i, &buf, ptr, 64, fmt, true);
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let after = phys_footprint();
            let set_bytes = heap.set.as_ref().map_or(0, |s| {
                s.commit();
                s.allocatedSize()
            });
            let delta_kib = (after as isize - before as isize) as f64 / 1024.0;
            println!(
                "materialize {count:>6} write_typed_buffer: {ms:>7.2} ms ({:.2} us/view), phys_footprint delta = {delta_kib:>8.1} KiB ({:.1} B/view), heap_set = {:.2} MiB",
                ms * 1000.0 / count as f64,
                (after as isize - before as isize) as f64 / count as f64,
                set_bytes as f64 / (1024.0 * 1024.0)
            );
            for i in 0..count {
                // SAFETY: `i * 64 < arena_len`.
                let ptr = unsafe { arena.primary.add(i * 64) };
                m.write_buffer(&heap, i, &buf, ptr, 64);
            }
        }
    }
}

#[cfg(test)]
mod heap_usage_tests {
    use super::*;

    #[test]
    fn heap_allocations_give_each_allocation_once_with_its_latest_use() {
        use objc2_foundation::NSObjectProtocol;
        let Ok(mut m) = Metal::new() else { return };
        if !m.dev.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
            return;
        }
        m.residency_sets = true;
        let heap = m.create_heap(HeapKind::Resource, 8).unwrap();
        let tex = |w| m.create_texture(&TextureDesc::d2(Format::Rgba8Unorm, w, w)).unwrap();
        let (a, b, c) = (tex(64), tex(128), tex(256));
        m.write_texture(&heap, 0, &a, 0.0);
        m.write_texture(&heap, 1, &a, 0.0);
        m.write_texture(&heap, 2, &b, 0.0);
        m.write_texture(&heap, 3, &c, 0.0);
        // A view of `c` is `c`'s memory: one allocation, its latest use theirs.
        let view = m.create_texture_view(&c, &ViewDesc { format: Format::Rgba8Unorm, mips: (0, 1), slices: (0, 1) }).unwrap();
        m.write_texture(&heap, 4, &view, 0.0);
        // `a` is named by entries last used at frames 10 and 50, `b` at 20, `c` never, its view at 5.
        let mut all = m.heap_allocations(&heap, &[10, 50, 20, 0, 5, 0, 0, 0]).unwrap();
        all.sort_by_key(|a| a.bytes);
        assert_eq!(all.iter().map(|a| a.latest).collect::<Vec<_>>(), [50, 20, 5], "{all:?}");
        assert!(all.iter().all(|a| a.texture && a.bytes > 0), "{all:?}");
        // Without a residency set nothing is kept resident by the heap.
        m.residency_sets = false;
        let plain = m.create_heap(HeapKind::Resource, 1).unwrap();
        assert_eq!(m.heap_allocations(&plain, &[0]), None);
    }
}
