//! Host pipelines of title shader sets made without the title (`victory translate --sets`): the
//! containers translated with the options the host uses for them, the refusals of the shader
//! path applied, and Metal asked for the pipeline. What only a run knows is stood in for: the
//! input layout is one made of the vertex shader's own input signature (each element at the
//! width it declares, appended in one slot), and the targets are as many as the pixel shader
//! writes (RGBA8, or 32-bit unsigned for an unsigned output) and a depth target. So a pipeline made here
//! says the shaders convert, link and compile for Metal, not that the title's state is right.

use super::shaders;
use super::state::Graphics;
use super::title::{Pipe, Title};
use std::sync::Arc;
use victory_backend::resources::Format;
use victory_metal::metal::Metal;
use victory_shader::emit::RECT_CORNER_SHIFT;

pub struct Offline {
    title: Title<Metal>,
}

/// The stage-in elements of a translated vertex shader's input signature: name, semantic index,
/// component type (1 unsigned, 2 signed, 3 float) and channels.
fn elements(dxil: &[u8]) -> Vec<(String, u32, u32, u32)> {
    use victory_shader::{Container, FourCC};
    let Some(d) = Container::parse(dxil).ok().and_then(|c| c.part(FourCC::ISG1).map(|p| p.data.to_vec())) else {
        return Vec::new();
    };
    let word = |at: usize| d.get(at..at + 4).map_or(0, |w| u32::from_le_bytes(w.try_into().unwrap()));
    (0..word(0) as usize)
        .filter_map(|i| {
            let row = 8 + 32 * i;
            let name = d.get(word(row + 4) as usize..)?;
            let name = String::from_utf8_lossy(&name[..name.iter().position(|&b| b == 0)?]).into_owned();
            let mask = u32::from(*d.get(row + 24)?);
            (word(row + 12) == 0).then(|| (name, word(row + 8), word(row + 16), 32 - mask.leading_zeros()))
        })
        .collect()
}

/// The render targets a translated pixel shader writes: index and component type (1 unsigned,
/// 2 signed, 3 float).
fn target_types(dxil: &[u8]) -> Vec<(u32, u32)> {
    use victory_shader::{Container, FourCC};
    let Some(d) = Container::parse(dxil).ok().and_then(|c| c.part(FourCC::OSG1).map(|p| p.data.to_vec())) else {
        return Vec::new();
    };
    let word = |at: usize| d.get(at..at + 4).map_or(0, |w| u32::from_le_bytes(w.try_into().unwrap()));
    // System value 64 is a render target.
    (0..word(0) as usize).map(|i| 8 + 32 * i).filter(|&row| word(row + 12) == 64).map(|row| (word(row + 8), word(row + 16))).collect()
}

impl Offline {
    pub fn new() -> Result<Offline, String> {
        Ok(Offline { title: Title::new(Metal::new()?, Arc::new(super::memory::PoolTable::default())) })
    }

    /// The refusal the shader path just counted for `name`, as the run's log gives it.
    fn refusal(&self, name: &str) -> String {
        let key = format!("{name}: ");
        self.title.shaders.refused.iter().rev().find_map(|r| r.0.strip_prefix(&key)).unwrap_or("no pipeline").to_owned()
    }

    /// The render pipeline of a described vertex and pixel container (`rect`: drawn as a
    /// rectangle list), or why there is none. `Ok` says what was made.
    pub fn graphics(&mut self, vertex: &[u8], pixel: Option<&[u8]>, rect: bool, name: &str) -> Result<String, String> {
        let vs = if rect { shaders::translate_rect(vertex) } else { shaders::translate(vertex) }.map_err(|e| format!("translation: {e}"))?;
        let ps = pixel.map(shaders::translate).transpose().map_err(|e| format!("translation: {e}"))?;
        // An input layout of the shader's own elements, once each (a rectangle variant declares
        // them for three corners), as wide as it declares them.
        let mut names = Vec::new();
        let mut layout = Vec::new();
        for (element, index, kind, channels) in elements(&vs.dxil) {
            if index >> RECT_CORNER_SHIFT != 0 {
                continue;
            }
            let format = match (kind, channels) {
                (3, 1) => 41, (3, 2) => 16, (3, 3) => 6, (3, _) => 2,
                (2, 1) => 43, (2, 2) => 18, (2, 3) => 8, (2, _) => 4,
                (_, 1) => 42, (_, 2) => 17, (_, 3) => 7, (_, _) => 3,
            };
            names.push(std::ffi::CString::new(element).map_err(|e| e.to_string())?);
            layout.push((index, format));
        }
        let mut records = Vec::new();
        for (name, (index, format)) in names.iter().zip(&layout) {
            let mut e = [0u8; 32];
            e[..8].copy_from_slice(&(name.as_ptr() as u64).to_le_bytes());
            // The semantic index, the format, slot 0, appended, per vertex.
            for (k, w) in [*index, *format, 0, u32::MAX, 0, 0].into_iter().enumerate() {
                e[8 + 4 * k..12 + 4 * k].copy_from_slice(&w.to_le_bytes());
            }
            records.extend(e);
        }
        let mut subobject = 12u32.to_le_bytes().to_vec();
        subobject.extend([0; 4]);
        subobject.extend((records.as_ptr() as u64).to_le_bytes());
        subobject.extend((layout.len() as u32).to_le_bytes());
        subobject.extend([0; 4]);
        let mut g = Graphics::default();
        // SAFETY: a whole input layout subobject over the live records and names above.
        unsafe { g.read(12, subobject.as_ptr()) };
        let outputs = ps.as_ref().map_or_else(Vec::new, |p| target_types(&p.dxil));
        let targets = outputs.iter().map(|t| t.0 + 1).max().unwrap_or(0).min(8);
        let color = (0..targets)
            .map(|n| match outputs.iter().find(|t| t.0 == n) {
                Some((_, 1)) => Format::Rgba32Uint,
                _ => Format::Rgba8Unorm,
            })
            .collect();
        // Always a depth target: a pixel shader may write depth.
        let g = g.with_targets(color, Some(Format::D32Float));
        let memory = |s: &victory_shader::emit::HostShader| if s.memory { " (page table)" } else { "" };
        let what = format!(
            "vertex {} inputs{}{}",
            layout.len(),
            memory(&vs),
            ps.as_ref().map_or(String::new(), |p| format!(", pixel {targets} targets{}", memory(p))),
        );
        match self.title.render_pipe(&vs, ps.as_ref(), &g, name) {
            Pipe::Render(_, binding) => Ok(format!("{what}{}", if binding.is_some_and(|b| b.rect_strip) { ", rectangle strips" } else { "" })),
            _ => Err(self.refusal(name)),
        }
    }

    /// The compute pipeline of a container, or why there is none.
    pub fn compute(&mut self, container: &[u8], name: &str) -> Result<String, String> {
        let cs = shaders::translate(container).map_err(|e| format!("translation: {e}"))?;
        match self.title.compute_pipe(&cs, name) {
            Pipe::Compute(pso) => Ok(format!("threads {:?}", pso.threads)),
            _ => Err(self.refusal(name)),
        }
    }
}
