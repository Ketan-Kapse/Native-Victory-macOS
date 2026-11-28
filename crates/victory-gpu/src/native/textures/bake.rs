//! Textures a draw takes from the resource heap at indices in its constants (`VICTORY_BAKE_SOURCES=1`,
//! docs/outfit-notes.md). The title's decoration bakes (`PS_BakeDecorations_*`,
//! `PS_DecorationBlend_*`) draw once into a corner of a large target, picking each source texture
//! by a heap index in a constant word, through no table. The host cannot tell which entries such a
//! shader reads ([`crate::native::list::Binding::heap`]), so nothing makes sure they show a texture:
//! a placed copy given up as covered, or a texture given up under the budget, leaves its entries
//! null, and a draw that reads one bakes black into a texture the title keeps for the rest of the
//! run (a face, a garment).
//!
//! Guess (marked, not console behaviour): every root-constant word and every word of the first
//! [`CBV_BYTES`] of each root constant buffer is a candidate heap index. Words that name a texture
//! record in the bound resource heap are made sure of before the draw is recorded: a texture
//! missing on the host (covered, evicted, never made) is made again and its entry written again;
//! its contents are uploaded at the submission that runs the draw, before the GPU does. Such
//! textures count as read ([`Title::heap_read`]) for [`super::HEAP_READ_KEEP`] submissions: neither
//! covered-clear nor the budget gives them up meanwhile. A word that only looks like an index
//! makes one texture more resident than needed; an index the shader computes otherwise (from a
//! buffer, a texture) is not seen.

use crate::native::host::DescriptorBackend;
use crate::native::list::{NativeHeap, Root};
use crate::native::title::Title;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use victory_backend::resources::{GraphicsBackend, PresentBackend};

/// Bytes of a root constant buffer read for candidate indices.
const CBV_BYTES: usize = 4096;

/// Counters: heap-indexed draws looked at, texture entries their words named, of those null at
/// the draw (no host texture: covered, evicted or not yet made), not yet uploaded at the draw (made
/// now or earlier but no upload yet: the submission uploads them first), made again, and still
/// null after the attempt.
pub(crate) static BAKE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Nanoseconds spent in [`Title::heap_draw_sources`] (and the reads of its constants, counted by
/// the caller).
pub(crate) static BAKE_NS: AtomicU64 = AtomicU64::new(0);

/// What [`Title::heap_draw_sources`] found for one draw.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Sources {
    pub textures: usize,
    pub null: usize,
    pub not_uploaded: usize,
    pub made: usize,
    pub still_null: usize,
}

/// The candidate heap indices in the words a draw's root parameters give it (sorted, once each):
/// below `count`, the heap's size. `read` fills a buffer from title memory at an address (false:
/// not readable).
pub(crate) fn candidates<B: GraphicsBackend>(root: &[Option<Root<B>>], count: usize, read: impl Fn(u64, &mut [u8]) -> bool) -> Vec<usize> {
    let mut out = Vec::new();
    let mut take = |w: u32| {
        if (w as usize) < count {
            out.push(w as usize);
        }
    };
    let mut buf = Vec::new();
    for r in root.iter().flatten() {
        match r {
            Root::Constants(c) => c.iter().copied().for_each(&mut take),
            Root::View(a) if *a != 0 => {
                buf.resize(CBV_BYTES, 0);
                // A buffer near the end of its mapping: the first 256 bytes then.
                let n = if read(*a, &mut buf) {
                    CBV_BYTES
                } else if read(*a, &mut buf[..256]) {
                    256
                } else {
                    0
                };
                buf[..n].as_chunks::<4>().0.iter().for_each(|c| take(u32::from_le_bytes(*c)));
            }
            _ => {}
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

impl<B: PresentBackend + DescriptorBackend> Title<B> {
    /// Before a draw whose shaders take views from `heap` at indices they compute
    /// ([`candidates`] of its root parameters): each texture an index names is on the host and
    /// shown by its entry, counted as read now. Counted in [`BAKE`]; a null or not yet uploaded
    /// source is logged (`pipeline` names the draw's shaders).
    pub(crate) fn heap_draw_sources(&mut self, heap: &Rc<NativeHeap<B>>, indices: &[usize], pipeline: &str) -> Sources {
        let mut found = Sources::default();
        BAKE[0].fetch_add(1, Relaxed);
        for &index in indices {
            let Some((key, placed)) = self.host.entry_texture(heap, index) else { continue };
            found.textures += 1;
            let shown = heap.shows_at(index) == Some(key) && self.host.texture_at(key).is_some();
            if !shown {
                found.null += 1;
                let covered = self.host.is_covered(key);
                let evicted = self.host.is_evicted(key);
                self.make_for_heap_read(key, placed);
                let now = self.host.mirror_index(heap, index) && heap.shows_at(index) == Some(key);
                found.made += usize::from(now);
                found.still_null += usize::from(!now);
                let n = BAKE[2].fetch_add(1, Relaxed);
                if n < 64 || n.is_power_of_two() {
                    tracing::warn!(pipeline, index, key = format!("{key:#x}"), placed = placed.is_some(), covered, evicted, shown_now = now, null_sources = n + 1, "a heap-indexed draw names a texture its entry did not show: made again before the draw");
                }
            }
            let owner = self.owner_of(key);
            let owner = self.sharing_object(owner).unwrap_or(owner);
            self.heap_read.insert(owner, self.sync_serial);
            self.heap_read.insert(key, self.sync_serial);
            let serial = self.budget.serial;
            if let Some(t) = self.textures.get_mut(&owner) {
                t.used = serial;
                if t.uploads == 0 && !t.info.render_target() && !t.gpu_written {
                    found.not_uploaded += 1;
                    let n = BAKE[3].fetch_add(1, Relaxed);
                    if n < 64 || n.is_power_of_two() {
                        tracing::warn!(pipeline, index, key = format!("{key:#x}"), not_uploaded = n + 1, "a heap-indexed draw names a texture not uploaded yet: its submission uploads it before the draw runs");
                    }
                }
            }
        }
        BAKE[1].fetch_add(found.textures as u64, Relaxed);
        BAKE[4].fetch_add(found.made as u64, Relaxed);
        BAKE[5].fetch_add(found.still_null as u64, Relaxed);
        let draws = BAKE[0].load(Relaxed);
        if draws.is_power_of_two() && draws >= 64 {
            let [d, t, n, u, m, s] = std::array::from_fn::<u64, 6, _>(|i| BAKE[i].load(Relaxed));
            let ms = BAKE_NS.load(Relaxed) / 1_000_000;
            tracing::info!(draws = d, textures = t, null = n, not_uploaded = u, made_again = m, still_null = s, ms, "heap-indexed draws and the textures their constants name (VICTORY_BAKE_SOURCES)");
        }
        found
    }

    /// Texture `key` (a placed description at `placed`'s address, or a resource object) on the
    /// host again for a heap-indexed read: a covered copy is placed again (the title still reads
    /// it, so the placement that covered it does not cover it again), an evicted or pending one
    /// made.
    fn make_for_heap_read(&mut self, key: usize, placed: Option<(u64, [u32; 14])>) {
        if self.host.is_covered(key) {
            self.host.placed_again(key);
            self.covered_history.revived(key);
        }
        match placed {
            Some((address, res)) if !self.host.knows(key) || self.host.texture_at(key).is_none() => {
                let Some(info) = crate::resource::placed_info(address, &res) else { return };
                let Some((pool, _)) = self.host.memory().resolve(address, 1) else { return };
                // Released with its pool like any placement, unless another owns the address now
                // (then it goes as a covered copy once no heap-indexed draw names it).
                let owner = *self.placed.entry(address).or_insert((key, pool.id));
                self.placed_reach = self.placed_reach.max(info.size as u64);
                if owner.0 != key {
                    self.covered.insert(key, owner.0);
                }
                if !self.alias_placed(address, key, &info) {
                    if !self.textures.contains_key(&key) && !self.copies_of.contains_key(&key) {
                        self.texture_created(key, info);
                    }
                    self.realize(key);
                }
            }
            Some(_) => {}
            None => {
                self.realize(key);
            }
        }
    }
}
