//! Content-addressed mirrors. Only successful decoding publishes an entry; replay retains the
//! backend allocation and reproduces the host's auxiliary tables and lifetime tracking too.
use super::*;

/// Native descriptor snapshots, kept here so the platform-neutral backend API need not change.
/// A snapshot owns everything needed to replay the entry after its original slot is overwritten.
pub trait DescriptorBackend: GraphicsBackend {
    type Descriptor;
    fn capture_descriptor(&self, heap: &Self::Heap, index: usize) -> Self::Descriptor;
    fn write_descriptor(&self, heap: &Self::Heap, index: usize, entry: &Self::Descriptor);
}

#[cfg(target_os = "macos")]
impl DescriptorBackend for victory_metal::metal::Metal {
    type Descriptor = victory_metal::gfx::Descriptor;
    fn capture_descriptor(&self, heap: &Self::Heap, index: usize) -> Self::Descriptor {
        heap.capture_descriptor(index)
    }
    fn write_descriptor(&self, heap: &Self::Heap, index: usize, entry: &Self::Descriptor) {
        self.write_descriptor(heap, index, entry);
    }
}

pub(super) const CACHE_LIMIT: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) enum Miss { Cold, Evicted, Stale, Uncacheable, Disabled }

#[cfg(target_os = "macos")]
impl Miss {
    pub(super) fn metric(self) -> Metric {
        match self {
            Miss::Cold => Metric::MissCold, Miss::Evicted => Metric::MissEvicted,
            Miss::Stale => Metric::MissStale, Miss::Uncacheable => Metric::MissUncacheable,
            Miss::Disabled => Metric::MissDisabled,
        }
    }
}

/// Count on every exit, including failed resolution; a successful direct buffer can select
/// bypass after resolution reveals that it needed neither a span nor an alias validation.
pub(super) struct MissCounter {
    pub reason: Cell<Miss>,
    #[cfg(target_os = "macos")]
    kind: Option<Metric>,
}
impl Drop for MissCounter {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        if let Some(kind) = self.kind {
            frame_time::count(self.reason.get().metric(), 1);
            frame_time::count(kind, 1);
        }
    }
}

pub(super) struct CachedMirror<E> {
    stamp: Rc<MirrorStamp>,
    kind: HeapKind,
    entry: E,
    typed: Option<(E, bool)>,
    pending: (bool, bool),
    span: Option<SpanRef>,
    title: Option<super::super::list::TitleView>,
    image: Option<usize>,
    twin_uav: Option<usize>,
    twin_srv: Option<usize>,
    depth_srv: Option<usize>,
    stride: u32,
    /// The view-stride table's minimum LOD clamp word (`descriptors::min_lod_word`).
    min_lod: u32,
    address: [u32; 4],
}

impl<E> CachedMirror<E> {
    pub(super) fn current_memory(&self, memory: u64) -> bool {
        self.stamp.fixed_null || self.stamp.memory == memory
    }
}

impl<B: DescriptorBackend> NativeHost<B> {
    /// Eviction/release must let cached allocations go too, even if the record is never read
    /// again. Multi-resource stamps use the global epoch and cannot outlive any release.
    pub(super) fn forget_cached_resource(&self, key: usize) {
        self.mirror_cache.borrow_mut().retain(|record, c| {
            let keep = c.stamp.fixed_null || c.stamp.resources.is_some_and(|(r, _)| r != key);
            self.note_cache_removal(*record, keep);
            keep
        });
    }

    pub(super) fn cached_mirror(&self, heap: &NativeHeap<B>, index: usize, record: [u8; 12], memory: u64, epoch: u64) -> Result<(), Miss> {
        if !self.cache_mirrors { return Err(Miss::Disabled); }
        let mut cache = self.mirror_cache.borrow_mut();
        let Some(c) = cache.get_mut(&record) else {
            return Err(self.mirror_misses.borrow().get(&record).copied().unwrap_or(Miss::Cold));
        };
        if c.kind != heap.kind || !self.stamp_current(&c.stamp, memory, epoch)
            || c.span.is_some_and(|(key, version)| self.spans.borrow().get(&key).is_none_or(|s| s.version != version))
        {
            self.note_cache_removal(record, false);
            cache.remove(&record);
            return Err(Miss::Stale);
        }
        self.observe_cache_record(record);
        self.backend.write_descriptor(&heap.heap, index, &c.entry);
        if let Some(pages) = &*heap.addresses.borrow() {
            pages.write_at(4 * index, &c.address);
        }
        if let Some(pages) = &*heap.strides.borrow() {
            Self::write_stride_words(pages, index, c.stride, c.min_lod);
        }
        if let Some(span) = c.span { heap.spanned.borrow_mut().insert(index, span); }
        if let Some(title) = c.title {
            heap.title_views.borrow_mut().insert(index, title);
        } else {
            let mut title = heap.title_views.borrow_mut();
            if !title.is_empty() { title.remove(&index); }
        }
        if let Some(key) = c.twin_uav { heap.twin_uavs.borrow_mut().insert(index, key); }
        if let Some(key) = c.twin_srv { heap.twin_srvs.borrow_mut().insert(index, key); }
        if let Some(key) = c.depth_srv { heap.depth_srvs.borrow_mut().insert(index, key); }
        heap.shows(index, c.image);
        heap.typed_pending.borrow_mut().set(index, c.pending.0, c.pending.1);
        heap.mark_mirrored(index, (*c.stamp).clone());
        #[cfg(target_os = "macos")]
        if self.trace_mirrors { frame_time::count(Metric::MirrorCached, 1); }
        Ok(())
    }

    pub(super) fn cache_mirror(&self, heap: &NativeHeap<B>, index: usize, stamp: &Rc<MirrorStamp>, view: &crate::descriptor::View) {
        if !self.cache_mirrors { return; }
        self.observe_cache_record(stamp.record);
        let pending = heap.typed_pending.borrow();
        let c = CachedMirror {
            stamp: stamp.clone(),
            kind: heap.kind,
            entry: self.backend.capture_descriptor(&heap.heap, index),
            typed: None,
            pending: (pending.contains(index), pending.is_explicit(index)),
            span: heap.spanned.borrow().get(&index).copied(),
            title: heap.title_views.borrow().get(&index).copied(),
            image: heap.shown(index),
            twin_uav: heap.twin_uavs.borrow().get(&index).copied(),
            twin_srv: heap.twin_srvs.borrow().get(&index).copied(),
            depth_srv: heap.depth_srvs.borrow().get(&index).copied(),
            stride: self.stride_word(Some(view)),
            min_lod: super::super::descriptors::min_lod_word(view, &self.resources.borrow()),
            address: self.address_words(Some(view)),
        };
        let mut cache = self.mirror_cache.borrow_mut();
        // A fixed bound on retained allocations as well as CPU metadata. Resetting on overflow
        // avoids a second hash lookup / LRU mutation on every hit.
        if cache.len() >= self.mirror_cache_limit {
            #[cfg(target_os = "macos")]
            if self.trace_mirrors {
                frame_time::count(Metric::CacheEvictions, cache.len() as u64);
                self.mirror_misses.borrow_mut().extend(cache.keys().map(|r| (*r, Miss::Evicted)));
            }
            cache.clear();
        }
        self.mirror_misses.borrow_mut().remove(&stamp.record);
        cache.insert(stamp.record, c);
        #[cfg(target_os = "macos")]
        if self.trace_mirrors { frame_time::maximum(Metric::CacheEntriesMax, cache.len() as u64); }
    }

    fn observe_cache_record(&self, record: [u8; 12]) {
        #[cfg(target_os = "macos")]
        if self.trace_mirrors {
            let frame = crate::frame::current();
            let mut working = self.cache_working.borrow_mut();
            if working.0 != frame { working.0 = frame; working.1.clear(); }
            if working.1.insert(record) { frame_time::count(Metric::CacheWorking, 1); }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = record;
    }

    pub(super) fn note_cache_removal(&self, record: [u8; 12], keep: bool) {
        #[cfg(target_os = "macos")]
        if !keep && self.trace_mirrors { self.mirror_misses.borrow_mut().insert(record, Miss::Stale); }
        #[cfg(not(target_os = "macos"))]
        let _ = (record, keep);
    }

    pub(super) fn miss_counter(&self, reason: Miss, decoded: Option<&Decoded>, view: Option<&crate::descriptor::View>, nothing: Nothing) -> MissCounter {
        #[cfg(target_os = "macos")]
        let kind = self.trace_mirrors.then(|| match decoded {
            Some(Decoded::Buffer { addr: 0, .. }) => Metric::MissNull,
            Some(Decoded::Buffer { .. }) if matches!(view, Some(crate::descriptor::View::Constant(..))) => Metric::MissConstant,
            Some(Decoded::Buffer { .. }) if view.is_some_and(|v| typed_buffer(v, &self.resources.borrow()).is_some()) => Metric::MissTyped,
            Some(Decoded::Buffer { .. }) => Metric::MissBuffer,
            Some(Decoded::Image { .. }) => Metric::MissImage,
            Some(Decoded::Sampler { .. }) => Metric::MissSampler,
            None => nothing.metric(),
        });
        #[cfg(not(target_os = "macos"))]
        let _ = (view, nothing);
        MissCounter {
            reason: Cell::new(if reason == Miss::Cold && decoded.is_none() { Miss::Uncacheable } else { reason }),
            #[cfg(target_os = "macos")]
            kind,
        }
    }

    pub(super) fn cached_typed(&self, heap: &NativeHeap<B>, index: usize, record: [u8; 12], memory: u64, epoch: u64) -> bool {
        if !self.cache_mirrors { return false; }
        let mut cache = self.mirror_cache.borrow_mut();
        let Some(c) = cache.get_mut(&record) else { return false };
        if !self.stamp_current(&c.stamp, memory, epoch) { return false; }
        let Some((entry, typed)) = &c.typed else { return false };
        self.backend.write_descriptor(&heap.heap, index, entry);
        self.typed_views.set(self.typed_views.get() + u64::from(*typed));
        #[cfg(target_os = "macos")]
        if self.trace_mirrors { frame_time::count(Metric::MirrorCached, 1); }
        true
    }

    pub(super) fn cache_typed(&self, heap: &NativeHeap<B>, index: usize, record: [u8; 12], typed: bool) {
        if self.cache_mirrors && let Some(c) = self.mirror_cache.borrow_mut().get_mut(&record) {
            c.typed = Some((self.backend.capture_descriptor(&heap.heap, index), typed));
        }
    }
}
