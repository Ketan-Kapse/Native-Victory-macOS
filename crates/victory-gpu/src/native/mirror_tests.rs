//! CPU-only host regressions: mirror stamps must follow precisely the resources and mappings
//! they resolved, without repeating backend writes for an unchanged entry.
use super::super::memory::{PoolInfo, PoolObserver};
use super::*;
use victory_backend::backend::{ComputeBackend, RootLayout, Submission};

#[derive(Default)]
pub(in crate::native) struct Recording {
    textures: std::cell::Cell<usize>,
    writes: std::cell::Cell<u64>,
    replays: std::cell::Cell<u64>,
    views: RefCell<Vec<(usize, ViewDesc)>>,
    details: RefCell<HashMap<(usize, usize), EntryDetail>>,
    capture_details: Cell<bool>,
    wraps: RefCell<Vec<(usize, usize)>>,
    spans: RefCell<Vec<Vec<(usize, usize, usize)>>>,
    /// Work tracking, when set: the mark of the work submitted so far, and the mark below which
    /// all work has finished.
    work: Cell<Option<(u64, u64)>>,
}
pub(in crate::native) struct RecordedBuffer {
    _memory: Option<Box<[u32]>>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::native) enum EntryDetail {
    Buffer(Option<(Format, bool)>),
    Image(Option<ViewDesc>, u32),
    Sampler(SamplerDesc, u32),
    Clear,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::native) struct RecordedEntry(usize, u32, Option<Rc<EntryDetail>>);
impl PartialEq<(usize, u32)> for RecordedEntry {
    fn eq(&self, other: &(usize, u32)) -> bool { (self.0, self.1) == *other }
}
type Heap = RefCell<HashMap<usize, (usize, u32)>>;
impl Recording {
    fn detail(&self, heap: &Heap, index: usize, detail: EntryDetail) {
        if self.capture_details.get() { self.details.borrow_mut().insert((heap as *const _ as usize, index), detail); }
    }
}

impl DescriptorBackend for Recording {
    type Descriptor = RecordedEntry;
    fn capture_descriptor(&self, heap: &Heap, index: usize) -> Self::Descriptor {
        let (a, b) = heap.borrow()[&index];
        let detail = self.capture_details.get().then(|| Rc::new(self.details.borrow()[&(heap as *const _ as usize, index)]));
        RecordedEntry(a, b, detail)
    }
    fn write_descriptor(&self, heap: &Heap, index: usize, entry: &Self::Descriptor) {
        self.writes.set(self.writes.get() + 1);
        self.replays.set(self.replays.get() + 1);
        heap.borrow_mut().insert(index, (entry.0, entry.1));
        if let Some(detail) = &entry.2 { self.detail(heap, index, **detail); }
    }
}

impl ComputeBackend for Recording {
    type Queue = ();
    type Fence = std::cell::Cell<u64>;
    type Buffer = RecordedBuffer;
    type Pipeline = ();
    type Heap = Heap;
    fn create_queue(&self) -> Result<(), String> {
        Ok(())
    }
    fn create_fence(&self, v: u64) -> Result<Self::Fence, String> {
        Ok(v.into())
    }
    fn wrap_title_memory(&self, p: *mut u8, n: usize) -> Result<RecordedBuffer, String> {
        self.wraps.borrow_mut().push((p as usize, n));
        Ok(RecordedBuffer { _memory: None })
    }
    fn wrap_title_span(&self, n: usize, pieces: &[(usize, *mut u8, usize)]) -> Result<(RecordedBuffer, *mut u8), String> {
        self.spans.borrow_mut().push(pieces.iter().map(|&(at, p, len)| (at, p as usize, len)).collect());
        self.host_buffer(n)
    }
    fn work_mark(&self) -> Option<u64> {
        self.work.get().map(|(mark, _)| mark)
    }
    fn work_done(&self, mark: u64) -> bool {
        self.work.get().is_none_or(|(_, done)| mark <= done)
    }
    fn host_buffer(&self, len: usize) -> Result<(RecordedBuffer, *mut u8), String> {
        let mut memory = vec![0u32; len.div_ceil(4)].into_boxed_slice();
        let ptr = memory.as_mut_ptr().cast();
        Ok((
            RecordedBuffer {
                _memory: Some(memory),
            },
            ptr,
        ))
    }
    fn gpu_address(&self, _: &RecordedBuffer, p: *const u8) -> u64 {
        p as u64
    }
    fn create_compute_pipeline(&self, _: &[u8]) -> Result<(), String> {
        unreachable!()
    }
    fn create_compute_pipeline_with(&self, _: &[u8], _: &RootLayout) -> Result<(), String> {
        unreachable!()
    }
    fn submit(&self, _: &(), _: &Submission<'_, Self>) -> Result<(), String> {
        unreachable!()
    }
    fn signal(&self, f: &Self::Fence, v: u64) {
        f.set(v);
    }
    fn completed(&self, f: &Self::Fence) -> u64 {
        f.get()
    }
    fn wait(&self, f: &Self::Fence, v: u64, _: u64) -> bool {
        f.get() >= v
    }
}

impl ResourceBackend for Recording {
    type Texture = usize;
    type Sampler = SamplerDesc;
    fn create_heap(&self, _: HeapKind, _: usize) -> Result<Heap, String> {
        Ok(Heap::default())
    }
    fn heap_address(&self, h: &Heap, i: usize) -> u64 {
        h as *const Heap as u64 + i as u64 * 32
    }
    fn write_buffer(&self, h: &Heap, i: usize, _: &RecordedBuffer, p: *const u8, n: u32) {
        self.writes.set(self.writes.get() + 1);
        h.borrow_mut().insert(i, (p as usize, n));
        self.detail(h, i, EntryDetail::Buffer(None));
    }
    fn write_typed_buffer(&self, h: &Heap, i: usize, b: &RecordedBuffer, p: *const u8, n: u32, f: Format, writable: bool) -> bool {
        self.write_buffer(h, i, b, p, n);
        self.detail(h, i, EntryDetail::Buffer(Some((f, writable))));
        true
    }
    fn write_texture(&self, h: &Heap, i: usize, t: &usize, lod: f32) {
        self.writes.set(self.writes.get() + 1);
        let (source, view) = if *t & (1usize << 63) != 0 {
            let (source, view) = self.views.borrow()[(*t & !(1usize << 63)) - 1];
            (source, Some(view))
        } else { (*t, None) };
        h.borrow_mut().insert(i, (source, u32::MAX));
        self.detail(h, i, EntryDetail::Image(view, lod.to_bits()));
    }
    fn write_sampler(&self, h: &Heap, i: usize, s: &SamplerDesc, bias: f32) {
        self.writes.set(self.writes.get() + 1);
        h.borrow_mut().insert(i, (0, u32::MAX - 1));
        self.detail(h, i, EntryDetail::Sampler(*s, bias.to_bits()));
    }
    fn clear_entry(&self, h: &Heap, i: usize) {
        self.writes.set(self.writes.get() + 1);
        h.borrow_mut().insert(i, (0, 0));
        self.detail(h, i, EntryDetail::Clear);
    }
    fn create_texture(&self, _: &TextureDesc) -> Result<usize, String> {
        self.textures.set(self.textures.get() + 1);
        Ok(self.textures.get())
    }
    fn create_texture_view(&self, t: &usize, view: &ViewDesc) -> Result<usize, String> {
        let mut views = self.views.borrow_mut();
        let key = (*t, *view);
        let index = views.iter().position(|v| *v == key).unwrap_or_else(|| { views.push(key); views.len() - 1 });
        Ok((1usize << 63) | (index + 1))
    }
    fn create_texture_over_memory(
        &self,
        _: &RecordedBuffer,
        _: *const u8,
        _: &TextureDesc,
        _: usize,
    ) -> Result<usize, String> {
        unreachable!()
    }
    fn linear_alignment(&self, _: Format) -> usize {
        unreachable!()
    }
    fn upload_texture(&self, _: &usize, _: usize, _: usize, _: &[u8], _: usize, _: usize) {}
    fn read_texture(&self, _: &usize, _: usize, _: usize, _: &mut [u8], _: usize, _: usize) {
        unreachable!()
    }
    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerDesc, String> {
        Ok(*desc)
    }
}

impl GraphicsBackend for Recording {
    type RenderPipeline = ();
    fn create_render_pipeline(&self, _: &RenderPipelineDesc<'_>) -> Result<(), String> {
        unreachable!()
    }
    fn submit_passes(&self, _: &(), _: &PassSubmission<'_, Self>) -> Result<(), SubmitFailure> {
        Ok(())
    }
}

impl PresentBackend for Recording {
    type Surface = ();
    fn create_surface(&self, _: *mut std::ffi::c_void) -> Result<(), String> { Ok(()) }
    fn present(&self, _: &(), _: &(), _: &[&usize]) -> Result<(), String> { Ok(()) }
}

/// Fresh descriptor slots, repeated view identities: the world's ring-heap workload.
#[test]
#[ignore = "CPU benchmark: run --release --ignored --nocapture"]
fn fresh_ring_mirror_benchmark() {
    use std::hint::black_box;
    use std::time::Instant;
    use Recording as Capture;
    const N: usize = 100_000;
    const M: usize = 2_000;
    let mut samples = Vec::new();
    for _ in 0..7 {
        let h = NativeHost::new(Capture::default());
        let mut bytes = vec![0u32; 32768];
        let pool = PoolInfo { id: 1, base: bytes.as_mut_ptr() as usize, len: bytes.len() * 4, memory_entry: None };
        h.memory.mapped(0x7000_0000, pool.len, &pool, 0);
        let mut records = vec![0u32; N * 8];
        let start = records.as_mut_ptr() as u64;
        let heap = h.heap(HeapKind::Resource, start, N).unwrap();
        for i in 0..N {
            crate::descriptor::record(start + (i * 32) as u64,
                crate::descriptor::View::Constant(0x7000_0000 + ((i % M) * 32) as u64, 32));
        }
        let before = Instant::now();
        for at in (0..N).step_by(NativeHost::<Capture>::TABLE_CHECK) {
            h.remirror_tables(black_box(&[Some(Root::Table(heap.clone(), at))]), None);
        }
        samples.push(before.elapsed().as_nanos() as f64 / N as f64);
        assert_eq!(h.backend.writes.get(), N as u64);
        for i in 0..N {
            assert_eq!(heap.heap.borrow()[&i], (pool.base + (i % M) * 32, 32));
        }
    }
    samples.sort_by(f64::total_cmp);
    println!("fresh ring N={N} M={M}: median {:.1} ns/mirror; samples {samples:.1?}", samples[3]);
}

/// Stable bound tables between draws, with a first scan at each submission boundary.
#[test]
#[ignore = "CPU benchmark: run --release --ignored --nocapture"]
fn repeated_table_benchmark() {
    use std::{hint::black_box, time::Instant};
    let h = NativeHost::new(Recording::default());
    let records = [0u32; 128];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 16).unwrap();
    for i in 0..16 { crate::descriptor::record(start + i * 32, crate::descriptor::View::Constant(0, 256)); }
    let roots = [Some(Root::Table(heap, 0))];
    let _tables = h.clean_tables.begin(h.table_clean);
    let mut samples = Vec::new();
    for _ in 0..7 {
        let now = Instant::now();
        for _ in 0..100_000 { h.remirror_tables(black_box(&roots), None); }
        samples.push(now.elapsed().as_nanos() as f64 / 100_000.0);
    }
    samples.sort_by(f64::total_cmp);
    println!("repeated table 16 entries: median {:.1} ns/call; samples {samples:.1?}", samples[3]);
}

/// Same table scan for all passes; registration and validation are outside the timed region.
#[test]
#[ignore = "CPU benchmark: run --release --ignored --nocapture"]
fn dense_table_workloads_benchmark() {
    use std::{hint::black_box, time::Instant};
    const TABLES: usize = 2048;
    // What the recording backend's own entry write (a hash map insert) costs in these numbers.
    let mut alone = Vec::new();
    for _ in 0..7 {
        let (backend, heap, buffer) = (Recording::default(), Heap::default(), RecordedBuffer { _memory: None });
        let now = Instant::now();
        for i in 0..TABLES * 16 {
            backend.write_buffer(black_box(&heap), i, &buffer, (0x3000_0000 + i * 32) as *const u8, 32);
        }
        alone.push(now.elapsed().as_nanos() as f64 / (TABLES * 16) as f64);
    }
    alone.sort_by(f64::total_cmp);
    println!("recording backend write alone: {:.1} ns/entry", alone[3]);
    for (name, width, fresh, constant) in [("distinct raw", 16, 16, false), ("distinct CBV", 16, 16, true), ("mixed 12 fresh + 80 clean", 92, 12, true)] {
        let n = TABLES * width;
        let mut records = vec![0u32; n * 8];
        let start = records.as_mut_ptr() as u64;
        for i in 0..n {
            let address = 0x7000_0000 + (i * 32) as u64;
            let view = if constant { crate::descriptor::View::Constant(address, 32) } else {
                let mut res = [0; 14]; res[0] = 1;
                let mut d = [0; 16];
                d[1] = 1; d[6] = 8; d[8] = 1; d[10] = address as u32; d[11] = (address >> 32) as u32;
                crate::descriptor::View::Placed(false, res, d)
            };
            crate::descriptor::record(start + (i * 32) as u64, view);
        }
        let binding = Binding { table_entries: vec![(0, width)], ..Binding::default() };
        let mut samples = Vec::new();
        for _ in 0..7 {
            let h = NativeHost::new(Recording::default());
            h.memory.mapped(0x7000_0000, n * 32, &PoolInfo { id: 1, base: 0x3000_0000, len: n * 32, memory_entry: None }, 0);
            let heap = h.heap(HeapKind::Resource, start, n).unwrap();
            // Prime just the clean suffix; fresh prefixes have no stamp or backend entry.
            if fresh < width {
                for i in (0..n).filter(|i| i % width >= fresh) { h.mirror(start + (i * 32) as u64).unwrap(); }
            }
            let now = Instant::now();
            for at in (0..n).step_by(width) {
                h.remirror_tables(black_box(&[Some(Root::Table(heap.clone(), at))]), Some(&binding));
            }
            samples.push(now.elapsed().as_nanos() as f64 / (TABLES * fresh) as f64);
            assert_eq!(h.backend.writes.get(), n as u64);
            for i in 0..n { assert_eq!(heap.heap.borrow()[&i], (0x3000_0000 + i * 32, 32)); }
        }
        samples.sort_by(f64::total_cmp);
        println!("{name}: {:.1} ns/fresh entry ({:.1} ns/table); samples {samples:.1?}", samples[3], samples[3] * fresh as f64);
    }
}

/// `VICTORY_DENSE_ENTRIES`: a host with the dense state, the quick mirrors, the pool list and
/// plain record loads against the reference host, over the same title records: after every
/// step the same backend entries and writes, stamps, pending retries and per-entry notes.
#[test]
fn dense_quick_mirrors_match_the_reference_host_through_random_writes() {
    use crate::descriptor::View;
    const N: usize = 700; // three pages of entry state, the last one short
    const A: u64 = 0x7000_0000;
    const B: u64 = 0x7100_0000;
    const BUFFER: usize = 0x5000_0010;
    const TEXTURE: usize = 0x5000_0020;
    let mut records = vec![0u32; N * 8];
    let start = records.as_mut_ptr() as u64;
    let pools = [
        PoolInfo { id: 1, base: 0x3000_0000, len: 65536, memory_entry: None },
        PoolInfo { id: 2, base: 0x3100_0000, len: 65536, memory_entry: None },
        PoolInfo { id: 3, base: 0x3200_0000, len: 65536, memory_entry: None },
    ];
    let hosts: Vec<_> = [true, false].into_iter().map(|dense| {
        let mut h = NativeHost::new(Recording::default());
        (h.dense_entries, h.record_loads, h.fast_buffers, h.cache_mirrors, h.table_clean) = (dense, true, true, true, true);
        let heap = h.heap(HeapKind::Resource, start, N).unwrap();
        h.memory.mapped(A as usize, 65536, &pools[0], 0);
        h.memory.mapped(B as usize, 65536, &pools[1], 0);
        h.register_buffer(BUFFER, A + 0x1000, 4096);
        let texture = h.texture(&TextureDesc::d2(Format::Rgba8Unorm, 8, 8)).unwrap();
        h.register_texture(TEXTURE, &texture);
        (h, heap)
    }).collect();
    assert!(hosts[0].0.records_readable(&hosts[0].1) && !hosts[1].0.records_readable(&hosts[1].1));
    // Everything the host keeps of a heap, with host-made buffers (the null view) by one name.
    let state = |(h, heap): &(NativeHost<Recording>, Rc<NativeHeap<Recording>>)| {
        let entries: Vec<_> = (0..N).map(|i| heap.heap.borrow().get(&i).map(|&(p, n)| {
            let named = n >= u32::MAX - 1 || (0x3000_0000..0x3300_0000).contains(&p) || (0x7000_0000..0x8000_0000).contains(&p);
            (if named { p } else { 1 }, n)
        })).collect();
        let pending = heap.typed_pending.borrow();
        let notes: Vec<_> = (0..N).map(|i| (
            heap.mirrored.borrow().get(&i).cloned(),
            heap.pending_mirror.borrow().get(&i).copied(),
            heap.title_views.borrow().get(&i).copied(),
            heap.spanned.borrow().get(&i).copied(),
            heap.unmapped.borrow().contains(&i),
            (pending.contains(i), pending.is_explicit(i)),
            heap.shown(i),
        )).collect();
        // The stride and address tables once made (the second half of the run).
        let words = |pages: &Option<Rc<Pages<Recording>>>, n: usize| pages.as_ref().map(|pages| {
            // SAFETY: both tables hold at least a dword (four for addresses) an entry of the heap.
            unsafe { std::slice::from_raw_parts(pages.ptr().cast::<u32>(), n) }.to_vec()
        });
        let tables = (words(&heap.strides.borrow(), N), words(&heap.addresses.borrow(), 4 * N));
        (entries, notes, tables, h.backend.writes.get(), h.backend.replays.get(), h.resources_epoch.get())
    };
    let placed = |address: u64, raw: bool| {
        let mut res = [0; 14];
        res[0] = 1;
        let mut d = [0; 16];
        (d[1], d[6], d[10], d[11]) = (1, 8, address as u32, (address >> 32) as u32);
        if raw { d[8] = 1 } else { d[7] = 4 }
        View::Placed(false, res, d)
    };
    let mut seed = 0x243f6a8885a308d3u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as usize
    };
    let mut dead = 0x6000_0000usize;
    for step in 0..1500 {
        // A few records rewritten, in a run or scattered (also across the page boundaries).
        let (first, run) = (next() % N, next() % 3 == 0);
        let written: Vec<usize> = (0..1 + next() % 24).map(|k| if run { (first + k) % N } else { next() % N }).collect();
        for &i in &written {
            let handle = start + (i * 32) as u64;
            let offset = (next() % 128 * 256) as u64;
            match next() % 13 {
                0 | 1 => crate::descriptor::record(handle, View::Constant(A + offset, 256)),
                2 => crate::descriptor::record(handle, View::Constant(B + offset, 64)),
                3 => crate::descriptor::record(handle, View::Constant(0, 256)),
                4 => crate::descriptor::record(handle, View::Constant(0x7f00_0000 + offset, 64)),
                5 => crate::descriptor::record(handle, placed(A + offset, true)),
                6 => crate::descriptor::record(handle, placed(B + offset, false)),
                7 => crate::descriptor::record(handle, View::Shader(BUFFER, None)),
                8 => crate::descriptor::record(handle, View::Shader(TEXTURE, None)),
                9 => crate::descriptor::record(handle, View::Shader(0x5000_0030, None)),
                // Bytes that are no record, an empty record, and a record of a released view.
                10 => records[i * 8..][..3].copy_from_slice(&[0xdead_beef, 1, 2]),
                11 => records[i * 8..][..3].fill(0),
                _ => {
                    dead += 16;
                    crate::descriptor::record(handle, View::Shader(dead, None));
                    crate::descriptor::forget(dead);
                }
            }
        }
        // The title's world changes now and then: a pool remapped or unmapped, a buffer gone
        // and back, a new submission (its clean marks and readability scope).
        let change = next() % 40;
        let (table, width, device) = (next() % N, 1 + next() % 92, next() % 3 == 0);
        for (h, heap) in &hosts {
            match change {
                0 => h.memory.unmapped(B as usize, 65536),
                1 => h.memory.mapped(B as usize, 65536, &pools[1 + step % 2], 0),
                2 => h.forget_resource(BUFFER),
                3 => h.register_buffer(BUFFER, A + 0x1000 + (step as u64 % 4) * 256, 4096),
                _ => {}
            }
            if step == 700 {
                h.view_strides(heap).unwrap();
                h.view_addresses(heap).unwrap();
            }
            h.prep_scope.set(h.prep_scope.get() + 1);
            let _scope = h.clean_tables.begin(step % 5 != 0);
            if device {
                for &i in &written {
                    let _ = h.mirror(start + (i * 32) as u64);
                }
            }
            let binding = Binding { table_entries: vec![(0, width)], ..Binding::default() };
            let roots = [Some(Root::Table(heap.clone(), table))];
            for _ in 0..2 {
                h.remirror_tables(&roots, Some(&binding));
            }
        }
        assert!(state(&hosts[0]) == state(&hosts[1]), "step {step}: the dense host differs from the reference");
    }
    let (quick, reference) = (hosts[0].0.quick_mirrored.get(), hosts[1].0.quick_mirrored.get());
    assert!(quick > 5_000 && reference == 0, "the quick path must carry the buffer and no-view records: {quick}, {reference}");
    assert!(hosts[0].0.backend.writes.get() > 5_000);
}

/// Plain record loads only while the heap's memory is known readable: asked of the kernel once
/// a preparation for title memory, never after the title released the heap.
#[test]
fn record_loads_follow_the_heaps_memory() {
    use super::super::list::Records;
    unsafe extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
        fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut u8, len: usize) -> i32;
    }
    const LEN: usize = 2 << 14; // two 16 KiB pages
    // SAFETY: an anonymous private mapping (PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON).
    let memory = unsafe { mmap(std::ptr::null_mut(), LEN, 3, 0x1002, -1, 0) };
    assert!(!memory.is_null() && memory as isize != -1);
    let start = memory as u64;
    assert!(crate::native::readable(start, LEN as u64));
    assert!(crate::native::readable(start + 100, 12));
    assert!(!crate::native::readable(0x10, 12), "nothing is mapped in page zero");
    assert!(!crate::native::readable(u64::MAX - 4, 12));

    let mut h = NativeHost::new(Recording::default());
    (h.dense_entries, h.record_loads, h.fast_buffers) = (true, true, true);
    h.memory.mapped(0x7000_0000, 65536, &PoolInfo { id: 1, base: 0x3000_0000, len: 65536, memory_entry: None }, 0);
    let count = LEN / 32;
    let heap = h.heap(HeapKind::Resource, start, count).unwrap();
    let last = count - 1;
    let roots = [Some(Root::Table(heap.clone(), last))];
    let write = |n: u64| crate::descriptor::record(start + (last * 32) as u64, crate::descriptor::View::Constant(0x7000_0000 + n * 256, 256));
    write(1);
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&last], (0x3000_0100, 256));
    assert_eq!(heap.records.get(), Records::Title(Some((0, true))));
    // The second page is taken away: the next preparation asks again and reads through the
    // kernel, which fails without a fault; the entry keeps what it showed.
    write(2);
    // SAFETY: the second half of the mapping made above (PROT_NONE).
    assert_eq!(unsafe { mprotect(memory.add(LEN / 2), LEN / 2, 0) }, 0);
    assert!(!crate::native::readable(start, LEN as u64) && crate::native::readable(start, (LEN / 2) as u64));
    h.prep_scope.set(1);
    h.remirror_tables(&roots, None);
    assert_eq!(heap.records.get(), Records::Title(Some((1, false))));
    assert_eq!(heap.heap.borrow()[&last], (0x3000_0100, 256));
    assert!(h.mirror(start + (last * 32) as u64).is_ok());
    // Readable again in a later preparation: the record written meanwhile is mirrored.
    // SAFETY: as above (PROT_READ | PROT_WRITE).
    assert_eq!(unsafe { mprotect(memory.add(LEN / 2), LEN / 2, 3) }, 0);
    h.prep_scope.set(2);
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&last], (0x3000_0200, 256));
    assert!(h.records_readable(&heap));
    // Memory the heap object owns needs no query; a released heap is never loaded from again,
    // though a recorded list may still bind it after its memory is gone.
    heap.records.set(Records::Owned);
    h.prep_scope.set(3);
    assert!(h.records_readable(&heap));
    assert_eq!(heap.records.get(), Records::Owned);
    h.drop_heap(start);
    assert_eq!(heap.records.get(), Records::Released);
    // SAFETY: the mapping made above, which nothing else names.
    assert_eq!(unsafe { munmap(memory, LEN) }, 0);
    assert!(!h.records_readable(&heap));
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&last], (0x3000_0200, 256));
    // The switches: either one keeps every read in the kernel.
    for (dense, loads) in [(false, true), (true, false)] {
        let mut h = NativeHost::new(Recording::default());
        (h.dense_entries, h.record_loads) = (dense, loads);
        let slots = [0u32; 8];
        let heap = h.heap(HeapKind::Resource, slots.as_ptr() as u64, 1).unwrap();
        heap.records.set(Records::Owned);
        assert!(!h.records_readable(&heap));
    }
}

/// The pool list: views alternating over more pools than it holds resolve as the pool table
/// does, and a change of the table drops every remembered mapping.
#[test]
fn the_pool_list_resolves_alternating_pools_and_follows_the_table() {
    for dense in [true, false] {
        let mut h = NativeHost::new(Recording::default());
        h.dense_entries = dense;
        let pools: Vec<_> = (0..6).map(|i| PoolInfo { id: 1 + i, base: 0x3000_0000 + (i << 24), len: 65536, memory_entry: None }).collect();
        for (i, pool) in pools.iter().enumerate() {
            h.memory.mapped(0x7000_0000 + (i << 20), 65536, pool, 0);
        }
        let mut seed = 0x9e3779b97f4a7c15u64;
        for n in 0..4000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // Mostly two or three pools, as a ring heap's views; sometimes any, a gap or an end.
            let pool = if n % 7 == 0 { seed as usize % 6 } else { seed as usize % 3 };
            let at = 0x7000_0000 + ((pool as u64) << 20) + (seed >> 16) % 65536;
            let len = [0, 1, 64, 65536][(seed >> 40) as usize % 4];
            let found = h.resolve_pool_cached(at, len).unwrap().map(|(_, ptr)| ptr);
            assert_eq!(found, h.memory.resolve(at, len).map(|(_, p)| p), "view {n}: {at:#x}+{len:#x}");
            if n % 500 == 499 {
                let moved = PoolInfo { id: 100 + n, base: 0x5000_0000 + (n << 16), len: 65536, memory_entry: None };
                h.memory.mapped(0x7000_0000, 65536, &moved, 0);
                assert_eq!(h.resolve_pool_cached(0x7000_0010, 16).unwrap().unwrap().1 as usize, moved.base + 16);
            }
        }
        for i in 1..6u64 {
            assert!(h.resolve_pool_cached(0x7000_0000 + (i << 20), 16).unwrap().is_some());
        }
        let held = h.last_pool.borrow().hits.iter().flatten().count();
        assert_eq!(held, if dense { POOL_WAYS } else { 1 });
        h.memory.pool_released(&pools[1]);
        h.forget_released();
        assert_eq!(h.last_pool.borrow().hits.iter().flatten().count(), 0);
        assert!(h.resolve_pool_cached(0x7010_0000, 16).unwrap().is_none());
    }
}

#[test]
fn unrelated_resources_do_not_rewrite_a_mirrored_table_entry() {
    let h = NativeHost::new(Recording::default());
    let (texture, other) = (Rc::new(11), Rc::new(22));
    let (key, unrelated) = (Rc::as_ptr(&texture) as usize, Rc::as_ptr(&other) as usize);
    h.register_texture(key, &texture);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1);

    h.register_texture(unrelated, &other);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1, "an unrelated registration only revalidates");
    assert_eq!(heap.mirrored.borrow()[&0].epoch.get(), h.resources_epoch.get(), "the next scan is cheap again");
    h.release_resource(unrelated);
    h.remirror_tables(&roots, None);
    h.register_buffer(unrelated, 0x1000, 256);
    h.remirror_tables(&roots, None);
    h.forget_resource(unrelated);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1, "unrelated buffers and removals are also irrelevant");
    assert_eq!(heap.heap.borrow()[&0], (11, u32::MAX));
    crate::descriptor::forget(key);
}

#[test]
fn a_texture_key_replacement_and_release_follow_the_new_texture() {
    let h = NativeHost::new(Recording::default());
    let texture = Rc::new(11);
    let key = Rc::as_ptr(&texture) as usize;
    h.register_texture(key, &texture);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    let record = record_at(handle);
    let first_version = h.resource_version(key);

    h.register_texture(key, &Rc::new(22));
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 2, "replacement without release invalidates too");
    assert_eq!(heap.heap.borrow()[&0], (22, u32::MAX));
    h.release_resource(key);
    assert_eq!(heap.heap.borrow()[&0], (0, 0), "released textures clear their entries immediately");
    assert_eq!(h.resource_version(key), 0);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 3, "an absent resource must not restore its old texture");
    h.register_texture(key, &Rc::new(33));
    assert!(h.resource_version(key) > first_version);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 4);
    assert_eq!(heap.heap.borrow()[&0], (33, u32::MAX));
    assert_eq!(record_at(handle), record, "resource identity, not descriptor bytes, changed");
    crate::descriptor::forget(key);
}

#[test]
fn a_device_mirror_skips_identical_records_but_writes_a_changed_record() {
    let h = NativeHost::new(Recording::default());
    let (a, b) = (Rc::new(11), Rc::new(22));
    let (key_a, key_b) = (Rc::as_ptr(&a) as usize, Rc::as_ptr(&b) as usize);
    h.register_texture(key_a, &a);
    h.register_texture(key_b, &b);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    let write = |key| crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    write(key_a);
    assert!(h.mirror(handle).unwrap());
    let record = record_at(handle);
    write(key_a);
    assert_eq!(record_at(handle), record, "writing the same view preserves the record");
    assert!(h.mirror(handle).unwrap());
    assert_eq!(h.backend.writes.get(), 1);
    write(key_b);
    assert_ne!(record_at(handle), record);
    assert!(h.mirror(handle).unwrap());
    assert_eq!(h.backend.writes.get(), 2);
    assert_eq!(heap.heap.borrow()[&0], (22, u32::MAX));
    crate::descriptor::forget(key_a);
    crate::descriptor::forget(key_b);
}

#[test]
fn placed_texture_stamps_follow_the_placement_key_and_keep_the_pool_generation() {
    let h = NativeHost::new(Recording::default());
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    let mut res = [0u32; 14];
    res[0] = 3;
    let mut view = [0u32; 16];
    (view[0], view[1], view[5], view[10]) = (28, 4, 1, 0x7000_0000);
    let key = super::super::textures::placed_key(u64::from(view[10]), &res);
    h.register_texture(key, &Rc::new(11));
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &res, &view, handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    h.register_texture(1, &Rc::new(99));
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1);
    h.register_texture(key, &Rc::new(22));
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 2);
    assert_eq!(heap.heap.borrow()[&0], (22, u32::MAX));
    h.memory.unmapped(0x7000_0000, 65536);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 3, "placed images still depend on pool mappings");
}

#[test]
fn a_buffer_resource_version_follows_its_address_and_forgetting_it() {
    let h = NativeHost::new(Recording::default());
    let mut bytes = vec![0u32; 16384];
    let pool = PoolInfo { id: 1, base: bytes.as_mut_ptr() as usize, len: 65536, memory_entry: None };
    h.memory.mapped(0x7000_0000, pool.len, &pool, 0);
    let key = bytes.as_ptr() as usize;
    h.register_buffer(key, 0x7000_0000, 256);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    h.register_texture(1, &Rc::new(99));
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1);
    h.register_buffer(key, 0x7000_0100, 512);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 2);
    assert_eq!(heap.heap.borrow()[&0], (pool.base + 256, 512));
    let version = h.resource_version(key);
    h.forget_resource(key);
    assert_eq!(h.resource_version(key), 0);
    h.register_buffer(key, 0x7000_0200, 128);
    assert!(h.resource_version(key) > version);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 3);
    assert_eq!(heap.heap.borrow()[&0], (pool.base + 512, 128));
    crate::descriptor::forget(key);
}

#[test]
fn a_uav_with_a_counter_keeps_the_global_resource_dependency() {
    let h = NativeHost::new(Recording::default());
    let (texture, counter) = (Rc::new(11), Box::new(0u32));
    let (key, counter_key) = (Rc::as_ptr(&texture) as usize, &*counter as *const u32 as usize);
    h.register_texture(key, &texture);
    h.register_buffer(counter_key, 0x1000, 4);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    let mut view = [0u32; 10];
    (view[0], view[1]) = (28, 4);
    crate::descriptor::create_uav(std::ptr::null_mut(), key as _, counter_key as _, &view, handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1);
    h.register_buffer(counter_key, 0x2000, 4);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 2, "both resources can change a counter UAV");
    crate::descriptor::forget(key);
    crate::descriptor::forget(counter_key);
}

#[test]
fn unchanged_uav_follows_remaps_and_recovers_after_an_unmapped_attempt() {
    let h = NativeHost::new(Recording::default());
    let mut old = vec![0u32; 16384];
    let mut new = vec![0u32; 16384];
    let pool = |id, base| PoolInfo {
        id,
        base,
        len: 65536,
        memory_entry: None,
    };
    let (a, b) = (
        pool(1, old.as_mut_ptr() as usize),
        pool(2, new.as_mut_ptr() as usize),
    );
    let address = 0x7000_0000usize;
    h.memory.pool_created(&a);
    h.memory.pool_created(&b);
    h.memory.mapped(address, 65536, &a, 0);
    let mut slots = [0u32; 8];
    let handle = slots.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    let mut res = [0u32; 14];
    res[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8], view[10]) = (1, 1, 1, address as u32);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &res, &view, handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    let write = |value: u32| {
        h.remirror_tables(&roots, None);
        let (ptr, size) = heap.heap.borrow()[&0];
        if size >= 4 {
            // SAFETY: the recording backend's non-null view resolves into the live vectors.
            unsafe {
                (ptr as *mut u32).write(value);
            }
        }
    };
    write(11);
    assert_eq!((old[0], new[0]), (11, 0));
    let record = record_at(handle);
    h.memory.mapped(address, 65536, &b, 0);
    write(22);
    assert_eq!(record_at(handle), record, "title descriptor did not change");
    assert_eq!(
        (old[0], new[0]),
        (11, 22),
        "recycled memory must not receive the next UAV write"
    );
    h.memory.unmapped(address, 65536);
    write(33);
    assert_eq!((old[0], new[0]), (11, 22));
    assert!(
        !heap.mirrored.borrow().contains_key(&0),
        "failed mirrors are not successful cache entries"
    );
    h.memory.mapped(address, 65536, &a, 0);
    write(44);
    assert_eq!((old[0], new[0]), (44, 22));
}

#[test]
fn submitted_table_buffer_owns_memory_after_more_than_eight_remaps() {
    let h = NativeHost::new(Recording::default());
    let heap = h.host_heap().unwrap();
    let first = h.memory_table(&heap).unwrap();
    first.write(&[0x1234abcd]);
    let ptr = first.ptr();
    // A backend submission retains the buffer, not the Pages wrapper.
    let submitted = first.buf.clone();
    let wrapper = Rc::downgrade(&first);
    drop(first);
    for _ in 0..12 {
        h.memory.unmapped(0x7000_0000, 65536);
        h.memory_table(&heap).unwrap();
    }
    assert!(
        wrapper.upgrade().is_none(),
        "no arbitrary retired-table queue"
    );
    assert!(submitted._memory.is_some());
    // SAFETY: the retained buffer owns the allocation at ptr.
    assert_eq!(unsafe { ptr.cast::<u32>().read() }, 0x1234abcd);
}

#[test]
fn forgetting_a_texture_keeps_retained_entries_until_the_resource_is_replaced() {
    let h = NativeHost::new(Recording::default());
    let texture = Rc::new(11);
    let key = Rc::as_ptr(&texture) as usize;
    h.register_texture(key, &texture);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    let record = record_at(handle);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1, "an unchanged entry does not rewrite the backend");
    h.forget_resource(key);
    assert_eq!(heap.heap.borrow()[&0], (11, u32::MAX), "upstream forget retains the heap view");
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&0], (11, u32::MAX));
    h.register_texture(key, &Rc::new(22));
    h.remirror_tables(&roots, None);
    assert_eq!(record_at(handle), record);
    assert_eq!(heap.heap.borrow()[&0], (22, u32::MAX));
    crate::descriptor::forget(key);
}

#[test]
fn shared_depth_keys_use_one_twin_and_colour_reads_follow_its_planes() {
    let h = NativeHost::new(Recording::default());
    let desc = TextureDesc::d2(Format::D32FloatS8, 8, 8);
    let depth = h.texture(&desc).unwrap();
    let keys = [0x123400, 0x123401];
    for key in keys {
        h.register_texture(key, &depth);
        h.register_depth(key, &depth, &desc);
    }
    let srv = crate::descriptor::View::Shader(keys[1], Some([41, 4, 0, 0, 0, 1, 0, 0, 0, 0]));
    assert!(h.twin_plane(keys[1], &depth, &srv).is_none(), "a colour read alone makes no twin");
    let uav = crate::descriptor::View::Unordered(keys[0], 0, Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
    let version = h.resource_version(keys[1]);
    let (write, format) = h.twin_plane(keys[0], &depth, &uav).unwrap();
    assert_eq!(format, Format::R32Float, "console 16-bit depth with stencil uses host float depth");
    assert!(h.resource_version(keys[1]) > version, "existing aliases are remirrored after twin creation");
    let (read, _) = h.twin_plane(keys[1], &depth, &srv).unwrap();
    assert!(Rc::ptr_eq(&read, &write));
    let stencil = crate::descriptor::View::Unordered(keys[1], 0, Some([62, 4, 0, 1, 0, 0, 0, 0, 0, 0]));
    let (stencil, format) = h.twin_plane(keys[1], &depth, &stencil).unwrap();
    assert_eq!(format, Format::R8Uint);
    assert!(!Rc::ptr_eq(&stencil, &write));
    assert_eq!(h.backend.textures.get(), 3, "one depth texture and its two planes");
    let twins = h.twins.borrow();
    assert!(Rc::ptr_eq(twins[&keys[0]].as_ref().unwrap(), twins[&keys[1]].as_ref().unwrap()));
    let twin = twins[&keys[0]].as_ref().unwrap();
    assert_eq!(twin.need(twin::TWIN), Some(false));
    twin.newer.set(twin::TWIN);
    assert_eq!(twin.need(twin::DEPTH), Some(true));
    assert_eq!(twin.copy(true).len(), 4, "both depth and stencil copy through staging");
}

/// Follow every copy through remapping, a gap (the whole view becomes null), and recovery.
#[test]
fn copied_wraps_follow_remaps_and_gaps_with_one_comparison_walk() {
    let h = NativeHost::new(Recording::default());
    const PAGE: usize = 16384;
    const VA: usize = 0x7100_0000;
    let pool = PoolInfo { id: 1, base: 0x1000_0000, len: 4 * PAGE, memory_entry: None };
    h.memory.mapped(VA, PAGE, &pool, 0);
    h.memory.mapped(VA + PAGE, PAGE, &pool, 2 * PAGE);
    let records = [0u32; 24];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 3).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[6], view[8], view[10]) = (1, 64, 1, (VA + PAGE - 128) as u32);
    for i in 0..3 {
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &view, start + 32 * i);
        assert!(h.mirror(start + 32 * i).unwrap());
    }
    let walks = || SIGNATURES.with(|n| n.replace(0));
    h.memory.pool_created(&PoolInfo { id: 2, base: 0x2000_0000, len: PAGE, memory_entry: None });
    walks();
    h.follow_title_pages();
    assert_eq!(walks(), 1);
    let old = h.aliases.borrow()[&(VA as u64, (VA + 2 * PAGE) as u64)].2.clone();
    let writes = h.backend.writes.get();
    h.memory.mapped(VA + PAGE, PAGE, &pool, 3 * PAGE);
    h.follow_title_pages();
    assert_eq!(walks(), if h.cache_mirrors { 2 } else { 4 }, "one comparison walk; cached copies reuse the first fresh wrap validation");
    assert_eq!(h.backend.writes.get() - writes, 3);
    let aliases = h.aliases.borrow();
    let (signature, _, new) = &aliases[&(VA as u64, (VA + 2 * PAGE) as u64)];
    assert!(!Rc::ptr_eq(&old, new));
    assert!(heap.title_views.borrow().values().all(|v| v.2 == Some(*signature)));
    drop(aliases);
    h.memory.unmapped(VA + PAGE, PAGE);
    h.follow_title_pages();
    assert!(heap.heap.borrow().values().all(|v| v.1 == 0), "a wrap with a gap is entirely null, not a partial span");
    assert_eq!(heap.title_views.borrow().len(), 3, "the null copies still follow their pages");
    assert!(heap.spanned.borrow().is_empty());
    walks();
    assert_eq!(h.alias_signature(VA as u64, (2 * PAGE) as u64), None);
    assert_eq!(h.alias_signature(VA as u64, (2 * PAGE) as u64), None);
    assert_eq!(walks(), 2, "standalone signature reads are fresh, including gaps");
    h.memory.mapped(VA + PAGE, PAGE, &pool, PAGE);
    assert!(h.alias_signature(VA as u64, (2 * PAGE) as u64).is_some());
    assert_eq!(walks(), 1);
    h.execute(&(), &[], &[], &[]).unwrap();
    assert!(heap.heap.borrow().values().all(|v| v.1 == 256), "all copies recover when the gap is mapped");
    // The recovered range is now one contiguous pool run, which is followed too.
    let signature = h.alias_signature(VA as u64, (2 * PAGE) as u64);
    assert!(signature.is_some() && heap.title_views.borrow().len() == 3);
    assert!(heap.title_views.borrow().values().all(|v| v.2 == signature), "a view one pool run holds follows its pages");
    assert!(heap.spanned.borrow().is_empty());
}

/// A view one pool run holds is followed on both mirror paths (the quick one takes its mapping's
/// signature, the general one walks), with the same signature, and is mirrored again when the
/// title maps another pool page under it; `VICTORY_FOLLOW_POOL_VIEWS=0` follows none.
#[test]
fn a_pool_run_view_is_followed_by_the_quick_and_the_general_mirror_alike() {
    const PAGE: usize = 65536;
    const VA: usize = 0x7200_0000;
    let pool = PoolInfo { id: 1, base: 0x1000_0000, len: 4 * PAGE, memory_entry: None };
    for (dense, follow) in [(true, true), (false, true), (true, false)] {
        let mut h = NativeHost::new(Recording::default());
        (h.dense_entries, h.follow_pool_views) = (dense, follow);
        h.memory.mapped(VA, PAGE, &pool, 0);
        let records = [0u32; 8];
        let start = records.as_ptr() as u64;
        let heap = h.heap(HeapKind::Resource, start, 1).unwrap();
        let mut resource = [0u32; 14];
        resource[0] = 1;
        let mut view = [0u32; 16];
        (view[1], view[6], view[8], view[10]) = (1, 64, 1, (VA + 256) as u32);
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &view, start);
        assert!(h.mirror(start).unwrap());
        assert_eq!(h.quick_mirrored.get(), u64::from(dense), "dense {dense}");
        let followed = heap.title_views.borrow().get(&0).copied();
        if !follow {
            assert_eq!(followed, None);
            continue;
        }
        let signature = h.alias_signature((VA + 256) as u64, 256);
        assert!(signature.is_some());
        assert_eq!(followed, Some(((VA + 256) as u64, 256, signature)), "dense {dense}");
        // The same record, another pool page under it: mirrored again, at the new page.
        let at = heap.heap.borrow()[&0].0;
        h.memory.mapped(VA, PAGE, &pool, 2 * PAGE);
        h.follow_title_pages();
        assert_eq!(heap.heap.borrow()[&0].0, at + 2 * PAGE, "dense {dense}: the entry kept the old pool page");
        assert_eq!(heap.title_views.borrow().get(&0).unwrap().2, h.alias_signature((VA + 256) as u64, 256));
        h.drop_heap(start);
    }
}

/// The title VA is a page-table key, not necessarily a pointer this process can dereference.
/// The merge tried wrap_title_memory(VA, ...) here before trying the pool-backed span.
#[test]
fn a_two_pool_view_uses_pool_pointers_and_keeps_replaced_spans() {
    let h = NativeHost::new(Recording::default());
    const PAGE: usize = 65536;
    const VA: usize = 0x23_4560_0000;
    let pools: Vec<_> = (0..3).map(|i| PoolInfo {
        id: i + 11, base: 0x1000_4000 + i * 2 * PAGE, len: 2 * PAGE, memory_entry: None,
    }).collect();
    h.memory.mapped(VA, PAGE, &pools[0], PAGE);
    h.memory.mapped(VA + PAGE, PAGE, &pools[1], 0);
    let records = [0u32; 8];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 1).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    let at = (VA + PAGE - 8) as u64;
    (view[1], view[4], view[8], view[10], view[11]) = (1, 64, 1, at as u32, (at >> 32) as u32);
    crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &view, start);
    assert!(h.mirror(start).unwrap());
    assert!(h.backend.wraps.borrow().is_empty(), "a logical title VA must never reach the no-copy wrapper");
    assert_eq!(h.backend.spans.borrow()[0], [(0, pools[0].base + PAGE, PAGE), (PAGE, pools[1].base, PAGE)]);
    let key = (VA, 2 * PAGE);
    let old = Rc::downgrade(&h.spans.borrow()[&key].buf);
    assert_eq!(heap.heap.borrow()[&0], (h.spans.borrow()[&key].base as usize + PAGE - 8, 256));
    assert!(heap.title_views.borrow().is_empty());

    // No descriptor rewrite or bound table: the next submission must refresh the span itself.
    h.memory.mapped(VA + PAGE, PAGE, &pools[2], 0);
    h.execute(&(), &[], &[], &[]).unwrap();
    assert_eq!(h.backend.spans.borrow()[1], [(0, pools[0].base + PAGE, PAGE), (PAGE, pools[2].base, PAGE)]);
    assert_eq!(heap.spanned.borrow()[&0], (key, h.spans.borrow()[&key].version));
    assert!(old.upgrade().is_some(), "the old buffer outlives the heap rewrite");

    h.memory.unmapped(VA + PAGE, PAGE);
    h.execute(&(), &[], &[], &[]).unwrap();
    assert_eq!(h.backend.spans.borrow()[2], [(0, pools[0].base + PAGE, PAGE)]);
    assert_eq!(heap.heap.borrow()[&0], (h.spans.borrow()[&key].base as usize + PAGE - 8, 256));
    h.memory.unmapped(VA, PAGE);
    h.execute(&(), &[], &[], &[]).unwrap();
    assert_eq!(heap.heap.borrow()[&0].1, 0, "a span without its starting page is null");
    assert!(heap.spanned.borrow().is_empty());
}

/// A replaced span buffer is let go once the work submitted before it was replaced has finished,
/// however many submissions that takes: a count of submissions let it go (and its alias be
/// unmapped) while command buffers that read it were still queued (docs/hang-notes.md). A backend
/// that cannot tell keeps the count.
#[test]
fn a_replaced_span_waits_for_the_work_submitted_before_it() {
    for tracked in [true, false] {
        let h = NativeHost::new(Recording::default());
        if tracked {
            h.backend.work.set(Some((7, 0)));
        }
        const PAGE: usize = 65536;
        const VA: usize = 0x23_4560_0000;
        let pools: Vec<_> = (0..3).map(|i| PoolInfo {
            id: i + 11, base: 0x1000_4000 + i * 2 * PAGE, len: 2 * PAGE, memory_entry: None,
        }).collect();
        h.memory.mapped(VA, PAGE, &pools[0], PAGE);
        h.memory.mapped(VA + PAGE, PAGE, &pools[1], 0);
        let records = [0u32; 8];
        let start = records.as_ptr() as u64;
        let _heap = h.heap(HeapKind::Resource, start, 1).unwrap();
        let mut resource = [0u32; 14];
        resource[0] = 1;
        let mut view = [0u32; 16];
        let at = (VA + PAGE - 8) as u64;
        (view[1], view[4], view[8], view[10], view[11]) = (1, 64, 1, at as u32, (at >> 32) as u32);
        crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &view, start);
        assert!(h.mirror(start).unwrap());
        let key = (VA, 2 * PAGE);
        let old = Rc::downgrade(&h.spans.borrow()[&key].buf);
        h.memory.mapped(VA + PAGE, PAGE, &pools[2], 0);
        // Many more submissions than the count while the earlier work has not finished.
        for _ in 0..2 * SPAN_GRACE {
            h.execute(&(), &[], &[], &[]).unwrap();
        }
        assert_eq!(old.upgrade().is_some(), tracked, "tracked {tracked}: kept while the earlier work runs");
        if tracked {
            h.backend.work.set(Some((9, 7)));
            h.execute(&(), &[], &[], &[]).unwrap();
            assert!(old.upgrade().is_none(), "let go once the work before it finished");
        }
    }
}

/// A released pool's buffer, and the pool's own mapping (`memory::let_go_of_pool`), are let go
/// once the work submitted before the host heard of the release has finished: a buffer over a
/// pool is reached by heap entries and the residency set, which retain nothing, so queued work
/// could write a pool the title had freed (docs/hang-notes.md). A backend that cannot tell lets go
/// at once, as before.
#[test]
fn a_released_pool_waits_for_the_work_submitted_before_it() {
    for tracked in [true, false] {
        let h = NativeHost::new(Recording::default());
        if tracked {
            h.backend.work.set(Some((7, 0)));
        }
        const VA: usize = 0x23_4570_0000;
        let pool = PoolInfo { id: 31 + tracked as usize, base: 0x1000_4000, len: 1 << 16, memory_entry: None };
        h.memory.pool_created(&pool);
        h.memory.mapped(VA, 1 << 16, &pool, 0);
        let buf = Rc::downgrade(&h.resolve(VA as u64, 64).unwrap().0);
        crate::native::memory::LET_GO.with(|l| l.borrow_mut().clear());
        h.memory.pool_released(&pool);
        for _ in 0..8 {
            h.execute(&(), &[], &[], &[]).unwrap();
        }
        let gone = crate::native::memory::LET_GO.with(|l| l.borrow().clone());
        if tracked {
            assert!(buf.upgrade().is_some(), "the buffer is kept while the earlier work runs");
            assert!(gone.is_empty(), "and the pool's mapping");
            h.backend.work.set(Some((9, 7)));
            h.execute(&(), &[], &[], &[]).unwrap();
        }
        assert!(buf.upgrade().is_none(), "tracked {tracked}: let go once the work before it finished");
        assert_eq!(crate::native::memory::LET_GO.with(|l| l.borrow().clone()), [pool.id]);
    }
}

#[test]
fn a_pool_goes_by_its_mark_or_at_once_without_one() {
    assert!(!pool_may_go(Some(5), |m| m <= 4));
    assert!(pool_may_go(Some(5), |m| m <= 5));
    assert!(pool_may_go(None, |_| unreachable!()));
}

#[test]
fn a_span_goes_by_its_mark_or_without_one_by_the_count() {
    assert!(!span_may_go(0, Some(5), 1000, |m| m <= 4));
    assert!(span_may_go(0, Some(5), 1, |m| m <= 5));
    assert!(!span_may_go(10, None, 10 + SPAN_GRACE - 1, |_| unreachable!()));
    assert!(span_may_go(10, None, 10 + SPAN_GRACE, |_| unreachable!()));
}

/// The cullers' buffer address includes dword 1's low 16 bits. The root SRV must point to
/// this table, and a later mirrored descriptor write must update it in place.
#[test]
fn view_addresses_bind_and_refresh_nonzero_buffer_high_words() {
    let h = NativeHost::new(Recording::default());
    let records = [0u32; 16];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 2).unwrap();
    let mut texture = [0u32; 14];
    (texture[0], texture[4], texture[6], texture[7], texture[9]) = (3, 64, 32, 1 | 1 << 16, 1);
    let mut image = [0u32; 16];
    (image[0], image[1], image[5]) = (28, 4, 1);
    crate::descriptor::create_placed_srv(std::ptr::null(), &texture, &image, start);
    let mut buffer = [0u32; 14];
    buffer[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[6], view[8], view[10], view[11]) = (1, 64, 1, 0x3450_0000, 0x12);
    crate::descriptor::create_placed_srv(std::ptr::null(), &buffer, &view, start + 32);
    let mut p = Prep {
        root: RootArgs::default(), ps_root: None, bufs: vec![], vertex: vec![],
        index: None, blit: vec![], indirect: None, count: None, heap: None,
    };
    h.view_address_args(Some(&heap), &mut p).unwrap();
    let table = heap.addresses.borrow().as_ref().unwrap().clone();
    assert_eq!(p.root.0, (table.ptr() as u64).to_le_bytes());
    assert!(Rc::ptr_eq(&p.bufs[0], &table.buf), "the submission retains the address table");
    // SAFETY: the table owns at least eight dwords and no backend work reads or writes it.
    let words = || unsafe { std::slice::from_raw_parts(table.ptr().cast::<u32>(), 8).to_vec() };
    assert_eq!(words(), [0, 0xC000_0000, 0x0007_C00F, 0x9000_0FAC, 0x3450_0000, 0x12, 0, 0]);
    view[11] = 0x23;
    h.memory.mapped(0x23_3450_0000, 65536, &PoolInfo {
        id: 1, base: 0x1000_0000, len: 65536, memory_entry: None,
    }, 0);
    crate::descriptor::create_placed_srv(std::ptr::null(), &buffer, &view, start + 32);
    assert!(h.mirror(start + 32).unwrap());
    assert_eq!(&words()[4..], [0x3450_0000, 0x23, 0, 0]);
    crate::descriptor::create_cbv(std::ptr::null(), &[0, 0, 256, 0], start + 32);
    assert!(h.mirror(start + 32).unwrap());
    assert_eq!(&words()[4..], [0; 4]);
}

/// Native registrations accept opaque keys, including placed hashes, not just Resource objects.
/// Preparing the address table walks even entries outside the dispatch's bound root tables.
#[test]
fn view_addresses_never_dereference_native_resource_keys() {
    use crate::descriptor::View;
    let h = NativeHost::new(Recording::default());
    let records = [0u32; 8];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 1).unwrap();
    let mut resource = [0u32; 14];
    (resource[0], resource[4], resource[6], resource[7], resource[9]) = (3, 64, 32, 1 | 1 << 16, 1);
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    let mut placed = [0u32; 16];
    placed[..10].copy_from_slice(&srv);
    let image_words = [0, 0xC000_0000, 0x0007_C00F, 0x9000_0FAC];
    let pso = Rc::new(ComputePso {
        pso: (), threads: [1; 3],
        binding: Some(Binding { view_addresses: true, ..Binding::default() }),
    });
    let mut list = NativeList::default();
    list.set_heaps(Some(&heap), None);
    list.set_compute_pipeline(&pso);
    list.dispatch(1, 1, 1).unwrap();

    for key in [0x4EA9_0000, super::super::textures::placed_key(0x1234_0000, &resource)] {
        assert!(!crate::resource::is_resource(key as _));
        h.register_texture(key, &Rc::new(11));
        for desc in [Some(srv), None] {
            for view in [View::Shader(key, desc), View::Unordered(key, 0, desc)] {
                // Force the initial heap walk through dispatch preparation, before any mirror.
                *heap.addresses.borrow_mut() = None;
                crate::descriptor::record(start, view.clone());
                let prepared = h.prep(&list.cmds[0]).unwrap();
                let table = heap.addresses.borrow().as_ref().unwrap().clone();
                assert_eq!(prepared.root.0, (table.ptr() as u64).to_le_bytes());
                // SAFETY: the CPU backend owns four initialized dwords; no GPU work is run.
                let words = || unsafe { std::slice::from_raw_parts(table.ptr().cast::<u32>(), 4).to_vec() };
                assert_eq!(words(), [0; 4], "an opaque key has no Resource description");
                crate::descriptor::record(start, View::Placed(false, resource, placed));
                let _ = h.mirror(start); // No native texture registered for this placed record.
                assert_eq!(words(), image_words, "placed records supply their own description");
                crate::descriptor::record(start, view);
                let _ = h.mirror(start);
                assert_eq!(words(), [0; 4], "mirrored writes use the same guard");
            }
        }
        h.forget_resource(key);
        assert_eq!(h.address_words(Some(&View::Shader(key, None))), [0; 4], "stale record");
        h.register_buffer(key, 0x23_3450_0000, 256);
        for view in [View::Shader(key, None), View::Unordered(key, 0, None)] {
            assert_eq!(h.address_words(Some(&view)), [0x3450_0000, 0x23, 0, 0]);
        }
        h.forget_resource(key);
        crate::descriptor::forget(key);
    }
    // Real objects still supply their title size (not the possibly downscaled host texture).
    let pixels = [0u32; 64 * 32];
    let object = crate::resource::test_texture_at(pixels.as_ptr() as u64, (64, 32), 1, 28, 0, 0);
    assert!(crate::resource::is_resource(object));
    for view in [View::Shader(object as usize, Some(srv)), View::Unordered(object as usize, 0, None)] {
        assert_eq!(h.address_words(Some(&view)), image_words);
    }
    let release: extern "system" fn(crate::object::This) -> u32 = crate::object::slot(object, 0x10);
    assert_eq!(release(object), 0);
}

/// Exercise upstream's covered-placement and byte-budget ownership on the CPU. The heap
/// protects a sampled copy until its entry is replaced; eviction drops the last host Rc.
#[test]
fn covered_and_budget_eviction_drop_unshown_copies() {
    use crate::native::title::Title;
    use victory_formats::texture::Swizzle;
    let mut t = Title::new(Recording::default(), Default::default());
    t.covered_clear = false; // Pin upstream's wait; the default clearing path is tested below.
    let memory = vec![0x44u32; 16384];
    let base = memory.as_ptr() as usize;
    let pool = PoolInfo { id: 1, base, len: memory.len() * 4, memory_entry: None };
    t.host.memory.mapped(base, pool.len, &pool, 0);
    let mut placed = |at: usize, size: u32| {
        let address = (base + at) as u64;
        let object = crate::resource::test_texture_at(address, (size.into(), size), 1, 28, Swizzle::LINEAR.0, 0);
        let words = crate::resource::texture_desc(object).unwrap();
        let info = crate::resource::placed_info(address, &words).unwrap();
        let key = super::super::textures::placed_key(address, &words);
        t.texture_placed(address, key, 1, info.size);
        t.texture_created(key, info);
        assert!(t.realize(key));
        key
    };
    let covered = placed(0, 32);
    let idle = placed(0x4000, 16);
    let covered_weak = Rc::downgrade(&t.host.texture_at(covered).unwrap());
    let idle_weak = Rc::downgrade(&t.host.texture_at(idle).unwrap());
    let records = [0u32; 8];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null(), covered as _, std::ptr::null(), start);
    assert!(t.host.mirror(start).unwrap());
    // A placement at a different start covers the sampled copy.
    t.texture_placed((base + 0x800) as u64, 123, 1, 1024);
    assert!(t.covered.contains_key(&covered));
    assert!(t.unloaded.contains(&covered) && !t.texture_watched(covered));
    t.sync_dirty();
    assert!(!t.unloaded.contains(&covered) && t.texture_watched(covered));
    assert_eq!(t.uploads_of(covered), 1);
    assert!(covered_weak.upgrade().is_some() && heap.shows_texture(covered));
    crate::descriptor::create_srv(std::ptr::null(), idle as _, std::ptr::null(), start);
    assert!(t.host.mirror(start).unwrap());
    t.sync_dirty();
    assert!(covered_weak.upgrade().is_none(), "covered eviction drops the copy");
    assert!(!heap.shows_texture(covered) && t.host.is_evicted(covered));
    assert!(t.pending.contains_key(&covered));
    let bytes = t.budget.live;
    t.set_texture_budget(1);
    t.keep_to_budget();
    assert!(idle_weak.upgrade().is_some(), "a shown copy stays over budget");
    // A null buffer replaces the last texture entry.
    crate::descriptor::record(start, crate::descriptor::View::Constant(0, 0));
    assert!(t.host.mirror(start).unwrap());
    for _ in 0..=super::super::textures::RECHECK { t.keep_to_budget(); }
    assert!(idle_weak.upgrade().is_none(), "budget eviction drops the copy");
    assert!(bytes > 0 && t.budget.live == 0);
    assert!(!heap.shows_texture(idle) && t.host.is_evicted(idle));
    assert_eq!(heap.heap.borrow()[&0].1, 0);
}

/// Synthetic title memory and descriptors; Recording never asks for a Metal device.
pub(in crate::native) struct CoveredCopies {
    pub(in crate::native) title: crate::native::title::Title<Recording>,
    memory: Vec<u32>,
    records: [u32; 8],
}

impl CoveredCopies {
    pub(in crate::native) fn new() -> Self {
        let mut title = crate::native::title::Title::new(Recording::default(), Default::default());
        // The clearing path under test, whatever the environment says.
        title.covered_clear = true;
        let memory = vec![0x44u32; 16384];
        let base = memory.as_ptr() as usize;
        let pool = PoolInfo { id: 1, base, len: memory.len() * 4, memory_entry: None };
        title.host.memory.mapped(base, pool.len, &pool, 0);
        Self { title, memory, records: [0; 8] }
    }

    pub(in crate::native) fn place(&mut self, at: usize, size: u32) -> usize {
        use crate::native::textures::{TextureInfo, placed_key};
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let address = self.memory.as_ptr() as u64 + at as u64;
        let key = placed_key(address, &[size; 14]);
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d, width: size, height: size, depth: 1,
            mips: 1, element_bytes: 4, block: 1, samples: 1, swizzle: Swizzle::LINEAR,
        }).unwrap();
        let info = TextureInfo {
            dimension: Dimension::Tex2d, format: 28, width: size, height: size,
            depth_or_layers: 1, flags: 0, size: layout.size as usize, layout,
            memory: address as *const u8,
        };
        self.title.placed_view(address, key, 1, info, true);
        assert!(self.title.host.knows(key));
        key
    }

    pub(in crate::native) fn show(&mut self, key: usize) -> Rc<NativeHeap<Recording>> {
        let start = self.records.as_ptr() as u64;
        let heap = self.title.host.heap(HeapKind::Resource, start, 1).unwrap();
        let mut srv = [0u32; 10];
        (srv[0], srv[1], srv[5]) = (28, 4, 1);
        crate::descriptor::create_srv(std::ptr::null(), key as _, &srv, start);
        assert!(self.title.host.mirror(start).unwrap());
        heap
    }

    fn list(heap: &Rc<NativeHeap<Recording>>) -> NativeList<Recording> {
        let mut list = NativeList::default();
        let pso = Rc::new(ComputePso { pso: (), threads: [1; 3], binding: None });
        list.set_compute_pipeline(&pso);
        list.set_compute_root_table(0, heap, 0);
        list.dispatch(1, 1, 1).unwrap();
        list
    }
}

impl Drop for CoveredCopies {
    fn drop(&mut self) {
        let keys: Vec<_> = self.title.placed.values().map(|&(k, _)| k).collect();
        for key in keys {
            self.title.host.release_resource(key);
            self.title.texture_released(key);
            crate::descriptor::forget(key);
        }
    }
}

#[test]
fn covered_clear_keeps_stale_records_null_until_a_title_placement() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    let weak = Rc::downgrade(&f.title.host.texture_at(old).unwrap());
    let record = f.records;
    let new = f.place(0x800, 16);
    // Both retry caches must be discarded, including a pending attempt on a shown entry.
    heap.pending_mirror.borrow_mut().insert(0, ([7; 12], 0, 0));
    heap.typed_pending.borrow_mut().set(0, true, true);
    f.title.sync_dirty();
    assert!(weak.upgrade().is_none());
    assert_eq!(heap.heap.borrow()[&0], (0, 0));
    assert!(!heap.shows_texture(old));
    assert!(heap.mirrored.borrow().is_empty() && heap.pending_mirror.borrow().is_empty());
    assert!(!heap.typed_pending.borrow().has_explicit());
    assert!(f.title.host.is_covered(old) && !f.title.host.is_evicted(old));
    assert!(f.title.pending.contains_key(&old));
    let bytes = f.title.budget.live;
    // Rebinding and internal realization are not the title's doing: nothing comes back.
    let roots = [Some(Root::Table(heap.clone(), 0))];
    let list = CoveredCopies::list(&heap);
    for _ in 0..3 {
        f.title.host.remirror_tables(&roots, None);
        assert!(f.title.host.take_wanted().is_empty());
        assert!(!f.title.realize(old));
        let named = f.title.keep_named(std::iter::once(&list));
        f.title.sync_dirty();
        f.title.let_named_go(named);
        assert_eq!(heap.heap.borrow()[&0], (0, 0));
        assert!(!heap.shows_texture(old) && f.title.host.knows(new));
        assert_eq!(f.title.budget.live, bytes);
        assert_eq!(f.title.covered_history.ping_pong, 0);
    }
    // Even a fresh slot holding the same stale record mirrors as null without a request.
    let copied = record;
    let copy_at = copied.as_ptr() as u64;
    let copy_heap = f.title.host.heap(HeapKind::Resource, copy_at, 1).unwrap();
    copy_heap.typed_pending.borrow_mut().set(0, true, true);
    copy_heap.pending_mirror.borrow_mut().insert(0, ([7; 12], 0, 0));
    assert!(f.title.host.mirror(copy_at).unwrap());
    assert_eq!(copy_heap.heap.borrow()[&0], (0, 0));
    assert!(!copy_heap.typed_pending.borrow().has_explicit());
    assert!(copy_heap.pending_mirror.borrow().is_empty());
    assert!(f.title.host.take_wanted().is_empty());
    assert_eq!(f.place(0, 32), old); // An explicit title placement can revive it.
    f.title.host.remirror_tables(&roots, None);
    f.title.sync_dirty();
    assert!(heap.shows_texture(old));
    assert_eq!(heap.heap.borrow()[&0].1, u32::MAX);
    assert_eq!(f.title.uploads_of(old), 1);
    assert_eq!(f.records, record, "the title never rewrote its record");
    assert_eq!(f.title.covered_history.ping_pong, 1);
}

/// The title copies the record of a view whose copy was given up as covered (`CopyDescriptors`
/// into the heap its shaders index): the view is in use, so its copy is made again, its entries
/// show it again, and the placement that covered it never clears it again. (Left null, every
/// material that samples such a view drew black: the walls' and the clothes' colour.)
#[test]
fn covered_clear_makes_a_copy_again_when_the_title_copies_its_record() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    let new = f.place(0x800, 16);
    f.title.sync_dirty();
    assert!(f.title.host.is_covered(old) && !heap.shows_texture(old));
    let info = f.title.pending[&old].clone();
    f.title.placed_view(info.memory as u64, old, 1, info, false);
    assert!(f.title.host.knows(old) && !f.title.host.is_covered(old), "made again");
    assert!(f.title.host.knows(new), "beside the placement that covered it");
    f.title.host.remirror_tables(&[Some(Root::Table(heap.clone(), 0))], None);
    f.title.sync_dirty();
    assert!(heap.shows_texture(old), "its entry shows it again");
    assert_eq!(f.title.uploads_of(old), 1, "filled from title memory again");
    assert_eq!((f.title.covered_history.copied, f.title.covered_history.ping_pong), (1, 0));
    // The same placement over it again does not cover it: the pair is exempt.
    for _ in 0..3 {
        assert_eq!(f.place(0x800, 16), new);
        f.title.sync_dirty();
        assert!(f.title.covered.is_empty() && f.title.host.knows(old) && heap.shows_texture(old));
    }
    assert_eq!(f.title.covered_history.copied, 1);
}

/// A copy of the record seen while the covered copy is still resident (named by work in flight)
/// keeps it: it is never given up, and no entry is cleared.
#[test]
fn covered_clear_keeps_a_copy_whose_record_the_title_copies_first() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    f.place(0x800, 16);
    assert!(f.title.covered.contains_key(&old));
    assert!(f.title.copied_while_covered(old));
    assert!(!f.title.copied_while_covered(old), "once");
    f.title.sync_dirty();
    assert!(f.title.host.knows(old) && heap.shows_texture(old));
    assert_eq!(f.title.covered_history.copied, 1);
    // With the switch off the copy waits for its entries as before; a copy of the record is no news.
    let mut f = CoveredCopies::new();
    f.title.covered_clear = false;
    let old = f.place(0, 32);
    f.place(0x800, 16);
    assert!(!f.title.copied_while_covered(old) && f.title.covered.contains_key(&old));
}

/// The title places a covered view again before its copy is given up (both views are placed
/// every frame): neither is ever cleared, and neither covers the other from then on.
#[test]
fn covered_clear_exempts_views_the_title_keeps_placing_before_either_is_cleared() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    let new = f.place(0x800, 16);
    assert!(f.title.covered.contains_key(&old));
    assert_eq!(f.place(0, 32), old);
    assert!(f.title.covered.is_empty(), "the pair is exempt: the newer view is not covered in turn");
    for _ in 0..3 {
        f.title.sync_dirty();
        assert!(f.title.host.knows(old) && f.title.host.knows(new) && heap.shows_texture(old));
        f.place(0x800, 16);
        f.place(0, 32);
        assert!(f.title.covered.is_empty());
    }
    assert_eq!((f.title.covered_history.copied, f.title.covered_history.ping_pong), (1, 0));
}

#[test]
fn covered_clear_keeps_copies_named_by_submissions_in_flight() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    f.place(0x800, 16);
    assert!(f.title.names_wanted(), "covered copies require names even with the budget off");
    let list = CoveredCopies::list(&heap);
    let first = f.title.keep_named(std::iter::once(&list));
    let second = f.title.keep_named(std::iter::once(&list));
    assert!(first.contains(&old) && second.contains(&old));
    f.title.sync_dirty();
    assert!(heap.shows_texture(old) && f.title.host.knows(old));
    f.title.let_named_go(first);
    f.title.sync_dirty();
    assert!(heap.shows_texture(old) && f.title.host.knows(old));
    f.title.let_named_go(second);
    f.title.sync_dirty();
    assert!(!f.title.host.knows(old));
    assert_eq!(heap.heap.borrow()[&0], (0, 0));
}

#[test]
fn covered_clear_keeps_a_copy_held_by_a_recorded_list() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    let new = f.place(0x800, 16);
    let mut list = NativeList::<Recording>::default();
    list.copy_texture_region(
        &f.title.host.texture_at(old).unwrap(), 0, 0, [0; 3], [1; 3],
        &f.title.host.texture_at(new).unwrap(), 0, 0, [0; 3],
    );
    f.title.sync_dirty();
    assert!(f.title.host.knows(old) && heap.shows_texture(old));
    drop(list);
    f.title.sync_dirty();
    assert!(!f.title.host.knows(old));
    assert_eq!(heap.heap.borrow()[&0], (0, 0));
}

#[test]
fn covered_clear_reverse_displacement_keeps_both_copies() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    let new = f.place(0x800, 16);
    f.title.sync_dirty();
    assert!(f.title.host.is_covered(old));
    assert!(!f.title.realize(old));
    assert_eq!(f.title.covered_history.ping_pong, 0);
    assert_eq!(f.place(0, 32), old);
    assert_eq!(f.title.covered_history.ping_pong, 1);
    // Explicit placements in both directions must preserve the exemption as well.
    for _ in 0..3 {
        f.place(0x800, 16);
        f.place(0, 32);
        f.title.sync_dirty();
        assert!(f.title.host.knows(old) && f.title.host.knows(new));
        assert!(f.title.covered.is_empty());
    }
    f.title.host.remirror_tables(&[Some(Root::Table(heap.clone(), 0))], None);
    assert!(heap.shows_texture(old));
    assert_eq!(f.title.covered_history.ping_pong, 1);
}

#[test]
fn covered_clear_switch_off_preserves_the_shown_copy() {
    let mut f = CoveredCopies::new();
    f.title.covered_clear = false;
    let old = f.place(0, 32);
    let heap = f.show(old);
    f.place(0x800, 16);
    f.title.sync_dirty();
    assert!(f.title.host.knows(old) && heap.shows_texture(old));
    assert_eq!(f.title.host.cleared(), 0);
    assert!(heap.mirrored.borrow().contains_key(&0));
    crate::descriptor::record(f.records.as_ptr() as u64, crate::descriptor::View::Constant(0, 0));
    assert!(f.title.host.mirror(f.records.as_ptr() as u64).unwrap());
    f.title.sync_dirty();
    assert!(f.title.host.is_evicted(old));
}

#[test]
fn covered_clear_keeps_gpu_written_and_unordered_copies() {
    for unordered in [false, true] {
        let mut f = CoveredCopies::new();
        let old = f.place(0, 32);
        let heap = f.show(old);
        f.place(0x800, 16);
        assert!(f.title.covered.contains_key(&old));
        if unordered {
            let mut uav = [0u32; 10];
            (uav[0], uav[1]) = (28, 4);
            crate::descriptor::record(f.records.as_ptr() as u64, crate::descriptor::View::Unordered(old, 0, Some(uav)));
            assert!(f.title.host.mirror(f.records.as_ptr() as u64).unwrap());
        } else {
            f.title.gpu_written(old);
        }
        f.title.sync_dirty();
        assert!(f.title.host.knows(old) && heap.shows_texture(old));
        assert_eq!(f.title.host.cleared(), 0);
    }
}

#[test]
fn covered_clear_reads_the_environment_switch() {
    let title = crate::native::title::Title::new(Recording::default(), Default::default());
    assert_eq!(title.covered_clear, std::env::var("VICTORY_COVERED_CLEAR").as_deref() != Ok("0"), "on unless turned off");
}

/// Compare raw and lazily promoted entries, auxiliary tables and host tracking after a miss
/// and a hit, against decoding the identical record with the cache disabled.
fn compare_replayed_view(h: &mut NativeHost<Recording>, kind: HeapKind, view: crate::descriptor::View) {
    h.backend.capture_details.set(true);
    let records = [0u32; 24];
    let start = records.as_ptr() as u64;
    let heap = h.heap(kind, start, 3).unwrap();
    if kind == HeapKind::Resource {
        // As `view_strides` makes it: a stride, then a minimum LOD clamp, for `HEAP_VIEWS` entries.
        *heap.strides.borrow_mut() = Some(Rc::new(Pages::new(&h.backend, victory_shader::emit::HEAP_VIEWS as usize * 8).unwrap()));
        *heap.addresses.borrow_mut() = Some(Rc::new(Pages::new(&h.backend, 48).unwrap()));
    }
    let mut expected = None;
    let before = h.backend.replays.get();
    for index in 0..3 {
        h.cache_mirrors = index != 0;
        crate::descriptor::record(start + index as u64 * NativeHost::<Recording>::step(kind), view.clone());
        assert!(h.mirror(start + index as u64 * NativeHost::<Recording>::step(kind)).unwrap(), "index={index} view={view:?}");
        let raw = h.backend.capture_descriptor(&heap.heap, index);
        let tracking = (
            heap.spanned.borrow().get(&index).copied(), heap.title_views.borrow().get(&index).copied(),
            heap.shown(index), heap.twin_uavs.borrow().get(&index).copied(),
            heap.twin_srvs.borrow().get(&index).copied(), heap.depth_srvs.borrow().get(&index).copied(),
            heap.typed_pending.borrow().contains(index), heap.typed_pending.borrow().is_explicit(index),
        );
        let mut tables = [0u32; 6];
        if let Some(strides) = &*heap.strides.borrow() {
            // SAFETY: this host-owned table has `2 * HEAP_VIEWS` dwords; no GPU is using it.
            tables[0] = unsafe { strides.ptr().cast::<u32>().add(index).read() };
            // SAFETY: as above; the entry's minimum LOD clamp.
            tables[5] = unsafe { strides.ptr().cast::<u32>().add(victory_shader::emit::HEAP_VIEWS as usize + index).read() };
            if kind == HeapKind::Resource {
                assert_eq!(tables[5], super::super::descriptors::min_lod_word(&view, &h.resources.borrow()), "index={index} {view:?}");
            }
        }
        if let Some(addresses) = &*heap.addresses.borrow() {
            // SAFETY: same live table, four words per entry.
            tables[1..5].copy_from_slice(unsafe { std::slice::from_raw_parts(addresses.ptr().cast::<u32>().add(index * 4), 4) });
        }
        h.materialize_typed_buffers(&[Some(Root::Table(heap.clone(), index))], None, Some(&heap));
        let result = (raw, h.backend.capture_descriptor(&heap.heap, index), tracking, tables);
        if let Some(expected) = &expected { assert_eq!(&result, expected, "{view:?}"); }
        else { expected = Some(result); }
    }
    let raw_buffer = matches!(decode(&view, &h.resources.borrow()), Some(Decoded::Buffer { addr, .. }) if addr != 0)
        && typed_buffer(&view, &h.resources.borrow()).is_none();
    assert!(h.backend.replays.get() > before || (h.fast_buffers && raw_buffer), "third copy must replay: {view:?}");
    h.drop_heap(start);
}

#[test]
fn cached_views_match_uncached_buffers_images_samplers_depth_and_spans() {
    use crate::descriptor::View;
    let mut h = NativeHost::new(Recording::default());
    let keys = [Box::new(1u8), Box::new(2), Box::new(3)];
    let keys = keys.each_ref().map(|b| &**b as *const u8 as usize);
    let pool = PoolInfo { id: 1, base: 0x3000_0000, len: 65536, memory_entry: None };
    h.memory.mapped(0x7000_0000, pool.len, &pool, 0);
    h.register_buffer(keys[0], 0x7000_0000, 65536);
    h.register_texture(keys[1], &Rc::new(50));
    let depth = h.texture(&TextureDesc::d2(Format::D32FloatS8, 8, 8)).unwrap();
    h.register_texture(keys[2], &depth);
    h.register_depth(keys[2], &depth, &TextureDesc::d2(Format::D32FloatS8, 8, 8));
    for view in [
        View::Constant(0x7000_0010, 256), View::Constant(0, 512),
        View::Shader(keys[0], None),
        View::Shader(keys[0], Some([41, 1, 0, 0, 4, 0, 16, 0, 0, 0])),
        View::Shader(keys[0], Some([0, 1, 0, 0, 4, 0, 16, 4, 0, 0])),
        View::Shader(keys[0], Some([0, 1, 0, 0, 4, 0, 16, 8, 0, 0])),
        View::Unordered(keys[0], 0, Some([0, 1, 4, 0, 16, 16, 0, 0, 0, 0])),
        View::Unordered(keys[0], 0, Some([42, 1, 4, 0, 16, 0, 0, 0, 0, 0])),
        View::Shader(keys[1], None),
        View::Shader(keys[1], Some([29, 5, 0, 0, 2, 3, 1, 2, 0, 0])),
        // A 2D view of every mip with a minimum LOD clamp of 1.5: the clamp's table word too.
        View::Shader(keys[1], Some([28, 4, 0, 0, 0, u32::MAX, 0, 1.5f32.to_bits(), 0, 0])),
        View::Unordered(keys[1], 0, Some([28, 4, 1, 0, 0, 0, 0, 0, 0, 0])),
        View::Unordered(keys[2], 0, Some([41, 4, 0, 0, 0, 0, 0, 0, 0, 0])),
        View::Shader(keys[2], Some([41, 4, 0, 0, 0, 1, 0, 0, 0, 0])),
        View::Shader(keys[2], Some([40, 4, 0, 0, 0, 1, 0, 0, 0, 0])),
    ] { compare_replayed_view(&mut h, HeapKind::Resource, view); }
    let mut sampler = [0u32; 27];
    sampler[1..4].fill(3); // clamp
    sampler[4] = 1.25f32.to_bits();
    sampler[5] = 1;
    compare_replayed_view(&mut h, HeapKind::Sampler, View::ConsoleSampler(sampler));
    let mut resource = [0u32; 14]; resource[0] = 1;
    let mut view = [0u32; 16];
    (view[0], view[1], view[6], view[10]) = (41, 1, 8, 0x7000_0040);
    compare_replayed_view(&mut h, HeapKind::Resource, View::Placed(false, resource, view));
    h.memory.mapped(0x7001_0000, 65536, &PoolInfo { id: 2, base: 0x4000_0000, ..pool }, 0);
    compare_replayed_view(&mut h, HeapKind::Resource, View::Constant(0x7000_fff0, 64));
    for key in keys { crate::descriptor::forget(key); }
}

#[test]
fn fresh_copies_invalidate_after_release_registration_remap_and_span_refresh() {
    use crate::descriptor::View;
    let mut h = NativeHost::new(Recording::default());
    h.cache_mirrors = true;
    let records = [0u32; 128];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 16).unwrap();
    let key = start as usize;
    let copy = |index: usize, view: View| {
        let handle = start + index as u64 * 32;
        crate::descriptor::record(handle, view);
        handle
    };
    h.register_texture(key, &Rc::new(11));
    let view = View::Shader(key, None);
    for i in 0..2 { assert!(h.mirror(copy(i, view.clone())).unwrap()); }
    let hits = h.backend.replays.get();
    assert_eq!(hits, 1);
    h.register_texture(key + 1, &Rc::new(99));
    assert!(h.mirror(copy(2, view.clone())).unwrap());
    assert_eq!(h.backend.replays.get(), hits + 1, "unrelated registration preserves a decoded image");
    h.register_texture(key, &Rc::new(22));
    assert!(h.mirror(copy(3, view.clone())).unwrap());
    assert_eq!(heap.heap.borrow()[&3], (22, u32::MAX));
    assert_eq!(h.backend.replays.get(), hits + 1, "replacement must decode");
    let record = record_at(start);
    h.release_resource(key);
    assert!(!h.mirror_cache.borrow().contains_key(&record), "release drops cached retention immediately");
    assert!(!h.mirror(copy(4, view.clone())).unwrap());
    crate::descriptor::forget(key);
    h.register_texture(key, &Rc::new(33));
    assert!(h.mirror(copy(5, view)).unwrap());
    assert_ne!(record_at(start + 5 * 32), record, "descriptor generation changed too");
    assert_eq!(heap.heap.borrow()[&5], (33, u32::MAX));

    let a = PoolInfo { id: 1, base: 0x3000_0000, len: 65536, memory_entry: None };
    let b = PoolInfo { id: 2, base: 0x4000_0000, ..a };
    let c = PoolInfo { id: 3, base: 0x5000_0000, ..a };
    h.memory.mapped(0x7000_0000, 65536, &a, 0);
    let buffer = View::Constant(0x7000_0010, 64);
    for i in 6..8 { assert!(h.mirror(copy(i, buffer.clone())).unwrap()); }
    h.memory.mapped(0x7000_0000, 65536, &b, 0);
    assert!(h.mirror(copy(8, buffer)).unwrap());
    assert_eq!(heap.heap.borrow()[&8], (b.base + 16, 64));
    h.memory.mapped(0x7001_0000, 65536, &a, 0);
    let span = View::Constant(0x7000_fff0, 64);
    for i in 9..11 { assert!(h.mirror(copy(i, span.clone())).unwrap()); }
    let old = heap.spanned.borrow()[&9];
    h.memory.mapped(0x7001_0000, 65536, &c, 0);
    h.refresh_spans();
    assert_ne!(heap.spanned.borrow()[&9], old);
    assert_eq!(heap.spanned.borrow()[&9], heap.spanned.borrow()[&10]);
    assert!(h.mirror(copy(11, span)).unwrap());
    assert_eq!(heap.spanned.borrow()[&11], heap.spanned.borrow()[&9]);
    assert_eq!(heap.heap.borrow()[&11], heap.heap.borrow()[&9]);
    h.memory.pool_released(&b);
    assert!(h.mirror(copy(12, View::Constant(0x7000_0010, 64))).is_err());
    assert!(!heap.mirrored.borrow().contains_key(&12));
    crate::descriptor::forget(key);
}

#[test]
fn cached_depth_reads_follow_first_twin_and_changed_mip_policy() {
    use crate::descriptor::View;
    let mut h = NativeHost::new(Recording::default());
    h.cache_mirrors = true;
    h.backend.capture_details.set(true);
    let records = [0u32; 64];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 8).unwrap();
    let key = start as usize;
    let desc = TextureDesc::d2(Format::D32Float, 8, 8);
    let texture = h.texture(&desc).unwrap();
    h.register_texture(key, &texture);
    h.register_depth(key, &texture, &desc);
    let srv = View::Shader(key, Some([41, 4, 0, 0, 0, 1, 0, 0, 0, 0]));
    let write = |index: u64, view| {
        crate::descriptor::record(start + index * 32, view);
        assert!(h.mirror(start + index * 32).unwrap());
    };
    write(0, srv.clone()); write(1, srv.clone());
    assert_eq!(heap.heap.borrow()[&0], (*texture, u32::MAX));
    write(2, View::Unordered(key, 0, Some([41, 4, 0, 0, 0, 0, 0, 0, 0, 0])));
    write(3, srv);
    assert_ne!(heap.heap.borrow()[&3], heap.heap.borrow()[&0]);
    assert_eq!(heap.heap.borrow()[&3], heap.heap.borrow()[&2]);
    assert_eq!(heap.twin_srvs.borrow().get(&3), Some(&key));
    h.register_texture(key, &Rc::new(55));
    let image = View::Shader(key, Some([28, 4, 0, 0, 2, 4, 0, 0, 0, 0]));
    write(4, image.clone());
    h.set_dropped_levels(key, 2, 3);
    write(5, image);
    assert_eq!(h.backend.capture_descriptor(&heap.heap, 5).2.as_deref(),
        Some(&EntryDetail::Image(Some(ViewDesc { format: Format::Rgba8Unorm, mips: (0, 3), slices: (0, usize::MAX) }), 0)));
    crate::descriptor::forget(key);
}

#[test]
fn mirror_cache_bound_evicts_and_redecodes_without_changing_entries() {
    let mut h = NativeHost::new(Recording::default());
    h.fast_buffers = false;
    h.mirror_cache_limit = super::mirror_cache::CACHE_LIMIT;
    h.cache_mirrors = true;
    let records = [0u32; 16];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 2).unwrap();
    h.memory.mapped(0x7000_0000, 65536, &PoolInfo { id: 1, base: 0x3000_0000, len: 65536, memory_entry: None }, 0);
    let limit = super::mirror_cache::CACHE_LIMIT;
    for i in 0..=limit {
        crate::descriptor::record(start, crate::descriptor::View::Constant(0x7000_0000 + (i * 4) as u64, 4));
        assert!(h.mirror(start).unwrap());
        assert!(h.mirror_cache.borrow().len() <= limit);
    }
    assert_eq!(h.mirror_cache.borrow().len(), 1);
    let hits = h.backend.replays.get();
    for i in 0..2 {
        crate::descriptor::record(start + i * 32, crate::descriptor::View::Constant(0x7000_0000, 4));
        assert!(h.mirror(start + i * 32).unwrap());
    }
    assert_eq!(h.backend.replays.get(), hits + 1, "evicted view is decoded then reusable again");
    assert_eq!(heap.heap.borrow()[&1], (0x3000_0000, 4));
}

#[test]
fn a_twin_created_during_a_table_scan_invalidates_later_shared_stamps() {
    use crate::descriptor::View;
    let mut h = NativeHost::new(Recording::default());
    h.cache_mirrors = true;
    let records = [0u32; 24];
    let start = records.as_ptr() as u64;
    let key = start as usize;
    let heap = h.heap(HeapKind::Resource, start, 3).unwrap();
    let desc = TextureDesc::d2(Format::D32Float, 8, 8);
    let texture = h.texture(&desc).unwrap();
    h.register_texture(key, &texture);
    h.register_depth(key, &texture, &desc);
    for i in 1..3 {
        crate::descriptor::record(start + i * 32, View::Shader(key, Some([41, 4, 0, 0, 0, 1, 0, 0, 0, 0])));
        assert!(h.mirror(start + i * 32).unwrap());
    }
    assert_eq!(heap.mirrored.borrow().get(&1), heap.mirrored.borrow().get(&2));
    crate::descriptor::record(start, View::Unordered(key, 0, Some([41, 4, 0, 0, 0, 0, 0, 0, 0, 0])));
    let _scope = h.clean_tables.begin(true);
    h.remirror_tables(&[Some(Root::Table(heap.clone(), 0))], None);
    assert_ne!(heap.heap.borrow()[&0], (*texture, u32::MAX));
    for i in 1..3 {
        assert_eq!(heap.heap.borrow()[&i], heap.heap.borrow()[&0]);
        assert_eq!(heap.twin_srvs.borrow().get(&i), Some(&key));
    }
    assert!(!table_clean_now(&h, &heap, 0, 3), "a twin changed the epoch within the scan");
    h.remirror_tables(&[Some(Root::Table(heap.clone(), 0))], None);
    assert!(table_clean_now(&h, &heap, 0, 3));
    crate::descriptor::forget(key);
}

fn table_clean_now(h: &NativeHost<Recording>, heap: &NativeHeap<Recording>, first: usize, end: usize) -> bool {
    h.clean_tables.contains((heap.start, first, end), table_clean::Dependencies {
        memory: h.memory.generation(), resources: h.resources_epoch.get(), spans: h.spans_made.get(),
    })
}

#[test]
fn clean_tables_follow_writes_dependencies_heap_lifetime_and_plain_stores() {
    use crate::descriptor::{View, record};
    let h = NativeHost::new(Recording::default());
    let records = [0u32; 256];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 32).unwrap();
    let key = start as usize;
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.register_texture(key, &Rc::new(11));
    record(start, View::Shader(key, None));
    {
        let _scope = h.clean_tables.begin(true);
        h.remirror_tables(&roots, None);
        assert!(table_clean_now(&h, &heap, 0, 16));
        let writes = h.backend.writes.get();
        h.remirror_tables(&roots, None);
        assert_eq!(h.backend.writes.get(), writes);
        // Reported writes in a disjoint range do not discard this table's mark.
        record(start + 16 * 32, View::Constant(0, 4));
        h.mirror(start + 16 * 32).unwrap();
        assert!(table_clean_now(&h, &heap, 0, 16));
        record(start + 15 * 32, View::Constant(0, 16));
        h.mirror(start + 15 * 32).unwrap();
        assert!(!table_clean_now(&h, &heap, 0, 16));
        h.remirror_tables(&roots, None);
        assert!(table_clean_now(&h, &heap, 0, 16));
        // Any resource mutation invalidates the table; per-entry stamps still avoid rewrites
        // for unrelated resources when the table is rechecked.
        h.register_buffer(key + 1, 0, 16);
        assert!(!table_clean_now(&h, &heap, 0, 16));
        h.remirror_tables(&roots, None);
        assert_eq!(heap.heap.borrow()[&0], (11, u32::MAX));
        h.register_texture(key, &Rc::new(22));
        assert!(!table_clean_now(&h, &heap, 0, 16));
        h.remirror_tables(&roots, None);
        assert_eq!(heap.heap.borrow()[&0], (22, u32::MAX));
        h.set_dropped_levels(key, 1, 2);
        assert!(!table_clean_now(&h, &heap, 0, 16));
        h.remirror_tables(&roots, None);
        h.release_resource(key);
        assert!(!table_clean_now(&h, &heap, 0, 16));
        h.remirror_tables(&roots, None);
        h.register_texture(key, &Rc::new(33));
        h.remirror_tables(&roots, None);
        assert_eq!(heap.heap.borrow()[&0], (33, u32::MAX));
        h.evict_texture(key);
        assert!(!table_clean_now(&h, &heap, 0, 16));
        h.remirror_tables(&roots, None);
        h.register_texture(key, &Rc::new(44));
        h.remirror_tables(&roots, None);
        assert_eq!(heap.heap.borrow()[&0], (44, u32::MAX));
        // An independent span identity change invalidates marks even without a new pool epoch.
        h.spans_made.set(h.spans_made.get() + 1);
        assert!(!table_clean_now(&h, &heap, 0, 16));
        h.remirror_tables(&roots, None);
        h.drop_heap(start);
        assert!(!table_clean_now(&h, &heap, 0, 16));
    }
    let replacement = h.heap(HeapKind::Resource, start, 32).unwrap();
    let roots = [Some(Root::Table(replacement.clone(), 0))];
    {
        let _scope = h.clean_tables.begin(true);
        h.remirror_tables(&roots, None);
        assert_eq!(replacement.heap.borrow()[&0], (44, u32::MAX));
    }
    // No host mirror notification: the next preparation must see a plain title store.
    record(start, View::Constant(0, 8));
    let _scope = h.clean_tables.begin(true);
    assert!(!table_clean_now(&h, &replacement, 0, 16));
    h.remirror_tables(&roots, None);
    assert_eq!(replacement.heap.borrow()[&0].1, 0);
    crate::descriptor::forget(key);
}

#[test]
fn clean_tables_retry_mapping_failures_pending_resources_and_span_refresh() {
    use crate::descriptor::{View, record};
    let h = NativeHost::new(Recording::default());
    let records = [0u32; 128];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 16).unwrap();
    let roots = [Some(Root::Table(heap.clone(), 0))];
    let a = PoolInfo { id: 1, base: 0x3000_0000, len: 65536, memory_entry: None };
    let b = PoolInfo { id: 2, base: 0x4000_0000, ..a };
    record(start, View::Constant(0x7000_0010, 64));
    let _scope = h.clean_tables.begin(true);
    h.remirror_tables(&roots, None);
    assert!(!table_clean_now(&h, &heap, 0, 16), "failed mirror cannot certify a table");
    h.memory.mapped(0x7000_0000, 65536, &a, 0);
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&0], (a.base + 16, 64));
    assert!(table_clean_now(&h, &heap, 0, 16));
    h.memory.mapped(0x7000_0000, 65536, &b, 0);
    assert!(!table_clean_now(&h, &heap, 0, 16));
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&0], (b.base + 16, 64));
    h.memory.unmapped(0x7000_0000, 65536);
    assert!(!table_clean_now(&h, &heap, 0, 16));
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&0].1, 0);
    h.memory.mapped(0x7000_0000, 65536, &a, 0);
    h.remirror_tables(&roots, None);
    h.memory.pool_released(&a);
    assert!(!table_clean_now(&h, &heap, 0, 16));
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&0].1, 0);
    h.memory.mapped(0x7000_0000, 65536, &a, 0);
    h.memory.mapped(0x7001_0000, 65536, &b, 0);
    record(start, View::Constant(0x7000_fff0, 64));
    h.mirror(start).unwrap();
    h.remirror_tables(&roots, None);
    let before = heap.spanned.borrow()[&0];
    h.memory.mapped(0x7001_0000, 65536, &a, 0);
    h.refresh_spans();
    assert!(!table_clean_now(&h, &heap, 0, 16));
    h.remirror_tables(&roots, None);
    assert_ne!(heap.spanned.borrow()[&0], before);
    let key = start as usize;
    record(start + 32, View::Shader(key, None));
    assert!(!h.mirror(start + 32).unwrap());
    h.remirror_tables(&roots, None);
    assert!(heap.pending_mirror.borrow().contains_key(&1));
    h.register_texture(key, &Rc::new(99));
    assert!(!table_clean_now(&h, &heap, 0, 16));
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&1], (99, u32::MAX));
    crate::descriptor::forget(key);
}

#[test]
fn pool_fast_lookup_matches_reference_for_boundaries_gaps_and_replacements() {
    let h = NativeHost::new(Recording::default());
    let a = PoolInfo { id: 1, base: 0x3000_0000, len: 4 * 65536, memory_entry: None };
    let b = PoolInfo { id: 2, base: 0x4000_0000, ..a };
    for (at, pool, offset) in [(0x7000_0000, &a, 0), (0x7001_0000, &a, 65536), (0x7003_0000, &b, 0)] {
        h.memory.mapped(at, 65536, pool, offset);
    }
    for _ in 0..2 {
        for (at, size) in [(0x7000_0000, 0), (0x7000_ffff, 1), (0x7000_ffff, 2),
            (0x7001_0000, 65536), (0x7002_0000, 1), (0x7003_0001, 4), (u64::MAX, 2)] {
            if let Some((_, ptr)) = h.resolve_pool_cached(at, size).unwrap() {
                assert_eq!(Some(ptr), h.memory.resolve(at, size).map(|(_, p)| p));
            }
        }
        h.memory.mapped(0x7000_0000, 65536, &b, 65536);
    }
    assert_eq!(h.resolve_view(0x7000_0010, 4).unwrap().1 as usize, b.base + 65536 + 16);
    h.memory.pool_released(&b);
    assert!(h.resolve_pool_cached(0x7000_0010, 4).unwrap().is_none());
}

#[test]
#[ignore = "CPU benchmark: run --release --ignored --nocapture"]
fn unique_buffer_mirror_benchmark() {
    use std::{hint::black_box, time::Instant};
    const N: usize = 50_000;
    let records = vec![0u32; N * 8];
    let start = records.as_ptr() as u64;
    for i in 0..N { crate::descriptor::record(start + (i * 32) as u64, crate::descriptor::View::Constant(0x7000_0000 + (i * 32) as u64, 32)); }
    for fast in [false, true] {
        let mut samples = Vec::new();
        let mut pool_samples = Vec::new();
        for _ in 0..7 {
            let mut h = NativeHost::new(Recording::default());
            h.fast_buffers = fast;
            h.cache_mirrors = true;
            h.memory.mapped(0x7000_0000, N * 32, &PoolInfo { id: 1, base: 0x3000_0000, len: N * 32, memory_entry: None }, 0);
            let heap = h.heap(HeapKind::Resource, start, N).unwrap();
            let memory = h.memory.generation();
            let now = Instant::now();
            for i in 0..N {
                let record = record_at(start + (i * 32) as u64);
                h.mirror_record(&heap, i, black_box(record), memory, 0).unwrap();
            }
            samples.push(now.elapsed().as_nanos() as f64 / N as f64);
            for i in 0..N { assert_eq!(heap.heap.borrow()[&i], (0x3000_0000 + i * 32, 32)); }
            let now = Instant::now();
            for i in 0..N {
                let (buffer, ptr, _) = h.resolve_view(black_box(0x7000_0000 + (i * 32) as u64), 32).unwrap();
                h.backend.write_buffer(&heap.heap, 0, &buffer, ptr, 32);
            }
            pool_samples.push(now.elapsed().as_nanos() as f64 / N as f64);
        }
        samples.sort_by(f64::total_cmp);
        pool_samples.sort_by(f64::total_cmp);
        println!("unique buffers fast={fast}: full mirror {:.1} ns; pool + recording write {:.1} ns; samples {samples:.1?}", samples[3], pool_samples[3]);
    }
    // The same records through the device-write entry point, as the title's view writes and
    // descriptor copies reach the host: record read, stamp check, mirror.
    let mut samples = Vec::new();
    for _ in 0..7 {
        let h = NativeHost::new(Recording::default());
        h.memory.mapped(0x7000_0000, N * 32, &PoolInfo { id: 1, base: 0x3000_0000, len: N * 32, memory_entry: None }, 0);
        let heap = h.heap(HeapKind::Resource, start, N).unwrap();
        let now = Instant::now();
        for i in 0..N {
            h.mirror(black_box(start + (i * 32) as u64)).unwrap();
        }
        samples.push(now.elapsed().as_nanos() as f64 / N as f64);
        for i in 0..N { assert_eq!(heap.heap.borrow()[&i], (0x3000_0000 + i * 32, 32)); }
    }
    samples.sort_by(f64::total_cmp);
    println!("unique buffers through mirror(handle): {:.1} ns; samples {samples:.1?}", samples[3]);
}

#[cfg(target_os = "macos")]
#[test]
fn plain_root_updates_detect_the_only_shared_reference_case() {
    let h = NativeHost::new(Recording::default());
    let heap = h.heap(HeapKind::Resource, 0, 1).unwrap();
    let mut list = NativeList::<Recording>::default();
    for compute in [false, true] {
        assert!(list.plain_root(compute, 0));
        list.set_root_constants_at(compute, 0, 0, &[1, 2]);
        assert!(list.plain_root(compute, 0));
        if compute { list.set_compute_root_table(0, &heap, 0); }
        else { list.set_graphics_root_table(0, &heap, 0); }
        let refs = Rc::strong_count(&heap);
        assert!(!list.plain_root(compute, 0));
        assert_eq!(Rc::strong_count(&heap), refs);
    }
}

#[test]
fn cache_miss_reasons_distinguish_capacity_dependency_and_disabled_paths() {
    use mirror_cache::Miss;
    let mut h = NativeHost::new(Recording::default());
    h.cache_mirrors = true;
    h.fast_buffers = false;
    h.trace_mirrors = true;
    h.mirror_cache_limit = 2;
    let records = [0u32; 24];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 3).unwrap();
    let pool = PoolInfo { id: 1, base: 0x3000_0000, len: 65536, memory_entry: None };
    h.memory.mapped(0x7000_0000, 65536, &pool, 0);
    let generation = h.memory.generation();
    assert_eq!(h.cached_mirror(&heap, 0, [0x55; 12], generation, 0), Err(Miss::Cold));
    assert_eq!(h.miss_counter(Miss::Cold, None, None, Nothing::Foreign).reason.get(), Miss::Uncacheable);
    assert_eq!(h.miss_counter(Miss::Stale, None, None, Nothing::Foreign).reason.get(), Miss::Stale);
    assert_eq!(h.miss_counter(Miss::Disabled, None, None, Nothing::Foreign).reason.get(), Miss::Disabled);
    for i in 0..3 {
        crate::descriptor::record(start + i * 32, crate::descriptor::View::Constant(0x7000_0000 + i * 256, 256));
        h.mirror(start + i * 32).unwrap();
    }
    assert_eq!(h.cached_mirror(&heap, 0, record_at(start), generation, 0), Err(Miss::Evicted));
    h.memory.unmapped(0x7000_0000, 65536);
    assert_eq!(h.cached_mirror(&heap, 2, record_at(start + 64), h.memory.generation(), 0), Err(Miss::Stale));
    h.cache_mirrors = false;
    assert_eq!(h.cached_mirror(&heap, 0, record_at(start), generation, 0), Err(Miss::Disabled));
}

#[test]
fn raw_buffer_mirrors_reuse_unique_stamps_without_retaining_cache_snapshots() {
    let mut h = NativeHost::new(Recording::default());
    h.fast_buffers = true;
    h.cache_mirrors = true;
    let records = [0u32; 8];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 1).unwrap();
    h.memory.mapped(0x7000_0000, 65536, &PoolInfo { id: 1, base: 0x3000_0000, len: 65536, memory_entry: None }, 0);
    crate::descriptor::record(start, crate::descriptor::View::Constant(0x7000_0000, 256));
    h.mirror(start).unwrap();
    let stamp = heap.mirrored.borrow().get(&0).cloned().unwrap();
    crate::descriptor::record(start, crate::descriptor::View::Constant(0x7000_0100, 256));
    h.mirror(start).unwrap();
    // The slot's stamp is rewritten in place: one stamp, of the new record.
    assert_eq!(heap.mirrored.borrow().len(), 1);
    assert_ne!(Some(&stamp), heap.mirrored.borrow().get(&0));
    assert_eq!(heap.mirrored.borrow().get(&0).unwrap().record, record_at(start));
    assert!(h.mirror_cache.borrow().is_empty());
    assert_eq!(heap.heap.borrow()[&0], (0x3000_0100, 256));
}

/// A description placed again at its address over other bytes (the title's allocator gave the
/// block to the next texture of the same description) has its copy read again at the next
/// submission: work that reads it once right after the placement sees the new texture, not the
/// one the block held before. A placement over the same bytes reads nothing again.
#[test]
fn a_description_placed_again_over_other_bytes_is_read_again_at_once() {
    let mut f = CoveredCopies::new();
    (f.title.covered_clear, f.title.reread_placed) = (false, true);
    let key = f.place(0, 32);
    let _heap = f.show(key);
    f.title.sync_dirty();
    assert_eq!(f.title.uploads_of(key), 1);
    assert_eq!(f.place(0, 32), key);
    assert!(!f.title.unloaded.contains(&key), "the same bytes: nothing to read again");
    f.memory[..1024].fill(0x55);
    assert_eq!(f.place(0, 32), key);
    assert!(f.title.unloaded.contains(&key), "other bytes: read at the next submission");
    f.title.sync_dirty();
    assert_eq!(f.title.uploads_of(key), 2);
    assert!(f.title.host.knows(key) && f.title.texture_watched(key));
}
