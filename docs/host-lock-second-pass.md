# Second host-lock pass

CPU work against `native-up`, with no Metal device or game run. Changes stay in
`victory-gpu`, `victory-metal`, and documentation. The coordinator must compare
identical world-frame windows before interpreting these microbenchmarks as FPS gains.

## Buffer mirrors and cache accounting

The host remembers its last pool mapping and buffer. A matching mapping generation
and containing range resolve without the pool mutex or a buffer hash lookup. Other
single-mapping views binary-search the existing immutable mapping snapshot. Views
crossing mappings, holes, spans, and host-owned buffers still use the established
resolver. Pool generations are published with release/acquire atomics after every
observer mutation; snapshot construction still takes the pool mutex. Releases clear
the remembered wrap and existing ownership. Unique slot stamps reuse their allocation.

Non-null raw buffer views resolved directly from a pool no longer retain decoded
backend snapshots. Nulls, images, samplers, spans/aliases with expensive validation,
and buffer views eligible for lazy typed promotion still use the cache.
Metal's same-allocation raw-buffer path writes only the three entry words. Fresh
slots share a weakly indexed ownership wrapper already held by other raw slots.
Reference counts still describe every slot; transitions still retain removed
allocations through reconciliation. `Heap::commit_set` is unchanged.

`VICTORY_TRACE_FRAME_TIME=1` adds:

- `miss_constant`, `miss_buffer`, `miss_typed`, `miss_image`, `miss_sampler`,
  `miss_null`, `miss_unknown`: decode-cache misses by kind.
- `miss_cold` (never cached), `miss_evicted` (capacity eviction), `miss_stale`
  (dependency invalidation), `miss_uncacheable` (raw-buffer bypass or no decodable
  view), `miss_disabled`: reasons for those misses. These are two marginal
  histograms over the same misses, not counts of backend entry writes.
- `cache_evictions`: snapshots discarded at the bound; `cache_entries_max`:
  maximum resident snapshot count in the interval; `cache_working`: distinct
  successfully cached/replayed records used in the title frame.
- `pool_last`, `pool_search`: successful remembered/binary-search pool resolutions;
  `buffer_fast`: same-allocation raw writes; `buffer_retained`: shared ownership
  reused for another slot.

The default snapshot bound remains 8,192. `VICTORY_MIRROR_CACHE_LIMIT=N` changes it
for a measured working set. No game trace here proves that live *cacheable* records
still exceed the bound after removing raw buffers. Compare `miss_evicted` with
`cache_working`, and size from that frame measurement. For the original buffer-miss
distribution, also run with `VICTORY_BUFFER_FAST=0`; otherwise those buffers appear
under `miss_uncacheable`. Exact eviction/stale attribution retains diagnostic record
keys for the run, but never their allocations; tracing off retains no such history.

## Clean tables

`(heap address, first entry, exclusive end)` identifies a clean table during one
`NativeHost::execute` preparation. Repeated checks compare its pool, resource, and
span epochs instead of reading title memory or walking entries. `table_clean`,
`table_scan`, and `table_entries_skipped` report the result.

Marks expire at every preparation boundary, including error returns. This is
necessary: the title copies descriptor records with plain stores, with no device
notification. Each submission's first use therefore scans the records again.
Descriptors used by submitted work must remain stable during its use, the same
descriptor-write exclusion required by the existing backend. Held queue segments
get separate preparation scopes.

Reported writes invalidate overlapping ranges; heap destruction invalidates every
range of that heap. Pool creation/remap/unmap/release, resource registration,
replacement/release/eviction, mip policy, depth-twin creation, and span replacement
invalidate dependency comparisons. A failed read or mirror does not establish a
clean mark. A dependency change *during* a scan, such as first twin creation, also
prevents marking that scan clean. Pending unrealized resources keep their existing
retry-on-resource-change behavior. Typed promotion is still done after remirroring.

CPU tests exercise range boundaries, disjoint writes, heap reuse, plain stores at
the next submission, mapping failures/recovery, resource changes, span refresh,
first twin creation during a scan, and the disabled path.

## Lock scope, texture probes, and timer

Each native `CopyDescriptors`/`CopyDescriptorsSimple` call mirrors all copied
destinations under one acquisition. `mirror_batches` counts batches and
`device_mirrors` still counts descriptors. `device_mirror` measures each batch's
whole duration including its one wait. Byte copies already run without `TITLE`.
Mirrors remain eager: heap-indexing shaders and auxiliary heap tables can observe
entries outside bound tables, so deferring arbitrary writes is not justified.

Pipeline/topology/viewport/scissor/blend/stencil/index/vertex setters, scalar root
updates, and plain buffer-copy/value-write recording use the list mutex alone.
`list_local` counts these calls. Replacing a root table can drop a shared `Rc` and
falls back to the ordered global-plus-list locks (`list_fallback`). Draw preparation,
table/heap binding, list reset/destruction, and fence queries retain their shared
state locks. Fence queries also perform completion/copy-back work.

Texture probes snapshot CPU metadata under `TITLE`, read/hash title memory outside
it, then apply only still-current results. Serial, last-upload hash, upload count,
and GPU ownership reject results invalidated during the unlocked interval.
`sync_probe` times the unlocked work; `probe_reads` counts reads and `probe_stale`
counts rejected results. The hot and blank working sets are capped at 1,024
snapshots each per submission, with the existing separate 2 ms budgets, minimum
32 visits, whole-read budgets, and round-robin continuation. Unvisited keys remain
tracked by the background scanner; tracking is not discarded to impose the work
bound. Large sets can take additional submissions to visit. Whole reads, hashing,
and staging detile already ran outside `TITLE` and continue to do so. Final Metal
uploads and host bookkeeping still hold it. CPU tests compare probe policy, reject
release/replacement/upload/GPU-write races, and check bounded work and fairness.

Before unlocking, `TitleGuard` publishes whether held work, outstanding fence
signals, or copy-back work remain. Timer ticks with none of these obligations skip
`TITLE` (`timer_skipped`); `timer_ticks` still counts actual refreshes. Pending GPU
signals stay pending until completion is observed, and memory/fence waits continue
polling. A producer that arrives after an idle check is seen by the next tick.

## A/B switches

All changes default on. Each switch below set to `0` restores its reference path:

| Switch | Reference behavior |
| --- | --- |
| `VICTORY_BUFFER_FAST` | Original pool lookup, raw-buffer snapshots, Metal entry ownership path |
| `VICTORY_TABLE_CLEAN` | Scan every bound table at every remirror |
| `VICTORY_MIRROR_BATCH` | One acquisition per copied descriptor |
| `VICTORY_LIST_LOCAL` | All list recording takes `TITLE` |
| `VICTORY_SYNC_PROBES_OFFLOCK` | Probe reads under `TITLE` |
| `VICTORY_TIMER_IDLE` | Every timer tick acquires `TITLE` |
| `VICTORY_MIRROR_CACHE` | Disable decoded snapshot replay, including typed snapshots |

Cache-off retains the independent pool, table, lock, and Metal optimizations; use
the other switches too when comparing with the original implementation.

## CPU measurements and validation

Offline release x86_64 under Rosetta, tracing off, median of seven samples, no GPU.
The recording backend verifies final entries and cannot measure Objective-C or
Metal residency cost.

| Benchmark | Before/reference | After | Unit |
| --- | ---: | ---: | --- |
| Repeated 16-entry table | 637.5 | 29.0 | ns/check |
| 50,000 distinct raw buffer records | 947.0 | 671.5 | ns/mirror, including fault-safe record read |
| Pool resolution + recording-backend write | 48.8 | 24.7 | ns/view |
| Existing 100,000-slot / 2,000-view ring, paired reference/fast | 190.5 | 231.9 | ns/mirror |

The first baseline was measured on starting HEAD `38806503`. The same-build
`VICTORY_TABLE_CLEAN=0` comparison is 658.7 versus 29.0 ns/check. The distinct
buffer benchmark compares reference and optimized paths in the same binary; both
share the atomic generation read and release-generation check.

The repeated-view ring **regresses about 22%** in paired runs (median of three
seven-sample runs per mode, alternating mode order). Its initial starting-HEAD
baseline was 184.4 ns/mirror; timing varied across the session, so the paired
reference is the more useful comparison. Direct raw mirrors no longer share the
cache's decode work and stamp across fresh slots. Reusing stamps in overwritten
slots helps those slots, but fresh slots still allocate a stamp. The distinct-view
path gains about 29%; the 45,000 game buffer mirrors' repeat distribution will
determine the net result. Use `VICTORY_BUFFER_FAST=0` for the coordinator's A/B run.
These measurements do not establish the game's miss distribution or gains for
lock batching, texture sync, or timer work.

Run CPU benchmarks using `cargo test -j4 --offline --release --target
x86_64-apple-darwin -p victory-gpu native::host::mirror_tests -- --ignored
--nocapture --test-threads=1`. The real-Metal companion test
`raw_buffer_fast_path_matches_reference_ownership_and_residency` is compiled here
for the coordinator, along with the existing descriptor/residency regressions.

The required workspace clippy command currently fails on two existing
`collapsible_if` diagnostics in `victory-shader/src/emit/bitcode/execute.rs`
(lines 379 and 394). That crate is outside this task's editable scope. Strict
Windows-target clippy scoped to `victory-gpu` and `victory-metal` passes.

Validation completed offline with the pinned compiler already on PATH and the
local dependency cache (`CARGO_HOME=/tmp/victory-covered-cargo`, target directory
inside this tree): release x86_64 `victory` build; release x86_64 GPU/Metal test
compilation without execution; 69 selected CPU tests, including residency and
frame accounting; the 33 host mirror tests also with mirror cache, table clean,
and buffer fast paths disabled; three CPU benchmarks and the table-clean-off
comparison. No Metal-dependent test or game was run. Changes remain uncommitted.
