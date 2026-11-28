# Third host-lock pass: a fresh heap entry

Follows [the second pass](host-lock-second-pass.md). In a world frame the title writes
about 94,000 fresh ring-heap entries, and each cost about 1.1 us to mirror under the host
lock. This pass cuts the host's share of a fresh buffer or constant-buffer view. Changes are
in `victory-gpu` and `victory-metal` only. CPU benchmarks and Metal unit tests were run; the
game was not, so the frame-time effect is still to be measured.

## What changed

**Per-entry state is dense.** A native heap kept seven maps keyed by entry index (stamps,
pending retries, spans, title views, three twin maps, shown textures). They are now
`native::entries::Entries`: lazily made pages of 256 slots (`descriptor::paged::Paged`).
Stamps are stored in the slot, not behind an `Rc`. Both forms iterate in index order, so
nothing depends on which one holds the state. Indices past the heap's end (none today) fall
back to a small ordered map instead of panicking.

**Metal heap ownership is a slab** (`victory-metal/src/dense.rs`). Each entry holds a small
id into a per-heap list of allocations; each allocation holds its Metal object, its entry
count, and whether the residency set held it at the last commit. A fresh entry that names a
buffer the heap already holds (the pool of every view into it) is an index and an increment:
no hash, no retain, no allocation. Four recent ids are checked before the one hash lookup.

The residency rules of `Heap::commit_set` are kept by construction:

- `reconcile` compares each allocation's committed membership with the desired one
  (`count != 0`) and issues at most one add or one remove for it, never both, never an add
  of a member or a remove of a non-member.
- An allocation with no entries keeps its object in the slab until `finish_commit`, which
  runs only after the set is published (or had no difference). Naming it again before the
  commit cancels the removal with the object the heap kept.
- Views of a retired allocation wait with it; without a set, objects go at once, as before.

**Quick mirrors.** `NativeHost::classify` decides what a record is from its view and the
registered resources, writing nothing. A table scan now reads, checks and classifies every
record in one pass with the stamps, pending retries and resources borrowed once and one lock
of the view table (released before anything is written). Only changed entries go to a second
pass. `mirror_quick` then handles the two kinds a ring heap's fresh entries mostly are:

- a plain buffer view (constant buffer, raw or structured view that needs no typed view)
  that one pool mapping holds;
- a record that is no view (see the miss kinds below).

Everything else (images, samplers, null and typed buffer views, spans, wraps, unmapped
memory, a record with a cached snapshot) takes the general path unchanged. If a mirror
changes the resource epoch mid-table (a UAV creating a depth twin), the rest of the table is
checked by the exact per-entry loop, as before.

**Pool lookup.** The last-pool slot is a four-entry most-recent list per pool-table
generation. A hit needs no search and no buffer hash lookup. A page-indexed table was not
built: the list already removes the search for views that alternate over a few pools, and
`pool_last` / `pool_search` will show whether more is needed.

**Record reads.** Descriptor records are read with plain loads when the heap's memory is
known readable (`NativeHost::records_readable`):

- memory the heap object owns (the title did not place the heap): checked readable once at
  creation, valid until the title releases the heap, which reaches `drop_heap` under the
  host lock before the memory is freed;
- memory the title placed, or of unknown origin: the whole heap range is asked of the kernel
  once per submission preparation (`mach_vm_region`), instead of one kernel read per table;
- a released heap (still bound by a recorded list) is never loaded from: kernel reads only.

**Smaller items.** `Buffer` caches `gpuAddress` (one message send per buffer, not per
entry). A device mirror no longer copies the whole view to decide whether a texture must be
realized first. The twin-map lookups per bound table no longer allocate.

## Counters (`VICTORY_TRACE_FRAME_TIME=1`)

- `miss_unknown` is split. `miss_empty`: a zeroed record over a mirrored entry.
  `miss_foreign`: bytes that are not a view record. `miss_dead_view`: a record of a view
  whose resource was released. `miss_unregistered`: a view of a resource object the host has
  not registered. `miss_unknown` keeps only views the host does not decode.
- `mirror_quick`: entries the quick path wrote or settled. `mirrors` counts them too;
  `mirror` (ms) times only general-path mirrors, so read quick-path time from `remirror` and
  `device_mirror`.
- `record_loads`: table scans read with plain loads. `region_queries`: kernel readability
  queries. `safe_reads` should fall by about the number of `record_loads`.
- `pool_last` now counts hits anywhere in the four-entry list.
- `hold_other` is split by caller. `hold_mirror`: device view writes and descriptor copies.
  `hold_fence`: queue and fence signal, wait and poll. `hold_resource`: buffer, texture,
  pipeline, queue, fence and heap creation and release. `hold_other` is the rest.
- `timer_pump` and `timer_fences`: the two parts of a timer tick's hold (moving held work
  along; raising the title's fence values).

## Switches

All default on; `0` restores the reference.

| Switch | Reference behavior |
| --- | --- |
| `VICTORY_DENSE_ENTRIES` | Ordered maps for per-entry state, the old Metal ownership maps, no quick mirrors, one remembered pool, kernel record reads |
| `VICTORY_RECORD_LOADS` | Kernel record reads only (everything else of this pass stays on) |

`VICTORY_MIRROR_CACHE`, `VICTORY_BUFFER_FAST` and `VICTORY_TABLE_CLEAN` work as before.
Quick mirrors also need `VICTORY_BUFFER_FAST` on. The dense reference uses ordered maps, so
it is slower than the previous build: use it to compare results, and compare frame times
against the previous build.

## CPU measurements

Release x86_64 under Rosetta, tracing off, recording backend, median of seven samples; the
machine was busy, so runs vary by about 10%. "Before" is the starting tree measured in the
same session. The recording backend's own entry write (a hash-map insert) is about 50 ns of
every fresh-entry number, before and after.

| Benchmark | Before | After | Unit |
| --- | ---: | ---: | --- |
| Fresh distinct raw buffer views, 16-entry tables | 194.0 | 122 - 147 | ns/fresh entry |
| Fresh distinct constant buffer views, 16-entry tables | 202.1 | 124 - 141 | ns/fresh entry |
| Mixed table, 12 fresh + 80 clean | 3381 | 2040 - 2380 | ns/table |
| Repeated-view ring (100,000 slots, 2,000 views) | 255.8 | 122 - 129 | ns/mirror |
| Distinct buffer records through `mirror(handle)` | 740.7 | 110 - 117 | ns/mirror |
| Repeated clean 16-entry table | 32.3 | 30 - 31 | ns/check |

The device-path "before" is the second pass's 670 to 740 ns case: one kernel read plus the
general path per record. The repeated-view ring regressed in the second pass (190 to 232 ns)
because fresh slots each allocated a stamp; stamps are now stored in the slot.

Run with `cargo test --release --target x86_64-apple-darwin -p victory-gpu --lib --
benchmark --ignored --nocapture --test-threads=1`.

## Tests

- `dense::tests`: 200,000 random writes against the old maps (`residency::replace` /
  `reconcile`): same counts and the same adds and removes at every commit, each allocation
  touched at most once, never an add of a member or a remove of a non-member.
- `gfx::heap_set_tests::dense_ownership_matches_the_reference_maps_through_random_writes_and_commits`:
  the same on real Metal objects and residency sets. The existing residency tests run in
  both modes.
- `mirror_tests::dense_quick_mirrors_match_the_reference_host_through_random_writes`: a
  dense host and a reference host over the same records, 1,500 random steps (all view
  kinds, pool remaps, resource changes, device and table paths, stride and address
  tables): same backend entries, writes, stamps and per-entry notes after every step.
- `record_loads_follow_the_heaps_memory`, `the_pool_list_resolves_alternating_pools_and_follows_the_table`,
  `entries::tests`.

## Risks and limits

- **Placed heap memory.** For a heap in title-placed memory, readability is asked once per
  submission preparation. If the title unmapped part of a live, bound heap inside that
  window, a plain load would fault where the kernel read only failed. D3D12 does not allow
  that, and a released heap is never loaded from, but it is the one new way to crash:
  `VICTORY_RECORD_LOADS=0` rules it out.
- **Stale snapshots.** A quick buffer mirror does not look the record up in the decode
  cache, so a stale snapshot of the same record is dropped later (at the bound or on
  release) instead of at once. It can never replay: its pool generation is past.
- **Memory.** Stamps take 56 bytes per touched entry, in pages of 256 entries that are kept
  for the heap's life: about 117 MB if a two-million-entry heap is touched everywhere.
- The Metal cost per entry (the three words, the slab) is not in the CPU benchmarks; only
  the Metal unit tests cover it. `hold_timer` itself was not reduced, only broken down.
