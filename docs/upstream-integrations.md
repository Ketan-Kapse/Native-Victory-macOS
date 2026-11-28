# Upstream integrations

What each merge of upstream `main` into this branch had to reconcile, newest last, and the
follow-ups to those merges. Moved out of [native-host.md](native-host.md), which the asset policy
limits to 512 KiB.

## Upstream integration at b998bb65

The current host uses upstream's shared texture copies, covered-placement release and byte
budget (`VICTORY_METAL_TEXTURE_BUDGET_MB`), including its R9G9B9E5 view over an R32_UINT object
(widened since: "Interior lighting after the merge" above).
The earlier `aliases`/`owns_placed` owner-selection mechanism, reverse-realization takeover,
placed overlap rule A, `VICTORY_PLACED_OVERLAP`, `VICTORY_PLACED_IDLE` and the separate
`texture memory:` line are retired. The historical notes below describe their experiments;
upstream's budget trace describes current retention. `forget_resource` retains heap entries
as upstream does; `release_resource` and budget eviction clear entries at their own boundary.

The Metal heap keeps the complete deferred residency reconciliation: `alloc_counts` is desired
membership, `in_set` is committed membership, and removed allocations and views are retained
until `Heap::commit_set` commits their removal. There are no entry-level Metal adds or removes.
This preserves the kernel-safety invariant across remove-then-readd and add-then-remove before
one commit. Upstream's request-residency option and residency/commit traces wrap that ownership.
A shared residency gate lets shutdown wait for active commits, empty the sets exclusively, and
prevent subsequent heap or pool changes using their obsolete membership. Its CPU regression
checks that shutdown waits for a pending commit and rejects later changes.
`VICTORY_TRACE_FRAME_TIME` remains, including completion and commit accounting for split command
buffers. Unchanged-entry skips, per-resource mirror versions, fitted texture views and typed
buffer view caches remain; upstream's lazy typed-buffer promotion decides when a view is made.
The mirror stamp also preserves upstream's permanent null-buffer rule: unrelated pool or resource
changes do not rewrite an address-only null view. The existing CPU regression checks this.

### Replaying descriptor content (27 November 2026)

Fresh ring-heap indices commonly contain copies of the same 12-byte descriptor record. A
per-host cache now keeps up to 8,192 successfully decoded records, independent of their index.
A miss runs the existing mirror decoder and captures the backend result. A hit replays its
three Metal words and shared retained allocation/view, plus the stride/address words, span
and direct-title-wrap tracking, image ownership, depth/twin use, and lazy typed-promotion bits.
Typed promotion has a separate optional snapshot in the same cache record: raw-only shaders
still do not create texture-buffer views, and a fresh raw mirror still drops a slot's previous
typed view. Unsupported views and unsuccessful mirrors are never cached as successes.

The cache validates the same `MirrorStamp` as slot mirroring: exact record bytes, pool
mapping generation, and the named resource's version (global resource epoch for address-only
views, samplers and two-resource UAVs). Spanned entries also check their span version. Resource
release/forget drops cached ownership immediately; pool refresh discards old-generation
snapshots. Registration, first depth-twin creation (including aliases), depth registration,
and changed mip-drop policy invalidate affected resource versions. Table scans recheck the
resource epoch per slot, since an early UAV can create a twin needed by later SRVs in that
same scan. Fixed address-only null
buffers remain independent of mappings and resources. Overflow clears the cache; it does not
change any already-written heap entry. Each slot stores a pointer to a shared stamp in a dense
vector (only the written title prefix, never reserved backend slots), removing index hashing
and repeated full-stamp copies. Revalidation of one shared stamp benefits all its copies.

Metal retains one shared ownership record per prepared descriptor. Entry writes borrow the
slot references once, and skip allocation-count changes for any rewrite that keeps the same
allocation, including raw/typed transitions and offset/size changes. Changed allocations update
the desired counts through the CPU-tested transition helper. `Heap::commit_set` still reconciles
only final membership: no allocation is both added and removed between commits, and removed
allocations remain retained until commit. The retirement threshold and shutdown gate are unchanged.
Table scans pass their already-read record and resolved heap directly to the mirror, and
per-entry tracing uses the host/heap's cached enabled flag. Empty sparse tracking maps skip
removal lookups explicitly: `HashMap::remove` still computes a hash for an empty map. Inspection
of the host's generated x86_64 code and the CPU benchmark found these redundant hashes were a
substantial part of the remaining Rosetta hit cost.

Device writes remain eager. `glue::mirror` also realizes placed/object textures; heap-indexing
shaders, `view_strides`, and `view_addresses` can consume entries outside every bound table.
A table-only dirty scheme cannot prove those consumers see current data, so no per-write lock
is removed here.

CPU proof (no Metal device): `native::host::mirror_tests` compares uncached, cold-cache and
hit results for raw, typed/structured, image mip/slice/format views, nulls, console samplers,
depth twins and spans; checks resource/pool/span invalidation and the cache bound. Metal's
`residency::tests` compares 100,000 random writes with a full recount and checks committed
membership and retention through remove/re-add. `prepared_descriptors_preserve_words_and_random_residency`
is the corresponding real-Metal test for the coordinator, including sampler snapshots.

The ignored CPU `fresh_ring_mirror_benchmark` scans 100,000 fresh slots in tables of 16,
cycling over 2,000 mapped buffer views, including cold misses. It reports the median of seven
samples in ns/mirror and checks every resulting slot. Run it with and without
`VICTORY_MIRROR_CACHE=0` using `cargo test -j4 --offline --release -p victory-gpu
fresh_ring_mirror_benchmark -- --ignored --nocapture`. It measures the host path with a mock
Capture backend; actual Metal residency cost and the target of remirror below 50 ms/world
frame require the coordinator's GPU/game A/B run.

Measured here, tracing off, median of seven samples (ns/mirror):

| CPU target | Starting HEAD `02ca25fc` | New code, cache off | Cache enabled | Speedup over HEAD |
| --- | ---: | ---: | ---: | ---: |
| arm64 | 649.0 | 514.8 | 92.6 | 7.0x |
| x86_64 under Rosetta | 842.3 | 728.0 | 141.4 | 6.0x |

The same-build switch comparison is 5.6x on arm64 and 5.1x under Rosetta; cache-off retains
shared stamp storage and the Metal residency changes. Validation: offline release x86_64
`victory` build and GPU/Metal test compilation, strict locked Windows-target workspace clippy,
39 CPU tests (the 32 host tests also run with cache disabled), the benchmark on both targets,
and `git diff --check` pass. GPU execution and world-frame timing remain for the coordinator.

Upstream's `title_alias` handles a fully mapped range outside one pool run. Our `span`,
`refresh_spans`, `NativeHeap::spanned`/`unmapped`, retired buffers and `Stitch` handle partly
mapped ranges and pool-generation changes. Both refresh paths run before submission. The
upstream stride table at root SRV 255 and view-address table remain; image entries retain only
the component-kind bits needed by our integer image lowering, with zero buffer stride.

Depth textures use one plane-aware `Twin` and one copy planner, `twin_copies_before`.
`register_depth`, `twin_plane` and `BlitOp::CopyDepthPlane` retain depth and stencil transfers.
Colour reads follow an existing twin as upstream does, resource aliases share that twin,
and creating a twin invalidates earlier colour-read mirrors. The existing upstream
`VICTORY_DEPTH_TWINS=0` switch also applies. This is a combined path awaiting Metal execution;
the CPU recording-backend regression checks shared keys, colour reads and both plane copies.

Presentation uses upstream's sequence of planes and straight-alpha overlay blend, including
its resulting alpha. The local premultiplied compositor and forced-opaque result are retired.
The composite lifetime test now uses the upstream plane interface and checks those alpha
values; single-plane selection and queued presentation still keep their resource lifetimes.
HDR reporting and drawable timing remain.

Described mesh stages and guessed mesh programs still use `Compiler::mesh_pipeline` with
`convert_mesh_with`/`reflect_mesh`, object stages and the 32 KiB limit. Upstream owns off-thread
compilation, archive lookup/add, the writer thread, the 60-second compile deadline and remembered
compiler failures. Described rectangle lists keep their replay and fourth-corner expansion;
upstream's guessed vertex-ID rectangles use its own lowering. Mesh dispatches share one path
and keep per-target refusals and both parents' counters. The console pipeline fallback,
console flow, BVH-builder policy, flow/target diagnostics and earlier native title patches
remain wired. Translation keys now include `VICTORY_NO_CONSOLE_FLOW` and
`VICTORY_EXPORT_INTERLEAVED`; the console-flow trace changes no translation. A scalar mask
instruction of the wrong wave width keeps upstream's precise refusal instead of being retried
through console flow.

Tests specific to the removed overlap/idle policy are retired with that policy. The two local
alias tests for reverse-realization takeover and most-recent-owner selection are retired with
that mechanism; upstream's sharing, lifetime and budget tests remain. Local sharing tests now
read the upstream registry instead of expecting a separately owned placed texture. Their
read-only R32_UINT exclusion is removed because upstream deliberately allows that sharing.
The former forget-clears test now verifies upstream's retained-entry behavior and replacement
invalidation. The mapping-list generation test uses the existing capture backend, since its
assertions require no Metal device. No failing runtime test was removed to obtain a passing suite.

## Earlier upstream integration at 2a3b03d1 (historical)

Described stream mesh stages and guessed mesh producers share `Compiler::mesh_pipeline`.
Their full mesh and amplification reflection, output linking, consecutive output stores and
32 KiB refusal are retained. The owned compile descriptor and pending-job key include the
mesh and amplification bytes and root layouts. Upstream's `metal::within`,
`Archive::mesh` and `compile_mesh` now own off-frame compilation, archive lookup/add, the
60-second compile deadline and remembered compiler crashes. The separate
`mesh_pipeline_state` timeout, `VICTORY_PIPELINE_TIMEOUT_S` and per-run archive writer are
retired. A timeout alone is refused for the run; upstream remembers a reported compiler crash,
including one reported after the deadline. Existing per-run archive files are no longer read.

Dual-source conversion uses upstream's pipeline-level decision and converter option. Its
cache key replaces the port's separate dual-source key. Successful descriptor mirrors keep
their mapping/resource stamps; pending mirrors use upstream's retry gating, sharing the
registered-resource epoch. Span refresh, partially mapped spans, stride/address tables and
depth twins remain active. The draw line includes both sets of counters, including separate
refused direct/indirect mesh dispatches and recorded texture copies.

The preferred console-flow lowering remains for named lane accesses that the fast path can
translate without preserving the other half. Upstream's checked lockstep retry follows a
refusal, including when `VICTORY_NO_CONSOLE_FLOW=1`; a lockstep retry cannot re-enter the
preferred console-flow path. Both use the same wide first-lane helper. The half-float wave
sum uses upstream's reduction helper with no prefix. Console-flow selection uses section
boundaries without simulating their values, keeping upstream's single-simulation invariant.

Both parents' regression cases remain. The local
`lds_a_wave_shares_in_divergent_code_takes_console_flow` keeps its deliberate acceptance and
barrier assertions in place of upstream's refusal test for the same program: console flow
makes the exchange uniform. The known failing
`a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves` is unchanged. The sampler-ID
test uses upstream's independent `min` and `mag` fields, alongside upstream's expanded tests.
Mesh compile ownership and cache identity cover both mesh and object stages. Early output-limit
refusals now retain `MeshRefusal::OverLimit`, so dispatch counters keep the specific reason.

3D compute image checks (2026-11-27): `native/volume_tests.rs` assembles synthetic GFX10 programs with 4 x 4 x 4 threads in wave64 and 3 x 2 x 3 groups, checking every texel of 12 x 8 x 12 title textures in DXGI 24, 26, 42 and 10. Stores use xyz-dependent values; SRV and UAV loads and point `image_sample_lz` at texel centres read CPU-uploaded volumes into a raw buffer, independently of image stores, including mip-1 views and preservation of mip 0. Code inspection found that object TEXTURE3D views bypassed mip decoding (now shared with placed views), while image loads/stores always annotated float components even for integer storage (now selected from the existing per-view format-kind bits, with integer DXIL operations for UINT/SINT and distinct cached handles for each component type). R32_UINT sampling is included as a diagnostic value test but still uses the existing float-typed sample lowering; its result needs the coordinator's GPU run. The 3D DXIL resource kind, xyz lowering, Metal type/depth, and whole-mip backend transfers already preserve depth; no MSC texture-type override was found. FirstWSlice/WSize are not array layers: W sub-ranges remain unsupported and expose the full selected 3D mip, covered by descriptor tests rather than a misleading sub-range GPU test. The flow tracer previously allocated one slice for a whole-volume backend read, which was rejected and left a zero-filled buffer; compute snapshots also copied only one slice into a 2D texture. Late statistics and snapshots now include every level-0 depth slice, print `depth N`, use the full-volume slice pitch/copy depth, and skip volumes exceeding `1 << 22` texels. GPU execution is pending; the diagnostic zero buffer alone does not establish that the volume store failed.

Historical placed texture reclamation, retired at b998bb65 (2026-11-27): `texture memory:` beside the 64-frame draw counters reports tracked placed/object/target counts and `info.size` host-byte estimates, aliases, tracked placed title-range sum/union/overlap, pending descriptions, overlap/idle releases and bytes, idle re-realizations and bytes, and `ping_pong` (these estimates exclude Metal padding and in-flight retains).
`VICTORY_PLACED_OVERLAP=0` disables the default overlap-reuse guess: a different-start placement retires intersecting placed copies through a sorted coverage index, preserving owners and same-owner aliases; reverse displacement exempts that pair, with 4096 recent displacements retained and up to 4096 persistent exemptions (only exhaustion of the exemption budget stops overlap eviction), and forgetting a resource now nulls its retained heap views and drops mirror stamps, including the old same-address path.
`VICTORY_PLACED_IDLE=<frames>` optionally evicts non-target placed copies after more than that many presented frames without a bound draw/dispatch table reaching them (unset: off); cached per-resource frame cells avoid rehashing unchanged records, and queued bound tables restore evicted descriptions and upload their bytes before execution, including records copied without another device view call.

## Follow-ups to the integration at b998bb65

Merge verification follow-ups (2026-11-27, after `8fa0b15a`): the second follow-up restores
`4f35fa7c`'s pool/span-first resolution, retaining upstream's own-buffer path and direct title
wrap as a heap-view fallback. In this clone the parent has spans but no `title_alias` or the
two wrapped-view tests; those came from `upstream-main`. Their different contracts matter:
spans serve whole 64 KiB title pages, including sparse ranges, using the pools' own pointers;
direct wraps require the entire aligned range to be mapped and become entirely null on a gap.
`span` now enforces the whole-title-page contract (offsets and lengths, not the absolute pool
base, which needs only host-page alignment). Smaller mapped fragments can use a direct wrap
when fully mapped, but cannot turn into a partial span when one fragment is unmapped.

The wrapped-view failure was a partial-span fallback, not a missing generation increment:
`PoolTable::mapped` and `unmapped` both call `Inner::cut`, which advances the generation.
The rejected direct wrap fell through to a span retaining the first mapped fragment, explaining
the nonzero first 32 words after unmap. The two-pool crash path was different: the merge tried
`wrap_title_memory` on the fixture's logical VA `0x23_4560_0000` (only recorded in the pool
table, never mapped in the process), ahead of the parent's stitch of the real pool pointers.
CPU regressions reproduce both mistakes before the fix and check nulling, recovery, pool
pointer selection, remapping, sparse holes and retention of the replaced span afterward.
Metal's span/deallocator implementation is unchanged from the parent: heap entries retain the
MTLBuffer and the stitch is unmapped by its deallocator when Metal lets it go.

The previous follow-up's generation-wide signature cache is removed. `follow_title_pages`
again shares comparisons only within that pass, and mirroring keeps the signature returned by
`title_alias` instead of walking again to infer whether resolution returned a wrap. The walk
counter now also counts the fresh checks in `title_alias`: the three-copy test takes 1 walk
on an unrelated mapping change, and 4 after remapping under the copies (one shared comparison
and three wrap validations). Only that test's remap count changes; its exact data assertions
and the unmap test's all-null assertion remain intact.

Upstream `c4faf9c4478448c1fdf5b47195dee63701f9dbf2` introduced the descriptor-field refusal.
It expects the existing `view_addresses` root SRV `t1`: buffer address dwords 0 and 1's low
16 bits, image size/level/type fields, zeros elsewhere. Its commit records that all 44 table
readers in its boot corpus still translate: 40 mask the buffer high word to 16 bits, and none
reads image dwords through the table. Our host already supplies nonzero buffer high words;
a CPU test checks `0x12` then `0x23`, the root binding, retained table and mirrored updates.
There is no host-table or translator change to hide a missing buffer address.

The synthetic `CS_VIEW_ADDRESSES` used by both heap tests previously moved all of image
dwords 1-3 and buffer dword 1 to the output, including fields the table never supplied.
Before the unchanged moves/stores, the fixture adds these instruction/literal pairs:
`8711_FF11 C000_0000`, `8712_FF12 3FFF_FFFF`, `8713_FF13 F00F_FFFF`,
`8719_FF19 0000_FFFF` (`s_and_b32` on s17, s18, s19, s25 respectively). No original
instruction word is replaced. These keep image size fields and all 48 buffer address bits;
the two GPU tests retain every output, canary and remap assertion. CPU translation checks
also reconstruct the unmasked original and assert the exact image-dword-1 refusal, then
check the buffer stride/swizzle refusal if the high-word mask is removed.

The forgotten-texture test now pins upstream's retained-entry behavior: forget preserves the
entry and stamp, registration invalidates the resource version, and final release clears the
entry. CPU covered-placement and byte-budget tests check that shown copies survive and that
eviction drops the host Rc after their last entry is replaced. The corresponding Metal tests
also check host-copy lifetime. Metal's `heap_entry` replaces/removes the entry's retained resource
and decrements its allocation count. With residency sets, the last removed reference is retired
until `commit_set` reconciles residency (also bounded by `RETIRED_OBJECTS`); without a set it drops
immediately. GPU work may still retain it until completion. The integer-target test now expects
the specific mesh conversion refusal for float
outputs, retaining its attachment diagnostic and exact integer pixel assertions. Sampled alias
re-creation checks unloaded/unwatched first, then one upload and dirty watching after submission
sync, matching upstream's deferred scanner registration.

Second-follow-up verification: offline release x86_64 builds with four jobs; 14 CPU mirror
tests, 6 pool-table tests, 9 address-contract tests, the shared address-program translation
test and upstream's view-address-field refusal test pass (31 total). Asset policy passes.
The named `a_wrapped_view` / `copies_of_a_wrapped_view` tests use `Host<Metal>` in this
checkout, not `Capture`; attempting them reports `no Metal device` here. All GPU tests compile,
but their GPU data assertions still need the coordinator's Mac. Logs are outside the repository
in `/tmp/native-merge-second-followup`. The earlier follow-up's broader shader run and parent
comparison are in `/tmp/native-merge-followup` (425 passes, 3 ignored, the known
`a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves` failure on both trees).

#### Third follow-up: opaque resource keys in the view-address table

The Windows native-host test path also needs a working `native::safe_read`: `mirror`
now decodes the bytes captured by `record_at`, rather than calling `descriptor::view`
to read the slot directly. The old non-macOS stub returned false and left a zero record,
so the first mirror failed before decoding even a registered texture. On Windows,
`safe_read` delegates to `host::read_into` (`ReadProcessMemory`); macOS keeps its Mach
read. The existing default-texture-SRV test covers the backend write and subsequent
default-view replacements. Resource keys still resolve through native registrations;
the separate object guard below remains required for texture size words.

The two view-address GPU tests' `0x4EA9_0000` fault is a native-host bug, not a fixture
that must create COM resources. `Heap::sized` still registers opaque texture keys;
`textures::placed_key` likewise produces hashes with the top bit set. Parent `4f35fa7c`
checked `resource::is_resource` before letting an object view reach `texture_words`.
The merge removed that check when moving `NativeHost::address_words`.

The caller chain is `NativeList::dispatch` / `dispatch_from` (records `Cmd::Dispatch`),
the tests' `run`, `NativeHost::execute`, `prep`, `view_address_args`, `view_addresses`,
`address_words`, `descriptor::texture_words`, `resource::texture_desc`, then
`resource::texture_memory`. The last function reads `Resource::desc` at key + 16.
The first address-table construction walks **every** heap entry, including entry 0's
fake texture key; binding a root table at entry 2 does not exclude it. Once the table
exists, `mirror` calls the same `address_words` when updating an entry.

The masked fixture's descriptor reads request this table; they do not change key
ownership. `stride_word`, descriptor decoding, typed-buffer materialization and ordinary
texture mirroring use native resource registrations or the placed record's description.
The tests call native lists directly, so `draw.rs`, the COM list copy path, `texture.rs`
and Windows-only `descriptor::host::{prepare,is_texture}` are not on this chain.
Optional diagnostic paths (`trace_args`, `table_views`) are separate from this table walk.

Restore the parent's guard at the native boundary: registered buffers decode through
`NativeHost::resources`, object texture words require the non-faulting `is_resource`
check, and placed views continue to supply their inline description. Opaque texture
keys have zero size words, as in the parent; real objects retain their title dimensions.
Neither GPU fixture nor its assertions changes in this follow-up.

The CPU recording-backend regression `view_addresses_never_dereference_native_resource_keys`
reproduced SIGSEGV without the guard. It exercises dispatch preparation and mirrored
updates for SRVs/UAVs with explicit and default descriptions, fake keys, placed hashes,
stale records, registered buffer addresses with nonzero high words, inline placed
descriptions and real texture objects. No Metal device or shader converter is used by
this regression.

Third-follow-up verification: pinned Rust 1.98.1, offline release builds with `-j4`,
native arm64 test binary. `cargo test --offline --release -j4 -p victory-gpu` with filters
`native::host::mirror_tests` (15), `native::descriptors::tests` (17) and
`view_address_program_translates_without_a_gpu` (1) passes all 33 tests. Asset policy and
`git diff --check` pass. Both named Metal tests compile as part of that test binary but
were not run; their GPU assertions still need verification on the coordinator's Mac.
Logs, including the CPU regression's pre-fix SIGSEGV, are in
`/tmp/native-merge-third-followup`. The two earlier rounds of uncommitted changes remain.


#### Merged-run regressions: descriptor reads and covered-copy revival

The five cached containers named in the November 27 report were analyzed locally through
SSA uses; no title instructions or bytes are recorded here. With `a7cac6c2`, both mesh
containers already translate. The two pixel partners and the probe compute program fail
because `c4faf9c4`'s reader analysis mistakes these field extractions for whole-word reads:

| Program | Descriptor data actually observed | Native value and parent behavior |
| --- | --- | --- |
| Compute `f22a76fcb967ee59` | Buffer dword 0, dword 1 bits 0–15 (also a subset 0–7) | `address_words` supplies the view's full low address word and the high 16 bits. The parent lowered both words from root SRV `t1`; stride/swizzle bits were zero. One `0xffff` mask lives in an SGPR defined by a scalar move, which the upstream check did not resolve. |
| Pixel `7d0f14ce44746d7e`, `6b6db9f1a526daeb` | Image dword 1 bits 30–31, dword 2 bits 0–29, dword 3 bits 12–15 | Width minus one is assembled from dword 1 shifted right 30 and dword 2 shifted left 2, then kept to 16 bits. Height minus one is a 16-bit extraction at bit 14; first mip is a four-bit extraction at bit 12. The existing `texture_words` already supplies all of these, as on `4f35fa7c`. |

Neither pixel reads image dword 1's address, minimum-LOD or format bits. The probe's
buffer dwords 2–3 are used as descriptors by resource instructions, not as numbers via
`view_addresses`; its address reads need no record-count or format value. The parent
marked numeric reads of those words missing too. The remaining field refusal,
`ceb5eb68cd99cb07`, actually compares all of buffer dword 2 (record count); that value is
not supplied and remains refused. The fix extends `Lower::seen_bits` to
prove these uses, preserving the same table, 16-byte entries, root binding and field refusal
masks. Unknown fields and observable SCC dependencies remain refused.

The original `CS_VIEW_ADDRESSES` words are restored intact. Its five vector moves followed
by raw stores expose **all** bits of image dwords 1–3 and buffer dword 1, including address,
format, tiling and stride bits that neither native host supplied. This is not what the three
affected programs do. Keeping its old zero-filled outputs would bless unsupported fields.
The CPU test therefore preserves the whole-dword refusal; the two execution tests now
perform synthetic width/height/first-mip extraction and the SGPR-held address mask, and
check dimensions, nonzero high address bits, canaries and two-pool remapping. They do not
pre-mask the image to make the refusal disappear. The refusal unit test covers these
queries and adjacent unsupported bits, as well as live SCC uses.

The covered-copy feedback loop was `give_up_covered` → `evict_texture` → an unchanged
record requested by `remirror_tables`/`keep_named` → `realize` → `give_up_overlapped`.
That last call treated internal revival as title activity and exempted the pair. At 4096
exemptions `may_cover` returned false globally. Covered copies now have separate host
state from budget evictions: internal realization is blocked and old or copied records
mirror as null until an explicit title placement. Descriptor creation and descriptor copy
carry distinct provenance into `placed_view`; only creation calls `texture_placed`, even
when a copy is already resident. History detection happens only there, while eviction
consults exemptions without mutating them. Saturation stops admitting pairs, not releases.
`VICTORY_COVERED_CLEAR=0` preserves the previous shown-entry wait and budget-style revival.
CPU recording-backend tests cover repeated remirroring/copying with no recreation or new
exemption, explicit title revival, full-set clearing, in-flight/recorded-list protection,
GPU-written/UAV protection, and the switch-off path.

Verification uses the local pinned Rust 1.98.1 binaries and cached dependencies, offline
with at most four jobs. The release `victory` build and x86_64 GPU test `--no-run` pass.
Native arm64 CPU checks pass: 22 mirror/lifecycle tests, the bounded-history/full-set
clearing test, and heap-query translation (24 tests); the switch-off and environment
checks also pass with `VICTORY_COVERED_CLEAR=0`. The x86_64 shader suite reports 425 passed,
3 ignored and the existing `a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves`
failure, with the same `(6, true)` versus `(0, false)` assertion recorded for the parent
in `/tmp/native-merge-followup/parent-wave32.log`. Asset policy passes. No Metal or game
run was attempted in this sandbox. Logs and local corpus analysis are under
`/tmp/victory-regressions-a7cac6c2`, outside the repository.

The CPU-only `victory translate --list --jobs 4` runs before and after use identical native
options and the same 19,104 container hashes. View-address-field refusals fall from **42 to 1**
(6 buffer-address and 35 image-size queries restored); total refusals fall from **95 to 54**.
There are **zero new refusals** and no changed outcomes or reasons outside those 41 restored
programs. The reported compute and both pixel partners now translate; both mesh partners
already did. The retained original unmasked fixture was checked word-for-word against
`4f35fa7c`. `git diff --check` passes; changes remain uncommitted.

## Upstream integration at 86cefd22 (27 November 2026)

What the merge had to reconcile with this branch:

- **Covered copies.** Upstream's round-robin sweep (`COVERED_CHECKS` a sync, `Title::covered_queue`)
  drives both this branch's clearing rule (`VICTORY_COVERED_CLEAR`, default on) and upstream's
  wait (`=0`). A queued copy's displacing key is looked up when it is checked. A copy is
  cleared up to one sweep late. The trim of stale heap entries was taken out again upstream.
- **`ExecuteIndirect`.** One command an execution (`NativeList::indirect_records`), also for
  mesh executions: `dispatch_mesh_from` takes all the records, finds the pipeline once, and
  counts every record as its own dispatch. During a `VICTORY_TRACE_FLOW_SNAP` capture, draws
  and compute dispatches are recorded a record at a time, so a snapshot stays beside its record.
- **Minimum LOD clamps.** Every writer of the view-stride table goes through
  `NativeHost::write_stride_words`: the general mirror, the quick mirror (clamp 0: no texture
  view takes it) and the replay of a cached mirror, whose snapshot now keeps the clamp word
  (a replayed view left the entry's old clamp: `mirror_tests::cached_views_match_*` failed).
- **The title lock.** `VICTORY_TRACE_TITLE_LOCK` wraps `with_title_kind`; `keep_to_budget` and
  `give_up_covered` are timed in `Title::begin_sync`. With unlocked probes (the default) its
  `probes` figure is 0.
- **Stop.** Upstream's longer wait first, then this branch's emptying of the residency sets.

## Upstream integration at a6a58361 (28 November 2026)

- **Views one pool run holds (#513).** Upstream follows them as it follows wraps. Here the
  general mirror does the same (not for a span, which `refresh_spans` follows), and the quick
  mirror of plain buffer views, which used to drop the entry from `title_views`, records the
  signature kept with the pool mapping it resolved. `VICTORY_FOLLOW_POOL_VIEWS=0` is the state
  before. Not measured in game: the follow pass now walks every followed pool view's range at
  each pool-table change (upstream: 4.7 ms a look at 570,000 views), on top of this branch's
  per-table generation stamps, which already mirror bound tables' entries again.
- **Indirect draws.** Upstream gates every indirect draw (2^20 instances, 2^26 vertices); this
  branch's rectangle-list rewrite (gate kind 3) runs after that limit in the same kernel.
- **Table records.** Upstream's `table_records` (one safe read) serves `note_written`;
  `remirror_tables` keeps this branch's plain loads where the heap is known readable.
- **Shader.** Barycentrics read as data (`SV_Barycentrics`) retry inside this branch's
  `lower_program`; a wave64 mesh program's first-lane exchange is tried before the console-flow
  and lockstep first-lane reads (those are compute only); `restored` takes upstream's shape
  after this branch's scalar folds.
- **Same fix on both sides:** `ps_binding` kept for a pixel shader that needs only
  `view_addresses` (#410).
