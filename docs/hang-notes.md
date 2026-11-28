# GPU hangs as the opening cutscene starts (notes, 2026-11-28)

Status: open. This file collects what is known; `docs/native-host.md` is at its size limit.

## The symptom

In most warm runs of 2026-11-27 (O1, O3, O4, O5, n88), about +92 to +112 s after launch, as the
opening cutscene starts, Metal fails a command buffer with "Caused GPU Hang Error
(00000003:kIOGPUCommandBufferCallbackErrorHang)", usually a second one 3 to 6 s later, and then
answers every later command buffer of the process with "Ignored (for causing prior/excessive GPU
errors)". Nothing the GPU should do after that is done, so the cutscene stays on one camera.

macOS writes a report for each restart (`/Library/Logs/DiagnosticReports/gpuEvent-victory-*.ips`,
bug type 284). It names only a reason and a data master: "firmware-detected lockup" (reason 4),
"CDM Kill timeout" (reason 9) or "MMU interrupt" (seen once, H5). No shader or address is in it.

## What the runs showed

| Run | Switches beyond the baseline | Failed command buffers |
|---|---|---|
| O1, O3, O4, O5, n88 (19:14-20:16 local) | `VICTORY_TRACE_TARGETS=1`, target dumps | 1 to 5, at +92..+112 s |
| n87 (18:59) | `VICTORY_TRACE_TARGETS=1` | 0 |
| H1, H2, H3 | `VICTORY_TRACE_GPU_FAILS` (`=1`, `=1`, `=names`) | 0 |
| H4 | `VICTORY_TRACE_GATE=1` | 0 |
| H5 | n88's switches + `VICTORY_TRACE_GATE=1` | 1 at +103.0 s ("MMU interrupt") |
| H6 | as H5, with the lifetime fix below | 1 at +96.0 s, then 4 at +102.6 s |

Baseline everywhere: `VICTORY_METAL_RESIDENCY_SETS=1 VICTORY_TEXTURE_DROP_TOP_MIP=1`.

- The command buffer that failed in H5 held **no compute dispatch** (`VICTORY_TRACE_GATE` lists
  them): only render and blit passes, with 12 fills. The first failed buffer of n88, O4 and O5 also
  owed 12 fills ("paid 0 fence signals and 12 value writes").
- No indirect dispatch record the gates let run was large: 14,899 groups at most in H4, 1,300 in the
  seconds before H5's failure. Garbage counts under the gate's limit are not the cause.
- The runs that failed had GPU memory in use near 14 GB and frames of 4 to 12 s (pipelines compiled
  inline) around the cutscene start; the Metal pipeline archive is shared by every tree, so a run is
  only as warm as the last tree that ran.

- H5 and the main session's P1b (another tree, 1 hang at +122 s) were both reported by macOS as
  "MMU interrupt": the GPU reached memory with no translation. That points at memory freed while
  queued work could still read it, not at a shader that loops.

## A lead that did not hold: objects let go while queued work may read them

With `VICTORY_METAL_RESIDENCY_SETS=1` nothing retains what a heap entry names: encoders do not name
it (`useResource` would make the command buffer retain it) and an `MTLResidencySet` retains nothing.
Two host paths let such objects go without waiting for the GPU:

1. `Heap::commit_set` released the allocations and views a heap no longer named right after
   publishing the set, while command buffers committed before could still be queued.
2. `NativeHost::refresh_spans` let a replaced span buffer go after 64 submissions (`SPAN_GRACE`, a
   documented shortcut); its deallocator unmaps the alias (`mem::Stitch::free`). The bad runs
   replaced 450 to 900 span buffers between +60 and +130 s.

The cutscene start is when both peak (texture streaming, remapped title ranges) while frames take
4 to 12 s and the GPU runs far behind the CPU. Offline experiment (Swift, M4 Pro, macOS 27.0.1):
textures reached only through resource IDs and a queue residency set, released after command
buffers that read them were committed but before the GPU ran them, faulted the GPU every time
("Caused GPU Address Fault Error", report "BIF0 page fault"); removing allocations from the set, or
adding them late, did not fault. Not shown: which object H5's fault reached (the report has no
address), nor why the game's error is "Hang"/"MMU interrupt" rather than "Address Fault".

## Lifetime fix (in native-next, uncommitted): correct, but not the cause

H6 ran with it and hung the same way; no span buffer would have gone early under the old count in
that run. Kept because the rule is right for residency sets (the experiment above faults), and it
costs little.


- `victory-backend`: `ComputeBackend::work_mark` / `work_done` (default: unknown, done).
  Metal: the next command buffer serial, and whether every committed command buffer below it has
  been reported on (`gfx.rs`, `flights()`).
- `victory-metal`: what `commit_set` lets go is kept by `Metal::keep_until_done` with the mark and
  let go at a later submission once that work is done (`let_go_of_done`). Without residency sets
  nothing changes (objects go at once; encoders retain them).
- `victory-gpu`: a replaced span buffer waits for `work_done` of the mark taken when it was
  replaced; only a backend that cannot tell keeps the 64-submission count. A line is logged when
  the count alone would have let one go while earlier work was unfinished ("a replaced span buffer
  outlives the 64-submission count").
- Tests: `victory-metal` `heap_set_tests::a_retired_texture_outlives_the_command_buffers_committed_before`;
  `victory-gpu` `mirror_tests::a_replaced_span_waits_for_the_work_submitted_before_it` and
  `a_span_goes_by_its_mark_or_without_one_by_the_count` (both fail with the old rules). Four
  heap-set lifetime tests now wait for earlier work before expecting the release.
- Cost: retired objects live about one GPU latency longer (normally one frame); no extra GPU work.

## Diagnostics added

- `VICTORY_TRACE_GPU_FAILS=names`: the labels and failure reports of `VICTORY_TRACE_GPU_FAILS=1`
  without encoder execution status (`MTLCommandBufferErrorOption::EncoderExecutionStatus`), which
  changes how the GPU runs the work. H1-H3 (all traced) did not hang.
- `VICTORY_TRACE_GATE=1`: the gates of indirect dispatch and mesh dispatch records keep, in a shared
  buffer, the largest record they let run and the largest they stopped, logged once a second
  (`gate: largest run ...`). Each command buffer also keeps the list of its compute dispatches
  (pipeline, group counts; indirect ones as zeros); a failed one is logged by name
  (`gate: failed command buffer #n ran k dispatches: ...`). Cost: one atomic per gated record and a
  vector push per dispatch.

## Ruled out or weakened

- A per-half `v_readfirstlane` branch that splits a console wave at an exchange barrier: the
  translation reads the console-wide first lane; GPU test
  `split_wave_tests::a_branch_on_a_first_lane_read_takes_the_whole_console_wave`. The long-standing
  failing shader test `a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves` predates
  console flow and expects a refusal the translator no longer needs.
- Indirect dispatch records with garbage counts under the gate's 2^22 groups (see above).
- A compute dispatch as the failing work: H5's and H6's first failed command buffers had none.

## Where to look next

The first failed command buffer of H5 and H6 held only render and blit work with exactly 12 buffer
fills (`ClearUnorderedAccessView` on buffers, `glue.rs` -> `NativeList::fill`); n88, O4 and O5's
first failures also owed 12 fills. `VICTORY_TRACE_COMMITS` in H1 shows such a command buffer every
frame, often a lone blit pass of 12 fills. A fill that writes memory with no GPU translation fits
"MMU interrupt". To check: log each fill's destination (title address, length, pool or span buffer,
offset in it) in that command buffer at the failure; and whether a title pool can be released
(`xmem_physical` `Pool::drop`) while queued work still writes through its no-copy buffer
(`wrap_title_memory` asks the caller to keep the range mapped while work lives).

## Title pools are held until the work before their release has finished (2026-11-28)

`xmem_physical`'s `Pool::drop` unmapped a pool's own mapping as soon as the title freed its last
page, while the GPU's buffer over that mapping (`NativeHost::pool_buffer`, a no-copy wrap that
joins the queue residency set; heap entries and the set retain nothing) could still be used by
command buffers queued behind slow frames. Now:

- `xmem_physical::Observer::holds_released_pools` (the runtime's `PoolFeed` says yes): a released
  pool's `PagePool` is kept in `HELD` until `xmem_physical::let_go_of_pool(id)`.
- `NativeHost::forget_released` keeps the released pools' buffers with the backend's work mark
  (`held_pools`); `NativeHost::let_go_of_pools`, at every submission, drops them and calls
  `memory::let_go_of_pool` (the hook `set_pool_release_hook`, set by `native_run.rs`) once
  `work_done(mark)`. A backend that cannot tell lets go at once, as before.
- Tests: `xmem_physical::tests::a_held_pool_stays_mapped_until_let_go`,
  `mirror_tests::a_released_pool_waits_for_the_work_submitted_before_it`,
  `a_pool_goes_by_its_mark_or_at_once_without_one`.

Diagnostic: with `VICTORY_TRACE_GATE=1` every fill also records what it writes (`NativeHost::
describe_title`: title address and length, pool id and offset or span buffer; then the Metal
buffer's base, length, pool flag, offset and the value), and a failed command buffer's fills are
logged: `gate: failed command buffer #n held k fills: ...`.

`VICTORY_TRACE_MEMORY=<seconds>` (10 if not a number): two lines "memory by kind: textures: ..."
and "memory by kind: buffers: ..." (`Title::trace_memory`, `NativeHost::buffer_breakdown`):
texture copies by kind, format and size, with Metal's own size of each; copies over the same
title range, the same start, or with identical bytes; covered copies; span buffers, wraps,
host buffers, held pools, depth textures and twins, and Metal's whole count.
