# Tiled cascade shadows, round 2: synchronized loops

Round 3 adds [LDS phases and atomics with a separate proof/report](shadow-lds-atomics.md).
The measurements and exclusions below describe the original round 2 result.

27 November 2026, on top of the [round 1 if-conversion](shadow-if-conversion.md).
Offline, four jobs, `CARGO_TARGET_DIR=target`; changes are uncommitted and limited
to the shader crate, synthetic fixtures and documentation. No game words were
copied into tests. No Metal device, converter validation or game run was available.

## Measured result

The full sweep covers the same 19,104 native cache containers, one program each,
with `wave32_host` and `zero_copy_memory` enabled. Fingerprints cover both the host
description and emitted DXIL, not translation time.

| Outcome | Round 2 baseline | Final |
|---|---:|---:|
| Translated | 19,057 | 19,069 |
| Refused, all reasons | 47 | 35 |
| Divergent cross-half exchange refusal | 31 | 19 |
| Other refusals | 16 | 16 |
| Previously translated programs now refused | — | 0 |
| Previously translated programs with changed fingerprints | — | 0 |

All twelve requested loop containers now translate. The original 38-container
set has 19 translated and 19 refused; round 1 recovered seven of those nineteen.

| Class | Containers | Translate now | Change this round / remaining boundary |
|---|---:|---:|---|
| A | 7 | 7 | Unchanged acyclic path |
| B | 9 | 9 | +9: scalar mask/range loops, including nested/fixed loops |
| C | 2 | 2 | +2: counters ending at 12 |
| D | 4 | 0 | 1024 threads and LDS phases |
| E | 2 | 0 | LDS ordering remains outside this lowering |
| F | 2 | 0 | Loops with LDS |
| G | 8 | 0 | LDS atomics and 512/1024 threads |
| H | 1 | 1 | +1: EXEC loops, whole-wave operations and elected buffer atomic add |
| I | 1 | 0 | Loop with LDS |
| J | 2 | 0 | 1024 threads and atomics outside the accepted cyclic subset |

The twelve changed **container filenames** are:

- B: `20b78ee12ecdb57d`, `466ec41d3f9d24c0`, `56f3bf2d9388e5a7`,
  `abd7a22106898c66`, `b4b0427232707598`, `ce79e9eba7ae0348`,
  `d7db57731ada919b`, `f133e35efed2a48a`, `fa0bd10a35bfef3c`.
- C: `922e1d0ee1e19a85`, `be6ce5d849d51127`.
- H: `a953d5b93efe8424`.

H also contains `buffer_atomic_add`; “no LDS” did not mean “no side effects.”
The new cyclic subset admits buffer/global **add** only, with tests for both its
effect count and returned values. Other atomic operations remain excluded.
Two B programs use PC-relative `s_setpc_b64` transfers. Only targets already
resolved to `Exit::Jump` by the decoder are admitted; indirect jumps still refuse.

## Choice of synchronization

The implementation extends the existing console-flow lowerer and group exchange,
using a **group-synchronous schedule of console PCs**. It introduces neither a
pair spin barrier nor an unroller. These are the tradeoffs against the three options:

1. Identical scalar trip decisions in both halves are necessary but insufficient
   for the existing group barrier: another console wave can have zero iterations
   while this wave has twelve. A pair barrier would need a per-console-wave
   generation protocol, release/acquire visibility, safe reuse, and a progress
   guarantee while another host wave is not resident. A bounded poll cannot
   silently proceed on timeout. There is no validated pair primitive here, so
   this change retains group barriers and brings **every launched thread** to
   every exchange, even after its own console wave has finished.
2. Class C could use checked full unrolling at 12, but that would also require
   handling early exits, nested loops and SSA repair, and multiply the large
   surrounding shadow paths. The same scheduler handles C without duplicating
   its body. There is no inferred or guessed trip count.
3. A mask scan uses the existing full-console first-lane/ballot machinery to
   obtain its numeric seed. Both halves then carry and update that same scalar:
   `ff1`, clear the selected bit, test for zero. The seed need not be exchanged
   again if it has no new cross-half input. Exchanges in the body still happen
   on every schedule visit; each wave commits results only on its live visits.

The cost is deliberately conservative: all waves visit all scheduled blocks,
including blocks skipped by their console PC. Scalar and vector arithmetic can
be computed speculatively; resource access cannot. A traversal also has two
group barriers for continuation. Translation success does not establish an
acceptable dispatch cost; measure that before treating this as a rendering fix.

## Invariants and proof obligations

`emit/lower/synchronized.rs` first uses the existing console-flow lowering,
full 64-bit mask operations, instruction EXEC guards and loop-limit counters.
`emit/bitcode/synchronized.rs` then rewrites only the control between regions.
A region is one console block and its instruction-local guards; generated
loop-edge/abort blocks are regions too. The entry/prologue executes once and
owns resource handles and input values.

At every region boundary:

1. **One console PC.** The PC and previous original block are identical in all
   launched lanes of a console wave. `lockstep::split_branch` rejects decisions
   derived from per-lane scalar values, vector inputs or waterfall reads. The
   established scalar-input/load semantics remain in force. Branches on EXEC,
   VCC or mask-derived SCC use full-console values. Named-lane reads exchange the
   selected value across the pair; first-lane reads elect from all 64 EXEC bits,
   including an empty lower half. No half-local substitute is introduced.
2. **Original state on live visits.** `on = (pc == region_head)`. Original phi
   inputs select on the previous original tail. Every numeric SSA value used
   across regions is carried through a scheduler-header phi. A live region
   commits its new values; an inactive region retains the old values. Phi
   selections use a snapshot taken before any state update, including mutually
   dependent loop phis. EXEC, SCC, masks, scalar counters and vector values obey
   the same rule. The original conditional edge changes the PC only when `on`.
3. **No extra memory effects.** The shared predicate helper puts resource loads
   and stores, including scalar loads and the admitted atomic adds, under `on`.
   Vector work also retains EXEC. Guarded results join before escaping the guard.
   The scheduler uses `select(on, exec, false)` so undefined arithmetic from an
   inactive visit cannot make its resource guard true. Atomic return values
   remain lane values until an explicit first/named-lane operation exchanges them.
4. **All exchanges are collective.** Each region must be acyclic internally,
   reconverge at its tail, and put every wave/quad operation and barrier on every
   path to that tail. A carried definition must dominate that tail. Incoming
   transfers can target region heads only, never an instruction guard or the
   one-time prologue. Original scalar branches become PC updates; they cannot
   skip a host exchange. Different console waves retain disjoint exchange slots.
5. **Only group-uniform continuation branches.** At the schedule latch, every
   launched thread writes its active flag to its own existing `xchg` slot. A
   group barrier with a TGSM fence publishes those flags. All threads OR the
   same host-wave representative slots (0, 32, ... below the launched count).
   The PC is uniform in each host wave, so these representatives suffice, also
   for a partial final wave. A second barrier prevents the next exchange from
   overwriting flags before all readers finish. The resulting identical `any`
   controls the only generated back edge. Finished waves keep their end PC and
   frozen state but continue participating until every wave has ended.

Induction follows the original sequence of live console blocks for each wave;
inactive visits change none of that wave's observable state. Every schedule
traversal executes the same finite sequence of exchange barriers for all threads,
regardless of each wave's original loop count. Completion is therefore a property
of the whole group, not a predicate of one half or one console wave.

After transformation, cutting just the generated back edge must produce an
acyclic emitted CFG in which every barrier is mandatory. A separate dominance
fixed point checks every emitted SSA use and phi predecessor, including the
cyclic graph. Failure returns a refusal, not partially transformed output.

The schedule transformation adds at most **65,536 instructions**, plus the small
group-continuation reduction bounded by the 256-thread group limit. It carries
at most **16,384 numeric SSA values**; handles and aggregate state crossing a
region boundary refuse. It never copies the original body per iteration. The
existing per-console-loop safety counters and `LOOP_LIMIT` remain unchanged;
schedule traversals do not consume an unrelated global trip budget.

## Refusal boundaries

This remains a fallback after the exact divergent-exchange refusal, for wave64
compute on a wave32 host with 65–256 threads. The early-exit retry keeps priority.
Already successful translations continue taking their old path.

LDS allocation/communication, title barriers, waterfalls, unresolved indirect
control, scalar memory effects, other atomics and conditional mode changes are
not admitted. These exclusions retain the original divergent-exchange message.
Cyclic candidates that enter this lowering report their precise failed proof:

- A nonuniform selector/branch reports the existing lockstep source reason, e.g.
  `a branch in lockstep on a scalar a vector instruction writes per lane`.
- An unproved mixed numeric/lane-mask backedge still reports
  `a register joining a lane mask and data, read as data`. The synthetic 64-bit
  `s_bitset0_b64` mask-scan fixture checks this exact refusal. This change does
  **not** claim every 64-bit numeric mask representation is supported.
- Structural failures identify non-reconverging guards, an exchange inside an
  instruction guard, or a value unavailable at its block boundary.
- Growth, carried-state and non-numeric-state failures have distinct messages;
  the final emitted checks report `a synchronized loop failed its barrier or SSA proof`.

`VICTORY_NO_CONSOLE_FLOW=1` also disables this fallback. The existing off-by-default
`VICTORY_TRACE_CONSOLE_FLOW` reports `console schedule: translated`/the refusal
and the candidate's region and live-instruction counts. No new diagnostic switch
or guessed console synchronization primitive is introduced.

## Tests and coordinator handoff

The required release build succeeds. The unfiltered shader suite is **449 passed,
1 failed, 3 ignored**; baseline was 434 passed, 1 failed, 3 ignored. All 15 added
tests pass. The sole failure remains
`emit::wave32_tests::a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves`,
with the same `(6, true)` versus `(0, false)` assertion. The run excluding only
that known failure also completes the remaining crate targets. `git diff --check`
is clean.

`synchronized_cases.rs` contains only hand-assembled synthetic GFX10. The test-only
integer executor in `emit/bitcode/execute.rs` runs the **emitted** graph, including
simultaneous phis, TGSM, wave operations, guarded memory effects and actual
barrier rendezvous. Unknown instructions, unavailable SSA values and mismatched
collectives fail the test. This is value/structure evidence, not Metal or DXIL
validator emulation. Tests run in both host-wave arrival orders and cover:

- Zero, one and twelve trips, plus literal bound 12; 112/128/256 launched threads.
- An inactive lower EXEC half containing a deliberately wrong seed, named reads
  of inactive lanes, and different counts in two or four console waves.
- Scalar-buffer bounds and per-iteration loads, checking exact load counts.
- `ff1`/clear-bit masks with zero, one and all 32 bits set, including the high bit.
- Nested loops, resetting inner counters, finished waves and simultaneous phi swaps.
- EXEC-shrinking loops through lane 127, whole-wave row permutes and lane reads.
- Elected atomic adds: exact effects, old values and subsequent first-lane results.
- Resolved far jumps and negative tests for nonuniform control, unsupported mask
  joins, guarded barriers, escaping SSA values and excessive code growth.

Six GPU fixtures are supplied in
`crates/victory-shader/tests/fixtures/synchronized_gpu.rs`. Their native root
constant/address bindings are checked by a shader lowering test. The fixture
file was also syntax/type-checked with a local stub of its harness; it was **not
executed on Metal**. The coordinator should append this test-only include to
`crates/victory-gpu/src/native/split_wave_tests.rs`:

```rust
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../victory-shader/tests/fixtures/synchronized_gpu.rs"));
```

Run on native arm64, not under Rosetta (the GPU tests crash there):

```bash
CARGO_TARGET_DIR=target cargo test -j4 --offline --release -p victory-gpu synchronized_multi_wave -- --nocapture
```

A `SKIPPED: Metal/MSC unavailable` message is not a pass on GPU values. Run the
round 1 fixtures as well, then create pipelines for the twelve recovered
containers, measure dispatch cost and check the game's shadows. Those steps
remain pending coordinator access to Metal.

## Class E: ordering assessment only

Both halves must finish the applicable LDS writes before either half's dependent
read, and both must finish reads of the old contents before either can overwrite
them. Title group barriers additionally order all participating console waves
at the same phase. It is insufficient to preserve only each half's instruction
order or to fence only the private cross-half exchange array.

Existing one-console-wave lockstep places TGSM group barriers before and after
each LDS write, outside EXEC guards, and proves all threads reach them. That
proves the half-to-half ordering in its one-wave scope. The multi-wave acyclic
predicate path and this scheduler deliberately exclude LDS and title barriers.
Their exchange barriers do not by themselves establish the required title LDS
phase ordering on paths that different console waves take. Class E needs an
explicit phase/order proof and synthetic shared-memory value tests before
enabling it; both E containers remain refused. No E relaxation is included.

## Reproduction and local evidence

Use the pinned Rust 1.98.1 toolchain. In this clone, offline crates are under
`CARGO_HOME="$PWD/../xcl-tools/cargo"`. Set `CARGO_TARGET_DIR=target` throughout.

```bash
cargo build -j4 --offline --release --target x86_64-apple-darwin -p victory
cargo test -j4 --offline --release --target x86_64-apple-darwin -p victory-shader
cargo test -j4 --offline --release --target x86_64-apple-darwin -p victory-shader -- \
  --skip emit::wave32_tests::a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves
git diff --check

VICTORY_CACHE_DIR=/tmp/native-shader-loop-native-cache \
  target/x86_64-apple-darwin/release/victory translate --list --jobs 4 \
  ../victory-test/crosslane-containers

cargo run -j4 --offline --release --target x86_64-apple-darwin \
  -p victory-shader --example sweep -- \
  "$HOME/.cache/victory/cache/native-shaders/containers" -j 4 \
  --list /tmp/native-shader-loop-files.txt > /tmp/native-shader-loop-final.tsv
```

The list contains all 19,104 container filenames because the cache files have
no `.bin` suffix. Compare status and fingerprint by filename; the timing column
is not part of the fingerprint. Baseline/final sweep rows are outside the repo
at `/tmp/native-shader-loop-baseline.tsv` and `/tmp/native-shader-loop-final.tsv`.
Build, suite and native-list logs use the same `/tmp/native-shader-loop-` prefix.
