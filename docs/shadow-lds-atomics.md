# Tiled cascade shadows, round 3: LDS phases and atomics

27 November 2026, on `0c0c7592`, following
[if-conversion](shadow-if-conversion.md) and
[synchronized loops](shadow-synchronized-loops.md). Changes are uncommitted and
limited to the shader crate, synthetic tests/fixtures, and documentation. All
inspection and translation was local, offline, with four jobs and
`CARGO_TARGET_DIR=target`. No title words, disassembly or generated shader blobs
are included here. No Metal device, converter validation or game run was available.

## Measured result

The baseline and final sweeps cover the same 19,104 cache containers, one program
each, with `wave32_host` and `zero_copy_memory` enabled. Fingerprints include the
host description and emitted DXIL; elapsed translation time is excluded.

| Outcome | Round 3 baseline | Final |
|---|---:|---:|
| Translated | 19,069 | 19,081 |
| Refused, all reasons | 35 | 23 |
| Remaining from the divergent-exchange set | 19 | 7 |
| Literal original divergent-exchange message | 19 | 2 |
| New precise title-barrier proof refusal | 0 | 5 |
| Other refusals | 16 | 16 |
| Previously translated programs now refused | — | 0 |
| Previously translated programs with changed fingerprints | — | 0 |

Five refusals now name the failed phase proof, so counting only the old message
would misleadingly suggest that just two unresolved containers remain.

| Class | Containers | Translate now | Ordering property / disposition |
|---|---:|---:|---|
| D | 4 | 4 | The existing empty-EXEC exit retry runs first. On that retained-lane CFG, each title barrier is outside cycles and on every path to an exit. Two or three title phases separate producers and consumers; scalar loops may take different numbers of schedule traversals. |
| E | 2 | 2 | One mandatory title barrier in an acyclic CFG. LDS writes complete before dependent reads; the later EXEC joins, counts and SCC decisions use full-console values. |
| G | 8 | 2 | Two programs have three mandatory, noncyclic title barriers. Their elected LDS atomics, per-lane buffer atomics and first-lane results retain each console wave's program order. Six programs remain excluded, as detailed below. |
| J | 2 | 2 | No title barriers. The two buffer adds and unsigned-min are per-thread effects, guarded by wave reach and EXEC; explicit lane operations exchange their returned values. No extra election is invented. |
| F | 2 | 2 | Five mandatory title barriers outside the per-lane loops. The full 64-bit EXEC determines each console wave's loop continuation; DS read/write2/st64 offsets and atomic results remain lane values. |
| I | 1 | 0 | A title barrier lies in the image/LDS loop. No proof of common group iteration/phase counts; still refused. |

A 7/7, B 9/9, C 2/2 and H 1/1 remain unchanged. Of the original 38-container set,
31 now translate and seven remain refused. The rebuilt native
`victory translate --list --jobs 4` command independently reports those same
31 successes and seven refusals over the 38-container directory, using a fresh
cache outside the repository.

The twelve newly translated **container filenames** are:

- D: `4e9b58cc637d9272`, `67353810417f6c19`, `b8f3f7979d6bbc50`, `eae3498e6553afa8`.
- E: `0c028454b58543f0`, `eab96ed5d1684967`.
- G: `6832a9da15c2e584`, `6b82bc7d22016cb5`.
- J: `c6a4ca484a40cc64`, `fd6e1cbec58e93a9`.
- F: `02384a60fda6f2d0`, `22338086ea4c5364`.

## Phase and memory proof

This extends the existing PC schedule to 65–1024-thread wave64 compute groups on
a 32-lane host. Acyclic programs that communicate through LDS or atomics use the
same schedule. Pure acyclic if-conversion retains its 256-thread bound. The new
path is still a fallback after the exact divergent-exchange refusal; successful
older translations and the existing empty-EXEC exit retry keep priority.

A title `s_barrier` is a **PC rendezvous**, not merely another physical fence in
the block traversal. Lowering splits the containing block before and after the
barrier, preserving SSA values and original predecessor identities. At the
barrier region every launched thread publishes whether its console PC has
arrived. A group-wide OR of missing participants, using both host-half
representatives, prevents *any* wave advancing until *all* PCs name that site.
A TGSM group barrier then fences the title phase, including UAV memory when the
scheduled program has buffer/image/global memory effects. Both halves remain
present even when their console EXEC has no active lanes.

Before doing this, `once_blocks` proves every reachable title barrier's block is
outside cycles and occurs on every entry-to-exit path. Multiple barriers in a
block are split in instruction order. Consequently all waves encounter the same
finite sequence of title barriers, and each logical barrier releases exactly
once. Physical polling fences repeat while waiting and on inactive schedule
visits; they do not replay source effects or allow an early phase transition.
The final emitted CFG and SSA checks still require all collectives to be
unconditional in each traversal and every value to dominate its use.

D's unmodified CFG initially fails that test: empty EXEC can branch to a bare
end. The **existing** `keep_lanes_to_end` retry removes those exits first. The
scheduler checks the resulting CFG rather than assuming the original barriers
were unconditional. The hand-assembled retained-wave test initializes LDS,
retires a whole console wave with EXEC zero, and checks that its threads still
participate in phases but perform no later LDS reads, writes or output stores.
D's scalar counters (including the previously identified bounds 25 and 15) need
no common-trip-count assumption: a wave reaching a barrier early parks there.

Every admitted LDS access has collective group fences before and after it,
outside the resource/EXEC guard. Reads as well as writes/atomics are guarded:
inactive visits cannot dereference stale or invalid addresses. Read and atomic
results join with the old VGPR value for inactive lanes. This preserves
wave-internal write→read and read→overwrite order across the two host halves,
including communication for which the console compiler omitted a title barrier.
Different console waves may still arrive in any order at an atomic; stronger
host rendezvous selects an allowed ordering while preserving source order and
title phase boundaries. No cross-wave lockstep arrival order is assumed.

The original loop counters and limits remain. For these new schedules with title
barriers, reaching a safety limit cancels the whole group at the schedule latch.
In particular, a conditional backedge must not turn a limit hit into its ordinary
exit and release a partially produced LDS phase. The limit edge has its own
scheduled region and correct phi predecessor; both unconditional and conditional
abort paths participate in the group cancellation vote. Schedules without title
barriers retain their previous limit behavior and fingerprints.

## Atomic election and returned values

`ds_add_rtn_u32`, compare-exchange and signed/unsigned min/max remain per-lane LDS
operations. An elected use is elected by the **source program's full 64-bit
EXEC**, not a new host `WaveIsFirstLane`. Full-console mask scans/`ff1` choose one
console lane, and the atomic is guarded by that lane's EXEC bit and its wave's
live PC. It therefore runs once per elected console wave, not once per host half.
The returned old value stays in the elected lane's VGPR until the source's
first/named-lane read exchanges it. All launched lanes execute that exchange,
including a host half with zero active console EXEC lanes.

Buffer/global add, unsigned-min and 32-bit compare-exchange use the existing
per-thread DXIL atomics under the same wave/EXEC predicates. They require no
implicit wave election. J's per-lane results can feed indexing directly; only
explicit wave-level uses require the exchange. G's two accepted programs also
contain buffer compare-exchange and unsigned-min, tested separately from LDS
compare-exchange (their new/compare operand order differs).

The atomic fixtures elect lane **33** using `ff1(EXEC)`, allocate from one counter
shared by multiple console waves, and check exact effect counts, distinct old
values and per-wave order in both simulated arrival orders. Both host halves
must receive each broadcast. Min/max and compare-exchange tests check old and
new values, including a failed compare-exchange. Buffer tests check every lane's
add/min/compare-exchange effects and the subsequent console-wide first-lane result.

## Per-iteration cost

The schedule visits each region once per traversal, even for inactive or finished
waves. An EXEC-narrowing backedge uses the OR of **both halves** of the console
mask; it costs one two-barrier full-console ballot when that mask needs
materialization. Existing cached numeric mask values can avoid redundant
materialization. There is no per-lane host loop exit that can strand the other half.

For a traversal with `E` full-console exchanges, `L` scheduled LDS access sites
and `B` title barriers, this lowering executes:

- `2E` exchange barriers;
- `2L` LDS ordering barriers;
- `3B` title rendezvous barriers (two for the arrival vote, one phase fence);
- two continuation barriers, plus two cancellation barriers when a loop can abort
  while peers wait at a title barrier.

The trace for F is **67 full-console exchanges, 38 LDS access sites, five title
barriers and two latch votes: 229 group barriers per complete traversal**. Its
one-time prologue has two more barriers. These counts include visits to other
loops/phases; they are not just the currently active source loop body. Each live
backedge requires another traversal. Both F containers have the same shape.

At 1024 threads there are 16 console waves / 32 host halves. The exchange scratch
array is 4096 bytes, separate from title LDS. Each group vote stores 1024 flags
and performs 32 representative-slot loads per thread: **32,768 loads per vote**.
A schedule with F's counts would have seven votes per traversal, or 229,376 such
loads, besides the 67 ordinary full-console exchanges and the actual LDS work.
The rendezvous count is per group, not multiplied into independent wave barriers:
all 1024 threads must attend each one. The two recovered G programs declare 32 KiB
of title LDS, so their 4 KiB exchange array brings declared shared storage to
**36 KiB**. Their pipeline/resource acceptance particularly needs coordinator
validation. This is a correctness implementation with potentially substantial
dispatch cost; no Metal timing is claimed.

The existing 65,536-instruction bound applies to the PC/SSA rewrite and its
16,384 carried numeric values. Additional LDS fences are linear in `L`; phase
votes are linear in `B × ceil(threads/32)`, bounded to 32 representative slots
per vote. There is no body unrolling or guessed wave trip count.

## Exact remaining refusals

| Class / container | Offset | Exact reason after `GFX10 code at ...:` |
|---|---|---|
| G `1d434307d981352a` | `0x1bc` | `a cross-lane operation between host waves in divergent code is not supported` |
| G `3b02415f67c6b0d8` | `0x1ac` | `a cross-lane operation between host waves in divergent code is not supported` |
| G `2365ac7d30dd6f0c`, `ea5d5c9a44e9cd1f` | `0x10c` | `a title barrier not proved exactly once on every group path is not supported` |
| G `4869d261ac95b961`, `dbf247e6efc048df` | `0xec` | `a title barrier not proved exactly once on every group path is not supported` |
| I `b0d5ffa0801b7450` | `0x218` | `a title barrier not proved exactly once on every group path is not supported` |

The first two G programs have 51 and 96 descriptor-waterfall first-lane reads,
respectively, as well as cyclic title barriers. Their existing eligibility
exclusion retains the original message. The other four G programs and I enter
the proof and fail at their cyclic title barrier. The new schedule does not
claim that a worklist loop's dynamically repeated barrier phases align between
all console waves.

GDS, append/consume and other unlisted DS operations, sub-dword LDS writes,
64-bit or unlisted buffer atomics, waterfalls, indirect control, scalar memory
writes and conditional mode changes remain outside this fallback. Existing
nonuniform-scalar, mixed mask/data, guarded-exchange, SSA and growth failures
retain their precise reasons. A bypassed noncyclic title barrier also refuses;
being outside a loop alone is insufficient.

## Tests and coordinator handoff

The release shader suite is **459 passed, 1 failed, 3 ignored**, with ten new
tests. The only failure is the unchanged
`emit::wave32_tests::a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves`
assertion `(6, true)` versus `(0, false)`. Two older negative tests now assert the
more precise title-barrier refusal; they still reject those shapes. The old
unused-LDS-allocation exclusion test now checks identical output with/without
unused allocation. The emitted integer-graph executor checks real guards,
phis, TGSM, atomic old/new values, and collective rendezvous. Unknown operations
and uninitialized shared reads fail. It is not a Metal/DXIL validator.

The required release build passed, and `git diff --check` is clean. With only
the known failure filtered out, the remaining 459 shader tests and all five
sweep-example tests passed; three shader tests remain ignored.

All five GPU fixtures are in
`crates/victory-shader/tests/fixtures/communicating_gpu.rs`, sharing only synthetic
words with `communicating_cases.rs`. Their native root bindings and LDS allocation
are checked by a lowering test; the fixture file is separately syntax/type-checked
against a harness stub. **No GPU fixture was executed here.**

Append this test-only include to
`crates/victory-gpu/src/native/split_wave_tests.rs` on the coordinator's checkout:

```rust
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../victory-shader/tests/fixtures/communicating_gpu.rs"));
```

Run on the native arm64 host, not under Rosetta:

```bash
CARGO_TARGET_DIR=target cargo test -j4 --offline --release -p victory-gpu communicating_multi_wave -- --nocapture
```

This checks partial-EXEC producers/consumers across title barriers; one elected
atomic per console wave with broadcasts into an empty half; two console waves
sharing repeated allocations; per-lane loops leaving in both halves; and
per-lane buffer add/min effects and returns. It includes 1024-thread fixtures.
The buffer GPU case uses root-address `global_atomic_*`; the CPU value tests also
check the descriptor-form `buffer_atomic_*` operations.

Keep the round 1 and round 2 includes and rerun their fixtures as well:

```bash
CARGO_TARGET_DIR=target cargo test -j4 --offline --release -p victory-gpu predicated_multi_wave -- --nocapture
CARGO_TARGET_DIR=target cargo test -j4 --offline --release -p victory-gpu synchronized_multi_wave -- --nocapture
```

`SKIPPED: Metal/MSC unavailable` is not a GPU value pass. Pipeline creation for
the twelve recovered containers, resource limits, dispatch cost and visible
shadows still require coordinator validation. Translation of the D helpers
alone does not establish a rendered-shadow fix.

## Reproduction

Use pinned Rust 1.98.1 with its own bin first on PATH. Offline dependencies in
this clone are at `CARGO_HOME="$PWD/../xcl-tools/cargo"`; set
`CARGO_TARGET_DIR=target`. Builds are run under `nice` (this sandbox reports
`setpriority: Operation not permitted` but executes the command).

```bash
cargo build -j4 --offline --release --target x86_64-apple-darwin -p victory
cargo test -j4 --offline --release --target x86_64-apple-darwin -p victory-shader
cargo test -j4 --offline --release --target x86_64-apple-darwin -p victory-shader -- \
  --skip emit::wave32_tests::a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves
git diff --check

VICTORY_CACHE_DIR=/tmp/native-shader-r3-native-cache \
  target/x86_64-apple-darwin/release/victory translate --list --jobs 4 \
  ../victory-test/crosslane-containers

cargo run -j4 --offline --release --target x86_64-apple-darwin \
  -p victory-shader --example sweep -- \
  "$HOME/.cache/victory/cache/native-shaders/containers" -j 4 \
  --list /tmp/native-shader-loop-files.txt > /tmp/native-shader-r3-final.tsv
```

The manifest explicitly lists all 19,104 filenames (these cache files have no
`.bin` suffix). Baseline rows are `/tmp/native-shader-lds-baseline.tsv`; final rows
are `/tmp/native-shader-r3-final.tsv`. Compare status and fingerprint by filename,
not the timing column. Local logs and inspection aggregates use the
`/tmp/native-shader-r3-` prefix and stay outside the repository.

The existing off-by-default `VICTORY_TRACE_CONSOLE_FLOW` now also reports title
barrier counts, prologue/scheduled LDS sites, and prologue/per-traversal barrier
counts. `VICTORY_NO_CONSOLE_FLOW=1` still disables the fallback. No new diagnostic
switch or undocumented console scheduling assumption is introduced.
