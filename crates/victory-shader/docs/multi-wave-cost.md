# Multi-wave translation, round 4: cost and switches

27 November 2026, compared with communicating-programs HEAD `fbc583bf`.
All corpus inputs, sweep rows and trace captures stayed outside the repository.
Only aggregate counts, container filenames and refusal messages appear here.
There was no Metal device, converter validation, game run or GPU timing in this
round. These are emitted-structure measurements and synthetic value tests.

The principal cost was the scheduler's **whole-program backedge**. Every loop
iteration revisited hundreds of cross-half exchanges in other branches and
outside the loop, even when the console PC was nowhere near those blocks.
In B that meant 626–1,678 group barriers per traversal; in C it meant 840.
The replacement schedules cyclic components separately and visits acyclic
components once. B now has 10–12 barriers per cyclic traversal, C has 10, and H
has 22. The substantial one-time costs remain visible in the tables below.

## Independent switches and cache behavior

All three new `emit::Options` booleans default to false. The native host reads
only the exact environment value `1` in its existing `options()` function:

| Environment variable | Option | Disabled fallback |
|---|---|---|
| `VICTORY_NO_IF_CONVERT=1` | `no_if_convert` | Round 1, acyclic noncommunicating multi-wave programs |
| `VICTORY_NO_SYNC_LOOPS=1` | `no_sync_loops` | Round 2, cyclic noncommunicating programs, including H's admitted atomic add |
| `VICTORY_NO_COMM_WAVES=1` | `no_comm_waves` | Round 3, LDS/title barriers, extended atomics or groups above 256 threads |

Classification selects the earliest round that could admit a candidate.
Keeping round 3 on cannot re-enable a disabled round 1 or 2. The switch gates
only the fallback after the original divergent-exchange refusal; successful
ordinary translations retain their original path. The old global
`VICTORY_NO_CONSOLE_FLOW` behavior is unchanged.

Native shader caching already includes the complete `Options` debug value in
its key. Thus switches affect cached successes and refusals, without hidden
environment reads in the shader library. A synthetic test checks all eight
distinct option keys, unchanged enabled output, and the exact original refusal
for each disabled round. `sweep` exposes matching explicit command-line flags
`--no-if-convert`, `--no-sync-loops`, and `--no-comm-waves`.

All eight combinations were also swept over the original 38 containers:

| Disabled rounds | Translated | Refused |
|---|---:|---:|
| None | 31 | 7 |
| 1 | 24 | 14 |
| 2 | 19 | 19 |
| 3 | 19 | 19 |
| 1, 2 | 12 | 26 |
| 1, 3 | 12 | 26 |
| 2, 3 | 7 | 31 |
| 1, 2, 3 | 0 | 38 |

Every surviving success has the same fingerprint as the new default build.
Every disabled success restores `a cross-lane operation between host waves in
divergent code`, including the original location in the formatted refusal.

## Counting convention

A barrier count means **group rendezvous phases**, not the number of threads
calling the barrier. A private exchange uses two `barrier(1 | 8)` calls:
one to publish the slot writes and one to finish reads before scratch reuse.
Each is both a group barrier and a TGSM fence. Ballots have one slot-store
instruction and two slot-load instructions per participating thread; named
lane reads have one store and one load. Ballot stores from a host wave write
the same host-wave slot; these are instruction counts, not unique addresses.

The group continuation vote has one store per thread and one load of each
host-wave representative per thread, followed by OR reduction. Write
`N = launched group threads`, `H = ceil(N / 32)`. A vote costs two barriers,
`N` stores and `N * H` loads. Votes are now in loop headers: `T` body traversals
need `T + 1` votes, including the final false test. An unused component has a
single vote and executes no body. An exchange-free loop has no vote at all.

`T_old` denotes traversals of the old whole-program schedule. `T_i` denotes
body traversals of cyclic component i in the new schedule; it is not an assumed
game trip count. For simple counters one traversal performs one counter step.
Nested/irreducible components retain their console-PC schedule, so use its
traversal count rather than assuming one outer source iteration. The tables
include all static branches that the schedule must rendezvous through.

For an ordinary completing group the new barrier count is exactly

`B_once + sum(B_body[i] * T_i) + 2 * sum(T_i + 1)`.

The last term applies only to collective components. The table's `B_iter`
includes a header vote, so the equivalent expression is
`B_once + 2 * K + sum(B_iter[i] * T_i)` for K collective components.
Sum these expressions over the groups in a dispatch to obtain its barrier
count. Multiply by group threads only when counting per-thread barrier calls.
A zero-coverage group returns before all these costs. Cancellation can also
shorten a dispatch; no average dynamic count or GPU speedup is inferred here.

## B, C and H: before and after

The before column includes the old continuation vote. B and C each have six
additional prologue barriers per group; H has none. `E_iter` is the number of
private exchange/vote sequences (each two barriers) in that traversal.

| Class / container | N | Before B_iter / E_iter | After B_once | After B_iter / E_iter for each cyclic component |
|---|---:|---:|---:|---|
| B `20b78ee12ecdb57d` | 128 | 1602 / 801 | 1528 | 10/5, 12/6, 10/5, 10/5, 12/6, 10/5, 12/6, 12/6 |
| B `466ec41d3f9d24c0` | 256 | 626 / 313 | 610 | 12/6, 12/6 |
| B `56f3bf2d9388e5a7` | 128 | 628 / 314 | 612 | 12/6, 12/6 |
| B `abd7a22106898c66` | 256 | 722 / 361 | 706 | 12/6, 12/6 |
| B `b4b0427232707598` | 256 | 718 / 359 | 702 | 12/6, 12/6 |
| B `ce79e9eba7ae0348` | 256 | 772 / 386 | 742 | 12/6, 12/6 |
| B `d7db57731ada919b` | 256 | 778 / 389 | 762 | 12/6, 12/6 |
| B `f133e35efed2a48a` | 128 | 1678 / 839 | 1574 | 10/5, 12/6, 10/5, 10/5, 12/6, 10/5, 12/6, 12/6 |
| B `fa0bd10a35bfef3c` | 256 | 750 / 375 | 714 | 12/6, 12/6 |
| C `922e1d0ee1e19a85` | 128 | 840 / 420 | 828 | 10/5, 10/5 |
| C `be6ce5d849d51127` | 128 | 840 / 420 | 828 | 10/5, 10/5 |
| H `a953d5b93efe8424` | 128 | 30 / 15 | 6 | 22/11 |

For example, C goes from `6 + 840*T_old` barriers per group to
`832 + 10*(T_1 + T_2)`. B `466e…` goes from `6 + 626*T_old` to
`614 + 12*(T_1 + T_2)`. H goes from `30*T_old` to `8 + 22*T_1`.
TGSM fence counts are identical to these barrier counts. B/C/H have **zero UAV
fences, zero synchronization atomics and zero polls/spins**, before and after.
B/C have no title atomic sites. H has one guarded title atomic-add site; it
executes only for original live, elected lanes. Its number of actual effects
depends on the original inputs/EXEC, not the scheduling traversal count.

These slot counts include the continuation vote and are per thread; multiply by
N for the old group total. For the new output they are upper bounds because
inactive private phases can skip their slots. The once columns exclude the K
terminal header votes. Add `K` stores and `K*H` loads per thread to obtain the
per-group constant, then add the iteration terms and multiply by N.

| Container | Before stores / loads per traversal | After once stores / loads | After stores / loads per cyclic traversal |
|---|---:|---:|---|
| `20b78ee12ecdb57d` | 801 / 1558 | 764 / 1488 | 5/12 or 6/14, in the barrier-table order |
| `466ec41d3f9d24c0` | 313 / 622 | 305 / 600 | 6 / 18 |
| `56f3bf2d9388e5a7` | 314 / 620 | 306 / 602 | 6 / 14 |
| `abd7a22106898c66` | 361 / 714 | 353 / 692 | 6 / 18 |
| `b4b0427232707598` | 359 / 709 | 351 / 687 | 6 / 18 |
| `ce79e9eba7ae0348` | 386 / 749 | 371 / 727 | 6 / 18 |
| `d7db57731ada919b` | 389 / 770 | 381 / 748 | 6 / 18 |
| `f133e35efed2a48a` | 839 / 1605 | 787 / 1535 | 5/12 or 6/14, in the barrier-table order |
| `fa0bd10a35bfef3c` | 375 / 722 | 357 / 700 | 6 / 18 |
| `922e1d0ee1e19a85` | 420 / 819 | 414 / 805 | 5 / 12 |
| `be6ce5d849d51127` | 420 / 811 | 414 / 797 | 5 / 12 |
| `a953d5b93efe8424` | 15 / 26 | 3 / 6 | 11 / 20 |

Old B/C prologues add three stores and six loads per thread; H adds zero.
Thus, for example, the first C's old dispatch traffic is the sum over groups
of `N*(3 + 420*T_old)` stores and `N*(6 + 819*T_old)` loads. Its new upper bounds
are `N*(416 + 5*(T_1+T_2))` stores and `N*(813 + 12*(T_1+T_2))` loads.

The remaining C loops still contain four full ballots each; most B loops have
five. A fixed scalar trip counter alone does not prove those ballot inputs
invariant. This pass retains them unless their complete input dependency is
proved invariant. It does **not** claim zero barriers for the real C bodies.
The synthetic invariant-counter case proves the stronger optimization: all
exchange costs are independent of whether the two console waves take 0, 1 or
12 trips, with zero barriers/votes/exchanges inside its loop.

## Work, ordering and termination proofs

1. **Separate cycles from unrelated code.** The condensation of the original
   region CFG is a DAG. Visit its strongly connected components in topological
   order; a wave cannot return to a completed component. Acyclic components run
   once. Each cyclic component carries its own PC, predecessor and numeric SSA
   state. At a matching PC, select the original phi inputs, perform the original
   guarded effects and commit the next state. Otherwise preserve previous state.
   Mutually dependent phis read the same pre-update snapshot. Existing loop
   limits, scalar-uniformity checks, growth bounds and refusal boundaries remain.

2. **Exchange only invariant private values outside loops.** Recorded private
   exchanges are complete write/fence/read/fence sequences. Integer identities
   and trivial `phi(seed, self)` aliases expose invariant operands. Hoisting
   requires every dependency to be an available entry value or a speculatable
   pure operation. A load, atomic, nontrivial phi, integer division or unknown
   call blocks hoisting. Entry-region exchanges keep their original order;
   moving past guarded descriptor consumers would break dominance. Each result
   stays in SSA, so later scratch reuse cannot destroy it. If no collective
   remains in a cyclic component, the original console-uniform PC test drives
   a local loop independently in each half. Different console waves may take
   different finite trip counts and meet at the next group barrier. No other
   wave is needed to make that loop progress.

3. **Remove unused exchanges as whole operations.** A private exchange whose
   result is dead can lose its slot accesses and both fences. Liveness treats
   its full sequence as one operation, including the payload dependency that
   a load alone cannot express in SSA. Title effects remain roots even when an
   atomic's returned value becomes dead. The direct regression test checks both
   a used exchange and an unused exchange fed by an observable atomic.

4. **Skip inactive work while retaining required collectives.** Previously all
   N threads executed every region's unguarded arithmetic on every schedule
   traversal, including both alternatives of a branch. If only n threads'
   console PCs matched a region, N-n executed arithmetic they did not need;
   with no matching PC this was all N threads. Effects already had PC/EXEC
   guards. Now exchange-free regions are skipped by a console-uniform PC
   branch. Straight-line regions containing only private exchanges split into
   guarded arithmetic/slot phases with the *same unconditional barriers* between
   them. Both halves share the PC predicate, and no third console wave reads
   that pair's slots. Live paths see the original definitions; inactive joins
   use harmless numeric values that the state commit discards. Complex guarded
   regions and title-LDS phases retain conservative arithmetic execution.
   Scalar instructions cannot in general be skipped merely because EXEC is
   empty: they may restore EXEC or determine later control. An EXEC-exit loop
   that has left its component does skip its body while its neighbour continues.

5. **Return wholly unlaunched groups.** For translations using thread-count
   dispatch masking, all three dimensions check `groupID * groupSize < count`.
   This group-uniform test dominates the exchanges and body resource loads.
   A wholly uncovered group returns; a partially covered group still sends all
   launched threads to required barriers. Zero coverage and a covered group
   are both checked by emitted-graph execution.

6. **Batch continuation and cancellation.** Title-communicating schedules used
   two votes, costing four barriers per global traversal. Active and abort are
   now bits 0 and 1 in the same slot write and the same representative-slot
   reads. Bitwise OR preserves both predicates independently, with two barriers
   and no additional storage. Abort remains group-uniform and cancels the group
   before any later title phase. Ordinary B/C/H already had only the active
   vote. General payload exchanges are not merged speculatively: they may
   depend on each other's results and reuse the same scratch. No extra payload
   arrays were added to the already large communicating programs.

The final verification cuts generated backedges, treats collective header tests
and cancellation as group-uniform, and proves each retained barrier mandatory
on the resulting traversal. Local loops contain no barrier. A separate full
SSA dominance/phi-predecessor check runs on the actual cyclic output. Any failed
proof refuses translation. Both host-wave arrival orders are exercised in the
integer executor; these tests do not emulate Metal's optimizer or scheduling.

### Pair-local handshake decision

No pair spin barrier was introduced. Release/acquire atomics can establish
visibility if the peer runs, but do not establish that both host waves of a
console wave are resident or eventually scheduled while one spins. There is no
validated Metal occupancy/forward-progress bound available in this offline
environment. A finite poll timeout cannot safely permit a dependent read; an
unbounded poll does not prove termination. A generation protocol alone would
not resolve that missing premise. Required cross-half exchanges therefore
retain group barriers. This adds no synchronization atomics or polling loops.

## Communicating programs

The same component schedule applies to round 3. These values include retained
title-LDS ordering, which can require UAV as well as TGSM fences. This change
does not weaken those fences or assume pair-local title storage.

| Class / containers | Old prologue + per traversal | New once barriers | New cyclic traversal barriers, including packed vote |
|---|---:|---:|---|
| D `4e9b58cc637d9272`, `b8f3f7979d6bbc50` | 6 + 847*T_old | 820 | 5, 8, 5, 10, 5, 8 |
| D `67353810417f6c19`, `eae3498e6553afa8` | 6 + 864*T_old | 846 | 5, 8, 5, 10 |
| E `0c028454b58543f0`, `eab96ed5d1684967` | 2 + 73*T_old | 70 | 5 |
| F `02384a60fda6f2d0`, `22338086ea4c5364` | 2 + 229*T_old | 82 | 5, 40, 46, 5, 5, 38, 5, 6, 6, 5, 6 |
| G `6832a9da15c2e584` | 6 + 163*T_old | 50 | 6, 5, 80, 5, 8, 5, 20 |
| G `6b82bc7d22016cb5` | 6 + 217*T_old | 50 | 6, 5, 134, 5, 8, 5, 20 |
| J `c6a4ca484a40cc64`, `fd6e1cbec58e93a9` | 94*T_old | 92 | None |

Add two terminal-vote barriers for each listed cyclic component, as above.
Five of F's components are title-barrier waits: these do not replay unrelated
loops. F hoists ten private exchanges. G's allocation remains **36 KiB**:
32 KiB title LDS plus 4 KiB existing exchange scratch. This round reduces
repeated synchronization/work; it does not solve G's shared-memory occupancy
cost. Title atomic sites remain effects, with their original predicates.

## Corpus and validation

The same 19,104-container full scan, with `wave32_host` and `zero_copy_memory`,
has **19,081 translated and 23 refused** before and after. There are no status
regressions, changed refusal messages, additions or omissions. Exactly 24
fingerprints changed: the twelve B/C/H entries and twelve D/E/F/G/J entries in
the tables above. Each changes because its synchronized schedule is replaced;
hoisting, dead-exchange removal and phase guards apply only on that path.
All seven A/round-1 fingerprints and the other 19,050 successful fingerprints
are unchanged. The fingerprint covers DXIL and host description, not timing.

The required unfiltered shader suite reports **464 passed, 1 failed, 3 ignored**.
The sole failure is the known
`emit::wave32_tests::a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves`,
with `(6, true)` versus `(0, false)`. Excluding exactly that test passes the
remaining library tests, all five sweep tests and doctests.

The application build and requested GPU test-compilation command were both
attempted. Both currently stop at `E0063` in
`crates/victory-gpu/src/shader_cache.rs:302`: that exhaustive `Options` literal
does not initialize the three new fields. This location is outside the user's
allowed GPU edit boundary, so it has been left untouched pending authorization.
The concrete compatibility fix is one line after `integer_targets: 0,`:

```rust
        ..victory_shader::emit::Options::default()
```

That preserves every explicitly set existing option and gives the D3D12 path
the three default-enabled fallbacks. The native environment reader is already
updated. `git diff --check` passes; changes remain uncommitted.

Existing round-1/2/3 emitted-value fixtures still pass. Added coverage checks
invariant exchanges with different wave trip counts and inactive named source
lanes, scratch reuse, an idle console wave beside an EXEC-shrinking neighbour,
zero dispatch coverage, all switch combinations and dead-exchange liveness.
The invariant fixture also counts actual emitted-graph barriers and slot traffic:
neither grows with its trip count. Idle lanes execute fewer arithmetic steps.
New GPU fixture bodies are checked with the native root-constant/memory layout,
and their Rust harness interface was type-checked with a local stub. No GPU
value result is claimed from those CPU checks.

Reproduce with the pinned Rust 1.98.1 toolchain. This clone's offline Cargo cache
is `CARGO_HOME="$PWD/../xcl-tools/cargo"`; use `CARGO_TARGET_DIR=target` throughout.

```bash
nice cargo build -j4 --offline --release --target x86_64-apple-darwin -p victory
nice cargo test -j4 --offline --release --target x86_64-apple-darwin -p victory-shader
nice cargo test -j4 --offline --release --target x86_64-apple-darwin -p victory-gpu --no-run
git diff --check

nice cargo test -j4 --offline --release --target x86_64-apple-darwin -p victory-shader -- \
  --skip emit::wave32_tests::a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves

nice cargo run -j4 --offline --release --target x86_64-apple-darwin \
  -p victory-shader --example sweep -- \
  "$HOME/.cache/victory/cache/native-shaders/containers" -j 4 \
  --list /tmp/native-shader-loop-files.txt > /tmp/native-shader-r4-final.tsv
```

The manifest is needed because these cache filenames have no `.bin` suffix.
Baseline and final rows are `/tmp/native-shader-r4-baseline.tsv` and
`/tmp/native-shader-r4-final.tsv`. Compare status/refusal and fingerprint by
filename, ignoring the elapsed-time column. The existing off-by-default
`VICTORY_TRACE_CONSOLE_FLOW` now prints aggregate original/prologue costs,
hoisted/dead exchanges, component costs and collective counts. Cost arrays are
`[group barriers, TGSM fences, UAV fences, slot stores, slot loads, atomic sites]`.
The new once/body trace costs exclude generated header votes; add those using
the formulas above. It does not print shader bytes or listings.

### Exact Metal fixture commands

The fixture include handoff from the prior rounds is still necessary: this
checkout's `native/split_wave_tests.rs` does not include these fixture modules.
That file is outside this task's allowed edit boundary. The coordinator must
include the following inside the existing native test module before running
the commands, and verify that nonzero fixture tests actually run:

```rust
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../victory-shader/tests/fixtures/predicated_gpu.rs"));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../victory-shader/tests/fixtures/synchronized_gpu.rs"));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../victory-shader/tests/fixtures/communicating_gpu.rs"));
```

Run on native arm64 with all three new switches unset, **not** under Rosetta:

```bash
CARGO_TARGET_DIR=target nice cargo test -j4 --offline --release -p victory-gpu predicated_multi_wave -- --nocapture
CARGO_TARGET_DIR=target nice cargo test -j4 --offline --release -p victory-gpu synchronized_multi_wave -- --nocapture
CARGO_TARGET_DIR=target nice cargo test -j4 --offline --release -p victory-gpu communicating_multi_wave -- --nocapture
```

The synchronized command includes the two new GPU tests:
`hoisted_exchange_survives_different_trip_counts_and_scratch_reuse` and
`an_idle_console_wave_waits_for_its_exec_shrinking_neighbour`.
`SKIPPED: Metal/MSC unavailable` is not a GPU-value pass. Metal execution,
pipeline creation for the changed corpus programs and a world-frame timing
comparison remain pending access to the device.
