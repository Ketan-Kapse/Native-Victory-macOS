# Multi-wave translation, round 5: cost of the communicating programs

27 November 2026, on `7bbd1c45` (round 4 plus the three option fields). It
follows [round 4](multi-wave-cost.md) and the round 3 report
(`docs/shadow-lds-atomics.md` at the repository root). Corpus inputs, sweep rows
and traces stayed outside the repository; only aggregate counts, container
filenames and refusal messages appear here. Unlike rounds 3 and 4, the GPU
fixtures were run on a Metal device this time (results below). No game run and
no frame timing was made.

Round 3 (LDS phases, title barriers, extended atomics, groups above 256
threads) was correct but cost 4,000 ms and more per world frame. This round
re-schedules those programs. Everything here applies only to programs that take
the cheaper communicating schedule (`predicate::shares`): rounds 1 and 2 emit
exactly what they did.

## Counts, before and after

Counts are group barriers of the emitted schedule, from
`VICTORY_TRACE_CONSOLE_FLOW`. "Once" is everything outside cyclic components.
"Per traversal" is one visit of one cyclic component, header vote included.
"Before" is round 4's trace on the same containers (component barriers per
traversal, and its once cost).

| Class | Containers | Once, before | Once, after | Per traversal, before | Per traversal, after |
|---|---|---:|---:|---|---|
| D | `4e9b58cc637d9272`, `b8f3f7979d6bbc50` | 820 | 13 | 8, 10, 8 and 5 per poll of each of 3 title barriers | 5, 9, 5 |
| D | `67353810417f6c19`, `eae3498e6553afa8` | 846 | 14 | 8, 10 and 5 per poll of each of 2 title barriers | 5, 9 |
| E | `0c028454b58543f0`, `eab96ed5d1684967` | 70 | 24 | 5 per poll of the title barrier | no cyclic component |
| F | `02384a60fda6f2d0`, `22338086ea4c5364` | 82 | 55 | 40, 46, 38, 6, 6, 6 and 5 per poll of each of 5 title barriers | 19, 20, 24, 6, 6, 6 |
| G | `6832a9da15c2e584` | 50 | 32 | 6, 80, 8, 20 and 5 per poll of each of 3 title barriers | 6, 20, 5, 10 |
| G | `6b82bc7d22016cb5` | 50 | 32 | 6, 134, 8, 20 and 5 per poll of each of 3 title barriers | 6, 26, 5, 10 |
| J | `c6a4ca484a40cc64`, `fd6e1cbec58e93a9` | 92 | 42 | no cyclic component | no cyclic component |

Full final cost arrays, `[group barriers, TGSM fences, UAV fences, slot stores,
slot loads, atomic sites]`, header votes included:

| Class | Once | Cyclic components |
|---|---|---|
| D (3 title barriers) | `[13, 13, 5, 9, 17, 0]` | `[5, 5, 2, 2, 34, 0]`, `[9, 9, 6, 2, 33, 0]`, `[5, 5, 2, 2, 34, 0]` |
| D (2 title barriers) | `[14, 14, 6, 9, 17, 0]` | `[5, 5, 2, 2, 34, 0]`, `[9, 9, 5, 2, 33, 0]` |
| E | `[24, 24, 11, 20, 29, 0]` | none |
| F | `[55, 55, 35, 18, 30, 0]` | `[19, 19, 8, 16, 28, 0]`, `[20, 20, 8, 15, 26, 0]`, `[24, 24, 12, 10, 22, 2]`, three of `[6, 6, 2, 2, 6, 0]` |
| G `6832...` | `[32, 32, 18, 12, 20, 2]` | `[6, 6, 2, 2, 34, 0]`, `[20, 20, 3, 17, 62, 6]`, `[5, 5, 0, 3, 35, 1]`, `[10, 10, 4, 4, 37, 1]` |
| G `6b82...` | `[32, 32, 18, 12, 20, 2]` | `[6, 6, 2, 2, 34, 0]`, `[26, 26, 3, 23, 74, 6]`, `[5, 5, 0, 3, 35, 1]`, `[10, 10, 4, 4, 37, 1]` |
| J | `[42, 42, 0, 53, 104, 3]` | none |

The exchange scratch is now eight banks of one slot per host half:
`ceil(threads / 64) * 2 * 8` dwords, 1 KiB at 1024 threads where it was 4 KiB.
The two G programs therefore declare 33 KiB of shared storage, not 36 KiB. A
vote stores one flag per host half and still loads every half's slot (32 loads
per thread at 1024 threads).

### The floor

D reaches the loop classes' order (5 to 9 per traversal, 13 or 14 once). The
others do not reach 10, for these reasons:

- **Dependent exchanges.** A batch shares one rendezvous only among exchanges
  none of whose stored values depends on another's result. A chain (a mask
  scan feeding a lane read feeding a count) needs one rendezvous per link.
  This is what is left in F's and G's large components and in J's and E's
  one-time work (J: 53 slot stores behind 42 barriers).
- **Title memory order.** A write that is not a proved constant keeps a fence
  on both sides, and a run of reads keeps one on both sides.
- **Unordered atomics.** Three of G's seven atomic order fences cannot be
  dropped: the atomic directly follows another LDS access with no other barrier
  between them on some path.
- **Votes.** A collective component needs its two vote barriers; a component
  with a limit that can cancel the group votes on both in the same rendezvous.

Going lower needs either a wider batch rule (speculating a dependent exchange,
which this round does not attempt) or knowledge of what each LDS address holds.

## What changed, and what each change must prove

### 1. A title barrier is one barrier, not a poll

Round 3 parked each wave at a title `s_barrier` and polled, with a group vote
per poll, until every wave's PC had arrived. Round 4 already runs the cyclic
components to completion one after another in topological order.

Obligation: when the schedule reaches the barrier's region, every console
wave's PC is at it. Proof: `once_blocks` (unchanged) requires the barrier's
block to be outside every cycle and on every path from entry to exit. Let X be
any block a wave could be waiting in. X lies on some entry-to-exit path, and
that path passes the barrier block B, so either X reaches B (X is before B in
every topological order, was already visited, and the wave left it) or B
reaches X (the wave has not passed B yet, so it cannot be there). Components
before B have finished, including a loop-limit exit, which cancels the whole
group at its vote. So the wave is at B, and one group barrier is the phase
fence. A barrier that fails `once_blocks` is still refused.

### 2. LDS fences per phase, not per access

Round 3 put a group barrier before and after every LDS access. Now
`lds_phase_barriers` places them, and the invariant is:

> Every two LDS accesses that one console wave makes in program order keep a
> group barrier between them, except inside a run of reads and inside a run of
> stores of one proved constant.

Obligations:

- **Reads commute.** No interleaving of the two host halves changes what a run
  of reads returns, because nothing in the run writes. Any other LDS
  instruction or a title barrier ends the run.
- **Stores of one constant commute, even where addresses alias.** Each written
  address ends up holding that constant in any order. `uniform_lds_store`
  proves the constant: every dword of the write comes from a `v_mov_b32` of an
  immediate, in the same block, with no title barrier between, under an EXEC
  that contains the store's EXEC. A phi, a value from another block, or a lane
  that could hold an older value fails the proof, and the write gets both
  fences. This is what removes D's roughly 800 initializer barriers.
- **Anything else is fenced.** Another write has a fence on both sides. An
  atomic has a fence before it unless the block's previous access already left
  one. A block's first access is always fenced, so nothing is assumed about the
  block the wave came from. A title barrier is a fence itself.

The fence keeps round 3's flags (thread group, TGSM, and UAV where the program
writes UAVs).

### 3. An atomic's order fence goes only where another barrier does its work

The interrupted first draft of this round removed every fence around LDS
atomics. That is not proved: two atomics of one wave with nothing between them
could then run in either order across the two host halves, which the console
never does. The fence before an atomic is back, and `elide_order_fences`
removes it only with a proof on the emitted schedule:

Obligation: on every path that arrives at the fence, a group barrier that
stays in the program lies after the last access to title memory (or there is
no such access). The pass walks the final CFG backwards over every edge,
backedges included. It does not count the other candidate fences, and does not
count scratch release barriers, because the next pass may remove those. Memory
that is not provably the exchange scratch counts as title memory. On the
corpus this proves 2 of 2 fences redundant in F and 4 of 7 in G; the other
three stay.

### 4. Batched scalar exchanges

`batch_exchanges` merges consecutive exchanges of one straight-line block:
all stores, one barrier, all loads, one barrier.

Obligations:

- **No dependence.** A store whose value depends on a load of the batch ends
  the batch (`exchange_batch_phases`).
- **Nothing else moves.** Between the exchanges only pure arithmetic, scratch
  loads and stores, `ReadNone` calls, `waveActiveBallot` and `waveReadLaneAt`
  may stand. A resource access, an atomic, any other barrier or a division
  ends the batch. No instruction crosses a CFG edge.
- **Disjoint slots.** Each exchange of a batch gets its own bank, at most
  eight, inside the declared array. Every slot index is a host half
  (`thread >> 5`), so `bank * halves + half` stays in range.
- **One value per slot.** All threads of a host half store the same value: a
  host-wave ballot, a `waveReadLaneAt` of a wave-uniform lane, or a vote that
  is uniform per console wave. Round 3's vote already read one representative
  per half.

### 5. Scratch releases

The barrier after an exchange's loads only protects the scratch from the next
store. `elide_exchange_releases` drops it where every path from it to a store
into the same bank crosses another group barrier that stays. Unannotated
scratch stores (votes, reductions) count as aliasing every bank. A barrier is
a release only if loads precede it and the exchange has a separate barrier
between its stores and loads; the walk treats every other release as already
gone and covers every reachable edge.

### 6. Mask words instead of repeated ballots

`wide_mask` evaluates a mask's BDD on the two 32-bit words of the console
mask. A new ballot is taken only for a per-lane compare; scalar EXEC edits,
uniform conditions and constants reuse words.

Obligations:

- **Same bits.** Each BDD node is `atom ? high : low`, computed bitwise. A
  uniform atom is all ones or zero, a constant is its words, a lane atom is
  its full-console ballot. Leaves are zero or the group's existing lanes, so
  the result never has a lane the group lacks.
- **Joins.** A mask phi carries both words through the console edge, which is
  the same for the whole console wave. A join of a mask with data is refused
  (`a register joining a lane mask and data, read as data`).
- **Where a cached ballot lives.** The schedule carries a value from the last
  visit of its region, so a cached ballot must sit in a region every user is
  behind. A lane atom's ballot goes to the block that defines the lane value.
  The first draft could place it in an edge's own region while filling a join
  and then reuse it elsewhere, where a wave that did not take that edge would
  read a stale ballot. It now goes to the defining block's tail.
- **Complete joins.** Filling one kind of join can request the other; both
  queues are drained together, so no word phi is left without its edges.

### 7. Early return and idle waves

Unchanged from round 4 and now tested for this class: a fully masked dispatch
returns before any exchange, vote or effect
(`a_masked_communicating_dispatch_returns_before_any_exchange_or_effect`).

## Class J on the GPU

The fixture was wrong, not the lowering. `global_atomic_*` takes a signed
12-bit offset, so the fixture's `0x800` was -2048 while the test put its data
at +2048: the atomic addressed memory outside the mapped input and read zero.
The fixture now adds 2048 to the address register and uses offset 0. A decoder
test (`root_atomic_fixtures_use_a_positive_address_not_a_negative_immediate`)
pins this, and four small GPU fixtures isolate one atomic each: add and
unsigned min, root address and descriptor form.

## Results

Run on an Apple GPU through Metal, x86_64 under Rosetta, release:

- `cargo test -p victory-gpu --lib`: 504 passed, 0 failed, 8 ignored.
- `predicated_multi_wave`: 4 of 4. `synchronized_multi_wave`: 8 of 8.
  `communicating_multi_wave`: 9 of 9, which includes
  `per_lane_buffer_add_and_unsigned_min_keep_all_results` and the four new
  `single_*_atomic_*` fixtures.
- `split_wave_tests::wave64_lane63_writelane_and_salu_or_reductions_run_on_the_gpu`
  asserted that a lane read behind an EXEC branch is refused. If-conversion
  translates it now, so that one assertion is gone and the value comparison
  after it stays.

Shader crate: every test passes except the known
`emit::wave32_tests::a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves`.
Clippy for the crate is clean with `-D warnings`.

Full scan of the 19,104 cache containers: 19,081 translated, 23 refused, as
before. No container changed outcome. Against a sweep of the tree before this
round, exactly twelve fingerprints changed: the twelve round 3 containers in
the first table.

## The seven still refused

Nothing in this round changes them. From the round 3 analysis:

- **G `1d434307d981352a`, `3b02415f67c6b0d8`** (`a cross-lane operation
  between host waves in divergent code`). They have 51 and 96 first-lane reads
  that select a descriptor (waterfalls), plus cyclic title barriers. They need
  a descriptor waterfall under the PC schedule: every console wave's index
  exchanged once, then a guarded access per distinct index, with a proof that
  the index is uniform per console wave at each use. They then meet the cyclic
  barrier problem below as well.
- **G `2365ac7d30dd6f0c`, `ea5d5c9a44e9cd1f`** (`0x10c`) and
  **G `4869d261ac95b961`, `dbf247e6efc048df`** (`0xec`): a title barrier inside
  a worklist loop. The console lets each wave pass the barrier a different
  number of times. A translation needs a rule for a wave that has left the
  loop while others still meet at the barrier (it must keep attending without
  effects), and a proof that waves which do attend see the same phases. The
  round 3 poll, kept per iteration with a group vote on "any wave still in the
  loop", is the likely shape; the proof that it matches a console order is the
  missing part.
- **I `b0d5ffa0801b7450`** (`0x218`): a title barrier inside an image and LDS
  loop. Same missing proof as the four above, with image effects between the
  phases, so the fence must also order UAV memory.

## Reproduce

Pinned Rust 1.98.1, offline, `--release --target x86_64-apple-darwin`:

```bash
cargo build -p victory
cargo test -p victory-shader
cargo test -p victory-gpu --lib predicated_multi_wave
cargo test -p victory-gpu --lib synchronized_multi_wave
cargo test -p victory-gpu --lib communicating_multi_wave
cargo test -p victory-gpu --lib
cargo clippy --all-targets --locked --target x86_64-pc-windows-gnullvm -p victory-shader -- -D warnings
git diff --check

VICTORY_CACHE_DIR=target/scan-cache target/x86_64-apple-darwin/release/victory \
  translate --list --jobs 6 ~/.cache/victory/cache/native-shaders/containers

VICTORY_TRACE_CONSOLE_FLOW=1 target/x86_64-apple-darwin/release/examples/sweep \
  ~/.cache/victory/cache/native-shaders/containers -j 1 --list <the 19 round 3 filenames>
```

The trace's last line per program is
`console schedule: final once [...]; final loops [...]`, the arrays above.
