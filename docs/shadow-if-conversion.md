# Tiled cascade shadows: bounded multi-wave if-conversion

Round 2 adds [synchronized loops and a separate proof/report](shadow-synchronized-loops.md).
The measurements and dispositions below are the original round 1 baseline and result.

Audit of `02ca25fc` and this uncommitted change, 27 November 2026. All inspection was
local and offline. This report contains identities and control-flow findings, not title
shader bytes or disassembly. Only the shader crate, its test fixtures and documentation
are changed. No GPU or game was run in this sandbox.

## Corpus result

The native cache contains 19,104 containers. Each has one native program in this audit.
The exact original baseline was recorded before editing the translator. Both versions
were also run through the native `victory translate --list --jobs 4` entry point, with
scratch `VICTORY_CACHE_DIR` directories under this clone's `target`.

| Outcome | Baseline | Final |
|---|---:|---:|
| Translated | 19,050 | 19,057 |
| Refused, all reasons | 54 | 47 |
| Cross-lane operation between host waves in divergent code | 38 | 31 |
| Other refusals | 16 | 16 |
| Previously translated programs now refused | — | 0 |
| Previously translated programs with changed output fingerprints | — | 0 |

These are cache-container counts, not the last run's 23 shader-set count. All 22 explicitly
listed identities in the task occur in this cache. It also contains other programs with
the same refusal. The table below inventories all 38.

Exactly these **container filenames** change outcome, each from the divergent-exchange
refusal to `translated Cs`:

- `191ecaed66eea3b1`
- `1d44a723bdebbf27`
- `240cb516c1af0bc3`
- `31d703207b51549a`
- `65de113e23fa4fdd`
- `804b342cb4b9d1e9`
- `ab5455fc6abb804f`

No other outcome changes. The full sweep also compares host-description/DXIL fingerprints
for every previously successful program. The existing early-exit retry retains priority;
otherwise two already successful containers would unnecessarily take this new path.

## Shapes and result use

The reported offset is the **first failing consumer**, which can be a branch that lazily
materializes a ballot. It is not necessarily a literal lane-read instruction. A branch
site outside a cycle does not make its whole program acyclic.

| Class | Count | Observed shape and use | Disposition |
|---|---:|---|---|
| A | 7 | Acyclic nested EXEC skips driven by vector/image comparisons. Mask population (`s_bcnt1_i32_b64`), mask boolean operations and VCC/SCC tests decide subsequent shadow-sampling paths. Whole-wave lane operations feed these decisions and scalar/vector joins. Groups have two or four console waves. | Translates through if-conversion. |
| B | 9 | The same shadow paths, plus scalar loops over masks and ranges. First-lane results seed masks; `s_ff1_i32_b32` selects a bit, which is removed, or computes a range index/bound with scalar-buffer data. `20b78ee12ecdb57d` and `f133e35efed2a48a` also have fixed-limit loops. The first reported branch is outside the cyclic blocks. | Refused: no multi-wave loop transformation or proof that all waves take equal iterations. |
| C | 2 | Shadow variants with scalar counters ending at 12; first reported SCC/VCC branches are outside those loops. | Refused: even fixed-trip loops need a checked unroll or synchronized loop lowering, not the acyclic path. |
| D | 4 | Helper-style 1024-thread LDS producers with scalar loops and barrier phases. The named `b8f3f7979d6bbc50` and `eae3498e6553afa8` helpers have counters ending at 25 and a counter comparison against 15. Mask scans/counts elect lanes and compute LDS destinations. The first failing instruction is an LDS write, whose data is consumed by later LDS reads/phases. | Refused: loops, LDS communication and sixteen console waves. |
| E | 2 | Acyclic 256-thread LDS/barrier phases followed by EXEC joins, population counts and SCC decisions. | Refused: LDS ordering is outside this proof even though the CFG is acyclic. |
| F | 2 | 128-thread per-lane loops narrow EXEC; scans/prefix work feeds LDS read/write2st64 offsets and later work. | Refused: cyclic control flow and LDS communication. |
| G | 8 | 512/1024-thread worklist loops with elected `ds_add_rtn_u32` atomics. Returned allocation/index values are broadcast with first-lane reads and used by subsequent work. | Refused: loops, LDS atomics and groups above the bound. |
| H | 1 | 128-thread loops containing whole-wave permutes, lane reads and mask scans, with an EXEC loop/skip decision. | Refused: cyclic, even without LDS. |
| I | 1 | A 256-thread image-load loop with population-count/SCC decisions and LDS. | Refused: cyclic and communicating through LDS. |
| J | 2 | Acyclic 1024-thread programs with divergent counts and buffer atomics (add/unsigned-min); atomic results participate in allocation/indexing. | Refused: atomics and sixteen console waves. |

All programs in this table are wave64 on a 32-lane host. Dimensions are thread-group
`x×y×z`; the following number is console waves per group. “Site cyclic / cyclic blocks”
distinguishes the first refusal's location from loops elsewhere in the program.

| Container filename | Dimensions / waves | First reported offset and instruction | Site cyclic / cyclic blocks | Class |
|---|---|---|---|---|
| `02384a60fda6f2d0` | 128×1×1 / 2 | `0x288` `ds_write2st64_b32` | yes / 54 | F |
| `0c028454b58543f0` | 16×16×1 / 4 | `0x6e8` `s_cbranch_scc1` | no / 0 | E |
| `191ecaed66eea3b1` | 16×16×1 / 4 | `0x3f8c` `s_cbranch_scc0` | no / 0 | A |
| `1d434307d981352a` | 512×1×1 / 8 | `0x1bc` `ds_add_rtn_u32` | yes / 283 | G |
| `1d44a723bdebbf27` | 16×16×1 / 4 | `0x484c` `s_cbranch_scc0` | no / 0 | A |
| `20b78ee12ecdb57d` | 8×16×1 / 2 | `0x2568` `s_cbranch_scc0` | no / 84 | B |
| `22338086ea4c5364` | 128×1×1 / 2 | `0x288` `ds_write2st64_b32` | yes / 54 | F |
| `2365ac7d30dd6f0c` | 1024×1×1 / 16 | `0xf4` `ds_add_rtn_u32` | yes / 3 | G |
| `240cb516c1af0bc3` | 16×16×1 / 4 | `0x4170` `s_cbranch_scc0` | no / 0 | A |
| `31d703207b51549a` | 16×16×1 / 4 | `0x16c0` `s_cbranch_scc0` | no / 0 | A |
| `3b02415f67c6b0d8` | 512×1×1 / 8 | `0x1ac` `ds_add_rtn_u32` | yes / 515 | G |
| `466ec41d3f9d24c0` | 16×16×1 / 4 | `0x2b74` `s_cbranch_scc0` | no / 24 | B |
| `4869d261ac95b961` | 1024×1×1 / 16 | `0xd4` `ds_add_rtn_u32` | yes / 3 | G |
| `4e9b58cc637d9272` | 1024×1×1 / 16 | `0xf8c` `ds_write_b32` | yes / 8 | D |
| `56f3bf2d9388e5a7` | 8×16×1 / 2 | `0x2c4c` `s_cbranch_scc0` | no / 24 | B |
| `65de113e23fa4fdd` | 16×16×1 / 4 | `0x2ab0` `s_cbranch_scc0` | no / 0 | A |
| `67353810417f6c19` | 1024×1×1 / 16 | `0xf7c` `ds_write_b32` | yes / 5 | D |
| `6832a9da15c2e584` | 1024×1×1 / 16 | `0x348c` `ds_add_rtn_u32` | yes / 97 | G |
| `6b82bc7d22016cb5` | 1024×1×1 / 16 | `0x3a14` `ds_add_rtn_u32` | yes / 150 | G |
| `804b342cb4b9d1e9` | 8×16×1 / 2 | `0x2ae0` `s_cbranch_scc0` | no / 0 | A |
| `922e1d0ee1e19a85` | 8×16×1 / 2 | `0x1d58` `s_cbranch_vccz` | no / 18 | C |
| `a953d5b93efe8424` | 128×1×1 / 2 | `0x2fc` `s_cbranch_execz` | yes / 6 | H |
| `ab5455fc6abb804f` | 16×16×1 / 4 | `0x1c44` `s_cbranch_vccz` | no / 0 | A |
| `abd7a22106898c66` | 16×16×1 / 4 | `0x4930` `s_cbranch_scc0` | no / 24 | B |
| `b0d5ffa0801b7450` | 16×16×1 / 4 | `0x16c` `s_cbranch_scc1` | yes / 6 | I |
| `b4b0427232707598` | 16×16×1 / 4 | `0x4218` `s_cbranch_scc0` | no / 24 | B |
| `b8f3f7979d6bbc50` | 1024×1×1 / 16 | `0xf9c` `ds_write_b32` | yes / 8 | D |
| `be6ce5d849d51127` | 8×16×1 / 2 | `0x2434` `s_cbranch_scc0` | no / 18 | C |
| `c6a4ca484a40cc64` | 32×32×1 / 16 | `0x1090` `buffer_atomic_add` | no / 0 | J |
| `ce79e9eba7ae0348` | 16×16×1 / 4 | `0x40c4` `s_cbranch_scc0` | no / 24 | B |
| `d7db57731ada919b` | 16×16×1 / 4 | `0x1ce4` `s_cbranch_vccz` | no / 24 | B |
| `dbf247e6efc048df` | 1024×1×1 / 16 | `0xd4` `ds_add_rtn_u32` | yes / 3 | G |
| `ea5d5c9a44e9cd1f` | 1024×1×1 / 16 | `0xf4` `ds_add_rtn_u32` | yes / 3 | G |
| `eab96ed5d1684967` | 16×16×1 / 4 | `0x71c` `s_cbranch_scc1` | no / 0 | E |
| `eae3498e6553afa8` | 1024×1×1 / 16 | `0xf8c` `ds_write_b32` | yes / 5 | D |
| `f133e35efed2a48a` | 8×16×1 / 2 | `0x1eb4` `s_cbranch_vccz` | no / 84 | B |
| `fa0bd10a35bfef3c` | 16×16×1 / 4 | `0x179c` `s_cbranch_scc0` | no / 24 | B |
| `fd6e1cbec58e93a9` | 32×32×1 / 16 | `0x1090` `buffer_atomic_add` | no / 0 | J |

## Lowering and invariants

`emit/lower/predicate.rs` extends the existing console-flow lowering. It is a fallback
only for the exact divergent-exchange refusal, with 65–256 threads and an acyclic CFG.
LDS allocation/communication, atomics, waterfalls, indirect flow, scalar writes to memory,
and conditional execution-mode changes are excluded. Mode writes already reached by every
wave retain the existing mode analysis. Unknown or unsupported operations still refuse.

All original blocks execute in topological order. Each original edge carries a predicate
uniform over one complete console wave, using the existing full-EXEC ballot/branch helpers.
Different console waves may have different predicates. Joins select on these edge predicates
for scalar data, SCC, both mask words, descriptors and spills. Vector writes additionally
retain EXEC. Named lane reads can therefore see inactive lanes in the other half;
first-lane reads use the full console EXEC, including an empty lower half.

Resource accesses, including scalar loads, are guarded by the wave predicate. Their results
join before use outside the guard. No cross-half barrier may be inside a resource guard.
Every wave uses disjoint slots in the existing exchange array; other console waves never
contribute to its ballot, lane read or reduction. The two existing barriers per exchange
bring the whole group together. No new synchronization primitive or guessed lane value is
introduced.

A final emitted-CFG check rejects cycles or any barrier absent from an entry-to-return path.
A separate SSA check requires every written value to dominate every use, including phi
incoming edges and values produced by resource guards. Failures return the original refusal.
The existing `VICTORY_NO_CONSOLE_FLOW=1` switch disables this fallback. Existing off-by-default
`VICTORY_TRACE_CONSOLE_FLOW` output identifies candidate successes and failed proofs.

## Verification and coordinator handoff

The required release build succeeds. The unfiltered shader suite reports **434 passed,
1 failed, 3 ignored**. Its sole failure is the pre-existing
`emit::wave32_tests::a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves`
(same baseline assertion: `(6, true)` versus `(0, false)`). The original baseline was
425 passed, 1 failed, 3 ignored. All nine new tests pass. A separate run excluding only
that known failure completes the remaining targets. `git diff --check` is clean.

The synthetic GFX10 tests cover named lane reads across an inactive half, first-lane reads
and counts with an empty lower half, scans, scalar and 64-bit mask joins, guarded scalar
loads, and refusal of loops/LDS/conditional mode changes/oversized groups. The CFG proof
also has negative tests for guarded barriers and values escaping guards without a phi.
No title words are copied into tests.

The four GPU value tests are in
`crates/victory-shader/tests/fixtures/predicated_gpu.rs`, sharing synthetic words with
`predicated_cases.rs`. The coordinator can append this one test-only include to the existing
`crates/victory-gpu/src/native/split_wave_tests.rs`; no production GPU changes are needed:

```rust
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../victory-shader/tests/fixtures/predicated_gpu.rs"));
```

Then run on the native arm64 host (not under Rosetta):

```bash
CARGO_TARGET_DIR=target cargo test -j4 --offline --release -p victory-gpu predicated_multi_wave -- --nocapture
```

They check every output lane for 128-thread split waves and 96/128/256-thread groups,
including distinct results in different console waves. Reads in partial groups name an
existing lane; they do not assume values for unlaunched lanes. The existing harness skips
when Metal/MSC is unavailable, so an empty/skipped GPU run is not value validation.

The existing GPU test `wave64_lane63_writelane_and_salu_or_reductions_run_on_the_gpu` also
contains an assertion that a two-wave program with a divergent named-lane read must refuse.
That assumption needs reevaluation when integrating this fallback; preserve its GPU value
comparisons. No GPU test file was changed in this clone.

Pipeline creation, GPU values, dispatch cost and visible sun shadows remain unverified.
The coordinator should run the four fixtures, create pipelines for the seven recovered
containers, and run the game before claiming a rendering fix. Both arms' arithmetic runs
for every wave, so translation success alone says nothing about runtime cost.

## Local reproduction

Use Rust 1.98.1, offline dependencies, `CARGO_TARGET_DIR=target`, and four build jobs. Build
and test commands used here:

```bash
cargo build -j4 --offline --release --target x86_64-apple-darwin -p victory
cargo test -j4 --offline --release --target x86_64-apple-darwin -p victory-shader
git diff --check
```

The native command is:

```bash
VICTORY_CACHE_DIR="$PWD/target/shadow-cache" \
  target/x86_64-apple-darwin/release/victory translate --list --jobs 4 \
  "$HOME/.cache/victory/cache/native-shaders/containers"
```

The native scanner parallelizes programs inside a file, not files. For this audit the
19,104 files were divided round-robin into four disjoint temporary directories of 4,776
regular file links each, outside the repository. Each directory was scanned by an unchanged
native executable; the rows were combined and deduplicated by container identity, verifying
19,104 distinct filenames. Symlinks are skipped by the scanner. An initial single-process
baseline scan was terminated before totals; it is not counted as completed evidence.

The faster `victory-shader` sweep was run before and after, four workers, with an explicit
`--list` of all filenames because these cache files have no `.bin` suffix:

```bash
cargo run -j4 --offline --release --target x86_64-apple-darwin \
  -p victory-shader --example sweep -- \
  "$HOME/.cache/victory/cache/native-shaders/containers" -j 4 \
  --list /tmp/native-shader-all-files.txt > /tmp/shadow-sweep.tsv
```

Raw native lists, sweep fingerprints, and local inspection summaries remain outside the
repository under `/tmp/native-shader-*`. Native aggregate rows are
`/tmp/native-shader-native-baseline.tsv` and `/tmp/native-shader-native-final.tsv`;
full fingerprint comparisons use `/tmp/native-shader-baseline-sweep.tsv` and
`/tmp/native-shader-final-verified-sweep.tsv`.
