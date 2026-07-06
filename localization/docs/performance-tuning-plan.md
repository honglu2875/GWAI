# Performance Tuning Plan

> Status: initial plan plus first implementation pass, 2026-07-06.
> Measurements below are from one 240-core GCP machine.  The first pass landed
> release debug symbols, stable-graph phase attribution, uncapped graph
> workers, graph-level dynamic scheduling, product ray parallelism, sparse
> entry accumulation, and parallel bidegree Birkhoff products.  The remaining
> tier ordering is still the working roadmap.

A cross-backend tuning plan grounded in release-build profiling, complementing
the harness rows in [performance-frontiers.md](performance-frontiers.md).
Where that document maps *which workloads* hit the one-minute frontier, this
one maps *where the cycles actually go* and orders the fixes by leverage.

## Where the cycles go (measured)

Release build, 240-core GCP machine, `perf` sampling, 2026-07-06.

**Rational tier** — `compute --n 1 --g 4 --d 2 --insert tau10(H)`, cold cache
66.5s total:

- ~50s: stable-graph generation for `(g,n)=(4,1)`, entirely serial, invisible
  in the `GW_PROFILE` phase breakdown (it happens under `prepared_stable_graphs`
  before the graphs phase timer).
- 15.7s: graph contraction on the default 8 workers.  With `GW_THREADS=64` the
  same contraction takes 3.2s wall, but at ~30% parallel efficiency.
- Instruction profile of the contraction phase:
  - ~55% in `num-bigint` arithmetic, of which ~40 points are GCD reduction
    (`biguint_shr2` + `sub_assign` + `gcd` — Stein's binary GCD driven by
    `num-rational`'s reduce-on-every-op),
  - ~30% in the allocator (tcmalloc is already `LD_PRELOAD`ed machine-wide, so
    this is churn volume, not allocator quality),
  - <4% in the graph-walk logic itself (`accumulate_graph_factors`,
    `QSeries::mul` frames).

**Factored/equivariant tier** — `twisted --n 1 --twist -1,-1 --g 1 --d 1
--insert tau2(H) --equivariant` (the README's stress case): killed after 16.5
minutes, still inside the factored Birkhoff calibration (no
`GW_PROFILE factored_twisted_calibration` line had printed), pinned on one
core, with **86% of samples inside Stein GCD**.  The symbolic frontier is not
graph combinatorics; it is BigRational reduction cost inside
`FactoredRatFun`/`SparsePoly` arithmetic.

**Backend microbenchmark** — num-bigint 0.4 vs GMP (`rug`), same operands:

| operand bits | num-bigint gcd | rug (GMP) gcd | ratio |
|---:|---:|---:|---:|
| 256 | 2.6us | 1.1us | 2.4x |
| 4096 | 100us | 19us | 5.3x |
| 65536 | 18.2ms | 1.3ms | 14.3x |

On fused mul–add chains where operands grow (the shape of every q-series
convolution), `BigRational` degrades 350–970x relative to `rug::Rational` at
identical operand sizes.  GMP's Lehmer/HGCD subquadratic GCD plus in-place
mutation is the difference.

## The levers, ordered

### Tier 0 — measurement hygiene (hours)

- The frontier harness times **debug** builds by default; algebra-heavy rows
  distort badly under debug (bigint inner loops are several times slower and
  allocation costs shift).  Keep debug rows for CI stability if desired, but
  record `--release` rows for tuning decisions.
- Add stable-graph generation to the `GW_PROFILE` phase output; today ~75% of
  a cold high-genus run is unattributed.
- Set `debug = true` in `[profile.release]` so `perf` gets symbols and inline
  frames without a separate profile.

### Tier 1 — cross-cutting, high leverage (days each)

1. **Bignum backend swap** (`algebra.rs` only — `Rational` is already a
   newtype and no `BigInt`/`BigRational` leaks outside the file).  Put a
   GMP-backed `rug::Rational` behind a cargo feature, keep `num` as the
   default/fallback for pure-Rust builds.  Expected: 1.5–2.5x on the rational
   contraction tier (~40% of it is GCD), and one order of magnitude or more on
   the factored/equivariant paths (86% GCD).  Every backend — twisted,
   product, bundle, psi — inherits the win.  License note: GMP is LGPL,
   dynamically linked; the feature gate keeps the MIT/Apache default intact.

2. **Parallelism overhaul in the contraction engine** (`givental/graph.rs`):
   - Landed: default worker count is `available_parallelism()` capped by work items,
     not by the hardcoded `MASTER_DEFAULT_MAX_WORKERS = 8` (this machine has
     240 cores; `GW_THREADS=64` already gives 5x today).
   - Landed partly: static contiguous `chunks()` over graphs are replaced with
     graph-level dynamic dispatch via a shared `AtomicUsize`.  Remaining:
     flatten to **(graph, coloring-orbit)** units, largest-estimated-cost
     first, to fix the single-value path's `worker_count = f(graph count)`
     degeneracy.
   - Landed: `product.rs::reconstruct_bidegree_invariants`'s ray loop is
     parallelized —
     `total_degree + 1` fully independent engine runs, mirroring the
     `thread::scope` pattern `bundle.rs` already uses (frontier doc item 2).
   - Landed for projective bundles: bidegree Birkhoff Laurent matrix products
     run in deterministic parallel chunks.  Remaining: parallelize the
     factored calibration's series-matrix products and one-variable Birkhoff
     steps entry-wise; the 16-minute equivariant calibration is still
     single-threaded matrix algebra.

3. **Stable-graph generation** (`graphs.rs`; frontier doc item 1, the
   `g=4,m=1` timeout):
   - Parallelize over `(vertex_count, edge_count)` buckets and over top-level
     first-edge branches of `for_each_connected_edge_multiset`; merge in
     deterministic bucket order so output order and the disk-cache format are
     unchanged.
   - Kill the per-candidate allocations: `DisjointSet` is **cloned for every
     candidate edge at every recursion node** — replace with a rollback DSU
     (union by rank + undo stack); hoist `edge_valence` clones; iterate
     `compositions`/leg assignments lazily instead of materializing.
   - The orderly-generation rewrite (README TODO) remains the asymptotic fix;
     the two items above are cheap and likely worth 1–2 orders of magnitude on
     this hardware first.

### Tier 2 — allocation and representation (a week-ish, compounding)

4. **In-place `Coeff` ops**: add `add_assign`, `mul_assign`, and a fused
   `accumulate(&mut self, a, b)` (acc += a*b) with default impls; use them in
   `QSeries::{add,mul}`, `SeriesMatrix::mul`, and the contraction leaf
   accumulation.  Directly attacks the ~30% allocator share; with GMP the
   in-place ops also avoid temporary re-reductions.

5. **Vertex-contribution cache**: store `Arc<QSeries<C>>` so the 5.3M cache
   hits in the genus-4 run stop deep-cloning series; build keys without
   re-allocating (SmallVec + sort once).

6. **`SparsePoly` mechanics**: initial `entry`-API accumulation landed for
   sparse polynomials and nearby Laurent helpers.  Remaining: consider
   SmallVec-backed `Monomial` (most monomials have ≤4 factors) and broader
   in-place coefficient accumulation.

7. **`FactoredRatFun` keys**: intern denominator factors (a per-process
   `SparsePoly -> u32` table like the existing symbol interner); term keys
   become sorted `SmallVec<u32>` so every BTreeMap comparison, sort in
   `normalize_factors`, and clone stops walking whole polynomials.  Implement
   `Sub` without cloning the rhs map.  Optionally cancel a numerator against a
   linear denominator factor by exact division when cheap.

### Tier 3 — structural / algorithmic (bigger bets)

8. **Integer-primitive `SparsePoly`**: terms over `BigInt` with one shared
   rational content, so polynomial arithmetic performs zero GCDs per term and
   normalizes once per operation.  This is the standard CAS representation and
   multiplies with the Tier-1 backend swap; it is also the fix that makes the
   *expanded* `RatFun` tier stop being a trap.

9. **Equivariant results by evaluation + reconstruction**: the fiber-
   equivariant outputs the README describes "often collapse to a finite
   polynomial in the `mu_i`" with dimension-bounded degree.  Evaluate the
   invariant at enough rational `mu` sample points through the *fast rational
   tier*, then solve the (multivariate Vandermonde) system exactly — the same
   trick ray reconstruction already plays for Novikov bidegrees, applied to
   fiber weights.  Guard with one extra sample point as an a-posteriori check,
   and keep the symbolic path as the validation twin.  This turns the
   16-minute-plus stress rows into seconds and sidesteps expression swell
   entirely where the polynomial-collapse hypothesis holds (rational-function
   reconstruction with degree bounds covers the rest).

10. **Genericize the J-calibration solve over `Coeff`** (README TODO) so the
    remaining twisted/series equivariant paths never materialize expanded
    `RatFun` matrices, and share warm calibrations across rays and
    degree/genus sweeps (the product-ray analogue of what bundles already do).

## Expected effect per frontier row

| frontier row (frontier doc) | dominant cost (measured/read) | tier items | plausible gain |
|---|---|---|---:|
| formula/stable graphs `g=4,m=1` timeout | serial enumeration + DSU clones + IR filter | 3 (then 8/orderly) | 10–50x |
| twisted equivariant stress (`g>=1` symbolic) | BigRational GCD in factored calibration, serial | 1, 2, 7, 9 | 10–1000x |
| product `g=2,d=3` was 19.4s, now 5.0s | serial rays x graph contraction | 2 landed; 1 remains | Product is no longer a leading sampled frontier. |
| bundle rank-3 was 34s, now 22s | bidegree Birkhoff improved; calibration correction and per-ray fundamental-S remain | 2 partly landed; 1, 10 remain | Further 2–5x likely needs shared ray calibration or bignum work. |
| ordinary `P^n` genus scaling | GCD + alloc churn in contraction | 1, 2, 4, 5 | 3–10x |
| psi `g=10` 6.2s | DVV recursion over big rationals | 1 (free rider) | 2–5x |

## Validation workflow

Unchanged from the crate's discipline: every tuning change gates on
`cargo test`, `gw-pn tests`, the A/B escape hatches
(`GWAI_DISABLE_RATIONAL_GRAPH`, `GWAI_DISABLE_FACTORED_GRAPH`), and a
frontier-harness pass against a saved baseline
(`scripts/run-perf-frontiers.sh --save-baseline` before, `--baseline` after —
adding `--release` rows for algebra-sensitive changes).  Determinism caveats:
parallel work-stealing must reduce results in a fixed order (sum per unit into
per-worker accumulators, then combine in unit order) so outputs stay
bit-identical, and the graph generator's output order is part of the disk
cache format.
