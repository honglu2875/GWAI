# Architecture

This document maps the crate's layers and points into the code, so a reader
can find where a concept lives without spelunking.  It complements
[lessons.md](lessons.md), which records the non-obvious findings behind
several of these design choices.

The crate computes Gromov–Witten invariants in exact rational arithmetic by
Givental–Teleman reconstruction: a semisimple CohFT is reassembled from its
genus-zero data (an R-matrix acting on a topological field theory) by a sum
over stable graphs, with each vertex reduced to Witten–Kontsevich psi
integrals.

## The layer cake

```
GwTarget            geometry: basis, classical ring, pairing,
                    fixed points + tangent weights, c1 data
     │
CalibrationRecipe   QuantumRing (QDE + flatness)   |   IFunction (mirror map
                    + Birkhoff + metric adjoint)   |   Direct (hand-built)
     │
CohFT contract      SemisimpleCalibration (metric, Psi, Delta, R),
                    descendant S-matrix, insertion dictionary,
                    optional dimension oracle
     │
Engine              stable-graph sum: TQFT x R-action x translation,
                    generic over the coefficient ring
     │
Coefficient tiers   Rational | FactoredRatFun | RatFun
```

Everything below the contract line is target-agnostic; everything above it
is one file per space.  The engine never inspects geometry.

## Module map

| Path | Contents |
|---|---|
| `src/algebra.rs` | `Rational` (BigRational wrapper), interned `Monomial`, `SparsePoly`, `RatFun`, the `Coeff` trait, lambda-line limits |
| `src/factored.rs` | `FactoredRatFun`: rational functions with denominator factor lists |
| `src/series.rs` | `QSeries<C>` (truncated Novikov series), `SeriesMatrix<C>`, plain-series utilities (exp, compose, mirror-map inversion) |
| `src/graphs.rs` | stable-graph generation, individualization–refinement canonicalization, automorphisms, disk cache |
| `src/tautological.rs` | Witten–Kontsevich psi integrals (string/dilaton/DVV), shared process-wide cache |
| `src/geometry.rs` | equivariant cohomology of `P^n`: classes, fixed-point restriction, Atiyah–Bott pairing |
| `src/frobenius.rs` | `P^n` quantum Frobenius data: root series, quantum idempotents (historical; the generic path lives in `recipe`) |
| `src/givental.rs` + submodules | the engine and everything target-facing (below) |
| `src/twisted.rs` + submodules | negative split-bundle twisted theories; also hosts the H-Laurent / mirror / Birkhoff machinery (see Recipes) |
| `src/formula/` | human-readable stable-graph formula rendering (text/TeX) |
| `src/resolvent.rs` | labelled resolvent generating functions |
| `src/validation.rs`, `src/testsuite.rs`, `src/validation_backends/` | seed formulas, oracle tables, legacy localization, the built-in `gw-pn tests` suite |
| `src/bin/gw-pn.rs` | the CLI |

Inside `src/givental/`:

| Path | Contents |
|---|---|
| `graph.rs` | the contraction engine: graph kernels, accumulators, external-leg tensors, parallel drivers, coefficient-tier dispatch, public compute/series/resolvent entry points |
| `provider.rs` | the provider traits and the production `ProjectiveSpaceProvider` (symbolic and lambda-line), calibration caches |
| `matrices.rs` | `SeriesRMatrix`, `SeriesSMatrix`, convention metadata, the symplectic-unitarity check |
| `r_solve.rs` | the R-matrix flatness recursion |
| `classical_limit.rs` | Bernoulli/Gamma diagonal asymptotics from tangent-weight differences |
| `recipe.rs` | target-agnostic calibration recipes (see below) |
| `target.rs` | `GwTarget`, `TargetProvider`, `ProjectiveTarget` |
| `product.rs` | `P^n x P^m` by Novikov ray reconstruction |

## The engine

**Stable graphs** (`graphs.rs`).  `stable_graphs(genus, legs)` enumerates the
boundary strata combinatorics once per `(g, n)`: connected edge multisets by
a pruned recursion (incremental disjoint-set bound), stability enforced by
per-vertex genus minimums, and a lazy skeleton-orbit quotient whose
canonicalization is individualization–refinement (`canonical_data`) — the
same search yields the canonical key and the complete vertex-automorphism
list.  Tables are cached in memory and, when generation is expensive, on
disk (`~/.cache/gw-pn`, versioned format, structural audit on load).

**Graph kernels** (`graph.rs`).  `GiventalGraphKernel<C>` is everything
graph-local derived from a calibration: `R^{-1}` (`inverse_r_coefficients`),
the translation `T(psi) = psi(1 - R^{-1})1` (`translation_coefficients`),
and the symplectic edge propagator (`edge_propagator_coefficients`).  Built
by `GiventalGraphKernel::from_calibration`, which is generic over `Coeff` —
this genericity is what lets the equivariant path construct its kernel
natively in factored arithmetic.

**Contraction.**  `compute_semisimple_graph_value/_series` walk every
prepared graph and coloring, choosing leg/edge factor options recursively
(`accumulate_graph_factors`) and closing each vertex with point-theory psi
integrals plus translation insertions
(`vertex_contribution_with_translations`).  Series/master modes precontract
graph sums into external-leg tensors (`ExternalLegKernel`,
`RestrictedExternalLegKernel`) shared across many coefficients.  Work is
chunked over scoped threads (`GW_THREADS`), with per-worker vertex caches
merged back into the kernel's shared cache.

**Coefficient tiers.**  The evaluator is generic over `Coeff`; dispatch
picks the cheapest faithful representation per call: constants run over
plain `Rational` (`evaluate_rational_graphs_if_possible`, also used for the
external-leg tensors), genuine symbolics run over `FactoredRatFun`
(`evaluate_factored_graphs`), and expanded `RatFun` remains the fallback.
Escape hatches: `GWAI_DISABLE_RATIONAL_GRAPH`, `GWAI_DISABLE_FACTORED_GRAPH`.

**Vertex theory.**  `WittenKontsevich::shared()` is the process-wide psi
oracle (string, dilaton, DVV recursion over big rationals).

## The CohFT contract

The engine consumes, via the provider traits in `provider.rs`:

- `SemisimpleCalibration<C>`: flat metric, `Psi`/`Psi^{-1}` (flat <->
  canonical frame), Dubrovin connection, Delta series, and the
  `SeriesRMatrix`.  Frame normalization is recorded in
  `CanonicalFrameConvention` (the recipes produce
  `RelativeNormalizedCanonicalIdempotents`).
- `SeriesSMatrix`: the descendant->ancestor calibration.  Conceptually this
  is *not* part of the abstract CohFT — it is the calibration that turns the
  ancestor theory the R-matrix reconstructs into the descendant invariants
  users ask for.  Convention: the engine consumes the metric adjoint
  (covector action); see lessons.md §2.
- An insertion dictionary (`insertion_vector`) mapping user-facing classes
  to flat-basis vectors, and optional dimension bookkeeping used only for
  pruning and zero-prediction.

Two parallel trait spellings exist: `SemisimpleCohftProvider` (the public
`RatFun` API) and `CoefficientSemisimpleCohftProvider<C>` (the generic
boundary, with a blanket impl for the former).  `CalibrationId` strings are
metadata that keep tests and error messages honest about which convention
produced an object.

## Recipes

Recipes manufacture the contract from more primitive data
(`givental/recipe.rs`):

**Quantum-ring route.**  `newton_root_series` (roots of the characteristic
polynomial, seeded at classical eigenvalues), `divisor_lagrange_frame`
(idempotents by Lagrange interpolation — valid for divisor-generated rings;
its `1/P'(u)` norms assume the residue pairing, see lessons.md §3),
`calibration_from_canonical_frame` (relative sqrt normalization, connection,
flatness recursion with `classical_r_asymptotics_for_point` constants from
tangent weights), and `descendant_s_from_divisor_qde` (S by integrating the
quantum differential equation).

**I-function route.**  `descendant_s_from_i_function`: mirror map read off
the `H z^{-1}` part, exponential gauge and re-expansion to J reduced by the
classical ring relation, fundamental solution by repeated
`z q d/dq + H`-cup, Birkhoff factorization, then the metric adjoint.  The
cohomology-valued Laurent machinery it composes (`HCoeffLaurentSeries`, the
mirror transforms, `birkhoff_factor.rs`) currently lives in the `twisted`
module for historical reasons; the entry point is the target-agnostic seam
and the physical relocation is mechanical follow-up.

**Direct route.**  `GiventalGraphKernel::from_parts` accepts hand-supplied
`R^{-1}` and translation data for experiments that bypass both recipes.

The two computed routes cross-validate on any theory that admits both
descriptions; the test
`i_function_and_qde_recipes_agree_on_projective_space` does exactly this via
a rank-zero twist.

## Targets

**`GwTarget` / `TargetProvider`** (`target.rs`).  A target supplies:
`dimension`, `c1_degree` (Fano-index datum for virtual dimensions),
`classical_eigenvalue_seeds` (divisor restrictions at fixed points, pairwise
distinct), `divisor_multiplication` (quantum and classical, in the
divisor-power flat basis — companion form is verified at runtime), and
`insertion_vector`.  `TargetProvider<T>` derives everything else through the
quantum-ring recipe.  Documented scope: one Novikov variable,
divisor-generated rings.  `ProjectiveTarget` is the reference — one struct
whose seeds are either symbolic `lambda_i` or rational weights, covering the
equivariant and specialized theories with the same code.  It is held equal
to the production `ProjectiveSpaceProvider` by tests.

**Products** (`product.rs`).  `P^n x P^m` runs on the single-variable engine
through exact Novikov ray specialization `(q1, q2) = (t, b t)`;
`reconstruct_bidegree_invariants` runs `total_degree + 1` rays and solves
the Vandermonde system over the rationals, then filters
dimension-mismatched bidegrees (nonzero equivariantly, zero
non-equivariantly).  The frame is built in the constant *classical*
`D`-power basis — quantum powers of `D` are t-dependent and must not be used
as a basis (lessons.md §1) — with quantum idempotents recovered from
factor-idempotent fixed-point restrictions and metric norms from the
Atiyah–Bott flat metric.  Validated against Behrend's product formula
(`R_{P1 x P1} = R_{P1} (x) R_{P1}` entrywise).

**Twisted theories** (`twisted.rs`).  Negative split-bundle twists over
`P^n`, historically the crate's second theory and the origin of the
I-function pipeline.  They are morally a second family of targets (state
space of the base with an inverse-Euler-modified pairing, calibrated by
quantum Lefschetz I-functions) and will migrate onto the target/recipe
interface; today they implement the provider traits directly, with their
own factored-coefficient wrappers for symbolic fiber weights.

## The algebra stack

- `Rational`: exact big rationals.
- `Monomial`/`SparsePoly`: multivariate polynomials with globally interned
  variable ids (multiplication is an allocation-light merge; display resolves
  and sorts names so output never depends on interning order).
- `RatFun`: expanded numerator/denominator pairs with light normalization
  and *no* GCD cancellation — the default for small exact work, the wrong
  representation for repeated products of linear factors.
- `FactoredRatFun`: terms keyed by denominator factor lists; the symbolic
  contraction default.
- `QSeries<C>`/`SeriesMatrix<C>`: truncated Novikov series and matrices over
  any `Coeff`.

The `Coeff` trait is the load-bearing abstraction: one contraction engine,
one kernel builder, one recipe layer, instantiated at whichever coefficient
ring a computation deserves.

## Caching

- In-memory statics keyed by problem shape: stable graphs and prepared
  colorings, projective/target/product calibrations, S-matrices, graph
  kernels (including the factored twins), the shared Witten–Kontsevich
  table.
- On disk: stable-graph tables only (pure combinatorics), versioned and
  audited (`GWAI_GRAPH_CACHE_DIR`, `GWAI_DISABLE_GRAPH_CACHE`).

## Validation infrastructure

Layered, from innermost out:

1. `debug_assert`s inside constructions (duplicate-isomorphism-class check
   in the graph quotient; automorphism-orientation check in
   canonicalization).
2. Brute-force reference implementations kept as tests (full `V!`
   automorphism sweeps, generate-then-filter enumeration) plus frozen
   reference counts from the pre-optimization generator.
3. Cross-path A/B tests: rational tier vs dense evaluator, factored tier vs
   symbolic evaluator, `TargetProvider` vs `ProjectiveSpaceProvider`,
   I-function recipe vs QDE recipe, product calibration vs tensor of factor
   calibrations.
4. Closed forms and classical numbers at extremes
   (`<tau_{2g}(H)>_{g,1} = 1/(2^{2g}(2g+1)!)`, the `(1,1)`-curve count).
5. The built-in oracle suite (`gw-pn tests`): Zinger cross-checks, Growi
   rows, local Calabi–Yau tables, legacy localization.

Every optimization or refactor lands gated on 1–5 plus `cargo fmt --check`
and clippy (CI lives at the repository root, one level above this package).

## How to extend

**A new rank-one target** (quadric, known-semisimple Fano): implement
`GwTarget` in one file under `givental/` — seeds, companion divisor
multiplication, insertion vectors — wrap it in `TargetProvider`, and add an
invariant-level test against whatever independent numbers exist.  If the
target also has a hypergeometric I-function, feed it through
`descendant_s_from_i_function` and assert the two S-matrices agree: that is
the cheapest strong validation available.

**A new coefficient ring**: implement `Coeff` (plus the structural fast
paths `is_structurally_zero/one` and the complexity hooks), and the entire
engine, kernel builder, and recipe layer are available at that ring.

**A new theory family** (like twisted): implement the provider traits
directly against the CohFT contract; promote shared constructions into
`recipe.rs` as they prove target-agnostic.
