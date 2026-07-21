# Architecture

This document maps the crate's layers and points into the code, so a reader
can find where a concept lives without spelunking.  It complements
[lessons.md](lessons.md), which records the non-obvious findings behind
several of these design choices, and [virasoro.md](virasoro.md), which fixes
the mathematical convention and audit semantics for Virasoro constraints.

The crate computes Gromov–Witten invariants in exact rational arithmetic by
Givental–Teleman reconstruction: a semisimple CohFT is reassembled from its
genus-zero data (an R-matrix acting on a topological field theory) by a sum
over stable graphs, with each vertex reduced to Witten–Kontsevich psi
integrals.  Universal identities, currently the Getzler Virasoro operators,
form a separate audit path over the same canonical theory data.

## The layer cake

```
GwTheory             sole canonical geometry: state space + unit, pairing,
                     grading, c1, curve lattice/cone/splits, characteristic data
    │                                      │
    ├─> Virasoro generator                 └─> CanonicalCorrelatorEvaluator
    │       │                                      │
    │       └─> constraint AST + renderer          │ maps canonical keys
    │                                              v
    │                               Givental providers and Novikov-ray adapters
    │                                              │
    │                               CalibrationRecipe: QuantumRing | IFunction
    │                                              │                | Direct
    │                                              v
    │                               CohFT contract: metric, Psi, Delta, R, S
    │                                              │
    │                               stable-graph contraction engine
    │                                              │
    └────────────────────────────────── exact coefficient tiers and reports
                                    Rational | FactoredRatFun | RatFun
```

`GwTheory` is the only authority for universal target geometry.  Providers,
calibration inputs, and ray objects are algorithms for evaluating canonical
correlator keys; they are not alternative theory descriptions.  Everything
below the CohFT contract is target-agnostic, and the graph engine never
inspects `GwTheory`.

For navigation, read the repository on two axes.  `src/spaces/` is the
mathematical discovery hierarchy: its peer modules answer "what is available
for this target?"  `src/reconstruction/`, `src/givental/`, and
`src/constraints/` are algorithm layers: they answer "which shared operation
does this adapter use?"  Geometry is not copied at their intersection; every
adapter holds or returns the same canonical `GwTheory`.

## Module map

| Path | Contents |
|---|---|
| `src/core/algebra.rs` | `Rational` (BigRational wrapper), interned `Monomial`, `SparsePoly`, `RatFun`, the `Coeff` trait, lambda-line limits |
| `src/factored.rs` | `FactoredRatFun`: rational functions with denominator factor lists |
| `src/core/series.rs` | `QSeries<C>` (truncated Novikov series), `SeriesMatrix<C>`, plain-series utilities (exp, compose, mirror-map inversion) |
| `src/core/theory.rs` | universal `GwTheory` contract and shared state-space, curve-class, pairing, grading, and characteristic-number types; no concrete target records |
| `src/core/{error,moduli,bounded_cache,fused}.rs` | public error and pointed-curve stability/recursion boundaries, plus crate-private target-neutral storage/arithmetic helpers |
| `src/graphs.rs` | stable-graph generation, individualization–refinement canonicalization, automorphisms, disk cache |
| `src/tautological.rs` | Witten–Kontsevich psi integrals (string/dilaton/DVV), shared process-wide cache |
| `src/spaces/` | target-oriented public and physical hierarchy; each peer owns its concrete theory, target-specific provider/API code, and concrete Virasoro evaluator without duplicating geometry |
| `src/spaces/projective_space/{api,batch,resolvent}.rs` | ordinary `P^n` request/results, sparse-potential and packed-resolvent orchestration, and labelled resolvent generating functions around generic graph kernels |
| `src/spaces/*/virasoro.rs` | concrete adapters from canonical correlator keys to each target's provider; the generic constraint engine reexports their historical names |
| `src/constraints/` | backend-independent identity ASTs; Getzler Virasoro generation, text/TeX rendering, exact evaluation reports, and bounded scans; `virasoro/compat.rs` alone restores historical concrete-evaluator paths |
| `src/algebra.rs`, `error.rs`, `series.rs`, `theory.rs`, `resolvent.rs` | compatibility facades for the historical public paths; new internals import `core::*` and concrete `spaces::*` paths directly |
| `src/geometry.rs`, `src/frobenius.rs` | compatibility reexports for the projective-space implementations under `src/spaces/projective_space/` |
| `src/givental.rs` + submodules | the target-neutral contraction engine and calibration machinery (below), plus compatibility reexports for the historical product/bundle paths |
| `src/twisted.rs` | compatibility reexport for `src/spaces/negative_split_projective/` |
| `src/reconstruction/` | crate-internal target-neutral reconstruction algebra shared by product, bundle, and negative-split adapters (details below) |
| `src/formula/` | human-readable stable-graph formula rendering (text/TeX), distinct from Virasoro constraint rendering |
| `src/lib.rs` | crate module tree and compatibility reexports of the historical crate-root projective-space API |
| `src/validation.rs`, `src/testsuite.rs`, `src/validation_backends/` | seed formulas, oracle tables, legacy localization, the built-in `gw-pn tests` suite |
| `src/bin/gw-pn.rs` | the CLI |

Inside `src/spaces/`, every top-level directory is a peer target family:

| Target directory | Target-owned pieces |
|---|---|
| `projective_space/` | `theory` (canonical geometry), `provider`/`target` (calibration adapters), `api` (requests/results), `batch` and `resolvent` (Pn-shaped orchestration), `virasoro` (canonical-key evaluator), plus `equivariant`, `frobenius`, and small target-owned seed formulas |
| `product_projective/` | `theory`, two-ray `provider`, and `virasoro` evaluator |
| `projective_bundle/` | `theory`, bidegree/ray `provider`, and `virasoro` evaluator |
| `negative_split_projective/` | canonical local `theory`, compact `completion`, twist/I-function/hypergeometric/mirror/calibration recipes, graph `provider`, and `virasoro` adapters for direct inverse-Euler QRR and compact-completion audits |

The repeated `theory`/`provider`/`virasoro` names are intentional navigation
anchors. A target-specific implementation belongs here even when it consumes
a generic algorithm from `givental`, `reconstruction`, or `constraints`.

Inside `src/givental/`:

| Path | Contents |
|---|---|
| `graph.rs` | the target-neutral contraction engine: graph kernels, accumulators, external-leg tensors, parallel drivers, coefficient-tier dispatch, and provider-generic scalar/series entry points |
| `provider.rs` | the coefficient-generic `SemisimpleCalibration` data and sole `SemisimpleCohftProvider` contract; concrete projective-provider names are reexported only by the historical `givental` facade |
| `matrices.rs` | `SeriesRMatrix`, `SeriesSMatrix`, convention metadata, the symplectic-unitarity check |
| `r_solve.rs` | the R-matrix flatness recursion |
| `classical_limit.rs` | Bernoulli/Gamma diagonal asymptotics from tangent-weight differences |
| `recipe.rs` | target-agnostic calibration recipes (see below) |
| `recipe/cone_point.rs` | Givental mirror normalization, fundamental-solution assembly, Birkhoff descendant-`S` extraction, and metric-adjoint conversion over reconstruction primitives |
| `target.rs` | insertion-generic rank-one quantum-ring input `SemisimpleTarget` and `TargetProvider`; historical fixed-projective-insertion `GwTarget` implementations adapt into that boundary, and concrete `ProjectiveTarget` is a compatibility reexport from `spaces/projective_space/target.rs` |

`givental::Truncation` is the generic caller cap for graph-calibration
`z`-order.  The crate root and projective API reexport the same type for
source compatibility; it is not target geometry.

Inside `src/reconstruction/`:

| Path | Contents |
|---|---|
| `birkhoff.rs` | coefficient-generic graded Laurent Birkhoff factorization |
| `h_laurent.rs` | coefficient-generic cohomology-valued Laurent series in a cyclic power basis |
| `series_matrix.rs` | target-neutral operations on `QSeries<C>` matrices |
| `cyclic.rs` | typed cyclic coordinates and multiplication in a quantum algebra generated by one operator |
| `interpolation.rs`, `linear.rs` | checked exact ray plans and rational Vandermonde recovery |
| `truncation.rs` | grading-generic closure of positive Laurent windows and required negative depths |

The cyclic `H`-power Laurent representation and its pure series operations are
shared by negative-split and projective-bundle reconstruction, so they live in
`reconstruction/`.  Calibration-specific cone-point normalization,
fundamental-solution assembly, and descendant-`S` extraction live under
`givental/recipe/`; this keeps the reconstruction layer independent of
Givental types.  Negative-split code still owns its hypergeometric twist
factors, while bundle code owns its bidegree indexing and toric coefficient
construction.  The public Laurent type names remain reexported from
`spaces/negative_split_projective/` for compatibility.  The historical
`givental/product.rs` and `givental/bundle.rs` modules are also compatibility
reexports.

## The canonical theory boundary

`GwTheory` (`core/theory.rs`) defines every datum on which a target-independent
identity may depend:

- the homogeneous basis, unit, complex degree, and parity;
- the Poincare or explicitly twisted pairing and its inverse;
- the classical cup product in the canonical basis, multiplication by `c_1`,
  and a theory-chosen divisor for unstable divisor-equation stabilization;
- the numerical curve lattice, `c_1 . beta`, virtual dimension, effectivity,
  admissible support cone, and ordered decompositions of a curve class; and
- characteristic numbers, including the genus-one Virasoro anomaly.

The shipped compact implementations are `ProjectiveSpaceTheory`,
`ProductProjectiveTheory`, and `ProjectiveBundleTheory`.  Their canonical
curve classes are respectively a degree, a geometric bidegree, and
`(H.beta, xi.beta)`; the bundle's second coordinate can be negative.  The
theory-owned shifted bundle cone is a conservative admissible
reconstruction grading, not a replacement for that geometric class.

`NegativeSplitTotalSpaceTheory` records the numerical local target but
deliberately omits an ordinary compact pairing, `c_1` action, and
characteristic numbers.  This is executable documentation: the compact
Virasoro generator refuses it rather than manufacturing compact data for a
noncompact total space.  The separate provider-owned QRR path derives the
inverse-Euler twisted pairing and operator from this same canonical record;
it currently specializes the closed `L_0` conjugation and does not add fake
compact fields to `GwTheory`.
Its compact projective-bundle audit geometry is a separate adapter in
`spaces/negative_split_projective/completion.rs`, rather than part of the
canonical local-theory record.

`givental::target::SemisimpleTarget` is the insertion-generic rank-one
calibration-recipe adapter.  Every implementation returns its canonical
`GwTheory`; it can supply quantum multiplication, fixed-point seeds, and
insertion vectors, but it cannot restate dimension, first-Chern degree, or
effectivity.  `TargetProvider<T>` derives that bookkeeping from the returned
theory and validates the calibration rank against its canonical state space.
The historical `GwTarget` contract fixes the projective-space insertion type;
a one-way blanket implementation adapts it into `SemisimpleTarget`, preserving
existing implementations without creating another behavior source.  Neither
trait is a synonym for `GwTheory`, and no universal identity queries one.
The projective, product-ray, bundle-ray, and Virasoro evaluator adapters each
store that canonical object privately and expose it by shared reference;
calibration weights and Novikov rays cannot replace or mutate its geometry.
In particular, product and bundle divisor recursion calls the same
`GwTheory::classical_product` and `GwTheory::stabilizing_divisor` methods as
any future universal recursion; an evaluator does not carry a second ring
relation.

Human-readable Virasoro notation follows the same boundary. `GwTheory`
supplies both plain and TeX basis/curve-coordinate labels, and the checked
`render_*_for_theory` entry points compare the theory fingerprint before
rendering. The generic renderer never guesses target symbols by rewriting
strings such as `H1` or `xi`.

## The Virasoro audit path

`constraints::virasoro::generate_constraint_with_term_limit` reads only the
canonical theory and builds one exact coefficient AST.  The per-equation term
limit is checked against an upper bound on the unaggregated expansion before
labelled marking partitions, admissible degree splits, or state-space matrix
powers are materialized.  A term-budget failure rejects generation; it does
not return a truncated equation.  The scan's separate equation limit is
checked against the full Cartesian-product count before the theory-owned
bounded curve cone and descendant profiles are allocated.
Standard compact operator generation also caps `k` at `64`, so bracket and
matrix work cannot grow without bound when coefficient aggregation emits few
terms.  A separate 64-marking cap bounds the descendant payload cloned into
one coefficient; bounded scans use the stricter cap of 20 markings because
they enumerate labelled partitions across every profile.  An aggregate
total-term limit separately bounds the AST
terms retained across all equations in one scan; dependency limits bound
report/cache growth only indirectly.

Evaluation first canonicalizes the unique correlator dependency closure.
`CorrelatorEvaluationBounds` then limits genus, markings, maximum individual
descendant power, and unique dependency count.  The CLI exposes the latter
three scan controls as `--dependency-markings-max`,
`--dependency-descendant-max`, and `--dependency-limit`; scan genus supplies
the genus bound.  Dependency order is deterministic.  Retained keys excluded
by a property bound are missing with reason `OutsideBounds`.  When the unique
closure exceeds the count limit, only the canonical smallest omitted key is
retained as a witness and the report is explicitly marked truncated.  Bounds
are applied before structural-zero or backend resolution, so an excluded
dependency cannot disappear as an assumed zero.

The scan reports outcomes and coverage separately.  The four disjoint
coverage categories are:

- **backend-exercised:** at least one canonical dependency reached the
  computation backend, whether or not other dependencies were unresolved;
- **structural-only:** a non-vacuous equation closed from constants and/or
  canonical-theory-certified structural zeros, with no backend request;
- **vacuous:** exact aggregation left no terms; and
- **unresolved-only:** a non-vacuous equation was incomplete and had no
  backend value.

A successful scan means that every equation is `VerifiedZero`; it does not
say how those equations were covered.  A green scan dominated by vacuous or
structural-only equations is weak evidence about the Givental backend.
Backend validation requires meaningful backend-exercised coverage in the
intended genus, curve, insertion, and descendant sectors.  See
[virasoro.md](virasoro.md) for the formulas and exact report semantics.

## The engine

**Stable graphs** (`graphs.rs`).  `stable_graphs(genus, legs)` enumerates the
boundary strata combinatorics once per `(g, n)`: connected edge multisets by
a pruned recursion (incremental disjoint-set bound), stability enforced by
per-vertex genus minimums, and a lazy skeleton-orbit quotient whose
canonicalization is individualization–refinement (`canonical_data`) — the
same search yields the canonical key and the complete vertex-automorphism
list.  Tables are cached in memory and, when generation is expensive, on
disk (`~/.cache/gw-pn`, versioned format, structural audit on load).
Untrusted callers use `try_stable_graphs`; the built-in enumerator has the
shared finite envelope `2g-2+n <= 8` and `n <= 8`.  Formula validation and
every fallible Givental entry point check that envelope before cache lookup,
calibration, bundle warmup, or ray-worker creation.
The topology table is target-independent.  Prepared colorings are not: their
keys include the state-space rank and their raw size grows as
`colors^vertices`.  Before materialization the evaluator applies a conservative
64 MiB aggregate storage estimate that charges for vector payloads and the
transient orbit-building copies.  Prepared coloring/orbit tables then live in
deterministic eight-entry bounded caches.

**Graph kernels** (`graph.rs`).  `GiventalGraphKernel<C>` is everything
graph-local derived from a calibration: `R^{-1}`, the translation
`T(psi) = psi(1 - R^{-1})1` (`translation_coefficients`), and the symplectic
edge propagator (`edge_propagator_coefficients`).  Kernel construction uses
the `SeriesRMatrix` contract `R(-z)^T eta R(z)=eta`, hence
`(R^{-1})_k=(-1)^k eta^{-1}R_k^T eta`; the diagonal canonical metric makes
this an entrywise row/column scaling instead of a dense formal-inverse
recurrence.  The recurrence remains as an independent test oracle, and
calibration builders can audit the contract with `check_unitarity`.  Built by
`GiventalGraphKernel::from_calibration`, the path is generic over `Coeff` — this
genericity is what lets the equivariant path construct its kernel natively in
factored arithmetic.

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
The factored twin is initialized once per cached symbolic kernel, so a family
of correlators reuses both the conversion and the factored vertex cache.
Escape hatches: `GWAI_DISABLE_RATIONAL_GRAPH`, `GWAI_DISABLE_FACTORED_GRAPH`.

**Vertex theory.**  `WittenKontsevich::shared()` is the process-wide psi
oracle (string, dilaton, DVV recursion over big rationals).

## The CohFT contract

The graph engine consumes, via the evaluator/provider traits in `provider.rs`:

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
- An insertion dictionary (`insertion_vector`) mapping backend-facing classes
  to flat-basis vectors.  At the audit boundary a
  `CanonicalCorrelatorEvaluator` maps canonical `BasisId` values to those
  backend classes; the dictionary does not redefine the state space.
- Optional evaluation-specific dimension bookkeeping used only for pruning.
  Canonical virtual dimension and curve splittings live on `GwTheory`.

`SemisimpleCohftProvider<C>` is the one coefficient-generic provider contract;
`C = RatFun` is only its default type parameter.  The deprecated
`CoefficientSemisimpleCohftProvider<C>` spelling remains only so callers of
the historical `coeff_*` methods keep compiling.  The graph engine calls
`SemisimpleCohftProvider<C>` directly.  A one-way blanket implementation
exposes every canonical provider through the deprecated view; there is no
reverse blanket implementation or second source of provider behavior.
`CalibrationId` strings are metadata that keep tests and error messages honest
about which convention produced an object.

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
classical ring relation, then `descendant_s_from_j_function` (fundamental
solution by repeated `z q d/dq + H`-cup, Birkhoff factorization, metric
adjoint). For multi-parameter projective bundles,
`spaces/projective_bundle/provider.rs` instead builds the raw bidegree
fundamental solution and Birkhoff-factors it before
ray restriction; this lets the positive factor carry the full cone
projection.  The projected bidegree cone point is then put in flat Novikov
coordinates by extracting the two divisor mirror-coordinate series, gauging
them away, and inverting the bidegree mirror map before any ray
specialization.  The cohomology-valued Laurent machinery behind these
operations is split at its actual abstraction boundary.  Target-neutral
series-matrix operations, the cyclic `H`-power Laurent representation, graded
Birkhoff factorization, cyclic-coordinate algebra, exact ray interpolation,
and Laurent-window planning live in `reconstruction/`.  Mirror normalization,
fundamental-solution generation, and descendant-`S` extraction live in
`givental/recipe/cone_point.rs`.  Negative-split code owns the hypergeometric
twist coefficient construction, while the bundle's bidegree container and
toric construction remain in
`spaces/projective_bundle/provider.rs`.  Neither target imports generic
reconstruction through the old root `twisted` compatibility path.

**Operator frame.**  `operator_lagrange_frame` builds a canonical frame from
an explicit quantum multiplication operator (spectral projectors applied to
the unit, eigenvalues by Newton on the Faddeev–LeVerrier characteristic
polynomial, norms from the flat metric).  Unlike `divisor_lagrange_frame` it
assumes neither companion form nor the residue pairing, so it serves targets
whose multiplication matrix is honest in a constant basis but not cyclic in
the naive sense — the projective-bundle path uses it.

**Direct route.**  `GiventalGraphKernel::from_parts` accepts hand-supplied
`R^{-1}` and translation data for experiments that bypass both recipes.

The two computed routes cross-validate on any theory that admits both
descriptions; the test
`i_function_and_qde_recipes_agree_on_projective_space` does exactly this via
a rank-zero twist.

## Canonical theories and evaluation adapters

**Projective space.** `ProjectiveSpaceTheory` supplies the canonical basis
`1,H,...,H^n`, pairing, `c_1` action, degree lattice, and Chern numbers.
`ProjectiveSpaceProvider`, `ProjectiveTarget`, and `TargetProvider<T>` are
alternative Givental evaluators/calibration paths for those same canonical
correlators.  `ProjectiveTarget` provides classical eigenvalue seeds,
quantum/classical divisor multiplication, and fixed-point data to the
quantum-ring recipe; tests hold it equal to the production provider.  Its
documented evaluator scope is one Novikov variable and a divisor-generated
ring.

**Products.** `ProductProjectiveTheory` owns the tensor-product state space
and geometric bidegree.  The evaluator in
`spaces/product_projective/provider.rs` runs each requested
canonical correlator on the single-variable engine
through exact Novikov ray specialization `(q1, q2) = (t, b t)`;
`reconstruct_bidegree_invariants` runs `total_degree + 1` rays and solves
the Vandermonde system over the rationals, then filters
dimension-mismatched bidegrees (nonzero equivariantly, zero
non-equivariantly).  The current dense interpolation and one-worker-per-ray
implementation rejects requests above `MAX_EXACT_RECONSTRUCTION_RAYS` before
allocation or thread spawning.  The frame is built in the constant *classical*
`D`-power basis — quantum powers of `D` are t-dependent and must not be used
as a basis (lessons.md §1) — with quantum idempotents recovered from
factor-idempotent fixed-point restrictions and metric norms from the
Atiyah–Bott flat metric.  Validated against Behrend's product formula
(`R_{P1 x P1} = R_{P1} (x) R_{P1}` entrywise).

**Projective bundles.** `ProjectiveBundleTheory` owns the canonical quotient-
ring basis, pairing, `c_1` action, Chern numbers, and geometric class
`(d1,d2)`.  Its full classical cup product, including reduction by the bundle
relation, is exposed through `GwTheory`; divisor stabilization therefore does
not need evaluator-local multiplication rules. The evaluator in
`spaces/projective_bundle/provider.rs` computes
`P(O(a_1) + ... + O(a_m))` over `P^n`
from its toric I-function.  This is Picard rank two and the first target with a
nontrivial mirror map and a curve class that can be *negative* against a
divisor (the exceptional section of a Hirzebruch surface).  The design that
makes it tractable: twists are normalized to `min a_l = 0`, the shifted
fiber degree `d2' = d2 + (max a) d1 >= 0` gives a nonnegative grading that
defines the theory-owned conservative admissible cone.  Classes outside it
are certified ineffective.  Classes inside have unknown effectivity: that
fact alone forces neither zero nor nonzero, and the backend is queried unless
another structural rule applies.  The whole ring is cyclic over the grading
divisor `D = xi + (A+1)H`, and the raw
fundamental solution is Birkhoff-factored over the finite bidegree Novikov
ring before any ray restriction.  Its projected first column is normalized
in the bidegree Novikov ring: the positive-degree unit coordinate is gauged
away, and the two divisor mirror coordinates are gauged away and inverted.
The result is accepted only if no positive-degree `z^-1` coordinate remains
in a higher cohomology direction.  Such a remainder describes a big-quantum
path for which `q d/dq` contains higher-primary derivatives, not just
insertion of the grading divisor, so the backend returns
the structured `UnsupportedFeature` variant, with the offending bidegree and
cohomology direction as its witness, until a generalized mirror
transformation is available.  Only a cone point that passes this check is
restricted to rays and regenerated into a one-variable fundamental solution.
Per ray, this fundamental solution gives quantum multiplication
`A_q = A_cl + t d/dt S_1`; its metric-adjoint gives the descendant insertion
operator, the frame comes from `operator_lagrange_frame`, and R from
flatness.  `reconstruct_bundle_invariants` runs `total_degree + 1` rays and
a rational Vandermonde solve, under the same shared explicit ray cap.
Validated against `P^1 x P^1` zero twists, the non-Fano `F_2` deformation
dictionary, and the normalized mixed-sign rank-three bundle
`P(O + O(3) + O(3)) -> P^2`.  The normalized `F_4` presentation
`P(O + O(4))` and tested non-nef rank-three presentations `[0,1,2]` and
`[0,4,5]` retain higher-primary `z^-1` coordinates and therefore fail closed;
their former isolated deformation coincidences are no longer validation
claims.  A separate validation-only toric fixed-tree backend covers
genus-zero primary bundle classes through shifted total degree two and checks
the `[0,3,3]` fiber/section/mixed values `1,-27,19` plus a degree-two fiber
conic, without using any reconstruction object.  See lessons.md §§15–17.

**Twisted/local evaluators.** `spaces/negative_split_projective/` evaluates
negative split-bundle twists over `P^n` and owns its target-specific I-function
pipeline, including factored-coefficient wrappers for symbolic fiber weights.
`twisted.rs` is only the old import path.  The provider is an evaluator adapter
for the canonical local record, not a second source of state-space or
curve-lattice semantics.  Ordinary invariant evaluation is available, but
the standard compact Virasoro generator remains invalid.  The peer
`virasoro/qrr.rs` specialization builds the fixed-fiber-equivariant
inverse-Euler twisted pairing and QRR-conjugated `L_0` from the canonical
negative-split degrees.  `NegativeSplitQrrEvaluator` uses the same provider
for stable degree-zero graphs and positive-degree correlators, including a
single full descendant-divisor stabilization policy for unstable pointed
ranges.  Its peer `NegativeSplitFixedFiberQrrEvaluator` specializes the same
theory at recorded nonzero rational `mu_i`, retaining a symbolic auxiliary
base lambda line so native-factored graph arithmetic is univariate until the final
limit.  The target-independent `SpecializedVirasoroConstraint` applies that
same named point to operator coefficients while preserving all equation
metadata.  General QRR Hamiltonian data live target-independently in
`constraints/virasoro/qrr.rs`; target specialization and evaluation stay with
the space provider.  The explicitly separate
`NegativeSplitCompletionEvaluator` is a compact-section audit adapter: its
compact projective-bundle theory remains the source of Virasoro data and
degree-zero invariants, while positive section classes `(d,-A d)` restrict by
`xi|_S=-A H` and are evaluated by the local provider.  It rejects positive
nonsection classes, invalid basis ids, and every local calibration except the
nonequivariant inverse-Euler mode.  It does not claim to implement generic
QRR-conjugated Virasoro.  See [virasoro.md](virasoro.md).

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

- Target-specific reconstruction caches (calibrations, descendant matrices,
  kernels, and bundle cone data) use the deterministic `BoundedCache` with a
  shared capacity of 64 entries.  Eviction is FIFO: hits do not make tests or
  memory retention order-dependent.  The bound is on entries, not bytes, so a
  single frontier-sized calibration can still be large.
- Universal stable-graph topology and the shared Witten–Kontsevich table
  remain process-wide under hard work envelopes; they are not byte-bounded.
  Prepared colorings and coloring orbits include target state-space rank in
  their keys, so they instead use an eight-entry `BoundedCache` plus the checked
  storage estimate described above.
- On disk: stable-graph tables only (pure combinatorics), versioned and
  audited (`GWAI_GRAPH_CACHE_DIR`, `GWAI_DISABLE_GRAPH_CACHE`).

## Error boundaries

`GwError` distinguishes two failures that must not be conflated in an audit:

- `ResourceLimit { operation, requested, limit }` says the mathematics is in
  scope but a declared work or retained-state guard rejected this request
  before the guarded operation or additional retention; and
- `UnsupportedFeature { target, feature, witness }` says the target was
  recognized but the witnessed case needs mathematics not implemented by the
  backend, such as higher-primary bundle mirror normalization.

The broader string-valued `UnsupportedInvariant` remains for legacy call
sites and support cases not yet migrated.  Display strings are for people;
tests and library callers should match the structured fields when those
variants apply.  Neither error category may be converted into a numerical
zero by a validation layer. Virasoro residual reports preserve the same
fields in structured `IncompleteReason` variants instead of relabeling them
as generic evaluation errors.

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
   calibrations, evaluator adapters vs their canonical `GwTheory`, disjoint
   ray-node choices, insertion permutations, and classical-ring laws.
4. Closed forms and classical numbers at extremes
   (`<tau_{2g}(H)>_{g,1} = 1/(2^{2g}(2g+1)!)`, the `(1,1)`-curve count).
5. The built-in oracle suite (`gw-pn tests`): Zinger cross-checks, Growi
   rows, local Calabi–Yau tables, and validation-only localization.  The old
   projective-space localization slice is genus-zero, primary, and degree one.
   The independent bundle fixed-tree module is genus-zero and primary through
   shifted total degree two; it includes a genuine degree-two cover, repeats
   dimension-matched rows at disjoint generic weights, and calls none of the
   I-function, Birkhoff, ray, `S`/`R`, or Givental reconstruction code.
6. Exact Virasoro coefficient audits: the generated human-readable equation,
   complete dependency closure, and an exact residual classified as
   `VerifiedZero`, `Nonzero`, or `Incomplete`.  A missing correlator can never
   turn into a passing zero.  Scan coverage additionally distinguishes
   backend-exercised, structural-only, vacuous, and unresolved-only equations,
   so a large collection of structural passes is not presented as backend
   evidence.

Every optimization or refactor lands gated on 1–6 plus `cargo fmt --check`
and clippy (CI lives at the repository root, one level above this package).

## How to extend

**A new compact theory** starts with one canonical `GwTheory`
implementation: basis and unit, grading/parity, pairing and inverse, classical
cup product, `c_1` action, a stabilizing-divisor policy, curve
lattice/effectivity/splittings, and characteristic numbers.  Give it a peer
discovery module under `spaces/`, then add one or more evaluator adapters.
A universal identity must depend on this canonical object, never on a
calibration provider or the discovery facade.

**A new rank-one Givental evaluator** (quadric, known-semisimple Fano): expose
it from the target's `spaces/` module and implement a `SemisimpleTarget`
calibration input--seeds, companion divisor multiplication, and a target-owned
insertion dictionary--using the shared machinery under `givental/`.  Wrap it
in `TargetProvider` and map canonical evaluator requests to the same
`GwTheory`.  Implement `GwTarget` only when preserving the historical
projective-insertion contract.  Add an invariant-level test against independent
numbers.  If the target also has a hypergeometric I-function, feed it through
`descendant_s_from_i_function` and assert the two S-matrices agree: that is
the cheapest strong validation available.

**A higher-Picard-rank target**: follow the `spaces/product_projective` and
`spaces/projective_bundle` entry points into their peer `provider.rs` files.  Build
the frame in a constant classical basis of a cyclic grading generator, keep
the Novikov multigrading through the quantum-ring or Birkhoff construction,
then restrict to flat-coordinate rays `(t, b t)` and reconstruct bidegrees from
`total + 1` rays by a rational Vandermonde solve.  This reuses the graph
engine.  Put the state space, numerical curve classes, effectivity, and
splittings--including any theory-owned conservative admissible grading--in
`GwTheory`; keep fixed-point weights, the I-function or quantum ring,
insertion conversion, ray choices, and interpolation in the evaluator
adapter.

**A new coefficient ring**: implement `Coeff` (plus the structural fast
paths `is_structurally_zero/one` and the complexity hooks), and the entire
engine, kernel builder, and recipe layer are available at that ring.

**A new noncompact or twisted theory** still begins with a canonical
`GwTheory`, but absent geometric data must remain absent rather than be
guessed.  Implement the evaluator against the CohFT contract and promote
shared constructions into `recipe.rs`.  Constraint support is a separate
step: use the target-independent `QrrConjugationFormula`, then let the provider
specialize the characteristic class, twisted pairing, degree-zero sector, and
exact operator conjugation.  Inverse-Euler `L_0` is the reference
implementation; arbitrary classes and higher modes must fail closed until all
four pieces exist.
