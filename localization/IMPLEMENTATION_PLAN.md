# `gw-pn` Implementation Plan

This is the working plan for turning the existing specification into an
end-to-end exact GW invariant calculator for `P^n`.

## Source Anchors

The implementation conventions follow the two Lee-Pandharipande notes linked in
the original request:

- Part 1: Frobenius manifolds, equivariant localization, materialization, and
  Givental's higher-genus formula.
  `https://people.math.ethz.ch/~rahul/Part1.pdf`
- Part 2: quantization, calibrations, descendant potentials, and Givental's
  descendant formula.
  `https://people.math.ethz.ch/~rahul/Part2.pdf`

The most important engineering consequence is that the code must keep
localization, Frobenius data, calibration, quantization, and graph expansion as
separate layers with explicit convention metadata.

## Critical Design Adjustments

1. The public interface should distinguish individual invariants from bounded
   potential coefficients.  For fixed homogeneous insertions, the virtual
   dimension determines at most one degree.  The series API should therefore not
   mean "same insertions, all degrees"; it should mean a sparse, bounded
   descendant-potential query over insertion monomials.
2. The first exact symbolic backend should be structured sparse algebra, not a
   generic expression tree.  Full polynomial gcd should be checkpointed rather
   than performed in inner graph loops.
3. The first production Givental engine should use the stable-graph/Feynman
   formula.  Direct operator expansion is useful for tests, but it creates large
   intermediate expressions.
4. Hodge integrals should be a backend boundary.  Pure psi integrals are built
   in; lambda/Hodge integrals should support tables and an offline
   `admcycles`/Sage generation path.
5. The localization and Givental engines should share graph, algebra,
   tautological, and projective-space modules so every convention can be tested
   twice.

## Implemented Now

- Arbitrary-precision rational arithmetic via `num-bigint`/`num-rational` and
  sparse rational functions.
- `P^n` cohomology classes in the hyperplane basis.
- Equivariant fixed-point restrictions, idempotents, Euler weights, and pairing.
- Cached Witten-Kontsevich psi integrals using string, dilaton, and DVV
  recursion.
- Bounded stable-graph data structures and a small exact generator for test
  cases.
- Bounded sparse potential coefficient API.  It enumerates monomials in
  `tau_k(H^r)` up to explicit marking/descendant bounds and degrees up to
  `d_max`; unsupported dimension-valid coefficients are reported rather than
  treated as zero.
- Equivariant classical and quantum `P^n` relation reduction using
  `prod(H-lambda_i)=q`.
- Truncated `q`-series algebra and matrix operations used by the Frobenius and
  Givental layers.
- Frobenius seed data: quantum multiplication by `H`, companion matrix checks,
  and formal `q`-series canonical root expansions by Newton iteration.
- Genus-zero localization fixed-locus graph enumeration with vertex colors,
  positive edge degrees, labelled markings, and cover automorphism factors.
- Genus-zero primary one-edge localization contribution evaluation for
  equivariant requests.  This is intentionally left as an equivariant rational
  function until the nonequivariant-limit backend is stronger.
- Validated genus-zero primary one-edge localization factor evaluation for
  degree-one equivariant requests.  A broader primary-tree attempt exposed
  convention gaps on the `P^2` conic count and is intentionally not used as a
  general evaluator yet.
- Kontsevich recursion shortcut for genus-zero `P^2` point invariants
  `<pt^(3d-1)>_{0,d}`, giving fast checks such as the conic count `1` and cubic
  count `12`.
- Exact lambda-line nonequivariant limit helper for rational functions along
  `lambda_i=c_i*t`.
- Dimension helper for fixed homogeneous insertions: the expected degree is
  derived from the virtual dimension constraint when possible.
- Classical canonical data at `q=0`: fixed-point idempotents, metric norms,
  inverse metric norms, and the unnormalized transition matrix to the
  hyperplane basis.
- Quantum canonical data as truncated `q`-series for the first validated target
  (`P^1` at low order): roots, unnormalized idempotents, metric norms,
  inverse norms `Delta_i=P'(x_i)`, and transition columns in the hyperplane
  basis.  Equivariant rational equality tests currently use exact numeric
  lambda specializations until polynomial gcd normalization is added.
- Canonical-frame metric checks: the unnormalized quantum transition matrix
  diagonalizes the flat metric in the validated `P^1` q-series case.
- Q-series `R`-matrix infrastructure with explicit calibration and
  canonical-frame convention metadata, identity calibration, and
  coefficient-wise `R(-z)^T eta R(z)=eta` unitarity validation.
- First nontrivial `P^n` J-calibrated `R`-matrix algorithm in a relative
  normalized idempotent frame: compute `Psi`, `Psi^{-1}`, `A=Psi^{-1}q dPsi/dq`,
  solve the projective-space recursion, and fix the diagonal gauge by the
  Bernoulli classical limit.  The validated regression currently covers `P^1`
  through `q^1 z^2`; higher bounds are available through the API but will need
  expression-control work before becoming routine tests.
- First Givental stable-graph evaluator wired into `ComputeMode::Givental` after
  seed formulas fail.  It uses the J-calibrated `R^{-1}` on legs, the
  symplectic edge propagator, the finite translation insertion
  `T(psi)=psi(1-R^{-1})1`, and the relative-frame diagonal TFT vertex factors.
  Current validation covers direct graph evaluation of `P^1`
  `<H,H,H>_{0,1}=1` and stationary descendant checks through degree 4.
- Small descendant `S`-matrix extraction for `P^n` from the quantum differential
  equation.  Ordinary descendant insertions are expanded into ancestor legs
  before the existing `R`-graph evaluation, with graph-dimension pruning of
  impossible ancestor psi powers.
- Practical non-equivariant `S/R` graph evaluation by early generic lambda-line
  specialization.  For nonequivariant requests, the canonical roots,
  idempotents, metric norms, `S` matrix, and `R` diagonal asymptotics are built
  directly over rational numbers after choosing generic weights, avoiding the
  symbolic equivariant rational-function blow-up in fixed-graph computations.
- Fixed-graph evaluator efficiency passes:
  - cached `R`/`S` calibrations keyed by target, truncation, and lambda-line
    weights;
  - cached stable-graph generation by `(g, markings)`;
  - reused vertex colorings and graph automorphism factors within a compute;
  - precomputed leg/edge expansion options by color;
  - memoized vertex translation sums by `(genus, color, psi powers)`;
  - pruned recursive graph branches once total ancestor psi power exceeds the
    graph dimension.
- Series mode now buckets homogeneous insertion monomials by the expected
  degree from the virtual dimension before calling the invariant API, avoiding
  the old all-degrees scan for each monomial.
- Genus-zero `P^1` stationary one-descendant divisor-family shortcut:
  `<tau_{2d-2}(H), H^m>_{0,d} = d^m/(d!)^2`, used as a fast public API path and
  cross-checked against direct `S/R` graph evaluation for initial degrees.
- Independent Zinger projective-space backend under `src/zinger/`, based on
  arXiv:1106.1633.  The implemented subset covers genus-zero degree-zero
  constants, the projective-space vanishing criterion, the Theorem 4 3-point
  generating-function extraction, and the explicitly stated 4-point primary
  degree-one/degree-two projective-space consequences.  It does not call the
  Givental `S/R` evaluator and is used for cross-check tests.
- Table-backed Hodge integral oracle boundary, with native psi integrals and
  externally supplied lambda/Hodge values.
- Public `compute` API with explicit `Localization`, `Givental`, and
  `CompareLocalizationAndGivental` modes.
- Seed computations:
  - `P^0` point theory;
  - genus-zero degree-zero constant maps;
  - genus-zero three-point primary small quantum products of `P^n`.
- Minimal CLI:
  - `gw-pn tests`
  - `gw-pn psi --g 2 --powers 4`
  - `gw-pn compute --n 2 --g 0 --d 1 --insert 'tau0(H^2)' --insert 'tau0(H^2)' --insert 'tau0(H)' --mode compare`
  - `gw-pn series --n 2 --g 0 --d-max 1 --max-markings 3 --mode compare`
- Built-in validation suite shared by `cargo test` and `gw-pn tests`.

The seed engines intentionally return `UnsupportedInvariant` for cases needing
full Hodge localization or a nontrivial `R`-matrix.

## Next Milestones

1. Fix and validate the full Kontsevich localization factor conventions for
   arbitrary genus-zero trees, using the `P^2` conic and cubic counts as
   regression tests.
2. Improve expression control for genuinely equivariant requests: checkpoint
   simplification, common-denominator batching, and graph-level memoization.
3. Extend Kontsevich localization factor evaluation from primary genus-zero
   trees to descendants and all remaining unstable-vertex cases.
3. Add Hodge table serialization and an offline `admcycles`/Sage import format.
4. Scale quantum canonical data from the validated `P^1` low-order case to
   `P^n`/higher `q` order with better rational-function normalization, then add
   normalized `Psi` after deciding how to represent square-root branches.
5. Harden the J-calibrated `R` extractor for equivariant output: add stronger
   simplification, higher `P^1`/`P^2` regression bounds, and direct comparison
   with J-function columns.
6. Harden the Givental graph expansion with graph-level memoization, stronger
   vertex/edge degree pruning, direct comparison against known low-degree
   invariants, and a CLI inspection/export command for graph contributions.
7. Extend the Zinger backend from the current 3-point/selected 4-point subset
   to the full 4-point descendant formula, then selected Theorem A N-point
   structure constants for projective spaces.
8. Make `SeriesRequest` reuse one graph/materialization pass per `(n,g,d_max)`
   instead of calling the individual-invariant API per coefficient.
9. Build comparison tests on `P^1` and `P^2` for `g <= 2`, `d <= 3` before
   optimizing for higher genus.
