# Lessons from building an exact Gromov–Witten engine

This crate computes Gromov–Witten invariants of projective spaces (and
friends) by Givental–Teleman reconstruction, in exact rational arithmetic.
Over a long stretch of optimization and generalization work — taking genus 4
from unreachable to interactive, equivariant computations from
non-terminating to seconds, and the engine from "ℙⁿ only" to a modular
target interface with ℙⁿ×ℙᵐ working — a number of genuinely non-obvious
things surfaced.  Most steps were standard; these were not.  Each entry below
is written symptom-first so a future developer (or AI agent) can
pattern-match against their own bug before re-deriving the explanation.

---

## Part I — Mathematical traps

### 1. Quantum powers of a divisor are not a basis

**Symptom.** A product target ℙ¹×ℙ¹ built through its graded divisor
`D = H₁+H₂` produced a Dubrovin connection with nonzero "mixed" entries
(idempotent pairs differing in *both* factor indices) and canonical metric
norms Δ that matched the tensor-product prediction at q⁰ but diverged at q¹.
The same construction pattern worked perfectly for ℙⁿ.

**Cause.** Presenting quantum multiplication by `D` as a companion matrix
implicitly chooses the basis `1, D, D•D, D•D•D, …` where `•` is the
*quantum* product.  Those are `t`-dependent elements of cohomology, not a
fixed basis.  The Dubrovin connection `Γ = Ψ⁻¹ q∂Ψ` and the descendant
quantum differential equation both differentiate the frame *against a
constant basis*; a moving basis injects spurious connection terms equal to
the basis's own derivative.  For ℙⁿ this is invisible because the quantum
relation only activates at the top power — `H^k` for `k ≤ n` is the same
classically and quantumly.  For a product, `H₁²` already contains `q₁` when
the first factor is ℙ¹, so the error enters at Novikov degree one.

**Fix.** Work in the constant *classical* power basis `1, D, D², …`
(classical cup products — genuinely fixed elements, a valid basis whenever
the classical eigenvalues of `D` are distinct).  Obtain the quantum
idempotents not from Lagrange interpolation in the quantum operator but from
their *classical fixed-point restrictions*: product idempotents tensor,
restrictions multiply, and any cohomology element is recovered from its
restrictions through the constant classical Lagrange transition.  Since the
restriction matrix is the identity at `q = 0`, its inverse is a cheap
truncated Neumann series.

**Portable rule.** Any time you write a matrix for a quantum-cohomology
operator, ask: *in which basis, and is that basis constant in the Novikov
variables?*  Companion form is only safe when quantum and classical powers
coincide below the top degree.

### 2. The metric-adjoint convention for S is invisible below z²

**Symptom.** Two independent constructions of the descendant S-matrix for
the *same* theory (one solving the quantum differential equation order by
order, one by mirror transformation and Birkhoff factorization of the
I-function) agreed identically at z⁰ and z¹ — every entry, every Novikov
order — then disagreed at z².

**Cause.** They compute metric adjoints of each other, `S* = η⁻¹ Sᵀ η`
(the vector action versus the covector action).  The symplectic condition
`S*(−z) S(z) = 1` *forces* `S₁* = S₁`, so the two conventions coincide
through z¹ for structural reasons and first diverge at z², where
`S₂* = S₁*S₁ − S₂`.

**Portable rule.** A validation of S-matrix conventions that only checks
z-order ≤ 1 is structurally incapable of catching an adjoint mix-up.  Always
test to z² or higher, and record explicitly which convention (vector or
covector action) your graph engine consumes.

### 3. The residue-pairing shortcut for metric norms is presentation-specific

For a cyclic presentation `Q[x]/(P(x))` the Lagrange denominators `P'(uᵢ)`
double as inverse metric norms **only when the Poincaré pairing is the
residue pairing of that presentation**.  True for ℙⁿ in the hyperplane
basis; false for a product presented through its graded divisor — there
`P'(dᵢ)` contains cross-differences of eigenvalue *sums*, which are not
tangent weights and not Euler classes.  Unless you have proven
residue = Poincaré for your specific presentation, compute norms honestly
from the flat metric: the diagonal of `Tᵀ G T` where `T` has the idempotents
as columns.

### 4. Classical R-matrix asymptotics come from tangent weights

The flatness ODE determines the R-matrix only up to constant diagonal
exponentials in z; the integration constants are the Bernoulli/Γ-function
series `exp(Σᵣ B₂ᵣ/(2r(2r−1)) Σ_w w^{−(2r−1)} z^{2r−1})`.  The sum runs over
the **tangent weights of the fixed point**, not over eigenvalue gaps of
whatever operator generates your ring.  For a product these are the
factor-wise weight differences — consistent with `R_{X×Y} = R_X ⊗ R_Y` —
while the D-eigenvalue gaps would be wrong.  Conveniently, this means the
constants are derivable from the same fixed-point weight data that defines
the equivariant frame, so a target interface never needs a separate
"asymptotics" input.

### 5. Novikov ray specialization is exact — with three sharp edges

Rank-two Novikov variables can be handled on a single-variable engine
without loss: `(q₁,q₂) = (t, b·t)` for rational `b` is a ring homomorphism,
a degree-k ray coefficient is `Σ_{d₁+d₂=k} b^{d₂} N_{(d₁,d₂)}`, and `k+1`
distinct rays plus a Vandermonde solve over ℚ recover every bidegree
*exactly*.  The sharp edges:

- **Cyclicity is generic, not automatic.** Along a ray the ring must be
  generated by the graded divisor, i.e. the fixed-point sums `λᵢ + μⱼ` must
  be pairwise distinct.  Choose default weights that guarantee it (e.g.
  `λᵢ = i+1`, `μⱼ = (n+2)(j+1)`).
- **Per-degree dimension pruning is impossible on a ray** when the factors
  have different dimensions: the virtual dimension varies across the
  bidegrees mixed into one total degree.  Prune after reconstruction.
- **Do not expect mismatched bidegrees to reconstruct to zero.**  See the
  next lesson.

### 6. Equivariant dimension mismatch has two different sides

For a proper target the equivariant pushforward lands in
`H*_T(pt) = ℚ[λ]`.  If the total insertion degree is below the virtual
dimension, the answer would have negative parameter degree and is therefore
zero.  Exact dimension gives a weight-independent scalar.  Excess insertion
degree can instead give a nonzero weight polynomial whose non-equivariant
limit vanishes.  Tests and candidate-degree pruning must distinguish these
deficit, exact, and excess cases rather than treating every mismatch alike.

Localized twisted theories require a separate policy: inverse Euler classes
live in a localized coefficient ring where negative parameter degrees can
occur, so the proper-target deficit argument cannot be applied blindly.

### 7. Free cross-validation oracles are everywhere — use them

The three that carried this project:

- **Rank-zero twist = untwisted target.**  A twisted-theory pipeline with an
  empty twist computes the plain theory, so the I-function/Birkhoff recipe
  and the quantum-ring/QDE recipe become two independent roads to the same
  S-matrix.  Held equal entrywise, this validates both.
- **Behrend's product formula.**  `R_{X×Y} = R_X ⊗ R_Y` in matching
  canonical frames (idempotents tensor, Δ multiply, relative normalizations
  are multiplicative).  An entrywise-exact test of a product implementation
  against two independently computed factor calibrations.
- **Closed forms at extremes.**  `⟨τ_{2g}(H)⟩_{g,d=1} = 1/(2^{2g}(2g+1)!)`
  on ℙ¹ pinned the entire genus-4 pipeline the first time it ever ran
  (1/92897280 at g = 4).  When pushing into a new regime, find one closed
  form that lives there.

### 8. Exact arithmetic overflows too

`(2d+1)!!` exceeds `i128` at `d ≥ 28` — reachable by one-point
Witten–Kontsevich integrals near genus 10 through the DVV recursion.
`usize` factorials overflow at `21!` — reachable by translation-insertion
multiplicities at high vertex dimension.  In release builds these wrap
*silently*, corrupting results that are advertised as exact.  Any
factorial-type coefficient in a recursion whose inputs scale with
genus/markings should be accumulated in big rationals from the start; the
cost is invisible because such values are cached.

---

## Part II — Performance truths

### 9. Profile buckets lie; untracked gaps are findings

Two expensive mislabelings:

- The genus-4 wall was not graph *evaluation* but graph *generation* — 100
  minutes to enumerate 2,666 stable graphs (string canonical labels
  minimized over all V! vertex permutations, inside a generate-then-filter
  loop).  The tell was an *untracked gap* in the profiler output: total
  0.75 s at genus 3 with only 0.2 s attributed to named stages.  The gap was
  the finding.
- The equivariant "calibration = 27 s" bucket actually contained kernel
  construction (R⁻¹ recursion and edge propagators, i.e. many products of
  calibration entries); the calibration itself took 120 ms.  One targeted
  probe (`time each stage separately in a scratch binary`) redirected the
  entire fix.

Rule: before optimizing a stage, verify by direct measurement what the stage
*contains*, and treat unattributed wall-clock as a first-class lead.

### 10. For symbolic blowup, representation beats micro-optimization

String interning, binary exponentiation, and reference-based big-rational
ops — all worthwhile hygiene — moved an equivariant genus-2 computation from
"never finishes" to 249 s.  Changing the *representation* did the rest:
keeping denominators as factor lists (never expanding products of linear
factors mid-computation) took the graph contraction from 221 s to 10 s, and
building the kernel natively in that representation — converting the small
calibration once instead of the big kernel afterwards — took the total to
1.6 s.  The general pattern:

> Convert small objects early; compute in the representation that respects
> your expression structure; expand exactly once at the end.

A coefficient-generic engine (a `Coeff` trait with plain-rational, expanded
symbolic, and factored implementations) makes this a per-call dispatch
rather than an architecture change, and cheap escape-hatch environment flags
make every tier A/B-testable against the naive path.

### 11. Isomorphism-class enumeration: what actually worked

For enumerating stable graphs up to isomorphism (vertices carry genera, legs
are labelled, multi-edges and loops allowed):

- Weisfeiler–Leman refinement plus a sweep over class-respecting
  relabelings beats brute V!, but stalls exactly on regular structures where
  all vertices look alike.
- **Individualization–refinement** fixes that and yields a bonus: the
  automorphism group acts freely and transitively on the minimal-key leaves
  of the search tree, so pairing one fixed minimal leaf with every minimal
  leaf enumerates the *complete* automorphism group from the same search —
  no group-closure step.
- Every branching and ordering choice inside a canonicalization must depend
  only on isomorphism-invariant data.  The subtlest bug of the project was
  comparing an identity-arrangement key against block-arranged keys with `<`
  (absence of anything smaller) instead of demanding *equality with the
  minimum*: the identity arrangement is generally not among the enumerated
  arrangements, so "nothing smaller" holds for several labelings of the same
  class at once, and the quotient overproduces.
- Therefore: **build a duplicate-class assertion into the quotient itself**
  (`debug_assert` that output canonical keys are pairwise distinct).  It
  caught two real bugs before any downstream test could.
- Two-stage quotients (canonicalize the edge skeleton once, then keep only
  orbit-minimal decorations under its automorphisms) amortize beautifully
  when decorations are dense and *lose* when they are sparse — and any
  routing heuristic between strategies must itself be isomorphism-invariant,
  or isomorphic objects take different paths and the quotient breaks.
  Compute such quotients lazily: skeletons admitting no valid decoration
  should never pay for canonicalization.

### 12. Cache pure combinatorics on disk, with a structural audit

Stable-graph tables depend only on `(genus, markings)` — never on the
target, degree, or insertions — and are expensive exactly once.  A versioned
plain-text cache with atomic writes turned a ~50 s generation into ~2 ms
loads for every later process.  The important detail: **audit on load**
(re-check each graph's marking count, total genus, connectivity, stability)
so a corrupt or stale file regenerates instead of silently poisoning every
computation built on it.  Version the format in the filename; bump it when
the generator or the canonical representatives change.

---

## Part III — Architecture

### 13. Name the engine's true contract, and interfaces fall out

The reconstruction engine is exactly a **semisimple CohFT evaluator**: it
consumes (state space with metric, semisimple frame data Δ/Ψ, R-matrix) plus
a *descendant calibration* S — which is not part of the abstract CohFT, but
the descendant↔ancestor dictionary — plus an insertion dictionary and an
optional dimension oracle used only for pruning.  Once stated this way,
"support a new space" decomposes cleanly:

- a **target** supplies geometry (basis, classical ring, pairing, fixed
  points with tangent weights, c₁ data);
- a **recipe** manufactures the contract from whichever datum the target
  naturally has: the quantum ring (S from the QDE, R from flatness +
  weight asymptotics) or an I-function (mirror map, Birkhoff factorization,
  metric adjoint) — or a direct hand-built calibration for experiments;
- the engine never changes.

Targets with both data (ℙⁿ; anything with a rank-zero-twist description)
give the cross-recipe oracle of Lesson 7 for free.

### 14. Miscellaneous scars, briefly

- **`E − V + 1` is not the first Betti number** of a possibly-disconnected
  graph; `E − V + C` is, and the difference underflows `usize` on forests.
- **Presence is not truth for env flags**: `FLAG=0` enabling a feature
  because the code checked `var_os().is_some()` — parse values, share one
  helper.
- **Stale build artifacts lie**: a debug harness linking `libX-*.rlib`
  picked by `find | head -1` bound the *oldest* rlib and produced hours of
  self-consistent nonsense.  Pick artifacts by mtime (`ls -t`), or better,
  through cargo itself.
- **`git add -A` in an automated loop** will eventually commit session junk;
  add tool/session directories to `.gitignore` before the first push, and
  gate every push on the same checks CI runs (the first CI failure of the
  project was a missing local `cargo fmt --check`).
- **Delete dead parameters ruthlessly**: a public struct carried four
  fields of which one was read; callers were configuring behavior that did
  not exist.  An API that accepts and ignores input is worse than a smaller
  API.

---

## Part IV — Multi-parameter targets

### 15. Grade first, restrict to rays second — Birkhoff before specialization

Higher Picard rank means multiple Novikov variables, and the naive fear is
multi-variable Birkhoff factorization.  The right simplification is not to
avoid it by restricting to rays too early; it is to use the grading.  Work
with coefficients indexed by the full multidegree `(d₁, d₂')`, finite in
number per total degree.  In total degree `k`, the Birkhoff recursion only
uses products of lower total degree, so the split is an exact finite
coefficientwise Laurent operation, not an analytic infinite-dimensional
problem.

After the raw bidegree fundamental solution is factored, keep working in the
same bidegree grading long enough to recover flat Novikov coordinates: take
the projected cone point from the negative factor, read the divisor
mirror-coordinate series, gauge them away, and invert the bidegree mirror map.
Only the resulting flat cone point should be restricted to rational rays
`(t, b·t)` for graph evaluation.  Reconstruction over `total+1` rays then
recovers every multidegree exactly.  The principle generalizes: *do the
loop-group projection and coordinate correction before throwing away Novikov
directions; use rays only as an exact coefficient-recovery device.*

### 16. You do not need the effective cone — just a cone that covers it

The honest obstruction to projective bundles is that the effective cone of
curve classes in `P(E)` is not something we want to compute (it depends on
the `a_l` in ways that get subtle).  We don't need it.  Non-effective
classes contribute *zero* — computationally, not just morally: their
I-function terms contain the full classical ring relation and vanish
identically.  So it suffices to enumerate a *convex cone guaranteed to
contain* the effective cone and let the zeros fall out.

For `P(O(a_1) ⊕ … ⊕ O(a_m))` the mechanism is a coordinate shift.  The fiber
degree `d₂ = ξ·β` can be negative (the exceptional section of a Hirzebruch
surface has `d₂ = −1`), which a nonnegative Novikov grading cannot index.
But `A = max a_l` bounds the negativity: every class with `d₂ < −A d₁` is
non-effective, so the shifted degree `d₂' = d₂ + A d₁ ≥ 0` is a valid
nonnegative grading covering the cone.  Two normalizations make this clean
and are worth internalizing:

- **`P(E) ≅ P(E ⊗ L)`**: twisting all summands by a line bundle preserves the
  variety but shifts the labelled tautological class and its curve coordinate.
  Normalize `min a_l = 0` together with that coordinate change.  This leaves
  fewer cases, and the shift `A` is then just `max a_l`.
- Choosing the grading divisor `D = ξ + (A+1)H` (rather than `ξ` alone)
  guarantees the classical eigenvalues `D|_{(i,j)}` are distinct at generic
  weights, so the ring is cyclic over `D` and the entire single-generator
  machinery applies to a rank-two Picard group.  The right cyclic generator
  turns "two divisors" back into "one divisor" for every purpose except the
  Novikov bookkeeping, which the grading already handles.

The lesson is general to toric-ish targets: negative or awkward curve
classes are a *grading* problem, not a *geometry* problem.  Find a shift and
a cyclic generator; the vanishing takes care of the rest.

### 17. Non-Fano is where the mirror map stops being a change of variables

The naive mirror transform — read the mirror map off the `1/z` divisor part,
exponential-gauge it away, invert, re-expand — works for Fano targets and
quietly fails for non-Fano ones.  The mechanism, made concrete on
`F_2 = P(O ⊕ O(2))`:

- `F_2` has an effective curve of *anticanonical degree zero*: the
  `(-2)`-section `B_-`, with `c_1·B_- = 0`.  (Fano surfaces like `F_0`, `F_1`
  have `c_1·β > 0` for every effective class.)
- For such a class the I-function term `I_β(z)` starts at `z^{0}` and can carry
  a **positive power of `z`** and a `z^0` part that is *not a multiple of the
  unit*.  So `I ≠ 1 + O(1/z)`, and projecting it onto the J-slice is a genuine
  Birkhoff (upper-triangular loop-group) step, not a divisor change of
  variables.
- The precise, cheap detector: the naive divisor-mirror exponent acquires a
  **nonzero unit component** exactly for these classes.  For Fano targets it is
  identically zero.  That detector was useful while the full projection was
  missing, but it is not the fix.
- The fix is to keep the bidegree Novikov grading through the Birkhoff stage:
  build the raw bidegree fundamental frame from the I-function and factor it
  directly over the finite bidegree Novikov ring.  The positive factor is the
  full cone projection, including mirror-coordinate and higher positive-`z`
  corrections.  A prior exponential "full-vector mirror gauge" is not
  equivalent; it passes `F_2` but fails already for `P(O(2) ⊕ O(-2))`.
  The negative factor's first column is still not ready for graph evaluation:
  the two divisor mirror-coordinate series must be extracted and inverted in
  bidegree before restricting to rays.  This is what fixes the higher
  `F_2` negative-section descendants.
- Truncating the Laurent tail is part of the math, not an implementation
  detail.  Once the positive factor has a term `z^p`, the recursive split can
  pull a lower-degree negative coefficient `z^{-s-p}` back into `z^{-s}`.
  The safe window is determined by the actual nonnegative `z` support of the
  raw bidegree fundamental frame and the maximum length of a lower-degree
  dependency chain.  A fixed local margin passed rank two and failed the
  rank-three deformation `P(O(2) ⊕ O(1) ⊕ O(-3)) ~ P^1 x P^2`.

Two meta-lessons outlived the specific bug.  First, **the diagnostic weight
`(t, b·t)` and full-vector debugging are worth building** — printing the raw
I-function's `z`-graded coefficient vectors is what turned "wrong rational
number" into "positive-`z` term at the `B_-` grade" in one look.  Second, and
more embarrassing: **every passing test used the R-matrix only to first
order.**  The genus-0, three-point curve counts that validated the Fano
bundles all have moduli dimension zero (`R`-order 1), so the `R`-matrix beyond
first order went entirely unexercised until a genus-1 four-point check caught
a disagreement.  The root cause was a fiber sign in the bundle R-asymptotic
constants: because `xi|_(i,j) = -c_ij`, the fiber difference for the
R-asymptotics is `(-c_il) - (-c_ij) = c_ij - c_il`, opposite to the Euler
factor used in the flat metric.  When a validation battery looks green, ask
what *order* of each object it actually reaches — coverage of cases is not
coverage of structure.
