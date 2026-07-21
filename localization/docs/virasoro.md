# Virasoro constraint auditing

The Virasoro subsystem is an audit layer, not another reconstruction route.
It generates a finite coefficient equation, records every correlator on which
that equation depends, and asks an evaluation backend for those correlators.
The symbolic equation can be rendered as text or TeX before it is evaluated.

## Convention

We use the corrected Eguchi--Hori--Xiong convention written down by Getzler.
For a smooth projective target `X` of complex dimension `r`, choose a
homogeneous even basis `gamma_a`, with `gamma_0 = 1`, and put

```text
eta_ab = integral_X gamma_a gamma_b
mu_a   = p_a - r/2
c1(TX) gamma_a = R_a^b gamma_b.
```

Here `p_a` is the Hodge `p`-degree.  It is the complex codimension for the
Hodge--Tate targets currently in scope, but it must not silently be replaced
by half the real degree for a general target.  Indices on powers of `R` are
raised with `eta^{-1}`.  In particular, the quadratic coefficient tensor is

```text
[mu_c + m + 1/2]^k_i eta^(ac) (R^i)^b_c.
```

The grading factor is attached to the lower input index `c` before contraction
with `eta^(ac)`; it cannot be moved outside that contraction and evaluated at
the raised index `a`.  The scalar term in `L_0` is

```text
rho(X) = (1/48) integral_X ((3-r)c_r(TX) - 2 c1(TX)c_(r-1)(TX)).
```

The connected descendant potentials and total partition function are

```text
F_g = sum_(beta,n) Q^beta/n! <prod_i tau_(d_i)(gamma_(a_i))>_(g,beta)
                           prod_i t_(d_i)^(a_i),
Z   = exp(sum_(g >= 0) hbar^(g-1) F_g).
```

The operators annihilate `Z`, not an individual connected potential.  We use
the standard dilaton shift

```text
q_k^a = t_k^a - delta_(k,1) delta_(a,0).
```

In Givental's loop-space notation the unquantized generators are

```text
l_-1 = z^-1,
l_0  = z d/dz + 1/2 + mu + R/z,
l_m  = l_0 (z l_0)^m,                 m >= 1.
```

Their quantizations use `qq -> qq/hbar`, `qp -> q d/dq`, and
`pp -> hbar d^2/dq^2`.  With the sign convention used here,
`[L_m,L_n] = (m-n)L_(m+n)`.  The coordinate coefficients use Getzler's
polynomials

```text
product_(j=0)^m (s+x+j) = sum_(i=0)^(m+1) s^i [x]^m_i.
```

The implementation keeps the dilaton-shift, genus-reduction, disconnected
splitting, unstable-correction, and scalar terms distinct in the generated
AST.  This is deliberate: a formula that merely prints a final residual is
too difficult to audit for a missing factor of two or a shift error.

The authoritative coordinate formulas are equations `(Lk)`, `(L-1)`, and
`(zkg)` in [Getzler's *The Virasoro conjecture for Gromov--Witten
invariants*](https://arxiv.org/abs/math/9812026).  The loop-space and
quantization description is in [Givental's *Gromov--Witten invariants and
quantization of quadratic Hamiltonians*](https://arxiv.org/abs/math/0108100).

## Coefficient extraction

A request fixes an operator `L_m`, a genus `g`, a geometric curve class
`beta`, and a monomial in descendant times.  The generated equation is

```text
[Q^beta prod_j d/dt_(d_j)^(a_j)]
    (Z^-1 L_m Z) |_(t=0, hbar power g-1) = 0.
```

Using labelled derivatives rather than raw polynomial coefficients makes the
`1/n!` normalization unambiguous, including repeated insertions.  Expanding
the second-order part of `L_m` produces both

- a connected genus-reduction correlator in genus `g-1`; and
- products of connected correlators, summed over every marking partition,
  every `g_1+g_2=g`, and every canonical-theory-admissible
  `beta_1+beta_2=beta`.

For a product or projective bundle, `beta` is the canonical geometric
multidegree.  The constraint is generated in that multidegree, and the
evaluator reconstructs each requested coefficient before substitution.  The
formula uses componentwise splittings of the geometric class; a coefficient
on a specialized Novikov ray is not a substitute for a bidegree coefficient.
Product splittings are certified effective.  A projective bundle instead owns
a conservative shifted admissible cone.  A class outside that cone is
ineffective; a class or summand inside it has unknown effectivity and is
queried from the backend unless another structural-zero rule applies.  Cone
membership alone is never used to force either a nonzero value or a zero.

Explicit quadratic corrections supply the string and degree-zero unstable
terms.  The backend must not invent unstable correlators to imitate those
corrections, or they will be counted twice.

## Supported compact theories

The compact generator is for ordinary, non-equivariant descendant theories
with an even state space.  A canonical target theory must supply the homogeneous
basis, Poincare pairing and inverse, unit, complex grading, multiplication by
`c_1(TX)`, the classical cup product and a stabilizing-divisor policy for
unstable recursion, characteristic number `rho(X)`, a tri-state effectivity
answer, and the theory-owned admissible degree splittings required by the
requested coefficient.  Unknown effectivity means "query the backend", not
"assume effective".

In this crate that scope covers ordinary projective spaces, products of
projective spaces, and compact projective toric bundles once their true
multidegrees have been reconstructed.  Virasoro constraints for semisimple
targets are treated by Givental, and the bundle case is supported by
[Coates--Givental--Tseng, *Virasoro Constraints for Toric
Bundles*](https://arxiv.org/abs/1508.06282).

Odd cohomology is rejected because the current correlator key canonicalizes
insertions without Koszul signs.  Ordinary equivariant theory is also not a
drop-in use of these operators: frozen equivariant parameters do not obey
the conformal grading used by `L_0`.  It needs a separately specified
extended Euler operator.

## Negative-split projective-completion audits

The CLI has an explicit `--local-completion-twist` mode for a narrower but
useful audit of negative-split theories.  Let

```text
V = direct sum_i O(-a_i) over P^n,       A = max_i a_i.
```

It constructs the normalized compact projective bundle

```text
Y = P(O(A) + direct sum_i O(A-a_i)).
```

In the library's line-projectivization convention `xi = -c1(S)`, the section
selected by `O(A)` obeys

```text
xi|_S = -A H,
[degree-d section curve] = (d,-A d),
H^h xi^j|_S = (-A)^j H^(h+j).
```

The universal generator therefore sees one ordinary compact target, `Y`, and
prints a human-readable compact Virasoro equation in the `H,xi` basis.  Its
mixed evaluator has one exact routing rule:

- a positive dependency in class `(d,-A d)`, `d > 0`, is restricted to the
  section by the formula above and sent to the negative-split twisted
  provider;
- a degree-zero dependency is sent to the compact projective-bundle backend,
  because positive-degree concavity does not identify the degree-zero local
  and compact theories; and
- any other positive compact curve class is unsupported.  It remains a
  missing dependency, so the report is `Incomplete`, never an assumed zero.

Construction also fails closed unless the supplied local provider is in its
nonequivariant inverse-Euler calibration.  Euler, alternate-QRR, symbolic-base,
and fiber-equivariant modes are different theories and are not silently
reinterpreted by this adapter.

For example, the conifold completion can be rendered and checked with

```bash
cargo run --quiet -- virasoro formula --n 1 \
  --local-completion-twist -1,-1 --k 1 --g 2 --d 1,-1 --insert 1

cargo run --quiet -- virasoro check --n 1 \
  --local-completion-twist -1,-1 --k 1 --g 2 --d 1,-1 --insert 1 \
  --show-formula
```

Insertion parsing is the compact bundle notation: for example,
`tau1(H*xi)` or bare `xi`.  Degree input is always the rank-two compact class,
not the rank-one local degree.  A generic bounded compact scan will encounter
nonsection classes and fail closed; focused `formula` and `check` requests in
section classes are the intended high-genus audit interface.

This construction is not the Virasoro conjecture for an arbitrary twisted
theory.  It remains separate from the direct `--local-twist` path: the latter
now has an actual QRR-conjugated inverse-Euler `L_0` generator, twisted
pairing, and stable degree-zero evaluator, whereas the completion path checks
ordinary compact Virasoro equations.  Neither path is silently substituted
for the other, and direct local operators other than `L_0` still fail closed.
See [Coates--Givental, *Quantum Riemann--Roch, Lefschetz and
Serre*](https://arxiv.org/abs/math/0110142) for that conjugation framework.
Ordinary Virasoro constraints for the compact toric-bundle target are covered
by [Coates--Givental--Tseng, *Virasoro Constraints for Toric
Bundles*](https://arxiv.org/abs/1508.06282).

The regression suite evaluates these equations rather than merely checking
their shape:

- the native `F_1 = P(O + O(1))` backend checks `L_2`, genus two, class
  `(1,-1)`.  Both genus-reduction and degree-splitting terms contribute
  nontrivially, and requested genus-two descendants are nonzero;
- the slow rank-three hold-out normalizes
  `P(O(-2) + O(1) + O(1)) -> P^2` to `P(O + O(3) + O(3))` and checks
  `L_2`, genus two, class `(1,-2)` in the normalized `(H.beta, xi.beta)`
  coordinates.  All 281 terms close using 55 backend dependencies, including
  the nonzero invariant `<H xi^2>_(g=2,(1,-2)) = 1/10`; genus reduction and
  the genuine split `(1,-2) = (0,1) + (1,-3)` both contribute, and perturbing
  that genus-two value makes the residual nonzero.  The selected invariant is
  also reconstructed at a second generic equivariant-weight specialization;
- the resolved-conifold completion checks an `L_1`, genus-two equation with
  nonzero genus-two descendants; and
- a non-Calabi--Yau twisted case, `O(-2) -> P^2`, checks an `L_2`, genus-two,
  degree-one equation with 73 terms and 29 backend dependencies.

The two `L_2` tests perturb a nonzero genus-two dependency and require the
formerly zero residual to become nonzero.  Full descendant divisor recursion
also has direct product, bundle, and twisted dilaton regressions.  The native
`F_2` exceptional-class probe checks an `L_1`, genus-two equation in class
`(1,-2)` directly: all positive-degree dependencies are evaluated by the
bundle backend and the exact residual must vanish.  The independent
`F_2 -> P^1 x P^1` deformation checks remain separate audit oracles rather
than runtime substitutions.

The rank-three hold-out initially stopped with one missing fiber-class
descendant.  This exposed an evaluator gap rather than a Virasoro residual:
stabilization by the fiber divisor must use the theory-owned classical
product by `xi`, including reduction through `xi (xi + 3H)^2 = 0`.  Bundle
divisor recursion now expands that product in the canonical cohomology basis
instead of rejecting all fiber-only descendant corrections.

## Exact-check semantics

`ResidualReport` has three outcomes:

- `VerifiedZero`: every required correlator was obtained or proved zero, and
  their exact sum is zero.
- `Nonzero`: every required correlator was obtained or proved zero, and their
  exact sum is nonzero.
- `Incomplete`: at least one dependency is unsupported, outside the requested
  bounds, or failed to evaluate.  An exact partial sum may be shown for
  diagnosis, but it is neither a pass nor a failure.

All arithmetic is exact.  There is no numerical tolerance, and an absent
coefficient is never interpreted as zero.  Only structural zeros--for
example an unstable correlator, an ineffective curve class, or a proved
dimension mismatch--may be eliminated without querying the backend.  The
generated constraint and its report together retain term origins, individual
exact contributions, missing correlator keys, conventions, and formula
provenance so that the displayed equation is independently reviewable.

`OutsideBounds` is an exact missing-dependency reason, not a structural zero.
Dependency bounds are checked before structural-zero and backend resolution.
Consequently, even an excluded dependency that might later have proved zero
makes the equation `Incomplete`; the audit never reasons beyond its declared
envelope.

## Bounded scans and coverage

A scan has guards at both construction and evaluation time:

- ordinary Getzler operator generation currently caps `k` at `64`, bounding
  bracket-polynomial and state-space matrix work independently of term count;
- every single coefficient is capped at 64 external markings, bounding the
  payload cloned into labelled correlator keys independently of term count;
- `--markings-max` has a stricter scan cap of `20`, because a nonlinear scan
  enumerates labelled marking partitions for every generated profile;
- `--equation-limit` bounds the full operator/genus/curve/profile Cartesian
  product before the theory-owned bounded cone and profiles are allocated;
- `--term-limit` (default `1000000`) is a per-equation upper bound on the
  unaggregated coefficient expansion, checked before marking partitions,
  admissible degree splits, or matrix powers are materialized;
- `--total-term-limit` (default `1000000`) bounds the generated AST terms
  retained across the complete scan.  Report and evaluator-cache storage is
  controlled only indirectly by the dependency limits;
- `--dependency-markings-max` bounds the number of markings in each unique
  correlator dependency and defaults to `--markings-max + 2`;
- `--dependency-descendant-max` bounds each individual psi power.  When not
  specified, it is `--descendant-max` if `--k-max < 0`, and otherwise
  `max(--descendant-max + --k-max, --k-max + 1)`; and
- `--dependency-limit` (default `100000`) bounds the deterministic,
  canonically ordered unique dependency closure of each equation; and
- product and projective-bundle backends cap exact multi-degree
  reconstruction at 64 Novikov rays (total degree at most 63).  Dependencies
  beyond this implementation frontier remain explicit missing dependencies;
  they are never converted to zeros.

Single `formula` and `check` requests use the same default per-equation term
limit and both expose `--term-limit` to change it.  `check` additionally
exposes `--dependency-limit` and `--show-missing`; the latter limits printed
diagnostics, not the mathematical dependency envelope.  The term limit
rejects equation generation rather than returning a partial AST.  Correlator
bounds are different: property-bounded retained keys remain visible as
`OutsideBounds`.  If the unique closure exceeds `--dependency-limit`, the
report retains its canonical smallest omitted key as a witness and marks the
closure truncated instead of allocating the full remainder.  Either case
makes the residual `Incomplete`, even if the exact partial sum is zero.  This
fail-closed distinction prevents resource limits from becoming mathematical
assumptions.

Construction guards return the structured
`GwError::ResourceLimit { operation, requested, limit }` variant.  This is a
work-envelope result, not evidence that the invariant or identity is
mathematically unsupported; an audit must preserve the distinction rather
than convert either case to zero.  Backend `ResourceLimit` and
`UnsupportedFeature` errors retain their fields in the corresponding
`IncompleteReason` variants of a residual report; they are not flattened into
generic evaluation failures.

Outcome counts (`VerifiedZero`, `Nonzero`, and `Incomplete`) answer whether an
equation closed and what its exact residual was.  The scan also partitions
equations into four separate coverage categories:

- `backend-exercised`: at least one dependency was resolved by the
  computation backend; other dependencies may still be unresolved;
- `structural-only`: the non-vacuous equation closed using only constants
  and/or certified structural zeros;
- `vacuous`: exact symbolic aggregation left no terms; and
- `unresolved-only`: the non-vacuous equation was incomplete and had no
  backend value.

A green scan means every generated equation is `VerifiedZero`, but it is not
automatically strong evidence for the invariant engine.  If most equations
are structural-only or vacuous, the backend was barely tested.  Audit reports
should quote the coverage counts and claim backend evidence only when
backend-exercised equations meaningfully cover the intended genera, geometric
curve classes, insertion types, and descendant powers.

## Negative-split and local theories

A negative split bundle is represented by an inverse-Euler-twisted theory on
its compact base.  It is not the ordinary compact GW theory of the
noncompact total space, so substituting the total-space dimension and
`c_1` into Getzler's compact operator is not valid.

For a multiplicative class

```text
c(V) = exp(sum_(k>=0) s_k ch_k(V))
```

and a bundle `E`, Quantum Riemann--Roch changes the pairing and identifies the
twisted Fock coordinate with the ordinary one by

```text
(a,b)_tw = integral_X c(E) a b,
q_CG(z)  = sqrt(c(E)) (t(z)-z).
```

Define

```text
A_- = sum_(l>0) s_(l-1) ch_l(E) z^-1,
A_+ = sum_(m>0,l>=0)
        s_(2m-1+l) B_(2m)/(2m)! ch_l(E) z^(2m-1).
```

In the Coates--Givental Hamiltonian convention,

```text
U_(c,E) = exp(hat_CG(A_+)) exp(hat_CG(A_-)).
```

The crate's Getzler normal-order map has the opposite infinitesimal sign, so
the same differential operator is

```text
U_(c,E) = exp(-hat_G(A_+)) exp(-hat_G(A_-)).
```

Recording both signs is essential; copying the first display into the
Getzler coordinate implementation would conjugate in the wrong direction.
The appropriate fixed-parameter annihilator is

```text
L_m^tw = U_(c,E) L_m^base U_(c,E)^-1,
```

not the ordinary `L_m` for a putative total space.  See
[Coates--Givental, *Quantum Riemann--Roch, Lefschetz and
Serre*](https://arxiv.org/abs/math/0110142), especially Theorem 1 and
equation (7).  The QRR genus-one determinant changes the partition function
by a central scalar; it cancels from this operator conjugation and is therefore
not a missing term in `L_m^tw`.

### Symbolic artifact

`QrrConjugationFormula` is a backend-independent, public representation of
the two Hamiltonians above.  Its finite term table records the exact
Bernoulli multiplier, Chern-character index, characteristic-parameter index,
and loop-space power of every retained term; its text and TeX renderers also
print the untruncated formal identity, pairing, Fock-space identification,
operator order, both hat conventions, and source.  It is intended to be
inspectable before a target backend or an invariant is selected.

The negative-split provider specializes that artifact to the inverse-Euler
class of

```text
E = direct sum_i O(-a_i) over P^n,
u_i = mu_i - a_i H,
c(E) = product_i u_i^-1.
```

The first closed specialization is `L_0`.  After pullback to native twisted
coordinates, where `q_tw=t-z`, it is

```text
B_0 = L_0^(P^n) + sum_r K_r(H) z^r,

K_-1 = sum_i sum_(l=2)^n a_i^l H^l/(l mu_i^(l-1)),
K_0  = 1/2 sum_i sum_(l=1)^n a_i^l H^l/mu_i^l,
K_(2m-1) = B_(2m)/(2m) sum_i mu_i/u_i^(2m).
```

All raised and lowered tensors use the twisted pairing.  Positive modes add
both connected genus-reduction terms and every labelled genus/degree/marking
splitting.  The possible new `L_0` projective quantization cocycle is the
trace of nilpotent multiplication and vanishes; the ordinary `P^n` anomaly
remains.  The positive odd mode cutoff is not a global heuristic.  For one
coefficient, with base virtual dimension `V` and external insertion degree
`D_M`, vector terms require `r <= V-D_M`, and second-order terms require
`r <= V-D_M+n`.  Construction refuses a requested expansion beyond the
public mode cap before allocating it.

This finite bound remains valid with symbolic fiber weights for a specific
inverse-Euler reason; it is not the ordinary nonequivariant dimension-equality
shortcut.  After the base-equivariant parameter tends to zero,
`1/e(R pi_* E)` has an expansion
`sum_(j>=0) mu^(-chi(E)-j) c_j`, with `c_j` of ordinary codimension `j`
(and the corresponding multi-index expansion for a split bundle).  A
correlator with insertion degree `D_M` can therefore be nonzero only when
`D_M+j=V`, so in particular `D_M<=V`.  For a genus-reduction or splitting
term, the sum of the boundary-stratum base virtual dimensions is `V+n-1`;
the positive mode contributes total psi degree `r-1`.  This gives the stated
second-order bound `r<=V-D_M+n`.  A different multiplicative class needs its
own finiteness proof before this specialization strategy can be reused.

The same provider owns evaluation.  It keeps each `mu_i` symbolic, evaluates
stable degree-zero twisted correlators, and evaluates positive degree through
the fiber-equivariant hypergeometric/Birkhoff/Givental backend.  Positive
degree does not make the underlying pointed curve stable: genus-zero one- and
two-point dependencies generated by QRR are reconstructed with the full
descendant divisor equation, including cup-product correction branches.
Unsupported or failed dependencies make the residual `Incomplete`.

This is a fixed-parameter Virasoro operator: each `mu_i` is frozen in the
coefficient field, and there is no `mu_i d/dmu_i` term.  An extended
equivariantly homogeneous Euler operator would be a different convention.

The public specialization artifact can evaluate those frozen parameters at
an exact rational point without losing provenance.  In outline:

```rust
let evaluator = NegativeSplitFixedFiberQrrEvaluator::new(n, degrees, mu_values)?;
let specialized = evaluator.specialize_constraint(&symbolic_constraint)?;
let report = evaluate_constraint(&evaluator, specialized.constraint());
```

`SpecializedVirasoroConstraint` retains the theory fingerprint, operator,
sector, time coefficient, conventions, and formula source verbatim, and
separately records the named `mu_i` assignments.  Every declared parameter
must be present; unknown names and coefficient poles fail closed.  The paired
provider fixes the nonzero fiber weights before calibration but keeps
`lambda_i=w_i lambda_0` symbolic through each complete graph sum, making the
graph arithmetic univariate before the final `lambda_0 -> 0` limit.  This
is evaluation at a point of the fiber-equivariant coefficient field, not the
ordinary all-weights nonequivariant limit, so the equivariant dimension policy
and the inverse-Euler upper-bound pruning remain in force.  Focused tests
compare stable degree-zero and positive-degree values with the fully symbolic
factored evaluator followed by the same exact `mu_i` substitution.

For example, the complete symbolic operator and a coefficient equation can
be rendered without evaluating any correlator:

```bash
cargo run --quiet -- virasoro formula --n 2 --local-twist -2 \
  --k 0 --g 2 --d 1
```

Here the absence of `--insert` selects the unmarked (empty time-monomial)
coefficient.  The resulting genus-two, degree-one equation contains the
positive QRR modes `z^1`, `z^3`, and `z^5`; both its genus-reduction and its
degree-splitting sector have live coefficients depending on `mu_0`.

The exact residual check uses the same operator object and cutoff, then
specializes both the equation and its backend to `mu_0=7`:

```bash
cargo run --quiet -- virasoro check --n 2 --local-twist -2 \
  --k 0 --g 2 --d 1 --fiber-weights 7 --show-formula
```

`--fiber-weights` parses exact nonzero rationals, not floating-point
approximations: a rank-one value may be `7` or `7/2`, while a rank-two twist
could use `3/2,5`.  Formula generation remains symbolic; specialization is a
separate, provenance-preserving step used to turn the human-readable artifact
into an executable invariant test.

The `mu_0=7` check is a scheduled acceptance test because its stable
degree-zero and positive-degree genus-two graph sectors are intentionally
expensive.  The acceptance registry runs it per case with a timeout and
durable JSONL and Markdown reports.

For inverse Euler classes the fiber-equivariant parameters must remain
invertible while the conjugated equation is formed; a non-equivariant limit
can be taken only after cancellation.  The twisted metric, degree-zero
sector, quantization scalar, and specialization order are all part of the
convention.  Setting `mu_i=0` term by term in the operator is not a valid
Fock-space specialization.

The current implementation boundary is deliberate: inverse-Euler `L_0` is
specialized and evaluable; arbitrary multiplicative classes have the general
symbolic QRR artifact but need theory/provider-owned characteristic data and
degree-zero evaluators before they can become invariant tests.  Higher
`L_m` operators require exact differential-operator conjugation, including
the projective quantization cocycle, rather than reusing the closed `L_0`
formula.  Those requests fail closed.
