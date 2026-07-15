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
`c_1(TX)`, characteristic number `rho(X)`, a tri-state effectivity answer,
and the theory-owned admissible degree splittings required by the requested
coefficient.  Unknown effectivity means "query the backend", not "assume
effective".

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

For a multiplicative class `c` and bundle `E`, Quantum Riemann--Roch changes
the pairing and Fock coordinate to

```text
(a,b)_tw = integral_X c(E) a b,
q_tw(z)  = sqrt(c(E)) (t(z)-z),
```

and relates the potentials by a quantized Bernoulli/Chern-character
operator `Delta`.  The appropriate fixed-parameter annihilator is therefore

```text
L_m^tw = Delta_hat L_m^base Delta_hat^-1,
```

not the ordinary `L_m` for a putative total space.  See
[Coates--Givental, *Quantum Riemann--Roch, Lefschetz and
Serre*](https://arxiv.org/abs/math/0110142), especially Theorem 1.

For inverse Euler classes the fiber-equivariant parameters must remain
invertible while the conjugated equation is formed; a non-equivariant limit
can be taken only after cancellation.  The twisted metric, degree-zero
sector, quantization scalar, and specialization order are all part of the
convention.  Until a backend supplies that QRR-conjugated operator and the
required degree-zero twisted correlators, a negative-split/local request is
reported as unsupported or incomplete, never checked with the compact
operator.
