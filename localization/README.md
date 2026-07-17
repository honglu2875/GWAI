# gw-pn

Experimental exact computations for Gromov-Witten invariants of projective
spaces, products, projective bundles, and negative split-bundle twists, with
exact symbolic Virasoro auditing for the compact theories.

Run commands through Cargo from the repository root:

```bash
cargo run --quiet -- <subcommand> <flags>
```

After installing/building the binary separately, replace `cargo run --quiet --`
with `gw-pn`.

The CLI is parsed with `clap`: it rejects unknown flags and suggests close
matches when possible (for example, `--max-descendants` reports
`tip: a similar argument exists: '--max-descendant'`). Run `gw-pn --help` or
`gw-pn <subcommand> --help` for the full flag list. Long flags also accept the
spelled-out aliases (`--genus` for `--g`, `--degree` for `--d`, and so on).

A code-pointer map of the layers below is in
[docs/architecture.md](docs/architecture.md); non-obvious findings and design
lessons from building this engine are collected in
[docs/lessons.md](docs/lessons.md).  The Virasoro convention, coefficient
extraction, and support boundary are documented separately in
[docs/virasoro.md](docs/virasoro.md).

## Quick Checks

Run the built-in validation suite:

```bash
cargo run --quiet -- tests
```

Run Rust unit tests:

```bash
cargo test --quiet
```

The optional GMP rational backend is behind a feature flag:

```bash
cargo test --quiet --features gmp-rational
```

## Insertions

Insertions are written as `tauK(CLASS)`.

Supported classes:

- `1`
- `H`
- `H^p`

Examples:

```bash
--insert 'tau0(H)'
--insert 'tau4(H)'
--insert 'tau3(H^2)'
--insert 'tau5(1)'
```

A bare class is shorthand for its primary insertion: `--insert H^2` means
`--insert 'tau0(H^2)'` (no shell quoting needed).

Pass multiple insertions by repeating `--insert`.

Library callers accepting untrusted class data should use
`CohomologyClass::try_new` or `CohomologyClass::try_h_power`. The legacy
infallible constructors now panic on coefficients outside the `P^n`
cohomology basis instead of silently discarding them; this is an intentional
pre-1.0 API tightening.

## `psi`

Computes pure Witten-Kontsevich psi intersections.

```bash
cargo run --quiet -- psi --g 2 --powers 4
```

Output:

```text
1/1152
```

Multiple markings use comma-separated powers:

```bash
cargo run --quiet -- psi --g 0 --powers 0,0,0
```

## `compute`

Computes ordinary `P^n` invariants through the Givental/S/R path.

Basic form:

```bash
cargo run --quiet -- compute --n <n> --g <genus> --d <degree> \
  --insert 'tauK(CLASS)' \
  --mode givental
```

`--d` may be omitted: the CLI then infers and reports the degree selected by
the nonequivariant dimension constraint.  Pass `--d` explicitly when requesting
an off-dimension equivariant class.  Nonequivariant requests vanish unless the
insertion degree equals a nonnegative virtual dimension; equivariant requests
can retain excess degree as a polynomial in the torus weights.

Supported `--mode` values:

- `givental`

Examples:

```bash
cargo run --quiet -- compute --n 2 --g 0 --d 1 \
  --insert 'tau0(H^2)' \
  --insert 'tau0(H^2)' \
  --insert 'tau0(H)' \
  --mode givental

# same invariant, with shorthand insertions and inferred degree
cargo run --quiet -- compute --n 2 --g 0 --insert H^2 --insert H^2 --insert H
```

## `twisted`

Computes negative split-bundle twists over `P^n`.

The flag `--twist -a,-b,-c` means:

```text
O(-a) + O(-b) + O(-c) -> P^n
```

The public CLI requires the minus signs. Internally the code still stores the
positive magnitudes because the sign is part of the negative split-bundle
convention.

Basic form:

```bash
cargo run --quiet -- twisted --n <n> --twist <negative-degrees> --g <genus> --d <degree> \
  --insert 'tauK(CLASS)'
```

### Verified Examples

`O(-1) -> P^2`, genus 2, degree 2:

```bash
cargo run --quiet -- twisted --n 2 --twist -1 --g 2 --d 2 --insert 'tau4(H)'
```

Output:

```text
-1/480
```

Other checked entries with the same target/genus/degree:

```bash
cargo run --quiet -- twisted --n 2 --twist -1 --g 2 --d 2 --insert 'tau5(1)'
cargo run --quiet -- twisted --n 2 --twist -1 --g 2 --d 2 --insert 'tau3(H^2)'
```

Expected outputs:

```text
0
-7/480
```

Local `P^2 = O(-3) -> P^2`, no insertions:

```bash
cargo run --quiet -- twisted --n 2 --twist -3 --g 2 --d 3
```

Output:

```text
3/20
```

Resolved conifold `O(-1) + O(-1) -> P^1`:

```bash
cargo run --quiet -- twisted --n 1 --twist -1,-1 --g 2 --d 3
```

Output:

```text
1/80
```

Notes:

- The current public twisted path computes non-equivariant negative split
  twists through an early rational lambda-line specialization.
- `--equivariant` on a twisted command keeps one symbolic fiber parameter
  `mu_i` for each summand `O(-a_i)`.  The CLI uses the factored rational
  coefficient engine by default in this mode because expanded symbolic output
  can be much slower even in small examples.  `--factored` is still accepted
  with `--equivariant` as an explicit spelling of the default behavior, and is
  rejected without `--equivariant`.
- In this fiber-equivariant mode the base `P^n` weights are still taken to the
  non-equivariant limit.  With ordinary non-equivariant insertions, the result
  often collapses to a finite polynomial in the `mu_i`; the factored engine is
  primarily a faster and safer evaluation strategy for these symbolic fiber
  weights, not a separate enumerative theory.
- Degree-zero local twisted invariants are not implemented in this path.

For formula/calibration inspection with fiber parameters:

```bash
cargo run --quiet -- formula --n 2 --twist -3 --g 2 --markings 1 \
  --basis raw \
  --equivariant
```

For symbolic fiber-equivariant invariant output:

```bash
cargo run --quiet -- twisted --n 2 --twist -1 --g 0 --d 1 \
  --insert 'tau1(H^2)' \
  --insert 'tau0(H)' \
  --equivariant
```

Two quick equivariant obstruction-polynomial checks over `P^2`:

```bash
cargo run --quiet -- twisted --n 2 --twist -3 --g 0 --d 1 \
  --insert 'tau0(H^2)' \
  --insert 'tau0(H^2)' \
  --insert 'tau0(H)' \
  --equivariant

cargo run --quiet -- twisted --n 2 --twist -2,-2 --g 0 --d 1 \
  --insert 'tau0(H^2)' \
  --insert 'tau0(H^2)' \
  --insert 'tau0(H)' \
  --equivariant
```

These print `mu_0^2` and `mu_0*mu_1`, respectively.  Their `mu_i=0` constant
terms match the corresponding non-equivariant local dimension check, while the
top fiber-weight coefficients match the ordinary untwisted degree-one line
count in `P^2`.

The same top-term check works in higher projective dimension:

```bash
cargo run --quiet -- twisted --n 3 --twist -4 --g 0 --d 1 \
  --insert 'tau0(H^3)' \
  --insert 'tau0(H^3)' \
  --insert 'tau0(H)' \
  --equivariant

cargo run --quiet -- twisted --n 3 --twist -2,-2 --g 0 --d 1 \
  --insert 'tau0(H^3)' \
  --insert 'tau0(H^3)' \
  --insert 'tau0(H)' \
  --equivariant
```

These print `mu_0^3` and `mu_0*mu_1`.  For a local-dimension constant-term
check, this command prints a factored expression that evaluates to `-40` for
generic `mu_0` and specializes to the same value as the non-equivariant twisted
path:

```bash
cargo run --quiet -- twisted --n 3 --twist -4 --g 0 --d 1 \
  --insert 'tau0(H^2)' \
  --insert 'tau0(H)' \
  --insert 'tau0(H)' \
  --equivariant
```

Genus-one and degree-two symbolic fiber-equivariant contractions are currently
useful stress tests rather than default quick checks.  For example, the
comparison values are already available from the familiar paths:

```bash
cargo run --quiet -- compute --n 2 --g 1 --d 1 --insert 'tau3(H)' --mode givental
cargo run --quiet -- compute --n 2 --g 0 --d 2 \
  --insert 'tau4(H^2)' \
  --insert 'tau0(H^2)' \
  --insert 'tau0(1)' \
  --mode givental
```

They return `1/8` and `1/2`.  The corresponding fiber-equivariant top terms
should have those coefficients after the appropriate obstruction-weight power,
but the full symbolic contractions are still part of the performance frontier.

## `product`

Computes `P^n x P^m` invariants by exact Novikov ray reconstruction: `--d` is
the total degree `d1 + d2`, and the command reports every bidegree.  Values
are non-equivariant invariants; bidegrees whose virtual dimension does not
match the insertions are reported as dimension mismatches.

Insertions are `tauK(CLASS)` with `CLASS` a `*`-product of `H1^a` and `H2^b`
factors (or `1`); a bare class means `tau0`.

```bash
# the unique (1,1)-curve through three general points on P^1 x P^1
cargo run --quiet -- product --n 1 --m 1 --g 0 --d 2 \
  --insert 'tau0(H1*H2)' --insert 'tau0(H1*H2)' --insert 'tau0(H1*H2)'

# ruling counts distinguish the two factors
cargo run --quiet -- product --n 1 --m 1 --g 0 --d 1 --insert H1*H2 --insert H1 --insert H1
```

Optional `--weights-x`/`--weights-y` set the rational equivariant weights;
the defaults are chosen so all fixed-point eigenvalue sums stay distinct.

## `bundle`

Computes invariants of a projective bundle `P(O(a_1) + ... + O(a_m))` over
`P^n` from its toric I-function.  `--twists` are nonnegative `a_l` and must
include zero.  An isomorphic presentation `P(E ⊗ L)` must be normalized by
the caller because tensoring changes the labelled tautological class `xi` and
the coordinate `xi . beta`; silently retaining those labels would request a
different invariant.  `--d` is
the *shifted* total degree `d1 + (d2 + (max a) d1)`, and the command reports
every curve class `(d1, d2)` in that slice — `d2 = xi . beta` may be negative
(the exceptional section of a Hirzebruch surface has `d2 < 0`).

Insertions are `tauK(CLASS)` with `CLASS` a `*`-product of `H^p` and `xi^q`
factors (or `1`); a bare class means `tau0`.

```bash
# F_1 = Bl_pt P^2 = P(O + O(1)) over P^1: classical int H xi = 1
cargo run --quiet -- bundle --n 1 --twists 0,1 --g 0 --d 0 --insert H --insert xi --insert 1

# the exceptional curve e = (1, -1): <xi, xi, xi>_e = -1
cargo run --quiet -- bundle --n 1 --twists 0,1 --g 0 --d 1 --insert xi --insert xi --insert xi
```

Optional `--weights-base`/`--weights-fiber` set the rational equivariant
weights; the defaults keep all grading eigenvalues distinct.

Validated scope includes genus-zero curve-counting invariants of Fano bundles
(`F_0`, `F_1`, and the like), the non-Fano `F_2 = P(O + O(2))` deformation
dictionary through genus one, the normalized mixed-sign rank-three bundle
`P(O + O(3) + O(3)) -> P^2`, and a zero-twist `P(O + O)` calibration check
against the product engine through higher `R` order.

Non-Fano support is deliberately fail-closed.  After the bidegree Birkhoff
projection, the backend removes positive-degree `z^-1` mirror coordinates in
the unit and the two divisor directions.  If a `z^-1` component remains in a
higher cohomology direction, the cone point lies on a genuinely big-quantum
path: `q d/dq` is no longer insertion of the grading divisor alone.  Such a
request returns `UnsupportedInvariant` until generalized mirror normalization
is implemented.  This currently includes the normalized `F_4` presentation
`P(O + O(4))` and the tested non-nef rank-three presentations
`P(O + O(1) + O(2))` and `P(O + O(4) + O(5))`.  Earlier isolated numerical
agreements for those targets are not treated as validation of their small GW
theories.  `F_2` and `P(O + O(3) + O(3)) -> P^2` pass the higher-primary check
and remain supported.

## `virasoro`

Generates and exactly audits finite coefficients of the corrected
Eguchi--Hori--Xiong/Getzler Virasoro equations.  This is separate from the
top-level `formula` command: `virasoro formula` displays a constraint on the
total descendant partition function, while `formula` displays a stable-graph
contraction skeleton.

The compact target selector is one of:

- `--n N` for `P^N`;
- `--n N --product-m M` for `P^N x P^M`; or
- `--n N --bundle-twists a_1,...,a_r` for
  `P(O(a_1)+...+O(a_r)) -> P^N`.

Bundle twists must use the canonical nonnegative presentation with minimum
zero, for the same `xi`-coordinate reason described under `bundle` above.

Two negative-split selectors have deliberately different meanings:

- `--n N --local-twist ...` names the local theory itself and returns the
  explicit QRR-required error described below; and
- `--n N --local-completion-twist ...` audits its distinguished section inside
  a compact projective bundle.  It generates the ordinary compact-completion
  equation, not a QRR-conjugated local Virasoro equation.

Use `--d d` for projective space and the geometric bidegree `--d d1,d2` for
products and bundles.  In particular, a bundle's `d2` may be negative; the
constraint generator receives the reconstructed geometric class, not a
Novikov-ray coefficient or shifted total degree.  Insertions use the same
homogeneous basis syntax as the corresponding computation command.

`P^n` examples:

```bash
# Render a non-linear L_1 coefficient for point theory.
cargo run --quiet -- virasoro formula --n 0 --k 1 --g 0 --d 0 \
  --insert 1 --insert 1 --insert 1 --insert 1

# Display and exactly check the genus-one L_0 anomaly equation.
cargo run --quiet -- virasoro check --n 0 --k 0 --g 1 --d 0 --show-formula

# This bounded scan closes with 90 verified-zero equations.
cargo run --quiet -- virasoro scan --n 0 --k-max 1 --g-max 1 --d-max 0 \
  --markings-max 4 --term-limit 1000000 \
  --dependency-markings-max 6 --dependency-descendant-max 2 \
  --dependency-limit 100000
```

Product examples:

```bash
cargo run --quiet -- virasoro formula --n 1 --product-m 1 \
  --k 0 --g 1 --d 0,0 --format tex

cargo run --quiet -- virasoro check --n 1 --product-m 1 \
  --k 0 --g 1 --d 0,0

cargo run --quiet -- virasoro scan --n 1 --product-m 1 \
  --k-min -1 --k-max -1 --g-max 0 --d-max 0 \
  --markings-max 2 --descendant-max 0 --equation-limit 100
```

Projective-bundle examples:

```bash
cargo run --quiet -- virasoro formula --n 1 --bundle-twists 0,1 \
  --k 0 --g 1 --d 0,0

cargo run --quiet -- virasoro check --n 1 --bundle-twists 0,1 \
  --k 0 --g 1 --d 0,0

# Nonlinear high-genus audit: F_1 exceptional class, with nonzero
# genus-reduction and degree-splitting contributions.
cargo run --quiet -- virasoro check --n 1 --bundle-twists 0,1 \
  --k 2 --g 2 --d 1,-1 --show-formula

cargo run --quiet -- virasoro scan --n 1 --bundle-twists 0,1 \
  --k-min -1 --k-max -1 --g-max 0 --d-max 0 \
  --markings-max 2 --descendant-max 0 --equation-limit 100
```

Negative-split projective-completion example:

```bash
# V = O(-1) + O(-1), A = 1, and the section class is (d,-A d).
cargo run --quiet -- virasoro formula --n 1 \
  --local-completion-twist -1,-1 --k 1 --g 2 --d 1,-1 --insert 1

cargo run --quiet -- virasoro check --n 1 \
  --local-completion-twist -1,-1 --k 1 --g 2 --d 1,-1 --insert 1 \
  --show-formula
```

For `V = direct sum_i O(-a_i)` and `A = max_i a_i`, this mode uses the
normalized compactification
`P(O(A) + direct sum_i O(A-a_i))`.  With `xi = -c1(S)`, the distinguished
section satisfies `xi|_S = -A H` and has class `(d,-A d)`.  Positive-degree
section dependencies are restricted by
`H^h xi^j -> (-A)^j H^(h+j)` and evaluated by the local twisted provider;
degree-zero dependencies are evaluated by the compact projective-bundle
backend.  A positive nonsection class is unsupported and therefore makes a
check incomplete rather than being treated as zero.  Insertions use bundle
syntax such as `tau1(H*xi)`.

This completion audit is particularly useful for testing high-genus twisted
values, but it is not generic twisted Virasoro.  Arbitrary multiplicative
twists still require the twisted pairing, degree-zero sector, and the
Quantum-Riemann--Roch-conjugated operators.  See
[docs/virasoro.md](docs/virasoro.md) for the precise boundary.

The point scan above reports 90 verified-zero equations, of which 26 are
backend-exercised and 64 are structural-only.  The two small product/bundle
scans each report 15 verified-zero equations, with 4 backend-exercised and 11
structural-only.  `--d-max` on a scan bounds the canonical theory's
theory-owned admissible grading: `d1+d2` for products and the shifted cone
grading for bundles.  The bundle's theory-owned shifted cone is a
conservative admissible support cone: a class outside it is certified
ineffective, while a class inside it has unknown effectivity and is queried
from the backend unless another structural rule applies.  Membership is
never interpreted as either a nonzero invariant or a zero invariant.
Standard compact operator generation caps `k` at `64`, independently
bounding bracket-polynomial and state-space matrix work when few terms are
emitted.  A single equation is also capped at 64 external markings so that
correlator-key payload cannot grow quadratically behind a small term count.
Exact product and bundle reconstruction is separately capped at 64 Novikov
rays (total reconstruction degree at most 63).  A dependency beyond that
frontier is reported as unsupported/incomplete instead of allocating a dense
unbounded interpolation system or spawning an unbounded thread family.
The shared stable-graph generator accepts `2g-2+n <= 8` and at most eight
labelled markings.  Formula and backend requests beyond that explicit work
envelope fail before graph-cache lookup, calibration, or worker creation.

Both `virasoro formula` and `virasoro check` expose `--term-limit` (default
`1000000`).  A check also accepts `--dependency-limit` (default `100000`) and
`--show-missing` (default `20`); the display limit does not change which
dependencies are required for a decisive result.

Scans have independent construction, retention, and evaluation envelopes:

- `--markings-max` bounds external markings in generated profiles and has a
  hard scan cap of `20`, because nonlinear equations enumerate labelled
  marking partitions;
- `--equation-limit` (default `10000`) bounds the complete Cartesian product
  of operators, genera, theory-owned curve classes, and external descendant
  profiles;
- `--term-limit` bounds the estimated unaggregated terms in each generated
  coefficient equation before marking partitions or matrix powers are
  materialized (default `1000000`);
- `--total-term-limit` bounds generated AST terms retained across the entire
  scan (default `1000000`);
- `--dependency-markings-max` bounds markings in each correlator dependency
  (default `--markings-max + 2`);
- `--dependency-descendant-max` bounds every individual psi power in a
  dependency (by default it is derived from `--k-max` and
  `--descendant-max`); and
- `--dependency-limit` bounds the number of canonical unique correlator
  dependencies considered per equation (default `100000`).

Exceeding the per-equation term budget rejects generation before the large
equation is allocated.  Dependency bounds are fail-closed instead: retained
keys outside a property bound are recorded as `OutsideBounds`; when the
unique closure exceeds `--dependency-limit`, one canonical omitted witness
is recorded and the report is explicitly marked truncated.  That equation is
`Incomplete`, never silently completed with a zero.

In addition to outcome counts, a scan prints four disjoint coverage counts:

- `backend-exercised`: at least one dependency was resolved by a computation
  backend; this category can still contain an incomplete equation;
- `structural-only`: a non-vacuous equation closed using only constants and
  canonical-theory-certified structural zeros, with no backend call;
- `vacuous`: no terms remained after exact symbolic aggregation; and
- `unresolved-only`: the equation was non-vacuous and incomplete, but no
  backend value was obtained.

These are coverage categories, not pass/fail categories.  A green scan can be
dominated by structural-only or vacuous equations and therefore provide very
little evidence about the reconstruction backend.  Treat it as a strong
backend audit only when `backend-exercised` coverage is meaningful in the
intended genera, curve classes, and descendant ranges.

An exact check has three API outcomes:

- `VerifiedZero` (CLI: `verified-zero`): every required correlator was
  evaluated or proved to be a structural zero, and the exact residual is
  zero;
- `Nonzero` (CLI: `NONZERO`): every dependency is known and the exact
  residual is nonzero; or
- `Incomplete` (CLI: `INCOMPLETE`): at least one correlator is unsupported,
  has `OutsideBounds` from the dependency envelope, or failed to evaluate.
  The displayed exact partial sum is diagnostic only.

Missing coefficients are never replaced by zero, and a nonzero partial sum
with missing dependencies is still `Incomplete`, not `Nonzero`.  `check`
exits successfully only for `VerifiedZero`; `scan` exits successfully only
when every generated equation is verified zero.

Direct negative-split/local targets are deliberately refused by the compact
checker.  For example,

```bash
cargo run --quiet -- virasoro check --n 2 --local-twist -3 \
  --k 0 --g 1 --d 0
```

reports that local Virasoro generation requires the twisted pairing and the
Quantum Riemann--Roch-conjugated operator.  The CLI does not substitute an
ordinary compact operator built from the noncompact total-space dimension and
`c_1`.  The separately named `--local-completion-twist` mode instead makes its
compact target explicit, so it does not change this refusal or claim to be a
generic local operator.  See [docs/virasoro.md](docs/virasoro.md) for the QRR
boundary and the precise Getzler convention.

## `formula`

Prints a human-readable Givental graph formula skeleton for fixed genus and
number of markings. This is an explanatory tool with two formula bases:

- `--basis raw`: the default engine-specialized symbolic graph formula. It
  substitutes the projective-space or twisted calibration into the packed graph
  kernels, leaving color/root sums, `R^{-1}`, `S`, `Psi`, `Delta`, and `eta`
  visible.
- `--basis coefficients`: the legacy fully unrolled coefficient basis in
  `R_k`, `S_k`, `T_k`, `Psi`, `Delta`, and point-theory psi integrals.

There is no separate `coefficient` subcommand; use
`formula --basis coefficients` for this view.

The default is `--basis raw`.  The older `--expand` flag is still accepted; it
now just requests the same engine dictionary that raw mode uses by default.

Use `--n` for `P^n` color count, or `--colors` for a provider-independent
semisimple CohFT skeleton:

```bash
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --max-descendant 5 \
  --d 3
```

For standalone TeX output, including a document preamble and TikZ graph
pictures, use either `--format tex` or the shorthand `--tex`:

```bash
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --max-descendant 5 \
  --format tex
```

Use `--format tex-fragment` when embedding the formula into an existing
document that already loads `amsmath`, `mathtools`, and `tikz`.

Use `--basis coefficients` for the provider-independent coefficient expansion.
The `--twist` flag is accepted in this mode but ignored, so the stable-graph
skeleton remains provider-independent:

```bash
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --basis coefficients \
  --twist -3 \
  --format tex-fragment
```

Use `--basis raw` to specialize the resolvent kernels to the current
projective-space or twisted calibration. Without `--twist`, this is ordinary
`P^n`; with `--twist`, this is the negative split backend:

```bash
cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --basis raw \
  --format tex

cargo run --quiet -- formula --n 2 --g 2 --markings 1 \
  --twist -3 \
  --basis raw \
  --format tex
```

The coefficient output defines the basis elements `S`, `PsiInv`, `RInv`, `T`,
`Delta`, `EtaInv`, and point-theory psi integrals, then lists the finite stable
graphs, truncation orders, and expanded graph terms. Marking and edge factors
are expanded directly in those basis elements rather than kept as separate
composite basis elements. The raw basis instead prints one compact graph
expression per stable graph, with the leg and edge kernel formulas substituted
directly into the graph bracket while descendant insertions stay packed in the
variables `z_l`.
Add `--no-glossary` for a shorter listing that still includes the graph
formulas. TeX mode uses standard Givental symbols such as `S_s`, `R^{-1}_r`,
`\Psi^{-1}`, `(T_p)_i`, `\Delta_i`, and
`\langle \tau_{p_1}\cdots\tau_{p_N}\rangle_h^{\mathrm{pt}}`. The renderer wraps
long displays itself: compact graph brackets use `multlined` (from
`mathtools`), while the fully expanded basis sums use a page-breakable `align*`,
so no display runs past the right margin or off the bottom of a page. It avoids
giant `\left...\right` delimiter pairs. Standalone `tex` output loads
`amsmath`, `mathtools`, `microtype`, and `tikz`; `tex-fragment` users should
load the first three (plus `tikz`) in their surrounding document.

## `resolvent`

Computes the fixed-degree labelled resolvent generating function

```text
sum_{a_i,k_i} <prod_i tau_{k_i}(H^{a_i})>_{g,d}
  prod_i t_i^{a_i}/a_i! * z_i^{-k_i-1}.
```

The resolvent is defined to be the exact virtual-dimension slice
`sum_i(a_i+k_i) = vdim`, so it is a finite Laurent polynomial in the `z_i^{-1}`
and a polynomial in the `t_i`.  This remains the definition with
`--equivariant`; that flag changes the coefficient ring, not the slice.  Each
coefficient is simplified by the same exact algebra engine used elsewhere in
the crate.

The command uses the packed S/R external-leg graph evaluator by default: it
precontracts the stable-graph sum once for fixed `(g,d,m)` and attaches all
resolvent coefficients to that shared kernel. Add `--validate` to also run the
older invariant-wise resolver and compare the two outputs.

For `--twist ... --equivariant`, the packed resolver uses the factored
coefficient engine so rational dependence on the fiber parameters `mu_i` is not
prematurely expanded. This is the preferred symbolic path for twisted
fiber-equivariant generating functions. Validation still works by expanding the
factored result only for the comparison step.

```bash
cargo run --quiet -- resolvent --n 2 --g 0 --d 1 --markings 3

cargo run --quiet -- resolvent --n 2 --twist -3 --g 2 --d 1 --markings 1 --validate

cargo run --quiet -- resolvent --n 1 --twist -1,-1 --g 1 --d 1 --markings 1 --equivariant
```

## `degree-series`

Computes invariants while varying the degree. Without `--twist`, this uses the
ordinary `P^n` Givental backend; with `--twist`, it uses the negative
split-bundle backend.

For a fixed insertion profile:

```bash
cargo run --quiet -- degree-series --n <n> --g <genus> --d-max <degree> \
  --insert 'tauK(CLASS)' \
  --mode givental
```

For a bounded scan over insertion profiles:

```bash
cargo run --quiet -- degree-series --n <n> --g <genus> --d-max <degree> \
  --max-markings <m> \
  --max-descendant <k> \
  --mode givental
```

Twisted local `P^2` example:

```bash
cargo run --quiet -- degree-series --n 2 --twist -3 --g 2 --d-max 3
```

Twisted scan example:

```bash
cargo run --quiet -- degree-series --n 2 --twist -1 --g 2 --d-max 2 \
  --max-markings 1 \
  --max-descendant 5
```

Degree ranges default to `--d-min 0` in the ordinary theory and `--d-min 1` in
the negative split-bundle theory. You can override this with `--d-min`.
When `--max-markings` is supplied, the command enumerates all monomials in
`tauK(1),...,tauK(H^n)` up to those bounds.  Nonequivariant scans prune profiles
outside the exact virtual dimension.  Equivariant scans retain excess-degree
profiles (and conservatively retain every bounded profile for localized
negative-split twists). `--include-zero` prints computed zero values too.

## `genus-series`

Computes invariants while varying the genus. As with `degree-series`, omitting
`--twist` selects ordinary `P^n`; adding `--twist` selects the negative
split-bundle theory.

For a fixed insertion profile:

```bash
cargo run --quiet -- genus-series --n <n> --d <degree> --g-max <genus> \
  --insert 'tauK(CLASS)' \
  --mode givental
```

For a bounded scan over insertion profiles:

```bash
cargo run --quiet -- genus-series --n <n> --d <degree> --g-max <genus> \
  --max-markings <m> \
  --max-descendant <k> \
  --mode givental
```

Twisted local `P^2` example:

```bash
cargo run --quiet -- genus-series --n 2 --twist -3 --d 1 --g-max 3
```

Twisted scan example:

```bash
cargo run --quiet -- genus-series --n 2 --twist -1 --d 2 --g-max 2 \
  --max-markings 1 \
  --max-descendant 5
```

Genus ranges default to `--g-min 0`. You can override this with `--g-min`.
When `--max-markings` is supplied, `--max-descendant` defaults to 0 if omitted.

## `series`

Enumerates a bounded sparse descendant potential for ordinary `P^n`.

Basic form:

```bash
cargo run --quiet -- series --n <n> --g <genus> --d-max <degree> \
  --max-markings <m> \
  --max-descendant <k> \
  --mode givental
```

Example:

```bash
cargo run --quiet -- series --n 2 --g 2 --d-max 4 \
  --max-markings 2 \
  --max-descendant 20 \
  --mode givental
```

Useful flags:

- `--include-zero` prints zero coefficients too.
- `--equivariant` requests equivariant projective-space data where supported;
  for negative split twists it means symbolic fiber parameters over an
  early-specialized base.

The library validates a finite sparse-series envelope before allocation:
state-space rank at most 64, `d-max <= 64`, at most eight markings, individual
descendant power at most 64, and at most 100,000 candidate coefficients under
the conservative profile-by-degree count.

If a series or resolvent command skips coefficients or falls back from a packed
path, the CLI writes those warnings to a temporary file and prints the path on
stderr. Informational engine notes are not classified as warnings. If the
temporary directory is unwritable, the warnings are printed directly to stderr
without changing an otherwise successful exit status. The library-level
`SeriesResult::is_complete()` reports whether any requested coefficient was
skipped.

## Environment Variables

Debug and tuning knobs, all off by default. Boolean flags are enabled by
`1`/`true`/`yes`/`on`/`full` (case-insensitive); anything else — including
`0` — leaves them off.

- `GW_PROFILE`: print timing and counter diagnostics for calibration, option
  construction, and graph contraction to stderr.
- `GW_THREADS`: worker thread count for parallel graph evaluation (defaults to
  available parallelism, capped at 8).
- `GWAI_DISABLE_RATIONAL_GRAPH`: disable the plain-rational fast path for
  graph sums with constant coefficients and force the symbolic evaluator.
- `GWAI_DISABLE_FACTORED_GRAPH`: disable the factored-denominator contraction
  tier for symbolic (equivariant) graph sums.
- `GWAI_DISABLE_GRAPH_CACHE`: do not read or write the on-disk stable-graph
  tables.
- `GWAI_GRAPH_CACHE_DIR`: directory for the on-disk stable-graph tables
  (default `$XDG_CACHE_HOME/gw-pn` or `~/.cache/gw-pn`).
- `GWAI_VALIDATE_TWISTED_CALIBRATION` (alias: `GW_VALIDATE_CALIBRATION`): run
  expensive self-adjointness, diagonalization, and unitarity checks on twisted
  calibrations before caching the graph kernel.

## Current Scope

The stable-graph/Givental engine is implemented for stable CohFT ranges. Some
unstable cases are handled by seed formulas or specialized paths; others report
an unsupported-invariant error rather than returning a guessed value.

Validation-only implementations and oracle tables live under
`src/validation_backends/`. This includes the Zinger cross-check path, Growi
rows, local Calabi-Yau tables, and the legacy direct stable-map localization
code. They are used by tests and diagnostics rather than as production
computation shortcuts.

## Architecture: Canonical Theories, Evaluators, and the CohFT Engine

`theory::GwTheory` is the sole canonical source of target geometry.  It owns
the homogeneous state space and unit, Poincare pairing, complex grading,
`c_1` action, numerical curve lattice, effectivity/admissible cone and degree
splittings, virtual dimension, and characteristic numbers.  The concrete
canonical theories are `ProjectiveSpaceTheory`, `ProductProjectiveTheory`,
`ProjectiveBundleTheory`, and the deliberately incomplete
`NegativeSplitTotalSpaceTheory`.

Universal identities consume only `GwTheory`.  In particular, the Virasoro
generator does not inspect a Givental calibration or reverse-engineer target
data from an evaluator.  It produces a backend-independent symbolic
constraint; a `CanonicalCorrelatorEvaluator` then maps its canonical
`BasisId` and `CurveClass` keys to a computation backend.

The Givental providers, `givental::target::GwTarget`, and product/bundle ray
objects are therefore evaluator and calibration adapters, not competing
descriptions of a theory.  They own algorithm-specific data such as fixed
point weights, I-functions, quantum multiplication, canonical frames,
`R`/`S` matrices, and ray interpolation.  Each production adapter exposes or
privately owns its canonical `GwTheory` and exposes it by shared reference;
compatibility is checked at that boundary.  In particular, `GwTarget` must
return its canonical theory and cannot independently restate dimension,
first-Chern degree, or effectivity.

Below the adapter boundary, the semisimple-CohFT engine consumes a
calibration and descendant `S`-matrix and contracts stable graphs without
inspecting target geometry.  `givental::recipe` constructs those calibrations
from a quantum ring or I-function, while `GiventalGraphKernel::from_parts`
accepts a direct calibration for experiments.  Picard-rank-two evaluators
specialize `(q1,q2)=(t,bt)` and reconstruct geometric bidegrees exactly by a
rational Vandermonde solve; the canonical theory, not the ray, remains the
authority for curve classes and their splittings.

See [docs/architecture.md](docs/architecture.md) for the full layer map and
[docs/virasoro.md](docs/virasoro.md) for the constraint convention.

## Performance Notes

The graph engine picks a coefficient representation per computation: plain
rationals when everything is constant (the non-equivariant production path),
factored rational functions for symbolic equivariant contractions, and
expanded `RatFun` only as a fallback.  Stable-graph tables are generated up to
isomorphism with individualization-refinement canonicalization and cached on
disk once generation is expensive.  Representative timings on `P^1` stationary
descendants: genus 3 in well under a second, genus 4 in about a minute (a few
seconds once the graph table is cached), genus-2 fully symbolic equivariant in
under two seconds.

## TODO

- A native multi-parameter Novikov series layer, as an alternative to ray
  reconstruction for higher Picard rank (rays scale as one engine run per
  reconstruction point and cannot carry symbolic equivariant weights; a
  native layer would share one run across all bidegrees).
- Relocate the H-Laurent / mirror-map / Birkhoff machinery from the twisted
  module into `givental` (the recipe entry points are already the seam).
  With the I-function recipe in place, further toric evaluators (complete
  intersections, more general toric varieties) can be paired with canonical
  `GwTheory` descriptions in the same way as projective bundles.
- Implement the generalized mirror transformation needed to return a
  Birkhoff-projected bundle cone point with higher-primary `z^-1` coordinates
  to the small quantum slice; until then those bundle presentations fail
  closed rather than feeding a big-quantum path to divisor reconstruction.
- Add the twisted pairing, degree-zero twisted sector, and independently
  generated QRR-conjugated Virasoro operators needed to audit
  negative-split/local theories without misusing the compact operator.
- Route the twisted and series/master equivariant paths through the same
  factored kernel construction the ordinary equivariant path now uses.
- Speed up high-genus stable-graph generation further: the remaining cost is
  the raw connected edge-multiset enumeration, which calls for orderly
  (canonical-extension) generation rather than enumerate-and-quotient.
- Genericize the J-calibration solve itself over the coefficient ring so the
  equivariant path never materializes expanded `RatFun` matrices at all.
