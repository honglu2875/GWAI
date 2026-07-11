# gw-pn

Experimental exact computations for Gromov-Witten invariants of projective
spaces and negative split-bundle twists.

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
[docs/lessons.md](docs/lessons.md).

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
`P^n` from its toric I-function.  `--twists` are the `a_l` (any integers,
normalized internally so `min a_l = 0`, since `P(E) = P(E ⊗ L)`).  `--d` is
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

Validated scope: genus-zero curve-counting invariants of Fano bundles (`F_0`,
`F_1`, and the like), non-Fano Hirzebruch deformation checks against
`P^1 x P^1` including `F_2` genus-one and `F_4 = P(O(2)+O(-2))` middle-class
cases, rank-three negative-direction checks against `P^1 x P^2`, and a
zero-twist `P(O + O)` calibration check against the product engine through
higher `R` order.

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

If the series command produces warnings or skipped coefficients, the CLI writes
them to a temporary file and prints the path on stderr.

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

## Architecture: Targets, Recipes, and the CohFT Engine

The computation core is a semisimple-CohFT evaluator: it consumes a
calibration (canonical frame data plus `R`-matrix) and a descendant
`S`-matrix, and contracts stable graphs without inspecting target geometry.
Three layers sit above it:

- **`givental::target::GwTarget`** describes a space: dimension, Fano-index
  datum, classical eigenvalue seeds at the torus fixed points, the
  quantum/classical divisor multiplication, and the insertion dictionary.
  `TargetProvider<T>` turns any implementation into an engine-facing
  provider.  `ProjectiveTarget` is the reference implementation and is held
  equal to the production `P^n` path by tests.
- **`givental::recipe`** holds the target-agnostic constructions: Newton
  root series and Lagrange frames from a quantum ring, the Dubrovin
  connection and `R`-flatness recursion with Bernoulli asymptotics derived
  from fixed-point weight differences, and two descendant `S`-matrix
  recipes — `descendant_s_from_divisor_qde` (from a quantum ring) and
  `descendant_s_from_i_function` (mirror map plus Birkhoff factorization of
  a cohomology-valued hypergeometric series, in the engine's metric-adjoint
  convention).  The two recipes cross-validate on untwisted `P^1`, where a
  rank-zero twist makes both available for the same theory.  The H-Laurent
  machinery the second recipe composes still lives in the `twisted` module;
  relocating it is mechanical follow-up.
- Providers may also supply calibrations directly
  (`GiventalGraphKernel::from_parts`) for experiments that bypass both
  recipes.

The `GwTarget` interface itself covers one Novikov variable and
divisor-generated rings; Picard-rank-two targets run on the same
single-variable engine through exact Novikov ray specialization
`(q1, q2) = (t, b t)` — a ring homomorphism, so each ray runs unchanged and
`total_degree + 1` rays determine every bidegree by a rational Vandermonde
solve.  Two rank-two targets ship:

- `P^n x P^m` (`givental::product`), validated against Behrend's product
  formula `R_{P^1 x P^1} = R_{P^1} (x) R_{P^1}` entrywise; see the `product`
  subcommand.
- projective bundles `P(O(a_1) + ... + O(a_m))` over `P^n`
  (`givental::bundle`) from their toric I-function, with bidegree Birkhoff
  projection plus bidegree mirror-coordinate correction before ray
  reconstruction, and a shifted grading that handles negative curve classes
  (the exceptional section of a Hirzebruch surface); see the `bundle`
  subcommand and
  [docs/lessons.md](docs/lessons.md) §§15–17.

The twisted theories are equivalent in spirit to a second family of targets
and will migrate onto the same interface once the I-function recipe is fully
extracted.

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
  module into `givental` (the recipe entry points are already the seam), and
  migrate the twisted theories themselves onto the target interface.  With
  the I-function recipe in place, further toric targets (complete
  intersections, more general toric varieties) register the same way the
  projective bundles do.
- Route the twisted and series/master equivariant paths through the same
  factored kernel construction the ordinary equivariant path now uses.
- Speed up high-genus stable-graph generation further: the remaining cost is
  the raw connected edge-multiset enumeration, which calls for orderly
  (canonical-extension) generation rather than enumerate-and-quotient.
- Genericize the J-calibration solve itself over the coefficient ring so the
  equivariant path never materializes expanded `RatFun` matrices at all.
