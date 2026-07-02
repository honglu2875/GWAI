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

## Quick Checks

Run the built-in validation suite:

```bash
cargo run --quiet -- tests
```

Run Rust unit tests:

```bash
cargo test --quiet
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

Supported `--mode` values:

- `givental`

Examples:

```bash
cargo run --quiet -- compute --n 2 --g 0 --d 1 \
  --insert 'tau0(H^2)' \
  --insert 'tau0(H^2)' \
  --insert 'tau0(H)' \
  --mode givental
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

For fixed genus, degree, and number of markings, the virtual dimension fixes
`sum_i(a_i+k_i)`, so this is a finite Laurent polynomial in the `z_i^{-1}` and
a polynomial in the `t_i`.  Each coefficient is simplified by the same exact
algebra engine used elsewhere in the crate.

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
`tauK(1),...,tauK(H^n)` up to those bounds and prunes dimension-incompatible
profiles before running the graph evaluator. `--include-zero` prints computed
zero values too.

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

## TODO

- Continue improving the factored rational-function engine for full symbolic
  equivariant negative split-bundle graph contractions.  It is the default for
  `twisted --equivariant`; non-equivariant computations use the ordinary
  rational engine.  Remaining work: avoid dense canonical-leg product blow-up
  in larger stable symbolic graph contractions.
- Generalize the reconstruction interfaces beyond `P^n`, with twisted,
  equivariant, and eventually other semisimple CohFT targets sharing the same
  Givental graph evaluator.
- Improve performance for the genus-4 local-curve frontier, especially
  `O(-1) + O(-1) -> P^1`, where graph recursion and calibration caching are the
  next bottlenecks.
- Continue optimizing the main untwisted path for larger genus, degree, and
  marking bounds. The likely targets are batched series evaluation, more
  aggressive graph pruning, and reduced repeated `S/R` materialization.
