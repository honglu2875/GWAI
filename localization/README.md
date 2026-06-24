# gw-pn

Experimental exact computations for Gromov-Witten invariants of projective
spaces and negative split-bundle twists.

Run commands through Cargo from the repository root:

```bash
cargo run --quiet -- <subcommand> <flags>
```

After installing/building the binary separately, replace `cargo run --quiet --`
with `gw-pn`.

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
- Full symbolic equivariant twisted output is not enabled from this command.
- Degree-zero local twisted invariants are not implemented in this path.

## `degree-series`

Computes one fixed insertion profile while varying the degree. Without
`--twist`, this uses the ordinary `P^n` Givental backend; with `--twist`, it
uses the negative split-bundle backend.

Basic form:

```bash
cargo run --quiet -- degree-series --n <n> --g <genus> --d-max <degree> \
  --insert 'tauK(CLASS)' \
  --mode givental
```

Twisted local `P^2` example:

```bash
cargo run --quiet -- degree-series --n 2 --twist -3 --g 2 --d-max 3
```

Degree ranges default to `--d-min 0` in the ordinary theory and `--d-min 1` in
the negative split-bundle theory. You can override this with `--d-min`.

## `genus-series`

Computes one fixed insertion profile while varying the genus. As with
`degree-series`, omitting `--twist` selects ordinary `P^n`; adding `--twist`
selects the negative split-bundle theory.

Basic form:

```bash
cargo run --quiet -- genus-series --n <n> --d <degree> --g-max <genus> \
  --insert 'tauK(CLASS)' \
  --mode givental
```

Twisted local `P^2` example:

```bash
cargo run --quiet -- genus-series --n 2 --twist -3 --d 1 --g-max 3
```

Genus ranges default to `--g-min 0`. You can override this with `--g-min`.

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
- `--equivariant` requests equivariant projective-space data where supported.

If the series command produces warnings or skipped coefficients, the CLI writes
them to a temporary file and prints the path on stderr.

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

- Enable full symbolic equivariant output for negative split-bundle twisted
  theories. The current public twisted command computes non-equivariant answers
  by early rational lambda-line specialization.
- Generalize the reconstruction interfaces beyond `P^n`, with twisted,
  equivariant, and eventually other semisimple CohFT targets sharing the same
  Givental graph evaluator.
- Improve performance for the genus-4 local-curve frontier, especially
  `O(-1) + O(-1) -> P^1`, where graph recursion and calibration caching are the
  next bottlenecks.
- Continue optimizing the main untwisted path for larger genus, degree, and
  marking bounds. The likely targets are batched series evaluation, more
  aggressive graph pruning, and reduced repeated `S/R` materialization.
