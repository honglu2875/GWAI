# Acceptance-test reporting

The ordinary Rust suite remains the push gate.  A smaller registry in
[`scripts/acceptance-tests.json`](../scripts/acceptance-tests.json) identifies
curated mathematical holdouts and regression fixtures: published tables, an
independent localization algorithm, cross-backend equivalences, deep-window
truncation oracles, backend-sensitive Virasoro relations, and explicitly
unsourced golden rows.
The registry is scheduling and reporting metadata, not a second source for
expected invariant values.  Expected values and assertions remain in the Rust
tests; independent-oracle provenance is recorded separately in
[`oracle-provenance.tsv`](oracle-provenance.tsv).

## Two cadences

The `push` acceptance suite contains the curated cases that already run in a
normal `cargo test`.  Running it explicitly is useful before a release or a
large reconstruction change because it produces one timing and log per case.
Normal push CI need not rerun it: an `--audit-only` invocation checks
the registry against Cargo's test inventory after the ordinary test jobs have
compiled the crate.

The cumulative `scheduled` suite contains the push cases and every ignored
acceptance test.  The currently ignored cases are:

| Category | Target | Scheduled evidence |
|---|---|---|
| external reference | negative split projective | eight published local-`P2` values through genus three and degree five |
| cross-backend equivalence | projective bundle | the rank-three primary shortcut against the generic graph series and insertion permutations |
| combinatorial reference | stable graphs | high-genus graph counts from the former brute-force generator |
| adaptive truncation | negative split projective | the O(-4), O(-5), O(-6), O(-8), O(-10) deep-Laurent grid |
| Virasoro backend | negative split projective | unmarked genus-two, degree-one `O(-2) -> P2` inverse-Euler QRR `L_0`: symbolic modes `z^1`, `z^3`, `z^5`, followed by the exact `mu_0=7` specialization, with live `mu_0`-dependent genus-reduction and degree-splitting sectors |
| Virasoro backend | product projective | genus-two `L_2` on the asymmetric six-color `P1 x P2` backend |
| Virasoro backend | projective bundle | the 281-term mixed-sign rank-three genus-two holdout |
| independent localization | projective bundle | production rank-three fiber-conic reconstruction versus fixed-tree localization |
| deformation equivalence | projective bundle | the full `F_2 -> P1 x P1` genus-zero/genus-one grid |
| deformation equivalence | projective bundle | harder descendants in the `F_2` class `(2,-1)` |
| deformation equivalence | projective bundle | the fixed genus-one `F_2` class `(2,-1)` value `-1/4` |

Run the suites from the package root:

```sh
scripts/run-acceptance-tests.sh --suite push
scripts/run-acceptance-tests.sh --suite scheduled --profile acceptance
scripts/run-acceptance-tests.sh --suite scheduled --profile acceptance --all-features
```

For a focused rerun, select one or more registry IDs:

```sh
scripts/run-acceptance-tests.sh \
  --case rank3_mixed_sign_genus2_virasoro \
  --profile acceptance
```

`--timeout-seconds` supplies the default per-case timeout, not an aggregate
timeout.  A registry case may carry a positive finite `timeout_seconds`
override when its established runtime needs a different safety ceiling.  Only
the inverse-Euler QRR genus-two holdout currently does so: its `1800`-second
ceiling accommodates cold and slower CI machines while all other cases retain
the command-line default.  The runner continues after a failure or timeout by
default, so one expensive failure does not hide the status and duration of
later oracles.

## Reports

Each run writes a timestamped directory below `target/acceptance-tests/` unless
`--output-dir` is supplied.  It contains:

- `results.jsonl`, with a run record, one record per test, and a summary record;
- `summary.md`, a human-readable table suitable for a CI step summary;
- `logs/<case-id>.log`, the complete output of each exact Cargo test; and
- a run-local stable-graph cache shared by the individually launched tests.

Every case record includes the stable registry ID, exact Rust test path, Cargo
target, cadence, category, mathematical target, oracle description, ignored
status, wall-clock duration, outcome, exit code, effective timeout, command,
and log path.  The run record separately reports the command-line default as
`default_timeout_seconds`, while every case record and the Markdown table show
the effective `timeout_seconds`.  The closed category vocabulary is `external-reference`,
`external-localization-golden`, `recorded-unsourced`,
`independent-localization`, `cross-backend-equivalence`,
`deformation-equivalence`, `adaptive-truncation`, `virasoro-backend`, and
`combinatorial-reference`.  Categories are therefore available without
parsing test names or prose, and the runner rejects category spelling drift.
Reports are build artifacts and stay under the ignored `target/` directory.

Before running cases, the script uses `cargo metadata` to enumerate every
testable package target, then asks each harness separately for its complete and
ignored test lists.  Test identities are `(cargo_target, test)`, so equal
function names in two harnesses cannot mask a missing registration or a stale
`ignored` flag.  It fails if a registered test disappeared, if an ignored test
in any harness is not registered, or if a registry entry's ignored status is
stale.  This makes the scheduled inventory closed under future `#[ignore]`
additions across library, binary, integration-test, example, and benchmark
harnesses while preserving the existing `lib` and `test:<name>` manifest
syntax.

The same discovery pass verifies every executable `cargo_target`/`cargo_test`
identity in the oracle-provenance TSV, including whether its `default_ci` flag
agrees with `#[ignore]`.  It also checks each package-relative
`source_locator`; Rust locators must name an existing function.  Rows backed by
one helper inside the built-in suite intentionally identify the aggregate
Cargo test and use the locator only to identify the contributing helper.

The runner's pure duplicate-name regression check can be run without compiling
the crate:

```sh
python3 scripts/acceptance_tests.py --self-check
```

## CI wiring

Push CI runs this after its existing test steps:

```sh
scripts/run-acceptance-tests.sh --audit-only --all-features
```

That performs discovery only and does not repeat mathematical computations.
The weekly workflow replaces opaque `cargo test -- --ignored` batches with a
two-entry backend matrix.  Each matrix entry runs the cumulative suite
with the optimized `acceptance` Cargo profile and chooses a distinct output
directory.  That profile omits the release profile's link-time optimization,
which materially reduces clean CI build latency without returning the exact
arithmetic-heavy holdouts to an unoptimized debug build:

```sh
scripts/run-acceptance-tests.sh --suite scheduled --profile acceptance \
  --timeout-seconds 480 \
  --output-dir target/acceptance-tests/default

scripts/run-acceptance-tests.sh --suite scheduled --profile acceptance --all-features \
  --timeout-seconds 480 \
  --output-dir target/acceptance-tests/all-features
```

Thus `480` seconds remains the workflow default; the manifest raises only the
QRR genus-two row to `1800` seconds.

The workflow appends `summary.md` to `GITHUB_STEP_SUMMARY` with an
`if: always()` step and uploads each whole report directory with
`actions/upload-artifact`.  The matrix keeps the default rational and optional
GMP runs independent and parallel; `fail-fast: false` preserves the second
backend's report when the first fails.
