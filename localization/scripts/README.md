# GW-Pn Scripts

This directory contains throw-away diagnostics that use the `gw-pn` library but
are not compiled by the root package by default.

Run the stable-graph stats script with:

```sh
scripts/run-graph-stats.sh --g-max 2 --markings-max 2
```

Useful options:

```sh
scripts/run-graph-stats.sh --release --g-max 3 --markings-max 2 --csv
scripts/run-graph-stats.sh --g-min 2 --g-max 2 --markings-min 0 --markings-max 2
scripts/run-twisted-raw-values.sh
scripts/run-twisted-raw-values.sh --target conifold --genus 2 --d-min 1 --d-max 4
scripts/run-perf-frontiers.sh
scripts/run-perf-frontiers.sh --suite smoke --save-baseline
scripts/run-perf-frontiers.sh --baseline target/perf-frontiers/baseline.csv --repeat 3
scripts/run-perf-frontiers.sh --suite extended --case extended --no-build
```

The current stable-graph generator is exact but naive for higher genus and
multiple markings; rows such as `g=3, markings=2` can be expensive.

`run-twisted-raw-values.sh` compares negative split-bundle graph values with
local Calabi-Yau oracle tables.  It prints whether the final rational value
matches the oracle, using the same rational fast path and finite lambda-line
limit fallback as the public twisted CLI.  The script remains useful as a
performance diagnostic outside the default test path.

`run-perf-frontiers.sh` wraps `perf_frontiers.py`, which runs curated CLI
workloads for the ordinary Givental, twisted, product, bundle, formula,
resolvent, series, and psi paths.  It treats roughly one minute as the
execution frontier, applies a per-case timeout, and writes timestamped plus
`latest.*` Markdown, CSV, and JSONL results under `target/perf-frontiers/`.
Use `--save-baseline` before an optimization pass and `--baseline PATH` after
it to get percentage deltas; use `--repeat N` for noisy rows.
