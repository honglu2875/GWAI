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
```

The current stable-graph generator is exact but naive for higher genus and
multiple markings; rows such as `g=3, markings=2` can be expensive.

`run-twisted-raw-values.sh` compares negative split-bundle graph values with
local Calabi-Yau oracle tables.  It prints whether the raw equivariant value is
already the oracle value and then compares the finite lambda-line limit.  The
default provider now uses early rational lambda-line specialization, so matching
rows should normally have `raw-equals-oracle=true`.  The script remains useful
as a performance diagnostic outside the default test path.
