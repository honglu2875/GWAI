# Performance Frontiers

Current working definition: a debug-build computation around one minute is an
execution frontier.  The reusable harness is `scripts/run-perf-frontiers.sh`
wrapping `scripts/perf_frontiers.py`; it writes timestamped and `latest.*`
Markdown, CSV, and JSONL results under `target/perf-frontiers/`.

Recommended tuning workflow:

```sh
# Before changing performance-sensitive code.
scripts/run-perf-frontiers.sh --suite smoke --save-baseline

# During a tuning pass; use --case to narrow the row under active work.
scripts/run-perf-frontiers.sh --baseline target/perf-frontiers/baseline.csv --case bundle --repeat 3

# Broader frontier pass when a change looks promising.
scripts/run-perf-frontiers.sh --baseline target/perf-frontiers/baseline.csv
```

Useful environment defaults for the wrapper:

```sh
GW_PERF_SUITE=frontier
GW_PERF_TIMEOUT=75
GW_PERF_FRONTIER_SECONDS=55
GW_PERF_REPEAT=1
```

The first pass below was run on 2026-07-06 with:

```sh
scripts/perf_frontiers.py --timeout 75 --frontier-seconds 55
scripts/perf_frontiers.py --timeout 75 --frontier-seconds 55 --case rank3_bundle --no-build
scripts/perf_frontiers.py --suite extended --case extended --timeout 75 --frontier-seconds 55 --no-build
```

Raw local artifacts from that run:

- `target/perf-frontiers/perf-frontiers-20260706T092435Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T092636Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T092721Z.*`

## Measured Rows

| mode | axis | probe | time | status | frontier signal |
|---|---|---|---:|---|---|
| psi | genus | `g=10`, one marking, `psi^28` | 6.24s | ok | Not a main frontier yet, but high-genus point theory is not free. |
| formula | stable graphs | `g=2`, `m=1` | 0.01s | ok | Baseline. |
| formula | stable graphs | `g=3`, `m=1` | 0.02s | ok | Still trivial. |
| formula | stable graphs | `g=3`, `m=2` | 20.76s | ok | Marking count causes a sharp jump. |
| formula | stable graphs | `g=4`, `m=1` | 75.07s | timeout | First clear one-minute frontier. |
| givental | primary/degree | `P^2`, `g=0`, `d=1`, three primaries | 0.04s | ok | Seed-sized. |
| givental | genus | `P^1`, `g=2`, `d=1`, `tau4(H)` | 0.19s | ok | Small. |
| givental | genus | `P^1`, `g=3`, `d=1`, `tau6(H)` | 1.95s | ok | Genus scaling visible but below frontier. |
| givental | dimension | `P^2`, `g=2`, `d=2`, `tau6(H^2)` | 0.80s | ok | Below frontier. |
| givental | dimension | `P^3`, `g=2`, `d=2`, `tau6(H^3)` | 3.10s | ok | Dimension/color scaling visible. |
| resolvent | markings | `P^2`, `g=0`, `d=1`, `m=3` | 0.12s | ok | Packed path is effective here. |
| resolvent | markings | `P^2`, `g=1`, `d=1`, `m=2` | 0.22s | ok | Below frontier. |
| series | markings/psi | `P^2`, `g=0`, `d<=2`, `m<=4`, `psi<=2` | 2.67s | ok | Candidate enumeration visible. |
| series | markings/psi | `P^2`, `g=1`, `d<=2`, `m<=3`, `psi<=3` | 5.54s | ok | Below frontier, but grows with candidate count. |
| series | degree/psi | `P^2`, `g=2`, `d<=3`, `m<=1`, `psi<=9` | 3.05s | ok | Degree sweep is still manageable. |
| twisted | degree | local `P^2`, `O(-3)`, `g=2`, `d=3` | 5.69s | ok | Current sampled local-P2 row is below frontier. |
| twisted | twist rank | conifold `P^1`, `O(-1)+O(-1)`, `g=2`, `d=3` | 0.80s | ok | Rank 2 is cheap in this sample. |
| twisted | twist rank | `P^2`, `O(-1)^3`, `g=2`, `d=2` | 2.02s | ok | Twist-factor count visible but below frontier. |
| twisted | equivariant | `P^2`, `O(-1)`, expanded symbolic, `g=0`, `d=1` | 0.23s | ok | Small case; no factored advantage visible. |
| twisted | equivariant | same, factored symbolic | 0.22s | ok | Same scale as expanded for this case. |
| product | degree | `P^1 x P^1`, `g=0`, total `d=2`, three points | 1.08s | ok | Ray reconstruction baseline. |
| product | genus/degree | `P^1 x P^1`, `g=1`, total `d=2`, `tau3(point)` | 1.44s | ok | Below frontier. |
| product | genus/degree | `P^1 x P^1`, `g=2`, total `d=3`, `tau6(point)` | 19.41s | ok | Approaching frontier; next degree/genus likely matters. |
| product | dimension | `P^1 x P^2`, `g=1`, total `d=2`, `tau3(H1*H2^2)` | 5.39s | ok | Color count matters but is not dominant yet. |
| bundle | degree | `P(O+O(2))`, `g=0`, shifted `d=3`, three primaries | 1.08s | ok | Non-Fano positive-z baseline. |
| bundle | genus/degree | `P(O+O(2))`, `g=1`, shifted `d=5`, three `tau1(point)` | 18.20s | ok | Approaching frontier. |
| bundle | twist rank | `P(O(2)+O(1)+O(-3))`, `g=0`, shifted `d=3`, primary ruling | 34.09s | ok | Current bundle frontier after bounded Birkhoff optimization. |

## Frontier Table

| mode | genus | degree | dimension/colors | markings | psi classes | twist/rank factors | current frontier |
|---|---|---|---|---|---|---|---|
| formula/stable graphs | Dominant. `g=4,m=1` timed out at 75s. | Indirect, only through expansion metadata. | Indirect via expansion size. | Dominant. `g=3,m=2` was 20.76s while `g=3,m=1` was 0.02s. | Affects rendered expansion size. | Twisted expansion adds labels, not the core count. | Stable-graph enumeration/canonicalization/rendering. |
| ordinary Givental `P^n` | Visible but below frontier in sampled single-invariant rows. | Increases q/R truncation and graph coefficient work. | Matrix size and color sums grow with `n+1`; `P^3` sample was 3.10s. | Expands external-leg states and graph contractions. | Raises `z_order`/`r_order`; high single psi still manageable here. | None. | For ordinary target algebra, not yet near one minute; stable graphs become frontier first. |
| twisted | Same stable-graph pressure as Givental. | Hypergeometric I-function, Birkhoff, and graph degree grow with `d`; local `P^2`, `g=2,d=3` was 5.69s. | Color count and relation degree grow with `n`. | Same graph/external-leg pressure. | Raises calibration order. | More negative factors increase hypergeometric products; sampled rank 3 was 2.02s. | Higher genus/degree symbolic leg products remain the suspected next frontier, not the sampled small rank. |
| product | Same graph pressure per ray. | Total degree gives `d+1` Novikov rays and larger q truncation. | Product colors multiply: `(n+1)(m+1)`. | Same graph/external-leg pressure. | Raises calibration order. | None. | `P^1 x P^1`, `g=2,d=3` is already 19.41s; ray parallelism/caching are likely next. |
| bundle | Same graph pressure per ray. | Shifted total degree gives `d+1` rays and bidegree Birkhoff windows. | Size is `(n+1) * rank`; rank 3 is costly. | Same graph/external-leg pressure. | Raises S/R order and prewarm depth. | Positive-z and mixed twist signs drive bidegree Birkhoff/fundamental-S cost. | Rank-3 bundle primary at shifted `d=3` is 34.09s; still the closest non-formula frontier. |
| resolvent/series | Depends on packed graph contraction and candidate count. | Degree ranges multiply candidate coefficients. | Target color count affects packed kernels. | Markings can explode candidate terms, but packed rows sampled stayed under 6s. | `max-descendant` directly multiplies candidates. | Twisted resolvent adds calibration cost. | No one-minute row in sampled cases; candidate enumeration should be watched at higher `m,k`. |
| psi point theory | Recursion/table cost grows with genus and markings. | Not applicable. | Not applicable. | More markings increases partitions. | Powers determine dimension-valid tuples. | Not applicable. | `g=10` one marking was 6.24s; not the first optimization target. |

## Optimization Frontiers

1. Stable-graph enumeration and formula rendering are the clearest global
   frontier.  The jump from `g=3,m=1` to `g=3,m=2`, and the `g=4,m=1` timeout,
   suggest that graph canonicalization/orbit handling and full formula
   materialization need a separate optimization pass.

2. Product and bundle ray reconstruction are the next practical targets.
   Product has no ray parallelism yet; bundle now does.  Product should likely
   get the same parallel ray reconstruction and warm shared calibration strategy
   that helped bundles.

3. Bundle rank/twist cost remains meaningful after the bounded Birkhoff fix.
   The rank-3 primary case is down to roughly 34s, but the dominant profile is
   now ray fundamental-S construction and direct quantum-product setup rather
   than the bidegree Birkhoff multiplication alone.

4. Twisted sampled rows are not currently frontier cases.  The existing module
   note that dense symbolic stable-graph leg products remain the likely
   frontier still looks plausible, but the next probe should intentionally use
   larger equivariant/factored rows rather than the small `g=0,d=1` comparison.

5. Packed resolvent and sparse series paths look healthy for the sampled rows.
   Future stress tests should increase `markings` and `max-descendant` together,
   because that is the axis most likely to create a candidate explosion.
