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
GW_PERF_GRAPH_CACHE_MODE=shared
```

The passes below were run on 2026-07-06 with:

```sh
# Earlier warm/cold audit runs.
scripts/run-perf-frontiers.sh --suite frontier --timeout 90 --no-build
scripts/run-perf-frontiers.sh --suite frontier --timeout 90 --no-build --graph-cache-mode cold
scripts/run-perf-frontiers.sh --suite extended --case formula_g4_m1 --timeout 90 --no-build --graph-cache-mode cold
scripts/run-perf-frontiers.sh --suite extended --case p2_g1_dmax2_m3_k3_series --timeout 90 --no-build --graph-cache-mode cold

# After stable-graph generation parallelism.
scripts/run-perf-frontiers.sh --suite frontier --timeout 90 --no-build --graph-cache-mode cold
scripts/run-perf-frontiers.sh --suite extended --case formula_g4_m1 --timeout 90 --no-build --graph-cache-mode cold
```

Raw local artifacts from these runs:

- `target/perf-frontiers/perf-frontiers-20260706T092435Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T092636Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T092721Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T105426Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T144907Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T145031Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T150315Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T150508Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T150643Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T195805Z.*`
- `target/perf-frontiers/perf-frontiers-20260706T195942Z.*`

Cache caveat: the frontier harness defaults to a shared project-local
`target/perf-frontiers/graph-cache/` directory because that is the useful
inner-loop tuning mode.  Use `--graph-cache-mode cold` when auditing execution
frontiers: it gives each case attempt a fresh stable-graph disk cache and
records the cache mode in the CSV/JSONL artifacts.  The table below keeps both
views.  Stable-graph cold generation remains visible, but parallel prefix
generation moved the sampled cold formula rows well below the previous timeout.

## Measured Rows

| mode | axis | probe | shared cache | cold graph cache | cold status | frontier signal |
|---|---|---|---:|---:|---|---|
| psi | genus | `g=10`, one marking, `psi^28` | 6.19s | 6.20s | ok | Not a main frontier yet, but high-genus point theory is not free. |
| formula | stable graphs | `g=2`, `m=1` | 0.01s | 0.02s | ok | Baseline. |
| formula | stable graphs | `g=3`, `m=1` | 0.02s | 0.06s | ok | Cold generation is now small. |
| formula | stable graphs | `g=3`, `m=2` | 0.11s | 1.46s | ok | Parallel prefix generation removed the previous sharp cold jump. |
| formula | stable graphs | `g=4`, `m=1` | not sampled | 31.70s | ok | Largest sampled formula row after graph-generation parallelism. |
| givental | primary/degree | `P^2`, `g=0`, `d=1`, three primaries | 0.03s | 0.03s | ok | Seed-sized. |
| givental | genus | `P^1`, `g=2`, `d=1`, `tau4(H)` | 0.18s | 0.18s | ok | Small. |
| givental | genus | `P^1`, `g=3`, `d=1`, `tau6(H)` | 1.25s | 0.97s | ok | Genus scaling visible but below frontier. |
| givental | dimension | `P^2`, `g=2`, `d=2`, `tau6(H^2)` | 0.78s | 0.77s | ok | Below frontier. |
| givental | dimension | `P^3`, `g=2`, `d=2`, `tau6(H^3)` | 3.01s | 3.01s | ok | Dimension/color scaling visible. |
| resolvent | markings | `P^2`, `g=0`, `d=1`, `m=3` | 0.11s | 0.11s | ok | Packed path is effective here. |
| resolvent | markings | `P^2`, `g=1`, `d=1`, `m=2` | 0.19s | 0.19s | ok | Below frontier. |
| series | markings/psi | `P^2`, `g=0`, `d<=2`, `m<=4`, `psi<=2` | 2.46s | 2.46s | ok | Candidate enumeration visible. |
| series | markings/psi | `P^2`, `g=1`, `d<=2`, `m<=3`, `psi<=3` | 5.54s | 4.80s | ok | Below frontier, but grows with candidate count. |
| series | degree/psi | `P^2`, `g=2`, `d<=3`, `m<=1`, `psi<=9` | 2.93s | 2.91s | ok | Degree sweep is still manageable. |
| twisted | degree | local `P^2`, `O(-3)`, `g=2`, `d=3` | 5.60s | 5.57s | ok | Current sampled local-P2 row is below frontier. |
| twisted | twist rank | conifold `P^1`, `O(-1)+O(-1)`, `g=2`, `d=3` | 0.76s | 0.76s | ok | Rank 2 is cheap in this sample. |
| twisted | twist rank | `P^2`, `O(-1)^3`, `g=2`, `d=2` | 1.96s | 1.95s | ok | Twist-factor count visible but below frontier. |
| twisted | equivariant | `P^2`, `O(-1)`, expanded symbolic, `g=0`, `d=1` | 0.18s | 0.18s | ok | Small case; no factored advantage visible. |
| twisted | equivariant | same, factored symbolic | 0.18s | 0.18s | ok | Same scale as expanded for this case. |
| product | degree | `P^1 x P^1`, `g=0`, total `d=2`, three points | 0.39s | 0.39s | ok | Ray reconstruction baseline. |
| product | genus/degree | `P^1 x P^1`, `g=1`, total `d=2`, `tau3(point)` | 0.51s | 0.51s | ok | Below frontier. |
| product | genus/degree | `P^1 x P^1`, `g=2`, total `d=3`, `tau6(point)` | 4.99s | 4.96s | ok | Ray parallelism moved this below the sampled frontier. |
| product | dimension | `P^1 x P^2`, `g=1`, total `d=2`, `tau3(H1*H2^2)` | 1.84s | 1.82s | ok | Color count matters but is not dominant yet. |
| bundle | degree | `P(O+O(2))`, `g=0`, shifted `d=3`, three primaries | 1.07s | 1.06s | ok | Non-Fano positive-z baseline. |
| bundle | genus/degree | `P(O+O(2))`, `g=1`, shifted `d=5`, three `tau1(point)` | 17.01s | 17.00s | ok | Still visible but below the one-minute frontier. |
| bundle | twist rank | `P(O(2)+O(1)+O(-3))`, `g=0`, shifted `d=3`, primary ruling | 22.25s | 22.09s | ok | Parallel bidegree Birkhoff moved this down, but bundle setup remains visible. |

## Frontier Table

| mode | genus | degree | dimension/colors | markings | psi classes | twist/rank factors | current frontier |
|---|---|---|---|---|---|---|---|
| formula/stable graphs | Still visible cold. `g=4,m=1` is now 31.70s instead of timing out at 90s. | Indirect, only through expansion metadata. | Indirect via expansion size. | Dominant cold axis. `g=3,m=2` is now 1.46s cold and 0.11s warm; `g=3,m=1` is 0.06s cold and 0.02s warm. | Affects rendered expansion size. | Twisted expansion adds labels, not the core count. | Parallel prefix generation moved sampled formula rows below one minute; larger `(g,n)` rows are the next graph frontier. |
| ordinary Givental `P^n` | Visible but below frontier in sampled single-invariant rows. | Increases q/R truncation and graph coefficient work. | Matrix size and color sums grow with `n+1`; `P^3` sample was 2.99s. | Expands external-leg states and graph contractions. | Raises `z_order`/`r_order`; high single psi still manageable here. | None. | For ordinary target algebra, not yet near one minute; stable graphs become frontier first. |
| twisted | Same stable-graph pressure as Givental. | Hypergeometric I-function, Birkhoff, and graph degree grow with `d`; local `P^2`, `g=2,d=3` was 5.62s. | Color count and relation degree grow with `n`. | Same graph/external-leg pressure. | Raises calibration order. | More negative factors increase hypergeometric products; sampled rank 3 was 1.95s. | Higher genus/degree symbolic leg products remain the suspected next frontier, not the sampled small rank. |
| product | Same graph pressure per ray. | Total degree gives `d+1` Novikov rays and larger q truncation. | Product colors multiply: `(n+1)(m+1)`. | Same graph/external-leg pressure. | Raises calibration order. | None. | Ray parallelism moved the sampled `P^1 x P^1`, `g=2,d=3` row to 4.99s; product is no longer a leading frontier in this suite. |
| bundle | Same graph pressure per ray. | Shifted total degree gives `d+1` rays and bidegree Birkhoff windows. | Size is `(n+1) * rank`; rank 3 is costly. | Same graph/external-leg pressure. | Raises S/R order and prewarm depth. | Positive-z and mixed twist signs drive bidegree Birkhoff/fundamental-S cost. | Rank-3 bundle primary at shifted `d=3` is 22.25s after parallel bidegree Birkhoff; per-ray fundamental-S/correction setup is now the visible bundle cost. |
| resolvent/series | Depends on packed graph contraction and candidate count. | Degree ranges multiply candidate coefficients. | Target color count affects packed kernels. | Markings can explode candidate terms, but packed rows sampled stayed under 6s. | `max-descendant` directly multiplies candidates. | Twisted resolvent adds calibration cost. | No one-minute row in sampled cases; candidate enumeration should be watched at higher `m,k`. |
| psi point theory | Recursion/table cost grows with genus and markings. | Not applicable. | Not applicable. | More markings increases partitions. | Powers determine dimension-valid tuples. | Not applicable. | `g=10` one marking was 6.19s; not the first optimization target. |

## Optimization Frontiers

1. Cold stable-graph enumeration is no longer the clearest sampled one-minute
   frontier after rollback DSU, once-per-key cache coordination, and parallel
   prefix generation.  A fresh `g=3,m=2` table is now 1.46s, and `g=4,m=1`
   finishes in 31.70s.  Larger `(g,n)` rows still need a separate
   canonicalization/orbit-handling pass.

2. Bundle setup is the next backend-specific practical target.  Product ray
   reconstruction is now parallel and the sampled product rows are below 5s;
   bundle rank/twist rows still spend visible time in shared bidegree
   calibration correction and per-ray fundamental-S construction.

3. Bundle rank/twist cost remains meaningful after the bounded Birkhoff fix.
   The rank-3 primary case is down to roughly 22s after parallel bidegree
   Birkhoff products, but the dominant profile is now ray fundamental-S
   construction and calibration correction rather than the Birkhoff matrix
   multiplication alone.

4. Twisted sampled rows are not currently frontier cases.  The existing module
   note that dense symbolic stable-graph leg products remain the likely
   frontier still looks plausible, but the next probe should intentionally use
   larger equivariant/factored rows rather than the small `g=0,d=1` comparison.

5. Packed resolvent and sparse series paths look healthy for the sampled rows.
   Future stress tests should increase `markings` and `max-descendant` together,
   because that is the axis most likely to create a candidate explosion.
