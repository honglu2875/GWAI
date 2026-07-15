# Performance Frontiers

Current working definition: a debug-build computation around one minute is an
execution frontier.  The reusable harness is `scripts/run-perf-frontiers.sh`
wrapping `scripts/perf_frontiers.py`; it writes timestamped and `latest.*`
Markdown, CSV, and JSONL results under `target/perf-frontiers/`.

Exact product and projective-bundle interpolation currently has an explicit
64-ray ceiling (total reconstruction degree at most 63).  The implementation
still uses a dense Vandermonde solve and one scoped worker per ray; larger
requests fail before warmup, allocation, or thread spawning.  Raising this
ceiling should follow a bounded worker pool and a more scalable interpolation
strategy, not merely a larger constant.

Stable-graph generation likewise has an explicit built-in envelope
`2g-2+n <= 8` with at most eight labelled markings.  The complexity-eight
`g=4,m=2` and `g=5,m=0` probes below remain deliberate boundary cases; larger
formula/backend requests now fail before graph enumeration or cache lookup.

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
GW_PERF_FEATURES=
GW_PERF_ALL_FEATURES=
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

# New frontier probes after graph generation stopped dominating the sampled suite.
GWAI_GRAPH_CACHE_DIR=/tmp/gwai-frontier-formula-g4m2-20260706 timeout 90s target/debug/gw-pn formula --g 4 --markings 2 --n 2 --d 2 --max-descendant 1 --no-glossary
GWAI_GRAPH_CACHE_DIR=/tmp/gwai-frontier-rank3-bundle-d4-20260706 timeout 90s target/debug/gw-pn bundle --n 1 --twists 5,4,0 --g 0 --d 4 --insert 'H*xi^2' --insert H --insert H --weights-base 1,2 --weights-fiber 0,10,30
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

The newest frontier probes were run as one-off commands timed with
`/usr/bin/time -f 'elapsed=%e'` and stdout redirected under `/tmp`, then
promoted into the `extended` harness suite for future repeats.

Cache caveat: the frontier harness defaults to a shared project-local
`target/perf-frontiers/graph-cache/` directory because that is the useful
inner-loop tuning mode.  Use `--graph-cache-mode cold` when auditing execution
frontiers: it gives each case attempt a fresh stable-graph disk cache and
records the cache mode in the CSV/JSONL artifacts.  The table below keeps both
views.  Stable-graph cold generation remains visible, but parallel prefix
generation moved the sampled cold formula rows well below the previous timeout.

## GMP Backend Snapshot

The optional `gmp-rational` feature was measured on 2026-07-07 with release
builds, one fresh CLI process per row, and `--graph-cache-mode off` so the
stable-graph disk cache could not hide cold costs.  These rows are the right
scale for algebra-heavy tuning decisions; debug-build measurements show much
larger speedups but overstate the production effect.

Command shape:

```sh
scripts/run-perf-frontiers.sh --release --graph-cache-mode off --case CASE
scripts/run-perf-frontiers.sh --release --features gmp-rational --graph-cache-mode off --case CASE
```

| mode | probe | default release | GMP release | speedup |
|---|---|---:|---:|---:|
| givental | `P^3`, `g=2`, `d=2`, `tau6(H^3)` | 0.141s | 0.060s | 2.34x |
| product | `P^1 x P^1`, `g=2`, total `d=3`, `tau6(point)` | 0.233s | 0.095s | 2.44x |
| bundle | `P(O+O(2))`, `g=1`, shifted `d=5`, three `tau1(point)` | 0.615s | 0.184s | 3.34x |
| bundle | `P(O(2)+O(1)+O(-3))`, `g=0`, shifted `d=3` | 0.556s | 0.113s | 4.94x |
| twisted | local `P^2`, `O(-3)`, `g=2`, `d=3` | 0.131s | 0.048s | 2.72x |
| twisted | `P^2`, `O(-1)`, factored equivariant, `g=0`, `d=1` | 0.009s | 0.007s | 1.20x |

Read: the GMP backend is a real cross-cutting win, especially in bundle and
twisted rational paths, while very small factored symbolic rows are dominated
by setup rather than rational arithmetic.  Keep the default backend pure Rust;
use `gmp-rational` for frontier hunting and production runs where the LGPL
system dependency is acceptable.

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

## New Frontier Probes

| mode | axis | probe | time | status | read |
|---|---|---|---:|---|---|
| formula | stable graphs | `g=4`, `m=1` | 31.70s | ok | Largest formula row below the cutoff. |
| formula | stable graphs | `g=4`, `m=2` | 90.01s | timeout | Marking jump is a hard graph/canonicalization frontier. |
| formula | stable graphs | `g=5`, `m=0` | 90.01s | timeout | Genus jump also crosses the cutoff. |
| psi | genus | `g=12`, one marking, `psi^34` | 37.87s | ok | Point theory is now visible. |
| psi | genus | `g=13`, one marking, `psi^37` | 90.00s | timeout | Point-theory recursion crosses the cutoff here. |
| product | genus/degree | `P^1 x P^1`, `g=3`, total `d=3`, `tau7(point)` | 28.13s | ok | Below frontier. |
| product | genus/degree | `P^1 x P^1`, `g=3`, total `d=4`, `tau9(point)` | 54.81s | ok | Product row right on the frontier threshold. |
| twisted | degree | local `P^2`, `O(-3)`, `g=2`, `d=5` | 40.49s | ok | Below frontier. |
| twisted | degree | local `P^2`, `O(-3)`, `g=2`, `d=6` | 90.00s | timeout | Twisted degree frontier bracket. |
| twisted | equivariant | `P^2`, `O(-1)`, `g=0`, `d=2`, expanded symbolic | 90.00s | timeout | Symbolic equivariant calibration/graph path crosses the cutoff. |
| twisted | equivariant | same, factored symbolic | 90.00s | timeout | Factored mode also crosses the cutoff. |
| series | markings/psi | `P^2`, `g=1`, `d<=2`, `m<=4`, `psi<=3` | 33.26s | ok | Candidate enumeration is visible. |
| series | markings/psi | `P^2`, `g=1`, `d<=2`, `m<=5`, `psi<=3` | 90.02s | timeout | One extra marking crosses the cutoff. |
| bundle | genus/degree | `P(O+O(2))`, `g=1`, shifted `d=7`, three `tau1(point)` | 46.50s | ok | Below frontier. |
| bundle | genus/degree | `P(O+O(2))`, `g=1`, shifted `d=8`, three `tau1(point)` | 67.48s | ok | Clear F2 bundle frontier after bounded one-variable Birkhoff. |
| bundle | twist rank | `P(O(2)+O(1)+O(-3))`, `g=0`, shifted `d=4`, primary ruling | 54.17s | ok | Improved from 73.23s by bounded one-variable Birkhoff. |

## Frontier Table

| mode | genus | degree | dimension/colors | markings | psi classes | twist/rank factors | current frontier |
|---|---|---|---|---|---|---|---|
| formula/stable graphs | `g=4,m=1` is 31.70s; `g=5,m=0` times out at 90s. | Indirect, only through expansion metadata. | Indirect via expansion size. | `g=4,m=2` times out at 90s. | Affects rendered expansion size. | Twisted expansion adds labels, not the core count. | Larger `(g,n)` canonicalization/orbit handling is again a graph frontier, but above the old sampled row. |
| ordinary Givental `P^n` | Visible but below frontier in sampled single-invariant rows. | Increases q/R truncation and graph coefficient work. | Matrix size and color sums grow with `n+1`; `P^3` sample was 2.99s. | Expands external-leg states and graph contractions. | Raises `z_order`/`r_order`; high single psi still manageable here. | None. | No one-minute ordinary target row found yet; product/bundle/twisted hit the frontier first. |
| twisted | Same stable-graph pressure as Givental. | Local `P^2`, `g=2,d=5` is 40.49s and `d=6` times out. | Color count and relation degree grow with `n`. | Same graph/external-leg pressure. | Raises calibration order. | More negative factors increase hypergeometric products; equivariant `O(-1),d=2` times out in both expanded and factored modes. | Twisted degree and symbolic equivariant calibration are now clear frontiers. |
| product | Same graph pressure per ray. | Total degree gives `d+1` Novikov rays and larger q truncation. | Product colors multiply: `(n+1)(m+1)`. | Same graph/external-leg pressure. | Raises calibration order. | None. | `P^1 x P^1`, `g=3,d=4`, `tau9(point)` is 54.81s: product is back on the frontier at higher genus. |
| bundle | Same graph pressure per ray. | Shifted total degree gives `d+1` rays and bidegree Birkhoff windows. | Size is `(n+1) * rank`; rank 3 is costly. | Same graph/external-leg pressure. | Raises S/R order and prewarm depth. | Positive-z and mixed twist signs drive bidegree Birkhoff/fundamental-S cost. | Rank-3 shifted `d=4` is now 54.17s; F2 `g=1,d=8` is 67.48s. Bundle setup remains a top backend frontier. |
| resolvent/series | Depends on packed graph contraction and candidate count. | Degree ranges multiply candidate coefficients. | Target color count affects packed kernels. | `P^2`, `g=1,d<=2,m<=4,k<=3` is 33.26s; `m<=5` times out. | `max-descendant` directly multiplies candidates. | Twisted resolvent adds calibration cost. | Sparse series candidate enumeration is a clear frontier with one more marking. |
| psi point theory | Recursion/table cost grows with genus and markings. | Not applicable. | Not applicable. | More markings increases partitions. | Powers determine dimension-valid tuples. | Not applicable. | `g=12` one marking is 37.87s; `g=13` times out. Point theory is now worth optimizing if high genus matters. |

## Optimization Frontiers

1. Bundle setup is still the most practical backend frontier in the sampled
   suite, but bounded one-variable Birkhoff reduced rank-3 shifted `d=4` from
   73.23s to 54.17s and F2 `g=1,d=8` from 71.01s to 67.48s.  The next bundle
   pass should focus on per-ray canonical-frame/kernel construction and
   bidegree setup growth.

2. Twisted equivariant symbolic rows are an equally important specialized
   frontier.  `O(-1)` over `P^2`, `g=0,d=2`, two insertions timed out at 90s in
   both expanded and factored modes, while local `P^2`, `g=2,d=6` also timed
   out.

3. Product is back on the frontier only at higher genus: `P^1 x P^1`,
   `g=3,d=4`, `tau9(point)` took 54.81s.  This points back to graph
   contraction/coloring plus ray reconstruction, not the low-genus product
   rows.

4. Stable-graph generation is no longer the first sampled bottleneck, but
   larger formula rows still cross the cutoff: `g=4,m=2` and `g=5,m=0` both
   timed out at 90s.  The next graph-specific pass should target
   canonicalization/orbit handling rather than broad prefix parallelism.

5. Sparse series and point theory now have clear high-end brackets:
   `P^2`, `g=1,d<=2,m<=5,k<=3` timed out after `m<=4` took 33.26s, and
   one-marking point theory goes from 37.87s at `g=12` to timeout at `g=13`.
