#!/usr/bin/env python3
"""Probe gw-pn execution frontiers with curated CLI workloads.

The default `frontier` suite is intentionally finite and debug-build friendly:
it samples each major execution mode and treats roughly-one-minute runs as the
practical frontier.  Results are written under target/perf-frontiers/ so the
same script can be rerun after optimization rounds.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Case:
    suite: str
    group: str
    axis: str
    name: str
    params: str
    cmd: tuple[str, ...]
    notes: str


CASES: tuple[Case, ...] = (
    Case(
        "smoke",
        "psi",
        "genus",
        "psi_g10_one_marking",
        "g=10, one psi class",
        ("psi", "--g", "10", "--powers", "28"),
        "Point-theory recursion/table path; should not be a frontier.",
    ),
    Case(
        "smoke",
        "formula",
        "stable_graphs",
        "formula_g2_m1",
        "g=2, markings=1",
        (
            "formula",
            "--g",
            "2",
            "--markings",
            "1",
            "--n",
            "2",
            "--d",
            "2",
            "--max-descendant",
            "3",
            "--no-glossary",
        ),
        "Small stable-graph rendering baseline.",
    ),
    Case(
        "smoke",
        "givental",
        "degree",
        "p2_g0_d1_three_primary",
        "P^2, g=0, d=1, 3 primary insertions",
        (
            "compute",
            "--n",
            "2",
            "--g",
            "0",
            "--d",
            "1",
            "--insert",
            "H^2",
            "--insert",
            "H^2",
            "--insert",
            "H",
        ),
        "Seed-sized ordinary invariant baseline.",
    ),
    Case(
        "smoke",
        "product",
        "degree",
        "p1xp1_g0_d2_three_points",
        "P^1 x P^1, g=0, total d=2, 3 point insertions",
        (
            "product",
            "--n",
            "1",
            "--m",
            "1",
            "--g",
            "0",
            "--d",
            "2",
            "--insert",
            "H1*H2",
            "--insert",
            "H1*H2",
            "--insert",
            "H1*H2",
        ),
        "Product ray reconstruction baseline.",
    ),
    Case(
        "smoke",
        "bundle",
        "degree",
        "f2_g0_d3_three_primary",
        "P(O+O(2)), g=0, shifted d=3, 3 primary insertions",
        (
            "bundle",
            "--n",
            "1",
            "--twists",
            "0,2",
            "--g",
            "0",
            "--d",
            "3",
            "--insert",
            "H*xi",
            "--insert",
            "H",
            "--insert",
            "H",
        ),
        "Non-Fano positive-z bundle baseline.",
    ),
    Case(
        "frontier",
        "givental",
        "genus",
        "p1_g2_d1_stationary",
        "P^1, g=2, d=1, tau4(H)",
        ("compute", "--n", "1", "--g", "2", "--d", "1", "--insert", "tau4(H)"),
        "Ordinary graph path with one high psi class.",
    ),
    Case(
        "frontier",
        "givental",
        "genus",
        "p1_g3_d1_stationary",
        "P^1, g=3, d=1, tau6(H)",
        ("compute", "--n", "1", "--g", "3", "--d", "1", "--insert", "tau6(H)"),
        "Genus scaling at fixed target and degree.",
    ),
    Case(
        "frontier",
        "givental",
        "dimension",
        "p2_g2_d2_one_descendant",
        "P^2, g=2, d=2, tau6(H^2)",
        ("compute", "--n", "2", "--g", "2", "--d", "2", "--insert", "tau6(H^2)"),
        "Ordinary calibration plus higher-dimensional graph contractions.",
    ),
    Case(
        "frontier",
        "givental",
        "dimension",
        "p3_g2_d2_one_descendant",
        "P^3, g=2, d=2, tau6(H^3)",
        ("compute", "--n", "3", "--g", "2", "--d", "2", "--insert", "tau6(H^3)"),
        "Dimension scaling of ordinary calibration and color sums.",
    ),
    Case(
        "frontier",
        "formula",
        "stable_graphs",
        "formula_g3_m1",
        "g=3, markings=1",
        (
            "formula",
            "--g",
            "3",
            "--markings",
            "1",
            "--n",
            "2",
            "--d",
            "2",
            "--max-descendant",
            "3",
            "--no-glossary",
        ),
        "Stable-graph enumeration/rendering without heavy target algebra.",
    ),
    Case(
        "frontier",
        "formula",
        "stable_graphs",
        "formula_g3_m2",
        "g=3, markings=2",
        (
            "formula",
            "--g",
            "3",
            "--markings",
            "2",
            "--n",
            "2",
            "--d",
            "2",
            "--max-descendant",
            "2",
            "--no-glossary",
        ),
        "Known expensive stable-graph axis: one extra marking.",
    ),
    Case(
        "frontier",
        "resolvent",
        "markings",
        "p2_g0_d1_m3_resolvent",
        "P^2, g=0, d=1, markings=3",
        ("resolvent", "--n", "2", "--g", "0", "--d", "1", "--markings", "3"),
        "Packed external-leg baseline.",
    ),
    Case(
        "frontier",
        "resolvent",
        "markings",
        "p2_g1_d1_m2_resolvent",
        "P^2, g=1, d=1, markings=2",
        ("resolvent", "--n", "2", "--g", "1", "--d", "1", "--markings", "2"),
        "Packed resolvent with nontrivial stable graphs.",
    ),
    Case(
        "frontier",
        "series",
        "markings_psi",
        "p2_g0_dmax2_m4_k2_series",
        "P^2, g=0, d<=2, markings<=4, psi<=2",
        (
            "series",
            "--n",
            "2",
            "--g",
            "0",
            "--d-max",
            "2",
            "--max-markings",
            "4",
            "--max-descendant",
            "2",
        ),
        "Sparse descendant-potential enumeration baseline.",
    ),
    Case(
        "frontier",
        "series",
        "degree_psi",
        "p2_g2_dmax3_m1_k9_degree_series",
        "P^2, g=2, d<=3, markings<=1, psi<=9",
        (
            "degree-series",
            "--n",
            "2",
            "--g",
            "2",
            "--d-max",
            "3",
            "--max-markings",
            "1",
            "--max-descendant",
            "9",
        ),
        "Degree sweep with high descendants.",
    ),
    Case(
        "frontier",
        "twisted",
        "degree",
        "local_p2_g2_d3",
        "P^2, O(-3), g=2, d=3",
        ("twisted", "--n", "2", "--twist", "-3", "--g", "2", "--d", "3"),
        "Local P^2 row; calibration and graph path both active.",
    ),
    Case(
        "frontier",
        "twisted",
        "twist_rank",
        "conifold_g2_d3",
        "P^1, O(-1)+O(-1), g=2, d=3",
        ("twisted", "--n", "1", "--twist", "-1,-1", "--g", "2", "--d", "3"),
        "Rank-2 negative split baseline.",
    ),
    Case(
        "frontier",
        "twisted",
        "twist_rank",
        "rank3_twist_p2_g2_d2",
        "P^2, O(-1)^3, g=2, d=2",
        ("twisted", "--n", "2", "--twist", "-1,-1,-1", "--g", "2", "--d", "2"),
        "Twisted factor count scaling.",
    ),
    Case(
        "frontier",
        "twisted",
        "equivariant",
        "o1_p2_equivariant_expanded",
        "P^2, O(-1), g=0, d=1, equivariant expanded coefficients",
        (
            "twisted",
            "--n",
            "2",
            "--twist",
            "-1",
            "--g",
            "0",
            "--d",
            "1",
            "--insert",
            "tau1(H^2)",
            "--insert",
            "H",
            "--equivariant",
        ),
        "Expanded symbolic coefficient path.",
    ),
    Case(
        "frontier",
        "twisted",
        "equivariant",
        "o1_p2_equivariant_factored",
        "P^2, O(-1), g=0, d=1, equivariant factored coefficients",
        (
            "twisted",
            "--n",
            "2",
            "--twist",
            "-1",
            "--g",
            "0",
            "--d",
            "1",
            "--insert",
            "tau1(H^2)",
            "--insert",
            "H",
            "--equivariant",
            "--factored",
        ),
        "Factored symbolic coefficient path.",
    ),
    Case(
        "frontier",
        "product",
        "genus_degree",
        "p1xp1_g1_d2_one_descendant",
        "P^1 x P^1, g=1, total d=2, tau3(point)",
        (
            "product",
            "--n",
            "1",
            "--m",
            "1",
            "--g",
            "1",
            "--d",
            "2",
            "--insert",
            "tau3(H1*H2)",
        ),
        "Product ray reconstruction with nonzero R contribution.",
    ),
    Case(
        "frontier",
        "product",
        "genus_degree",
        "p1xp1_g2_d3_one_descendant",
        "P^1 x P^1, g=2, total d=3, tau6(point)",
        (
            "product",
            "--n",
            "1",
            "--m",
            "1",
            "--g",
            "2",
            "--d",
            "3",
            "--insert",
            "tau6(H1*H2)",
        ),
        "Higher genus/degree product reconstruction.",
    ),
    Case(
        "frontier",
        "product",
        "dimension",
        "p1xp2_g1_d2_one_descendant",
        "P^1 x P^2, g=1, total d=2, tau3(H1*H2^2)",
        (
            "product",
            "--n",
            "1",
            "--m",
            "2",
            "--g",
            "1",
            "--d",
            "2",
            "--insert",
            "tau3(H1*H2^2)",
        ),
        "Product color-count scaling.",
    ),
    Case(
        "frontier",
        "bundle",
        "genus_degree",
        "f2_g1_d5_three_descendants",
        "P(O+O(2)), g=1, shifted d=5, three tau1(point)",
        (
            "bundle",
            "--n",
            "1",
            "--twists",
            "0,2",
            "--g",
            "1",
            "--d",
            "5",
            "--insert",
            "tau1(H*xi)",
            "--insert",
            "tau1(H*xi)",
            "--insert",
            "tau1(H*xi)",
        ),
        "Slow F2 negative-section acceptance shape.",
    ),
    Case(
        "frontier",
        "bundle",
        "twist_rank",
        "rank3_bundle_g0_d3_primary",
        "P(O(2)+O(1)+O(-3)), g=0, shifted d=3, primary ruling",
        (
            "bundle",
            "--n",
            "1",
            "--twists",
            "2,1,-3",
            "--g",
            "0",
            "--d",
            "3",
            "--insert",
            "H*xi^2",
            "--insert",
            "H",
            "--insert",
            "H",
            "--weights-base",
            "1,2",
            "--weights-fiber",
            "0,10,30",
        ),
        "Current rank-3 bundle frontier case.",
    ),
    Case(
        "extended",
        "formula",
        "stable_graphs",
        "formula_g4_m1",
        "g=4, markings=1",
        (
            "formula",
            "--g",
            "4",
            "--markings",
            "1",
            "--n",
            "2",
            "--d",
            "2",
            "--max-descendant",
            "2",
            "--no-glossary",
        ),
        "Extended stable-graph stress test.",
    ),
    Case(
        "extended",
        "series",
        "markings_psi",
        "p2_g1_dmax2_m3_k3_series",
        "P^2, g=1, d<=2, markings<=3, psi<=3",
        (
            "series",
            "--n",
            "2",
            "--g",
            "1",
            "--d-max",
            "2",
            "--max-markings",
            "3",
            "--max-descendant",
            "3",
        ),
        "Extended sparse potential stress test.",
    ),
)


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        print("error: --repeat must be at least 1", file=sys.stderr)
        return 2
    selected = select_cases(args)
    if args.list:
        for case in selected:
            print(f"{case.suite:8} {case.group:10} {case.axis:14} {case.name}")
        return 0

    baseline_rows = load_baseline(args.baseline) if args.baseline else {}
    binary = resolve_binary(args)
    if not args.no_build:
        build(args.release)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_cache_dir = None
    cold_graph_cache_base = None
    if args.graph_cache_mode == "shared":
        graph_cache_dir = out_dir / "graph-cache"
        graph_cache_dir.mkdir(parents=True, exist_ok=True)
    elif args.graph_cache_mode == "cold":
        cold_graph_cache_base = out_dir / "cold-graph-cache" / stamp
        cold_graph_cache_base.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {case.group}/{case.name}", flush=True)
        row = run_case(
            binary,
            case,
            args.timeout,
            args.frontier_seconds,
            args.repeat,
            args.graph_cache_mode,
            graph_cache_dir,
            cold_graph_cache_base,
        )
        attach_baseline(row, baseline_rows.get(case.name), args.regression_percent)
        rows.append(row)
        status = row["status"]
        elapsed = row["elapsed_s"]
        marker = " frontier" if row["frontier"] else ""
        comparison = format_comparison(row)
        print(f"  {status} {elapsed:.3f}s{marker}{comparison}", flush=True)

    base = out_dir / f"perf-frontiers-{stamp}"
    write_jsonl(base.with_suffix(".jsonl"), rows)
    write_csv(base.with_suffix(".csv"), rows)
    write_markdown(base.with_suffix(".md"), rows, args)
    write_jsonl(out_dir / "latest.jsonl", rows)
    write_csv(out_dir / "latest.csv", rows)
    write_markdown(out_dir / "latest.md", rows, args)
    if args.save_baseline:
        write_csv(Path(args.save_baseline), rows)

    print()
    print(render_markdown_table(rows))
    print()
    print(f"wrote {base.with_suffix('.md')}")
    print(f"wrote {base.with_suffix('.csv')}")
    print(f"wrote {base.with_suffix('.jsonl')}")
    print(f"wrote {out_dir / 'latest.md'}")
    if args.save_baseline:
        print(f"wrote baseline {args.save_baseline}")
    if any(row.get("regression") for row in rows):
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("smoke", "frontier", "extended", "all"),
        default="frontier",
        help="case set to run; frontier includes smoke cases",
    )
    parser.add_argument("--case", action="append", default=[], help="substring filter")
    parser.add_argument("--list", action="store_true", help="list selected cases and exit")
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument("--frontier-seconds", type=float, default=55.0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run each successful case this many times and compare medians",
    )
    parser.add_argument("--output-dir", default="target/perf-frontiers")
    parser.add_argument(
        "--graph-cache-mode",
        choices=("shared", "cold", "off"),
        default="shared",
        help=(
            "stable-graph disk cache mode: shared uses the project-local cache, "
            "cold creates a fresh cache per sample, off disables disk caching"
        ),
    )
    parser.add_argument(
        "--baseline",
        help="CSV from an earlier run; adds timing deltas and regression flags",
    )
    parser.add_argument(
        "--save-baseline",
        nargs="?",
        const="target/perf-frontiers/baseline.csv",
        help="write the current CSV in baseline format, default target/perf-frontiers/baseline.csv",
    )
    parser.add_argument(
        "--regression-percent",
        type=float,
        default=20.0,
        help="mark rows slower than baseline by this percent as regressions",
    )
    parser.add_argument("--binary", help="path to gw-pn binary")
    parser.add_argument("--release", action="store_true", help="benchmark target/release/gw-pn")
    parser.add_argument("--no-build", action="store_true", help="skip cargo build")
    return parser.parse_args()


def select_cases(args: argparse.Namespace) -> list[Case]:
    if args.suite == "smoke":
        suites = {"smoke"}
    elif args.suite == "frontier":
        suites = {"smoke", "frontier"}
    elif args.suite == "extended":
        suites = {"smoke", "frontier", "extended"}
    else:
        suites = {"smoke", "frontier", "extended"}

    cases = [case for case in CASES if case.suite in suites]
    for pattern in args.case:
        needle = pattern.lower()
        cases = [
            case
            for case in cases
            if needle
            in " ".join((case.group, case.axis, case.name, case.params, case.notes)).lower()
        ]
    return cases


def resolve_binary(args: argparse.Namespace) -> Path:
    if args.binary:
        return Path(args.binary)
    profile = "release" if args.release else "debug"
    return Path("target") / profile / "gw-pn"


def build(release: bool) -> None:
    cmd = ["cargo", "build", "--quiet"]
    if release:
        cmd.insert(2, "--release")
    subprocess.run(cmd, check=True)


def run_case(
    binary: Path,
    case: Case,
    timeout: float,
    frontier_seconds: float,
    repeat: int,
    graph_cache_mode: str,
    graph_cache_dir: Path | None,
    cold_graph_cache_base: Path | None,
) -> dict[str, object]:
    cmd = [str(binary), *case.cmd]
    base_env = os.environ.copy()
    base_env.pop("GW_PROFILE", None)
    samples = []
    statuses = []
    graph_cache_dirs = []
    stdout = ""
    stderr = ""
    for attempt in range(repeat):
        env = base_env.copy()
        graph_cache_path = configure_graph_cache(
            env,
            graph_cache_mode,
            graph_cache_dir,
            cold_graph_cache_base,
            case.name,
            attempt,
        )
        if graph_cache_path is not None:
            graph_cache_dirs.append(str(graph_cache_path))
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=env,
            )
            elapsed = time.perf_counter() - started
            status = "ok" if completed.returncode == 0 else f"exit_{completed.returncode}"
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - started
            status = "timeout"
            stdout = as_text(exc.stdout)
            stderr = as_text(exc.stderr)

        samples.append(elapsed)
        statuses.append(status)
        if status != "ok":
            break
        if attempt + 1 < repeat:
            print(f"  sample {attempt + 1}/{repeat}: {elapsed:.3f}s", flush=True)

    elapsed = median(samples)
    status = statuses[-1]

    return {
        "suite": case.suite,
        "group": case.group,
        "axis": case.axis,
        "name": case.name,
        "params": case.params,
        "command": " ".join(cmd),
        "status": status,
        "elapsed_s": round(elapsed, 6),
        "elapsed_min_s": round(min(samples), 6),
        "elapsed_max_s": round(max(samples), 6),
        "elapsed_samples_s": ",".join(f"{sample:.6f}" for sample in samples),
        "repeat": len(samples),
        "graph_cache_mode": graph_cache_mode,
        "graph_cache_dirs": ",".join(graph_cache_dirs),
        "frontier": status == "timeout" or elapsed >= frontier_seconds,
        "stdout_bytes": len(stdout.encode()),
        "stderr_bytes": len(stderr.encode()),
        "stdout_tail": tail(stdout),
        "stderr_tail": tail(stderr),
        "notes": case.notes,
    }


def configure_graph_cache(
    env: dict[str, str],
    mode: str,
    shared_dir: Path | None,
    cold_base: Path | None,
    case_name: str,
    attempt: int,
) -> Path | None:
    if mode == "shared":
        if shared_dir is not None:
            env.setdefault("GWAI_GRAPH_CACHE_DIR", str(shared_dir))
        path = env.get("GWAI_GRAPH_CACHE_DIR")
        return Path(path) if path else None
    if mode == "cold":
        if cold_base is None:
            raise RuntimeError("cold graph-cache mode requires a base directory")
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_name)
        path = Path(
            tempfile.mkdtemp(
                prefix=f"{safe_name}-sample{attempt + 1}-",
                dir=cold_base,
            )
        )
        env.pop("GWAI_DISABLE_GRAPH_CACHE", None)
        env["GWAI_GRAPH_CACHE_DIR"] = str(path)
        return path
    if mode == "off":
        env["GWAI_DISABLE_GRAPH_CACHE"] = "1"
        env.pop("GWAI_GRAPH_CACHE_DIR", None)
        return None
    raise ValueError(f"unknown graph cache mode: {mode}")


def median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def load_baseline(path: str) -> dict[str, dict[str, str]]:
    baseline_path = Path(path)
    if not baseline_path.exists():
        print(f"warning: baseline `{path}` does not exist; comparison disabled", file=sys.stderr)
        return {}
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle) if row.get("name")}


def attach_baseline(
    row: dict[str, object],
    baseline: dict[str, str] | None,
    regression_percent: float,
) -> None:
    row["baseline_s"] = ""
    row["delta_s"] = ""
    row["change_percent"] = ""
    row["speedup"] = ""
    row["regression"] = False
    if not baseline:
        return
    try:
        baseline_s = float(baseline.get("elapsed_s", ""))
    except ValueError:
        return
    current_s = float(row["elapsed_s"])
    if baseline_s <= 0 or not math.isfinite(baseline_s):
        return
    delta_s = current_s - baseline_s
    change_percent = 100.0 * delta_s / baseline_s
    speedup = baseline_s / current_s if current_s > 0 else math.inf
    row["baseline_s"] = round(baseline_s, 6)
    row["delta_s"] = round(delta_s, 6)
    row["change_percent"] = round(change_percent, 3)
    row["speedup"] = round(speedup, 4) if math.isfinite(speedup) else "inf"
    row["regression"] = (
        row["status"] == "ok"
        and baseline.get("status") == "ok"
        and change_percent > regression_percent
    )


def format_comparison(row: dict[str, object]) -> str:
    if row.get("change_percent") == "":
        return ""
    regression = " regression" if row.get("regression") else ""
    return f" ({float(row['change_percent']):+.1f}% vs baseline{regression})"


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def tail(value: str, limit: int = 800) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[-limit:]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite",
        "group",
        "axis",
        "name",
        "params",
        "status",
        "elapsed_s",
        "elapsed_min_s",
        "elapsed_max_s",
        "elapsed_samples_s",
        "repeat",
        "graph_cache_mode",
        "graph_cache_dirs",
        "baseline_s",
        "delta_s",
        "change_percent",
        "speedup",
        "regression",
        "frontier",
        "stdout_bytes",
        "stderr_bytes",
        "command",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_markdown(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GW-Pn Performance Frontier Run",
        "",
        f"- suite: `{args.suite}`",
        f"- timeout: `{args.timeout:.1f}s`",
        f"- frontier threshold: `{args.frontier_seconds:.1f}s`",
        f"- repeat: `{args.repeat}`",
        f"- profile: `{'release' if args.release else 'debug'}`",
        f"- graph cache mode: `{args.graph_cache_mode}`",
        f"- baseline: `{args.baseline or 'none'}`",
        "",
        render_markdown_table(rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def render_markdown_table(rows: list[dict[str, object]]) -> str:
    has_baseline = any(row.get("baseline_s") != "" for row in rows)
    header = "| group | axis | case | params | status | seconds | frontier | notes |"
    divider = "|---|---|---|---|---:|---:|:---:|---|"
    if has_baseline:
        header = (
            "| group | axis | case | params | status | seconds | baseline | change | frontier | notes |"
        )
        divider = "|---|---|---|---|---:|---:|---:|---:|:---:|---|"
    lines = [header, divider]
    for row in rows:
        if has_baseline:
            baseline = ""
            change = ""
            if row.get("baseline_s") != "":
                baseline = f"{float(row['baseline_s']):.3f}"
                change = f"{float(row['change_percent']):+.1f}%"
                if row.get("regression"):
                    change += " regressed"
            lines.append(
                "| {group} | {axis} | `{name}` | {params} | {status} | {elapsed:.3f} | {baseline} | {change} | {frontier} | {notes} |".format(
                    group=md(row["group"]),
                    axis=md(row["axis"]),
                    name=md(row["name"]),
                    params=md(row["params"]),
                    status=md(row["status"]),
                    elapsed=float(row["elapsed_s"]),
                    baseline=baseline,
                    change=change,
                    frontier="yes" if row["frontier"] else "",
                    notes=md(row["notes"]),
                )
            )
        else:
            lines.append(
                "| {group} | {axis} | `{name}` | {params} | {status} | {elapsed:.3f} | {frontier} | {notes} |".format(
                    group=md(row["group"]),
                    axis=md(row["axis"]),
                    name=md(row["name"]),
                    params=md(row["params"]),
                    status=md(row["status"]),
                    elapsed=float(row["elapsed_s"]),
                    frontier="yes" if row["frontier"] else "",
                    notes=md(row["notes"]),
                )
            )
    return "\n".join(lines)


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    sys.exit(main())
