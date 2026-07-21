#!/usr/bin/env python3
"""Run curated mathematical holdouts with durable per-test reporting.

The registry is intentionally separate from Cargo's broad test inventory. It
names curated mathematical holdouts and regression fixtures and records the
provenance or internal-consistency category of each one. The runner audits every
Rust ``#[ignore]`` test against that registry so expensive acceptance coverage
cannot silently vanish from scheduled CI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "scripts" / "acceptance-tests.json"
DEFAULT_ORACLE_MANIFEST = ROOT / "docs" / "oracle-provenance.tsv"
VALID_CADENCES = {"push", "scheduled"}
VALID_CATEGORIES = {
    "adaptive-truncation",
    "combinatorial-reference",
    "cross-backend-equivalence",
    "deformation-equivalence",
    "external-localization-golden",
    "external-reference",
    "independent-localization",
    "recorded-unsourced",
    "virasoro-backend",
}
VALID_TARGETS = {
    "negative-split-completion",
    "negative-split-projective",
    "product-projective",
    "projective-bundle",
    "stable-graphs",
}
ORACLE_HEADER = (
    "id",
    "target_family",
    "evidence",
    "production_exercised",
    "default_ci",
    "production_path",
    "oracle_path",
    "cargo_target",
    "cargo_test",
    "source_locator",
    "source",
    "shared_code",
    "claim",
)


@dataclass(frozen=True)
class Case:
    id: str
    test: str
    cargo_target: str
    cadence: str
    category: str
    target: str
    oracle: str
    ignored: bool
    description: str
    timeout_seconds: float | None = None

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Case":
        required = {
            "id",
            "test",
            "cargo_target",
            "cadence",
            "category",
            "target",
            "oracle",
            "ignored",
            "description",
        }
        optional = {"timeout_seconds"}
        missing = required - value.keys()
        extra = value.keys() - required - optional
        if missing or extra:
            raise ValueError(
                f"invalid case fields for {value.get('id', '<unknown>')}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        normalized = dict(value)
        if "timeout_seconds" in normalized:
            timeout_seconds = normalized["timeout_seconds"]
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0
            ):
                raise ValueError(
                    "case timeout_seconds must be a positive finite number for "
                    f"{value.get('id', '<unknown>')}"
                )
            normalized["timeout_seconds"] = float(timeout_seconds)
        case = cls(**normalized)
        if re.fullmatch(r"[a-z0-9_]+", case.id) is None:
            raise ValueError(f"case id is not a safe stable slug: {case.id}")
        if case.cadence not in VALID_CADENCES:
            raise ValueError(f"invalid cadence for {case.id}: {case.cadence}")
        if case.category not in VALID_CATEGORIES:
            raise ValueError(f"invalid category for {case.id}: {case.category}")
        if case.target not in VALID_TARGETS:
            raise ValueError(f"invalid target for {case.id}: {case.target}")
        if case.cargo_target != "lib" and re.fullmatch(
            r"(?:test|bin|example|bench):[^:]+", case.cargo_target
        ) is None:
            raise ValueError(f"invalid cargo_target for {case.id}: {case.cargo_target}")
        if not all(
            isinstance(field, str) and field
            for field in (
                case.id,
                case.test,
                case.category,
                case.target,
                case.oracle,
                case.description,
            )
        ):
            raise ValueError(f"case {case.id} has an empty or non-string field")
        if not isinstance(case.ignored, bool):
            raise ValueError(f"case {case.id} has non-boolean ignored field")
        return case


@dataclass(frozen=True)
class CargoHarness:
    """One testable Cargo target, named in the registry's stable syntax."""

    id: str
    source: Path


@dataclass(frozen=True)
class OracleIdentity:
    id: str
    default_ci: str
    cargo_target: str
    cargo_test: str
    source_locator: str


@dataclass(frozen=True)
class RegistryDrift:
    missing_registered: tuple[tuple[str, str], ...]
    unexpected_ignored: tuple[tuple[str, str], ...]
    no_longer_ignored: tuple[tuple[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--oracle-manifest",
        type=Path,
        default=DEFAULT_ORACLE_MANIFEST,
        help="independent-oracle provenance inventory audited with Cargo discovery",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run the runner's target-identity regression checks and exit",
    )
    parser.add_argument(
        "--suite",
        choices=("push", "scheduled"),
        default="push",
        help="push selects default-suite holdouts; scheduled includes every registered holdout",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="run one case id (repeatable); overrides --suite selection",
    )
    parser.add_argument("--list", action="store_true", help="list selected cases and exit")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="check registry/test inventory agreement without running holdouts",
    )
    parser.add_argument(
        "--skip-registry-audit",
        action="store_true",
        help="skip Cargo test discovery (for local iteration only; do not use in CI)",
    )
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument("--features", help="comma- or space-separated Cargo features")
    feature_group.add_argument("--all-features", action="store_true")
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument("--release", action="store_true")
    profile_group.add_argument(
        "--profile",
        choices=("dev", "acceptance", "release"),
        default="dev",
        help="Cargo profile; acceptance is optimized without release-profile LTO",
    )
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--cargo", default=os.environ.get("CARGO", "cargo"))
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=1_800.0,
        help="per-test timeout; the runner continues with later cases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="report directory (default: target/acceptance-tests/<UTC timestamp>)",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> tuple[int, list[Case]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"schema_version", "cases"}:
        raise ValueError("manifest root must contain only schema_version and cases")
    if data["schema_version"] != 1:
        raise ValueError(f"unsupported manifest schema: {data['schema_version']}")
    cases = [Case.from_json(value) for value in data["cases"]]
    ids = [case.id for case in cases]
    tests = [(case.cargo_target, case.test) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate case ids")
    if len(tests) != len(set(tests)):
        raise ValueError("manifest contains duplicate cargo-target/test pairs")
    return data["schema_version"], cases


def load_oracle_identities(path: Path) -> list[OracleIdentity]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("oracle provenance inventory is empty")
    header = tuple(lines[0].split("\t"))
    if header != ORACLE_HEADER:
        raise ValueError(
            "oracle provenance header differs from the audited schema: "
            f"expected={ORACLE_HEADER}, actual={header}"
        )
    rows: list[OracleIdentity] = []
    ids: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != len(ORACLE_HEADER):
            raise ValueError(
                f"oracle provenance line {line_number} has {len(fields)} fields; "
                f"expected {len(ORACLE_HEADER)}"
            )
        values = dict(zip(ORACLE_HEADER, fields, strict=True))
        if any(not value.strip() for value in fields):
            raise ValueError(f"oracle provenance line {line_number} has an empty field")
        row_id = values["id"]
        if row_id in ids:
            raise ValueError(f"duplicate oracle provenance id: {row_id}")
        ids.add(row_id)
        if values["default_ci"] not in {"yes", "no"}:
            raise ValueError(
                f"oracle {row_id} has invalid default_ci={values['default_ci']}"
            )
        rows.append(
            OracleIdentity(
                id=row_id,
                default_ci=values["default_ci"],
                cargo_target=values["cargo_target"],
                cargo_test=values["cargo_test"],
                source_locator=values["source_locator"],
            )
        )
    return rows


def cargo_common(args: argparse.Namespace) -> list[str]:
    command = [args.cargo, "test", "--locked"]
    profile = selected_profile(args)
    if profile == "release":
        command.append("--release")
    elif profile != "dev":
        command.extend(("--profile", profile))
    if args.all_features:
        command.append("--all-features")
    elif args.features:
        command.extend(("--features", args.features))
    return command


def selected_profile(args: argparse.Namespace) -> str:
    return "release" if args.release else args.profile


def cargo_target_args(cargo_target: str) -> list[str]:
    if cargo_target == "lib":
        return ["--lib"]
    prefix, separator, name = cargo_target.partition(":")
    option = {
        "test": "--test",
        "bin": "--bin",
        "example": "--example",
        "bench": "--bench",
    }.get(prefix)
    if separator != ":" or option is None or not name:
        raise ValueError(f"invalid Cargo target identity: {cargo_target}")
    return [option, name]


def metadata_target_id(target: dict[str, Any]) -> str | None:
    if not target.get("test", False):
        return None
    kinds = set(target.get("kind", []))
    if kinds & {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"}:
        return "lib"
    for kind in ("test", "bin", "example", "bench"):
        if kind in kinds:
            return f"{kind}:{target['name']}"
    raise ValueError(
        f"unsupported testable Cargo target {target.get('name')}: kinds={sorted(kinds)}"
    )


def discover_cargo_harnesses(args: argparse.Namespace) -> tuple[list[CargoHarness], float]:
    command = [
        args.cargo,
        "metadata",
        "--locked",
        "--no-deps",
        "--format-version=1",
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Cargo metadata failed with exit code {result.returncode}")
    metadata = json.loads(result.stdout)
    manifest = (ROOT / "Cargo.toml").resolve()
    package = next(
        (
            candidate
            for candidate in metadata["packages"]
            if Path(candidate["manifest_path"]).resolve() == manifest
        ),
        None,
    )
    if package is None:
        raise RuntimeError(f"Cargo metadata did not contain package {manifest}")
    harnesses: list[CargoHarness] = []
    ids: set[str] = set()
    for target in package["targets"]:
        target_id = metadata_target_id(target)
        if target_id is None:
            continue
        if target_id in ids:
            raise RuntimeError(f"Cargo metadata contains duplicate harness identity {target_id}")
        ids.add(target_id)
        harnesses.append(CargoHarness(target_id, Path(target["src_path"])))
    if not harnesses:
        raise RuntimeError("Cargo metadata contained no testable harnesses")
    return sorted(harnesses, key=lambda harness: harness.id), elapsed


def parse_test_list(output: str) -> set[str]:
    tests = set()
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.endswith(": test"):
            tests.add(stripped.removesuffix(": test"))
    return tests


def run_discovery(
    args: argparse.Namespace,
    ignored_only: bool,
    cargo_target: str,
) -> tuple[set[str], float]:
    command = cargo_common(args) + cargo_target_args(cargo_target)
    command.append("--")
    if ignored_only:
        command.append("--ignored")
    command.append("--list")
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"Cargo test discovery failed with exit code {result.returncode}")
    return parse_test_list(result.stdout), elapsed


def registry_drift(
    cases: list[Case],
    all_tests: dict[str, set[str]],
    ignored_tests: dict[str, set[str]],
) -> RegistryDrift:
    all_pairs = {
        (cargo_target, test)
        for cargo_target, tests in all_tests.items()
        for test in tests
    }
    ignored_pairs = {
        (cargo_target, test)
        for cargo_target, tests in ignored_tests.items()
        for test in tests
    }
    registered_pairs = {(case.cargo_target, case.test) for case in cases}
    registered_ignored = {
        (case.cargo_target, case.test) for case in cases if case.ignored
    }
    return RegistryDrift(
        missing_registered=tuple(sorted(registered_pairs - all_pairs)),
        unexpected_ignored=tuple(sorted(ignored_pairs - registered_ignored)),
        no_longer_ignored=tuple(sorted(registered_ignored - ignored_pairs)),
    )


def validate_source_locator(row: OracleIdentity) -> None:
    path_text, separator, member = row.source_locator.partition("::")
    relative = Path(path_text)
    if relative.is_absolute():
        raise ValueError(f"oracle {row.id} has an absolute source locator")
    source_path = (ROOT / relative).resolve()
    try:
        source_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"oracle {row.id} source escapes the package root") from error
    if not source_path.is_file():
        raise ValueError(f"oracle {row.id} source does not exist: {path_text}")
    if source_path.suffix == ".rs":
        if separator != "::" or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", member) is None:
            raise ValueError(
                f"oracle {row.id} Rust source must name one member as path.rs::member"
            )
        source = source_path.read_text(encoding="utf-8")
        if re.search(rf"\bfn\s+{re.escape(member)}\b", source) is None:
            raise ValueError(
                f"oracle {row.id} source member {member} is missing from {path_text}"
            )
        cargo_member = row.cargo_test.rsplit("::", maxsplit=1)[-1]
        is_builtin_helper = (
            row.cargo_target == "lib"
            and row.cargo_test == "testsuite::tests::builtin_suite_passes"
            and path_text == "src/testsuite.rs"
        )
        if member != cargo_member and not is_builtin_helper:
            raise ValueError(
                f"oracle {row.id} source member {member} does not identify its "
                f"Cargo test {cargo_member}"
            )
        if is_builtin_helper:
            table = re.search(
                r"\bpub\s+fn\s+run_builtin_tests\s*\(\s*\).*?"
                r"\blet\s+tests\s*:.*?=\s*&\[(?P<members>.*?)\n\s*\];",
                source,
                re.DOTALL,
            )
            aggregate_test = re.search(
                r"#\[test\]\s*\n\s*fn\s+builtin_suite_passes\s*\(\s*\)\s*"
                r"\{(?P<body>.*?)\n\s*\}",
                source,
                re.DOTALL,
            )
            member_is_registered = table is not None and re.search(
                rf"\b{re.escape(member)}\b", table.group("members")
            )
            aggregate_is_executed = aggregate_test is not None and re.search(
                r"\brun_builtin_tests\s*\(", aggregate_test.group("body")
            )
            if not member_is_registered or not aggregate_is_executed:
                raise ValueError(
                    f"oracle {row.id} helper {member} is not linked through "
                    "run_builtin_tests to testsuite::tests::builtin_suite_passes"
                )
    elif separator:
        raise ValueError(f"oracle {row.id} non-Rust source locator cannot name a member")


def audit_oracle_identities(
    rows: list[OracleIdentity],
    all_tests: dict[str, set[str]],
    ignored_tests: dict[str, set[str]],
) -> None:
    failures: list[str] = []
    for row in rows:
        try:
            validate_source_locator(row)
        except ValueError as error:
            failures.append(str(error))
        tests = all_tests.get(row.cargo_target)
        if tests is None:
            failures.append(
                f"oracle {row.id} names unknown Cargo target {row.cargo_target}"
            )
            continue
        if row.cargo_test not in tests:
            failures.append(
                f"oracle {row.id} Cargo test is missing from {row.cargo_target}: "
                f"{row.cargo_test}"
            )
            continue
        is_ignored = row.cargo_test in ignored_tests[row.cargo_target]
        expected_ignored = row.default_ci == "no"
        if is_ignored != expected_ignored:
            expectation = "ignored" if expected_ignored else "non-ignored"
            failures.append(
                f"oracle {row.id} declares default_ci={row.default_ci}, but "
                f"{row.cargo_target}::{row.cargo_test} is not {expectation}"
            )
    if failures:
        print("oracle provenance identity audit failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise RuntimeError("oracle provenance audit failed")


def audit_registry(
    args: argparse.Namespace,
    cases: list[Case],
    oracle_rows: list[OracleIdentity],
) -> float:
    harnesses, metadata_seconds = discover_cargo_harnesses(args)
    all_tests: dict[str, set[str]] = {}
    ignored_tests: dict[str, set[str]] = {}
    discovery_seconds = 0.0
    for harness in harnesses:
        tests, seconds = run_discovery(args, ignored_only=False, cargo_target=harness.id)
        ignored, ignored_seconds = run_discovery(
            args, ignored_only=True, cargo_target=harness.id
        )
        all_tests[harness.id] = tests
        ignored_tests[harness.id] = ignored
        discovery_seconds += seconds + ignored_seconds
    drift = registry_drift(cases, all_tests, ignored_tests)
    if any(
        (
            drift.missing_registered,
            drift.unexpected_ignored,
            drift.no_longer_ignored,
        )
    ):
        all_pairs = {
            (cargo_target, test)
            for cargo_target, tests in all_tests.items()
            for test in tests
        }
        if drift.missing_registered:
            print("registered tests missing from their declared Cargo target:", file=sys.stderr)
            for cargo_target, test in drift.missing_registered:
                elsewhere = (
                    " (same name found in another target)"
                    if any(
                        other_test == test and other_target != cargo_target
                        for other_target, other_test in all_pairs
                    )
                    else ""
                )
                print(f"  {cargo_target}: {test}{elsewhere}", file=sys.stderr)
        if drift.unexpected_ignored:
            print("ignored tests missing from acceptance registry:", file=sys.stderr)
            for cargo_target, test in drift.unexpected_ignored:
                print(f"  {cargo_target}: {test}", file=sys.stderr)
        if drift.no_longer_ignored:
            print("registry entries marked ignored but no longer ignored:", file=sys.stderr)
            for cargo_target, test in drift.no_longer_ignored:
                print(f"  {cargo_target}: {test}", file=sys.stderr)
        raise RuntimeError("acceptance registry audit failed")
    audit_oracle_identities(oracle_rows, all_tests, ignored_tests)
    ignored_count = sum(len(tests) for tests in ignored_tests.values())
    print(
        f"registry audit: {len(cases)} curated cases, "
        f"{ignored_count} ignored tests across {len(harnesses)} Cargo harnesses, "
        f"{len(oracle_rows)} oracle identities, no drift"
    )
    return metadata_seconds + discovery_seconds


def run_self_check() -> None:
    case = Case(
        id="shadowed_test",
        test="duplicate_name",
        cargo_target="test:declared",
        cadence="scheduled",
        category="external-reference",
        target="stable-graphs",
        oracle="self-check",
        ignored=True,
        description="self-check",
    )
    drift = registry_drift(
        [case],
        {
            "lib": {"duplicate_name"},
            "test:declared": {"duplicate_name"},
        },
        {
            "lib": {"duplicate_name"},
            "test:declared": set(),
        },
    )
    assert drift.missing_registered == ()
    assert drift.unexpected_ignored == (("lib", "duplicate_name"),)
    assert drift.no_longer_ignored == (("test:declared", "duplicate_name"),)
    assert cargo_target_args("lib") == ["--lib"]
    assert cargo_target_args("test:verified_examples") == [
        "--test",
        "verified_examples",
    ]
    print("acceptance runner self-check: target-qualified drift detection passed")


def select_cases(args: argparse.Namespace, cases: list[Case]) -> list[Case]:
    if args.case_ids:
        by_id = {case.id: case for case in cases}
        unknown = sorted(set(args.case_ids) - by_id.keys())
        if unknown:
            raise ValueError(f"unknown case ids: {', '.join(unknown)}")
        return [by_id[case_id] for case_id in args.case_ids]
    if args.suite == "push":
        return [case for case in cases if case.cadence == "push"]
    return list(cases)


def target_args(case: Case) -> list[str]:
    return cargo_target_args(case.cargo_target)


def case_command(args: argparse.Namespace, case: Case) -> list[str]:
    return cargo_common(args) + target_args(case) + [
        case.test,
        "--",
        "--exact",
        "--include-ignored",
        "--nocapture",
    ]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "target" / "acceptance-tests" / stamp


def backend_label(args: argparse.Namespace) -> str:
    if args.all_features:
        return "all-features"
    if args.features:
        return args.features.replace(" ", ",")
    return "default"


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_summary(
    path: Path,
    suite: str,
    profile: str,
    backend: str,
    audit_seconds: float,
    build_seconds: float,
    results: list[dict[str, Any]],
) -> None:
    passed = sum(result["status"] == "passed" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    timed_out = sum(result["status"] == "timed-out" for result in results)
    skipped = sum(result["status"] == "not-run" for result in results)
    total_seconds = sum(result["duration_seconds"] for result in results)
    lines = [
        "# Acceptance-test report",
        "",
        f"- Suite: `{suite}`",
        f"- Profile: `{profile}`",
        f"- Feature set: `{backend}`",
        f"- Registry audit: `{audit_seconds:.3f}s`",
        f"- Build: `{build_seconds:.3f}s`",
        f"- Cases: {passed} passed, {failed} failed, {timed_out} timed out, {skipped} not run",
        f"- Sum of individual case times: `{total_seconds:.3f}s`",
        "",
        "| Status | Seconds | Timeout | Category | Target | Case | Oracle |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| {status} | {seconds:.3f} | {timeout:g}s | {category} | {target} | `{id}` | {oracle} |".format(
                status=result["status"],
                seconds=result["duration_seconds"],
                timeout=result["timeout_seconds"],
                category=markdown_escape(result["category"]),
                target=markdown_escape(result["target"]),
                id=result["id"],
                oracle=markdown_escape(result["oracle"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def decode_timeout_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def run_case_process(
    command: list[str], env: dict[str, str], timeout_seconds: float
) -> tuple[int | None, str, bool]:
    """Run one case and terminate its whole process group on timeout."""
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        return process.returncode, output, False
    except subprocess.TimeoutExpired as error:
        partial = decode_timeout_output(error.stdout)
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            remainder, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            remainder, _ = process.communicate()
        if remainder.startswith(partial):
            partial = remainder
        elif remainder:
            partial += remainder
        return None, partial, True


def main() -> int:
    args = parse_args()
    if args.self_check:
        run_self_check()
        return 0
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.audit_only and args.skip_registry_audit:
        raise ValueError("--audit-only cannot be combined with --skip-registry-audit")
    schema_version, cases = load_cases(args.manifest)
    oracle_rows = load_oracle_identities(args.oracle_manifest)
    selected = select_cases(args, cases)

    if args.list:
        for case in selected:
            ignored = "ignored" if case.ignored else "normal"
            print(
                f"{case.id}\t{case.cadence}\t{ignored}\t{case.category}\t"
                f"{case.target}\t{case.test}"
            )
        return 0

    if args.audit_only:
        if not args.skip_registry_audit:
            audit_registry(args, cases, oracle_rows)
        return 0

    output_dir = (args.output_dir or default_output_dir()).resolve()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.md"
    profile = selected_profile(args)
    backend = backend_label(args)
    suite_label = "selected" if args.case_ids else args.suite

    build_seconds = 0.0
    if not args.no_build:
        build_command = cargo_common(args) + ["--all-targets", "--no-run"]
        print(f"build: {' '.join(build_command)}", flush=True)
        started = time.monotonic()
        build = subprocess.run(build_command, cwd=ROOT, check=False)
        build_seconds = time.monotonic() - started
        if build.returncode != 0:
            print("acceptance-test build failed", file=sys.stderr)
            return build.returncode

    audit_seconds = 0.0
    if not args.skip_registry_audit:
        audit_seconds = audit_registry(args, cases, oracle_rows)

    run_record = {
        "record_type": "run",
        "schema_version": schema_version,
        "started_at": utc_now(),
        "suite": suite_label,
        "selected_case_ids": [case.id for case in selected],
        "profile": profile,
        "feature_set": backend,
        "default_timeout_seconds": args.timeout_seconds,
        "registry_audit_seconds": round(audit_seconds, 6),
    }
    jsonl_path.write_text(json.dumps(run_record, sort_keys=True) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["GWAI_GRAPH_CACHE_DIR"] = str(output_dir / "graph-cache")
    env.setdefault("RUST_TEST_THREADS", "1")
    results: list[dict[str, Any]] = []
    failed = False
    with jsonl_path.open("a", encoding="utf-8", buffering=1) as jsonl:
        for index, case in enumerate(selected, start=1):
            command = case_command(args, case)
            timeout_seconds = (
                case.timeout_seconds
                if case.timeout_seconds is not None
                else args.timeout_seconds
            )
            print(
                f"[{index}/{len(selected)}] {case.id} "
                f"({case.category}, {case.target}, timeout={timeout_seconds:g}s)",
                flush=True,
            )
            started_at = utc_now()
            started = time.monotonic()
            status = "failed"
            exit_code: int | None = None
            output = ""
            exit_code, output, timed_out = run_case_process(
                command, env, timeout_seconds
            )
            if timed_out:
                status = "timed-out"
            else:
                status = "passed" if exit_code == 0 else "failed"
            duration = time.monotonic() - started
            log_relative = Path("logs") / f"{case.id}.log"
            (output_dir / log_relative).write_text(output, encoding="utf-8")
            result = {
                "record_type": "case",
                "id": case.id,
                "test": case.test,
                "cargo_target": case.cargo_target,
                "cadence": case.cadence,
                "category": case.category,
                "target": case.target,
                "oracle": case.oracle,
                "ignored": case.ignored,
                "description": case.description,
                "status": status,
                "duration_seconds": round(duration, 6),
                "started_at": started_at,
                "exit_code": exit_code,
                "timeout_seconds": timeout_seconds,
                "command": command,
                "log": str(log_relative),
            }
            results.append(result)
            jsonl.write(json.dumps(result, sort_keys=True) + "\n")
            print(f"  {status}: {duration:.3f}s", flush=True)
            if status != "passed":
                failed = True
                tail = "\n".join(output.splitlines()[-40:])
                if tail:
                    print(tail, file=sys.stderr)
                if args.fail_fast:
                    break

        completed_ids = {result["id"] for result in results}
        for case in selected:
            if case.id not in completed_ids:
                result = {
                    "record_type": "case",
                    "id": case.id,
                    "test": case.test,
                    "cargo_target": case.cargo_target,
                    "cadence": case.cadence,
                    "category": case.category,
                    "target": case.target,
                    "oracle": case.oracle,
                    "ignored": case.ignored,
                    "description": case.description,
                    "status": "not-run",
                    "duration_seconds": 0.0,
                    "started_at": None,
                    "exit_code": None,
                    "timeout_seconds": (
                        case.timeout_seconds
                        if case.timeout_seconds is not None
                        else args.timeout_seconds
                    ),
                    "command": case_command(args, case),
                    "log": None,
                }
                results.append(result)
                jsonl.write(json.dumps(result, sort_keys=True) + "\n")

        summary_record = {
            "record_type": "summary",
            "finished_at": utc_now(),
            "passed": sum(result["status"] == "passed" for result in results),
            "failed": sum(result["status"] == "failed" for result in results),
            "timed_out": sum(result["status"] == "timed-out" for result in results),
            "not_run": sum(result["status"] == "not-run" for result in results),
            "case_seconds": round(sum(result["duration_seconds"] for result in results), 6),
            "build_seconds": round(build_seconds, 6),
            "registry_audit_seconds": round(audit_seconds, 6),
        }
        jsonl.write(json.dumps(summary_record, sort_keys=True) + "\n")

    write_summary(
        summary_path,
        suite_label,
        profile,
        backend,
        audit_seconds,
        build_seconds,
        results,
    )
    print(f"JSONL: {jsonl_path}")
    print(f"Markdown: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"acceptance runner error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
