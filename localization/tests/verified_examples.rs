//! End-to-end regression tests that pin the documented "Verified Examples" and
//! source-backed validation tables to the public library API.
//!
//! These guard the twisted (negative split-bundle), ordinary Givental, and
//! Witten-Kontsevich paths against silent numerical drift. Each value here is a
//! documented or independently published output; if one changes, either the
//! corresponding reference/convention or a computation path regressed.

use gw_pn::algebra::{RatFun, Rational};
use gw_pn::geometry::CohomologyClass;
use gw_pn::tautological::{TautologicalOracle, WittenKontsevich};
use gw_pn::twisted::{compute_negative_split_twisted, TwistedInvariantRequest};
use gw_pn::{compute, tau, Insertion, InvariantRequest};
use std::process::Command;

/// Compute a non-equivariant negative split-bundle twisted invariant and return
/// it as a scalar rational.
///
/// `degrees` are the positive magnitudes: `[1]` is `O(-1)`, `[3]` is `O(-3)`,
/// `[1, 1]` is `O(-1) + O(-1)`. The public CLI spells these as `--twist -1`,
/// `--twist -3`, `--twist -1,-1`.
fn twisted_value(
    n: usize,
    degrees: Vec<usize>,
    genus: usize,
    degree: usize,
    insertions: Vec<Insertion>,
) -> Option<Rational> {
    let req = TwistedInvariantRequest::new(n, degrees, genus, degree, insertions)
        .expect("valid twisted request");
    compute_negative_split_twisted(&req)
        .expect("twisted computation succeeds")
        .value
        .as_rational()
}

// --- twisted: O(-1) -> P^2, genus 2, degree 2 ---

#[test]
fn twisted_o1_p2_g2_d2_tau4_h() {
    let value = twisted_value(
        2,
        vec![1],
        2,
        2,
        vec![tau(4, CohomologyClass::h_power(2, 1))],
    );
    assert_eq!(value, Some(Rational::new(-1, 480)));
}

#[test]
fn twisted_o1_p2_g2_d2_tau5_one() {
    let value = twisted_value(2, vec![1], 2, 2, vec![tau(5, CohomologyClass::one(2))]);
    assert_eq!(value, Some(Rational::zero()));
}

#[test]
fn twisted_o1_p2_g2_d2_tau3_h2() {
    let value = twisted_value(
        2,
        vec![1],
        2,
        2,
        vec![tau(3, CohomologyClass::h_power(2, 2))],
    );
    assert_eq!(value, Some(Rational::new(-7, 480)));
}

// --- twisted: local P^2 = O(-3) -> P^2, genus 2, degree 3, no insertions ---

#[test]
fn twisted_local_p2_g2_d3() {
    let value = twisted_value(2, vec![3], 2, 3, vec![]);
    assert_eq!(value, Some(Rational::new(3, 20)));
}

#[test]
fn twisted_local_p2_matches_published_higher_genus_grid() {
    // Coates-Iritani, "Gromov-Witten Invariants of Local P^2 and Modular
    // Forms", Appendix C, Tables 2 and 3 (arXiv:1804.03292).
    let cases = [
        (2, 1, Rational::new(1, 80)),
        (2, 2, Rational::zero()),
        (2, 3, Rational::new(3, 20)),
        (2, 4, Rational::new(-514, 5)),
        (2, 5, Rational::new(43_497, 8)),
        (3, 1, Rational::new(1, 2_016)),
        (3, 2, Rational::new(1, 336)),
        (3, 3, Rational::new(1, 56)),
    ];

    for (genus, degree, expected) in cases {
        assert_eq!(
            twisted_value(2, vec![3], genus, degree, vec![]),
            Some(expected),
            "local P2 g={genus}, d={degree}"
        );
    }
}

// --- twisted: resolved conifold O(-1) + O(-1) -> P^1, genus 2, degree 3 ---

#[test]
fn twisted_conifold_p1_g2_d3() {
    let value = twisted_value(1, vec![1, 1], 2, 3, vec![]);
    assert_eq!(value, Some(Rational::new(1, 80)));
}

// --- ordinary P^2: line through two points, genus 0, degree 1 = 1 ---

#[test]
fn ordinary_p2_line_through_two_points() {
    let insertions = vec![
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 2)),
        tau(0, CohomologyClass::h_power(2, 1)),
    ];
    let value = compute(InvariantRequest::new(2, 0, 1, insertions))
        .expect("ordinary compute succeeds")
        .value;
    assert_eq!(value, RatFun::one());
}

// --- Witten-Kontsevich psi: <tau_4>_2 = 1/1152 ---

#[test]
fn psi_genus2_power4() {
    let value = WittenKontsevich::new().psi_integral(2, &[4]);
    assert_eq!(value, Rational::new(1, 1152));
}

#[test]
fn fixed_degree_scan_honors_include_zero() {
    let args = [
        "degree-series",
        "--n",
        "1",
        "--g",
        "0",
        "--d-min",
        "2",
        "--d-max",
        "2",
        "--insert",
        "H",
        "--insert",
        "H",
        "--insert",
        "H",
        "--equivariant",
    ];
    let sparse = Command::new(env!("CARGO_BIN_EXE_gw-pn"))
        .args(args)
        .output()
        .expect("degree-series command runs");
    assert!(sparse.status.success());
    assert!(String::from_utf8_lossy(&sparse.stdout).trim().is_empty());

    let dense = Command::new(env!("CARGO_BIN_EXE_gw-pn"))
        .args(args)
        .arg("--include-zero")
        .output()
        .expect("degree-series --include-zero command runs");
    assert!(dense.status.success());
    assert!(String::from_utf8_lossy(&dense.stdout).contains("q^2 [tau0(H) tau0(H) tau0(H)] = 0"));
}

#[test]
fn unwritable_diagnostics_directory_does_not_fail_computation() {
    let missing_tmp = std::env::temp_dir().join(format!(
        "gw-pn-missing-diagnostics-parent-{}",
        std::process::id()
    ));
    let output = Command::new(env!("CARGO_BIN_EXE_gw-pn"))
        .args([
            "series",
            "--n",
            "1",
            "--g",
            "1",
            "--d-max",
            "0",
            "--max-markings",
            "0",
        ])
        .env("TMPDIR", &missing_tmp)
        .output()
        .expect("series command runs");

    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("failed to write diagnostics"), "{stderr}");
    assert!(stderr.contains("skipped q^0"), "{stderr}");
}
