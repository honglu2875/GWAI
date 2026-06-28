//! End-to-end regression tests that pin the documented "Verified Examples" from
//! `README.md` to the public library API.
//!
//! These guard the twisted (negative split-bundle), ordinary Givental, and
//! Witten-Kontsevich paths against silent numerical drift. Each value here is a
//! README-documented output; if one of these changes, the README is wrong or a
//! computation path regressed.

use gw_pn::algebra::{RatFun, Rational};
use gw_pn::geometry::CohomologyClass;
use gw_pn::tautological::{TautologicalOracle, WittenKontsevich};
use gw_pn::twisted::{compute_negative_split_twisted, TwistedInvariantRequest};
use gw_pn::{compute, tau, Insertion, InvariantRequest};

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
