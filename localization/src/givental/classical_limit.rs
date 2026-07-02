//! Classical (q=0) limit diagonal R-matrix coefficients and the small
//! numeric helpers (exp series, Bernoulli numbers, binomials) they use.

use crate::algebra::{lambda, RatFun, Rational};

pub(crate) fn classical_limit_diagonal_coefficients(n: usize, z_order: usize) -> Vec<Vec<RatFun>> {
    (0..=n)
        .map(|branch| classical_limit_diagonal_coefficients_for_branch(n, branch, z_order))
        .collect()
}

pub(crate) fn classical_limit_diagonal_coefficients_at_lambda_weights(
    n: usize,
    z_order: usize,
    weights: &[Rational],
) -> Vec<Vec<RatFun>> {
    (0..=n)
        .map(|branch| {
            classical_limit_diagonal_coefficients_for_branch_at_lambda_weights(
                n, branch, z_order, weights,
            )
        })
        .collect()
}

pub(crate) fn classical_limit_diagonal_coefficients_for_branch(
    n: usize,
    branch: usize,
    z_order: usize,
) -> Vec<RatFun> {
    let differences = (0..=n)
        .filter(|&other| other != branch)
        .map(|other| &lambda(other) - &lambda(branch))
        .collect::<Vec<_>>();
    classical_r_asymptotics_for_point(&differences, z_order)
}

pub(crate) fn classical_limit_diagonal_coefficients_for_branch_at_lambda_weights(
    n: usize,
    branch: usize,
    z_order: usize,
    weights: &[Rational],
) -> Vec<RatFun> {
    let differences = (0..=n)
        .filter(|&other| other != branch)
        .map(|other| RatFun::from_rational(weights[other].clone() - weights[branch].clone()))
        .collect::<Vec<_>>();
    classical_r_asymptotics_for_point(&differences, z_order)
}

/// Diagonal `R`-matrix asymptotics at one semisimple point, from the pairwise
/// eigenvalue differences at that point.
///
/// This is the Gamma-function/Bernoulli expansion
/// `exp(sum_r B_{2r}/(2r(2r-1)) sum_w w^{-(2r-1)} z^{2r-1})`, where `w` runs
/// over the supplied differences.  For a target with isolated torus fixed
/// points, the differences at fixed point `p` are `s_j - s_p` for the
/// classical eigenvalue seeds `s_i` (for `P^n`: `lambda_j - lambda_i`), so
/// these constants are derivable from the same weight data that defines the
/// equivariant frame.
pub(crate) fn classical_r_asymptotics_for_point(
    eigenvalue_differences: &[RatFun],
    z_order: usize,
) -> Vec<RatFun> {
    let mut exponent = vec![RatFun::zero(); z_order + 1];
    for r in 1..=z_order.div_ceil(2) {
        let order = 2 * r - 1;
        let coefficient =
            bernoulli_number(2 * r) / (Rational::from(2 * r) * Rational::from(2 * r - 1));
        let coefficient = RatFun::from_rational(coefficient);
        let mut weight_sum = RatFun::zero();
        for difference in eigenvalue_differences {
            let term = &RatFun::one() / &difference.pow_usize(order);
            weight_sum = &weight_sum + &term;
        }
        exponent[order] = &coefficient * &weight_sum;
    }
    exp_scalar_z_series(&exponent)
}

pub(crate) fn exp_scalar_z_series(exponent: &[RatFun]) -> Vec<RatFun> {
    let z_order = exponent.len().saturating_sub(1);
    let mut out = vec![RatFun::zero(); z_order + 1];
    out[0] = RatFun::one();
    for degree in 1..=z_order {
        let mut total = RatFun::zero();
        for part in 1..=degree {
            if exponent[part].is_zero() {
                continue;
            }
            let scaled = &exponent[part] * &RatFun::from(part);
            let term = &scaled * &out[degree - part];
            total = &total + &term;
        }
        out[degree] = &total / &RatFun::from(degree);
    }
    out
}

pub(crate) fn bernoulli_number(n: usize) -> Rational {
    let mut bernoulli = vec![Rational::zero(); n + 1];
    bernoulli[0] = Rational::one();
    for degree in 1..=n {
        let mut sum = Rational::zero();
        for idx in 0..degree {
            sum += Rational::from(binomial(degree + 1, idx)) * bernoulli[idx].clone();
        }
        bernoulli[degree] = -sum / Rational::from(degree + 1);
    }
    bernoulli[n].clone()
}

pub(crate) fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut out = 1usize;
    for step in 1..=k {
        out = out * (n + 1 - step) / step;
    }
    out
}
