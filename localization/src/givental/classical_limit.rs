//! Classical (q=0) limit diagonal R-matrix coefficients and the small
//! numeric helpers (exp series, Bernoulli numbers, binomials) they use.

use crate::core::algebra::{lambda, Coeff, RatFun, Rational};

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
        let coefficient = bernoulli_asymptotic_coefficient::<RatFun>(r);
        let mut weight_sum = RatFun::zero();
        for difference in eigenvalue_differences {
            let term = &RatFun::one() / &difference.pow_usize(order);
            weight_sum = &weight_sum + &term;
        }
        exponent[order] = &coefficient * &weight_sum;
    }
    exp_scalar_z_series(&exponent)
}

/// Exponentiates a scalar formal `z`-series with zero constant term.
///
/// The recurrence follows from `(exp f)' = f' exp f` and works over every
/// exact coefficient representation used by the calibration engines.
pub(crate) fn exp_scalar_z_series<C: Coeff>(exponent: &[C]) -> Vec<C> {
    let z_order = exponent.len().saturating_sub(1);
    let mut out = vec![C::zero(); z_order + 1];
    out[0] = C::one();
    for degree in 1..=z_order {
        let mut total = C::zero();
        for part in 1..=degree {
            if exponent[part].is_zero() {
                continue;
            }
            let term = C::from_usize(part)
                .mul(&exponent[part])
                .mul(&out[degree - part]);
            total = total.add(&term);
        }
        out[degree] = total.div(&C::from_usize(degree));
    }
    out
}

/// The coefficient `B_(2r) / (2r(2r-1))` in the Gamma/Bernoulli
/// asymptotic exponent, converted to the requested exact coefficient type.
pub(crate) fn bernoulli_asymptotic_coefficient<C: Coeff>(r: usize) -> C {
    C::from_rational(bernoulli_number(2 * r) / (Rational::from(2 * r) * Rational::from(2 * r - 1)))
}

pub(crate) fn bernoulli_number(n: usize) -> Rational {
    let mut bernoulli = vec![Rational::zero(); n + 1];
    bernoulli[0] = Rational::one();
    for degree in 1..=n {
        let mut sum = Rational::zero();
        for idx in 0..degree {
            sum += binomial_rational(degree + 1, idx) * bernoulli[idx].clone();
        }
        bernoulli[degree] = -sum / Rational::from(degree + 1);
    }
    bernoulli[n].clone()
}

fn binomial_rational(n: usize, k: usize) -> Rational {
    if k > n {
        return Rational::zero();
    }
    let k = k.min(n - k);
    let mut out = Rational::one();
    for idx in 0..k {
        out = out * Rational::from(n - idx) / Rational::from(idx + 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_numbers_use_the_standard_even_convention() {
        assert_eq!(bernoulli_number(0), Rational::one());
        assert_eq!(bernoulli_number(1), Rational::new(-1, 2));
        assert_eq!(bernoulli_number(2), Rational::new(1, 6));
        assert_eq!(bernoulli_number(4), Rational::new(-1, 30));
    }

    #[test]
    fn scalar_exponential_is_coefficient_generic() {
        let rational_exponent = vec![
            Rational::zero(),
            Rational::one(),
            Rational::zero(),
            Rational::zero(),
        ];
        let rational = exp_scalar_z_series(&rational_exponent);
        assert_eq!(
            rational,
            vec![
                Rational::one(),
                Rational::one(),
                Rational::new(1, 2),
                Rational::new(1, 6),
            ]
        );

        let symbolic_exponent = rational_exponent
            .into_iter()
            .map(RatFun::from_rational)
            .collect::<Vec<_>>();
        let symbolic = exp_scalar_z_series(&symbolic_exponent);
        assert_eq!(
            symbolic,
            rational
                .into_iter()
                .map(RatFun::from_rational)
                .collect::<Vec<_>>()
        );
    }
}
