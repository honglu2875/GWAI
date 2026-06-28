//! Small numeric helpers for the twisted calibration: scalar exp series,
//! local Bernoulli numbers, polynomial-by-(affine) multiplication, and
//! constant-matrix extraction.

use super::binomial_rational;
use crate::algebra::{Coeff, RatFun, Rational};
use crate::series::{QSeries, SeriesMatrix};

pub(crate) fn exp_scalar_z_series_local(exponent: &[RatFun]) -> Vec<RatFun> {
    exp_scalar_z_series_coeff(exponent)
}

pub(crate) fn exp_scalar_z_series_coeff<C: Coeff>(exponent: &[C]) -> Vec<C> {
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

pub(crate) fn bernoulli_number_local(n: usize) -> Rational {
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

pub(crate) fn multiply_polynomial_by_linear_series<C: Coeff>(
    poly: &[QSeries<C>],
    constant: &QSeries<C>,
    max_q_degree: usize,
) -> Vec<QSeries<C>> {
    multiply_polynomial_by_affine_h_series(
        poly,
        constant,
        &QSeries::<C>::one(max_q_degree),
        max_q_degree,
    )
}

pub(crate) fn multiply_polynomial_by_affine_h_series<C: Coeff>(
    poly: &[QSeries<C>],
    constant: &QSeries<C>,
    h_coeff: &QSeries<C>,
    max_q_degree: usize,
) -> Vec<QSeries<C>> {
    let mut out = vec![QSeries::<C>::zero(max_q_degree); poly.len() + 1];
    for (degree, coeff) in poly.iter().enumerate() {
        out[degree] = out[degree].add(&coeff.mul(constant));
        out[degree + 1] = out[degree + 1].add(&coeff.mul(h_coeff));
    }
    out
}

pub(crate) fn constant_matrix_at_q_degree(matrix: &SeriesMatrix, q_degree: usize) -> SeriesMatrix {
    SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| {
                        QSeries::constant(
                            entry.coeff(0).cloned().unwrap_or_else(RatFun::zero),
                            q_degree,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
}
