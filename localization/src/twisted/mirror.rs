//! Mirror map, mirror-transformed J-coefficients, fundamental solution
//! matrix, the Birkhoff descendant-S factor, and the H-Laurent q-series
//! operations they are built from.

use super::*;
use crate::algebra::{Coeff, Rational};
use crate::error::GwError;
use crate::givental::{CalibrationId, SeriesSMatrix};
use crate::series::SeriesMatrix;
use std::collections::BTreeMap;

pub fn negative_split_inverse_mirror_map_coefficients(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
) -> Vec<Rational> {
    let mirror = negative_split_mirror_map_coefficients(n, twist, q_degree);
    invert_mirror_map(&mirror, q_degree)
}

pub(crate) fn mirror_map_coefficients_from_i_function(
    i_coefficients: &[HLaurentSeries],
    q_degree: usize,
) -> Vec<Rational> {
    mirror_map_coefficients_from_i_function_coeff(i_coefficients, q_degree)
}

pub(crate) fn mirror_map_coefficients_from_i_function_coeff<C: Coeff>(
    i_coefficients: &[HCoeffLaurentSeries<C>],
    q_degree: usize,
) -> Vec<C> {
    let mut out = vec![C::zero(); q_degree + 1];
    let Some(first) = i_coefficients.first() else {
        return out;
    };
    if first.max_h_power() == 0 {
        return out;
    }
    for (degree, coeff) in out.iter_mut().enumerate().take(q_degree + 1).skip(1) {
        *coeff = i_coefficients
            .get(degree)
            .map(|i_degree| i_degree.coefficient(1, -1))
            .unwrap_or_else(C::zero);
    }
    out
}

pub(crate) fn mirror_transformed_j_coefficients_from_i_function(
    n: usize,
    i_coefficients: &[HLaurentSeries],
    mirror: &[Rational],
    inverse_mirror: &[Rational],
    q_degree: usize,
) -> Vec<HLaurentSeries> {
    // Applies the usual mirror transform: remove the H/z term by the exponential
    // gauge, then re-expand in the flat coordinate using the inverse mirror map.
    let gauge = exp_minus_h_mirror_over_z_coefficients(n, mirror, q_degree);
    let gauged = multiply_h_laurent_q_series(&gauge, i_coefficients, q_degree);
    compose_h_laurent_q_series(&gauged, inverse_mirror, q_degree)
}

pub(crate) fn mirror_transformed_j_coefficients_from_i_function_mod_relation(
    n: usize,
    i_coefficients: &[HLaurentSeries],
    _mirror: &[Rational],
    _inverse_mirror: &[Rational],
    q_degree: usize,
    h_power_relation: &[Rational],
) -> Vec<HLaurentSeries> {
    mirror_transformed_j_coefficients_from_i_function_mod_relation_coeff(
        n,
        i_coefficients,
        _mirror,
        _inverse_mirror,
        q_degree,
        h_power_relation,
    )
}

pub(crate) fn mirror_transformed_j_coefficients_from_i_function_mod_relation_coeff<C: Coeff>(
    n: usize,
    i_coefficients: &[HCoeffLaurentSeries<C>],
    _mirror: &[C],
    inverse_mirror: &[C],
    q_degree: usize,
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let gauge =
        full_vector_mirror_gauge_coefficients_coeff(n, i_coefficients, q_degree, h_power_relation);
    let gauged = multiply_h_laurent_q_series_mod_relation_coeff(
        &gauge,
        i_coefficients,
        q_degree,
        h_power_relation,
    );
    compose_h_laurent_q_series_coeff(&gauged, inverse_mirror, q_degree)
}

pub(crate) fn fundamental_solution_matrix_from_j_coefficients(
    n: usize,
    q_degree: usize,
    j_coefficients: &[HLaurentSeries],
) -> BTreeMap<i32, SeriesMatrix> {
    // The quantum connection fundamental solution is generated from J by
    // repeated application of z q d/dq.  Each derivative gives one flat-basis
    // column.
    let mut columns = Vec::with_capacity(n + 1);
    let mut current = j_coefficients.to_vec();
    for _ in 0..=n {
        columns.push(current.clone());
        current = quantum_derivative_h_laurent_q_series(&current);
    }
    h_laurent_columns_to_laurent_matrix(n, q_degree, &columns)
}

pub(crate) fn fundamental_solution_matrix_from_j_coefficients_mod_relation(
    n: usize,
    q_degree: usize,
    j_coefficients: &[HLaurentSeries],
    h_power_relation: &[Rational],
) -> BTreeMap<i32, SeriesMatrix> {
    let mut columns = Vec::with_capacity(n + 1);
    let mut current = j_coefficients.to_vec();
    for _ in 0..=n {
        columns.push(current.clone());
        current = quantum_derivative_h_laurent_q_series_mod_relation(&current, h_power_relation);
    }
    h_laurent_columns_to_laurent_matrix(n, q_degree, &columns)
}

pub(crate) fn fundamental_solution_matrix_from_j_coefficients_mod_relation_coeff<C: Coeff>(
    n: usize,
    q_degree: usize,
    j_coefficients: &[HCoeffLaurentSeries<C>],
    h_power_relation: &[C],
) -> BTreeMap<i32, SeriesMatrix<C>> {
    let mut columns = Vec::with_capacity(n + 1);
    let mut current = j_coefficients.to_vec();
    for _ in 0..=n {
        columns.push(current.clone());
        current =
            quantum_derivative_h_laurent_q_series_mod_relation_coeff(&current, h_power_relation);
    }
    h_coeff_laurent_columns_to_laurent_matrix(n, q_degree, &columns)
}

pub(crate) fn birkhoff_descendant_s_matrix_from_fundamental(
    size: usize,
    q_degree: usize,
    z_order: usize,
    fundamental: &BTreeMap<i32, SeriesMatrix>,
    calibration: CalibrationId,
) -> Result<SeriesSMatrix, GwError> {
    birkhoff_descendant_s_matrix_from_fundamental_coeff(
        size,
        q_degree,
        z_order,
        fundamental,
        calibration,
    )
}

pub(crate) fn birkhoff_descendant_s_matrix_from_fundamental_coeff<C: Coeff>(
    size: usize,
    q_degree: usize,
    z_order: usize,
    fundamental: &BTreeMap<i32, SeriesMatrix<C>>,
    calibration: CalibrationId,
) -> Result<SeriesSMatrix<C>, GwError> {
    // Birkhoff factorization splits the Laurent fundamental solution into
    // S(z^{-1})^{-1} * P(z).  We keep the negative factor and convert its
    // z^{-k} terms into the descendant S-matrix coefficients.
    let (_, s_factor) = birkhoff_factor_by_q_degree(size, q_degree, fundamental)?;
    let coefficients = negative_factor_to_s_coefficients(size, q_degree, z_order, &s_factor);
    SeriesSMatrix::from_coefficients(size, q_degree, z_order, coefficients, calibration)
}

pub(crate) fn exp_minus_h_mirror_over_z_coefficients(
    n: usize,
    mirror: &[Rational],
    max_degree: usize,
) -> Vec<HLaurentSeries> {
    let mut exponent = vec![HLaurentSeries::zero(n); max_degree + 1];
    for degree in 1..=max_degree {
        let coeff = mirror.get(degree).cloned().unwrap_or_else(Rational::zero);
        if !coeff.is_zero() && n >= 1 {
            exponent[degree].add_term(1, -1, -coeff);
        }
    }

    let mut out = vec![HLaurentSeries::zero(n); max_degree + 1];
    out[0] = HLaurentSeries::one(n);
    for degree in 1..=max_degree {
        let mut sum = HLaurentSeries::zero(n);
        for split in 1..=degree {
            if exponent[split].coeffs.iter().all(BTreeMap::is_empty) {
                continue;
            }
            let term = exponent[split]
                .multiply(&out[degree - split])
                .scale(Rational::from(split));
            sum = sum.add(&term);
        }
        out[degree] = sum.scale(Rational::one() / Rational::from(degree));
    }
    out
}

pub(crate) fn multiply_h_laurent_q_series(
    left: &[HLaurentSeries],
    right: &[HLaurentSeries],
    max_degree: usize,
) -> Vec<HLaurentSeries> {
    let max_h_power = left
        .first()
        .or_else(|| right.first())
        .map(HLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HLaurentSeries::zero(max_h_power); max_degree + 1];
    for left_degree in 0..=max_degree {
        for right_degree in 0..=max_degree - left_degree {
            let term = left[left_degree].multiply(&right[right_degree]);
            out[left_degree + right_degree] = out[left_degree + right_degree].add(&term);
        }
    }
    out
}

pub(crate) fn multiply_h_laurent_q_series_mod_relation_coeff<C: Coeff>(
    left: &[HCoeffLaurentSeries<C>],
    right: &[HCoeffLaurentSeries<C>],
    max_degree: usize,
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let max_h_power = left
        .first()
        .or_else(|| right.first())
        .map(HCoeffLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HCoeffLaurentSeries::<C>::zero(max_h_power); max_degree + 1];
    for left_degree in 0..=max_degree {
        for right_degree in 0..=max_degree - left_degree {
            let term =
                left[left_degree].multiply_mod_relation(&right[right_degree], h_power_relation);
            out[left_degree + right_degree] = out[left_degree + right_degree].add(&term);
        }
    }
    out
}

pub(crate) fn full_vector_mirror_gauge_coefficients_coeff<C: Coeff>(
    n: usize,
    i_coefficients: &[HCoeffLaurentSeries<C>],
    max_degree: usize,
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let mut exponent = vec![HCoeffLaurentSeries::<C>::zero(n); max_degree + 1];
    let mut gauge = vec![HCoeffLaurentSeries::<C>::zero(n); max_degree + 1];
    gauge[0] = HCoeffLaurentSeries::<C>::one(n);

    for degree in 1..=max_degree {
        let mut known_gauge = HCoeffLaurentSeries::<C>::zero(n);
        for split in 1..degree {
            if exponent[split].is_empty() {
                continue;
            }
            let term = exponent[split]
                .multiply_mod_relation(&gauge[degree - split], h_power_relation)
                .scale(C::from_usize(split));
            known_gauge = known_gauge.add(&term);
        }
        known_gauge = known_gauge.scale(C::one().div(&C::from_usize(degree)));
        gauge[degree] = known_gauge;

        let mut gauged_degree = HCoeffLaurentSeries::<C>::zero(n);
        for split in 0..=degree {
            let term = gauge[split]
                .multiply_mod_relation(&i_coefficients[degree - split], h_power_relation);
            gauged_degree = gauged_degree.add(&term);
        }
        let tau = z_power_part_coeff(&gauged_degree, -1);
        exponent[degree] = tau.shift_z(-1).scale(C::from_rational(-Rational::one()));
        gauge[degree] = gauge[degree].add(&exponent[degree]);
    }

    gauge
}

pub(crate) fn z_power_part_coeff<C: Coeff>(
    series: &HCoeffLaurentSeries<C>,
    z_power: i32,
) -> HCoeffLaurentSeries<C> {
    let mut out = HCoeffLaurentSeries::<C>::zero(series.max_h_power());
    for h_power in 0..=series.max_h_power() {
        let coeff = series.coefficient(h_power, z_power);
        if !coeff.is_zero() {
            out.add_term(h_power, 0, coeff);
        }
    }
    out
}

pub(crate) fn compose_h_laurent_q_series(
    series: &[HLaurentSeries],
    input: &[Rational],
    max_degree: usize,
) -> Vec<HLaurentSeries> {
    compose_h_laurent_q_series_coeff(series, input, max_degree)
}

pub(crate) fn compose_h_laurent_q_series_coeff<C: Coeff>(
    series: &[HCoeffLaurentSeries<C>],
    input: &[C],
    max_degree: usize,
) -> Vec<HCoeffLaurentSeries<C>> {
    let max_h_power = series
        .first()
        .map(HCoeffLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HCoeffLaurentSeries::<C>::zero(max_h_power); max_degree + 1];
    let mut power = vec![C::zero(); max_degree + 1];
    power[0] = C::one();
    for source_degree in 0..=max_degree {
        for target_degree in 0..=max_degree {
            if power[target_degree].is_zero() {
                continue;
            }
            let term = series[source_degree].scale(power[target_degree].clone());
            out[target_degree] = out[target_degree].add(&term);
        }
        power = mul_plain_series(&power, input, max_degree);
    }
    out
}

pub(crate) fn quantum_derivative_h_laurent_q_series(
    series: &[HLaurentSeries],
) -> Vec<HLaurentSeries> {
    let max_degree = series.len().saturating_sub(1);
    let max_h_power = series.first().map(HLaurentSeries::max_h_power).unwrap_or(0);
    let mut out = vec![HLaurentSeries::zero(max_h_power); max_degree + 1];
    for degree in 0..=max_degree {
        out[degree] = out[degree].add(&series[degree].multiply_by_h());
        if degree > 0 {
            let derivative_term = series[degree].shift_z(1).scale(Rational::from(degree));
            out[degree] = out[degree].add(&derivative_term);
        }
    }
    out
}

pub(crate) fn quantum_derivative_h_laurent_q_series_mod_relation_coeff<C: Coeff>(
    series: &[HCoeffLaurentSeries<C>],
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let max_degree = series.len().saturating_sub(1);
    let max_h_power = series
        .first()
        .map(HCoeffLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HCoeffLaurentSeries::<C>::zero(max_h_power); max_degree + 1];
    for degree in 0..=max_degree {
        out[degree] = out[degree].add(&series[degree].multiply_by_affine_mod_relation(
            C::one(),
            C::zero(),
            C::zero(),
            h_power_relation,
        ));
        if degree > 0 {
            let derivative_term = series[degree].shift_z(1).scale(C::from_usize(degree));
            out[degree] = out[degree].add(&derivative_term);
        }
    }
    out
}

pub(crate) fn quantum_derivative_h_laurent_q_series_mod_relation(
    series: &[HLaurentSeries],
    h_power_relation: &[Rational],
) -> Vec<HLaurentSeries> {
    quantum_derivative_h_laurent_q_series_mod_relation_coeff(series, h_power_relation)
}
