//! Givental cone-point normalization, fundamental solutions, and descendant-S
//! extraction in a cyclic cohomology basis.

use crate::core::algebra::{Coeff, Rational};
use crate::core::error::GwError;
use crate::core::series::SeriesMatrix;
use crate::givental::{CalibrationId, SeriesSMatrix};
use crate::reconstruction::{
    birkhoff_negative_factor_by_q_degree_with_z_bounds, coeff_matrix_is_zero,
    compose_h_laurent_q_series, compose_h_laurent_q_series_coeff,
    exp_minus_h_mirror_over_z_coefficients, full_vector_mirror_gauge_coefficients_coeff,
    h_coeff_laurent_columns_to_laurent_matrix, h_laurent_columns_to_laurent_matrix,
    invert_series_matrix_coeff, matrix_q_coefficient, multiply_h_laurent_q_series,
    multiply_h_laurent_q_series_mod_relation_coeff, negative_factor_to_s_coefficients,
    plan_birkhoff_windows, quantum_derivative_h_laurent_q_series,
    quantum_derivative_h_laurent_q_series_mod_relation,
    quantum_derivative_h_laurent_q_series_mod_relation_coeff, HCoeffLaurentSeries, HLaurentSeries,
};
use std::collections::BTreeMap;

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
    validate_birkhoff_request_bounds(q_degree, z_order)?;
    if z_order == 0 {
        return SeriesSMatrix::from_coefficients(
            size,
            q_degree,
            0,
            vec![SeriesMatrix::identity(size, q_degree)],
            calibration,
        );
    }
    // Birkhoff factorization splits the Laurent fundamental solution into
    // S(z^{-1})^{-1} * P(z).  We keep the negative factor and convert its
    // z^{-k} terms into the descendant S-matrix coefficients.
    let (positive_windows, negative_depths) = birkhoff_q_z_bounds(fundamental, q_degree, z_order)?;
    let s_factor = birkhoff_negative_factor_by_q_degree_with_z_bounds(
        size,
        q_degree,
        fundamental,
        &positive_windows,
        &negative_depths,
    )?;
    let coefficients = negative_factor_to_s_coefficients(size, q_degree, z_order, &s_factor);
    SeriesSMatrix::from_coefficients(size, q_degree, z_order, coefficients, calibration)
}

pub(crate) fn validate_birkhoff_request_bounds(
    q_degree: usize,
    z_order: usize,
) -> Result<(), GwError> {
    q_degree.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("Birkhoff q-degree count overflow".to_string())
    })?;
    z_order.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("Birkhoff z-order count overflow".to_string())
    })?;
    i32::try_from(z_order).map_err(|_| {
        GwError::UnsupportedInvariant("Birkhoff z-order exceeds i32 range".to_string())
    })?;
    Ok(())
}

pub(crate) fn required_birkhoff_negative_z_depth<C: Coeff>(
    fundamental: &BTreeMap<i32, SeriesMatrix<C>>,
    q_degree: usize,
    z_order: usize,
) -> Result<usize, GwError> {
    let (_, negative_depths) = birkhoff_q_z_bounds(fundamental, q_degree, z_order)?;
    Ok(negative_depths.values().copied().max().unwrap_or(0))
}

fn birkhoff_q_z_bounds<C: Coeff>(
    fundamental: &BTreeMap<i32, SeriesMatrix<C>>,
    q_degree: usize,
    z_order: usize,
) -> Result<(BTreeMap<usize, usize>, BTreeMap<usize, usize>), GwError> {
    let raw_positive_windows = max_nonnegative_z_power_by_q_degree(fundamental, q_degree);
    let grades = (1..=q_degree).collect::<Vec<_>>();
    let plan = plan_birkhoff_windows(
        &grades,
        &raw_positive_windows,
        z_order,
        |degree| (1..*degree).map(|left| (left, degree - left)).collect(),
        "cone-point Birkhoff negative Laurent depth overflow",
    )?;
    Ok((plan.positive_windows, plan.negative_depths))
}

fn max_nonnegative_z_power_by_q_degree<C: Coeff>(
    fundamental: &BTreeMap<i32, SeriesMatrix<C>>,
    q_degree: usize,
) -> BTreeMap<usize, usize> {
    let mut out = BTreeMap::new();
    for degree in 1..=q_degree {
        if let Some(max_z) = fundamental
            .iter()
            .filter(|(z_power, _)| **z_power >= 0)
            .filter_map(|(z_power, matrix)| {
                let coefficient = matrix_q_coefficient(matrix, degree);
                (!coeff_matrix_is_zero(&coefficient)).then_some(*z_power as usize)
            })
            .max()
        {
            out.insert(degree, max_z);
        }
    }
    out
}

/// Convert a descendant fundamental solution to its action on covector
/// insertions: `S^* = eta^{-1} S^T eta`.
pub(crate) fn metric_adjoint_descendant_s_matrix_coeff<C: Coeff>(
    s_matrix: SeriesSMatrix<C>,
    flat_metric: &SeriesMatrix<C>,
) -> Result<SeriesSMatrix<C>, GwError> {
    let metric_inverse = invert_series_matrix_coeff(flat_metric)?;
    metric_adjoint_descendant_s_matrix_with_inverse_coeff(s_matrix, flat_metric, &metric_inverse)
}

/// Metric-adjoint conversion when the inverse pairing is already available.
pub(crate) fn metric_adjoint_descendant_s_matrix_with_inverse_coeff<C: Coeff>(
    s_matrix: SeriesSMatrix<C>,
    flat_metric: &SeriesMatrix<C>,
    metric_inverse: &SeriesMatrix<C>,
) -> Result<SeriesSMatrix<C>, GwError> {
    let size = s_matrix.size();
    let q_degree = s_matrix.q_degree();
    let coefficients = s_matrix
        .coefficients()
        .iter()
        .map(|matrix| {
            // Birkhoff normalization gives S_0 = I exactly.  Multiplying
            // eta^{-1} I eta is algebraically redundant, but over a factored
            // coefficient ring it can materialize a large sum which is only
            // recognized as I after expanding common denominators.  Preserve
            // an already structural identity before taking the metric
            // adjoint.  Nonidentity coefficients still follow the complete
            // eta^{-1} S_k^T eta formula below.
            let structural_identity = matrix.rows() == size
                && matrix.cols() == size
                && (0..size).all(|row| {
                    (0..size).all(|col| {
                        let entry = matrix.entry(row, col);
                        if row == col {
                            entry.is_structurally_one()
                        } else {
                            entry.is_structurally_zero()
                        }
                    })
                });
            if structural_identity {
                SeriesMatrix::identity(size, q_degree)
            } else {
                metric_inverse.mul(&matrix.transpose()).mul(flat_metric)
            }
        })
        .collect::<Vec<_>>();
    SeriesSMatrix::from_coefficients(
        s_matrix.size(),
        s_matrix.q_degree(),
        s_matrix.z_order(),
        coefficients,
        CalibrationId(format!("{}-metric-adjoint", s_matrix.calibration().0)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algebra::RatFun;
    use crate::factored::FactoredRatFun;

    #[test]
    fn metric_adjoint_preserves_structural_identity_without_changing_other_modes() {
        let a = FactoredRatFun::variable("a");
        let one = FactoredRatFun::one();
        let zero = FactoredRatFun::zero();
        let determinant = &(&a * &a) - &one;
        let inverse_determinant = &one / &determinant;
        let metric = SeriesMatrix::constant(
            vec![vec![a.clone(), one.clone()], vec![one.clone(), a.clone()]],
            0,
        );
        let metric_inverse = SeriesMatrix::constant(
            vec![
                vec![&a * &inverse_determinant, -inverse_determinant.clone()],
                vec![-inverse_determinant.clone(), &a * &inverse_determinant],
            ],
            0,
        );
        let identity = SeriesMatrix::<FactoredRatFun>::identity(2, 0);
        let nonidentity =
            SeriesMatrix::constant(vec![vec![one.clone(), one.clone()], vec![zero, one]], 0);

        // This is the pre-optimization formula.  It is semantically I, but
        // the cross-denominator cancellation is not a structural identity in
        // the factored representation.
        let expanded_identity_adjoint = metric_inverse.mul(&identity.transpose()).mul(&metric);
        assert!(!expanded_identity_adjoint.entry(0, 0).is_structurally_one());
        for row in 0..2 {
            for col in 0..2 {
                let expected = if row == col {
                    RatFun::one()
                } else {
                    RatFun::zero()
                };
                assert_eq!(
                    expanded_identity_adjoint
                        .entry(row, col)
                        .coeff(0)
                        .unwrap()
                        .to_ratfun(),
                    expected
                );
            }
        }

        let old_nonidentity_adjoint = metric_inverse.mul(&nonidentity.transpose()).mul(&metric);
        let s = SeriesSMatrix::from_coefficients(
            2,
            0,
            1,
            vec![identity.clone(), nonidentity],
            CalibrationId("structural-identity-regression".to_string()),
        )
        .unwrap();
        let adjoint =
            metric_adjoint_descendant_s_matrix_with_inverse_coeff(s, &metric, &metric_inverse)
                .unwrap();

        assert_eq!(adjoint.coefficient(0).unwrap(), &identity);
        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(
                    adjoint
                        .coefficient(1)
                        .unwrap()
                        .entry(row, col)
                        .coeff(0)
                        .unwrap()
                        .to_ratfun(),
                    old_nonidentity_adjoint
                        .entry(row, col)
                        .coeff(0)
                        .unwrap()
                        .to_ratfun()
                );
            }
        }
        assert_ne!(
            adjoint.coefficient(1).unwrap(),
            &SeriesMatrix::identity(2, 0)
        );
    }
}
