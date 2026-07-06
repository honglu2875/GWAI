//! Generic coefficient-matrix and Birkhoff q-degree factorization helpers
//! used by the twisted calibration path.

use crate::algebra::Coeff;
use crate::error::GwError;
use crate::series::{QSeries, SeriesMatrix};
use std::collections::BTreeMap;

pub(crate) type CoeffMatrix<C> = Vec<Vec<C>>;
pub(crate) type LaurentCoeffMatrix<C> = BTreeMap<i32, CoeffMatrix<C>>;
pub(crate) type QDegreeLaurentFactor<C> = Vec<LaurentCoeffMatrix<C>>;
pub(crate) type Bidegree = (usize, usize);
pub(crate) type BidegreeLaurentFactor<C> = BTreeMap<Bidegree, LaurentCoeffMatrix<C>>;

pub(crate) fn birkhoff_factor_by_q_degree<C: Coeff>(
    size: usize,
    q_degree: usize,
    matrix: &BTreeMap<i32, SeriesMatrix<C>>,
) -> Result<(QDegreeLaurentFactor<C>, QDegreeLaurentFactor<C>), GwError> {
    // Recursive Birkhoff split in Novikov degree.  At each q^d, all lower-degree
    // products are known; the remaining Laurent matrix is uniquely split into
    // nonnegative and negative z-powers.
    validate_identity_at_q_zero(size, matrix)?;
    let mut positive = vec![BTreeMap::new(); q_degree + 1];
    let mut negative = vec![BTreeMap::new(); q_degree + 1];
    positive[0].insert(0, identity_coeff_matrix(size));
    negative[0].insert(0, identity_coeff_matrix(size));

    for degree in 1..=q_degree {
        let mut raw = q_degree_slice(matrix, degree, size);
        let known = multiply_laurent_matrix_q_slices(&negative, &positive, degree, size);
        subtract_laurent_matrix(&mut raw, &known);
        for (z_power, coeff) in raw {
            if coeff_matrix_is_zero(&coeff) {
                continue;
            }
            if z_power >= 0 {
                positive[degree].insert(z_power, coeff);
            } else {
                negative[degree].insert(z_power, coeff);
            }
        }
    }

    Ok((positive, negative))
}

#[cfg(test)]
pub(crate) fn birkhoff_factor_by_bidegree<C: Coeff>(
    size: usize,
    max_total_degree: usize,
    matrix: &BidegreeLaurentFactor<C>,
) -> Result<(BidegreeLaurentFactor<C>, BidegreeLaurentFactor<C>), GwError> {
    // Same recursive split as the one-variable factorization, but over the
    // completed bidegree Novikov ring.  Each nonzero bidegree only depends on
    // products of strictly lower total degree, so the z-Laurent split remains
    // an ordinary coefficientwise operation.
    validate_bidegree_identity_at_zero(size, matrix)?;

    let mut positive = BidegreeLaurentFactor::new();
    let mut negative = BidegreeLaurentFactor::new();
    positive
        .entry((0, 0))
        .or_default()
        .insert(0, identity_coeff_matrix(size));
    negative
        .entry((0, 0))
        .or_default()
        .insert(0, identity_coeff_matrix(size));

    for total in 1..=max_total_degree {
        for first in 0..=total {
            let grade = (first, total - first);
            let mut raw = bidegree_slice(matrix, grade, size);
            let known = multiply_laurent_matrix_bidegree_slices(&negative, &positive, grade, size);
            subtract_laurent_matrix(&mut raw, &known);
            for (z_power, coeff) in raw {
                if coeff_matrix_is_zero(&coeff) {
                    continue;
                }
                if z_power >= 0 {
                    positive.entry(grade).or_default().insert(z_power, coeff);
                } else {
                    negative.entry(grade).or_default().insert(z_power, coeff);
                }
            }
        }
    }

    Ok((positive, negative))
}

pub(crate) fn birkhoff_negative_factor_by_bidegree_with_z_bounds<C: Coeff>(
    size: usize,
    max_total_degree: usize,
    matrix: &BidegreeLaurentFactor<C>,
    positive_z_windows: &BTreeMap<Bidegree, usize>,
    negative_z_depths: &BTreeMap<Bidegree, usize>,
) -> Result<BidegreeLaurentFactor<C>, GwError> {
    validate_bidegree_identity_at_zero(size, matrix)?;

    let mut positive = BidegreeLaurentFactor::new();
    let mut negative = BidegreeLaurentFactor::new();
    positive
        .entry((0, 0))
        .or_default()
        .insert(0, identity_coeff_matrix(size));
    negative
        .entry((0, 0))
        .or_default()
        .insert(0, identity_coeff_matrix(size));

    for total in 1..=max_total_degree {
        for first in 0..=total {
            let grade = (first, total - first);
            let min_z = -i32::try_from(negative_z_depths.get(&grade).copied().unwrap_or(0))
                .map_err(|_| {
                    GwError::AlgebraFailure("negative z-depth does not fit in i32".to_string())
                })?;
            let max_z = i32::try_from(positive_z_windows.get(&grade).copied().unwrap_or(0))
                .map_err(|_| {
                    GwError::AlgebraFailure("positive z-window does not fit in i32".to_string())
                })?;
            let mut raw = bidegree_slice_z_window(matrix, grade, size, min_z, max_z);
            let known = multiply_laurent_matrix_bidegree_slices_z_window(
                &negative, &positive, grade, size, min_z, max_z,
            );
            subtract_laurent_matrix(&mut raw, &known);
            for (z_power, coeff) in raw {
                if coeff_matrix_is_zero(&coeff) {
                    continue;
                }
                if z_power >= 0 {
                    positive.entry(grade).or_default().insert(z_power, coeff);
                } else {
                    negative.entry(grade).or_default().insert(z_power, coeff);
                }
            }
        }
    }

    Ok(negative)
}

pub(crate) fn validate_bidegree_identity_at_zero<C: Coeff>(
    size: usize,
    matrix: &BidegreeLaurentFactor<C>,
) -> Result<(), GwError> {
    for (grade, laurent) in matrix {
        for (z_power, coefficient) in laurent {
            let expected = if *grade == (0, 0) && *z_power == 0 {
                identity_coeff_matrix(size)
            } else {
                zero_coeff_matrix(size)
            };
            if *grade == (0, 0) && *z_power == 0 {
                if coefficient != &expected {
                    return Err(GwError::ConventionMismatch(
                        "bidegree Birkhoff input must be identity at degree (0,0)".to_string(),
                    ));
                }
            } else if *grade == (0, 0) && !coeff_matrix_is_zero(coefficient) {
                return Err(GwError::ConventionMismatch(format!(
                    "bidegree Birkhoff input must have no nonzero z^{z_power} term at degree (0,0)"
                )));
            }
        }
    }
    if !matrix
        .get(&(0, 0))
        .and_then(|laurent| laurent.get(&0))
        .is_some_and(|coefficient| coefficient == &identity_coeff_matrix(size))
    {
        return Err(GwError::ConventionMismatch(
            "bidegree Birkhoff input is missing the identity at degree (0,0)".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn bidegree_slice<C: Coeff>(
    matrix: &BidegreeLaurentFactor<C>,
    grade: Bidegree,
    size: usize,
) -> LaurentCoeffMatrix<C> {
    matrix.get(&grade).cloned().unwrap_or_else(|| {
        let mut out = LaurentCoeffMatrix::new();
        out.insert(0, zero_coeff_matrix(size));
        out
    })
}

pub(crate) fn bidegree_slice_z_window<C: Coeff>(
    matrix: &BidegreeLaurentFactor<C>,
    grade: Bidegree,
    size: usize,
    min_z: i32,
    max_z: i32,
) -> LaurentCoeffMatrix<C> {
    let mut out = LaurentCoeffMatrix::new();
    if let Some(laurent) = matrix.get(&grade) {
        for (&z_power, coeff) in laurent.range(min_z..=max_z) {
            if !coeff_matrix_is_zero(coeff) {
                out.insert(z_power, coeff.clone());
            }
        }
    }
    if out.is_empty() {
        out.insert(0, zero_coeff_matrix(size));
    }
    out
}

#[cfg(test)]
pub(crate) fn multiply_laurent_matrix_bidegree_slices<C: Coeff>(
    left: &BidegreeLaurentFactor<C>,
    right: &BidegreeLaurentFactor<C>,
    grade: Bidegree,
    size: usize,
) -> LaurentCoeffMatrix<C> {
    multiply_laurent_matrix_bidegree_slices_z_window(left, right, grade, size, i32::MIN, i32::MAX)
}

pub(crate) fn multiply_laurent_matrix_bidegree_slices_z_window<C: Coeff>(
    left: &BidegreeLaurentFactor<C>,
    right: &BidegreeLaurentFactor<C>,
    grade: Bidegree,
    size: usize,
    min_z: i32,
    max_z: i32,
) -> LaurentCoeffMatrix<C> {
    let mut out = BTreeMap::new();
    for left_first in 0..=grade.0 {
        for left_second in 0..=grade.1 {
            let left_grade = (left_first, left_second);
            if left_grade == (0, 0) || left_grade == grade {
                continue;
            }
            let right_grade = (grade.0 - left_first, grade.1 - left_second);
            let Some(left_laurent) = left.get(&left_grade) else {
                continue;
            };
            let Some(right_laurent) = right.get(&right_grade) else {
                continue;
            };
            for (left_z, left_matrix) in left_laurent {
                for (right_z, right_matrix) in right_laurent {
                    let z_power = left_z + right_z;
                    if z_power < min_z || z_power > max_z {
                        continue;
                    }
                    add_product_matrix_to_laurent(
                        &mut out,
                        z_power,
                        left_matrix,
                        right_matrix,
                        size,
                    );
                }
            }
        }
    }
    out
}

pub(crate) fn validate_identity_at_q_zero<C: Coeff>(
    size: usize,
    matrix: &BTreeMap<i32, SeriesMatrix<C>>,
) -> Result<(), GwError> {
    for (z_power, coefficient) in matrix {
        let q0 = matrix_q_coefficient(coefficient, 0);
        let expected = if *z_power == 0 {
            identity_coeff_matrix(size)
        } else {
            zero_coeff_matrix(size)
        };
        if q0 != expected {
            return Err(GwError::ConventionMismatch(format!(
                "Birkhoff input must be identity at q=0; z^{z_power} coefficient is nonstandard"
            )));
        }
    }
    Ok(())
}

pub(crate) fn q_degree_slice<C: Coeff>(
    matrix: &BTreeMap<i32, SeriesMatrix<C>>,
    degree: usize,
    size: usize,
) -> LaurentCoeffMatrix<C> {
    let mut out = BTreeMap::new();
    for (z_power, coefficient) in matrix {
        let q_coeff = matrix_q_coefficient(coefficient, degree);
        if !coeff_matrix_is_zero(&q_coeff) {
            out.insert(*z_power, q_coeff);
        }
    }
    if out.is_empty() {
        out.insert(0, zero_coeff_matrix(size));
    }
    out
}

pub(crate) fn matrix_q_coefficient<C: Coeff>(
    matrix: &SeriesMatrix<C>,
    degree: usize,
) -> CoeffMatrix<C> {
    matrix
        .entries()
        .iter()
        .map(|row| {
            row.iter()
                .map(|entry| entry.coeff(degree).cloned().unwrap_or_else(C::zero))
                .collect()
        })
        .collect()
}

pub(crate) fn multiply_laurent_matrix_q_slices<C: Coeff>(
    left: &[LaurentCoeffMatrix<C>],
    right: &[LaurentCoeffMatrix<C>],
    degree: usize,
    size: usize,
) -> LaurentCoeffMatrix<C> {
    let mut out = BTreeMap::new();
    for split in 1..degree {
        for (left_z, left_matrix) in &left[split] {
            for (right_z, right_matrix) in &right[degree - split] {
                add_product_matrix_to_laurent(
                    &mut out,
                    left_z + right_z,
                    left_matrix,
                    right_matrix,
                    size,
                );
            }
        }
    }
    out
}

pub(crate) fn subtract_laurent_matrix<C: Coeff>(
    target: &mut LaurentCoeffMatrix<C>,
    rhs: &LaurentCoeffMatrix<C>,
) {
    for (z_power, matrix) in rhs {
        add_matrix_to_laurent(target, *z_power, matrix, true);
    }
}

pub(crate) fn add_matrix_to_laurent<C: Coeff>(
    target: &mut LaurentCoeffMatrix<C>,
    z_power: i32,
    matrix: &CoeffMatrix<C>,
    negate: bool,
) {
    if coeff_matrix_is_zero(matrix) {
        return;
    }
    let size = matrix.len();
    let entry = target
        .entry(z_power)
        .or_insert_with(|| zero_coeff_matrix(size));
    for row in 0..size {
        for col in 0..size {
            if matrix[row][col].is_structurally_zero() {
                continue;
            }
            let term = if negate {
                matrix[row][col].neg()
            } else {
                matrix[row][col].clone()
            };
            entry[row][col] = entry[row][col].add(&term);
        }
    }
    if coeff_matrix_is_zero(entry) {
        target.remove(&z_power);
    }
}

pub(crate) fn add_product_matrix_to_laurent<C: Coeff>(
    target: &mut LaurentCoeffMatrix<C>,
    z_power: i32,
    left: &CoeffMatrix<C>,
    right: &CoeffMatrix<C>,
    size: usize,
) {
    for row in 0..size {
        for mid in 0..size {
            if left[row][mid].is_structurally_zero() {
                continue;
            }
            for col in 0..size {
                if right[mid][col].is_structurally_zero() {
                    continue;
                }
                let term = left[row][mid].mul(&right[mid][col]);
                if term.is_zero() {
                    continue;
                }
                let entry = target
                    .entry(z_power)
                    .or_insert_with(|| zero_coeff_matrix::<C>(size));
                entry[row][col] = entry[row][col].add(&term);
            }
        }
    }
}

pub(crate) fn negative_factor_to_s_coefficients<C: Coeff>(
    size: usize,
    q_degree: usize,
    z_order: usize,
    negative: &[LaurentCoeffMatrix<C>],
) -> Vec<SeriesMatrix<C>> {
    let mut coefficients = Vec::with_capacity(z_order + 1);
    for order in 0..=z_order {
        let mut entries = vec![vec![vec![C::zero(); q_degree + 1]; size]; size];
        if order == 0 {
            for idx in 0..size {
                entries[idx][idx][0] = C::one();
            }
        } else {
            let z_power = -(order as i32);
            for degree in 1..=q_degree {
                if let Some(matrix) = negative[degree].get(&z_power) {
                    for row in 0..size {
                        for col in 0..size {
                            entries[row][col][degree] = matrix[row][col].clone();
                        }
                    }
                }
            }
        }
        coefficients.push(SeriesMatrix::from_entries(
            entries
                .into_iter()
                .map(|row| row.into_iter().map(QSeries::from_coeffs).collect())
                .collect(),
        ));
    }
    coefficients
}

pub(crate) fn identity_coeff_matrix<C: Coeff>(size: usize) -> CoeffMatrix<C> {
    let mut out = zero_coeff_matrix(size);
    for idx in 0..size {
        out[idx][idx] = C::one();
    }
    out
}

pub(crate) fn zero_coeff_matrix<C: Coeff>(size: usize) -> CoeffMatrix<C> {
    vec![vec![C::zero(); size]; size]
}

pub(crate) fn coeff_matrix_is_zero<C: Coeff>(matrix: &CoeffMatrix<C>) -> bool {
    matrix.iter().all(|row| row.iter().all(Coeff::is_zero))
}
