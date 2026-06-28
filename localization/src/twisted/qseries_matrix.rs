//! Generic QSeries-polynomial and series-matrix linear algebra
//! (determinants, polynomial arithmetic, matrix inversion) over any Coeff.

use crate::algebra::Coeff;
use crate::error::GwError;
use crate::series::{QSeries, SeriesMatrix};

pub(crate) fn determinant_qseries_polynomial_matrix<C: Coeff>(
    matrix: &[Vec<Vec<QSeries<C>>>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let size = matrix.len();
    if size == 0 {
        return vec![QSeries::<C>::one(q_degree)];
    }
    if size == 1 {
        return matrix[0][0].clone();
    }

    let mut total = vec![QSeries::<C>::zero(q_degree)];
    for col in 0..size {
        let mut minor = Vec::with_capacity(size - 1);
        for source_row in matrix.iter().skip(1) {
            let mut row = Vec::with_capacity(size - 1);
            for (source_col, entry) in source_row.iter().enumerate() {
                if source_col != col {
                    row.push(entry.clone());
                }
            }
            minor.push(row);
        }
        let term = qseries_polynomial_mul(
            &matrix[0][col],
            &determinant_qseries_polynomial_matrix(&minor, q_degree),
            q_degree,
        );
        total = if col % 2 == 0 {
            qseries_polynomial_add(&total, &term, q_degree)
        } else {
            qseries_polynomial_sub(&total, &term, q_degree)
        };
    }
    total
}

pub(crate) fn qseries_polynomial_add<C: Coeff>(
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let size = left.len().max(right.len());
    let mut out = vec![QSeries::<C>::zero(q_degree); size];
    for degree in 0..size {
        let left_coeff = left
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::<C>::zero(q_degree));
        let right_coeff = right
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::<C>::zero(q_degree));
        out[degree] = left_coeff.add(&right_coeff);
    }
    out
}

pub(crate) fn qseries_polynomial_sub<C: Coeff>(
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let size = left.len().max(right.len());
    let mut out = vec![QSeries::<C>::zero(q_degree); size];
    for degree in 0..size {
        let left_coeff = left
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::<C>::zero(q_degree));
        let right_coeff = right
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::<C>::zero(q_degree));
        out[degree] = left_coeff.sub(&right_coeff);
    }
    out
}

pub(crate) fn qseries_polynomial_mul<C: Coeff>(
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let mut out = vec![QSeries::<C>::zero(q_degree); left.len() + right.len() - 1];
    for (left_degree, left_coeff) in left.iter().enumerate() {
        if left_coeff.is_zero() {
            continue;
        }
        for (right_degree, right_coeff) in right.iter().enumerate() {
            if right_coeff.is_zero() {
                continue;
            }
            out[left_degree + right_degree] =
                out[left_degree + right_degree].add(&left_coeff.mul(right_coeff));
        }
    }
    out
}

pub(crate) fn series_matrix_scale<C: Coeff>(
    matrix: &SeriesMatrix<C>,
    scalar: &QSeries<C>,
) -> SeriesMatrix<C> {
    SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| row.iter().map(|entry| entry.mul(scalar)).collect())
            .collect(),
    )
}

pub(crate) fn invert_series_matrix_coeff<C: Coeff>(
    matrix: &SeriesMatrix<C>,
) -> Result<SeriesMatrix<C>, GwError> {
    if matrix.rows() != matrix.cols() {
        return Err(GwError::ConventionMismatch(
            "matrix inversion requires a square matrix".to_string(),
        ));
    }
    let size = matrix.rows();
    let q_degree = matrix.max_degree();
    let mut augmented = vec![vec![QSeries::<C>::zero(q_degree); 2 * size]; size];
    for (row, augmented_row) in augmented.iter_mut().enumerate() {
        for col in 0..size {
            augmented_row[col] = matrix.entry(row, col).clone();
        }
        augmented_row[size + row] = QSeries::one(q_degree);
    }

    for col in 0..size {
        let pivot = (col..size)
            .find(|row| qseries_has_invertible_constant_coeff(&augmented[*row][col]))
            .ok_or(GwError::NonSemisimplePoint)?;
        if pivot != col {
            augmented.swap(pivot, col);
        }
        let pivot_inv = augmented[col][col].inverse()?;
        for entry in &mut augmented[col] {
            *entry = entry.mul(&pivot_inv);
        }
        let pivot_row = augmented[col].clone();
        for row in 0..size {
            if row == col {
                continue;
            }
            let factor = augmented[row][col].clone();
            if factor.is_zero() {
                continue;
            }
            for (entry, pivot_entry) in augmented[row].iter_mut().zip(&pivot_row) {
                *entry = entry.sub(&factor.mul(pivot_entry));
            }
        }
    }

    Ok(SeriesMatrix::from_entries(
        augmented
            .into_iter()
            .map(|row| row.into_iter().skip(size).collect())
            .collect(),
    ))
}

pub(crate) fn qseries_has_invertible_constant_coeff<C: Coeff>(series: &QSeries<C>) -> bool {
    series.coeff(0).is_some_and(|constant| !constant.is_zero())
}

pub(crate) fn derivative_qseries_polynomial_coefficients<C: Coeff>(
    coefficients: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coeff)| coeff.scale(&C::from_usize(power)))
        .chain(std::iter::once(QSeries::<C>::zero(q_degree)))
        .collect::<Vec<_>>()
}

pub(crate) fn evaluate_qseries_polynomial<C: Coeff>(
    coefficients: &[QSeries<C>],
    x: &QSeries<C>,
) -> QSeries<C> {
    let q_degree = x.max_degree();
    let mut out = QSeries::<C>::zero(q_degree);
    for coeff in coefficients.iter().rev() {
        out = out.mul(x).add(coeff);
    }
    out
}

pub(crate) fn canonical_evaluation_matrix_local(roots: &[QSeries]) -> SeriesMatrix {
    SeriesMatrix::from_entries(
        roots
            .iter()
            .map(|root| {
                (0..roots.len())
                    .map(|power| root.pow_usize(power))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
}

pub(crate) fn relative_sqrt_delta_series_local(delta: &QSeries) -> Result<QSeries, GwError> {
    relative_sqrt_delta_series_coeff(delta)
}

pub(crate) fn relative_sqrt_delta_series_coeff<C: Coeff>(
    delta: &QSeries<C>,
) -> Result<QSeries<C>, GwError> {
    let delta0 = delta
        .coeff(0)
        .ok_or_else(|| GwError::AlgebraFailure("empty twisted Delta series".to_string()))?;
    if delta0.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    let inv_delta0 = C::one().div(delta0);
    delta.scale(&inv_delta0).sqrt_with_constant_one()
}
