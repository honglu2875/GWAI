//! Projective-space R-matrix solving: elementary symmetric functions,
//! sqrt-delta normalization, canonical evaluation matrix, and the flatness
//! recursion that fixes the R-matrix coefficients.

use crate::algebra::{RatFun, Rational};
use crate::error::GwError;
use crate::series::{QSeries, SeriesMatrix};

pub(crate) fn elementary_symmetric_rational(weights: &[Rational]) -> Vec<Rational> {
    let mut elementary = vec![Rational::zero(); weights.len() + 1];
    elementary[0] = Rational::one();
    for (idx, weight) in weights.iter().enumerate() {
        for k in (1..=idx + 1).rev() {
            elementary[k] = elementary[k].clone() + elementary[k - 1].clone() * weight.clone();
        }
    }
    elementary
}

pub(crate) fn relative_sqrt_delta_series(delta: &QSeries) -> Result<QSeries, GwError> {
    let delta0 = delta
        .coeff(0)
        .ok_or_else(|| GwError::AlgebraFailure("empty Delta series".to_string()))?;
    if delta0.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    let inv_delta0 = &RatFun::one() / delta0;
    delta.scale(&inv_delta0).sqrt_with_constant_one()
}

pub(crate) fn canonical_evaluation_matrix(roots: &[QSeries]) -> SeriesMatrix {
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

pub(crate) fn solve_projective_r_coefficients(
    roots: &[QSeries],
    connection: &SeriesMatrix,
    _metric: &SeriesMatrix,
    classical_diagonal: &[Vec<RatFun>],
    q_degree: usize,
    z_order: usize,
) -> Result<Vec<SeriesMatrix>, GwError> {
    let size = roots.len();
    let mut coefficients = Vec::with_capacity(z_order + 1);
    coefficients.push(SeriesMatrix::identity(size, q_degree));

    for order in 1..=z_order {
        let previous = &coefficients[order - 1];
        let recursion_source = previous.q_derivative().add(&connection.mul(previous));
        let mut entries = vec![vec![QSeries::zero(q_degree); size]; size];

        for row in 0..size {
            for col in 0..size {
                if row == col {
                    continue;
                }
                let root_difference = roots[col].sub(&roots[row]);
                entries[row][col] = recursion_source
                    .entry(row, col)
                    .neg()
                    .div(&root_difference)?;
            }
        }

        for branch in 0..size {
            entries[branch][branch] = solve_r_diagonal_from_flatness(
                connection,
                &entries,
                branch,
                classical_diagonal[branch][order].clone(),
                q_degree,
            );
        }

        let next = SeriesMatrix::from_entries(entries);
        coefficients.push(next);
    }

    Ok(coefficients)
}

pub(crate) fn solve_r_diagonal_from_flatness(
    connection: &SeriesMatrix,
    entries: &[Vec<QSeries>],
    branch: usize,
    constant: RatFun,
    q_degree: usize,
) -> QSeries {
    let mut known = QSeries::zero(q_degree);
    for (source, row) in entries.iter().enumerate() {
        if source == branch {
            continue;
        }
        known = known.add(&connection.entry(branch, source).mul(&row[branch]));
    }
    let target = known.neg();
    let diagonal_connection = connection.entry(branch, branch);
    let a0 = diagonal_connection
        .coeff(0)
        .cloned()
        .unwrap_or_else(RatFun::zero);

    let mut coeffs = vec![RatFun::zero(); q_degree + 1];
    coeffs[0] = constant;
    for degree in 1..=q_degree {
        let mut numerator = target.coeff(degree).cloned().unwrap_or_else(RatFun::zero);
        for connection_degree in 1..=degree {
            let term = diagonal_connection
                .coeff(connection_degree)
                .cloned()
                .unwrap_or_else(RatFun::zero)
                * coeffs[degree - connection_degree].clone();
            numerator = numerator - term;
        }
        let denominator = RatFun::from(degree) + a0.clone();
        coeffs[degree] = numerator / denominator;
    }
    QSeries::from_coeffs(coeffs)
}
