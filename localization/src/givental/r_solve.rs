//! Semisimple R-matrix solving: elementary symmetric functions, sqrt-delta
//! normalization, canonical evaluation matrices, and the coefficient-generic
//! flatness recursion that fixes the R-matrix coefficients.

use crate::core::algebra::{Coeff, RatFun, Rational};
use crate::core::error::GwError;
use crate::core::series::{QSeries, SeriesMatrix};

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

/// Solves the canonical-coordinate `R`-matrix flatness recursion over any
/// exact coefficient representation.
///
/// Root differences determine the off-diagonal entries, while the supplied
/// classical diagonal asymptotics are the integration constants for the
/// diagonal equations.
pub(crate) fn solve_r_coefficients_from_flatness<C: Coeff>(
    roots: &[QSeries<C>],
    connection: &SeriesMatrix<C>,
    classical_diagonal: &[Vec<C>],
    q_degree: usize,
    z_order: usize,
) -> Result<Vec<SeriesMatrix<C>>, GwError> {
    let size = roots.len();
    let mut coefficients = Vec::with_capacity(z_order + 1);
    coefficients.push(SeriesMatrix::<C>::identity(size, q_degree));

    for order in 1..=z_order {
        let previous = &coefficients[order - 1];
        let recursion_source = previous.q_derivative().add(&connection.mul(previous));
        let mut entries = vec![vec![QSeries::<C>::zero(q_degree); size]; size];

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

pub(crate) fn solve_r_diagonal_from_flatness<C: Coeff>(
    connection: &SeriesMatrix<C>,
    entries: &[Vec<QSeries<C>>],
    branch: usize,
    constant: C,
    q_degree: usize,
) -> QSeries<C> {
    let mut known = QSeries::<C>::zero(q_degree);
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
        .unwrap_or_else(C::zero);

    let mut coeffs = vec![C::zero(); q_degree + 1];
    coeffs[0] = constant;
    for degree in 1..=q_degree {
        let mut numerator = target.coeff(degree).cloned().unwrap_or_else(C::zero);
        for connection_degree in 1..=degree {
            let term = diagonal_connection
                .coeff(connection_degree)
                .cloned()
                .unwrap_or_else(C::zero)
                .mul(&coeffs[degree - connection_degree]);
            numerator = numerator.sub(&term);
        }
        let denominator = C::from_usize(degree).add(&a0);
        coeffs[degree] = numerator.div(&denominator);
    }
    QSeries::from_coeffs(coeffs)
}
