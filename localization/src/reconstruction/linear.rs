use crate::core::algebra::Rational;
use crate::core::error::GwError;

/// Solve a square linear system over the exact rationals in place.
///
/// `values` is replaced by the solution vector.  Callers are responsible for
/// supplying a square matrix with the same number of rows as values; keeping
/// this primitive target-neutral lets interpolation and mirror-coordinate
/// reconstruction share one implementation.
pub(crate) fn solve_rational_system(
    matrix: &mut [Vec<Rational>],
    values: &mut [Rational],
) -> Result<(), GwError> {
    let size = matrix.len();
    if values.len() != size || matrix.iter().any(|row| row.len() != size) {
        return Err(GwError::AlgebraFailure(
            "linear-system dimensions do not agree".to_string(),
        ));
    }
    for pivot in 0..size {
        let row = (pivot..size)
            .find(|&row| !matrix[row][pivot].is_zero())
            .ok_or_else(|| GwError::AlgebraFailure("singular linear system".to_string()))?;
        matrix.swap(pivot, row);
        values.swap(pivot, row);
        let inverse = Rational::one() / matrix[pivot][pivot].clone();
        for col in pivot..size {
            matrix[pivot][col] = matrix[pivot][col].clone() * inverse.clone();
        }
        values[pivot] = values[pivot].clone() * inverse;
        for other in 0..size {
            if other == pivot || matrix[other][pivot].is_zero() {
                continue;
            }
            let factor = matrix[other][pivot].clone();
            for col in pivot..size {
                let term = matrix[pivot][col].clone() * factor.clone();
                matrix[other][col] = matrix[other][col].clone() - term;
            }
            let term = values[pivot].clone() * factor;
            values[other] = values[other].clone() - term;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_solver_rejects_inconsistent_shapes() {
        let mut matrix = vec![vec![Rational::one()]];
        let mut values = Vec::new();
        let error = solve_rational_system(&mut matrix, &mut values).unwrap_err();
        assert!(matches!(error, GwError::AlgebraFailure(_)));

        let mut matrix = vec![vec![Rational::one(), Rational::zero()]];
        let mut values = vec![Rational::one()];
        let error = solve_rational_system(&mut matrix, &mut values).unwrap_err();
        assert!(matches!(error, GwError::AlgebraFailure(_)));
    }
}
