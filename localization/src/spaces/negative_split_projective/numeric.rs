//! Small target-specific numeric helpers for the twisted calibration.

use crate::core::algebra::RatFun;
use crate::core::series::{QSeries, SeriesMatrix};

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
