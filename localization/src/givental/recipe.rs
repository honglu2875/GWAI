//! Target-agnostic calibration recipes.
//!
//! The graph engine is a semisimple-CohFT evaluator: it consumes a
//! [`SemisimpleCalibration`] (TQFT frame data plus `R`-matrix) and a
//! descendant [`SeriesSMatrix`], and never inspects target geometry.  This
//! module holds the *recipes* that manufacture that contract from more
//! primitive semisimple data:
//!
//! - [`calibration_from_canonical_frame`]: canonical frame (roots,
//!   idempotent transition, metric norms) + classical `R`-asymptotics
//!   -> full calibration, via the Dubrovin connection and the flatness
//!   recursion.
//! - [`descendant_s_from_divisor_qde`]: quantum and classical divisor
//!   multiplication -> descendant `S`-matrix, by integrating the quantum
//!   differential equation order by order in `z`.
//!
//! Target-specific builders (projective space today; other targets through
//! the same shape) reduce to assembling a [`CanonicalFrame`] and choosing
//! integration constants.  The mirror-map/Birkhoff route used by twisted
//! theories is an alternative recipe for the same contract and lives in the
//! `twisted` module pending the same extraction.

use super::*;

/// Canonical (idempotent-frame) data of a semisimple quantum ring at a fixed
/// Novikov truncation.
///
/// `transition_to_flat` has the unnormalized canonical idempotents as
/// columns; `flat_to_canonical` restricts flat classes to the canonical
/// branches (for a divisor-generated ring with monomial flat basis this is
/// the Vandermonde matrix of the roots).  `roots` are the canonical
/// eigenvalue series of the divisor multiplication, used by the flatness
/// recursion's off-diagonal denominators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalFrame {
    pub roots: Vec<QSeries>,
    pub transition_to_flat: SeriesMatrix,
    pub flat_to_canonical: SeriesMatrix,
    pub metric_norms: Vec<QSeries>,
    pub inverse_metric_norms: Vec<QSeries>,
}

/// Builds the full semisimple calibration from a canonical frame.
///
/// This is the universal part of the quantum-ring recipe: relative
/// square-root normalization of the frame, `Psi`/`Psi^{-1}`, the Dubrovin
/// connection `Psi^{-1} q d(Psi)/dq`, and the `R`-matrix flatness recursion
/// with the supplied classical diagonal asymptotics as integration
/// constants.
pub fn calibration_from_canonical_frame(
    frame: &CanonicalFrame,
    classical_diagonal: &[Vec<RatFun>],
    q_degree: usize,
    z_order: usize,
    calibration: CalibrationId,
) -> Result<SemisimpleCalibration, GwError> {
    let size = frame.roots.len();

    let relative_sqrt_delta = frame
        .inverse_metric_norms
        .iter()
        .map(relative_sqrt_delta_series)
        .collect::<Result<Vec<_>, _>>()?;
    let relative_sqrt_delta_inv = relative_sqrt_delta
        .iter()
        .map(QSeries::inverse)
        .collect::<Result<Vec<_>, _>>()?;

    let relative_scale = SeriesMatrix::diagonal(relative_sqrt_delta.clone());
    let relative_scale_inv = SeriesMatrix::diagonal(relative_sqrt_delta_inv.clone());
    let psi = frame.transition_to_flat.mul(&relative_scale);
    let psi_inverse = relative_scale_inv.mul(&frame.flat_to_canonical);
    let connection = psi_inverse.mul(&psi.q_derivative());

    let metric = SeriesMatrix::diagonal(
        frame
            .metric_norms
            .iter()
            .map(|norm| {
                QSeries::constant(
                    norm.coeff(0).cloned().unwrap_or_else(RatFun::zero),
                    q_degree,
                )
            })
            .collect(),
    );
    let coefficients = solve_projective_r_coefficients(
        &frame.roots,
        &connection,
        &metric,
        classical_diagonal,
        q_degree,
        z_order,
    )?;

    let r_matrix = SeriesRMatrix {
        size,
        q_degree,
        z_order,
        coefficients,
        calibration,
        convention: CanonicalFrameConvention::RelativeNormalizedCanonicalIdempotents,
    };

    Ok(SemisimpleCalibration {
        r_matrix,
        metric,
        psi,
        psi_inverse,
        connection,
        delta: frame.inverse_metric_norms.clone(),
        inverse_delta: frame.metric_norms.clone(),
        relative_sqrt_delta,
        relative_sqrt_delta_inverse: relative_sqrt_delta_inv,
    })
}

/// Solves the descendant `S`-matrix from the quantum differential equation.
///
/// `S_k` is determined order by order in `z` by
/// `q d/dq S_k = (A_quantum) S_{k-1} - S_{k-1} (A_classical)` with `S_0 = 1`,
/// where `A` is multiplication by the divisor in the flat basis.  The zero
/// integration constant is the small-J-function calibration convention.
pub fn descendant_s_from_divisor_qde(
    quantum_multiplication: &SeriesMatrix,
    classical_multiplication: &SeriesMatrix,
    z_order: usize,
    calibration: CalibrationId,
) -> Result<SeriesSMatrix, GwError> {
    let size = quantum_multiplication.rows();
    let q_degree = quantum_multiplication.max_degree();
    let mut coefficients = Vec::with_capacity(z_order + 1);
    coefficients.push(SeriesMatrix::identity(size, q_degree));

    for order in 1..=z_order {
        let previous = &coefficients[order - 1];
        let source = quantum_multiplication
            .mul(previous)
            .sub(&previous.mul(classical_multiplication));
        coefficients.push(integrate_q_derivative_zero_constant_matrix(&source)?);
    }

    Ok(SeriesSMatrix {
        size,
        q_degree,
        z_order,
        coefficients,
        calibration,
    })
}

/// Evaluates a polynomial with `q`-series coefficients at a `q`-series point
/// (Horner form).
pub(crate) fn evaluate_series_polynomial(coefficients: &[QSeries], point: &QSeries) -> QSeries {
    let q_degree = point.max_degree();
    let mut out = QSeries::zero(q_degree);
    for coefficient in coefficients.iter().rev() {
        out = out.mul(point).add(coefficient);
    }
    out
}

pub(crate) fn series_polynomial_derivative(coefficients: &[QSeries]) -> Vec<QSeries> {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coefficient)| coefficient.scale(&RatFun::from(power)))
        .collect()
}

/// Newton iteration for one root branch of a polynomial with `q`-series
/// coefficients, seeded at its classical (`q = 0`) value.
///
/// Converges in the `q`-adic topology as long as the seed is a simple root of
/// the classical polynomial — the semisimplicity assumption.
pub fn newton_root_series(
    charpoly: &[QSeries],
    seed: &RatFun,
    q_degree: usize,
) -> Result<QSeries, GwError> {
    let derivative = series_polynomial_derivative(charpoly);
    let mut root = QSeries::constant(seed.clone(), q_degree);
    for _ in 0..=q_degree {
        let value = evaluate_series_polynomial(charpoly, &root);
        if value.coeffs().iter().all(RatFun::is_zero) {
            break;
        }
        let slope = evaluate_series_polynomial(&derivative, &root);
        root = root.sub(&value.div(&slope)?);
    }
    Ok(root)
}

/// Canonical frame of a divisor-generated semisimple ring from its root
/// series, by Lagrange interpolation.
///
/// Assumes the flat basis is `1, D, D^2, ...` for the divisor generator `D`,
/// so idempotents are `prod_{j != i}(D - u_j)/(u_i - u_j)`, the evaluation
/// matrix is the Vandermonde of the roots, and the metric norms are
/// `1/P'(u_i)` (the residue pairing of the presentation).
pub fn divisor_lagrange_frame(
    roots: Vec<QSeries>,
    q_degree: usize,
) -> Result<CanonicalFrame, GwError> {
    let size = roots.len();
    let mut inverse_metric_norms = Vec::with_capacity(size);
    let mut metric_norms = Vec::with_capacity(size);
    let mut transition_to_flat = vec![vec![QSeries::zero(q_degree); size]; size];

    for branch in 0..size {
        let mut numerator = vec![QSeries::one(q_degree)];
        let mut denominator = QSeries::one(q_degree);
        for other in 0..size {
            if other == branch {
                continue;
            }
            numerator =
                multiply_qseries_polynomial_by_linear(&numerator, &roots[other].neg(), q_degree);
            denominator = denominator.mul(&roots[branch].sub(&roots[other]));
        }
        let denominator_inv = denominator.inverse()?;
        for (row, coefficient) in numerator.into_iter().enumerate() {
            transition_to_flat[row][branch] = coefficient.mul(&denominator_inv);
        }
        metric_norms.push(denominator.inverse()?);
        inverse_metric_norms.push(denominator);
    }

    Ok(CanonicalFrame {
        flat_to_canonical: canonical_evaluation_matrix(&roots),
        transition_to_flat: SeriesMatrix::from_entries(transition_to_flat),
        roots,
        metric_norms,
        inverse_metric_norms,
    })
}
