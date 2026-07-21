//! Twisted calibration: the quantum relation, specialized canonical data,
//! Birkhoff/relation calibration skeletons and candidates, and the twisted
//! Frobenius / quantum-product / R-matrix machinery they are solved from.

use super::hypergeometric::{
    NegativeSplitEquivariantHypergeometricModel, NegativeSplitLineHypergeometricModel,
};
use super::numeric::constant_matrix_at_q_degree;
use super::provider::{TwistedCalibrationMode, TwistedCalibrationValidation};
use super::twist::NegativeSplitBundleTwist;
use crate::core::algebra::{Coeff, RatFun, Rational};
use crate::core::error::GwError;
use crate::core::series::{
    compose_plain_series, integrate_q_derivative_zero_constant_matrix, QSeries, SeriesMatrix,
};
use crate::givental::{
    bernoulli_asymptotic_coefficient, exp_scalar_z_series, solve_r_coefficients_from_flatness,
    CalibrationId, CanonicalFrameConvention, SemisimpleCalibration, SeriesRMatrix, SeriesSMatrix,
};
use crate::reconstruction::{
    canonical_evaluation_matrix_local, derivative_qseries_polynomial_coefficients,
    determinant_qseries_polynomial_matrix, evaluate_qseries_polynomial,
    multiply_qseries_polynomial_by_affine, multiply_qseries_polynomial_by_linear,
    relative_sqrt_delta_series_coeff, relative_sqrt_delta_series_local, series_matrix_scale,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwistedQuantumRelation {
    pub n: usize,
    pub twist: NegativeSplitBundleTwist,
    pub weights: Vec<Rational>,
}

impl TwistedQuantumRelation {
    pub fn new(
        n: usize,
        twist: NegativeSplitBundleTwist,
        weights: Vec<Rational>,
    ) -> Result<Self, GwError> {
        if weights.len() != n + 1 {
            return Err(GwError::AlgebraFailure(format!(
                "expected {} lambda weights, got {}",
                n + 1,
                weights.len()
            )));
        }
        Ok(Self { n, twist, weights })
    }

    pub fn coefficients(&self, q_degree: usize) -> Vec<QSeries> {
        debug_assert_eq!(
            self.twist.degree_sum(),
            self.n + 1,
            "current twisted relation builder is for local Calabi-Yau rank"
        );
        let mut base = vec![QSeries::one(q_degree)];
        for weight in &self.weights {
            base = multiply_qseries_polynomial_by_linear(
                &base,
                &QSeries::constant(RatFun::from_rational(-weight.clone()), q_degree),
                q_degree,
            );
        }

        let mut fiber = vec![QSeries::one(q_degree)];
        for degree in self.twist.degrees() {
            let factor = RatFun::from_rational(-Rational::from(*degree));
            for _ in 0..*degree {
                fiber = multiply_qseries_polynomial_by_linear(
                    &fiber,
                    &QSeries::zero(q_degree),
                    q_degree,
                );
                for coeff in &mut fiber {
                    *coeff = coeff.scale(&factor);
                }
            }
        }

        let size = self.n + 2;
        let mut out = vec![QSeries::zero(q_degree); size.max(fiber.len())];
        for (power, coeff) in base.into_iter().enumerate() {
            out[power] = out[power].add(&coeff);
        }
        let q = QSeries::q(q_degree);
        for (power, coeff) in fiber.into_iter().enumerate() {
            out[power] = out[power].sub(&q.mul(&coeff));
        }
        out.truncate(size);
        out
    }

    pub fn multiplication_matrix(&self, q_degree: usize) -> Result<SeriesMatrix, GwError> {
        if self.twist.degree_sum() != self.n + 1 {
            return Err(GwError::UnsupportedInvariant(
                "twisted multiplication matrix is currently implemented for local Calabi-Yau split bundles only"
                    .to_string(),
            ));
        }
        let coefficients = self.coefficients(q_degree);
        let size = self.n + 1;
        let leading = coefficients[size].inverse()?;
        let mut entries = vec![vec![QSeries::zero(q_degree); size]; size];
        for col in 0..self.n {
            entries[col + 1][col] = QSeries::one(q_degree);
        }
        for row in 0..size {
            entries[row][self.n] = coefficients[row].mul(&leading).neg();
        }
        Ok(SeriesMatrix::from_entries(entries))
    }

    /// Builds the formal S-like solution of the principal quantum relation.
    ///
    /// This is useful for algebra diagnostics, but it is not the calibrated
    /// twisted descendant S-matrix used by the provider.  The provider uses
    /// the hypergeometric mirror/Birkhoff path instead.
    pub fn diagnostic_relation_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        let size = self.n + 1;
        let quantum_h = self.multiplication_matrix(q_degree)?;
        let classical = Self {
            n: self.n,
            twist: self.twist.clone(),
            weights: self.weights.clone(),
        };
        let classical_h = classical.multiplication_matrix(0)?;
        let classical_h = constant_matrix_at_q_degree(&classical_h, q_degree);
        let mut coefficients = Vec::with_capacity(z_order + 1);
        coefficients.push(SeriesMatrix::identity(size, q_degree));

        for order in 1..=z_order {
            let previous = &coefficients[order - 1];
            let source = quantum_h.mul(previous).sub(&previous.mul(&classical_h));
            coefficients.push(integrate_q_derivative_zero_constant_matrix(&source)?);
        }

        SeriesSMatrix::from_coefficients(
            size,
            q_degree,
            z_order,
            coefficients,
            CalibrationId("negative-split-local-cy-relation-diagnostic".to_string()),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecializedTwistedCanonicalData {
    pub roots: Vec<QSeries>,
    pub metric_norms: Vec<QSeries>,
    pub inverse_metric_norms: Vec<QSeries>,
    pub transition_to_flat: Vec<Vec<QSeries>>,
    pub relation_derivatives: Vec<QSeries>,
    pub fiber_eulers: Vec<QSeries>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecializedTwistedBirkhoffCanonicalData<C = RatFun> {
    pub roots: Vec<QSeries<C>>,
    pub metric_norms: Vec<QSeries<C>>,
    pub inverse_metric_norms: Vec<QSeries<C>>,
    pub transition_to_flat: Vec<Vec<QSeries<C>>>,
    pub flat_to_canonical: Vec<Vec<QSeries<C>>>,
    pub quantum_h: SeriesMatrix<C>,
}

/// Inverts an idempotent transition through the Frobenius pairing.
///
/// If the columns of `transition` are the canonical idempotents, Frobenius
/// orthogonality says
///
/// `transition^T flat_metric transition = diag(metric_norms)`.
///
/// Therefore its inverse is exactly
///
/// `diag(metric_norms^{-1}) transition^T flat_metric`.
///
/// This form avoids a second symbolic Gaussian elimination after the
/// canonical metric and its diagonal inverse have already been computed.
fn frobenius_adjoint_transition_inverse<C: Coeff>(
    transition: &SeriesMatrix<C>,
    flat_metric: &SeriesMatrix<C>,
    inverse_metric_norms: &[QSeries<C>],
) -> Result<SeriesMatrix<C>, GwError> {
    let size = transition.rows();
    let q_degree = transition.max_degree();
    if transition.cols() != size
        || flat_metric.rows() != size
        || flat_metric.cols() != size
        || flat_metric.max_degree() != q_degree
        || inverse_metric_norms.len() != size
        || inverse_metric_norms
            .iter()
            .any(|norm| norm.max_degree() != q_degree)
    {
        return Err(GwError::ConventionMismatch(
            "canonical transition, flat metric, and norm truncations do not agree".to_string(),
        ));
    }

    let mut entries = vec![vec![QSeries::<C>::zero(q_degree); size]; size];
    for branch in 0..size {
        for flat_col in 0..size {
            let mut pairing = QSeries::<C>::zero(q_degree);
            for flat_row in 0..size {
                pairing.add_product_assign(
                    transition.entry(flat_row, branch),
                    flat_metric.entry(flat_row, flat_col),
                );
            }
            entries[branch][flat_col] = inverse_metric_norms[branch].mul(&pairing);
        }
    }
    Ok(SeriesMatrix::from_entries(entries))
}

pub fn specialized_twisted_birkhoff_canonical_data(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    max_q_degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SpecializedTwistedBirkhoffCanonicalData, GwError> {
    // Semisimple data extracted from the Birkhoff-normalized quantum
    // multiplication operator.  This is the twisted analogue of finding
    // canonical roots/idempotents for ordinary P^n.
    specialized_twisted_birkhoff_canonical_data_with_mode(
        n,
        twist,
        max_q_degree,
        base_weights,
        fiber_weights,
        TwistedCalibrationMode::InverseEuler,
    )
}

pub(crate) fn specialized_twisted_birkhoff_canonical_data_with_mode(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    max_q_degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    mode: TwistedCalibrationMode,
) -> Result<SpecializedTwistedBirkhoffCanonicalData, GwError> {
    specialized_twisted_birkhoff_canonical_data_with_mode_and_validation(
        n,
        twist,
        max_q_degree,
        base_weights,
        fiber_weights,
        mode,
        TwistedCalibrationValidation::Full,
    )
}

pub(crate) fn specialized_twisted_birkhoff_canonical_data_with_mode_and_validation(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    max_q_degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    mode: TwistedCalibrationMode,
    validation: TwistedCalibrationValidation,
) -> Result<SpecializedTwistedBirkhoffCanonicalData, GwError> {
    // Build quantum multiplication by H from either Picard-Fuchs data or from
    // the Birkhoff S-matrix, then diagonalize it.  Full validation checks
    // self-adjointness and diagonalization of the twisted flat pairing; the fast
    // graph path skips those identities after they have been covered by tests.
    validate_twisted_weights(n, twist, base_weights, fiber_weights)?;
    let model = NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
        n,
        twist.clone(),
        max_q_degree,
        1,
        base_weights.to_vec(),
        fiber_weights.to_vec(),
    )?;
    let quantum_h = match mode {
        TwistedCalibrationMode::Euler => twisted_quantum_multiplication_from_picard_fuchs(
            n,
            twist,
            max_q_degree,
            base_weights,
            fiber_weights,
            &model.mirror_map_coefficients()?,
            &model.inverse_mirror_map_coefficients()?,
        )?,
        TwistedCalibrationMode::InverseEuler | TwistedCalibrationMode::InverseEulerFiberPlus => {
            let descendant_s = model.birkhoff_descendant_s_matrix(1)?;
            let classical_h =
                twisted_classical_h_multiplication_matrix(n, max_q_degree, base_weights)?;
            twisted_quantum_multiplication_from_s(&descendant_s, &classical_h, &mode)?
        }
    };
    let flat_metric = match mode {
        TwistedCalibrationMode::InverseEuler | TwistedCalibrationMode::InverseEulerFiberPlus => {
            twisted_inverse_euler_flat_metric_matrix(
                n,
                max_q_degree,
                twist,
                base_weights,
                fiber_weights,
            )?
        }
        TwistedCalibrationMode::Euler => {
            twisted_flat_metric_matrix(n, max_q_degree, twist, base_weights, fiber_weights)?
        }
    };

    if validation.runs_expensive_checks() {
        let self_adjoint_defect = quantum_h
            .transpose()
            .mul(&flat_metric)
            .sub(&flat_metric.mul(&quantum_h));
        if !self_adjoint_defect.is_zero() {
            return Err(GwError::ValidationFailure(
                "Birkhoff quantum multiplication is not self-adjoint for the twisted flat pairing"
                    .to_string(),
            ));
        }
    }

    let charpoly = charpoly_qseries_coefficients(&quantum_h)?;
    let roots = (0..=n)
        .map(|branch| {
            root_series_from_charpoly(&charpoly, base_weights[branch].clone(), max_q_degree)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transition = spectral_transition_matrix_from_roots(&quantum_h, &roots)?;
    let transition_to_flat = transition.entries().to_vec();
    let canonical_metric = transition.transpose().mul(&flat_metric).mul(&transition);

    let mut metric_norms = Vec::with_capacity(n + 1);
    let mut inverse_metric_norms = Vec::with_capacity(n + 1);
    for row in 0..=n {
        if validation.runs_expensive_checks() {
            for col in 0..=n {
                if row != col && !canonical_metric.entry(row, col).is_zero() {
                    return Err(GwError::ValidationFailure(
                        "Birkhoff idempotents do not diagonalize the twisted flat pairing"
                            .to_string(),
                    ));
                }
            }
        }
        let norm = canonical_metric.entry(row, row).clone();
        inverse_metric_norms.push(norm.inverse()?);
        metric_norms.push(norm);
    }
    let flat_to_canonical =
        frobenius_adjoint_transition_inverse(&transition, &flat_metric, &inverse_metric_norms)?
            .entries()
            .to_vec();

    Ok(SpecializedTwistedBirkhoffCanonicalData {
        roots,
        metric_norms,
        inverse_metric_norms,
        transition_to_flat,
        flat_to_canonical,
        quantum_h,
    })
}

pub(crate) fn specialized_twisted_birkhoff_canonical_data_for_coeff_weights_with_validation<
    C: Coeff,
>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    max_q_degree: usize,
    base_weights: &[C],
    fiber_weights: &[C],
    validation: TwistedCalibrationValidation,
) -> Result<SpecializedTwistedBirkhoffCanonicalData<C>, GwError> {
    let model = NegativeSplitLineHypergeometricModel::<C>::from_coeff_weights(
        n,
        twist.clone(),
        max_q_degree,
        1,
        base_weights.to_vec(),
        fiber_weights,
    )?;
    let descendant_s = model.birkhoff_descendant_s_matrix(1)?;
    let classical_h =
        twisted_classical_h_multiplication_matrix_coeff(n, max_q_degree, &model.base_weights)?;
    let quantum_h = twisted_quantum_multiplication_from_s_coeff(
        &descendant_s,
        &classical_h,
        &TwistedCalibrationMode::InverseEuler,
    )?;
    let flat_metric = twisted_inverse_euler_flat_metric_matrix_coeff(
        n,
        max_q_degree,
        twist,
        &model.base_weights,
        &model.fiber_weights,
    )?;

    if validation.runs_expensive_checks() {
        let self_adjoint_defect = quantum_h
            .transpose()
            .mul(&flat_metric)
            .sub(&flat_metric.mul(&quantum_h));
        if !self_adjoint_defect.is_zero() {
            return Err(GwError::ValidationFailure(
                "Birkhoff quantum multiplication is not self-adjoint for the twisted lambda-line pairing"
                    .to_string(),
            ));
        }
    }

    let charpoly = charpoly_qseries_coefficients_coeff(&quantum_h)?;
    let roots = (0..=n)
        .map(|branch| {
            root_series_from_charpoly_coeff(
                &charpoly,
                model.base_weights[branch].clone(),
                max_q_degree,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transition = spectral_transition_matrix_from_roots_coeff(&quantum_h, &roots)?;
    let transition_to_flat = transition.entries().to_vec();
    let canonical_metric = transition.transpose().mul(&flat_metric).mul(&transition);

    let mut metric_norms = Vec::with_capacity(n + 1);
    let mut inverse_metric_norms = Vec::with_capacity(n + 1);
    for row in 0..=n {
        if validation.runs_expensive_checks() {
            for col in 0..=n {
                if row != col && !canonical_metric.entry(row, col).is_zero() {
                    return Err(GwError::ValidationFailure(
                        "Birkhoff idempotents do not diagonalize the twisted lambda-line pairing"
                            .to_string(),
                    ));
                }
            }
        }
        let norm = canonical_metric.entry(row, row).clone();
        inverse_metric_norms.push(norm.inverse()?);
        metric_norms.push(norm);
    }
    let flat_to_canonical =
        frobenius_adjoint_transition_inverse(&transition, &flat_metric, &inverse_metric_norms)?
            .entries()
            .to_vec();

    Ok(SpecializedTwistedBirkhoffCanonicalData {
        roots,
        metric_norms,
        inverse_metric_norms,
        transition_to_flat,
        flat_to_canonical,
        quantum_h,
    })
}

pub fn negative_split_twisted_birkhoff_calibration_skeleton(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SemisimpleCalibration, GwError> {
    // Calibration without a nontrivial R-matrix.  This is useful for isolating
    // whether an error is in the Birkhoff/Psi side or in the R-recursion.
    negative_split_twisted_birkhoff_calibration_skeleton_with_mode(
        n,
        twist,
        q_degree,
        z_order,
        base_weights,
        fiber_weights,
        TwistedCalibrationMode::InverseEuler,
    )
}

pub(crate) fn negative_split_twisted_birkhoff_calibration_skeleton_with_mode(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    mode: TwistedCalibrationMode,
) -> Result<SemisimpleCalibration, GwError> {
    let canonical = specialized_twisted_birkhoff_canonical_data_with_mode(
        n,
        twist,
        q_degree,
        base_weights,
        fiber_weights,
        mode,
    )?;
    negative_split_twisted_birkhoff_calibration_skeleton_from_canonical(
        q_degree, z_order, &canonical,
    )
}

pub(crate) fn negative_split_twisted_birkhoff_calibration_skeleton_from_canonical(
    q_degree: usize,
    z_order: usize,
    canonical: &SpecializedTwistedBirkhoffCanonicalData,
) -> Result<SemisimpleCalibration, GwError> {
    negative_split_twisted_birkhoff_calibration_skeleton_from_canonical_coeff(
        q_degree, z_order, canonical,
    )
}

pub(crate) fn negative_split_twisted_birkhoff_calibration_skeleton_from_canonical_coeff<
    C: Coeff,
>(
    q_degree: usize,
    z_order: usize,
    canonical: &SpecializedTwistedBirkhoffCanonicalData<C>,
) -> Result<SemisimpleCalibration<C>, GwError> {
    let size = canonical.roots.len();
    let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());
    let relative_sqrt_delta = canonical
        .inverse_metric_norms
        .iter()
        .map(relative_sqrt_delta_series_coeff)
        .collect::<Result<Vec<_>, _>>()?;
    let relative_sqrt_delta_inv = relative_sqrt_delta
        .iter()
        .map(QSeries::inverse)
        .collect::<Result<Vec<_>, _>>()?;

    let relative_scale = SeriesMatrix::diagonal(relative_sqrt_delta.clone());
    let relative_scale_inv = SeriesMatrix::diagonal(relative_sqrt_delta_inv.clone());
    let transition_inverse = SeriesMatrix::from_entries(canonical.flat_to_canonical.clone());
    let psi = transition.mul(&relative_scale);
    let psi_inverse = relative_scale_inv.mul(&transition_inverse);
    let connection = psi_inverse.mul(&psi.q_derivative());
    let metric = SeriesMatrix::diagonal(
        canonical
            .metric_norms
            .iter()
            .map(|norm| QSeries::constant(norm.coeff(0).cloned().unwrap_or_else(C::zero), q_degree))
            .collect(),
    );

    Ok(SemisimpleCalibration {
        r_matrix: SeriesRMatrix::<C>::identity(
            size,
            q_degree,
            z_order,
            CanonicalFrameConvention::RelativeNormalizedCanonicalIdempotents,
        ),
        metric,
        psi,
        psi_inverse,
        connection,
        delta: canonical.inverse_metric_norms.clone(),
        inverse_delta: canonical.metric_norms.clone(),
        relative_sqrt_delta,
        relative_sqrt_delta_inverse: relative_sqrt_delta_inv,
    })
}

pub fn negative_split_twisted_birkhoff_calibration_candidate(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SemisimpleCalibration, GwError> {
    // Full candidate calibration: Birkhoff canonical data plus the QRR
    // Bernoulli classical limit used to integrate the R-matrix flatness
    // equation.
    negative_split_twisted_birkhoff_calibration_candidate_with_mode(
        n,
        twist,
        q_degree,
        z_order,
        base_weights,
        fiber_weights,
        TwistedCalibrationMode::InverseEuler,
    )
}

pub(crate) fn negative_split_twisted_birkhoff_calibration_candidate_with_mode(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    mode: TwistedCalibrationMode,
) -> Result<SemisimpleCalibration, GwError> {
    negative_split_twisted_birkhoff_calibration_candidate_with_mode_and_validation(
        n,
        twist,
        q_degree,
        z_order,
        base_weights,
        fiber_weights,
        mode,
        TwistedCalibrationValidation::Full,
    )
}

pub(crate) fn negative_split_twisted_birkhoff_calibration_candidate_with_mode_and_validation(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    mode: TwistedCalibrationMode,
    validation: TwistedCalibrationValidation,
) -> Result<SemisimpleCalibration, GwError> {
    let canonical = specialized_twisted_birkhoff_canonical_data_with_mode_and_validation(
        n,
        twist,
        q_degree,
        base_weights,
        fiber_weights,
        mode.clone(),
        validation,
    )?;
    let mut calibration = negative_split_twisted_birkhoff_calibration_skeleton_from_canonical(
        q_degree, z_order, &canonical,
    )?;
    let classical_diagonal = twisted_classical_limit_diagonal_coefficients_with_mode(
        n,
        twist,
        z_order,
        base_weights,
        fiber_weights,
        mode,
    )?;
    let coefficients = solve_r_coefficients_from_flatness(
        &canonical.roots,
        &calibration.connection,
        &classical_diagonal,
        q_degree,
        z_order,
    )?;
    calibration.r_matrix = SeriesRMatrix::from_coefficients(
        n + 1,
        q_degree,
        z_order,
        coefficients,
        CalibrationId("negative-split-equivariant-birkhoff-qrr-candidate".to_string()),
        CanonicalFrameConvention::RelativeNormalizedCanonicalIdempotents,
    )?;
    if validation.runs_expensive_checks() {
        calibration.r_matrix.check_unitarity(&calibration.metric)?;
    }
    Ok(calibration)
}

pub(crate) fn negative_split_twisted_birkhoff_calibration_candidate_for_ratfun_weights_with_validation(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[RatFun],
    fiber_weights: &[RatFun],
    validation: TwistedCalibrationValidation,
) -> Result<SemisimpleCalibration, GwError> {
    negative_split_twisted_birkhoff_calibration_candidate_for_coeff_weights_with_validation(
        n,
        twist,
        q_degree,
        z_order,
        base_weights,
        fiber_weights,
        validation,
    )
}

pub(crate) fn negative_split_twisted_birkhoff_calibration_candidate_for_coeff_weights_with_validation<
    C: Coeff,
>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[C],
    fiber_weights: &[C],
    validation: TwistedCalibrationValidation,
) -> Result<SemisimpleCalibration<C>, GwError> {
    let canonical = specialized_twisted_birkhoff_canonical_data_for_coeff_weights_with_validation(
        n,
        twist,
        q_degree,
        base_weights,
        fiber_weights,
        validation,
    )?;
    let mut calibration =
        negative_split_twisted_birkhoff_calibration_skeleton_from_canonical_coeff(
            q_degree, z_order, &canonical,
        )?;
    let classical_diagonal = twisted_classical_limit_diagonal_coefficients_coeff(
        n,
        twist,
        z_order,
        base_weights,
        fiber_weights,
    )?;
    let coefficients = solve_r_coefficients_from_flatness(
        &canonical.roots,
        &calibration.connection,
        &classical_diagonal,
        q_degree,
        z_order,
    )?;
    calibration.r_matrix = SeriesRMatrix::<C>::from_coefficients(
        n + 1,
        q_degree,
        z_order,
        coefficients,
        CalibrationId("negative-split-ratfun-birkhoff-qrr-candidate".to_string()),
        CanonicalFrameConvention::RelativeNormalizedCanonicalIdempotents,
    )?;
    if validation.runs_expensive_checks() {
        calibration.r_matrix.check_unitarity(&calibration.metric)?;
    }

    Ok(calibration)
}

pub fn negative_split_twisted_relation_calibration_skeleton(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SemisimpleCalibration, GwError> {
    let canonical = specialized_twisted_quantum_canonical_data(
        n,
        twist,
        q_degree,
        base_weights,
        fiber_weights,
    )?;
    let size = n + 1;
    let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());
    let relative_sqrt_delta = canonical
        .inverse_metric_norms
        .iter()
        .map(relative_sqrt_delta_series_local)
        .collect::<Result<Vec<_>, _>>()?;
    let relative_sqrt_delta_inv = relative_sqrt_delta
        .iter()
        .map(QSeries::inverse)
        .collect::<Result<Vec<_>, _>>()?;

    let relative_scale = SeriesMatrix::diagonal(relative_sqrt_delta.clone());
    let relative_scale_inv = SeriesMatrix::diagonal(relative_sqrt_delta_inv.clone());
    let evaluation = canonical_evaluation_matrix_local(&canonical.roots);
    let psi = transition.mul(&relative_scale);
    let psi_inverse = relative_scale_inv.mul(&evaluation);
    let connection = psi_inverse.mul(&psi.q_derivative());
    let metric = SeriesMatrix::diagonal(
        canonical
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

    Ok(SemisimpleCalibration {
        r_matrix: SeriesRMatrix::identity(
            size,
            q_degree,
            z_order,
            CanonicalFrameConvention::RelativeNormalizedCanonicalIdempotents,
        ),
        metric,
        psi,
        psi_inverse,
        connection,
        delta: canonical.inverse_metric_norms,
        inverse_delta: canonical.metric_norms,
        relative_sqrt_delta,
        relative_sqrt_delta_inverse: relative_sqrt_delta_inv,
    })
}

/// Candidate twisted calibration from the principal relation and Euler/QRR
/// Bernoulli diagonal gauge.
///
/// This is intentionally not wired into `TwistedProjectiveSpaceProvider` yet.
/// It is a validation target for the remaining R-matrix convention work.
pub fn negative_split_twisted_relation_calibration_candidate(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SemisimpleCalibration, GwError> {
    let calibration = negative_split_twisted_relation_calibration_raw_candidate(
        n,
        twist,
        q_degree,
        z_order,
        base_weights,
        fiber_weights,
    )?;
    calibration.r_matrix.check_unitarity(&calibration.metric)?;
    Ok(calibration)
}

pub fn negative_split_twisted_relation_calibration_raw_candidate(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SemisimpleCalibration, GwError> {
    let canonical = specialized_twisted_quantum_canonical_data(
        n,
        twist,
        q_degree,
        base_weights,
        fiber_weights,
    )?;
    let size = n + 1;
    let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());
    let relative_sqrt_delta = canonical
        .inverse_metric_norms
        .iter()
        .map(relative_sqrt_delta_series_local)
        .collect::<Result<Vec<_>, _>>()?;
    let relative_sqrt_delta_inv = relative_sqrt_delta
        .iter()
        .map(QSeries::inverse)
        .collect::<Result<Vec<_>, _>>()?;

    let relative_scale = SeriesMatrix::diagonal(relative_sqrt_delta.clone());
    let relative_scale_inv = SeriesMatrix::diagonal(relative_sqrt_delta_inv.clone());
    let evaluation = canonical_evaluation_matrix_local(&canonical.roots);
    let psi = transition.mul(&relative_scale);
    let psi_inverse = relative_scale_inv.mul(&evaluation);
    let connection = psi_inverse.mul(&psi.q_derivative());
    let metric = SeriesMatrix::diagonal(
        canonical
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

    let classical_diagonal = twisted_classical_limit_diagonal_coefficients(
        n,
        twist,
        z_order,
        base_weights,
        fiber_weights,
    )?;
    let coefficients = solve_r_coefficients_from_flatness(
        &canonical.roots,
        &connection,
        &classical_diagonal,
        q_degree,
        z_order,
    )?;
    let r_matrix = SeriesRMatrix::from_coefficients(
        size,
        q_degree,
        z_order,
        coefficients,
        CalibrationId("negative-split-relation-qrr-candidate".to_string()),
        CanonicalFrameConvention::RelativeNormalizedCanonicalIdempotents,
    )?;

    Ok(SemisimpleCalibration {
        r_matrix,
        metric,
        psi,
        psi_inverse,
        connection,
        delta: canonical.inverse_metric_norms,
        inverse_delta: canonical.metric_norms,
        relative_sqrt_delta,
        relative_sqrt_delta_inverse: relative_sqrt_delta_inv,
    })
}

pub fn specialized_twisted_quantum_canonical_data(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    max_q_degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SpecializedTwistedCanonicalData, GwError> {
    validate_twisted_local_cy_weights(n, twist, base_weights, fiber_weights)?;

    let roots = (0..=n)
        .map(|branch| {
            twisted_canonical_root_series_at_weights(
                n,
                twist,
                branch,
                max_q_degree,
                base_weights,
                fiber_weights,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let relation_derivatives = roots
        .iter()
        .map(|root| {
            twisted_relation_derivative_series_at_weights(
                n,
                twist,
                root,
                base_weights,
                fiber_weights,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fiber_coefficients =
        twisted_fiber_polynomial_coefficients(twist, max_q_degree, fiber_weights)?;
    let fiber_eulers = roots
        .iter()
        .map(|root| evaluate_qseries_polynomial(&fiber_coefficients, root))
        .collect::<Vec<_>>();

    let mut metric_norms = Vec::with_capacity(n + 1);
    let mut inverse_metric_norms = Vec::with_capacity(n + 1);
    let mut transition_to_flat = vec![vec![QSeries::zero(max_q_degree); n + 1]; n + 1];
    let flat_metric =
        twisted_flat_metric_matrix(n, max_q_degree, twist, base_weights, fiber_weights)?;

    for branch in 0..=n {
        let mut numerator = vec![QSeries::one(max_q_degree)];
        let mut denominator = QSeries::one(max_q_degree);
        for other in 0..=n {
            if other == branch {
                continue;
            }
            numerator = multiply_qseries_polynomial_by_linear(
                &numerator,
                &roots[other].neg(),
                max_q_degree,
            );
            denominator = denominator.mul(&roots[branch].sub(&roots[other]));
        }
        let denominator_inv = denominator.inverse()?;
        for (row, coeff) in numerator.into_iter().enumerate() {
            transition_to_flat[row][branch] = coeff.mul(&denominator_inv);
        }

        let metric_norm =
            canonical_metric_norm_from_flat_metric(&transition_to_flat, branch, &flat_metric);
        let inverse_metric_norm = metric_norm.inverse()?;
        metric_norms.push(metric_norm);
        inverse_metric_norms.push(inverse_metric_norm);
    }

    Ok(SpecializedTwistedCanonicalData {
        roots,
        metric_norms,
        inverse_metric_norms,
        transition_to_flat,
        relation_derivatives,
        fiber_eulers,
    })
}

pub(crate) fn validate_twisted_local_cy_weights(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<(), GwError> {
    validate_twisted_weights(n, twist, base_weights, fiber_weights)?;
    if twist.degree_sum() != n + 1 {
        return Err(GwError::UnsupportedInvariant(
            "twisted canonical relation skeleton currently supports local Calabi-Yau split bundles only"
                .to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_twisted_weights(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<(), GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    if fiber_weights.len() != twist.rank() {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} fiber weights, got {}",
            twist.rank(),
            fiber_weights.len()
        )));
    }
    Ok(())
}

pub(crate) fn twisted_canonical_root_series_at_weights(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    branch: usize,
    max_q_degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<QSeries, GwError> {
    let mut root = QSeries::constant(
        RatFun::from_rational(base_weights[branch].clone()),
        max_q_degree,
    );
    for _ in 0..=max_q_degree {
        let p = twisted_relation_series_at_weights(n, twist, &root, base_weights, fiber_weights)?;
        if p.coeffs().iter().all(RatFun::is_zero) {
            break;
        }
        let dp = twisted_relation_derivative_series_at_weights(
            n,
            twist,
            &root,
            base_weights,
            fiber_weights,
        )?;
        root = root.sub(&p.div(&dp)?);
    }
    Ok(root)
}

pub(crate) fn twisted_relation_series_at_weights(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    x: &QSeries,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<QSeries, GwError> {
    let q_degree = x.max_degree();
    let base_coefficients = twisted_base_polynomial_coefficients(n, q_degree, base_weights)?;
    let fiber_coefficients = twisted_fiber_polynomial_coefficients(twist, q_degree, fiber_weights)?;
    Ok(evaluate_qseries_polynomial(&base_coefficients, x)
        .sub(&QSeries::q(q_degree).mul(&evaluate_qseries_polynomial(&fiber_coefficients, x))))
}

pub(crate) fn twisted_relation_derivative_series_at_weights(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    x: &QSeries,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<QSeries, GwError> {
    let q_degree = x.max_degree();
    let base_coefficients = derivative_qseries_polynomial_coefficients(
        &twisted_base_polynomial_coefficients(n, q_degree, base_weights)?,
        q_degree,
    );
    let fiber_coefficients = derivative_qseries_polynomial_coefficients(
        &twisted_fiber_polynomial_coefficients(twist, q_degree, fiber_weights)?,
        q_degree,
    );
    Ok(evaluate_qseries_polynomial(&base_coefficients, x)
        .sub(&QSeries::q(q_degree).mul(&evaluate_qseries_polynomial(&fiber_coefficients, x))))
}

pub(crate) fn twisted_base_polynomial_coefficients(
    n: usize,
    q_degree: usize,
    base_weights: &[Rational],
) -> Result<Vec<QSeries>, GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    let mut out = vec![QSeries::one(q_degree)];
    for weight in base_weights {
        out = multiply_qseries_polynomial_by_linear(
            &out,
            &QSeries::constant(RatFun::from_rational(-weight.clone()), q_degree),
            q_degree,
        );
    }
    Ok(out)
}

pub(crate) fn twisted_base_polynomial_coefficients_coeff<C: Coeff>(
    n: usize,
    q_degree: usize,
    base_weights: &[C],
) -> Result<Vec<QSeries<C>>, GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    let mut out = vec![QSeries::<C>::one(q_degree)];
    for weight in base_weights {
        out = multiply_qseries_polynomial_by_linear(
            &out,
            &QSeries::constant(weight.neg(), q_degree),
            q_degree,
        );
    }
    Ok(out)
}

pub(crate) fn twisted_fiber_polynomial_coefficients(
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    fiber_weights: &[Rational],
) -> Result<Vec<QSeries>, GwError> {
    if fiber_weights.len() != twist.rank() {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} fiber weights, got {}",
            twist.rank(),
            fiber_weights.len()
        )));
    }
    let mut out = vec![QSeries::one(q_degree)];
    for (degree, weight) in twist.degrees().iter().zip(fiber_weights) {
        for _ in 0..*degree {
            out = multiply_qseries_polynomial_by_affine(
                &out,
                &QSeries::constant(RatFun::from_rational(weight.clone()), q_degree),
                &QSeries::constant(RatFun::from_rational(-Rational::from(*degree)), q_degree),
                q_degree,
            );
        }
    }
    Ok(out)
}

pub(crate) fn twisted_flat_metric_matrix(
    n: usize,
    q_degree: usize,
    twist: &NegativeSplitBundleTwist,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SeriesMatrix, GwError> {
    let mut entries = vec![vec![QSeries::zero(q_degree); n + 1]; n + 1];
    for a in 0..=n {
        for b in 0..=n {
            let mut value = Rational::zero();
            for branch in 0..=n {
                let lambda = base_weights[branch].clone();
                let mut tangent = Rational::one();
                for (other, weight) in base_weights.iter().enumerate() {
                    if other != branch {
                        tangent = tangent * (lambda.clone() - weight.clone());
                    }
                }
                if tangent.is_zero() {
                    return Err(GwError::NonSemisimplePoint);
                }
                let fiber = twisted_fiber_euler_at_fixed_point(twist, fiber_weights, &lambda);
                value += lambda.pow_usize(a + b) * fiber / tangent;
            }
            entries[a][b] = QSeries::constant(RatFun::from_rational(value), q_degree);
        }
    }
    Ok(SeriesMatrix::from_entries(entries))
}

pub(crate) fn twisted_inverse_euler_flat_metric_matrix(
    n: usize,
    q_degree: usize,
    twist: &NegativeSplitBundleTwist,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SeriesMatrix, GwError> {
    let mut entries = vec![vec![QSeries::zero(q_degree); n + 1]; n + 1];
    for a in 0..=n {
        for b in 0..=n {
            let mut value = Rational::zero();
            for branch in 0..=n {
                let lambda = base_weights[branch].clone();
                let mut tangent = Rational::one();
                for (other, weight) in base_weights.iter().enumerate() {
                    if other != branch {
                        tangent = tangent * (lambda.clone() - weight.clone());
                    }
                }
                if tangent.is_zero() {
                    return Err(GwError::NonSemisimplePoint);
                }
                let fiber = twisted_fiber_euler_at_fixed_point(twist, fiber_weights, &lambda);
                if fiber.is_zero() {
                    return Err(GwError::NonSemisimplePoint);
                }
                value += lambda.pow_usize(a + b) / (tangent * fiber);
            }
            entries[a][b] = QSeries::constant(RatFun::from_rational(value), q_degree);
        }
    }
    Ok(SeriesMatrix::from_entries(entries))
}

pub(crate) fn twisted_inverse_euler_flat_metric_matrix_ratfun(
    n: usize,
    q_degree: usize,
    twist: &NegativeSplitBundleTwist,
    base_weights: &[RatFun],
    fiber_weights: &[RatFun],
) -> Result<SeriesMatrix, GwError> {
    twisted_inverse_euler_flat_metric_matrix_coeff(n, q_degree, twist, base_weights, fiber_weights)
}

pub(crate) fn twisted_inverse_euler_flat_metric_matrix_coeff<C: Coeff>(
    n: usize,
    q_degree: usize,
    twist: &NegativeSplitBundleTwist,
    base_weights: &[C],
    fiber_weights: &[C],
) -> Result<SeriesMatrix<C>, GwError> {
    let mut entries = vec![vec![QSeries::<C>::zero(q_degree); n + 1]; n + 1];
    for a in 0..=n {
        for b in 0..=n {
            let mut value = C::zero();
            for branch in 0..=n {
                let lambda = base_weights[branch].clone();
                let mut tangent = C::one();
                for (other, weight) in base_weights.iter().enumerate() {
                    if other != branch {
                        tangent = tangent.mul(&lambda.sub(weight));
                    }
                }
                if tangent.is_zero() {
                    return Err(GwError::NonSemisimplePoint);
                }
                let fiber = twisted_fiber_euler_at_fixed_point_coeff(twist, fiber_weights, &lambda);
                if fiber.is_zero() {
                    return Err(GwError::NonSemisimplePoint);
                }
                value = value.add(&lambda.pow_usize(a + b).div(&tangent.mul(&fiber)));
            }
            entries[a][b] = QSeries::constant(value, q_degree);
        }
    }
    Ok(SeriesMatrix::from_entries(entries))
}

pub(crate) fn twisted_inverse_euler_flat_metric_pair_from_rational_base<C: Coeff>(
    n: usize,
    q_degree: usize,
    twist: &NegativeSplitBundleTwist,
    base_weights: &[Rational],
    fiber_weights: &[C],
) -> Result<(SeriesMatrix<C>, SeriesMatrix<C>), GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    if fiber_weights.len() != twist.rank() {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} fiber weights, got {}",
            twist.rank(),
            fiber_weights.len()
        )));
    }

    let mut metric_entries = vec![vec![QSeries::<C>::zero(q_degree); n + 1]; n + 1];
    let mut inverse_entries = vec![vec![QSeries::<C>::zero(q_degree); n + 1]; n + 1];
    for branch in 0..=n {
        let lambda = &base_weights[branch];
        let mut tangent = Rational::one();
        for (other, weight) in base_weights.iter().enumerate() {
            if other != branch {
                tangent = tangent * (lambda.clone() - weight.clone());
            }
        }
        if tangent.is_zero() {
            return Err(GwError::NonSemisimplePoint);
        }
        let fiber_lambda = C::from_rational(lambda.clone());
        let fiber = twisted_fiber_euler_at_fixed_point_coeff(twist, fiber_weights, &fiber_lambda);
        if fiber.is_zero() {
            return Err(GwError::NonSemisimplePoint);
        }
        let lagrange = lagrange_basis_coefficients(branch, base_weights)?;

        for a in 0..=n {
            for b in 0..=n {
                let metric_scalar = lambda.pow_usize(a + b) / tangent.clone();
                let metric_term = C::from_rational(metric_scalar).div(&fiber);
                let metric_value = metric_entries[a][b]
                    .coeff(0)
                    .cloned()
                    .unwrap_or_else(C::zero)
                    .add(&metric_term);
                metric_entries[a][b] = QSeries::constant(metric_value, q_degree);

                let inverse_scalar = lagrange[a].clone() * lagrange[b].clone() * tangent.clone();
                let inverse_term = C::from_rational(inverse_scalar).mul(&fiber);
                let inverse_value = inverse_entries[a][b]
                    .coeff(0)
                    .cloned()
                    .unwrap_or_else(C::zero)
                    .add(&inverse_term);
                inverse_entries[a][b] = QSeries::constant(inverse_value, q_degree);
            }
        }
    }

    Ok((
        SeriesMatrix::from_entries(metric_entries),
        SeriesMatrix::from_entries(inverse_entries),
    ))
}

pub(crate) fn lagrange_basis_coefficients(
    branch: usize,
    base_weights: &[Rational],
) -> Result<Vec<Rational>, GwError> {
    let n = base_weights.len().saturating_sub(1);
    if branch >= base_weights.len() {
        return Err(GwError::AlgebraFailure(format!(
            "Lagrange branch {branch} out of range for {} base weights",
            base_weights.len()
        )));
    }
    let lambda = &base_weights[branch];
    let mut denominator = Rational::one();
    let mut coefficients = vec![Rational::one()];
    for (other, weight) in base_weights.iter().enumerate() {
        if other == branch {
            continue;
        }
        denominator = denominator * (lambda.clone() - weight.clone());
        let mut next = vec![Rational::zero(); coefficients.len() + 1];
        for (power, coeff) in coefficients.iter().enumerate() {
            next[power] = next[power].clone() - coeff.clone() * weight.clone();
            next[power + 1] = next[power + 1].clone() + coeff.clone();
        }
        coefficients = next;
    }
    if denominator.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    coefficients.resize(n + 1, Rational::zero());
    for coeff in &mut coefficients {
        *coeff = coeff.clone() / denominator.clone();
    }
    Ok(coefficients)
}

pub(crate) fn twisted_fiber_euler_at_fixed_point(
    twist: &NegativeSplitBundleTwist,
    fiber_weights: &[Rational],
    lambda: &Rational,
) -> Rational {
    twisted_fiber_euler_at_fixed_point_coeff(twist, fiber_weights, lambda)
}

pub(crate) fn twisted_fiber_euler_at_fixed_point_coeff<C: Coeff>(
    twist: &NegativeSplitBundleTwist,
    fiber_weights: &[C],
    lambda: &C,
) -> C {
    twist
        .degrees()
        .iter()
        .zip(fiber_weights)
        .fold(C::one(), |acc, (degree, weight)| {
            acc.mul(&weight.sub(&C::from_usize(*degree).mul(lambda)))
        })
}

pub(crate) fn canonical_metric_norm_from_flat_metric(
    transition_to_flat: &[Vec<QSeries>],
    branch: usize,
    flat_metric: &SeriesMatrix,
) -> QSeries {
    let q_degree = flat_metric.max_degree();
    let mut norm = QSeries::zero(q_degree);
    for a in 0..transition_to_flat.len() {
        for b in 0..transition_to_flat.len() {
            let term = transition_to_flat[a][branch]
                .mul(flat_metric.entry(a, b))
                .mul(&transition_to_flat[b][branch]);
            norm = norm.add(&term);
        }
    }
    norm
}

pub(crate) fn twisted_classical_h_multiplication_matrix(
    n: usize,
    q_degree: usize,
    base_weights: &[Rational],
) -> Result<SeriesMatrix, GwError> {
    let coefficients = twisted_base_polynomial_coefficients(n, q_degree, base_weights)?;
    companion_multiplication_matrix_from_monic_polynomial(n + 1, &coefficients)
}

pub(crate) fn twisted_classical_h_multiplication_matrix_coeff<C: Coeff>(
    n: usize,
    q_degree: usize,
    base_weights: &[C],
) -> Result<SeriesMatrix<C>, GwError> {
    let coefficients = twisted_base_polynomial_coefficients_coeff(n, q_degree, base_weights)?;
    companion_multiplication_matrix_from_monic_polynomial(n + 1, &coefficients)
}

pub(crate) fn companion_multiplication_matrix_from_monic_polynomial<C: Coeff>(
    size: usize,
    coefficients: &[QSeries<C>],
) -> Result<SeriesMatrix<C>, GwError> {
    if coefficients.len() != size + 1 {
        return Err(GwError::ConventionMismatch(format!(
            "expected monic polynomial of degree {size}, got degree {}",
            coefficients.len().saturating_sub(1)
        )));
    }
    let q_degree = coefficients
        .first()
        .map(QSeries::max_degree)
        .unwrap_or_default();
    let leading = coefficients[size].inverse()?;
    let mut entries = vec![vec![QSeries::<C>::zero(q_degree); size]; size];
    for col in 0..size.saturating_sub(1) {
        entries[col + 1][col] = QSeries::<C>::one(q_degree);
    }
    for row in 0..size {
        entries[row][size - 1] = coefficients[row].mul(&leading).neg();
    }
    Ok(SeriesMatrix::from_entries(entries))
}

pub(crate) fn twisted_quantum_multiplication_from_s(
    descendant_s: &SeriesSMatrix,
    classical_h: &SeriesMatrix,
    mode: &TwistedCalibrationMode,
) -> Result<SeriesMatrix, GwError> {
    twisted_quantum_multiplication_from_s_coeff(descendant_s, classical_h, mode)
}

pub(crate) fn twisted_quantum_multiplication_from_s_coeff<C: Coeff>(
    descendant_s: &SeriesSMatrix<C>,
    classical_h: &SeriesMatrix<C>,
    mode: &TwistedCalibrationMode,
) -> Result<SeriesMatrix<C>, GwError> {
    let s1 = descendant_s.coefficient(1).ok_or_else(|| {
        GwError::ConventionMismatch("need S-matrix through z^{-1} to recover product".to_string())
    })?;
    Ok(match mode {
        TwistedCalibrationMode::Euler => classical_h.sub(&s1.q_derivative()),
        TwistedCalibrationMode::InverseEuler | TwistedCalibrationMode::InverseEulerFiberPlus => {
            classical_h.add(&s1.q_derivative())
        }
    })
}

pub(crate) fn twisted_quantum_multiplication_from_picard_fuchs(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    mirror: &[Rational],
    inverse_mirror: &[Rational],
) -> Result<SeriesMatrix, GwError> {
    if twist.degree_sum() > n + 1 {
        return Err(GwError::UnsupportedInvariant(
            "Euler Picard-Fuchs product extraction currently requires total twisting degree <= n+1"
                .to_string(),
        ));
    }
    let mut log_jacobian = vec![Rational::zero(); q_degree + 1];
    log_jacobian[0] = Rational::one();
    for degree in 1..=q_degree {
        log_jacobian[degree] =
            Rational::from(degree) * mirror.get(degree).cloned().unwrap_or_else(Rational::zero);
    }
    let jacobian = QSeries::from_coeffs(
        compose_plain_series(&log_jacobian, inverse_mirror, q_degree)
            .into_iter()
            .map(RatFun::from_rational)
            .collect(),
    );
    let q_in_flat = QSeries::from_coeffs(
        inverse_mirror
            .iter()
            .take(q_degree + 1)
            .cloned()
            .map(RatFun::from_rational)
            .collect(),
    );
    let base = substitute_scaled_generator_in_polynomial(
        &twisted_base_polynomial_coefficients(n, q_degree, base_weights)?,
        &jacobian,
    );
    let fiber = substitute_scaled_generator_in_polynomial(
        &twisted_fiber_polynomial_coefficients(twist, q_degree, fiber_weights)?,
        &jacobian,
    );
    let mut relation = vec![QSeries::zero(q_degree); n + 2];
    for (power, coeff) in base.into_iter().enumerate().take(n + 2) {
        relation[power] = relation[power].add(&coeff);
    }
    for (power, coeff) in fiber.into_iter().enumerate() {
        if power > n + 1 {
            if !coeff.is_zero() {
                return Err(GwError::UnsupportedInvariant(
                    "Euler Picard-Fuchs relation exceeded the ambient state-space rank".to_string(),
                ));
            }
            continue;
        }
        relation[power] = relation[power].sub(&q_in_flat.mul(&coeff));
    }
    companion_multiplication_matrix_from_monic_polynomial(n + 1, &relation)
}

pub(crate) fn substitute_scaled_generator_in_polynomial(
    coefficients: &[QSeries],
    scale: &QSeries,
) -> Vec<QSeries> {
    let q_degree = scale.max_degree();
    let mut powers = Vec::with_capacity(coefficients.len());
    powers.push(QSeries::one(q_degree));
    for power in 1..coefficients.len() {
        powers.push(powers[power - 1].mul(scale));
    }
    coefficients
        .iter()
        .enumerate()
        .map(|(power, coeff)| coeff.mul(&powers[power]))
        .collect()
}

pub(crate) fn charpoly_qseries_coefficients(
    matrix: &SeriesMatrix,
) -> Result<Vec<QSeries>, GwError> {
    charpoly_qseries_coefficients_coeff(matrix)
}

pub(crate) fn charpoly_qseries_coefficients_coeff<C: Coeff>(
    matrix: &SeriesMatrix<C>,
) -> Result<Vec<QSeries<C>>, GwError> {
    if matrix.rows() != matrix.cols() {
        return Err(GwError::ConventionMismatch(
            "characteristic polynomial requires a square matrix".to_string(),
        ));
    }
    let size = matrix.rows();
    let q_degree = matrix.max_degree();
    let mut polynomial_matrix = vec![vec![vec![QSeries::<C>::zero(q_degree)]; size]; size];
    for (row, out_row) in polynomial_matrix.iter_mut().enumerate() {
        for (col, entry) in out_row.iter_mut().enumerate() {
            let mut poly = vec![matrix.entry(row, col).neg()];
            if row == col {
                poly.push(QSeries::<C>::one(q_degree));
            }
            *entry = poly;
        }
    }
    let mut charpoly = determinant_qseries_polynomial_matrix(&polynomial_matrix, q_degree);
    charpoly.resize(size + 1, QSeries::<C>::zero(q_degree));
    Ok(charpoly)
}

pub(crate) fn root_series_from_charpoly(
    coefficients: &[QSeries],
    branch_root: Rational,
    max_q_degree: usize,
) -> Result<QSeries, GwError> {
    let derivative = derivative_qseries_polynomial_coefficients(coefficients, max_q_degree);
    let mut root = QSeries::constant(RatFun::from_rational(branch_root), max_q_degree);
    for _ in 0..=max_q_degree {
        let value = evaluate_qseries_polynomial(coefficients, &root);
        if value.is_zero() {
            break;
        }
        let differential = evaluate_qseries_polynomial(&derivative, &root);
        root = root.sub(&value.div(&differential)?);
    }
    Ok(root)
}

pub(crate) fn root_series_from_charpoly_coeff<C: Coeff>(
    coefficients: &[QSeries<C>],
    branch_root: C,
    max_q_degree: usize,
) -> Result<QSeries<C>, GwError> {
    let derivative = derivative_qseries_polynomial_coefficients(coefficients, max_q_degree);
    let mut root = QSeries::constant(branch_root, max_q_degree);
    for _ in 0..=max_q_degree {
        let value = evaluate_qseries_polynomial(coefficients, &root);
        if value.is_zero() {
            break;
        }
        let differential = evaluate_qseries_polynomial(&derivative, &root);
        root = root.sub(&value.div(&differential)?);
    }
    Ok(root)
}

pub(crate) fn spectral_transition_matrix_from_roots(
    multiplication: &SeriesMatrix,
    roots: &[QSeries],
) -> Result<SeriesMatrix, GwError> {
    spectral_transition_matrix_from_roots_coeff(multiplication, roots)
}

pub(crate) fn spectral_transition_matrix_from_roots_coeff<C: Coeff>(
    multiplication: &SeriesMatrix<C>,
    roots: &[QSeries<C>],
) -> Result<SeriesMatrix<C>, GwError> {
    let size = roots.len();
    let q_degree = multiplication.max_degree();
    let identity = SeriesMatrix::<C>::identity(size, q_degree);
    let mut columns = vec![vec![QSeries::<C>::zero(q_degree); size]; size];

    for branch in 0..size {
        let mut projector = SeriesMatrix::<C>::identity(size, q_degree);
        for other in 0..size {
            if other == branch {
                continue;
            }
            let shifted = multiplication.sub(&series_matrix_scale(&identity, &roots[other]));
            let denominator = roots[branch].sub(&roots[other]).inverse()?;
            projector = projector.mul(&shifted);
            projector = series_matrix_scale(&projector, &denominator);
        }
        for (row, column_row) in columns.iter_mut().enumerate() {
            column_row[branch] = projector.entry(row, 0).clone();
        }
    }

    Ok(SeriesMatrix::from_entries(columns))
}

pub(crate) fn twisted_classical_limit_diagonal_coefficients(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<Vec<Vec<RatFun>>, GwError> {
    twisted_classical_limit_diagonal_coefficients_with_mode(
        n,
        twist,
        z_order,
        base_weights,
        fiber_weights,
        TwistedCalibrationMode::InverseEuler,
    )
}

pub(crate) fn twisted_classical_limit_diagonal_coefficients_with_mode(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    mode: TwistedCalibrationMode,
) -> Result<Vec<Vec<RatFun>>, GwError> {
    validate_twisted_weights(n, twist, base_weights, fiber_weights)?;
    (0..=n)
        .map(|branch| {
            twisted_classical_limit_diagonal_coefficients_for_branch(
                n,
                twist,
                branch,
                z_order,
                base_weights,
                fiber_weights,
                mode.clone(),
            )
        })
        .collect()
}

pub(crate) fn twisted_classical_limit_diagonal_coefficients_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    z_order: usize,
    base_weights: &[C],
    fiber_weights: &[C],
) -> Result<Vec<Vec<C>>, GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    if fiber_weights.len() != twist.rank() {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} fiber weights, got {}",
            twist.rank(),
            fiber_weights.len()
        )));
    }
    (0..=n)
        .map(|branch| {
            twisted_classical_limit_diagonal_coefficients_for_branch_coeff(
                n,
                twist,
                branch,
                z_order,
                base_weights,
                fiber_weights,
            )
        })
        .collect()
}

pub(crate) fn twisted_classical_limit_diagonal_coefficients_for_branch(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    branch: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    mode: TwistedCalibrationMode,
) -> Result<Vec<RatFun>, GwError> {
    let mut exponent = vec![RatFun::zero(); z_order + 1];
    for r in 1..=z_order.div_ceil(2) {
        let order = 2 * r - 1;
        let coefficient = bernoulli_asymptotic_coefficient::<Rational>(r);
        let mut weight_sum = Rational::zero();
        for other in 0..=n {
            if other == branch {
                continue;
            }
            let difference = base_weights[other].clone() - base_weights[branch].clone();
            if difference.is_zero() {
                return Err(GwError::NonSemisimplePoint);
            }
            weight_sum += Rational::one() / difference.pow_usize(order);
        }
        for (bundle_degree, fiber_weight) in twist.degrees().iter().zip(fiber_weights) {
            let fiber_root = fiber_weight.clone()
                - Rational::from(*bundle_degree) * base_weights[branch].clone();
            if fiber_root.is_zero() {
                return Err(GwError::NonSemisimplePoint);
            }
            let fiber_term = Rational::one() / fiber_root.pow_usize(order);
            weight_sum = match mode {
                TwistedCalibrationMode::InverseEuler => weight_sum - fiber_term,
                TwistedCalibrationMode::InverseEulerFiberPlus => weight_sum + fiber_term,
                TwistedCalibrationMode::Euler => weight_sum + fiber_term,
            };
        }
        exponent[order] = RatFun::from_rational(coefficient * weight_sum);
    }
    Ok(exp_scalar_z_series(&exponent))
}

pub(crate) fn twisted_classical_limit_diagonal_coefficients_for_branch_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    branch: usize,
    z_order: usize,
    base_weights: &[C],
    fiber_weights: &[C],
) -> Result<Vec<C>, GwError> {
    let mut exponent = vec![C::zero(); z_order + 1];
    for r in 1..=z_order.div_ceil(2) {
        let order = 2 * r - 1;
        let coefficient = bernoulli_asymptotic_coefficient::<C>(r);
        let mut weight_sum = C::zero();
        for other in 0..=n {
            if other == branch {
                continue;
            }
            let difference = base_weights[other].sub(&base_weights[branch]);
            if difference.is_zero() {
                return Err(GwError::NonSemisimplePoint);
            }
            weight_sum = weight_sum.add(&C::one().div(&difference.pow_usize(order)));
        }
        for (bundle_degree, fiber_weight) in twist.degrees().iter().zip(fiber_weights) {
            let fiber_root =
                fiber_weight.sub(&C::from_usize(*bundle_degree).mul(&base_weights[branch]));
            if fiber_root.is_zero() {
                return Err(GwError::NonSemisimplePoint);
            }
            weight_sum = weight_sum.sub(&C::one().div(&fiber_root.pow_usize(order)));
        }
        exponent[order] = coefficient.mul(&weight_sum);
    }
    Ok(exp_scalar_z_series(&exponent))
}

#[cfg(test)]
mod frobenius_transition_inverse_tests {
    use super::*;
    use crate::factored::FactoredRatFun;

    fn assert_factored_matrix_semantically_equal(
        left: &SeriesMatrix<FactoredRatFun>,
        right: &SeriesMatrix<FactoredRatFun>,
    ) {
        assert_eq!((left.rows(), left.cols()), (right.rows(), right.cols()));
        for row in 0..left.rows() {
            for col in 0..left.cols() {
                for degree in 0..=left.max_degree() {
                    let difference = left.entry(row, col).coeff(degree).unwrap().to_ratfun()
                        - right.entry(row, col).coeff(degree).unwrap().to_ratfun();
                    assert!(
                        difference.is_zero(),
                        "matrix entries differ at ({row},{col}), q^{degree}: {difference}"
                    );
                }
            }
        }
    }

    #[test]
    fn frobenius_adjoint_is_positive_degree_transition_inverse_over_ratfun() {
        let twist = NegativeSplitBundleTwist::new(vec![2]).unwrap();
        let base_weights = [Rational::from(1), Rational::from(2), Rational::from(4)]
            .into_iter()
            .map(RatFun::from_rational)
            .collect::<Vec<_>>();
        let fiber_weights = [Rational::from(11)]
            .into_iter()
            .map(RatFun::from_rational)
            .collect::<Vec<_>>();
        let canonical =
            specialized_twisted_birkhoff_canonical_data_for_coeff_weights_with_validation(
                2,
                &twist,
                1,
                &base_weights,
                &fiber_weights,
                TwistedCalibrationValidation::Full,
            )
            .unwrap();
        let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());
        let frobenius_inverse = SeriesMatrix::from_entries(canonical.flat_to_canonical);
        let reference = crate::reconstruction::invert_series_matrix_coeff(&transition).unwrap();

        assert_eq!(frobenius_inverse, reference);
        assert_eq!(
            frobenius_inverse.mul(&transition),
            SeriesMatrix::identity(3, 1)
        );
    }

    #[test]
    fn frobenius_adjoint_is_positive_degree_transition_inverse_over_factored() {
        let twist = NegativeSplitBundleTwist::new(vec![2]).unwrap();
        let base_weights = [Rational::from(1), Rational::from(2), Rational::from(4)]
            .into_iter()
            .map(FactoredRatFun::from_rational)
            .collect::<Vec<_>>();
        let fiber_weights = [Rational::from(11)]
            .into_iter()
            .map(FactoredRatFun::from_rational)
            .collect::<Vec<_>>();
        let canonical =
            specialized_twisted_birkhoff_canonical_data_for_coeff_weights_with_validation(
                2,
                &twist,
                1,
                &base_weights,
                &fiber_weights,
                TwistedCalibrationValidation::Full,
            )
            .unwrap();
        let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());
        let frobenius_inverse = SeriesMatrix::from_entries(canonical.flat_to_canonical);
        let reference = crate::reconstruction::invert_series_matrix_coeff(&transition).unwrap();

        assert_factored_matrix_semantically_equal(&frobenius_inverse, &reference);
        assert_factored_matrix_semantically_equal(
            &frobenius_inverse.mul(&transition),
            &SeriesMatrix::identity(3, 1),
        );
    }
}
