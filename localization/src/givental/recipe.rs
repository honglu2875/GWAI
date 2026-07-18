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
//! theories is an alternative recipe for the same contract.  Its
//! calibration-specific cone-point assembly lives in this module, over the
//! target-neutral Laurent and Birkhoff algebra in `reconstruction`.

use super::{
    canonical_evaluation_matrix, relative_sqrt_delta_series, solve_r_coefficients_from_flatness,
    CalibrationId, CanonicalFrameConvention, SemisimpleCalibration, SeriesRMatrix, SeriesSMatrix,
};
use crate::core::algebra::{Coeff, RatFun, Rational};
use crate::core::error::GwError;
use crate::core::series::{integrate_q_derivative_zero_constant_matrix, QSeries, SeriesMatrix};
pub(crate) use crate::reconstruction::multiply_qseries_polynomial_by_linear;
use crate::reconstruction::HCoeffLaurentSeries;

mod cone_point;
pub(crate) use cone_point::*;

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
    let coefficients = solve_r_coefficients_from_flatness(
        &frame.roots,
        &connection,
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

/// Descendant `S`-matrix from an `I`-function, by mirror transformation and
/// Birkhoff factorization.
///
/// This is the second calibration recipe, for targets whose natural datum is
/// a cohomology-valued hypergeometric series rather than a quantum ring
/// (toric complete intersections, twisted theories).  The steps are: read
/// the mirror map off the `H z^{-1}` part of `I`, gauge it away and re-expand
/// in the flat coordinate to obtain `J`, generate the fundamental solution by
/// repeated applications of `z q d/dq + H`-cup (reduced by the classical ring
/// relation), and Birkhoff-factor the resulting loop-group element; the
/// negative factor's `z^{-k}` coefficients are the descendant `S`-matrix.
///
/// `classical_h_relation` expresses `H^{n+1}` in lower powers in the
/// classical ring.  The `I`-coefficients are indexed by Novikov degree.
///
pub fn descendant_s_from_i_function<C: Coeff>(
    n: usize,
    i_coefficients: &[HCoeffLaurentSeries<C>],
    classical_h_relation: &[C],
    flat_metric: &SeriesMatrix<C>,
    q_degree: usize,
    z_order: usize,
    calibration: CalibrationId,
) -> Result<SeriesSMatrix<C>, GwError> {
    let mirror = mirror_map_coefficients_from_i_function_coeff(i_coefficients, q_degree);
    let inverse_mirror = crate::core::series::invert_mirror_map(&mirror, q_degree);
    let j_coefficients = mirror_transformed_j_coefficients_from_i_function_mod_relation_coeff(
        n,
        i_coefficients,
        &mirror,
        &inverse_mirror,
        q_degree,
        classical_h_relation,
    );
    descendant_s_from_j_function(
        n,
        &j_coefficients,
        classical_h_relation,
        flat_metric,
        q_degree,
        z_order,
        calibration,
    )
}

/// Descendant `S`-matrix from already-mirror-transformed `J`-coefficients:
/// fundamental solution, Birkhoff factorization, metric adjoint.
///
/// Use this instead of [`descendant_s_from_i_function`] when the mirror
/// transformation happened elsewhere — e.g. for multi-parameter targets whose
/// mirror map is computed in bidegree-graded form before restricting to a
/// Novikov ray.
pub fn descendant_s_from_j_function<C: Coeff>(
    n: usize,
    j_coefficients: &[HCoeffLaurentSeries<C>],
    classical_h_relation: &[C],
    flat_metric: &SeriesMatrix<C>,
    q_degree: usize,
    z_order: usize,
    calibration: CalibrationId,
) -> Result<SeriesSMatrix<C>, GwError> {
    descendant_s_from_cone_point_function(
        n,
        j_coefficients,
        classical_h_relation,
        flat_metric,
        q_degree,
        z_order,
        calibration,
    )
}

/// Descendant `S`-matrix from a cohomology-valued point on Givental's cone:
/// generate the fundamental solution, Birkhoff-factor it, and take the metric
/// adjoint of the negative factor.
///
/// If the input is already on the small-J slice this is the usual J -> S
/// construction.  If the input has a nontrivial polynomial/positive-z part
/// (as for non-Fano toric bundles), the positive Birkhoff factor performs the
/// projection to the small-J calibration while the negative factor still gives
/// the descendant S-matrix.
pub fn descendant_s_from_cone_point_function<C: Coeff>(
    n: usize,
    cone_point_coefficients: &[HCoeffLaurentSeries<C>],
    classical_h_relation: &[C],
    flat_metric: &SeriesMatrix<C>,
    q_degree: usize,
    z_order: usize,
    calibration: CalibrationId,
) -> Result<SeriesSMatrix<C>, GwError> {
    let fundamental = fundamental_solution_matrix_from_j_coefficients_mod_relation_coeff(
        n,
        q_degree,
        cone_point_coefficients,
        classical_h_relation,
    );
    let birkhoff = birkhoff_descendant_s_matrix_from_fundamental_coeff(
        n + 1,
        q_degree,
        z_order,
        &fundamental,
        calibration,
    )?;
    // The Birkhoff factor acts on vectors; the engine consumes the metric
    // adjoint S^* = eta^{-1} S^T eta (the action on covector insertions).
    // The symplectic condition makes the two agree through z^1 and diverge
    // from z^2 on, so getting this convention wrong is invisible in
    // low-order checks.
    metric_adjoint_descendant_s_matrix_coeff(birkhoff, flat_metric)
}

/// Classical Lagrange transition for distinct rational eigenvalues: column
/// `p` holds the coefficients of the idempotent `E_p(x)` in the power basis
/// `1, x, x^2, ...`.
pub(crate) fn classical_lagrange_transition(seeds: &[Rational]) -> Vec<Vec<Rational>> {
    let size = seeds.len();
    let mut transition = vec![vec![Rational::zero(); size]; size];
    for point in 0..size {
        let mut projector = vec![Rational::one()];
        let mut denominator = Rational::one();
        for other in 0..size {
            if other == point {
                continue;
            }
            let mut next = vec![Rational::zero(); projector.len() + 1];
            for (power, coefficient) in projector.iter().enumerate() {
                next[power] += -(seeds[other].clone()) * coefficient.clone();
                next[power + 1] += coefficient.clone();
            }
            projector = next;
            denominator = denominator * (seeds[point].clone() - seeds[other].clone());
        }
        for (power, coefficient) in projector.into_iter().enumerate() {
            transition[power][point] = coefficient / denominator.clone();
        }
    }
    transition
}

/// Inverse of a series matrix whose constant term is the identity, by the
/// truncated Neumann series `I - N + N^2 - ...` for `N = M - I`.
pub(crate) fn neumann_inverse(
    matrix: &SeriesMatrix,
    q_degree: usize,
) -> Result<SeriesMatrix, GwError> {
    let size = matrix.rows();
    let identity = SeriesMatrix::identity(size, q_degree);
    let nilpotent_part = matrix.sub(&identity);
    let constant_term_vanishes = nilpotent_part.entries().iter().all(|row| {
        row.iter().all(|series| {
            series
                .coeff(0)
                .map(|constant| constant.is_zero())
                .unwrap_or(true)
        })
    });
    if !constant_term_vanishes {
        return Err(GwError::AlgebraFailure(
            "Neumann inversion requires identity constant term".to_string(),
        ));
    }
    let mut inverse = identity.clone();
    let mut power = identity;
    for _ in 0..q_degree {
        power = power.mul(&nilpotent_part).neg();
        inverse = inverse.add(&power);
    }
    Ok(inverse)
}

/// Monic characteristic polynomial of a series matrix by Faddeev-LeVerrier,
/// in ascending powers (constant first, leading `1` last).
pub(crate) fn series_matrix_charpoly(matrix: &SeriesMatrix) -> Vec<QSeries> {
    let size = matrix.rows();
    let q_degree = matrix.max_degree();
    let trace = |m: &SeriesMatrix| {
        let mut total = QSeries::zero(q_degree);
        for idx in 0..size {
            total = total.add(m.entry(idx, idx));
        }
        total
    };

    // charpoly = x^size + c_1 x^{size-1} + ... + c_size.
    let mut descending = vec![QSeries::one(q_degree)];
    let mut auxiliary = SeriesMatrix::identity(size, q_degree);
    for step in 1..=size {
        let product = matrix.mul(&auxiliary);
        let coefficient = trace(&product)
            .scale(&(&RatFun::from_rational(-Rational::one()) / &RatFun::from(step)));
        auxiliary = product.add(&scaled_identity(&coefficient, size, q_degree));
        descending.push(coefficient);
    }
    descending.reverse();
    descending
}

fn scaled_identity(scalar: &QSeries, size: usize, q_degree: usize) -> SeriesMatrix {
    let mut entries = vec![vec![QSeries::zero(q_degree); size]; size];
    for (idx, row) in entries.iter_mut().enumerate() {
        row[idx] = scalar.clone();
    }
    SeriesMatrix::from_entries(entries)
}

/// Canonical frame from an explicit quantum multiplication operator.
///
/// Unlike [`divisor_lagrange_frame`], this makes no assumption about the flat
/// basis beyond containing the unit at `unit_coordinates`: idempotents are the
/// spectral projectors `prod_{q != p} (M - u_q)/(u_p - u_q)` of the operator
/// applied to the unit element, eigenvalue series come from Newton iteration
/// on the Faddeev-LeVerrier characteristic polynomial, and metric norms come
/// from the supplied flat metric.  This is the frame builder for targets whose
/// multiplication matrix is honest in a constant basis but not in companion
/// form (multi-divisor rings presented through a cyclic generator).
pub fn operator_lagrange_frame(
    multiplication: &SeriesMatrix,
    seeds: &[Rational],
    unit_coordinates: &[RatFun],
    flat_metric: &SeriesMatrix,
) -> Result<CanonicalFrame, GwError> {
    let size = multiplication.rows();
    let q_degree = multiplication.max_degree();
    let charpoly = series_matrix_charpoly(multiplication);
    let roots = seeds
        .iter()
        .map(|seed| newton_root_series(&charpoly, &RatFun::from_rational(seed.clone()), q_degree))
        .collect::<Result<Vec<_>, _>>()?;

    let mut transition_columns = vec![vec![QSeries::zero(q_degree); size]; size];
    for branch in 0..size {
        // E_p(M) applied to the unit element.
        let mut vector = unit_coordinates
            .iter()
            .map(|coordinate| QSeries::constant(coordinate.clone(), q_degree))
            .collect::<Vec<_>>();
        let mut denominator = QSeries::one(q_degree);
        for other in 0..size {
            if other == branch {
                continue;
            }
            let mut next = vec![QSeries::zero(q_degree); size];
            for (row, entry) in next.iter_mut().enumerate() {
                let mut total = QSeries::zero(q_degree);
                for col in 0..size {
                    total = total.add(&multiplication.entry(row, col).mul(&vector[col]));
                }
                *entry = total.sub(&roots[other].mul(&vector[row]));
            }
            vector = next;
            denominator = denominator.mul(&roots[branch].sub(&roots[other]));
        }
        let denominator_inv = denominator.inverse()?;
        for (row, entry) in vector.into_iter().enumerate() {
            transition_columns[row][branch] = entry.mul(&denominator_inv);
        }
    }
    let transition_to_flat = SeriesMatrix::from_entries(transition_columns);
    let flat_to_canonical = crate::reconstruction::invert_series_matrix_coeff(&transition_to_flat)?;

    let canonical_metric = transition_to_flat
        .transpose()
        .mul(flat_metric)
        .mul(&transition_to_flat);
    let mut metric_norms = Vec::with_capacity(size);
    let mut inverse_metric_norms = Vec::with_capacity(size);
    for branch in 0..size {
        let norm = canonical_metric.entry(branch, branch).clone();
        inverse_metric_norms.push(norm.inverse()?);
        metric_norms.push(norm);
    }

    Ok(CanonicalFrame {
        roots,
        transition_to_flat,
        flat_to_canonical,
        metric_norms,
        inverse_metric_norms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::negative_split_projective::{
        NegativeSplitBundleTwist, NegativeSplitLineHypergeometricModel,
    };
    use crate::spaces::projective_space::provider::projective_space_descendant_s_matrix_at_lambda_weights;

    #[test]
    fn i_function_and_qde_recipes_agree_on_projective_space() {
        // A rank-zero twist is untwisted P^1, so its hypergeometric
        // I-function is the P^1 I-function and the mirror/Birkhoff recipe
        // must reproduce the QDE-solved descendant S-matrix: the two
        // calibration recipes cross-validate on the same target.
        let q_degree = 2;
        let z_order = 2;
        let weights = [Rational::from(2), Rational::from(5)];
        let ratfun_weights = weights
            .iter()
            .map(|weight| RatFun::from_rational(weight.clone()))
            .collect::<Vec<_>>();
        let twist = NegativeSplitBundleTwist::new(Vec::new()).unwrap();
        let model = NegativeSplitLineHypergeometricModel::from_ratfun_weights(
            1,
            twist,
            q_degree,
            z_order,
            ratfun_weights.clone(),
            &[],
        )
        .unwrap();
        let relation =
            crate::reconstruction::base_h_power_relation_coeff(1, &ratfun_weights).unwrap();
        // Atiyah-Bott flat metric of P^1 in the H-power basis:
        // G_{rs} = sum_i w_i^{r+s} / prod_{j != i} (w_i - w_j).
        let metric_entry = |row: usize, col: usize| {
            let mut total = Rational::zero();
            for i in 0..2usize {
                let mut euler = Rational::one();
                for j in 0..2usize {
                    if j != i {
                        euler = euler * (weights[i].clone() - weights[j].clone());
                    }
                }
                total += weights[i].pow_usize(row + col) / euler;
            }
            RatFun::from_rational(total)
        };
        let flat_metric = SeriesMatrix::constant(
            (0..2)
                .map(|row| (0..2).map(|col| metric_entry(row, col)).collect())
                .collect(),
            q_degree,
        );
        let s_from_i = descendant_s_from_i_function(
            1,
            &model.i_coefficients().unwrap(),
            &relation,
            &flat_metric,
            q_degree,
            z_order,
            CalibrationId("test-i-function-recipe".to_string()),
        )
        .unwrap();
        let s_from_qde =
            projective_space_descendant_s_matrix_at_lambda_weights(1, q_degree, z_order, &weights)
                .unwrap();

        for order in 0..=z_order {
            for row in 0..2 {
                for col in 0..2 {
                    for degree in 0..=q_degree {
                        assert_eq!(
                            s_from_i
                                .coefficient(order)
                                .unwrap()
                                .entry(row, col)
                                .coeff(degree)
                                .unwrap()
                                .as_rational(),
                            s_from_qde
                                .coefficient(order)
                                .unwrap()
                                .entry(row, col)
                                .coeff(degree)
                                .unwrap()
                                .as_rational(),
                            "S recipe mismatch at z^{order} ({row},{col}) q^{degree}"
                        );
                    }
                }
            }
        }
    }
}
