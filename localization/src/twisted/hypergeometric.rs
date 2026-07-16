//! Genus-zero QRR/Lefschetz and hypergeometric models for negative split
//! bundles (the I-function / mirror-map / Birkhoff machinery).

use super::*;
use crate::algebra::{Coeff, RatFun, Rational};
use crate::error::GwError;
use crate::givental::{CalibrationId, SeriesSMatrix};
use crate::series::{invert_mirror_map, SeriesMatrix};
use std::collections::BTreeMap;

/// Genus-zero QRR/Lefschetz operator for a negative split bundle.
///
/// This is the hypergeometric part of quantum Riemann-Roch: it applies the
/// concave Euler factor degree-by-degree to the untwisted projective
/// `I`-function.  It is deliberately narrower than the all-genus quantized QRR
/// operator; the latter still needs a normalized semisimple calibration before
/// it can feed the graph evaluator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeSplitQrrOperator {
    twist: NegativeSplitBundleTwist,
}

impl NegativeSplitQrrOperator {
    pub fn new(twist: NegativeSplitBundleTwist) -> Self {
        Self { twist }
    }

    pub fn twist(&self) -> &NegativeSplitBundleTwist {
        &self.twist
    }

    pub fn degree_factor(&self, n: usize, degree: usize) -> HLaurentSeries {
        negative_split_qrr_euler_factor(n, &self.twist, degree)
    }

    pub fn apply_to_projective_i_coefficient(&self, n: usize, degree: usize) -> HLaurentSeries {
        self.degree_factor(n, degree)
            .multiply(&projective_i_function_coefficient(n, degree))
    }

    pub fn i_coefficients(&self, n: usize, q_degree: usize) -> Vec<HLaurentSeries> {
        (0..=q_degree)
            .map(|degree| self.apply_to_projective_i_coefficient(n, degree))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeSplitQrrModel {
    n: usize,
    operator: NegativeSplitQrrOperator,
    q_degree: usize,
}

impl NegativeSplitQrrModel {
    pub fn new(n: usize, twist: NegativeSplitBundleTwist, q_degree: usize) -> Self {
        Self {
            n,
            operator: NegativeSplitQrrOperator::new(twist),
            q_degree,
        }
    }

    pub fn i_coefficients(&self) -> Vec<HLaurentSeries> {
        self.operator.i_coefficients(self.n, self.q_degree)
    }

    pub fn mirror_map_coefficients(&self) -> Vec<Rational> {
        mirror_map_coefficients_from_i_function(&self.i_coefficients(), self.q_degree)
    }

    pub fn inverse_mirror_map_coefficients(&self) -> Vec<Rational> {
        invert_mirror_map(&self.mirror_map_coefficients(), self.q_degree)
    }

    pub fn mirror_transformed_j_coefficients(&self) -> Vec<HLaurentSeries> {
        let i_coefficients = self.i_coefficients();
        let mirror = mirror_map_coefficients_from_i_function(&i_coefficients, self.q_degree);
        let inverse_mirror = invert_mirror_map(&mirror, self.q_degree);
        mirror_transformed_j_coefficients_from_i_function(
            self.n,
            &i_coefficients,
            &mirror,
            &inverse_mirror,
            self.q_degree,
        )
    }

    pub fn fundamental_solution_matrix(&self) -> BTreeMap<i32, SeriesMatrix> {
        fundamental_solution_matrix_from_j_coefficients(
            self.n,
            self.q_degree,
            &self.mirror_transformed_j_coefficients(),
        )
    }

    pub fn birkhoff_descendant_s_matrix(&self, z_order: usize) -> Result<SeriesSMatrix, GwError> {
        validate_birkhoff_request_bounds(self.q_degree, z_order)?;
        birkhoff_descendant_s_matrix_from_fundamental(
            self.n + 1,
            self.q_degree,
            z_order,
            &self.fundamental_solution_matrix(),
            CalibrationId("negative-split-qrr-hypergeometric-birkhoff".to_string()),
        )
    }
}

pub fn negative_split_mirror_map_coefficients(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
) -> Vec<Rational> {
    // The scalar mirror map is read from the H/z coefficient of the I-function.
    mirror_map_coefficients_from_i_function(
        &negative_split_i_function_coefficients(n, twist, q_degree),
        q_degree,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeSplitHypergeometricModel {
    n: usize,
    twist: NegativeSplitBundleTwist,
    q_degree: usize,
}

impl NegativeSplitHypergeometricModel {
    pub fn new(n: usize, twist: NegativeSplitBundleTwist, q_degree: usize) -> Self {
        Self { n, twist, q_degree }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn twist(&self) -> &NegativeSplitBundleTwist {
        &self.twist
    }

    pub fn q_degree(&self) -> usize {
        self.q_degree
    }

    pub fn i_coefficients(&self) -> Vec<HLaurentSeries> {
        negative_split_i_function_coefficients(self.n, &self.twist, self.q_degree)
    }

    pub fn mirror_map_coefficients(&self) -> Vec<Rational> {
        mirror_map_coefficients_from_i_function(&self.i_coefficients(), self.q_degree)
    }

    pub fn inverse_mirror_map_coefficients(&self) -> Vec<Rational> {
        invert_mirror_map(&self.mirror_map_coefficients(), self.q_degree)
    }

    pub fn mirror_transformed_j_coefficients(&self) -> Vec<HLaurentSeries> {
        // J(q) = exp(-H t(q)/z) I(Q(q)), with Q(q) the inverse mirror map.
        let i_coefficients = self.i_coefficients();
        let mirror = mirror_map_coefficients_from_i_function(&i_coefficients, self.q_degree);
        let inverse_mirror = invert_mirror_map(&mirror, self.q_degree);
        mirror_transformed_j_coefficients_from_i_function(
            self.n,
            &i_coefficients,
            &mirror,
            &inverse_mirror,
            self.q_degree,
        )
    }

    pub fn fundamental_solution_matrix(&self) -> BTreeMap<i32, SeriesMatrix> {
        // The columns are J, z q dJ/dq, ..., (z q d/dq)^n J, written in the
        // hyperplane basis.  This is the fundamental solution before Birkhoff
        // factorization.
        fundamental_solution_matrix_from_j_coefficients(
            self.n,
            self.q_degree,
            &self.mirror_transformed_j_coefficients(),
        )
    }

    pub fn birkhoff_descendant_s_matrix(&self, z_order: usize) -> Result<SeriesSMatrix, GwError> {
        // Extract the negative z-power factor from the fundamental solution;
        // that factor is the descendant S-calibration used on insertions.
        validate_birkhoff_request_bounds(self.q_degree, z_order)?;
        birkhoff_descendant_s_matrix_from_fundamental(
            self.n + 1,
            self.q_degree,
            z_order,
            &self.fundamental_solution_matrix(),
            CalibrationId("negative-split-hypergeometric-birkhoff".to_string()),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeSplitEquivariantHypergeometricModel {
    n: usize,
    twist: NegativeSplitBundleTwist,
    q_degree: usize,
    base_weights: Vec<Rational>,
    fiber_weights: Vec<Rational>,
    min_z_power: i32,
}

fn default_negative_split_min_z_power(
    n: usize,
    q_degree: usize,
    z_order: usize,
) -> Result<i32, GwError> {
    let state_space_size = n.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant(
            "negative-split hypergeometric state-space size overflow".to_string(),
        )
    })?;
    let depth = state_space_size
        .checked_mul(q_degree)
        .and_then(|value| value.checked_add(z_order))
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "negative-split default Laurent z-window overflow".to_string(),
            )
        })?;
    let depth = i32::try_from(depth).map_err(|_| {
        GwError::UnsupportedInvariant(
            "negative-split default Laurent z-window exceeds i32 range".to_string(),
        )
    })?;
    depth.checked_neg().ok_or_else(|| {
        GwError::UnsupportedInvariant(
            "negative-split default Laurent z-window exceeds i32 range".to_string(),
        )
    })
}

fn negative_laurent_depth_to_min_z(depth: usize) -> Result<i32, GwError> {
    let depth = i32::try_from(depth).map_err(|_| {
        GwError::UnsupportedInvariant(
            "negative-split Birkhoff Laurent depth exceeds i32 range".to_string(),
        )
    })?;
    depth.checked_neg().ok_or_else(|| {
        GwError::UnsupportedInvariant(
            "negative-split Birkhoff Laurent depth exceeds i32 range".to_string(),
        )
    })
}

fn birkhoff_ready_fundamental<C, F>(
    n: usize,
    q_degree: usize,
    z_order: usize,
    initial_min_z_power: i32,
    mut build_fundamental: F,
) -> Result<BTreeMap<i32, SeriesMatrix<C>>, GwError>
where
    C: Coeff,
    F: FnMut(i32) -> Result<BTreeMap<i32, SeriesMatrix<C>>, GwError>,
{
    validate_birkhoff_request_bounds(q_degree, z_order)?;
    // D = H + z q d/dq can raise z-power once in each of the n derivatives
    // used to form the flat-basis columns.  Start deeply enough that the
    // nonnegative window used to plan Birkhoff dependencies is itself exact.
    let derivative_probe_min_z = negative_laurent_depth_to_min_z(n)?;
    let mut working_min_z = initial_min_z_power.min(derivative_probe_min_z);

    loop {
        let fundamental = build_fundamental(working_min_z)?;
        if q_degree == 0 || z_order == 0 {
            return Ok(fundamental);
        }
        let required_final_depth =
            required_birkhoff_negative_z_depth(&fundamental, q_degree, z_order)?;
        let required_source_depth = required_final_depth.checked_add(n).ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "negative-split Birkhoff source Laurent depth overflow".to_string(),
            )
        })?;
        let required_source_min_z = negative_laurent_depth_to_min_z(required_source_depth)?;
        if working_min_z <= required_source_min_z {
            return Ok(fundamental);
        }
        working_min_z = required_source_min_z;
    }
}

impl NegativeSplitEquivariantHypergeometricModel {
    pub fn new(
        n: usize,
        twist: NegativeSplitBundleTwist,
        q_degree: usize,
        base_weights: Vec<Rational>,
        fiber_weights: Vec<Rational>,
        min_z_power: i32,
    ) -> Result<Self, GwError> {
        n.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "negative-split hypergeometric state-space size overflow".to_string(),
            )
        })?;
        q_degree.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "negative-split hypergeometric q-degree count overflow".to_string(),
            )
        })?;
        validate_twisted_weights(n, &twist, &base_weights, &fiber_weights)?;
        Ok(Self {
            n,
            twist,
            q_degree,
            base_weights,
            fiber_weights,
            min_z_power,
        })
    }

    pub fn with_default_z_truncation(
        n: usize,
        twist: NegativeSplitBundleTwist,
        q_degree: usize,
        z_order: usize,
        base_weights: Vec<Rational>,
        fiber_weights: Vec<Rational>,
    ) -> Result<Self, GwError> {
        let min_z_power = default_negative_split_min_z_power(n, q_degree, z_order)?;
        Self::new(n, twist, q_degree, base_weights, fiber_weights, min_z_power)
    }

    pub fn i_coefficients(&self) -> Result<Vec<HLaurentSeries>, GwError> {
        (0..=self.q_degree)
            .map(|degree| {
                negative_split_equivariant_i_function_coefficient(
                    self.n,
                    &self.twist,
                    degree,
                    &self.base_weights,
                    &self.fiber_weights,
                    self.min_z_power,
                )
            })
            .collect()
    }

    pub fn mirror_map_coefficients(&self) -> Result<Vec<Rational>, GwError> {
        Ok(mirror_map_coefficients_from_i_function(
            &self.i_coefficients()?,
            self.q_degree,
        ))
    }

    pub fn inverse_mirror_map_coefficients(&self) -> Result<Vec<Rational>, GwError> {
        Ok(invert_mirror_map(
            &self.mirror_map_coefficients()?,
            self.q_degree,
        ))
    }

    pub fn mirror_transformed_j_coefficients(&self) -> Result<Vec<HLaurentSeries>, GwError> {
        let h_power_relation = base_h_power_relation(self.n, &self.base_weights)?;
        let i_coefficients = self.i_coefficients()?;
        let mirror = mirror_map_coefficients_from_i_function(&i_coefficients, self.q_degree);
        let inverse_mirror = invert_mirror_map(&mirror, self.q_degree);
        Ok(
            mirror_transformed_j_coefficients_from_i_function_mod_relation(
                self.n,
                &i_coefficients,
                &mirror,
                &inverse_mirror,
                self.q_degree,
                &h_power_relation,
            ),
        )
    }

    pub fn fundamental_solution_matrix(&self) -> Result<BTreeMap<i32, SeriesMatrix>, GwError> {
        let h_power_relation = base_h_power_relation(self.n, &self.base_weights)?;
        Ok(
            fundamental_solution_matrix_from_j_coefficients_mod_relation(
                self.n,
                self.q_degree,
                &self.mirror_transformed_j_coefficients()?,
                &h_power_relation,
            ),
        )
    }

    pub fn birkhoff_descendant_s_matrix(&self, z_order: usize) -> Result<SeriesSMatrix, GwError> {
        let fundamental = birkhoff_ready_fundamental(
            self.n,
            self.q_degree,
            z_order,
            self.min_z_power,
            |min_z_power| {
                let mut model = self.clone();
                model.min_z_power = min_z_power;
                model.fundamental_solution_matrix()
            },
        )?;
        birkhoff_descendant_s_matrix_from_fundamental(
            self.n + 1,
            self.q_degree,
            z_order,
            &fundamental,
            CalibrationId("negative-split-equivariant-hypergeometric-birkhoff".to_string()),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NegativeSplitLineHypergeometricModel<C = RatFun> {
    pub(crate) n: usize,
    pub(crate) twist: NegativeSplitBundleTwist,
    pub(crate) q_degree: usize,
    pub(crate) base_weights: Vec<C>,
    pub(crate) fiber_weights: Vec<C>,
    pub(crate) min_z_power: i32,
}

impl NegativeSplitLineHypergeometricModel<RatFun> {
    pub(crate) fn from_ratfun_weights(
        n: usize,
        twist: NegativeSplitBundleTwist,
        q_degree: usize,
        z_order: usize,
        base_weights: Vec<RatFun>,
        fiber_weights: &[RatFun],
    ) -> Result<Self, GwError> {
        Self::from_coeff_weights(n, twist, q_degree, z_order, base_weights, fiber_weights)
    }
}

impl<C: Coeff> NegativeSplitLineHypergeometricModel<C> {
    pub(crate) fn from_coeff_weights(
        n: usize,
        twist: NegativeSplitBundleTwist,
        q_degree: usize,
        z_order: usize,
        base_weights: Vec<C>,
        fiber_weights: &[C],
    ) -> Result<Self, GwError> {
        let state_space_size = n.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "negative-split hypergeometric state-space size overflow".to_string(),
            )
        })?;
        if base_weights.len() != state_space_size {
            return Err(GwError::AlgebraFailure(format!(
                "expected {} base weights, got {}",
                state_space_size,
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
        let min_z_power = default_negative_split_min_z_power(n, q_degree, z_order)?;
        Ok(Self {
            n,
            twist,
            q_degree,
            base_weights,
            fiber_weights: fiber_weights.to_vec(),
            min_z_power,
        })
    }

    pub(crate) fn i_coefficients(&self) -> Result<Vec<HCoeffLaurentSeries<C>>, GwError> {
        (0..=self.q_degree)
            .map(|degree| {
                negative_split_equivariant_i_function_coefficient_coeff(
                    self.n,
                    &self.twist,
                    degree,
                    &self.base_weights,
                    &self.fiber_weights,
                    self.min_z_power,
                )
            })
            .collect()
    }

    fn mirror_transformed_j_coefficients(&self) -> Result<Vec<HCoeffLaurentSeries<C>>, GwError> {
        let h_power_relation = base_h_power_relation_coeff(self.n, &self.base_weights)?;
        let i_coefficients = self.i_coefficients()?;
        let mirror = mirror_map_coefficients_from_i_function_coeff(&i_coefficients, self.q_degree);
        let inverse_mirror = invert_mirror_map(&mirror, self.q_degree);
        Ok(
            mirror_transformed_j_coefficients_from_i_function_mod_relation_coeff(
                self.n,
                &i_coefficients,
                &mirror,
                &inverse_mirror,
                self.q_degree,
                &h_power_relation,
            ),
        )
    }

    fn fundamental_solution_matrix(&self) -> Result<BTreeMap<i32, SeriesMatrix<C>>, GwError> {
        let h_power_relation = base_h_power_relation_coeff(self.n, &self.base_weights)?;
        Ok(
            fundamental_solution_matrix_from_j_coefficients_mod_relation_coeff(
                self.n,
                self.q_degree,
                &self.mirror_transformed_j_coefficients()?,
                &h_power_relation,
            ),
        )
    }

    pub(crate) fn birkhoff_descendant_s_matrix(
        &self,
        z_order: usize,
    ) -> Result<SeriesSMatrix<C>, GwError> {
        let fundamental = birkhoff_ready_fundamental(
            self.n,
            self.q_degree,
            z_order,
            self.min_z_power,
            |min_z_power| {
                let mut model = self.clone();
                model.min_z_power = min_z_power;
                model.fundamental_solution_matrix()
            },
        )?;
        birkhoff_descendant_s_matrix_from_fundamental_coeff(
            self.n + 1,
            self.q_degree,
            z_order,
            &fundamental,
            CalibrationId("negative-split-lambda-line-hypergeometric-birkhoff".to_string()),
        )
    }
}
