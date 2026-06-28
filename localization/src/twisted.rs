//! Twisted projective-space theories by negative split bundles.
//!
//! This module currently provides the target metadata, non-equivariant
//! hypergeometric `I`-function coefficients, bounded line-specialized
//! equivariant `I` coefficients with base and fiber weights, a genus-zero
//! QRR/Lefschetz factorization of the same coefficients, mirror-map
//! normalization, a formal Birkhoff extraction of the descendant `S`-factor, and
//! two semisimple skeletons.  The principal-relation skeleton is diagnostic
//! only: it fails the flat-pairing diagonalization beyond q=0.  The equivariant
//! Birkhoff skeleton validates the inverse-Euler product and low-order
//! Birkhoff/QRR `R` unitarity, including local `P^2`.  The non-equivariant
//! graph path uses an early rational specialization of the one-parameter lambda
//! line.  A fiber-equivariant mode keeps independent symbolic parameters
//! `mu_i` for the split summands while keeping the base weights
//! early-specialized; calibration-level specialization tests cover this mode.
//! The factored coefficient path keeps fiber-equivariant denominators
//! unexpanded through S/R calibration and graph-kernel construction.  Dense
//! symbolic stable-graph leg products remain the main performance frontier.
//! Fast validation currently covers several resolved-conifold rows and the
//! first local-`P^2` genus-2 row; genus-4 local curve computations are the next
//! observed performance frontier.

use crate::algebra::{Coeff, RatFun, Rational};
use crate::error::GwError;
use crate::factored::FactoredRatFun;
use crate::givental::{
    compute_semisimple_graph_value_with_coeff, CalibrationId, CanonicalFrameConvention,
    CoefficientSemisimpleCohftProvider, GiventalGraphKernel, ProjectiveSpaceProvider,
    SemisimpleCalibration, SemisimpleCohftProvider, SeriesRMatrix, SeriesSMatrix,
};
use crate::resolvent::{ResolventRequest, ResolventResult};
use crate::series::{
    compose_plain_series, integrate_q_derivative_zero_constant_matrix, invert_mirror_map,
    mul_plain_series, QSeries, SeriesMatrix,
};
use crate::{Insertion, InvariantResult, Truncation};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

mod birkhoff_factor;
use birkhoff_factor::*;
mod qseries_matrix;
use qseries_matrix::*;
mod numeric;
use numeric::*;
mod hypergeometric;
pub use hypergeometric::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NegativeSplitBundleTwist {
    degrees: Vec<usize>,
}

impl NegativeSplitBundleTwist {
    /// Split bundle `O(-a_1) + ... + O(-a_r)` over `P^n`.
    ///
    /// The stored degrees are the positive integers `a_i`; the signs are part
    /// of the type convention.  Negativity is what gives the concave Euler
    /// factors in the hypergeometric `I`-function below.
    pub fn new(degrees: Vec<usize>) -> Result<Self, GwError> {
        if degrees.contains(&0) {
            return Err(GwError::ParseError(
                "negative split-bundle degrees must be positive".to_string(),
            ));
        }
        Ok(Self { degrees })
    }

    pub fn degrees(&self) -> &[usize] {
        &self.degrees
    }

    pub fn rank(&self) -> usize {
        self.degrees.len()
    }

    pub fn degree_sum(&self) -> usize {
        self.degrees.iter().sum()
    }

    pub fn total_space_dimension(&self, base_n: usize) -> usize {
        base_n + self.rank()
    }

    pub fn is_calabi_yau(&self, base_n: usize) -> bool {
        self.degree_sum() == base_n + 1
    }

    pub fn virtual_dimension(
        &self,
        base_n: usize,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> isize {
        // Virtual dimension of maps to the total space of the split bundle:
        // (1-g)(dim total - 3) + c1(total).d + markings.
        let total_dimension = self.total_space_dimension(base_n) as isize;
        let degree_slope = base_n as isize + 1 - self.degree_sum() as isize;
        (1 - genus as isize) * (total_dimension - 3)
            + degree_slope * degree as isize
            + markings as isize
    }

    pub fn candidate_degrees(
        &self,
        base_n: usize,
        genus: usize,
        degree_max: usize,
        markings: usize,
        insertion_degree: Option<usize>,
    ) -> Vec<usize> {
        let Some(insertion_degree) = insertion_degree else {
            return (0..=degree_max).collect();
        };
        let constant_dimension = (1 - genus as isize)
            * (self.total_space_dimension(base_n) as isize - 3)
            + markings as isize;
        let numerator = insertion_degree as isize - constant_dimension;
        let slope = base_n as isize + 1 - self.degree_sum() as isize;
        if slope == 0 {
            return if numerator == 0 {
                (0..=degree_max).collect()
            } else {
                Vec::new()
            };
        }
        if numerator % slope != 0 {
            return Vec::new();
        }
        let degree = numerator / slope;
        if degree < 0 || degree as usize > degree_max {
            Vec::new()
        } else {
            vec![degree as usize]
        }
    }
}

/// The non-equivariant twisted Laurent series is the rational specialization
/// of the generic coefficient Laurent series.
pub type HLaurentSeries = HCoeffLaurentSeries<Rational>;

fn base_h_power_relation(n: usize, base_weights: &[Rational]) -> Result<Vec<Rational>, GwError> {
    base_h_power_relation_coeff(n, base_weights)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HCoeffLaurentSeries<C = RatFun> {
    max_h_power: usize,
    coeffs: Vec<BTreeMap<i32, C>>,
}

impl<C: Coeff> HCoeffLaurentSeries<C> {
    fn multiply_by_h(&self) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power + 1, *z_power, coeff.clone());
            }
        }
        out
    }

    fn multiply_by_linear(&self, h_coeff: C, z_coeff: C) -> Self {
        self.multiply_by_affine(h_coeff, C::zero(), z_coeff)
    }

    fn multiply_by_affine(&self, h_coeff: C, constant: C, z_coeff: C) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                if !constant.is_zero() {
                    out.add_term(h_power, *z_power, coeff.mul(&constant));
                }
                if !z_coeff.is_zero() {
                    out.add_term(h_power, z_power + 1, coeff.mul(&z_coeff));
                }
                if !h_coeff.is_zero() && h_power < self.max_h_power {
                    out.add_term(h_power + 1, *z_power, coeff.mul(&h_coeff));
                }
            }
        }
        out
    }

    fn multiply(&self, rhs: &Self) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        let mut out = Self::zero(self.max_h_power);
        for left_h in 0..=self.max_h_power {
            for (left_z, left_coeff) in &self.coeffs[left_h] {
                for right_h in 0..=self.max_h_power - left_h {
                    for (right_z, right_coeff) in &rhs.coeffs[right_h] {
                        out.add_term(
                            left_h + right_h,
                            left_z + right_z,
                            left_coeff.mul(right_coeff),
                        );
                    }
                }
            }
        }
        out
    }

    fn zero(max_h_power: usize) -> Self {
        Self {
            max_h_power,
            coeffs: vec![BTreeMap::new(); max_h_power + 1],
        }
    }

    fn one(max_h_power: usize) -> Self {
        let mut out = Self::zero(max_h_power);
        out.coeffs[0].insert(0, C::one());
        out
    }

    fn coefficient(&self, h_power: usize, z_power: i32) -> C {
        self.coeffs
            .get(h_power)
            .and_then(|terms| terms.get(&z_power))
            .cloned()
            .unwrap_or_else(C::zero)
    }

    fn max_h_power(&self) -> usize {
        self.max_h_power
    }

    fn is_empty(&self) -> bool {
        self.coeffs.iter().all(BTreeMap::is_empty)
    }

    fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        let mut out = self.clone();
        for h_power in 0..=rhs.max_h_power {
            for (z_power, coeff) in &rhs.coeffs[h_power] {
                out.add_term(h_power, *z_power, coeff.clone());
            }
        }
        out
    }

    fn scale(&self, scalar: C) -> Self {
        if scalar.is_zero() {
            return Self::zero(self.max_h_power);
        }
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power, *z_power, coeff.mul(&scalar));
            }
        }
        out
    }

    fn shift_z(&self, shift: i32) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power, z_power + shift, coeff.clone());
            }
        }
        out
    }

    fn multiply_mod_relation(&self, rhs: &Self, h_power_relation: &[C]) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        assert_eq!(h_power_relation.len(), self.max_h_power + 1);
        let basis_powers = h_basis_powers_mod_relation_coeff(self.max_h_power, h_power_relation);
        let mut out = Self::zero(self.max_h_power);
        for left_h in 0..=self.max_h_power {
            for (left_z, left_coeff) in &self.coeffs[left_h] {
                for right_h in 0..=self.max_h_power {
                    for (right_z, right_coeff) in &rhs.coeffs[right_h] {
                        let scalar = left_coeff.mul(right_coeff);
                        if scalar.is_zero() {
                            continue;
                        }
                        for (reduced_h, reduced_coeff) in
                            basis_powers[left_h + right_h].iter().enumerate()
                        {
                            if reduced_coeff.is_zero() {
                                continue;
                            }
                            out.add_term(reduced_h, left_z + right_z, scalar.mul(reduced_coeff));
                        }
                    }
                }
            }
        }
        out
    }

    fn multiply_by_affine_mod_relation(
        &self,
        h_coeff: C,
        constant: C,
        z_coeff: C,
        h_power_relation: &[C],
    ) -> Self {
        assert_eq!(h_power_relation.len(), self.max_h_power + 1);
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                if !constant.is_zero() {
                    out.add_term(h_power, *z_power, coeff.mul(&constant));
                }
                if !z_coeff.is_zero() {
                    out.add_term(h_power, z_power + 1, coeff.mul(&z_coeff));
                }
                if !h_coeff.is_zero() {
                    if h_power < self.max_h_power {
                        out.add_term(h_power + 1, *z_power, coeff.mul(&h_coeff));
                    } else {
                        for (reduced_h, reduced_coeff) in h_power_relation.iter().enumerate() {
                            if !reduced_coeff.is_zero() {
                                out.add_term(
                                    reduced_h,
                                    *z_power,
                                    coeff.mul(&h_coeff).mul(reduced_coeff),
                                );
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn truncated_z_below(&self, min_z_power: i32) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                if *z_power >= min_z_power {
                    out.add_term(h_power, *z_power, coeff.clone());
                }
            }
        }
        out
    }

    fn add_term(&mut self, h_power: usize, z_power: i32, coeff: C) {
        if coeff.is_zero() || h_power > self.max_h_power {
            return;
        }
        let terms = &mut self.coeffs[h_power];
        let next = terms
            .get(&z_power)
            .cloned()
            .unwrap_or_else(C::zero)
            .add(&coeff);
        if next.is_zero() {
            terms.remove(&z_power);
        } else {
            terms.insert(z_power, next);
        }
    }
}

fn h_basis_powers_mod_relation_coeff<C: Coeff>(
    max_h_power: usize,
    h_power_relation: &[C],
) -> Vec<Vec<C>> {
    let mut powers = vec![vec![C::zero(); max_h_power + 1]; 2 * max_h_power + 1];
    powers[0][0] = C::one();
    for power in 1..=2 * max_h_power {
        for h_power in 0..max_h_power {
            powers[power][h_power + 1] =
                powers[power][h_power + 1].add(&powers[power - 1][h_power]);
        }
        let top_coeff = powers[power - 1][max_h_power].clone();
        if !top_coeff.is_zero() {
            for (reduced_h, relation_coeff) in h_power_relation.iter().enumerate() {
                powers[power][reduced_h] =
                    powers[power][reduced_h].add(&top_coeff.mul(relation_coeff));
            }
        }
    }
    powers
}

fn h_affine_power_mod_relation_coeff<C: Coeff>(
    max_h_power: usize,
    h_coeff: C,
    constant: C,
    exponent: usize,
    h_power_relation: &[C],
) -> Vec<C> {
    let mut out = vec![C::zero(); max_h_power + 1];
    out[0] = C::one();
    for _ in 0..exponent {
        let mut next = vec![C::zero(); max_h_power + 1];
        for h_power in 0..=max_h_power {
            if out[h_power].is_zero() {
                continue;
            }
            if !constant.is_zero() {
                next[h_power] = next[h_power].add(&out[h_power].mul(&constant));
            }
            if !h_coeff.is_zero() {
                if h_power < max_h_power {
                    next[h_power + 1] = next[h_power + 1].add(&out[h_power].mul(&h_coeff));
                } else {
                    for (reduced_h, relation_coeff) in h_power_relation.iter().enumerate() {
                        next[reduced_h] =
                            next[reduced_h].add(&out[h_power].mul(&h_coeff).mul(relation_coeff));
                    }
                }
            }
        }
        out = next;
    }
    out
}

pub fn negative_split_i_function_coefficient(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
) -> HLaurentSeries {
    // Non-equivariant hypergeometric coefficient for local P^n.  The numerator
    // is the concave Euler factor for H^1(C,f^*O(-a)); the denominator is the
    // usual projective-space I-function denominator.
    let mut out = HLaurentSeries::one(n);
    for bundle_degree in twist.degrees() {
        for m in (-(bundle_degree.saturating_mul(degree) as isize) + 1)..=0 {
            out = out.multiply_by_linear(-Rational::from(*bundle_degree), Rational::from(m));
        }
    }
    for m in 1..=degree {
        let inverse = inverse_h_plus_mz_power(n, m, n + 1);
        out = out.multiply(&inverse);
    }
    out
}

pub fn negative_split_i_function_coefficients(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
) -> Vec<HLaurentSeries> {
    (0..=q_degree)
        .map(|degree| negative_split_i_function_coefficient(n, twist, degree))
        .collect()
}

pub fn projective_equivariant_i_function_coefficient(
    n: usize,
    degree: usize,
    base_weights: &[Rational],
    min_z_power: i32,
) -> Result<HLaurentSeries, GwError> {
    projective_equivariant_i_function_coefficient_coeff(n, degree, base_weights, min_z_power)
}

pub fn projective_i_function_coefficient(n: usize, degree: usize) -> HLaurentSeries {
    let mut out = HLaurentSeries::one(n);
    for m in 1..=degree {
        let inverse = inverse_h_plus_mz_power(n, m, n + 1);
        out = out.multiply(&inverse);
    }
    out
}

pub fn negative_split_equivariant_qrr_euler_factor(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<HLaurentSeries, GwError> {
    negative_split_equivariant_qrr_euler_factor_coeff(n, twist, degree, base_weights, fiber_weights)
}

pub fn negative_split_qrr_euler_factor(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
) -> HLaurentSeries {
    let mut out = HLaurentSeries::one(n);
    for bundle_degree in twist.degrees() {
        for m in (-(bundle_degree.saturating_mul(degree) as isize) + 1)..=0 {
            out = out.multiply_by_linear(-Rational::from(*bundle_degree), Rational::from(m));
        }
    }
    out
}

pub fn negative_split_equivariant_i_function_coefficient(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
    min_z_power: i32,
) -> Result<HLaurentSeries, GwError> {
    negative_split_equivariant_i_function_coefficient_coeff(
        n,
        twist,
        degree,
        base_weights,
        fiber_weights,
        min_z_power,
    )
}

fn negative_split_equivariant_i_function_coefficient_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[C],
    fiber_weights: &[C],
    min_z_power: i32,
) -> Result<HCoeffLaurentSeries<C>, GwError> {
    let h_power_relation = base_h_power_relation_coeff(n, base_weights)?;
    let factor = negative_split_equivariant_qrr_euler_factor_coeff(
        n,
        twist,
        degree,
        base_weights,
        fiber_weights,
    )?;
    let projective =
        projective_equivariant_i_function_coefficient_coeff(n, degree, base_weights, min_z_power)?;
    Ok(factor
        .multiply_mod_relation(&projective, &h_power_relation)
        .truncated_z_below(min_z_power))
}

fn projective_equivariant_i_function_coefficient_coeff<C: Coeff>(
    n: usize,
    degree: usize,
    base_weights: &[C],
    min_z_power: i32,
) -> Result<HCoeffLaurentSeries<C>, GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    let h_power_relation = base_h_power_relation_coeff(n, base_weights)?;
    let mut out = HCoeffLaurentSeries::<C>::one(n);
    for m in 1..=degree {
        for weight in base_weights {
            let inverse = inverse_affine_z_laurent_coeff(
                n,
                C::one(),
                weight.neg(),
                C::from_usize(m),
                min_z_power,
                Some(&h_power_relation),
            )?;
            out = out
                .multiply_mod_relation(&inverse, &h_power_relation)
                .truncated_z_below(min_z_power);
        }
    }
    Ok(out)
}

fn negative_split_equivariant_qrr_euler_factor_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[C],
    fiber_weights: &[C],
) -> Result<HCoeffLaurentSeries<C>, GwError> {
    if fiber_weights.len() != twist.rank() {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} fiber weights, got {}",
            twist.rank(),
            fiber_weights.len()
        )));
    }
    let h_power_relation = base_h_power_relation_coeff(n, base_weights)?;
    let mut out = HCoeffLaurentSeries::<C>::one(n);
    for (bundle_degree, fiber_weight) in twist.degrees().iter().zip(fiber_weights) {
        for m in (-(bundle_degree.saturating_mul(degree) as isize) + 1)..=0 {
            out = out.multiply_by_affine_mod_relation(
                C::from_rational(-Rational::from(*bundle_degree)),
                fiber_weight.clone(),
                C::from_rational(Rational::from(m)),
                &h_power_relation,
            );
        }
    }
    Ok(out)
}

fn base_h_power_relation_coeff<C: Coeff>(n: usize, base_weights: &[C]) -> Result<Vec<C>, GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    let mut coefficients = vec![C::one()];
    for weight in base_weights {
        let mut next = vec![C::zero(); coefficients.len() + 1];
        for (power, coeff) in coefficients.iter().enumerate() {
            next[power] = next[power].sub(&coeff.mul(weight));
            next[power + 1] = next[power + 1].add(coeff);
        }
        coefficients = next;
    }
    let leading = coefficients[n + 1].clone();
    if leading.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    Ok((0..=n)
        .map(|power| coefficients[power].neg().div(&leading))
        .collect())
}

pub fn negative_split_inverse_mirror_map_coefficients(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
) -> Vec<Rational> {
    let mirror = negative_split_mirror_map_coefficients(n, twist, q_degree);
    invert_mirror_map(&mirror, q_degree)
}

fn mirror_map_coefficients_from_i_function(
    i_coefficients: &[HLaurentSeries],
    q_degree: usize,
) -> Vec<Rational> {
    mirror_map_coefficients_from_i_function_coeff(i_coefficients, q_degree)
}

fn mirror_map_coefficients_from_i_function_coeff<C: Coeff>(
    i_coefficients: &[HCoeffLaurentSeries<C>],
    q_degree: usize,
) -> Vec<C> {
    let mut out = vec![C::zero(); q_degree + 1];
    let Some(first) = i_coefficients.first() else {
        return out;
    };
    if first.max_h_power() == 0 {
        return out;
    }
    for (degree, coeff) in out.iter_mut().enumerate().take(q_degree + 1).skip(1) {
        *coeff = i_coefficients
            .get(degree)
            .map(|i_degree| i_degree.coefficient(1, -1))
            .unwrap_or_else(C::zero);
    }
    out
}

fn mirror_transformed_j_coefficients_from_i_function(
    n: usize,
    i_coefficients: &[HLaurentSeries],
    mirror: &[Rational],
    inverse_mirror: &[Rational],
    q_degree: usize,
) -> Vec<HLaurentSeries> {
    // Applies the usual mirror transform: remove the H/z term by the exponential
    // gauge, then re-expand in the flat coordinate using the inverse mirror map.
    let gauge = exp_minus_h_mirror_over_z_coefficients(n, mirror, q_degree);
    let gauged = multiply_h_laurent_q_series(&gauge, i_coefficients, q_degree);
    compose_h_laurent_q_series(&gauged, inverse_mirror, q_degree)
}

fn mirror_transformed_j_coefficients_from_i_function_mod_relation(
    n: usize,
    i_coefficients: &[HLaurentSeries],
    _mirror: &[Rational],
    _inverse_mirror: &[Rational],
    q_degree: usize,
    h_power_relation: &[Rational],
) -> Vec<HLaurentSeries> {
    mirror_transformed_j_coefficients_from_i_function_mod_relation_coeff(
        n,
        i_coefficients,
        _mirror,
        _inverse_mirror,
        q_degree,
        h_power_relation,
    )
}

fn mirror_transformed_j_coefficients_from_i_function_mod_relation_coeff<C: Coeff>(
    n: usize,
    i_coefficients: &[HCoeffLaurentSeries<C>],
    _mirror: &[C],
    inverse_mirror: &[C],
    q_degree: usize,
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let gauge =
        full_vector_mirror_gauge_coefficients_coeff(n, i_coefficients, q_degree, h_power_relation);
    let gauged = multiply_h_laurent_q_series_mod_relation_coeff(
        &gauge,
        i_coefficients,
        q_degree,
        h_power_relation,
    );
    compose_h_laurent_q_series_coeff(&gauged, inverse_mirror, q_degree)
}

fn fundamental_solution_matrix_from_j_coefficients(
    n: usize,
    q_degree: usize,
    j_coefficients: &[HLaurentSeries],
) -> BTreeMap<i32, SeriesMatrix> {
    // The quantum connection fundamental solution is generated from J by
    // repeated application of z q d/dq.  Each derivative gives one flat-basis
    // column.
    let mut columns = Vec::with_capacity(n + 1);
    let mut current = j_coefficients.to_vec();
    for _ in 0..=n {
        columns.push(current.clone());
        current = quantum_derivative_h_laurent_q_series(&current);
    }
    h_laurent_columns_to_laurent_matrix(n, q_degree, &columns)
}

fn fundamental_solution_matrix_from_j_coefficients_mod_relation(
    n: usize,
    q_degree: usize,
    j_coefficients: &[HLaurentSeries],
    h_power_relation: &[Rational],
) -> BTreeMap<i32, SeriesMatrix> {
    let mut columns = Vec::with_capacity(n + 1);
    let mut current = j_coefficients.to_vec();
    for _ in 0..=n {
        columns.push(current.clone());
        current = quantum_derivative_h_laurent_q_series_mod_relation(&current, h_power_relation);
    }
    h_laurent_columns_to_laurent_matrix(n, q_degree, &columns)
}

fn fundamental_solution_matrix_from_j_coefficients_mod_relation_coeff<C: Coeff>(
    n: usize,
    q_degree: usize,
    j_coefficients: &[HCoeffLaurentSeries<C>],
    h_power_relation: &[C],
) -> BTreeMap<i32, SeriesMatrix<C>> {
    let mut columns = Vec::with_capacity(n + 1);
    let mut current = j_coefficients.to_vec();
    for _ in 0..=n {
        columns.push(current.clone());
        current =
            quantum_derivative_h_laurent_q_series_mod_relation_coeff(&current, h_power_relation);
    }
    h_coeff_laurent_columns_to_laurent_matrix(n, q_degree, &columns)
}

fn birkhoff_descendant_s_matrix_from_fundamental(
    size: usize,
    q_degree: usize,
    z_order: usize,
    fundamental: &BTreeMap<i32, SeriesMatrix>,
    calibration: CalibrationId,
) -> Result<SeriesSMatrix, GwError> {
    birkhoff_descendant_s_matrix_from_fundamental_coeff(
        size,
        q_degree,
        z_order,
        fundamental,
        calibration,
    )
}

fn birkhoff_descendant_s_matrix_from_fundamental_coeff<C: Coeff>(
    size: usize,
    q_degree: usize,
    z_order: usize,
    fundamental: &BTreeMap<i32, SeriesMatrix<C>>,
    calibration: CalibrationId,
) -> Result<SeriesSMatrix<C>, GwError> {
    // Birkhoff factorization splits the Laurent fundamental solution into
    // S(z^{-1})^{-1} * P(z).  We keep the negative factor and convert its
    // z^{-k} terms into the descendant S-matrix coefficients.
    let (_, s_factor) = birkhoff_factor_by_q_degree(size, q_degree, fundamental)?;
    let coefficients = negative_factor_to_s_coefficients(size, q_degree, z_order, &s_factor);
    SeriesSMatrix::from_coefficients(size, q_degree, z_order, coefficients, calibration)
}

fn exp_minus_h_mirror_over_z_coefficients(
    n: usize,
    mirror: &[Rational],
    max_degree: usize,
) -> Vec<HLaurentSeries> {
    let mut exponent = vec![HLaurentSeries::zero(n); max_degree + 1];
    for degree in 1..=max_degree {
        let coeff = mirror.get(degree).cloned().unwrap_or_else(Rational::zero);
        if !coeff.is_zero() && n >= 1 {
            exponent[degree].add_term(1, -1, -coeff);
        }
    }

    let mut out = vec![HLaurentSeries::zero(n); max_degree + 1];
    out[0] = HLaurentSeries::one(n);
    for degree in 1..=max_degree {
        let mut sum = HLaurentSeries::zero(n);
        for split in 1..=degree {
            if exponent[split].coeffs.iter().all(BTreeMap::is_empty) {
                continue;
            }
            let term = exponent[split]
                .multiply(&out[degree - split])
                .scale(Rational::from(split));
            sum = sum.add(&term);
        }
        out[degree] = sum.scale(Rational::one() / Rational::from(degree));
    }
    out
}

fn multiply_h_laurent_q_series(
    left: &[HLaurentSeries],
    right: &[HLaurentSeries],
    max_degree: usize,
) -> Vec<HLaurentSeries> {
    let max_h_power = left
        .first()
        .or_else(|| right.first())
        .map(HLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HLaurentSeries::zero(max_h_power); max_degree + 1];
    for left_degree in 0..=max_degree {
        for right_degree in 0..=max_degree - left_degree {
            let term = left[left_degree].multiply(&right[right_degree]);
            out[left_degree + right_degree] = out[left_degree + right_degree].add(&term);
        }
    }
    out
}

fn multiply_h_laurent_q_series_mod_relation_coeff<C: Coeff>(
    left: &[HCoeffLaurentSeries<C>],
    right: &[HCoeffLaurentSeries<C>],
    max_degree: usize,
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let max_h_power = left
        .first()
        .or_else(|| right.first())
        .map(HCoeffLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HCoeffLaurentSeries::<C>::zero(max_h_power); max_degree + 1];
    for left_degree in 0..=max_degree {
        for right_degree in 0..=max_degree - left_degree {
            let term =
                left[left_degree].multiply_mod_relation(&right[right_degree], h_power_relation);
            out[left_degree + right_degree] = out[left_degree + right_degree].add(&term);
        }
    }
    out
}

fn full_vector_mirror_gauge_coefficients_coeff<C: Coeff>(
    n: usize,
    i_coefficients: &[HCoeffLaurentSeries<C>],
    max_degree: usize,
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let mut exponent = vec![HCoeffLaurentSeries::<C>::zero(n); max_degree + 1];
    let mut gauge = vec![HCoeffLaurentSeries::<C>::zero(n); max_degree + 1];
    gauge[0] = HCoeffLaurentSeries::<C>::one(n);

    for degree in 1..=max_degree {
        let mut known_gauge = HCoeffLaurentSeries::<C>::zero(n);
        for split in 1..degree {
            if exponent[split].is_empty() {
                continue;
            }
            let term = exponent[split]
                .multiply_mod_relation(&gauge[degree - split], h_power_relation)
                .scale(C::from_usize(split));
            known_gauge = known_gauge.add(&term);
        }
        known_gauge = known_gauge.scale(C::one().div(&C::from_usize(degree)));
        gauge[degree] = known_gauge;

        let mut gauged_degree = HCoeffLaurentSeries::<C>::zero(n);
        for split in 0..=degree {
            let term = gauge[split]
                .multiply_mod_relation(&i_coefficients[degree - split], h_power_relation);
            gauged_degree = gauged_degree.add(&term);
        }
        let tau = z_power_part_coeff(&gauged_degree, -1);
        exponent[degree] = tau.shift_z(-1).scale(C::from_rational(-Rational::one()));
        gauge[degree] = gauge[degree].add(&exponent[degree]);
    }

    gauge
}

fn z_power_part_coeff<C: Coeff>(
    series: &HCoeffLaurentSeries<C>,
    z_power: i32,
) -> HCoeffLaurentSeries<C> {
    let mut out = HCoeffLaurentSeries::<C>::zero(series.max_h_power());
    for h_power in 0..=series.max_h_power() {
        let coeff = series.coefficient(h_power, z_power);
        if !coeff.is_zero() {
            out.add_term(h_power, 0, coeff);
        }
    }
    out
}

fn compose_h_laurent_q_series(
    series: &[HLaurentSeries],
    input: &[Rational],
    max_degree: usize,
) -> Vec<HLaurentSeries> {
    compose_h_laurent_q_series_coeff(series, input, max_degree)
}

fn compose_h_laurent_q_series_coeff<C: Coeff>(
    series: &[HCoeffLaurentSeries<C>],
    input: &[C],
    max_degree: usize,
) -> Vec<HCoeffLaurentSeries<C>> {
    let max_h_power = series
        .first()
        .map(HCoeffLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HCoeffLaurentSeries::<C>::zero(max_h_power); max_degree + 1];
    let mut power = vec![C::zero(); max_degree + 1];
    power[0] = C::one();
    for source_degree in 0..=max_degree {
        for target_degree in 0..=max_degree {
            if power[target_degree].is_zero() {
                continue;
            }
            let term = series[source_degree].scale(power[target_degree].clone());
            out[target_degree] = out[target_degree].add(&term);
        }
        power = mul_plain_series(&power, input, max_degree);
    }
    out
}

fn quantum_derivative_h_laurent_q_series(series: &[HLaurentSeries]) -> Vec<HLaurentSeries> {
    let max_degree = series.len().saturating_sub(1);
    let max_h_power = series.first().map(HLaurentSeries::max_h_power).unwrap_or(0);
    let mut out = vec![HLaurentSeries::zero(max_h_power); max_degree + 1];
    for degree in 0..=max_degree {
        out[degree] = out[degree].add(&series[degree].multiply_by_h());
        if degree > 0 {
            let derivative_term = series[degree].shift_z(1).scale(Rational::from(degree));
            out[degree] = out[degree].add(&derivative_term);
        }
    }
    out
}

fn quantum_derivative_h_laurent_q_series_mod_relation_coeff<C: Coeff>(
    series: &[HCoeffLaurentSeries<C>],
    h_power_relation: &[C],
) -> Vec<HCoeffLaurentSeries<C>> {
    let max_degree = series.len().saturating_sub(1);
    let max_h_power = series
        .first()
        .map(HCoeffLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HCoeffLaurentSeries::<C>::zero(max_h_power); max_degree + 1];
    for degree in 0..=max_degree {
        out[degree] = out[degree].add(&series[degree].multiply_by_affine_mod_relation(
            C::one(),
            C::zero(),
            C::zero(),
            h_power_relation,
        ));
        if degree > 0 {
            let derivative_term = series[degree].shift_z(1).scale(C::from_usize(degree));
            out[degree] = out[degree].add(&derivative_term);
        }
    }
    out
}

fn quantum_derivative_h_laurent_q_series_mod_relation(
    series: &[HLaurentSeries],
    h_power_relation: &[Rational],
) -> Vec<HLaurentSeries> {
    quantum_derivative_h_laurent_q_series_mod_relation_coeff(series, h_power_relation)
}

fn h_coeff_laurent_columns_to_laurent_matrix<C: Coeff>(
    n: usize,
    q_degree: usize,
    columns: &[Vec<HCoeffLaurentSeries<C>>],
) -> BTreeMap<i32, SeriesMatrix<C>> {
    let size = n + 1;
    let mut by_power: BTreeMap<i32, Vec<Vec<Vec<C>>>> = BTreeMap::new();
    for (col, column_series) in columns.iter().enumerate() {
        for (degree, h_series) in column_series.iter().enumerate().take(q_degree + 1) {
            for h_power in 0..=n {
                for (z_power, coeff) in &h_series.coeffs[h_power] {
                    let entries = by_power
                        .entry(*z_power)
                        .or_insert_with(|| vec![vec![vec![C::zero(); q_degree + 1]; size]; size]);
                    entries[h_power][col][degree] = entries[h_power][col][degree].add(coeff);
                }
            }
        }
    }

    by_power
        .into_iter()
        .map(|(z_power, entries)| {
            let matrix = SeriesMatrix::from_entries(
                entries
                    .into_iter()
                    .map(|row| row.into_iter().map(QSeries::from_coeffs).collect())
                    .collect(),
            );
            (z_power, matrix)
        })
        .collect()
}

fn h_laurent_columns_to_laurent_matrix(
    n: usize,
    q_degree: usize,
    columns: &[Vec<HLaurentSeries>],
) -> BTreeMap<i32, SeriesMatrix> {
    let size = n + 1;
    let mut by_power: BTreeMap<i32, Vec<Vec<Vec<RatFun>>>> = BTreeMap::new();
    for (col, column_series) in columns.iter().enumerate() {
        for (degree, h_series) in column_series.iter().enumerate().take(q_degree + 1) {
            for h_power in 0..=n {
                for (z_power, coeff) in &h_series.coeffs[h_power] {
                    let entries = by_power.entry(*z_power).or_insert_with(|| {
                        vec![vec![vec![RatFun::zero(); q_degree + 1]; size]; size]
                    });
                    entries[h_power][col][degree] = entries[h_power][col][degree].clone()
                        + RatFun::from_rational(coeff.clone());
                }
            }
        }
    }

    by_power
        .into_iter()
        .map(|(z_power, entries)| {
            let matrix = SeriesMatrix::from_entries(
                entries
                    .into_iter()
                    .map(|row| row.into_iter().map(QSeries::from_coeffs).collect())
                    .collect(),
            );
            (z_power, matrix)
        })
        .collect()
}

fn inverse_h_plus_mz_power(max_h_power: usize, m: usize, power: usize) -> HLaurentSeries {
    let mut out = HLaurentSeries::zero(max_h_power);
    let m = Rational::from(m);
    for h_power in 0..=max_h_power {
        let coefficient =
            signed_binomial_negative_power(power, h_power) / m.pow_usize(power + h_power);
        out.add_term(h_power, -((power + h_power) as i32), coefficient);
    }
    out
}

fn inverse_affine_z_laurent_coeff<C: Coeff>(
    max_h_power: usize,
    h_coeff: C,
    constant: C,
    z_coeff: C,
    min_z_power: i32,
    h_power_relation: Option<&[C]>,
) -> Result<HCoeffLaurentSeries<C>, GwError> {
    if z_coeff.is_zero() {
        return Err(GwError::AlgebraFailure(
            "cannot expand affine inverse at z=infinity with zero z coefficient".to_string(),
        ));
    }
    if min_z_power >= 0 {
        return Ok(HCoeffLaurentSeries::<C>::zero(max_h_power));
    }

    let mut out = HCoeffLaurentSeries::<C>::zero(max_h_power);
    let max_k = (-min_z_power - 1) as usize;
    for k in 0..=max_k {
        let sign = if k % 2 == 0 {
            C::one()
        } else {
            C::from_rational(-Rational::one())
        };
        let denominator = z_coeff.pow_usize(k + 1);
        if let Some(relation) = h_power_relation {
            let affine_power = h_affine_power_mod_relation_coeff(
                max_h_power,
                h_coeff.clone(),
                constant.clone(),
                k,
                relation,
            );
            for (h_power, coeff) in affine_power.into_iter().enumerate() {
                out.add_term(
                    h_power,
                    -((k + 1) as i32),
                    sign.mul(&coeff).div(&denominator),
                );
            }
        } else {
            for h_power in 0..=max_h_power.min(k) {
                let coeff = sign
                    .mul(&C::from_rational(binomial_rational(k, h_power)))
                    .mul(&constant.pow_usize(k - h_power))
                    .mul(&h_coeff.pow_usize(h_power))
                    .div(&denominator);
                out.add_term(h_power, -((k + 1) as i32), coeff);
            }
        }
    }
    Ok(out)
}

fn signed_binomial_negative_power(power: usize, h_power: usize) -> Rational {
    let sign = if h_power.is_multiple_of(2) {
        Rational::one()
    } else {
        -Rational::one()
    };
    sign * binomial_rational(power + h_power - 1, h_power)
}

fn binomial_rational(n: usize, k: usize) -> Rational {
    if k > n {
        return Rational::zero();
    }
    let k = k.min(n - k);
    let mut out = Rational::one();
    for idx in 0..k {
        out = out * Rational::from(n - idx) / Rational::from(idx + 1);
    }
    out
}

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
            base = multiply_polynomial_by_linear_series(
                &base,
                &QSeries::constant(RatFun::from_rational(-weight.clone()), q_degree),
                q_degree,
            );
        }

        let mut fiber = vec![QSeries::one(q_degree)];
        for degree in self.twist.degrees() {
            let factor = RatFun::from_rational(-Rational::from(*degree));
            for _ in 0..*degree {
                fiber = multiply_polynomial_by_linear_series(
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
    pub quantum_h: SeriesMatrix<C>,
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

fn specialized_twisted_birkhoff_canonical_data_with_mode(
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

fn specialized_twisted_birkhoff_canonical_data_with_mode_and_validation(
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

    Ok(SpecializedTwistedBirkhoffCanonicalData {
        roots,
        metric_norms,
        inverse_metric_norms,
        transition_to_flat,
        quantum_h,
    })
}

fn specialized_twisted_birkhoff_canonical_data_for_coeff_weights_with_validation<C: Coeff>(
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

    Ok(SpecializedTwistedBirkhoffCanonicalData {
        roots,
        metric_norms,
        inverse_metric_norms,
        transition_to_flat,
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

fn negative_split_twisted_birkhoff_calibration_skeleton_with_mode(
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

fn negative_split_twisted_birkhoff_calibration_skeleton_from_canonical(
    q_degree: usize,
    z_order: usize,
    canonical: &SpecializedTwistedBirkhoffCanonicalData,
) -> Result<SemisimpleCalibration, GwError> {
    negative_split_twisted_birkhoff_calibration_skeleton_from_canonical_coeff(
        q_degree, z_order, canonical,
    )
}

fn negative_split_twisted_birkhoff_calibration_skeleton_from_canonical_coeff<C: Coeff>(
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
    let transition_inverse = invert_series_matrix_coeff(&transition)?;
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

fn negative_split_twisted_birkhoff_calibration_candidate_with_mode(
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

fn negative_split_twisted_birkhoff_calibration_candidate_with_mode_and_validation(
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
    let coefficients = solve_twisted_r_coefficients(
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

fn negative_split_twisted_birkhoff_calibration_candidate_for_ratfun_weights_with_validation(
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

fn negative_split_twisted_birkhoff_calibration_candidate_for_coeff_weights_with_validation<
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
    let coefficients = solve_twisted_r_coefficients_coeff(
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
    let coefficients = solve_twisted_r_coefficients(
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
            numerator =
                multiply_polynomial_by_linear_series(&numerator, &roots[other].neg(), max_q_degree);
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

fn validate_twisted_local_cy_weights(
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

fn validate_twisted_weights(
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

fn twisted_canonical_root_series_at_weights(
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

fn twisted_relation_series_at_weights(
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

fn twisted_relation_derivative_series_at_weights(
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

fn twisted_base_polynomial_coefficients(
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
        out = multiply_polynomial_by_linear_series(
            &out,
            &QSeries::constant(RatFun::from_rational(-weight.clone()), q_degree),
            q_degree,
        );
    }
    Ok(out)
}

fn twisted_base_polynomial_coefficients_coeff<C: Coeff>(
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
        out = multiply_polynomial_by_linear_series(
            &out,
            &QSeries::constant(weight.neg(), q_degree),
            q_degree,
        );
    }
    Ok(out)
}

fn twisted_fiber_polynomial_coefficients(
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
            out = multiply_polynomial_by_affine_h_series(
                &out,
                &QSeries::constant(RatFun::from_rational(weight.clone()), q_degree),
                &QSeries::constant(RatFun::from_rational(-Rational::from(*degree)), q_degree),
                q_degree,
            );
        }
    }
    Ok(out)
}

fn twisted_flat_metric_matrix(
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

fn twisted_inverse_euler_flat_metric_matrix(
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

fn twisted_inverse_euler_flat_metric_matrix_ratfun(
    n: usize,
    q_degree: usize,
    twist: &NegativeSplitBundleTwist,
    base_weights: &[RatFun],
    fiber_weights: &[RatFun],
) -> Result<SeriesMatrix, GwError> {
    twisted_inverse_euler_flat_metric_matrix_coeff(n, q_degree, twist, base_weights, fiber_weights)
}

fn twisted_inverse_euler_flat_metric_matrix_coeff<C: Coeff>(
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

fn twisted_inverse_euler_flat_metric_pair_from_rational_base<C: Coeff>(
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

fn lagrange_basis_coefficients(
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

fn twisted_fiber_euler_at_fixed_point(
    twist: &NegativeSplitBundleTwist,
    fiber_weights: &[Rational],
    lambda: &Rational,
) -> Rational {
    twisted_fiber_euler_at_fixed_point_coeff(twist, fiber_weights, lambda)
}

fn twisted_fiber_euler_at_fixed_point_coeff<C: Coeff>(
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

fn canonical_metric_norm_from_flat_metric(
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

fn twisted_classical_h_multiplication_matrix(
    n: usize,
    q_degree: usize,
    base_weights: &[Rational],
) -> Result<SeriesMatrix, GwError> {
    let coefficients = twisted_base_polynomial_coefficients(n, q_degree, base_weights)?;
    companion_multiplication_matrix_from_monic_polynomial(n + 1, &coefficients)
}

fn twisted_classical_h_multiplication_matrix_coeff<C: Coeff>(
    n: usize,
    q_degree: usize,
    base_weights: &[C],
) -> Result<SeriesMatrix<C>, GwError> {
    let coefficients = twisted_base_polynomial_coefficients_coeff(n, q_degree, base_weights)?;
    companion_multiplication_matrix_from_monic_polynomial(n + 1, &coefficients)
}

fn companion_multiplication_matrix_from_monic_polynomial<C: Coeff>(
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

fn twisted_quantum_multiplication_from_s(
    descendant_s: &SeriesSMatrix,
    classical_h: &SeriesMatrix,
    mode: &TwistedCalibrationMode,
) -> Result<SeriesMatrix, GwError> {
    twisted_quantum_multiplication_from_s_coeff(descendant_s, classical_h, mode)
}

fn twisted_quantum_multiplication_from_s_coeff<C: Coeff>(
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

fn twisted_quantum_multiplication_from_picard_fuchs(
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

fn substitute_scaled_generator_in_polynomial(
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

fn charpoly_qseries_coefficients(matrix: &SeriesMatrix) -> Result<Vec<QSeries>, GwError> {
    charpoly_qseries_coefficients_coeff(matrix)
}

fn charpoly_qseries_coefficients_coeff<C: Coeff>(
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

fn root_series_from_charpoly(
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

fn root_series_from_charpoly_coeff<C: Coeff>(
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

fn spectral_transition_matrix_from_roots(
    multiplication: &SeriesMatrix,
    roots: &[QSeries],
) -> Result<SeriesMatrix, GwError> {
    spectral_transition_matrix_from_roots_coeff(multiplication, roots)
}

fn spectral_transition_matrix_from_roots_coeff<C: Coeff>(
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

fn twisted_classical_limit_diagonal_coefficients(
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

fn twisted_classical_limit_diagonal_coefficients_with_mode(
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

fn twisted_classical_limit_diagonal_coefficients_coeff<C: Coeff>(
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

fn twisted_classical_limit_diagonal_coefficients_for_branch(
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
        let coefficient =
            bernoulli_number_local(2 * r) / (Rational::from(2 * r) * Rational::from(2 * r - 1));
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
    Ok(exp_scalar_z_series_local(&exponent))
}

fn twisted_classical_limit_diagonal_coefficients_for_branch_coeff<C: Coeff>(
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
        let coefficient = C::from_rational(
            bernoulli_number_local(2 * r) / (Rational::from(2 * r) * Rational::from(2 * r - 1)),
        );
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
    Ok(exp_scalar_z_series_coeff(&exponent))
}

fn solve_twisted_r_coefficients(
    roots: &[QSeries],
    connection: &SeriesMatrix,
    classical_diagonal: &[Vec<RatFun>],
    q_degree: usize,
    z_order: usize,
) -> Result<Vec<SeriesMatrix>, GwError> {
    solve_twisted_r_coefficients_coeff(roots, connection, classical_diagonal, q_degree, z_order)
}

fn solve_twisted_r_coefficients_coeff<C: Coeff>(
    roots: &[QSeries<C>],
    connection: &SeriesMatrix<C>,
    classical_diagonal: &[Vec<C>],
    q_degree: usize,
    z_order: usize,
) -> Result<Vec<SeriesMatrix<C>>, GwError> {
    // Flatness recursion in canonical coordinates.  Off-diagonal entries are
    // determined by dividing by root differences u_j-u_i; diagonal entries are
    // integrated from the diagonal flatness equation with the classical
    // Bernoulli/QRR value as the q^0 constant.
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
            entries[branch][branch] = solve_twisted_r_diagonal_from_flatness_coeff(
                connection,
                &entries,
                branch,
                classical_diagonal[branch][order].clone(),
                q_degree,
            );
        }

        coefficients.push(SeriesMatrix::from_entries(entries));
    }

    Ok(coefficients)
}

fn solve_twisted_r_diagonal_from_flatness_coeff<C: Coeff>(
    connection: &SeriesMatrix<C>,
    entries: &[Vec<QSeries<C>>],
    branch: usize,
    constant: C,
    q_degree: usize,
) -> QSeries<C> {
    // Solves (q d/dq + A_ii) R_k,ii = known one q-coefficient at a time.
    // This is the piece that prevents the diagonal gauge from being silently
    // frozen at its q^0 value.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwistedProjectiveSpaceProvider {
    base: ProjectiveSpaceProvider,
    twist: NegativeSplitBundleTwist,
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
    fiber_weight_scale: Rational,
    custom_fiber_weights: Option<Vec<Rational>>,
    fiber_parameter_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TwistedLineMode {
    EarlyRational,
    SymbolicLimit,
    FiberEquivariant,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TwistedCalibrationMode {
    InverseEuler,
    InverseEulerFiberPlus,
    Euler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TwistedCalibrationValidation {
    /// Skip diagnostic identities that are expensive and do not change the
    /// produced calibration.  This is the default in the graph-evaluation path.
    Fast,
    /// Run full self-adjointness, diagonalization, and R-unitarity checks.
    Full,
}

impl TwistedCalibrationValidation {
    fn runs_expensive_checks(self) -> bool {
        matches!(self, Self::Full)
    }
}

fn twisted_calibration_validation_from_env() -> TwistedCalibrationValidation {
    // Expensive identities are off in the hot graph-evaluation path by default.
    // Set either variable to 1/true/yes/on/full when debugging a calibration
    // change and wanting self-adjointness, diagonalization, and unitarity checks
    // to run before caching the graph kernel.
    if env_flag_enabled("GWAI_VALIDATE_TWISTED_CALIBRATION")
        || env_flag_enabled("GW_VALIDATE_CALIBRATION")
    {
        TwistedCalibrationValidation::Full
    } else {
        TwistedCalibrationValidation::Fast
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "full"
            )
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TwistedGraphKernelCacheKey {
    n: usize,
    twist_degrees: Vec<usize>,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
    base_weights: Vec<Rational>,
    fiber_weights: Vec<Rational>,
    fiber_parameter_names: Vec<String>,
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
    validation: TwistedCalibrationValidation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwistedInvariantRequest {
    pub n: usize,
    pub twist: NegativeSplitBundleTwist,
    pub genus: usize,
    pub degree: usize,
    pub insertions: Vec<Insertion>,
    pub equivariant: bool,
    pub truncation: Option<Truncation>,
}

impl TwistedInvariantRequest {
    pub fn new(
        n: usize,
        degrees: Vec<usize>,
        genus: usize,
        degree: usize,
        insertions: Vec<Insertion>,
    ) -> Result<Self, GwError> {
        Ok(Self {
            n,
            twist: NegativeSplitBundleTwist::new(degrees)?,
            genus,
            degree,
            insertions,
            equivariant: false,
            truncation: None,
        })
    }
}

pub fn compute_negative_split_twisted(
    req: &TwistedInvariantRequest,
) -> Result<InvariantResult, GwError> {
    // Public local/twisted path: construct a semisimple provider from the
    // hypergeometric/Birkhoff/QRR calibration, run the generic Givental graph
    // evaluator, and either take the one-parameter lambda-line limit or keep the
    // fiber equivariant parameters as symbolic rational variables.
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = if req.equivariant {
        TwistedProjectiveSpaceProvider::fiber_equivariant(req.n, req.twist.degrees().to_vec())?
    } else {
        TwistedProjectiveSpaceProvider::new(req.n, req.twist.degrees().to_vec(), false)?
    };
    if !req.equivariant {
        if let Some((virtual_dimension, total_degree)) =
            twisted_dimension_mismatch(&provider, req.genus, req.degree, &req.insertions)
        {
            return Ok(InvariantResult {
                value: RatFun::zero(),
                engine: "twisted-negative-split-dimension",
                notes: vec![format!(
                    "dimension mismatch gives zero: virtual dimension {virtual_dimension}, insertion degree {total_degree}"
                )],
            });
        }
    }

    let unstable_two_point = req.genus == 0 && req.insertions.len() == 2;
    let primary_three_point = req.genus == 0 && genus_zero_three_primary_layout(&req.insertions);
    let raw = crate::givental::compute_semisimple_graph_value(
        &provider,
        req.genus,
        req.degree,
        &req.insertions,
        req.truncation.as_ref(),
    )?;
    let value = if req.equivariant {
        raw.lambda_line_limit_preserving_variables(req.n, provider.base_weights())?
    } else {
        match raw.as_rational() {
            Some(value) => RatFun::from_rational(value),
            None => RatFun::from_rational(raw.nonequivariant_limit_line(0, &[Rational::one()])?),
        }
    };
    Ok(InvariantResult {
        value,
        engine: if req.equivariant {
            "twisted-negative-split-fiber-equivariant-givental-birkhoff"
        } else {
            "twisted-negative-split-givental-birkhoff-early-line"
        },
        notes: vec![if unstable_two_point {
            if req.equivariant {
                "computed by the genus-zero two-point unstable S-matrix convention from the fiber-equivariant hypergeometric/Birkhoff calibration; base weights are early-specialized and fiber weights are symbolic mu_i"
                    .to_string()
            } else {
                "computed by the genus-zero two-point unstable S-matrix convention from the same early rational one-parameter lambda-line hypergeometric/Birkhoff calibration; no local oracle shortcut is used"
                    .to_string()
            }
        } else if primary_three_point {
            if req.equivariant {
                "computed by the fiber-equivariant twisted genus-zero Frobenius quantum product from the same hypergeometric/Birkhoff calibration; base weights are early-specialized and fiber weights are symbolic mu_i"
                    .to_string()
            } else {
                "computed by the twisted genus-zero Frobenius quantum product from the same early rational hypergeometric/Birkhoff calibration"
                    .to_string()
            }
        } else if req.equivariant {
            "computed by fiber-equivariant hypergeometric/Birkhoff S and QRR R stable-graph expansion; base weights are early-specialized and fiber weights are symbolic mu_i"
                .to_string()
        } else {
            "computed by early rational one-parameter lambda-line hypergeometric/Birkhoff S and QRR R stable-graph expansion; no local oracle shortcut is used; fast validation currently covers resolved conifold genus 2 degree 1"
                .to_string()
        }],
    })
}

pub fn compute_negative_split_twisted_factored(
    req: &TwistedInvariantRequest,
) -> Result<FactoredRatFun, GwError> {
    if !req.equivariant {
        return Err(GwError::UnsupportedInvariant(
            "factored twisted mode is currently for fiber-equivariant computations; pass --equivariant"
                .to_string(),
        ));
    }
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(
        req.n,
        req.twist.degrees().to_vec(),
    )?;
    if twisted_dimension_mismatch(provider.inner(), req.genus, req.degree, &req.insertions)
        .is_some()
    {
        return compute_negative_split_twisted(req)
            .map(|result| FactoredRatFun::from_ratfun(result.value));
    }
    compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
        &provider,
        req.genus,
        req.degree,
        &req.insertions,
        req.truncation.as_ref(),
    )
}

pub fn compute_negative_split_twisted_resolvent_packed(
    target_n: usize,
    degrees: Vec<usize>,
    req: &ResolventRequest,
    equivariant: bool,
) -> Result<ResolventResult, GwError> {
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = if equivariant {
        TwistedProjectiveSpaceProvider::fiber_equivariant(target_n, degrees)?
    } else {
        TwistedProjectiveSpaceProvider::new(target_n, degrees, false)?
    };
    crate::givental::compute_packed_resolvent_with_provider(
        req,
        provider,
        if equivariant {
            "twisted-negative-split-fiber-equivariant-packed-resolvent"
        } else {
            "twisted-negative-split-packed-resolvent"
        },
        if equivariant {
            "computed by packed fiber-equivariant twisted S/R external-leg graph kernel; base weights are early-specialized and fiber weights are symbolic mu_i"
        } else {
            "computed by packed twisted S/R external-leg graph kernel; all resolvent coefficients share one stable-graph contraction"
        },
        move |raw| {
            if equivariant {
                return Ok(raw);
            }
            match raw.as_rational() {
                Some(value) => Ok(RatFun::from_rational(value)),
                None => Ok(RatFun::from_rational(
                    raw.nonequivariant_limit_line(0, &[Rational::one()])?,
                )),
            }
        },
    )
}

pub fn compute_negative_split_twisted_resolvent_packed_factored(
    target_n: usize,
    degrees: Vec<usize>,
    req: &ResolventRequest,
) -> Result<ResolventResult<FactoredRatFun>, GwError> {
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(target_n, degrees)?;
    crate::givental::compute_packed_resolvent_with_coeff_provider(
        req,
        provider,
        "twisted-negative-split-fiber-equivariant-factored-packed-resolvent",
        "computed by packed fiber-equivariant twisted S/R external-leg graph kernel with factored symbolic coefficients; base weights are early-specialized and fiber weights are symbolic mu_i",
        Ok::<FactoredRatFun, GwError>,
    )
}

fn twisted_dimension_mismatch(
    provider: &TwistedProjectiveSpaceProvider,
    genus: usize,
    degree: usize,
    insertions: &[Insertion],
) -> Option<(isize, usize)> {
    let total_degree = provider.insertion_degree(insertions)?;
    let virtual_dimension = provider.virtual_dimension(genus, degree, insertions.len())?;
    (virtual_dimension >= 0 && total_degree as isize != virtual_dimension)
        .then_some((virtual_dimension, total_degree))
}

impl TwistedProjectiveSpaceProvider {
    pub fn new(n: usize, degrees: Vec<usize>, equivariant: bool) -> Result<Self, GwError> {
        let mut base = ProjectiveSpaceProvider::new(n, equivariant);
        base.weights = twisted_default_base_weights(n);
        Ok(Self {
            base,
            twist: NegativeSplitBundleTwist::new(degrees)?,
            line_mode: TwistedLineMode::EarlyRational,
            calibration_mode: TwistedCalibrationMode::InverseEuler,
            fiber_weight_scale: Rational::one(),
            custom_fiber_weights: None,
            fiber_parameter_names: Vec::new(),
        })
    }

    pub fn fiber_equivariant(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.line_mode = TwistedLineMode::FiberEquivariant;
        out.fiber_parameter_names = default_fiber_parameter_names(out.twist.rank());
        Ok(out)
    }

    pub fn symbolic_lambda_line(
        n: usize,
        degrees: Vec<usize>,
        equivariant: bool,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, equivariant)?;
        out.line_mode = TwistedLineMode::SymbolicLimit;
        Ok(out)
    }

    pub fn euler_twist(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.calibration_mode = TwistedCalibrationMode::Euler;
        Ok(out)
    }

    pub fn inverse_euler_with_positive_fiber_qrr(
        n: usize,
        degrees: Vec<usize>,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.calibration_mode = TwistedCalibrationMode::InverseEulerFiberPlus;
        Ok(out)
    }

    fn flat_metric_matrix(&self, q_degree: usize) -> Result<SeriesMatrix, GwError> {
        let fiber_weights = self.rational_fiber_weights();
        match self.line_mode {
            TwistedLineMode::EarlyRational => twisted_inverse_euler_flat_metric_matrix(
                self.base.n,
                q_degree,
                &self.twist,
                self.base_weights(),
                &fiber_weights,
            ),
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                twisted_inverse_euler_flat_metric_matrix_ratfun(
                    self.base.n,
                    q_degree,
                    &self.twist,
                    &base_weights,
                    &fiber_weights,
                )
            }
        }
    }

    fn genus_zero_two_point_fallback(
        &self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<Option<RatFun>, GwError> {
        let Some((descendant_idx, primary_idx, s_order)) =
            genus_zero_two_point_descendant_layout(insertions)
        else {
            return Ok(None);
        };
        let s_matrix = self.descendant_s_matrix(degree, s_order)?;
        let metric = self.flat_metric_matrix(degree)?;
        let descendant = self.insertion_vector(&insertions[descendant_idx], degree)?;
        let primary = self.insertion_vector(&insertions[primary_idx], degree)?;
        genus_zero_two_point_s_matrix_pairing_coeff(
            self.colors(),
            degree,
            s_order,
            &s_matrix,
            &metric,
            &descendant,
            &primary,
        )
        .map(Some)
    }

    pub fn rational_lambda_line_with_scale(
        n: usize,
        degrees: Vec<usize>,
        scale: Rational,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        for weight in &mut out.base.weights {
            *weight = weight.clone() * scale.clone();
        }
        out.fiber_weight_scale = scale;
        Ok(out)
    }

    pub fn rational_lambda_line_with_weights(
        n: usize,
        degrees: Vec<usize>,
        base_weights: Vec<Rational>,
        fiber_weights: Vec<Rational>,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        validate_twisted_weights(n, &out.twist, &base_weights, &fiber_weights)?;
        out.base.weights = base_weights;
        out.fiber_weight_scale = Rational::one();
        out.custom_fiber_weights = Some(fiber_weights);
        Ok(out)
    }

    pub fn n(&self) -> usize {
        self.base.n
    }

    pub fn twist(&self) -> &NegativeSplitBundleTwist {
        &self.twist
    }

    fn base_weights(&self) -> &[Rational] {
        &self.base.weights
    }

    fn rational_fiber_weights(&self) -> Vec<Rational> {
        if let Some(weights) = &self.custom_fiber_weights {
            return weights.clone();
        }
        let start = (1usize << (self.base.n + 1)).saturating_sub(1);
        (0..self.twist.rank())
            .map(|idx| Rational::from(start + 2 * idx) * self.fiber_weight_scale.clone())
            .collect()
    }

    fn ratfun_base_weights(&self) -> Vec<RatFun> {
        match self.line_mode {
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let lambda = crate::algebra::lambda(0);
                self.base_weights()
                    .iter()
                    .map(|weight| lambda.clone() * RatFun::from_rational(weight.clone()))
                    .collect()
            }
            TwistedLineMode::EarlyRational => self
                .base_weights()
                .iter()
                .cloned()
                .map(RatFun::from_rational)
                .collect(),
        }
    }

    fn ratfun_fiber_weights(&self) -> Vec<RatFun> {
        match self.line_mode {
            TwistedLineMode::SymbolicLimit => {
                let lambda = crate::algebra::lambda(0);
                self.rational_fiber_weights()
                    .into_iter()
                    .map(|weight| lambda.clone() * RatFun::from_rational(weight))
                    .collect()
            }
            TwistedLineMode::FiberEquivariant => self
                .fiber_parameter_names
                .iter()
                .cloned()
                .map(RatFun::variable)
                .collect(),
            TwistedLineMode::EarlyRational => self
                .rational_fiber_weights()
                .into_iter()
                .map(RatFun::from_rational)
                .collect(),
        }
    }
}

fn genus_zero_two_point_descendant_layout(
    insertions: &[Insertion],
) -> Option<(usize, usize, usize)> {
    if insertions.len() != 2 {
        return None;
    }
    let descendant_positions = insertions
        .iter()
        .enumerate()
        .filter_map(|(idx, insertion)| (insertion.descendant_power > 0).then_some(idx))
        .collect::<Vec<_>>();
    if descendant_positions.len() != 1 {
        return None;
    }

    let descendant_idx = descendant_positions[0];
    let primary_idx = 1 - descendant_idx;
    let s_order = insertions[descendant_idx].descendant_power + 1;
    Some((descendant_idx, primary_idx, s_order))
}

fn genus_zero_two_point_s_matrix_pairing_coeff<C: Coeff>(
    colors: usize,
    degree: usize,
    s_order: usize,
    s_matrix: &SeriesSMatrix<C>,
    metric: &SeriesMatrix<C>,
    descendant: &[QSeries<C>],
    primary: &[QSeries<C>],
) -> Result<C, GwError> {
    let s_coeff = s_matrix
        .coefficient(s_order)
        .ok_or(GwError::TruncationTooLow)?;

    let mut transformed = vec![QSeries::<C>::zero(degree); colors];
    for (row, target) in transformed.iter_mut().enumerate() {
        let mut total = QSeries::<C>::zero(degree);
        for (col, class_coeff) in descendant.iter().enumerate() {
            total = total.add(&s_coeff.entry(row, col).mul(class_coeff));
        }
        *target = total;
    }

    let mut paired = QSeries::<C>::zero(degree);
    for (left, transformed_coeff) in transformed.iter().enumerate() {
        for (right, primary_coeff) in primary.iter().enumerate() {
            let term = transformed_coeff
                .mul(metric.entry(left, right))
                .mul(primary_coeff);
            paired = paired.add(&term);
        }
    }
    Ok(paired.coeff(degree).cloned().unwrap_or_else(C::zero))
}

fn genus_zero_three_primary_layout(insertions: &[Insertion]) -> bool {
    insertions.len() == 3
        && insertions
            .iter()
            .all(|insertion| insertion.descendant_power == 0)
}

fn twisted_genus_zero_three_primary_value_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    calibration_mode: &TwistedCalibrationMode,
    base_weights: &[C],
    fiber_weights: &[C],
    insertions: &[Vec<QSeries<C>>],
) -> Result<C, GwError> {
    debug_assert_eq!(insertions.len(), 3);
    let model = NegativeSplitLineHypergeometricModel::<C>::from_coeff_weights(
        n,
        twist.clone(),
        degree,
        1,
        base_weights.to_vec(),
        fiber_weights,
    )?;
    let descendant_s = model.birkhoff_descendant_s_matrix(1)?;
    let classical_h = twisted_classical_h_multiplication_matrix_coeff(n, degree, base_weights)?;
    let quantum_h =
        twisted_quantum_multiplication_from_s_coeff(&descendant_s, &classical_h, calibration_mode)?;
    let metric = twisted_inverse_euler_flat_metric_matrix_coeff(
        n,
        degree,
        twist,
        base_weights,
        fiber_weights,
    )?;
    let product = quantum_product_vectors_coeff(&quantum_h, &insertions[0], &insertions[1], degree);
    pair_vectors_coeff(&metric, &product, &insertions[2], degree)
}

fn quantum_product_vectors_coeff<C: Coeff>(
    quantum_h: &SeriesMatrix<C>,
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let colors = quantum_h.rows();
    debug_assert_eq!(left.len(), colors);
    debug_assert_eq!(right.len(), colors);
    let mut result = vec![QSeries::<C>::zero(q_degree); colors];
    let mut h_power_right = right.to_vec();
    for (power, left_coeff) in left.iter().enumerate() {
        if !left_coeff.is_structurally_zero() {
            for row in 0..colors {
                result[row] = result[row].add(&h_power_right[row].mul(left_coeff));
            }
        }
        if power + 1 < left.len() {
            h_power_right =
                apply_series_matrix_to_vector_coeff(quantum_h, &h_power_right, q_degree);
        }
    }
    result
}

fn apply_series_matrix_to_vector_coeff<C: Coeff>(
    matrix: &SeriesMatrix<C>,
    vector: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    debug_assert_eq!(matrix.cols(), vector.len());
    (0..matrix.rows())
        .map(|row| {
            let mut total = QSeries::<C>::zero(q_degree);
            for (col, vector_coeff) in vector.iter().enumerate() {
                if vector_coeff.is_structurally_zero()
                    || matrix.entry(row, col).is_structurally_zero()
                {
                    continue;
                }
                total = total.add(&matrix.entry(row, col).mul(vector_coeff));
            }
            total
        })
        .collect()
}

fn pair_vectors_coeff<C: Coeff>(
    metric: &SeriesMatrix<C>,
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    degree: usize,
) -> Result<C, GwError> {
    let colors = metric.rows();
    debug_assert_eq!(metric.cols(), colors);
    debug_assert_eq!(left.len(), colors);
    debug_assert_eq!(right.len(), colors);
    let mut paired = QSeries::<C>::zero(degree);
    for (left_idx, left_coeff) in left.iter().enumerate() {
        if left_coeff.is_structurally_zero() {
            continue;
        }
        for (right_idx, right_coeff) in right.iter().enumerate() {
            if right_coeff.is_structurally_zero()
                || metric.entry(left_idx, right_idx).is_structurally_zero()
            {
                continue;
            }
            paired = paired.add(
                &left_coeff
                    .mul(metric.entry(left_idx, right_idx))
                    .mul(right_coeff),
            );
        }
    }
    Ok(paired.coeff(degree).cloned().unwrap_or_else(C::zero))
}

fn twisted_default_base_weights(n: usize) -> Vec<Rational> {
    (0..=n).map(|idx| Rational::from(1usize << idx)).collect()
}

fn default_fiber_parameter_names(rank: usize) -> Vec<String> {
    (0..rank).map(|idx| format!("mu_{idx}")).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoredTwistedProjectiveSpaceProvider {
    inner: TwistedProjectiveSpaceProvider,
}

impl FactoredTwistedProjectiveSpaceProvider {
    pub fn fiber_equivariant(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        Ok(Self {
            inner: TwistedProjectiveSpaceProvider::fiber_equivariant(n, degrees)?,
        })
    }

    pub fn inner(&self) -> &TwistedProjectiveSpaceProvider {
        &self.inner
    }

    fn factored_base_weights(&self) -> Vec<FactoredRatFun> {
        self.inner
            .base_weights()
            .iter()
            .cloned()
            .map(FactoredRatFun::from_rational)
            .collect()
    }

    fn factored_fiber_weights(&self) -> Vec<FactoredRatFun> {
        self.inner
            .fiber_parameter_names
            .iter()
            .cloned()
            .map(FactoredRatFun::variable)
            .collect()
    }

    fn factored_flat_metric_matrix(
        &self,
        q_degree: usize,
    ) -> Result<SeriesMatrix<FactoredRatFun>, GwError> {
        let (metric, _) = twisted_inverse_euler_flat_metric_pair_from_rational_base(
            self.inner.base.n,
            q_degree,
            &self.inner.twist,
            self.inner.base_weights(),
            &self.factored_fiber_weights(),
        )?;
        Ok(metric)
    }

    fn factored_genus_zero_two_point_fallback(
        &self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<Option<FactoredRatFun>, GwError> {
        let Some((descendant_idx, primary_idx, s_order)) =
            genus_zero_two_point_descendant_layout(insertions)
        else {
            return Ok(None);
        };
        let s_matrix = self.coeff_descendant_s_matrix(degree, s_order)?;
        let metric = self.factored_flat_metric_matrix(degree)?;
        let descendant = self.coeff_insertion_vector(&insertions[descendant_idx], degree)?;
        let primary = self.coeff_insertion_vector(&insertions[primary_idx], degree)?;
        genus_zero_two_point_s_matrix_pairing_coeff(
            self.coeff_colors(),
            degree,
            s_order,
            &s_matrix,
            &metric,
            &descendant,
            &primary,
        )
        .map(Some)
    }
}

impl CoefficientSemisimpleCohftProvider<FactoredRatFun> for FactoredTwistedProjectiveSpaceProvider {
    type Insertion = Insertion;

    fn coeff_colors(&self) -> usize {
        self.inner.colors()
    }

    fn coeff_descendant_power(&self, insertion: &Self::Insertion) -> usize {
        self.inner.descendant_power(insertion)
    }

    fn coeff_insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        self.inner.insertion_degree(insertions)
    }

    fn coeff_virtual_dimension(
        &self,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Option<isize> {
        self.inner.virtual_dimension(genus, degree, markings)
    }

    fn coeff_expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        self.inner.expected_degree_from_dimension(genus, insertions)
    }

    fn coeff_candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        self.inner
            .candidate_degrees_from_dimension(genus, degree_max, insertions)
    }

    fn coeff_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
        let base_weights = self.factored_base_weights();
        let fiber_weights = self.factored_fiber_weights();
        let s_matrix = NegativeSplitLineHypergeometricModel::<FactoredRatFun>::from_coeff_weights(
            self.inner.base.n,
            self.inner.twist.clone(),
            q_degree,
            z_order,
            base_weights.clone(),
            &fiber_weights,
        )?
        .birkhoff_descendant_s_matrix(z_order)?;
        let (flat_metric, flat_metric_inverse) =
            twisted_inverse_euler_flat_metric_pair_from_rational_base(
                self.inner.base.n,
                q_degree,
                &self.inner.twist,
                self.inner.base_weights(),
                &fiber_weights,
            )?;
        metric_adjoint_descendant_s_matrix_with_inverse_coeff(
            s_matrix,
            &flat_metric,
            &flat_metric_inverse,
        )
    }

    fn coeff_graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel<FactoredRatFun>>, GwError> {
        let profile_enabled = std::env::var_os("GW_PROFILE").is_some();
        let started = std::time::Instant::now();
        let base_weights = self.factored_base_weights();
        let fiber_weights = self.factored_fiber_weights();
        let calibration =
            negative_split_twisted_birkhoff_calibration_candidate_for_coeff_weights_with_validation(
                self.inner.base.n,
                &self.inner.twist,
                q_degree,
                r_order,
                &base_weights,
                &fiber_weights,
                twisted_calibration_validation_from_env(),
            )?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE factored_twisted_calibration={:.3}s",
                started.elapsed().as_secs_f64()
            );
        }
        let kernel_started = std::time::Instant::now();
        let kernel = Arc::new(GiventalGraphKernel::from_calibration(
            calibration,
            graph_dimension,
        )?);
        if profile_enabled {
            eprintln!(
                "GW_PROFILE factored_graph_kernel={:.3}s",
                kernel_started.elapsed().as_secs_f64()
            );
        }
        Ok(kernel)
    }

    fn coeff_insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries<FactoredRatFun>>, GwError> {
        Ok(self
            .inner
            .insertion_vector(insertion, q_degree)?
            .iter()
            .map(qseries_to_factored)
            .collect())
    }

    fn coeff_direct_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<FactoredRatFun>, GwError> {
        if genus != 0 || !genus_zero_three_primary_layout(insertions) {
            return Ok(None);
        }
        let base_weights = self.factored_base_weights();
        let fiber_weights = self.factored_fiber_weights();
        let insertion_vectors = insertions
            .iter()
            .map(|insertion| self.coeff_insertion_vector(insertion, degree))
            .collect::<Result<Vec<_>, _>>()?;
        twisted_genus_zero_three_primary_value_coeff(
            self.inner.base.n,
            &self.inner.twist,
            degree,
            &self.inner.calibration_mode,
            &base_weights,
            &fiber_weights,
            &insertion_vectors,
        )
        .map(Some)
    }

    fn coeff_scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<Option<FactoredRatFun>, GwError> {
        if genus == 0 {
            return self.factored_genus_zero_two_point_fallback(degree, insertions);
        }
        Ok(self
            .inner
            .scalar_fallback_value(genus, degree, insertions, truncation)?
            .map(FactoredRatFun::from_ratfun))
    }
}

#[cfg(test)]
fn series_s_matrix_to_factored(
    matrix: &SeriesSMatrix,
) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
    SeriesSMatrix::from_coefficients(
        matrix.size(),
        matrix.q_degree(),
        matrix.z_order(),
        matrix
            .coefficients()
            .iter()
            .map(series_matrix_to_factored)
            .collect(),
        matrix.calibration().clone(),
    )
}

#[cfg(test)]
fn series_matrix_to_factored(matrix: &SeriesMatrix) -> SeriesMatrix<FactoredRatFun> {
    SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| row.iter().map(qseries_to_factored).collect())
            .collect(),
    )
}

fn qseries_to_factored(series: &QSeries) -> QSeries<FactoredRatFun> {
    QSeries::from_coeffs(
        series
            .coeffs()
            .iter()
            .cloned()
            .map(FactoredRatFun::from_ratfun)
            .collect(),
    )
}

impl SemisimpleCohftProvider for TwistedProjectiveSpaceProvider {
    type Insertion = Insertion;

    fn colors(&self) -> usize {
        self.base.colors()
    }

    fn descendant_power(&self, insertion: &Self::Insertion) -> usize {
        self.base.descendant_power(insertion)
    }

    fn insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        self.base.insertion_degree(insertions)
    }

    fn virtual_dimension(&self, genus: usize, degree: usize, markings: usize) -> Option<isize> {
        Some(
            self.twist
                .virtual_dimension(self.base.n, genus, degree, markings),
        )
    }

    fn expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        if self.line_mode == TwistedLineMode::FiberEquivariant {
            return None;
        }
        let insertion_degree = self.insertion_degree(insertions)? as isize;
        let constant_dimension = (1 - genus as isize)
            * (self.twist.total_space_dimension(self.base.n) as isize - 3)
            + insertions.len() as isize;
        let slope = self.base.n as isize + 1 - self.twist.degree_sum() as isize;
        if slope == 0 {
            return None;
        }
        let numerator = insertion_degree - constant_dimension;
        if numerator < 0 || numerator % slope != 0 {
            return None;
        }
        Some((numerator / slope) as usize)
    }

    fn candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        if self.line_mode == TwistedLineMode::FiberEquivariant {
            return (0..=degree_max).collect();
        }
        self.twist.candidate_degrees(
            self.base.n,
            genus,
            degree_max,
            insertions.len(),
            self.insertion_degree(insertions),
        )
    }

    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        // The Birkhoff factor produces the fundamental-solution convention.
        // The graph evaluator expects the descendant insertion operator, which
        // is the adjoint with respect to the twisted flat metric.
        let rational_fiber_weights = self.rational_fiber_weights();
        let (s_matrix, flat_metric) = match self.line_mode {
            TwistedLineMode::EarlyRational => {
                let s_matrix =
                    NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
                        self.base.n,
                        self.twist.clone(),
                        q_degree,
                        z_order,
                        self.base_weights().to_vec(),
                        rational_fiber_weights.clone(),
                    )?
                    .birkhoff_descendant_s_matrix(z_order)?;
                let flat_metric = twisted_inverse_euler_flat_metric_matrix(
                    self.base.n,
                    q_degree,
                    &self.twist,
                    self.base_weights(),
                    &rational_fiber_weights,
                )?;
                (s_matrix, flat_metric)
            }
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                let s_matrix = NegativeSplitLineHypergeometricModel::from_ratfun_weights(
                    self.base.n,
                    self.twist.clone(),
                    q_degree,
                    z_order,
                    base_weights.clone(),
                    &fiber_weights,
                )?
                .birkhoff_descendant_s_matrix(z_order)?;
                let flat_metric = twisted_inverse_euler_flat_metric_matrix_ratfun(
                    self.base.n,
                    q_degree,
                    &self.twist,
                    &base_weights,
                    &fiber_weights,
                )?;
                (s_matrix, flat_metric)
            }
        };
        metric_adjoint_descendant_s_matrix(s_matrix, &flat_metric)
    }

    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        // This is where the twisted target becomes a generic semisimple CohFT:
        // Birkhoff canonical data + QRR R-recursion are packaged into the same
        // kernel interface used by ordinary P^n.
        static CACHE: OnceLock<
            Mutex<HashMap<TwistedGraphKernelCacheKey, Arc<GiventalGraphKernel>>>,
        > = OnceLock::new();
        let rational_fiber_weights = self.rational_fiber_weights();
        let validation = twisted_calibration_validation_from_env();
        let key = TwistedGraphKernelCacheKey {
            n: self.base.n,
            twist_degrees: self.twist.degrees().to_vec(),
            q_degree,
            r_order,
            graph_dimension,
            base_weights: self.base_weights().to_vec(),
            fiber_weights: rational_fiber_weights.clone(),
            fiber_parameter_names: self.fiber_parameter_names.clone(),
            line_mode: self.line_mode.clone(),
            calibration_mode: self.calibration_mode.clone(),
            validation,
        };
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(kernel) = cache.lock().unwrap().get(&key).cloned() {
            return Ok(kernel);
        }

        let calibration = match self.line_mode {
            TwistedLineMode::EarlyRational => {
                negative_split_twisted_birkhoff_calibration_candidate_with_mode_and_validation(
                    self.base.n,
                    &self.twist,
                    q_degree,
                    r_order,
                    self.base_weights(),
                    &rational_fiber_weights,
                    self.calibration_mode.clone(),
                    validation,
                )?
            }
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                negative_split_twisted_birkhoff_calibration_candidate_for_ratfun_weights_with_validation(
                    self.base.n,
                    &self.twist,
                    q_degree,
                    r_order,
                    &base_weights,
                    &fiber_weights,
                    validation,
                )?
            }
        };
        let kernel = Arc::new(GiventalGraphKernel::from_calibration(
            calibration,
            graph_dimension,
        )?);
        cache.lock().unwrap().insert(key, kernel.clone());
        Ok(kernel)
    }

    fn insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError> {
        self.base.insertion_vector(insertion, q_degree)
    }

    fn direct_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        if genus != 0 || !genus_zero_three_primary_layout(insertions) {
            return Ok(None);
        }
        let base_weights = self.ratfun_base_weights();
        let fiber_weights = self.ratfun_fiber_weights();
        let insertion_vectors = insertions
            .iter()
            .map(|insertion| self.insertion_vector(insertion, degree))
            .collect::<Result<Vec<_>, _>>()?;
        twisted_genus_zero_three_primary_value_coeff(
            self.base.n,
            &self.twist,
            degree,
            &self.calibration_mode,
            &base_weights,
            &fiber_weights,
            &insertion_vectors,
        )
        .map(Some)
    }

    fn scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        if genus == 0 {
            return self.genus_zero_two_point_fallback(degree, insertions);
        }
        Ok(None)
    }
}

fn metric_adjoint_descendant_s_matrix(
    s_matrix: SeriesSMatrix,
    flat_metric: &SeriesMatrix,
) -> Result<SeriesSMatrix, GwError> {
    metric_adjoint_descendant_s_matrix_coeff(s_matrix, flat_metric)
}

fn metric_adjoint_descendant_s_matrix_coeff<C: Coeff>(
    s_matrix: SeriesSMatrix<C>,
    flat_metric: &SeriesMatrix<C>,
) -> Result<SeriesSMatrix<C>, GwError> {
    // Converts S to its metric adjoint: S^* = eta^{-1} S^T eta.  This is the
    // correct action on covector insertions in the graph formula, and is
    // especially important after inverse-Euler twisting changes the pairing.
    let metric_inverse = invert_series_matrix_coeff(flat_metric)?;
    metric_adjoint_descendant_s_matrix_with_inverse_coeff(s_matrix, flat_metric, &metric_inverse)
}

fn metric_adjoint_descendant_s_matrix_with_inverse_coeff<C: Coeff>(
    s_matrix: SeriesSMatrix<C>,
    flat_metric: &SeriesMatrix<C>,
    metric_inverse: &SeriesMatrix<C>,
) -> Result<SeriesSMatrix<C>, GwError> {
    // Converts S to its metric adjoint: S^* = eta^{-1} S^T eta.  This overload
    // accepts eta^{-1} directly, which is important for factored twisted
    // theories where eta has a closed Vandermonde inverse and Gaussian
    // elimination would expand symbolic rational sums.
    let coefficients = s_matrix
        .coefficients()
        .iter()
        .map(|matrix| metric_inverse.mul(&matrix.transpose()).mul(flat_metric))
        .collect::<Vec<_>>();
    SeriesSMatrix::from_coefficients(
        s_matrix.size(),
        s_matrix.q_degree(),
        s_matrix.z_order(),
        coefficients,
        CalibrationId(format!("{}-metric-adjoint", s_matrix.calibration().0)),
    )
}

#[cfg(test)]
mod tests;
