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
pub use crate::constraints::virasoro::NegativeSplitCompletionEvaluator;
use crate::error::GwError;
use crate::factored::FactoredRatFun;
use crate::givental::{
    compute_semisimple_graph_value_with_coeff, CalibrationId, CanonicalFrameConvention,
    GiventalGraphKernel, ProjectiveSpaceProvider, SemisimpleCalibration, SemisimpleCohftProvider,
    SeriesRMatrix, SeriesSMatrix,
};
pub use crate::resolvent::{ResolventRequest, ResolventResult};
use crate::series::{
    compose_plain_series, integrate_q_derivative_zero_constant_matrix, invert_mirror_map,
    mul_plain_series, QSeries, SeriesMatrix,
};
pub use crate::spaces::projective_space::CohomologyClass;
use crate::theory::{CurveClass, CurveEffectivity, GwTheory, ProjectiveSpaceTheory};
pub use crate::theory::{NegativeSplitProjectiveCompletion, NegativeSplitTotalSpaceTheory};
pub use crate::{Insertion, InvariantResult, Truncation};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

// Compatibility flattening for the target-specific submodules, which
// historically imported these generic helpers through `super::*`.  New
// target-neutral callers should import them from `crate::reconstruction`.
pub(crate) use crate::reconstruction::*;
mod numeric;
pub(crate) use numeric::*;
mod hypergeometric;
pub use hypergeometric::*;
mod mirror;
pub use mirror::*;
mod calibration;
pub use calibration::*;
mod provider;
pub use provider::*;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct NegativeSplitBundleTwist {
    degrees: Vec<usize>,
}

impl NegativeSplitBundleTwist {
    /// Split bundle `O(-a_1) + ... + O(-a_r)` over `P^n`.
    ///
    /// The stored degrees are the positive integers `a_i`; the signs are part
    /// of the type convention.  Negativity is what gives the concave Euler
    /// factors in the hypergeometric `I`-function below.
    pub fn new(mut degrees: Vec<usize>) -> Result<Self, GwError> {
        if degrees.contains(&0) {
            return Err(GwError::ParseError(
                "negative split-bundle degrees must be positive".to_string(),
            ));
        }
        degrees.iter().try_fold(0usize, |sum, degree| {
            sum.checked_add(*degree).ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "negative split-bundle degree sum overflow".to_string(),
                )
            })
        })?;
        // A direct sum has no preferred summand order.  Canonicalizing here
        // gives every downstream model the same theory presentation.
        degrees.sort_unstable();
        Ok(Self { degrees })
    }

    /// Build the hypergeometric twist recipe from canonical target geometry.
    ///
    /// Providers use this path so summand order, rank, and degree data come
    /// from [`NegativeSplitTotalSpaceTheory`] rather than a second parsing and
    /// normalization pass.
    pub fn from_theory(theory: &NegativeSplitTotalSpaceTheory) -> Self {
        Self {
            degrees: theory.degrees().to_vec(),
        }
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

    fn with_canonical_theory<T>(
        &self,
        base_n: usize,
        use_theory: impl FnOnce(&dyn GwTheory) -> Result<T, GwError>,
    ) -> Result<T, GwError> {
        if self.degrees.is_empty() {
            let theory = ProjectiveSpaceTheory::try_new(base_n)?;
            use_theory(&theory)
        } else {
            let theory = NegativeSplitTotalSpaceTheory::new(base_n, self.degrees.clone())?;
            use_theory(&theory)
        }
    }

    pub fn try_total_space_dimension(&self, base_n: usize) -> Result<usize, GwError> {
        self.with_canonical_theory(base_n, |theory| Ok(theory.target_dimension()))
    }

    pub fn total_space_dimension(&self, base_n: usize) -> usize {
        self.try_total_space_dimension(base_n)
            .expect("negative-split total-space dimension must be representable")
    }

    pub fn try_is_calabi_yau(&self, base_n: usize) -> Result<bool, GwError> {
        self.with_canonical_theory(base_n, |theory| {
            Ok(theory.c1_pairing(&CurveClass::new(vec![1]))? == 0)
        })
    }

    pub fn is_calabi_yau(&self, base_n: usize) -> bool {
        self.try_is_calabi_yau(base_n)
            .expect("negative-split canonical theory must be representable")
    }

    pub fn try_virtual_dimension(
        &self,
        base_n: usize,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Result<isize, GwError> {
        let degree = i64::try_from(degree).map_err(|_| {
            GwError::UnsupportedInvariant(
                "local curve degree does not fit the canonical signed lattice".to_string(),
            )
        })?;
        self.with_canonical_theory(base_n, |theory| {
            theory.virtual_dimension(genus, &CurveClass::new(vec![degree]), markings)
        })
    }

    pub fn virtual_dimension(
        &self,
        base_n: usize,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> isize {
        self.try_virtual_dimension(base_n, genus, degree, markings)
            .expect("negative-split virtual dimension must be representable")
    }

    pub fn try_candidate_degrees(
        &self,
        base_n: usize,
        genus: usize,
        degree_max: usize,
        markings: usize,
        insertion_degree: Option<usize>,
    ) -> Result<Vec<usize>, GwError> {
        i64::try_from(degree_max).map_err(|_| {
            GwError::UnsupportedInvariant(
                "local degree bound does not fit the canonical signed lattice".to_string(),
            )
        })?;
        self.with_canonical_theory(base_n, |theory| {
            let effective_degrees = || -> Result<Vec<usize>, GwError> {
                let count = degree_max.checked_add(1).ok_or_else(|| {
                    GwError::UnsupportedInvariant("local degree bound is too large".to_string())
                })?;
                let mut degrees = Vec::new();
                degrees.try_reserve_exact(count).map_err(|_| {
                    GwError::UnsupportedInvariant(format!(
                        "cannot allocate {count} local degree candidates"
                    ))
                })?;
                for degree in 0..=degree_max {
                    let degree_i64 = i64::try_from(degree).map_err(|_| {
                        GwError::UnsupportedInvariant(
                            "local curve degree does not fit the canonical signed lattice"
                                .to_string(),
                        )
                    })?;
                    if theory.effectivity(&CurveClass::new(vec![degree_i64]))?
                        == CurveEffectivity::Effective
                    {
                        degrees.push(degree);
                    }
                }
                Ok(degrees)
            };
            let Some(insertion_degree) = insertion_degree else {
                return effective_degrees();
            };
            let insertion_degree = i128::try_from(insertion_degree).map_err(|_| {
                GwError::UnsupportedInvariant(
                    "local insertion degree does not fit in i128".to_string(),
                )
            })?;
            let constant_dimension =
                theory.virtual_dimension(genus, &CurveClass::new(vec![0]), markings)? as i128;
            let slope = i128::from(theory.c1_pairing(&CurveClass::new(vec![1]))?);
            let numerator = insertion_degree - constant_dimension;
            if slope == 0 {
                return if numerator == 0 {
                    effective_degrees()
                } else {
                    Ok(Vec::new())
                };
            }
            if numerator % slope != 0 {
                return Ok(Vec::new());
            }
            let Ok(degree) = usize::try_from(numerator / slope) else {
                return Ok(Vec::new());
            };
            if degree > degree_max {
                return Ok(Vec::new());
            }
            let degree_i64 = i64::try_from(degree).map_err(|_| {
                GwError::UnsupportedInvariant(
                    "local curve degree does not fit the canonical signed lattice".to_string(),
                )
            })?;
            if theory.effectivity(&CurveClass::new(vec![degree_i64]))?
                == CurveEffectivity::Effective
            {
                Ok(vec![degree])
            } else {
                Ok(Vec::new())
            }
        })
    }

    pub fn candidate_degrees(
        &self,
        base_n: usize,
        genus: usize,
        degree_max: usize,
        markings: usize,
        insertion_degree: Option<usize>,
    ) -> Vec<usize> {
        self.try_candidate_degrees(base_n, genus, degree_max, markings, insertion_degree)
            .expect("negative-split degree candidates must be representable")
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

    pub(crate) fn zero(max_h_power: usize) -> Self {
        Self {
            max_h_power,
            coeffs: vec![BTreeMap::new(); max_h_power + 1],
        }
    }

    pub(crate) fn one(max_h_power: usize) -> Self {
        let mut out = Self::zero(max_h_power);
        out.coeffs[0].insert(0, C::one());
        out
    }

    pub(crate) fn coefficient(&self, h_power: usize, z_power: i32) -> C {
        self.coeffs
            .get(h_power)
            .and_then(|terms| terms.get(&z_power))
            .cloned()
            .unwrap_or_else(C::zero)
    }

    pub(crate) fn max_h_power(&self) -> usize {
        self.max_h_power
    }

    pub(crate) fn terms_at_h_power(&self, h_power: usize) -> Option<&BTreeMap<i32, C>> {
        self.coeffs.get(h_power)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.coeffs.iter().all(BTreeMap::is_empty)
    }

    pub(crate) fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        let mut out = self.clone();
        for h_power in 0..=rhs.max_h_power {
            for (z_power, coeff) in &rhs.coeffs[h_power] {
                out.add_term(h_power, *z_power, coeff.clone());
            }
        }
        out
    }

    pub(crate) fn scale(&self, scalar: C) -> Self {
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

    pub(crate) fn shift_z(&self, shift: i32) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power, z_power + shift, coeff.clone());
            }
        }
        out
    }

    pub(crate) fn multiply_mod_relation(&self, rhs: &Self, h_power_relation: &[C]) -> Self {
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

    pub(crate) fn add_term(&mut self, h_power: usize, z_power: i32, coeff: C) {
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
        // The factors commute, so enumerate m = 0, -1, ..., -ad + 1.
        // Nested ranges preserve the infallible public API without forming the
        // potentially overflowing product a*d or casting it through isize.
        let mut z_offset = Rational::zero();
        for _ in 0..degree {
            for _ in 0..*bundle_degree {
                out = out.multiply_by_linear(-Rational::from(*bundle_degree), -z_offset.clone());
                z_offset += Rational::one();
            }
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
        let mut z_offset = Rational::zero();
        for _ in 0..degree {
            for _ in 0..*bundle_degree {
                out = out.multiply_by_linear(-Rational::from(*bundle_degree), -z_offset.clone());
                z_offset += Rational::one();
            }
        }
    }
    out
}

fn negative_split_euler_factor_count(
    bundle_degree: usize,
    curve_degree: usize,
) -> Result<usize, GwError> {
    bundle_degree.checked_mul(curve_degree).ok_or_else(|| {
        GwError::UnsupportedInvariant("negative-split Euler-factor count overflow".to_string())
    })
}

fn negative_split_positive_z_degree(
    twist: &NegativeSplitBundleTwist,
    degree: usize,
) -> Result<i32, GwError> {
    // For O(-a) in degree d the concave factor contains ad affine factors,
    // one with m=0.  Its largest z power is therefore max(ad-1, 0).  Terms
    // this far below the requested output floor can be shifted back into the
    // retained window when the projective denominator is multiplied by QRR.
    let positive_z_degree = twist.degrees().iter().try_fold(
        0usize,
        |total, bundle_degree| -> Result<usize, GwError> {
            let factor_count = negative_split_euler_factor_count(*bundle_degree, degree)?;
            let summand_z_degree = if factor_count == 0 {
                0
            } else {
                factor_count - 1
            };
            total.checked_add(summand_z_degree).ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "negative-split Euler-factor z-degree overflow".to_string(),
                )
            })
        },
    )?;
    i32::try_from(positive_z_degree).map_err(|_| {
        GwError::UnsupportedInvariant(
            "negative-split Euler-factor z-degree exceeds i32 range".to_string(),
        )
    })
}

fn negative_split_projective_source_min_z_power(
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    retained_min_z_power: i32,
) -> Result<i32, GwError> {
    let positive_z_degree = negative_split_positive_z_degree(twist, degree)?;
    retained_min_z_power
        .checked_sub(positive_z_degree)
        .ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "negative-split I-function source z-window exceeds i32 range".to_string(),
            )
        })
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
    let projective_min_z_power =
        negative_split_projective_source_min_z_power(twist, degree, min_z_power)?;
    let h_power_relation = base_h_power_relation_coeff(n, base_weights)?;
    let factor = negative_split_equivariant_qrr_euler_factor_coeff(
        n,
        twist,
        degree,
        base_weights,
        fiber_weights,
    )?;
    let projective = projective_equivariant_i_function_coefficient_coeff(
        n,
        degree,
        base_weights,
        projective_min_z_power,
    )?;
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
    let state_space_size = negative_split_state_space_size(n)?;
    if base_weights.len() != state_space_size {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            state_space_size,
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
    Ok(out.truncated_z_below(min_z_power))
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
    // `HCoeffLaurentSeries` stores z-exponents in i32.  Validate the
    // cumulative numerator degree before multiplying so the public fallible
    // path reports an error instead of eventually overflowing `z_power + 1`.
    negative_split_positive_z_degree(twist, degree)?;
    let h_power_relation = base_h_power_relation_coeff(n, base_weights)?;
    let mut out = HCoeffLaurentSeries::<C>::one(n);
    for (bundle_degree, fiber_weight) in twist.degrees().iter().zip(fiber_weights) {
        let factor_count = negative_split_euler_factor_count(*bundle_degree, degree)?;
        for z_offset in 0..factor_count {
            out = out.multiply_by_affine_mod_relation(
                C::from_rational(-Rational::from(*bundle_degree)),
                fiber_weight.clone(),
                C::from_rational(-Rational::from(z_offset)),
                &h_power_relation,
            );
        }
    }
    Ok(out)
}

pub(crate) fn base_h_power_relation_coeff<C: Coeff>(
    n: usize,
    base_weights: &[C],
) -> Result<Vec<C>, GwError> {
    let state_space_size = negative_split_state_space_size(n)?;
    if base_weights.len() != state_space_size {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            state_space_size,
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
    let leading = coefficients[state_space_size].clone();
    if leading.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    Ok((0..=n)
        .map(|power| coefficients[power].neg().div(&leading))
        .collect())
}

fn negative_split_state_space_size(n: usize) -> Result<usize, GwError> {
    n.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant(
            "negative-split projective state-space size overflow".to_string(),
        )
    })
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
    let max_k = min_z_power
        .checked_neg()
        .and_then(|depth| depth.checked_sub(1))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "Laurent expansion z-window exceeds the supported range".to_string(),
            )
        })?;
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

#[cfg(test)]
mod tests;
