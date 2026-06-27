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
use crate::series::{QSeries, SeriesMatrix};
use crate::{Insertion, InvariantResult, Truncation};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

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
        if degrees.iter().any(|degree| *degree == 0) {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HLaurentSeries {
    max_h_power: usize,
    coeffs: Vec<BTreeMap<i32, Rational>>,
}

impl HLaurentSeries {
    pub fn zero(max_h_power: usize) -> Self {
        Self {
            max_h_power,
            coeffs: vec![BTreeMap::new(); max_h_power + 1],
        }
    }

    pub fn one(max_h_power: usize) -> Self {
        let mut out = Self::zero(max_h_power);
        out.coeffs[0].insert(0, Rational::one());
        out
    }

    pub fn coefficient(&self, h_power: usize, z_power: i32) -> Rational {
        self.coeffs
            .get(h_power)
            .and_then(|terms| terms.get(&z_power))
            .cloned()
            .unwrap_or_else(Rational::zero)
    }

    pub fn max_h_power(&self) -> usize {
        self.max_h_power
    }

    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        let mut out = self.clone();
        for h_power in 0..=rhs.max_h_power {
            for (z_power, coeff) in &rhs.coeffs[h_power] {
                out.add_term(h_power, *z_power, coeff.clone());
            }
        }
        out
    }

    pub fn scale(&self, scalar: Rational) -> Self {
        if scalar.is_zero() {
            return Self::zero(self.max_h_power);
        }
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power, *z_power, coeff.clone() * scalar.clone());
            }
        }
        out
    }

    pub fn multiply_by_h(&self) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power + 1, *z_power, coeff.clone());
            }
        }
        out
    }

    pub fn shift_z(&self, shift: i32) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power, z_power + shift, coeff.clone());
            }
        }
        out
    }

    pub fn multiply_by_linear(&self, h_coeff: Rational, z_coeff: Rational) -> Self {
        self.multiply_by_affine(h_coeff, Rational::zero(), z_coeff)
    }

    pub fn multiply_by_affine(
        &self,
        h_coeff: Rational,
        constant: Rational,
        z_coeff: Rational,
    ) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                if !constant.is_zero() {
                    out.add_term(h_power, *z_power, coeff.clone() * constant.clone());
                }
                if !z_coeff.is_zero() {
                    out.add_term(h_power, z_power + 1, coeff.clone() * z_coeff.clone());
                }
                if !h_coeff.is_zero() && h_power < self.max_h_power {
                    out.add_term(h_power + 1, *z_power, coeff.clone() * h_coeff.clone());
                }
            }
        }
        out
    }

    pub fn truncated_z_below(&self, min_z_power: i32) -> Self {
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

    pub fn multiply(&self, rhs: &Self) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        let mut out = Self::zero(self.max_h_power);
        for left_h in 0..=self.max_h_power {
            for (left_z, left_coeff) in &self.coeffs[left_h] {
                for right_h in 0..=self.max_h_power - left_h {
                    for (right_z, right_coeff) in &rhs.coeffs[right_h] {
                        out.add_term(
                            left_h + right_h,
                            left_z + right_z,
                            left_coeff.clone() * right_coeff.clone(),
                        );
                    }
                }
            }
        }
        out
    }

    pub fn multiply_mod_relation(&self, rhs: &Self, h_power_relation: &[Rational]) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        assert_eq!(h_power_relation.len(), self.max_h_power + 1);
        let basis_powers = h_basis_powers_mod_relation(self.max_h_power, h_power_relation);
        let mut out = Self::zero(self.max_h_power);
        for left_h in 0..=self.max_h_power {
            for (left_z, left_coeff) in &self.coeffs[left_h] {
                for right_h in 0..=self.max_h_power {
                    for (right_z, right_coeff) in &rhs.coeffs[right_h] {
                        let scalar = left_coeff.clone() * right_coeff.clone();
                        if scalar.is_zero() {
                            continue;
                        }
                        for (reduced_h, reduced_coeff) in
                            basis_powers[left_h + right_h].iter().enumerate()
                        {
                            if reduced_coeff.is_zero() {
                                continue;
                            }
                            out.add_term(
                                reduced_h,
                                left_z + right_z,
                                scalar.clone() * reduced_coeff.clone(),
                            );
                        }
                    }
                }
            }
        }
        out
    }

    pub fn multiply_by_affine_mod_relation(
        &self,
        h_coeff: Rational,
        constant: Rational,
        z_coeff: Rational,
        h_power_relation: &[Rational],
    ) -> Self {
        assert_eq!(h_power_relation.len(), self.max_h_power + 1);
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                if !constant.is_zero() {
                    out.add_term(h_power, *z_power, coeff.clone() * constant.clone());
                }
                if !z_coeff.is_zero() {
                    out.add_term(h_power, z_power + 1, coeff.clone() * z_coeff.clone());
                }
                if !h_coeff.is_zero() {
                    if h_power < self.max_h_power {
                        out.add_term(h_power + 1, *z_power, coeff.clone() * h_coeff.clone());
                    } else {
                        for (reduced_h, reduced_coeff) in h_power_relation.iter().enumerate() {
                            if !reduced_coeff.is_zero() {
                                out.add_term(
                                    reduced_h,
                                    *z_power,
                                    coeff.clone() * h_coeff.clone() * reduced_coeff.clone(),
                                );
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn add_term(&mut self, h_power: usize, z_power: i32, coeff: Rational) {
        if coeff.is_zero() || h_power > self.max_h_power {
            return;
        }
        let terms = &mut self.coeffs[h_power];
        let next = terms.get(&z_power).cloned().unwrap_or_else(Rational::zero) + coeff;
        if next.is_zero() {
            terms.remove(&z_power);
        } else {
            terms.insert(z_power, next);
        }
    }
}

fn h_basis_powers_mod_relation(
    max_h_power: usize,
    h_power_relation: &[Rational],
) -> Vec<Vec<Rational>> {
    let mut powers = vec![vec![Rational::zero(); max_h_power + 1]; 2 * max_h_power + 1];
    powers[0][0] = Rational::one();
    for power in 1..=2 * max_h_power {
        for h_power in 0..max_h_power {
            powers[power][h_power + 1] =
                powers[power][h_power + 1].clone() + powers[power - 1][h_power].clone();
        }
        let top_coeff = powers[power - 1][max_h_power].clone();
        if !top_coeff.is_zero() {
            for (reduced_h, relation_coeff) in h_power_relation.iter().enumerate() {
                powers[power][reduced_h] =
                    powers[power][reduced_h].clone() + top_coeff.clone() * relation_coeff.clone();
            }
        }
    }
    powers
}

fn base_h_power_relation(n: usize, base_weights: &[Rational]) -> Result<Vec<Rational>, GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    let mut coefficients = vec![Rational::one()];
    for weight in base_weights {
        let mut next = vec![Rational::zero(); coefficients.len() + 1];
        for (power, coeff) in coefficients.iter().enumerate() {
            next[power] = next[power].clone() - coeff.clone() * weight.clone();
            next[power + 1] = next[power + 1].clone() + coeff.clone();
        }
        coefficients = next;
    }
    let leading = coefficients[n + 1].clone();
    if leading.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    Ok((0..=n)
        .map(|power| -coefficients[power].clone() / leading.clone())
        .collect())
}

fn h_affine_power_mod_relation(
    max_h_power: usize,
    h_coeff: Rational,
    constant: Rational,
    exponent: usize,
    h_power_relation: &[Rational],
) -> Vec<Rational> {
    let mut out = vec![Rational::zero(); max_h_power + 1];
    out[0] = Rational::one();
    for _ in 0..exponent {
        let mut next = vec![Rational::zero(); max_h_power + 1];
        for h_power in 0..=max_h_power {
            if out[h_power].is_zero() {
                continue;
            }
            if !constant.is_zero() {
                next[h_power] = next[h_power].clone() + out[h_power].clone() * constant.clone();
            }
            if !h_coeff.is_zero() {
                if h_power < max_h_power {
                    next[h_power + 1] =
                        next[h_power + 1].clone() + out[h_power].clone() * h_coeff.clone();
                } else {
                    for (reduced_h, relation_coeff) in h_power_relation.iter().enumerate() {
                        next[reduced_h] = next[reduced_h].clone()
                            + out[h_power].clone() * h_coeff.clone() * relation_coeff.clone();
                    }
                }
            }
        }
        out = next;
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HCoeffLaurentSeries<C = RatFun> {
    max_h_power: usize,
    coeffs: Vec<BTreeMap<i32, C>>,
}

impl<C: Coeff> HCoeffLaurentSeries<C> {
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
    // Equivariant projective I_d:
    // product_{m=1}^d product_i (H-lambda_i+mz)^{-1},
    // reduced in H_T(P^n) along the chosen lambda specialization.
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    let h_power_relation = base_h_power_relation(n, base_weights)?;
    let mut out = HLaurentSeries::one(n);
    for m in 1..=degree {
        for weight in base_weights {
            let inverse = inverse_affine_z_laurent(
                n,
                Rational::one(),
                -weight.clone(),
                Rational::from(m),
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
    // Equivariant inverse-Euler/QRR factor for the negative fibers:
    // product_{bundle a} product_{m=-ad+1}^0 (-aH + fiber_lambda + mz).
    if fiber_weights.len() != twist.rank() {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} fiber weights, got {}",
            twist.rank(),
            fiber_weights.len()
        )));
    }
    let h_power_relation = base_h_power_relation(n, base_weights)?;
    let mut out = HLaurentSeries::one(n);
    for (bundle_degree, fiber_weight) in twist.degrees().iter().zip(fiber_weights) {
        for m in (-(bundle_degree.saturating_mul(degree) as isize) + 1)..=0 {
            out = out.multiply_by_affine_mod_relation(
                -Rational::from(*bundle_degree),
                fiber_weight.clone(),
                Rational::from(m),
                &h_power_relation,
            );
        }
    }
    Ok(out)
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
    // Twisted I_d = projective I_d times the concave fiber Euler factor.
    let h_power_relation = base_h_power_relation(n, base_weights)?;
    let factor =
        negative_split_equivariant_qrr_euler_factor(n, twist, degree, base_weights, fiber_weights)?;
    let projective =
        projective_equivariant_i_function_coefficient(n, degree, base_weights, min_z_power)?;
    Ok(factor
        .multiply_mod_relation(&projective, &h_power_relation)
        .truncated_z_below(min_z_power))
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
        mirror_transformed_j_coefficients_from_i_function(
            self.n,
            &self.i_coefficients(),
            &self.mirror_map_coefficients(),
            &self.inverse_mirror_map_coefficients(),
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
        mirror_transformed_j_coefficients_from_i_function(
            self.n,
            &self.i_coefficients(),
            &self.mirror_map_coefficients(),
            &self.inverse_mirror_map_coefficients(),
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

impl NegativeSplitEquivariantHypergeometricModel {
    pub fn new(
        n: usize,
        twist: NegativeSplitBundleTwist,
        q_degree: usize,
        base_weights: Vec<Rational>,
        fiber_weights: Vec<Rational>,
        min_z_power: i32,
    ) -> Result<Self, GwError> {
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
        let min_z_power = -(((n + 1) * q_degree + z_order + 2) as i32);
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
        Ok(
            mirror_transformed_j_coefficients_from_i_function_mod_relation(
                self.n,
                &self.i_coefficients()?,
                &self.mirror_map_coefficients()?,
                &self.inverse_mirror_map_coefficients()?,
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
        birkhoff_descendant_s_matrix_from_fundamental(
            self.n + 1,
            self.q_degree,
            z_order,
            &self.fundamental_solution_matrix()?,
            CalibrationId("negative-split-equivariant-hypergeometric-birkhoff".to_string()),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NegativeSplitLineHypergeometricModel<C = RatFun> {
    n: usize,
    twist: NegativeSplitBundleTwist,
    q_degree: usize,
    base_weights: Vec<C>,
    fiber_weights: Vec<C>,
    min_z_power: i32,
}

impl NegativeSplitLineHypergeometricModel<RatFun> {
    fn from_ratfun_weights(
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
    fn from_coeff_weights(
        n: usize,
        twist: NegativeSplitBundleTwist,
        q_degree: usize,
        z_order: usize,
        base_weights: Vec<C>,
        fiber_weights: &[C],
    ) -> Result<Self, GwError> {
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
        let min_z_power = -(((n + 1) * q_degree + z_order + 2) as i32);
        Ok(Self {
            n,
            twist,
            q_degree,
            base_weights,
            fiber_weights: fiber_weights.to_vec(),
            min_z_power,
        })
    }

    fn i_coefficients(&self) -> Result<Vec<HCoeffLaurentSeries<C>>, GwError> {
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

    fn mirror_map_coefficients(&self) -> Result<Vec<C>, GwError> {
        Ok(mirror_map_coefficients_from_i_function_coeff(
            &self.i_coefficients()?,
            self.q_degree,
        ))
    }

    fn inverse_mirror_map_coefficients(&self) -> Result<Vec<C>, GwError> {
        Ok(invert_mirror_map_coeff(
            &self.mirror_map_coefficients()?,
            self.q_degree,
        ))
    }

    fn mirror_transformed_j_coefficients(&self) -> Result<Vec<HCoeffLaurentSeries<C>>, GwError> {
        let h_power_relation = base_h_power_relation_coeff(self.n, &self.base_weights)?;
        Ok(
            mirror_transformed_j_coefficients_from_i_function_mod_relation_coeff(
                self.n,
                &self.i_coefficients()?,
                &self.mirror_map_coefficients()?,
                &self.inverse_mirror_map_coefficients()?,
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

    fn birkhoff_descendant_s_matrix(&self, z_order: usize) -> Result<SeriesSMatrix<C>, GwError> {
        birkhoff_descendant_s_matrix_from_fundamental_coeff(
            self.n + 1,
            self.q_degree,
            z_order,
            &self.fundamental_solution_matrix()?,
            CalibrationId("negative-split-lambda-line-hypergeometric-birkhoff".to_string()),
        )
    }
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
    // In the one-parameter local models here, the mirror coordinate is the
    // coefficient of H z^{-1} in I/I_0.  The implemented examples have I_0=1
    // after the chosen normalization.
    let mut out = vec![Rational::zero(); q_degree + 1];
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
            .unwrap_or_else(Rational::zero);
    }
    out
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
    let gauge =
        full_vector_mirror_gauge_coefficients(n, i_coefficients, q_degree, h_power_relation);
    let gauged = multiply_h_laurent_q_series_mod_relation(
        &gauge,
        i_coefficients,
        q_degree,
        h_power_relation,
    );
    compose_h_laurent_q_series(&gauged, _inverse_mirror, q_degree)
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

fn invert_mirror_map(mirror: &[Rational], q_degree: usize) -> Vec<Rational> {
    let exp_mirror = exp_series(mirror, q_degree);
    let mut q_of_q = vec![Rational::zero(); q_degree + 1];
    if q_degree >= 1 {
        q_of_q[1] = Rational::one();
    }
    let target = mul_plain_series(&q_of_q, &exp_mirror, q_degree);
    invert_series_with_linear_term_one(&target, q_degree)
}

fn invert_mirror_map_coeff<C: Coeff>(mirror: &[C], q_degree: usize) -> Vec<C> {
    let exp_mirror = exp_series_coeff(mirror, q_degree);
    let mut q_of_q = vec![C::zero(); q_degree + 1];
    if q_degree >= 1 {
        q_of_q[1] = C::one();
    }
    let target = mul_plain_series_coeff(&q_of_q, &exp_mirror, q_degree);
    invert_series_with_linear_term_one_coeff(&target, q_degree)
}

fn exp_series(series: &[Rational], max_degree: usize) -> Vec<Rational> {
    let mut out = vec![Rational::zero(); max_degree + 1];
    out[0] = Rational::one();
    for degree in 1..=max_degree {
        let mut sum = Rational::zero();
        for split in 1..=degree {
            let coeff = series.get(split).cloned().unwrap_or_else(Rational::zero);
            sum += Rational::from(split) * coeff * out[degree - split].clone();
        }
        out[degree] = sum / Rational::from(degree);
    }
    out
}

fn exp_series_coeff<C: Coeff>(series: &[C], max_degree: usize) -> Vec<C> {
    let mut out = vec![C::zero(); max_degree + 1];
    out[0] = C::one();
    for degree in 1..=max_degree {
        let mut sum = C::zero();
        for split in 1..=degree {
            let coeff = series.get(split).cloned().unwrap_or_else(C::zero);
            let term = C::from_usize(split).mul(&coeff).mul(&out[degree - split]);
            sum = sum.add(&term);
        }
        out[degree] = sum.div(&C::from_usize(degree));
    }
    out
}

fn invert_series_with_linear_term_one(series: &[Rational], max_degree: usize) -> Vec<Rational> {
    assert_eq!(series.first(), Some(&Rational::zero()));
    assert_eq!(series.get(1), Some(&Rational::one()));
    let mut inverse = vec![Rational::zero(); max_degree + 1];
    if max_degree >= 1 {
        inverse[1] = Rational::one();
    }
    for degree in 2..=max_degree {
        let mut trial = inverse.clone();
        trial[degree] = Rational::one();
        let contribution = compose_plain_series(series, &trial, max_degree)[degree].clone();
        let mut baseline = inverse.clone();
        baseline[degree] = Rational::zero();
        let current = compose_plain_series(series, &baseline, max_degree)[degree].clone();
        let sensitivity = contribution - current.clone();
        inverse[degree] = -current / sensitivity;
    }
    inverse
}

fn invert_series_with_linear_term_one_coeff<C: Coeff>(series: &[C], max_degree: usize) -> Vec<C> {
    assert_eq!(series.first(), Some(&C::zero()));
    assert_eq!(series.get(1), Some(&C::one()));
    let mut inverse = vec![C::zero(); max_degree + 1];
    if max_degree >= 1 {
        inverse[1] = C::one();
    }
    for degree in 2..=max_degree {
        let mut trial = inverse.clone();
        trial[degree] = C::one();
        let contribution = compose_plain_series_coeff(series, &trial, max_degree)[degree].clone();
        let mut baseline = inverse.clone();
        baseline[degree] = C::zero();
        let current = compose_plain_series_coeff(series, &baseline, max_degree)[degree].clone();
        let sensitivity = contribution.sub(&current);
        inverse[degree] = current.neg().div(&sensitivity);
    }
    inverse
}

fn compose_plain_series(
    series: &[Rational],
    input: &[Rational],
    max_degree: usize,
) -> Vec<Rational> {
    let mut out = vec![Rational::zero(); max_degree + 1];
    let mut power = vec![Rational::zero(); max_degree + 1];
    power[0] = Rational::one();
    for degree in 0..=max_degree {
        let coefficient = series.get(degree).cloned().unwrap_or_else(Rational::zero);
        if !coefficient.is_zero() {
            for idx in 0..=max_degree {
                out[idx] += coefficient.clone() * power[idx].clone();
            }
        }
        power = mul_plain_series(&power, input, max_degree);
    }
    out
}

fn compose_plain_series_coeff<C: Coeff>(series: &[C], input: &[C], max_degree: usize) -> Vec<C> {
    let mut out = vec![C::zero(); max_degree + 1];
    let mut power = vec![C::zero(); max_degree + 1];
    power[0] = C::one();
    for degree in 0..=max_degree {
        let coefficient = series.get(degree).cloned().unwrap_or_else(C::zero);
        if !coefficient.is_zero() {
            for idx in 0..=max_degree {
                out[idx] = out[idx].add(&coefficient.mul(&power[idx]));
            }
        }
        power = mul_plain_series_coeff(&power, input, max_degree);
    }
    out
}

fn mul_plain_series(left: &[Rational], right: &[Rational], max_degree: usize) -> Vec<Rational> {
    let mut out = vec![Rational::zero(); max_degree + 1];
    for left_degree in 0..=max_degree {
        if left[left_degree].is_zero() {
            continue;
        }
        for right_degree in 0..=max_degree - left_degree {
            if right[right_degree].is_zero() {
                continue;
            }
            out[left_degree + right_degree] +=
                left[left_degree].clone() * right[right_degree].clone();
        }
    }
    out
}

fn mul_plain_series_coeff<C: Coeff>(left: &[C], right: &[C], max_degree: usize) -> Vec<C> {
    let mut out = vec![C::zero(); max_degree + 1];
    for left_degree in 0..=max_degree {
        if left[left_degree].is_zero() {
            continue;
        }
        for right_degree in 0..=max_degree - left_degree {
            if right[right_degree].is_zero() {
                continue;
            }
            out[left_degree + right_degree] =
                out[left_degree + right_degree].add(&left[left_degree].mul(&right[right_degree]));
        }
    }
    out
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

fn multiply_h_laurent_q_series_mod_relation(
    left: &[HLaurentSeries],
    right: &[HLaurentSeries],
    max_degree: usize,
    h_power_relation: &[Rational],
) -> Vec<HLaurentSeries> {
    let max_h_power = left
        .first()
        .or_else(|| right.first())
        .map(HLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HLaurentSeries::zero(max_h_power); max_degree + 1];
    for left_degree in 0..=max_degree {
        for right_degree in 0..=max_degree - left_degree {
            let term =
                left[left_degree].multiply_mod_relation(&right[right_degree], h_power_relation);
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

fn full_vector_mirror_gauge_coefficients(
    n: usize,
    i_coefficients: &[HLaurentSeries],
    max_degree: usize,
    h_power_relation: &[Rational],
) -> Vec<HLaurentSeries> {
    let mut exponent = vec![HLaurentSeries::zero(n); max_degree + 1];
    let mut gauge = vec![HLaurentSeries::zero(n); max_degree + 1];
    gauge[0] = HLaurentSeries::one(n);

    for degree in 1..=max_degree {
        let mut known_gauge = HLaurentSeries::zero(n);
        for split in 1..degree {
            if exponent[split].coeffs.iter().all(BTreeMap::is_empty) {
                continue;
            }
            let term = exponent[split]
                .multiply_mod_relation(&gauge[degree - split], h_power_relation)
                .scale(Rational::from(split));
            known_gauge = known_gauge.add(&term);
        }
        known_gauge = known_gauge.scale(Rational::one() / Rational::from(degree));
        gauge[degree] = known_gauge;

        let mut gauged_degree = HLaurentSeries::zero(n);
        for split in 0..=degree {
            let term = gauge[split]
                .multiply_mod_relation(&i_coefficients[degree - split], h_power_relation);
            gauged_degree = gauged_degree.add(&term);
        }
        let tau = z_power_part(&gauged_degree, -1);
        exponent[degree] = tau.shift_z(-1).scale(-Rational::one());
        gauge[degree] = gauge[degree].add(&exponent[degree]);
    }

    gauge
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

fn z_power_part(series: &HLaurentSeries, z_power: i32) -> HLaurentSeries {
    let mut out = HLaurentSeries::zero(series.max_h_power());
    for h_power in 0..=series.max_h_power() {
        let coeff = series.coefficient(h_power, z_power);
        if !coeff.is_zero() {
            out.add_term(h_power, 0, coeff);
        }
    }
    out
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
    let max_h_power = series.first().map(HLaurentSeries::max_h_power).unwrap_or(0);
    let mut out = vec![HLaurentSeries::zero(max_h_power); max_degree + 1];
    let mut power = vec![Rational::zero(); max_degree + 1];
    power[0] = Rational::one();
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
        power = mul_plain_series_coeff(&power, input, max_degree);
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
    let max_degree = series.len().saturating_sub(1);
    let max_h_power = series.first().map(HLaurentSeries::max_h_power).unwrap_or(0);
    let mut out = vec![HLaurentSeries::zero(max_h_power); max_degree + 1];
    for degree in 0..=max_degree {
        out[degree] = out[degree].add(&series[degree].multiply_by_affine_mod_relation(
            Rational::one(),
            Rational::zero(),
            Rational::zero(),
            h_power_relation,
        ));
        if degree > 0 {
            let derivative_term = series[degree].shift_z(1).scale(Rational::from(degree));
            out[degree] = out[degree].add(&derivative_term);
        }
    }
    out
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

type CoeffMatrix<C> = Vec<Vec<C>>;
type LaurentCoeffMatrix<C> = BTreeMap<i32, CoeffMatrix<C>>;
type QDegreeLaurentFactor<C> = Vec<LaurentCoeffMatrix<C>>;

fn birkhoff_factor_by_q_degree<C: Coeff>(
    size: usize,
    q_degree: usize,
    matrix: &BTreeMap<i32, SeriesMatrix<C>>,
) -> Result<(QDegreeLaurentFactor<C>, QDegreeLaurentFactor<C>), GwError> {
    // Recursive Birkhoff split in Novikov degree.  At each q^d, all lower-degree
    // products are known; the remaining Laurent matrix is uniquely split into
    // nonnegative and negative z-powers.
    validate_identity_at_q_zero(size, matrix)?;
    let mut positive = vec![BTreeMap::new(); q_degree + 1];
    let mut negative = vec![BTreeMap::new(); q_degree + 1];
    positive[0].insert(0, identity_coeff_matrix(size));
    negative[0].insert(0, identity_coeff_matrix(size));

    for degree in 1..=q_degree {
        let mut raw = q_degree_slice(matrix, degree, size);
        let known = multiply_laurent_matrix_q_slices(&negative, &positive, degree, size);
        subtract_laurent_matrix(&mut raw, &known);
        for (z_power, coeff) in raw {
            if coeff_matrix_is_zero(&coeff) {
                continue;
            }
            if z_power >= 0 {
                positive[degree].insert(z_power, coeff);
            } else {
                negative[degree].insert(z_power, coeff);
            }
        }
    }

    Ok((positive, negative))
}

fn validate_identity_at_q_zero<C: Coeff>(
    size: usize,
    matrix: &BTreeMap<i32, SeriesMatrix<C>>,
) -> Result<(), GwError> {
    for (z_power, coefficient) in matrix {
        let q0 = matrix_q_coefficient(coefficient, 0);
        let expected = if *z_power == 0 {
            identity_coeff_matrix(size)
        } else {
            zero_coeff_matrix(size)
        };
        if q0 != expected {
            return Err(GwError::ConventionMismatch(format!(
                "Birkhoff input must be identity at q=0; z^{z_power} coefficient is nonstandard"
            )));
        }
    }
    Ok(())
}

fn q_degree_slice<C: Coeff>(
    matrix: &BTreeMap<i32, SeriesMatrix<C>>,
    degree: usize,
    size: usize,
) -> LaurentCoeffMatrix<C> {
    let mut out = BTreeMap::new();
    for (z_power, coefficient) in matrix {
        let q_coeff = matrix_q_coefficient(coefficient, degree);
        if !coeff_matrix_is_zero(&q_coeff) {
            out.insert(*z_power, q_coeff);
        }
    }
    if out.is_empty() {
        out.insert(0, zero_coeff_matrix(size));
    }
    out
}

fn matrix_q_coefficient<C: Coeff>(matrix: &SeriesMatrix<C>, degree: usize) -> CoeffMatrix<C> {
    matrix
        .entries()
        .iter()
        .map(|row| {
            row.iter()
                .map(|entry| entry.coeff(degree).cloned().unwrap_or_else(C::zero))
                .collect()
        })
        .collect()
}

fn multiply_laurent_matrix_q_slices<C: Coeff>(
    left: &[LaurentCoeffMatrix<C>],
    right: &[LaurentCoeffMatrix<C>],
    degree: usize,
    size: usize,
) -> LaurentCoeffMatrix<C> {
    let mut out = BTreeMap::new();
    for split in 1..degree {
        for (left_z, left_matrix) in &left[split] {
            for (right_z, right_matrix) in &right[degree - split] {
                let product = multiply_coeff_matrix(left_matrix, right_matrix, size);
                add_matrix_to_laurent(&mut out, left_z + right_z, product);
            }
        }
    }
    out
}

fn subtract_laurent_matrix<C: Coeff>(
    target: &mut LaurentCoeffMatrix<C>,
    rhs: &LaurentCoeffMatrix<C>,
) {
    for (z_power, matrix) in rhs {
        add_matrix_to_laurent(target, *z_power, neg_coeff_matrix(matrix));
    }
}

fn add_matrix_to_laurent<C: Coeff>(
    target: &mut LaurentCoeffMatrix<C>,
    z_power: i32,
    matrix: CoeffMatrix<C>,
) {
    if coeff_matrix_is_zero(&matrix) {
        return;
    }
    let size = matrix.len();
    let entry = target
        .entry(z_power)
        .or_insert_with(|| zero_coeff_matrix(size));
    for row in 0..size {
        for col in 0..size {
            entry[row][col] = entry[row][col].add(&matrix[row][col]);
        }
    }
    if coeff_matrix_is_zero(entry) {
        target.remove(&z_power);
    }
}

fn negative_factor_to_s_coefficients<C: Coeff>(
    size: usize,
    q_degree: usize,
    z_order: usize,
    negative: &[LaurentCoeffMatrix<C>],
) -> Vec<SeriesMatrix<C>> {
    let mut coefficients = Vec::with_capacity(z_order + 1);
    for order in 0..=z_order {
        let mut entries = vec![vec![vec![C::zero(); q_degree + 1]; size]; size];
        if order == 0 {
            for idx in 0..size {
                entries[idx][idx][0] = C::one();
            }
        } else {
            let z_power = -(order as i32);
            for degree in 1..=q_degree {
                if let Some(matrix) = negative[degree].get(&z_power) {
                    for row in 0..size {
                        for col in 0..size {
                            entries[row][col][degree] = matrix[row][col].clone();
                        }
                    }
                }
            }
        }
        coefficients.push(SeriesMatrix::from_entries(
            entries
                .into_iter()
                .map(|row| row.into_iter().map(QSeries::from_coeffs).collect())
                .collect(),
        ));
    }
    coefficients
}

fn multiply_coeff_matrix<C: Coeff>(
    left: &CoeffMatrix<C>,
    right: &CoeffMatrix<C>,
    size: usize,
) -> CoeffMatrix<C> {
    let mut out = zero_coeff_matrix(size);
    for row in 0..size {
        for col in 0..size {
            let mut total = C::zero();
            for mid in 0..size {
                total = total.add(&left[row][mid].mul(&right[mid][col]));
            }
            out[row][col] = total;
        }
    }
    out
}

fn identity_coeff_matrix<C: Coeff>(size: usize) -> CoeffMatrix<C> {
    let mut out = zero_coeff_matrix(size);
    for idx in 0..size {
        out[idx][idx] = C::one();
    }
    out
}

fn zero_coeff_matrix<C: Coeff>(size: usize) -> CoeffMatrix<C> {
    vec![vec![C::zero(); size]; size]
}

fn neg_coeff_matrix<C: Coeff>(matrix: &CoeffMatrix<C>) -> CoeffMatrix<C> {
    matrix
        .iter()
        .map(|row| row.iter().map(Coeff::neg).collect())
        .collect()
}

fn coeff_matrix_is_zero<C: Coeff>(matrix: &CoeffMatrix<C>) -> bool {
    matrix.iter().all(|row| row.iter().all(Coeff::is_zero))
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

fn inverse_affine_z_laurent(
    max_h_power: usize,
    h_coeff: Rational,
    constant: Rational,
    z_coeff: Rational,
    min_z_power: i32,
    h_power_relation: Option<&[Rational]>,
) -> Result<HLaurentSeries, GwError> {
    if z_coeff.is_zero() {
        return Err(GwError::AlgebraFailure(
            "cannot expand affine inverse at z=infinity with zero z coefficient".to_string(),
        ));
    }
    if min_z_power >= 0 {
        return Ok(HLaurentSeries::zero(max_h_power));
    }

    let mut out = HLaurentSeries::zero(max_h_power);
    let max_k = (-min_z_power - 1) as usize;
    for k in 0..=max_k {
        let sign = if k % 2 == 0 {
            Rational::one()
        } else {
            -Rational::one()
        };
        let denominator = z_coeff.pow_usize(k + 1);
        if let Some(relation) = h_power_relation {
            let affine_power = h_affine_power_mod_relation(
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
                    sign.clone() * coeff / denominator.clone(),
                );
            }
        } else {
            for h_power in 0..=max_h_power.min(k) {
                let binom = binomial_rational(k, h_power);
                let coeff = sign.clone()
                    * binom
                    * constant.clone().pow_usize(k - h_power)
                    * h_coeff.clone().pow_usize(h_power)
                    / denominator.clone();
                out.add_term(h_power, -((k + 1) as i32), coeff);
            }
        }
    }
    Ok(out)
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
    let sign = if h_power % 2 == 0 {
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
    twist
        .degrees()
        .iter()
        .zip(fiber_weights)
        .fold(Rational::one(), |acc, (degree, weight)| {
            acc * (weight.clone() - Rational::from(*degree) * lambda.clone())
        })
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

fn determinant_qseries_polynomial_matrix<C: Coeff>(
    matrix: &[Vec<Vec<QSeries<C>>>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let size = matrix.len();
    if size == 0 {
        return vec![QSeries::<C>::one(q_degree)];
    }
    if size == 1 {
        return matrix[0][0].clone();
    }

    let mut total = vec![QSeries::<C>::zero(q_degree)];
    for col in 0..size {
        let mut minor = Vec::with_capacity(size - 1);
        for source_row in matrix.iter().skip(1) {
            let mut row = Vec::with_capacity(size - 1);
            for (source_col, entry) in source_row.iter().enumerate() {
                if source_col != col {
                    row.push(entry.clone());
                }
            }
            minor.push(row);
        }
        let term = qseries_polynomial_mul(
            &matrix[0][col],
            &determinant_qseries_polynomial_matrix(&minor, q_degree),
            q_degree,
        );
        total = if col % 2 == 0 {
            qseries_polynomial_add(&total, &term, q_degree)
        } else {
            qseries_polynomial_sub(&total, &term, q_degree)
        };
    }
    total
}

fn qseries_polynomial_add<C: Coeff>(
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let size = left.len().max(right.len());
    let mut out = vec![QSeries::<C>::zero(q_degree); size];
    for degree in 0..size {
        let left_coeff = left
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::<C>::zero(q_degree));
        let right_coeff = right
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::<C>::zero(q_degree));
        out[degree] = left_coeff.add(&right_coeff);
    }
    out
}

fn qseries_polynomial_sub<C: Coeff>(
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let size = left.len().max(right.len());
    let mut out = vec![QSeries::<C>::zero(q_degree); size];
    for degree in 0..size {
        let left_coeff = left
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::<C>::zero(q_degree));
        let right_coeff = right
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::<C>::zero(q_degree));
        out[degree] = left_coeff.sub(&right_coeff);
    }
    out
}

fn qseries_polynomial_mul<C: Coeff>(
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let mut out = vec![QSeries::<C>::zero(q_degree); left.len() + right.len() - 1];
    for (left_degree, left_coeff) in left.iter().enumerate() {
        if left_coeff.is_zero() {
            continue;
        }
        for (right_degree, right_coeff) in right.iter().enumerate() {
            if right_coeff.is_zero() {
                continue;
            }
            out[left_degree + right_degree] =
                out[left_degree + right_degree].add(&left_coeff.mul(right_coeff));
        }
    }
    out
}

fn series_matrix_scale<C: Coeff>(matrix: &SeriesMatrix<C>, scalar: &QSeries<C>) -> SeriesMatrix<C> {
    SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| row.iter().map(|entry| entry.mul(scalar)).collect())
            .collect(),
    )
}

fn invert_series_matrix_coeff<C: Coeff>(
    matrix: &SeriesMatrix<C>,
) -> Result<SeriesMatrix<C>, GwError> {
    if matrix.rows() != matrix.cols() {
        return Err(GwError::ConventionMismatch(
            "matrix inversion requires a square matrix".to_string(),
        ));
    }
    let size = matrix.rows();
    let q_degree = matrix.max_degree();
    let mut augmented = vec![vec![QSeries::<C>::zero(q_degree); 2 * size]; size];
    for (row, augmented_row) in augmented.iter_mut().enumerate() {
        for col in 0..size {
            augmented_row[col] = matrix.entry(row, col).clone();
        }
        augmented_row[size + row] = QSeries::one(q_degree);
    }

    for col in 0..size {
        let pivot = (col..size)
            .find(|row| qseries_has_invertible_constant_coeff(&augmented[*row][col]))
            .ok_or_else(|| GwError::NonSemisimplePoint)?;
        if pivot != col {
            augmented.swap(pivot, col);
        }
        let pivot_inv = augmented[col][col].inverse()?;
        for entry in &mut augmented[col] {
            *entry = entry.mul(&pivot_inv);
        }
        let pivot_row = augmented[col].clone();
        for row in 0..size {
            if row == col {
                continue;
            }
            let factor = augmented[row][col].clone();
            if factor.is_zero() {
                continue;
            }
            for (entry, pivot_entry) in augmented[row].iter_mut().zip(&pivot_row) {
                *entry = entry.sub(&factor.mul(pivot_entry));
            }
        }
    }

    Ok(SeriesMatrix::from_entries(
        augmented
            .into_iter()
            .map(|row| row.into_iter().skip(size).collect())
            .collect(),
    ))
}

fn qseries_has_invertible_constant_coeff<C: Coeff>(series: &QSeries<C>) -> bool {
    series.coeff(0).is_some_and(|constant| !constant.is_zero())
}

fn derivative_qseries_polynomial_coefficients<C: Coeff>(
    coefficients: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coeff)| coeff.scale(&C::from_usize(power)))
        .chain(std::iter::once(QSeries::<C>::zero(q_degree)))
        .collect::<Vec<_>>()
}

fn evaluate_qseries_polynomial<C: Coeff>(
    coefficients: &[QSeries<C>],
    x: &QSeries<C>,
) -> QSeries<C> {
    let q_degree = x.max_degree();
    let mut out = QSeries::<C>::zero(q_degree);
    for coeff in coefficients.iter().rev() {
        out = out.mul(x).add(coeff);
    }
    out
}

fn canonical_evaluation_matrix_local(roots: &[QSeries]) -> SeriesMatrix {
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

fn relative_sqrt_delta_series_local(delta: &QSeries) -> Result<QSeries, GwError> {
    relative_sqrt_delta_series_coeff(delta)
}

fn relative_sqrt_delta_series_coeff<C: Coeff>(delta: &QSeries<C>) -> Result<QSeries<C>, GwError> {
    let delta0 = delta
        .coeff(0)
        .ok_or_else(|| GwError::AlgebraFailure("empty twisted Delta series".to_string()))?;
    if delta0.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    let inv_delta0 = C::one().div(delta0);
    delta.scale(&inv_delta0).sqrt_with_constant_one()
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
    for r in 1..=((z_order + 1) / 2) {
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
    for r in 1..=((z_order + 1) / 2) {
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

fn exp_scalar_z_series_local(exponent: &[RatFun]) -> Vec<RatFun> {
    exp_scalar_z_series_coeff(exponent)
}

fn exp_scalar_z_series_coeff<C: Coeff>(exponent: &[C]) -> Vec<C> {
    let z_order = exponent.len().saturating_sub(1);
    let mut out = vec![C::zero(); z_order + 1];
    out[0] = C::one();
    for degree in 1..=z_order {
        let mut total = C::zero();
        for part in 1..=degree {
            if exponent[part].is_zero() {
                continue;
            }
            let term = C::from_usize(part)
                .mul(&exponent[part])
                .mul(&out[degree - part]);
            total = total.add(&term);
        }
        out[degree] = total.div(&C::from_usize(degree));
    }
    out
}

fn bernoulli_number_local(n: usize) -> Rational {
    let mut bernoulli = vec![Rational::zero(); n + 1];
    bernoulli[0] = Rational::one();
    for degree in 1..=n {
        let mut sum = Rational::zero();
        for idx in 0..degree {
            sum += binomial_rational(degree + 1, idx) * bernoulli[idx].clone();
        }
        bernoulli[degree] = -sum / Rational::from(degree + 1);
    }
    bernoulli[n].clone()
}

fn multiply_polynomial_by_linear_series<C: Coeff>(
    poly: &[QSeries<C>],
    constant: &QSeries<C>,
    max_q_degree: usize,
) -> Vec<QSeries<C>> {
    multiply_polynomial_by_affine_h_series(
        poly,
        constant,
        &QSeries::<C>::one(max_q_degree),
        max_q_degree,
    )
}

fn multiply_polynomial_by_affine_h_series<C: Coeff>(
    poly: &[QSeries<C>],
    constant: &QSeries<C>,
    h_coeff: &QSeries<C>,
    max_q_degree: usize,
) -> Vec<QSeries<C>> {
    let mut out = vec![QSeries::<C>::zero(max_q_degree); poly.len() + 1];
    for (degree, coeff) in poly.iter().enumerate() {
        out[degree] = out[degree].add(&coeff.mul(constant));
        out[degree + 1] = out[degree + 1].add(&coeff.mul(h_coeff));
    }
    out
}

fn constant_matrix_at_q_degree(matrix: &SeriesMatrix, q_degree: usize) -> SeriesMatrix {
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

fn integrate_q_derivative_zero_constant_matrix(
    matrix: &SeriesMatrix,
) -> Result<SeriesMatrix, GwError> {
    Ok(SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| {
                row.iter()
                    .map(integrate_q_derivative_zero_constant)
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

fn integrate_q_derivative_zero_constant(series: &QSeries) -> Result<QSeries, GwError> {
    if series.coeff(0).is_some_and(|constant| !constant.is_zero()) {
        return Err(GwError::AlgebraFailure(
            "cannot integrate q d/dq with nonzero constant term and zero integration constant"
                .to_string(),
        ));
    }
    let max_degree = series.max_degree();
    let mut coeffs = vec![RatFun::zero(); max_degree + 1];
    for degree in 1..=max_degree {
        coeffs[degree] =
            series.coeff(degree).cloned().unwrap_or_else(RatFun::zero) / RatFun::from(degree);
    }
    Ok(QSeries::from_coeffs(coeffs))
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
mod tests {
    use super::*;
    use crate::algebra::Rational;
    use crate::geometry::CohomologyClass;
    use crate::tau;
    use std::collections::BTreeMap;

    #[test]
    fn negative_split_degrees_must_be_positive() {
        assert!(NegativeSplitBundleTwist::new(vec![3]).is_ok());
        assert!(NegativeSplitBundleTwist::new(vec![1, 1]).is_ok());
        assert!(NegativeSplitBundleTwist::new(vec![0]).is_err());
    }

    #[test]
    fn local_cy_threefold_dimension_is_degree_independent_without_insertions() {
        let local_p2 = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let conifold = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();

        for genus in 0..=4 {
            for degree in 0..=5 {
                assert_eq!(local_p2.virtual_dimension(2, genus, degree, 0), 0);
                assert_eq!(conifold.virtual_dimension(1, genus, degree, 0), 0);
            }
        }
    }

    #[test]
    fn local_cy_provider_returns_all_degree_candidates_when_dimension_matches() {
        let provider = TwistedProjectiveSpaceProvider::new(2, vec![3], false).unwrap();
        assert_eq!(
            provider.candidate_degrees_from_dimension(2, 4, &[]),
            vec![0, 1, 2, 3, 4]
        );
        let h = tau(0, CohomologyClass::h_power(2, 2));
        assert!(provider
            .candidate_degrees_from_dimension(2, 4, &[h])
            .is_empty());
    }

    #[test]
    fn twisted_quantum_relation_records_local_p2_symbol() {
        let relation = TwistedQuantumRelation::new(
            2,
            NegativeSplitBundleTwist::new(vec![3]).unwrap(),
            vec![
                Rational::from(1usize),
                Rational::from(2usize),
                Rational::from(3usize),
            ],
        )
        .unwrap();
        let coefficients = relation.coefficients(1);

        assert_eq!(
            coefficients[0].coeff(0),
            Some(&RatFun::from_rational(Rational::from(-6)))
        );
        assert_eq!(
            coefficients[1].coeff(0),
            Some(&RatFun::from_rational(Rational::from(11)))
        );
        assert_eq!(
            coefficients[2].coeff(0),
            Some(&RatFun::from_rational(Rational::from(-6)))
        );
        assert_eq!(coefficients[3].coeff(0), Some(&RatFun::one()));
        assert_eq!(
            coefficients[3].coeff(1),
            Some(&RatFun::from_rational(Rational::from(27usize)))
        );
    }

    #[test]
    fn local_p2_hypergeometric_i_function_has_expected_first_mirror_term() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let coefficient = negative_split_i_function_coefficient(2, &twist, 1);
        assert_eq!(coefficient.coefficient(1, -1), Rational::from(-6));
        assert_eq!(
            negative_split_mirror_map_coefficients(2, &twist, 2)[1],
            Rational::from(-6)
        );
        assert_eq!(
            negative_split_inverse_mirror_map_coefficients(2, &twist, 2)[2],
            Rational::from(6usize)
        );
    }

    #[test]
    fn projective_i_function_coefficient_records_denominator_series() {
        let coefficient = projective_i_function_coefficient(2, 1);

        assert_eq!(coefficient.coefficient(0, -3), Rational::one());
        assert_eq!(coefficient.coefficient(1, -4), Rational::from(-3));
        assert_eq!(coefficient.coefficient(2, -5), Rational::from(6usize));
    }

    #[test]
    fn equivariant_projective_i_specializes_to_nonequivariant_i_at_zero_weights() {
        let equivariant = projective_equivariant_i_function_coefficient(
            2,
            1,
            &[Rational::zero(), Rational::zero(), Rational::zero()],
            -5,
        )
        .unwrap();

        assert_eq!(equivariant, projective_i_function_coefficient(2, 1));
    }

    #[test]
    fn equivariant_projective_i_records_base_weight_correction() {
        let coefficient = projective_equivariant_i_function_coefficient(
            1,
            1,
            &[Rational::from(2usize), Rational::from(5usize)],
            -3,
        )
        .unwrap();

        assert_eq!(coefficient.coefficient(0, -2), Rational::one());
        assert_eq!(coefficient.coefficient(0, -3), Rational::from(7usize));
        assert_eq!(coefficient.coefficient(1, -3), Rational::from(-2));
    }

    #[test]
    fn qrr_factorized_i_function_matches_direct_hypergeometric_i_function() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let qrr = NegativeSplitQrrOperator::new(twist.clone());

        for degree in 0..=4 {
            assert_eq!(
                qrr.apply_to_projective_i_coefficient(2, degree),
                negative_split_i_function_coefficient(2, &twist, degree)
            );
        }
    }

    #[test]
    fn equivariant_negative_split_i_specializes_to_direct_local_i_at_zero_weights() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let equivariant = negative_split_equivariant_i_function_coefficient(
            2,
            &twist,
            1,
            &[Rational::zero(), Rational::zero(), Rational::zero()],
            &[Rational::zero()],
            -5,
        )
        .unwrap();

        assert_eq!(
            equivariant,
            negative_split_i_function_coefficient(2, &twist, 1)
        );
    }

    #[test]
    fn equivariant_negative_split_qrr_factor_records_fiber_weight() {
        let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
        let factor = negative_split_equivariant_qrr_euler_factor(
            1,
            &twist,
            1,
            &[Rational::zero(), Rational::zero()],
            &[Rational::from(3usize), Rational::from(7usize)],
        )
        .unwrap();

        assert_eq!(factor.coefficient(0, 0), Rational::from(21usize));
        assert_eq!(factor.coefficient(1, 0), Rational::from(-10));
    }

    #[test]
    fn twisted_canonical_roots_solve_principal_relation() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let base_weights = vec![
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ];
        let fiber_weights = vec![Rational::from(7usize)];
        let canonical =
            specialized_twisted_quantum_canonical_data(2, &twist, 3, &base_weights, &fiber_weights)
                .unwrap();

        for root in &canonical.roots {
            let value =
                twisted_relation_series_at_weights(2, &twist, root, &base_weights, &fiber_weights)
                    .unwrap();
            assert!(
                value.coeffs().iter().all(RatFun::is_zero),
                "root {root} does not solve relation: {value}"
            );
        }
    }

    #[test]
    fn twisted_metric_norm_q_zero_is_euler_over_tangent_euler() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let base_weights = vec![
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ];
        let fiber_weights = vec![Rational::from(7usize)];
        let canonical =
            specialized_twisted_quantum_canonical_data(2, &twist, 1, &base_weights, &fiber_weights)
                .unwrap();

        for branch in 0..=2 {
            let lambda_i = base_weights[branch].clone();
            let fiber = fiber_weights[0].clone() - Rational::from(3usize) * lambda_i.clone();
            let mut tangent = Rational::one();
            for (other, weight) in base_weights.iter().enumerate() {
                if other != branch {
                    tangent = tangent * (lambda_i.clone() - weight.clone());
                }
            }
            let expected = RatFun::from_rational(fiber / tangent);
            assert_eq!(canonical.metric_norms[branch].coeff(0), Some(&expected));
        }
    }

    #[test]
    fn relation_idempotents_do_not_diagonalize_flat_pairing_beyond_q_zero() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let base_weights = vec![
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ];
        let fiber_weights = vec![Rational::from(7usize)];
        let q_degree = 3;
        let canonical = specialized_twisted_quantum_canonical_data(
            2,
            &twist,
            q_degree,
            &base_weights,
            &fiber_weights,
        )
        .unwrap();
        let flat_metric =
            twisted_flat_metric_matrix(2, q_degree, &twist, &base_weights, &fiber_weights).unwrap();
        let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());
        let canonical_metric = transition.transpose().mul(&flat_metric).mul(&transition);

        let mut found_quantum_off_diagonal = false;
        for row in 0..=2 {
            for col in 0..=2 {
                if row != col
                    && canonical_metric
                        .entry(row, col)
                        .coeffs()
                        .iter()
                        .skip(1)
                        .any(|coeff| !coeff.is_zero())
                {
                    found_quantum_off_diagonal = true;
                }
            }
        }
        assert!(found_quantum_off_diagonal);
    }

    #[test]
    fn equivariant_birkhoff_s_matrix_builds_from_weighted_i_function() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let model = NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
            2,
            twist,
            1,
            1,
            vec![
                Rational::from(1usize),
                Rational::from(2usize),
                Rational::from(4usize),
            ],
            vec![Rational::from(7usize)],
        )
        .unwrap();
        let descendant_s = model.birkhoff_descendant_s_matrix(1).unwrap();

        assert_eq!(descendant_s.size(), 3);
        assert_eq!(descendant_s.q_degree(), 1);
        assert_eq!(descendant_s.z_order(), 1);
        assert_eq!(
            descendant_s.calibration(),
            &CalibrationId("negative-split-equivariant-hypergeometric-birkhoff".to_string())
        );
    }

    #[test]
    fn conifold_birkhoff_quantum_product_is_self_adjoint_and_semisimple() {
        let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
        let base_weights = vec![Rational::from(1usize), Rational::from(3usize)];
        let fiber_weights = vec![Rational::from(5usize), Rational::from(7usize)];
        let canonical = specialized_twisted_birkhoff_canonical_data(
            1,
            &twist,
            1,
            &base_weights,
            &fiber_weights,
        )
        .unwrap();
        assert_birkhoff_idempotents_diagonalize_inverse_euler_pairing(
            1,
            1,
            &twist,
            &base_weights,
            &fiber_weights,
            &canonical,
        );
        assert_eq!(canonical.roots.len(), 2);
    }

    #[test]
    fn local_p2_birkhoff_quantum_product_is_self_adjoint_and_semisimple() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let base_weights = vec![
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ];
        let fiber_weights = vec![Rational::from(7usize)];
        let canonical = specialized_twisted_birkhoff_canonical_data(
            2,
            &twist,
            1,
            &base_weights,
            &fiber_weights,
        )
        .unwrap();

        assert_birkhoff_idempotents_diagonalize_inverse_euler_pairing(
            2,
            1,
            &twist,
            &base_weights,
            &fiber_weights,
            &canonical,
        );
        assert_eq!(canonical.roots.len(), 3);
    }

    #[test]
    fn birkhoff_calibration_skeleton_has_inverse_psi() {
        let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
        let calibration = negative_split_twisted_birkhoff_calibration_skeleton(
            1,
            &twist,
            1,
            1,
            &[Rational::from(1usize), Rational::from(3usize)],
            &[Rational::from(5usize), Rational::from(7usize)],
        )
        .unwrap();

        assert_eq!(
            calibration.psi_inverse.mul(&calibration.psi),
            SeriesMatrix::identity(2, 1)
        );
        calibration.r_matrix.check_identity_calibration().unwrap();
    }

    #[test]
    fn local_p2_birkhoff_r_candidate_is_unitary_at_low_order() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let calibration = negative_split_twisted_birkhoff_calibration_candidate(
            2,
            &twist,
            2,
            2,
            &[
                Rational::from(1usize),
                Rational::from(2usize),
                Rational::from(4usize),
            ],
            &[Rational::from(7usize)],
        )
        .unwrap();

        calibration
            .r_matrix
            .check_unitarity(&calibration.metric)
            .unwrap();
        assert_eq!(
            calibration.r_matrix.calibration(),
            &CalibrationId("negative-split-equivariant-birkhoff-qrr-candidate".to_string())
        );
    }

    #[test]
    fn local_p2_birkhoff_graph_recovers_known_genus_zero_divisor_row() {
        let provider = TwistedProjectiveSpaceProvider::new(2, vec![3], false).unwrap();
        let insertions = vec![
            tau(0, CohomologyClass::h_power(2, 1)),
            tau(0, CohomologyClass::h_power(2, 1)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ];
        let expected = [
            (1, RatFun::from(3usize)),
            (2, RatFun::from(-45)),
            (3, RatFun::from(732usize)),
        ];

        for (degree, oracle) in expected {
            let value = crate::givental::compute_semisimple_graph_value(
                &provider,
                0,
                degree,
                &insertions,
                None,
            )
            .unwrap();
            assert_eq!(value, oracle, "local P2 <H,H,H>_0,{degree}");
        }
    }

    #[test]
    fn o_minus_one_p2_birkhoff_graph_matches_localization_row() {
        let provider = TwistedProjectiveSpaceProvider::rational_lambda_line_with_weights(
            2,
            vec![1],
            vec![
                Rational::from(1usize),
                Rational::from(2usize),
                Rational::from(4usize),
            ],
            vec![Rational::from(0usize)],
        )
        .unwrap();
        let cases = [
            (tau(5, CohomologyClass::one(2)), RatFun::zero(), "tau5(1)"),
            (
                tau(4, CohomologyClass::h_power(2, 1)),
                RatFun::from_rational(Rational::new(-1, 480)),
                "tau4(H)",
            ),
            (
                tau(3, CohomologyClass::h_power(2, 2)),
                RatFun::from_rational(Rational::new(-7, 480)),
                "tau3(H^2)",
            ),
        ];

        for (insertion, oracle, label) in cases {
            let value = crate::givental::compute_semisimple_graph_value(
                &provider,
                2,
                2,
                &[insertion],
                None,
            )
            .unwrap();
            assert_eq!(value, oracle, "O(-1)->P2 g=2 d=2 {label}");
        }
    }

    fn assert_birkhoff_idempotents_diagonalize_inverse_euler_pairing(
        n: usize,
        q_degree: usize,
        twist: &NegativeSplitBundleTwist,
        base_weights: &[Rational],
        fiber_weights: &[Rational],
        canonical: &SpecializedTwistedBirkhoffCanonicalData,
    ) {
        let flat_metric = twisted_inverse_euler_flat_metric_matrix(
            n,
            q_degree,
            twist,
            base_weights,
            fiber_weights,
        )
        .unwrap();
        let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());
        let canonical_metric = transition.transpose().mul(&flat_metric).mul(&transition);

        for row in 0..=n {
            for col in 0..=n {
                let expected = if row == col {
                    canonical.metric_norms[row].clone()
                } else {
                    QSeries::zero(q_degree)
                };
                assert_eq!(canonical_metric.entry(row, col), &expected);
            }
        }
    }

    #[test]
    fn twisted_relation_calibration_skeleton_has_inverse_psi() {
        let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
        let calibration = negative_split_twisted_relation_calibration_skeleton(
            1,
            &twist,
            2,
            2,
            &[Rational::from(1usize), Rational::from(3usize)],
            &[Rational::from(5usize), Rational::from(7usize)],
        )
        .unwrap();

        assert_eq!(
            calibration.psi_inverse.mul(&calibration.psi),
            SeriesMatrix::identity(2, 2)
        );
        calibration.r_matrix.check_identity_calibration().unwrap();
    }

    #[test]
    fn twisted_classical_diagonal_subtracts_inverse_euler_fiber_qrr_term() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let diagonal = twisted_classical_limit_diagonal_coefficients(
            2,
            &twist,
            2,
            &[
                Rational::from(1usize),
                Rational::from(2usize),
                Rational::from(4usize),
            ],
            &[Rational::from(7usize)],
        )
        .unwrap();

        let tangent =
            Rational::one() / Rational::from(1usize) + Rational::one() / Rational::from(3usize);
        let fiber = Rational::one() / Rational::from(4usize);
        let expected = RatFun::from_rational((tangent - fiber) / Rational::from(12usize));
        assert_eq!(diagonal[0][1], expected);
    }

    #[test]
    fn twisted_candidate_r_calibration_fails_unitarity_guard() {
        let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
        let err = negative_split_twisted_relation_calibration_candidate(
            1,
            &twist,
            2,
            2,
            &[Rational::from(1usize), Rational::from(3usize)],
            &[Rational::from(5usize), Rational::from(7usize)],
        )
        .unwrap_err();

        match err {
            GwError::ValidationFailure(message) => {
                assert!(message.contains("R(-z)^T eta R(z)"));
            }
            other => panic!("expected unitarity validation failure, got {other:?}"),
        }
    }

    #[test]
    fn twisted_raw_r_candidate_has_nontrivial_r_coefficients() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let calibration = negative_split_twisted_relation_calibration_raw_candidate(
            2,
            &twist,
            1,
            2,
            &[
                Rational::from(1usize),
                Rational::from(2usize),
                Rational::from(4usize),
            ],
            &[Rational::from(7usize)],
        )
        .unwrap();

        assert!(calibration
            .r_matrix
            .coefficient(1)
            .unwrap()
            .entries()
            .iter()
            .flat_map(|row| row.iter())
            .any(|entry| !entry.is_zero()));
    }

    #[test]
    fn qrr_and_direct_hypergeometric_paths_have_same_mirror_data() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let direct = NegativeSplitHypergeometricModel::new(2, twist.clone(), 3);
        let qrr = NegativeSplitQrrModel::new(2, twist, 3);

        assert_eq!(
            qrr.i_coefficients(),
            direct.i_coefficients(),
            "QRR-factorized coefficients should reproduce the direct local I-function"
        );
        assert_eq!(
            qrr.mirror_map_coefficients(),
            direct.mirror_map_coefficients()
        );
        assert_eq!(
            qrr.inverse_mirror_map_coefficients(),
            direct.inverse_mirror_map_coefficients()
        );
        assert_eq!(
            qrr.mirror_transformed_j_coefficients(),
            direct.mirror_transformed_j_coefficients()
        );
    }

    #[test]
    fn local_p2_mirror_transform_cancels_j_h_over_z_terms() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let model = NegativeSplitHypergeometricModel::new(2, twist, 3);
        let j_coefficients = model.mirror_transformed_j_coefficients();

        for coefficient in j_coefficients.iter().take(4).skip(1) {
            assert_eq!(coefficient.coefficient(1, -1), Rational::zero());
        }
    }

    #[test]
    fn local_p2_fundamental_solution_is_identity_at_q_zero() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let model = NegativeSplitHypergeometricModel::new(2, twist, 2);
        let fundamental = model.fundamental_solution_matrix();

        for (z_power, matrix) in &fundamental {
            let q0 = matrix_q_coefficient(matrix, 0);
            let expected = if *z_power == 0 {
                identity_coeff_matrix(3)
            } else {
                zero_coeff_matrix(3)
            };
            assert_eq!(q0, expected);
        }
    }

    #[test]
    fn twisted_quantum_relation_records_resolved_conifold_symbol() {
        let relation = TwistedQuantumRelation::new(
            1,
            NegativeSplitBundleTwist::new(vec![1, 1]).unwrap(),
            vec![Rational::from(1usize), Rational::from(2usize)],
        )
        .unwrap();
        let coefficients = relation.coefficients(1);

        assert_eq!(
            coefficients[0].coeff(0),
            Some(&RatFun::from_rational(Rational::from(2usize)))
        );
        assert_eq!(
            coefficients[1].coeff(0),
            Some(&RatFun::from_rational(Rational::from(-3)))
        );
        assert_eq!(coefficients[2].coeff(0), Some(&RatFun::one()));
        assert_eq!(
            coefficients[2].coeff(1),
            Some(&RatFun::from_rational(Rational::from(-1)))
        );
    }

    #[test]
    fn twisted_descendant_s_matrix_uses_hypergeometric_birkhoff_model() {
        let provider = TwistedProjectiveSpaceProvider::new(2, vec![3], false).unwrap();
        let descendant_s = provider.descendant_s_matrix(2, 2).unwrap();

        assert_eq!(descendant_s.size(), 3);
        assert_eq!(descendant_s.q_degree(), 2);
        assert_eq!(descendant_s.z_order(), 2);
        assert_eq!(
            descendant_s.calibration(),
            &CalibrationId(
                "negative-split-equivariant-hypergeometric-birkhoff-metric-adjoint".to_string()
            )
        );
        assert_eq!(
            descendant_s.coefficient(0),
            Some(&SeriesMatrix::identity(3, 2))
        );
        assert!(descendant_s
            .coefficient(1)
            .unwrap()
            .entries()
            .iter()
            .flat_map(|row| row.iter())
            .any(|entry| !entry.coeff(1).unwrap().is_zero()));
    }

    #[test]
    fn qrr_birkhoff_s_matches_direct_hypergeometric_birkhoff_s() {
        let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
        let direct = NegativeSplitHypergeometricModel::new(1, twist.clone(), 3);
        let qrr = NegativeSplitQrrModel::new(1, twist, 3);

        assert_eq!(
            qrr.birkhoff_descendant_s_matrix(2).unwrap().coefficients(),
            direct
                .birkhoff_descendant_s_matrix(2)
                .unwrap()
                .coefficients()
        );
    }

    #[test]
    fn qrr_birkhoff_s_has_separate_calibration_label() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let qrr = NegativeSplitQrrModel::new(2, twist, 2);
        let descendant_s = qrr.birkhoff_descendant_s_matrix(2).unwrap();

        assert_eq!(
            descendant_s.calibration(),
            &CalibrationId("negative-split-qrr-hypergeometric-birkhoff".to_string())
        );
    }

    #[test]
    fn negative_split_compute_matches_resolved_conifold_closed_formula() {
        let cases = [(2, 1), (2, 2), (2, 3), (2, 4), (3, 1)];
        for (genus, degree) in cases {
            let req =
                TwistedInvariantRequest::new(1, vec![1, 1], genus, degree, Vec::new()).unwrap();
            let result = compute_negative_split_twisted(&req).unwrap();
            assert_eq!(
                result.value,
                RatFun::from_rational(
                    crate::validation_backends::local_cy::resolved_conifold_gw(genus, degree)
                        .unwrap()
                ),
                "resolved conifold g={genus} d={degree}"
            );
            assert_eq!(
                result.engine,
                "twisted-negative-split-givental-birkhoff-early-line"
            );
        }
    }

    #[test]
    fn o_minus_one_p2_genus_zero_two_point_descendant_uses_unstable_s_matrix() {
        let req = TwistedInvariantRequest::new(
            2,
            vec![1],
            0,
            2,
            vec![
                tau(2, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 2)),
            ],
        )
        .unwrap();
        let result = compute_negative_split_twisted(&req).unwrap();
        assert_eq!(result.value, RatFun::from_rational(Rational::new(-1, 2)));
        assert!(result.notes[0].contains("two-point unstable S-matrix"));
    }

    #[test]
    fn o_minus_one_p2_genus_zero_three_primary_uses_frobenius_product() {
        let mut req = TwistedInvariantRequest::new(
            2,
            vec![1],
            0,
            1,
            vec![
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 1)),
            ],
        )
        .unwrap();
        req.equivariant = true;

        let result = compute_negative_split_twisted(&req).unwrap();
        assert_eq!(result.value, RatFun::one());
        assert!(result.notes[0].contains("Frobenius quantum product"));
    }

    #[test]
    fn factored_o_minus_one_p2_three_primary_matches_expanded_product() {
        let mut req = TwistedInvariantRequest::new(
            2,
            vec![1],
            0,
            1,
            vec![
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 1)),
            ],
        )
        .unwrap();
        req.equivariant = true;

        let factored = compute_negative_split_twisted_factored(&req).unwrap();
        assert_eq!(factored.to_ratfun(), RatFun::one());
    }

    #[test]
    fn fiber_equivariant_twisted_does_not_prune_dimension_mismatch() {
        let mut zero_req = TwistedInvariantRequest::new(
            2,
            vec![1],
            0,
            1,
            vec![
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 2)),
            ],
        )
        .unwrap();

        let nonequivariant = compute_negative_split_twisted(&zero_req).unwrap();
        assert_eq!(nonequivariant.value, RatFun::zero());
        assert_eq!(nonequivariant.engine, "twisted-negative-split-dimension");

        zero_req.equivariant = true;
        let expanded_zero = compute_negative_split_twisted(&zero_req).unwrap();
        assert_eq!(expanded_zero.value, RatFun::zero());
        let factored_zero = compute_negative_split_twisted_factored(&zero_req).unwrap();
        assert_eq!(factored_zero.to_ratfun(), RatFun::zero());

        let mut nonzero_req = TwistedInvariantRequest::new(
            2,
            vec![2],
            0,
            1,
            vec![
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 1)),
            ],
        )
        .unwrap();
        let expected = RatFun::variable("mu_0");

        let nonequivariant = compute_negative_split_twisted(&nonzero_req).unwrap();
        assert_eq!(nonequivariant.value, RatFun::zero());
        assert_eq!(nonequivariant.engine, "twisted-negative-split-dimension");

        nonzero_req.equivariant = true;
        let expanded = compute_negative_split_twisted(&nonzero_req).unwrap();
        assert_eq!(expanded.value, expected);
        assert!(expanded.notes[0].contains("Frobenius quantum product"));

        let factored = compute_negative_split_twisted_factored(&nonzero_req).unwrap();
        assert_eq!(factored.to_ratfun(), expected);
    }

    #[test]
    fn fiber_equivariant_degree_one_top_terms_match_untwisted_p2() {
        let insertions = vec![
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ];
        let untwisted =
            crate::compute(crate::InvariantRequest::new(2, 0, 1, insertions.clone())).unwrap();
        assert_eq!(untwisted.value, RatFun::one());

        let cases = [
            (vec![2], RatFun::variable("mu_0")),
            {
                let mu = RatFun::variable("mu_0");
                (vec![3], &mu * &mu)
            },
            {
                let mu0 = RatFun::variable("mu_0");
                let mu1 = RatFun::variable("mu_1");
                (vec![2, 2], &mu0 * &mu1)
            },
        ];

        for (twist, expected) in cases {
            let nonequivariant =
                TwistedInvariantRequest::new(2, twist.clone(), 0, 1, insertions.clone()).unwrap();
            let local_constant_term = compute_negative_split_twisted(&nonequivariant).unwrap();
            assert_eq!(
                local_constant_term.value,
                RatFun::zero(),
                "constant term for twist {twist:?}"
            );

            let mut equivariant = nonequivariant;
            equivariant.equivariant = true;
            let value = compute_negative_split_twisted_factored(&equivariant).unwrap();
            let mut zero_fiber_weights = BTreeMap::new();
            for idx in 0..twist.len() {
                zero_fiber_weights.insert(format!("mu_{idx}"), Rational::zero());
            }

            assert_eq!(value.to_ratfun(), expected, "top term for twist {twist:?}");
            assert_eq!(
                value.evaluate_variables(&zero_fiber_weights).unwrap(),
                Rational::zero(),
                "constant term by mu=0 for twist {twist:?}"
            );
        }
    }

    #[test]
    fn early_rational_twisted_graph_value_matches_lambda_line_limit() {
        let provider = TwistedProjectiveSpaceProvider::new(1, vec![1, 1], false).unwrap();
        let raw =
            crate::givental::compute_semisimple_graph_value(&provider, 2, 1, &[], None).unwrap();
        let oracle = RatFun::from_rational(
            crate::validation_backends::local_cy::resolved_conifold_gw(2, 1).unwrap(),
        );

        assert_eq!(raw, oracle);
    }

    #[test]
    fn symbolic_raw_twisted_graph_value_has_correct_lambda_line_limit() {
        let provider =
            TwistedProjectiveSpaceProvider::symbolic_lambda_line(1, vec![1, 1], false).unwrap();
        let raw =
            crate::givental::compute_semisimple_graph_value(&provider, 2, 1, &[], None).unwrap();
        let oracle = RatFun::from_rational(
            crate::validation_backends::local_cy::resolved_conifold_gw(2, 1).unwrap(),
        );
        let limit = RatFun::from_rational(
            raw.nonequivariant_limit_line(0, &[Rational::one()])
                .unwrap(),
        );

        assert_eq!(limit, oracle);
    }

    #[test]
    fn fiber_equivariant_i_function_specializes_to_rational_fiber_weights() {
        let twist = NegativeSplitBundleTwist::new(vec![1, 1]).unwrap();
        let base = vec![Rational::from(1usize), Rational::from(2usize)];
        let rational_fiber = vec![Rational::from(3usize), Rational::from(5usize)];
        let symbolic_base = base
            .iter()
            .cloned()
            .map(RatFun::from_rational)
            .collect::<Vec<_>>();
        let symbolic_fiber = vec![RatFun::variable("mu_0"), RatFun::variable("mu_1")];
        let symbolic = negative_split_equivariant_i_function_coefficient_coeff(
            1,
            &twist,
            1,
            &symbolic_base,
            &symbolic_fiber,
            -4,
        )
        .unwrap();
        let rational = negative_split_equivariant_i_function_coefficient(
            1,
            &twist,
            1,
            &base,
            &rational_fiber,
            -4,
        )
        .unwrap();
        let mut values = BTreeMap::new();
        values.insert("mu_0".to_string(), Rational::from(3usize));
        values.insert("mu_1".to_string(), Rational::from(5usize));

        let rendered = (0..=1)
            .flat_map(|h_power| (-4..=0).map(move |z_power| (h_power, z_power)))
            .map(|(h_power, z_power)| symbolic.coefficient(h_power, z_power).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(rendered.contains("mu_0"));
        assert!(rendered.contains("mu_1"));
        for h_power in 0..=1 {
            for z_power in -4..=0 {
                let specialized = symbolic
                    .coefficient(h_power, z_power)
                    .evaluate_variables(&values)
                    .unwrap();
                assert_eq!(specialized, rational.coefficient(h_power, z_power));
            }
        }
    }

    #[test]
    fn fiber_equivariant_inverse_euler_pairing_specializes_to_rational_weights() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let base = vec![
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ];
        let rational_fiber = vec![Rational::from(7usize)];
        let symbolic_base = base
            .iter()
            .cloned()
            .map(RatFun::from_rational)
            .collect::<Vec<_>>();
        let symbolic_fiber = vec![RatFun::variable("mu_0")];
        let symbolic = twisted_inverse_euler_flat_metric_matrix_ratfun(
            2,
            0,
            &twist,
            &symbolic_base,
            &symbolic_fiber,
        )
        .unwrap();
        let rational =
            twisted_inverse_euler_flat_metric_matrix(2, 0, &twist, &base, &rational_fiber).unwrap();
        let mut values = BTreeMap::new();
        values.insert("mu_0".to_string(), Rational::from(7usize));

        assert!(symbolic
            .entry(0, 0)
            .coeff(0)
            .unwrap()
            .to_string()
            .contains("mu_0"));
        for row in 0..=2 {
            for col in 0..=2 {
                let specialized = symbolic
                    .entry(row, col)
                    .coeff(0)
                    .unwrap()
                    .evaluate_variables(&values)
                    .unwrap();
                let expected = rational
                    .entry(row, col)
                    .coeff(0)
                    .unwrap()
                    .as_rational()
                    .unwrap();
                assert_eq!(specialized, expected);
            }
        }
    }

    #[test]
    fn negative_split_compute_matches_local_p2_degree_one() {
        let req = TwistedInvariantRequest::new(2, vec![3], 2, 1, Vec::new()).unwrap();
        let result = compute_negative_split_twisted(&req).unwrap();
        assert_eq!(
            result.value,
            RatFun::from_rational(crate::validation_backends::local_cy::local_p2_gw(2, 1).unwrap(),)
        );
    }

    #[test]
    fn factored_twisted_compute_requires_equivariant_mode() {
        let req = TwistedInvariantRequest::new(
            2,
            vec![1],
            0,
            2,
            vec![
                tau(2, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 2)),
            ],
        )
        .unwrap();
        let err = compute_negative_split_twisted_factored(&req).unwrap_err();

        assert!(err.to_string().contains("--equivariant"));
    }

    #[test]
    fn factored_twisted_two_point_uses_native_s_matrix() {
        let mut req = TwistedInvariantRequest::new(
            2,
            vec![1],
            0,
            1,
            vec![
                tau(1, CohomologyClass::h_power(2, 2)),
                tau(0, CohomologyClass::h_power(2, 1)),
            ],
        )
        .unwrap();
        req.equivariant = true;

        let value = compute_negative_split_twisted_factored(&req).unwrap();

        assert_eq!(value.to_ratfun(), RatFun::one());
    }

    #[test]
    fn factored_flat_metric_vandermonde_inverse_is_identity() {
        let twist = NegativeSplitBundleTwist::new(vec![3]).unwrap();
        let base = vec![
            Rational::from(1usize),
            Rational::from(2usize),
            Rational::from(4usize),
        ];
        let fiber = vec![FactoredRatFun::variable("mu_0")];
        let (metric, inverse) =
            twisted_inverse_euler_flat_metric_pair_from_rational_base(2, 0, &twist, &base, &fiber)
                .unwrap();
        let product = metric.mul(&inverse);
        let mut values = BTreeMap::new();
        values.insert("mu_0".to_string(), Rational::from(7usize));

        for row in 0..=2 {
            for col in 0..=2 {
                let actual = product
                    .entry(row, col)
                    .coeff(0)
                    .unwrap()
                    .evaluate_variables(&values)
                    .unwrap();
                let expected = if row == col {
                    Rational::one()
                } else {
                    Rational::zero()
                };
                assert_eq!(actual, expected, "entry ({row},{col})");
            }
        }
    }

    #[test]
    fn factored_s_matrix_conversion_round_trips_to_ratfun() {
        let mu = RatFun::variable("mu_0");
        let entry = &mu / &(&mu + &RatFun::from(1usize));
        let matrix = SeriesMatrix::constant(vec![vec![entry.clone()]], 0);
        let expanded =
            SeriesSMatrix::from_coefficients(1, 0, 0, vec![matrix], CalibrationId("test".into()))
                .unwrap();
        let factored = series_s_matrix_to_factored(&expanded).unwrap();

        assert_eq!(
            factored
                .coefficient(0)
                .unwrap()
                .entry(0, 0)
                .coeff(0)
                .unwrap()
                .to_ratfun(),
            expanded
                .coefficient(0)
                .unwrap()
                .entry(0, 0)
                .coeff(0)
                .unwrap()
                .clone()
        );
    }

    #[test]
    fn generic_h_laurent_series_preserves_factored_coefficients() {
        let mu = FactoredRatFun::variable("mu_0");
        let mut series = HCoeffLaurentSeries::<FactoredRatFun>::one(1);
        series.add_term(1, -1, mu.clone());
        let relation = vec![FactoredRatFun::one(), FactoredRatFun::zero()];
        let product = series.multiply_mod_relation(&series, &relation);

        assert_eq!(product.coefficient(0, 0).to_ratfun(), RatFun::one());
        assert_eq!(
            product.coefficient(1, -1).to_ratfun(),
            &RatFun::from(2usize) * &RatFun::variable("mu_0")
        );
    }

    #[test]
    fn generic_birkhoff_split_preserves_factored_coefficients() {
        let mu = FactoredRatFun::variable("mu_0");
        let mut fundamental = BTreeMap::new();
        fundamental.insert(
            0,
            SeriesMatrix::from_entries(vec![vec![QSeries::from_coeffs(vec![
                FactoredRatFun::one(),
                FactoredRatFun::zero(),
            ])]]),
        );
        fundamental.insert(
            -1,
            SeriesMatrix::from_entries(vec![vec![QSeries::from_coeffs(vec![
                FactoredRatFun::zero(),
                mu.clone(),
            ])]]),
        );

        let (_, negative) = birkhoff_factor_by_q_degree(1, 1, &fundamental).unwrap();
        let coefficients = negative_factor_to_s_coefficients(1, 1, 1, &negative);

        assert_eq!(
            coefficients[1].entry(0, 0).coeff(1).unwrap().to_ratfun(),
            RatFun::variable("mu_0")
        );
    }

    #[test]
    fn packed_resolvent_matches_invariant_wise_local_p2() {
        let req = crate::resolvent::ResolventRequest {
            target_n: 2,
            genus: 2,
            degree: 1,
            markings: 1,
            virtual_dimension: 1,
        };
        let packed =
            compute_negative_split_twisted_resolvent_packed(2, vec![3], &req, false).unwrap();
        let invariant_wise =
            crate::resolvent::compute_resolvent_generating_function(&req, |insertions| {
                let invariant_req =
                    TwistedInvariantRequest::new(2, vec![3], 2, 1, insertions.to_vec())?;
                compute_negative_split_twisted(&invariant_req)
            })
            .unwrap();

        assert_eq!(packed.value, invariant_wise.value);
        assert_eq!(packed.candidate_terms, invariant_wise.candidate_terms);
        assert_eq!(packed.nonzero_terms, invariant_wise.nonzero_terms);
    }

    #[test]
    fn non_cy_twist_can_still_select_one_degree() {
        let twist = NegativeSplitBundleTwist::new(vec![1]).unwrap();
        let insertion_degree = Some(3);
        assert_eq!(
            twist.candidate_degrees(2, 0, 5, 1, insertion_degree),
            vec![1]
        );
    }
}
