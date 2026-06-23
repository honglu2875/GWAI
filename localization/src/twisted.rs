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
//! Birkhoff/QRR `R` unitarity, including local `P^2`.  The public graph path
//! uses an early rational specialization of the one-parameter lambda line,
//! while the symbolic lambda-line provider remains available as a finite-limit
//! diagnostic.  Fast validation currently covers several resolved-conifold rows
//! and the first local-`P^2` genus-2 row; genus-4 local curve computations are
//! the next observed performance frontier.

use crate::algebra::{RatFun, Rational};
use crate::error::GwError;
use crate::givental::{
    CalibrationId, CanonicalFrameConvention, GiventalGraphKernel, ProjectiveSpaceProvider,
    SemisimpleCalibration, SemisimpleCohftProvider, SeriesRMatrix, SeriesSMatrix,
};
use crate::series::{QSeries, SeriesMatrix};
use crate::{Insertion, InvariantResult, Truncation};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NegativeSplitBundleTwist {
    degrees: Vec<usize>,
}

impl NegativeSplitBundleTwist {
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
struct HRatFunLaurentSeries {
    max_h_power: usize,
    coeffs: Vec<BTreeMap<i32, RatFun>>,
}

impl HRatFunLaurentSeries {
    fn zero(max_h_power: usize) -> Self {
        Self {
            max_h_power,
            coeffs: vec![BTreeMap::new(); max_h_power + 1],
        }
    }

    fn one(max_h_power: usize) -> Self {
        let mut out = Self::zero(max_h_power);
        out.coeffs[0].insert(0, RatFun::one());
        out
    }

    fn coefficient(&self, h_power: usize, z_power: i32) -> RatFun {
        self.coeffs
            .get(h_power)
            .and_then(|terms| terms.get(&z_power))
            .cloned()
            .unwrap_or_else(RatFun::zero)
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

    fn scale(&self, scalar: RatFun) -> Self {
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

    fn shift_z(&self, shift: i32) -> Self {
        let mut out = Self::zero(self.max_h_power);
        for h_power in 0..=self.max_h_power {
            for (z_power, coeff) in &self.coeffs[h_power] {
                out.add_term(h_power, z_power + shift, coeff.clone());
            }
        }
        out
    }

    fn multiply_mod_relation(&self, rhs: &Self, h_power_relation: &[RatFun]) -> Self {
        assert_eq!(self.max_h_power, rhs.max_h_power);
        assert_eq!(h_power_relation.len(), self.max_h_power + 1);
        let basis_powers = h_basis_powers_mod_relation_ratfun(self.max_h_power, h_power_relation);
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

    fn multiply_by_affine_mod_relation(
        &self,
        h_coeff: RatFun,
        constant: RatFun,
        z_coeff: RatFun,
        h_power_relation: &[RatFun],
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

    fn add_term(&mut self, h_power: usize, z_power: i32, coeff: RatFun) {
        if coeff.is_zero() || h_power > self.max_h_power {
            return;
        }
        let terms = &mut self.coeffs[h_power];
        let next = terms.get(&z_power).cloned().unwrap_or_else(RatFun::zero) + coeff;
        if next.is_zero() {
            terms.remove(&z_power);
        } else {
            terms.insert(z_power, next);
        }
    }
}

fn h_basis_powers_mod_relation_ratfun(
    max_h_power: usize,
    h_power_relation: &[RatFun],
) -> Vec<Vec<RatFun>> {
    let mut powers = vec![vec![RatFun::zero(); max_h_power + 1]; 2 * max_h_power + 1];
    powers[0][0] = RatFun::one();
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

fn h_affine_power_mod_relation_ratfun(
    max_h_power: usize,
    h_coeff: RatFun,
    constant: RatFun,
    exponent: usize,
    h_power_relation: &[RatFun],
) -> Vec<RatFun> {
    let mut out = vec![RatFun::zero(); max_h_power + 1];
    out[0] = RatFun::one();
    for _ in 0..exponent {
        let mut next = vec![RatFun::zero(); max_h_power + 1];
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

pub fn negative_split_i_function_coefficient(
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
    let h_power_relation = base_h_power_relation(n, base_weights)?;
    let factor =
        negative_split_equivariant_qrr_euler_factor(n, twist, degree, base_weights, fiber_weights)?;
    let projective =
        projective_equivariant_i_function_coefficient(n, degree, base_weights, min_z_power)?;
    Ok(factor
        .multiply_mod_relation(&projective, &h_power_relation)
        .truncated_z_below(min_z_power))
}

fn negative_split_equivariant_i_function_coefficient_ratfun(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[RatFun],
    fiber_weights: &[RatFun],
    min_z_power: i32,
) -> Result<HRatFunLaurentSeries, GwError> {
    let h_power_relation = base_h_power_relation_ratfun(n, base_weights)?;
    let factor = negative_split_equivariant_qrr_euler_factor_ratfun(
        n,
        twist,
        degree,
        base_weights,
        fiber_weights,
    )?;
    let projective =
        projective_equivariant_i_function_coefficient_ratfun(n, degree, base_weights, min_z_power)?;
    Ok(factor
        .multiply_mod_relation(&projective, &h_power_relation)
        .truncated_z_below(min_z_power))
}

fn projective_equivariant_i_function_coefficient_ratfun(
    n: usize,
    degree: usize,
    base_weights: &[RatFun],
    min_z_power: i32,
) -> Result<HRatFunLaurentSeries, GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    let h_power_relation = base_h_power_relation_ratfun(n, base_weights)?;
    let mut out = HRatFunLaurentSeries::one(n);
    for m in 1..=degree {
        for weight in base_weights {
            let inverse = inverse_affine_z_laurent_ratfun(
                n,
                RatFun::one(),
                -weight.clone(),
                RatFun::from(m),
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

fn negative_split_equivariant_qrr_euler_factor_ratfun(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    base_weights: &[RatFun],
    fiber_weights: &[RatFun],
) -> Result<HRatFunLaurentSeries, GwError> {
    if fiber_weights.len() != twist.rank() {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} fiber weights, got {}",
            twist.rank(),
            fiber_weights.len()
        )));
    }
    let h_power_relation = base_h_power_relation_ratfun(n, base_weights)?;
    let mut out = HRatFunLaurentSeries::one(n);
    for (bundle_degree, fiber_weight) in twist.degrees().iter().zip(fiber_weights) {
        for m in (-(bundle_degree.saturating_mul(degree) as isize) + 1)..=0 {
            out = out.multiply_by_affine_mod_relation(
                RatFun::from_rational(-Rational::from(*bundle_degree)),
                fiber_weight.clone(),
                RatFun::from_rational(Rational::from(m)),
                &h_power_relation,
            );
        }
    }
    Ok(out)
}

fn base_h_power_relation_ratfun(n: usize, base_weights: &[RatFun]) -> Result<Vec<RatFun>, GwError> {
    if base_weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} base weights, got {}",
            n + 1,
            base_weights.len()
        )));
    }
    let mut coefficients = vec![RatFun::one()];
    for weight in base_weights {
        let mut next = vec![RatFun::zero(); coefficients.len() + 1];
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
struct NegativeSplitLineHypergeometricModel {
    n: usize,
    twist: NegativeSplitBundleTwist,
    q_degree: usize,
    base_weights: Vec<RatFun>,
    fiber_weights: Vec<RatFun>,
    min_z_power: i32,
}

impl NegativeSplitLineHypergeometricModel {
    fn new(
        n: usize,
        twist: NegativeSplitBundleTwist,
        q_degree: usize,
        z_order: usize,
        base_weight_factors: &[Rational],
        fiber_weight_factors: &[Rational],
    ) -> Result<Self, GwError> {
        if base_weight_factors.len() != n + 1 {
            return Err(GwError::AlgebraFailure(format!(
                "expected {} base weight factors, got {}",
                n + 1,
                base_weight_factors.len()
            )));
        }
        if fiber_weight_factors.len() != twist.rank() {
            return Err(GwError::AlgebraFailure(format!(
                "expected {} fiber weight factors, got {}",
                twist.rank(),
                fiber_weight_factors.len()
            )));
        }
        let lambda = crate::algebra::lambda(0);
        let base_weights = base_weight_factors
            .iter()
            .map(|weight| lambda.clone() * RatFun::from_rational(weight.clone()))
            .collect::<Vec<_>>();
        let fiber_weights = fiber_weight_factors
            .iter()
            .map(|weight| lambda.clone() * RatFun::from_rational(weight.clone()))
            .collect::<Vec<_>>();
        let min_z_power = -(((n + 1) * q_degree + z_order + 2) as i32);
        Ok(Self {
            n,
            twist,
            q_degree,
            base_weights,
            fiber_weights,
            min_z_power,
        })
    }

    fn i_coefficients(&self) -> Result<Vec<HRatFunLaurentSeries>, GwError> {
        (0..=self.q_degree)
            .map(|degree| {
                negative_split_equivariant_i_function_coefficient_ratfun(
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

    fn mirror_map_coefficients(&self) -> Result<Vec<RatFun>, GwError> {
        Ok(mirror_map_coefficients_from_i_function_ratfun(
            &self.i_coefficients()?,
            self.q_degree,
        ))
    }

    fn inverse_mirror_map_coefficients(&self) -> Result<Vec<RatFun>, GwError> {
        Ok(invert_mirror_map_ratfun(
            &self.mirror_map_coefficients()?,
            self.q_degree,
        ))
    }

    fn mirror_transformed_j_coefficients(&self) -> Result<Vec<HRatFunLaurentSeries>, GwError> {
        let h_power_relation = base_h_power_relation_ratfun(self.n, &self.base_weights)?;
        Ok(
            mirror_transformed_j_coefficients_from_i_function_mod_relation_ratfun(
                self.n,
                &self.i_coefficients()?,
                &self.mirror_map_coefficients()?,
                &self.inverse_mirror_map_coefficients()?,
                self.q_degree,
                &h_power_relation,
            ),
        )
    }

    fn fundamental_solution_matrix(&self) -> Result<BTreeMap<i32, SeriesMatrix>, GwError> {
        let h_power_relation = base_h_power_relation_ratfun(self.n, &self.base_weights)?;
        Ok(
            fundamental_solution_matrix_from_j_coefficients_mod_relation_ratfun(
                self.n,
                self.q_degree,
                &self.mirror_transformed_j_coefficients()?,
                &h_power_relation,
            ),
        )
    }

    fn birkhoff_descendant_s_matrix(&self, z_order: usize) -> Result<SeriesSMatrix, GwError> {
        birkhoff_descendant_s_matrix_from_fundamental(
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

fn mirror_map_coefficients_from_i_function_ratfun(
    i_coefficients: &[HRatFunLaurentSeries],
    q_degree: usize,
) -> Vec<RatFun> {
    let mut out = vec![RatFun::zero(); q_degree + 1];
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
            .unwrap_or_else(RatFun::zero);
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

fn mirror_transformed_j_coefficients_from_i_function_mod_relation_ratfun(
    n: usize,
    i_coefficients: &[HRatFunLaurentSeries],
    _mirror: &[RatFun],
    inverse_mirror: &[RatFun],
    q_degree: usize,
    h_power_relation: &[RatFun],
) -> Vec<HRatFunLaurentSeries> {
    let gauge =
        full_vector_mirror_gauge_coefficients_ratfun(n, i_coefficients, q_degree, h_power_relation);
    let gauged = multiply_h_laurent_q_series_mod_relation_ratfun(
        &gauge,
        i_coefficients,
        q_degree,
        h_power_relation,
    );
    compose_h_laurent_q_series_ratfun(&gauged, inverse_mirror, q_degree)
}

fn fundamental_solution_matrix_from_j_coefficients(
    n: usize,
    q_degree: usize,
    j_coefficients: &[HLaurentSeries],
) -> BTreeMap<i32, SeriesMatrix> {
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

fn fundamental_solution_matrix_from_j_coefficients_mod_relation_ratfun(
    n: usize,
    q_degree: usize,
    j_coefficients: &[HRatFunLaurentSeries],
    h_power_relation: &[RatFun],
) -> BTreeMap<i32, SeriesMatrix> {
    let mut columns = Vec::with_capacity(n + 1);
    let mut current = j_coefficients.to_vec();
    for _ in 0..=n {
        columns.push(current.clone());
        current =
            quantum_derivative_h_laurent_q_series_mod_relation_ratfun(&current, h_power_relation);
    }
    h_ratfun_laurent_columns_to_laurent_matrix(n, q_degree, &columns)
}

fn birkhoff_descendant_s_matrix_from_fundamental(
    size: usize,
    q_degree: usize,
    z_order: usize,
    fundamental: &BTreeMap<i32, SeriesMatrix>,
    calibration: CalibrationId,
) -> Result<SeriesSMatrix, GwError> {
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

fn invert_mirror_map_ratfun(mirror: &[RatFun], q_degree: usize) -> Vec<RatFun> {
    let exp_mirror = exp_series_ratfun(mirror, q_degree);
    let mut q_of_q = vec![RatFun::zero(); q_degree + 1];
    if q_degree >= 1 {
        q_of_q[1] = RatFun::one();
    }
    let target = mul_plain_series_ratfun(&q_of_q, &exp_mirror, q_degree);
    invert_series_with_linear_term_one_ratfun(&target, q_degree)
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

fn exp_series_ratfun(series: &[RatFun], max_degree: usize) -> Vec<RatFun> {
    let mut out = vec![RatFun::zero(); max_degree + 1];
    out[0] = RatFun::one();
    for degree in 1..=max_degree {
        let mut sum = RatFun::zero();
        for split in 1..=degree {
            let coeff = series.get(split).cloned().unwrap_or_else(RatFun::zero);
            sum = sum + RatFun::from(split) * coeff * out[degree - split].clone();
        }
        out[degree] = sum / RatFun::from(degree);
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

fn invert_series_with_linear_term_one_ratfun(series: &[RatFun], max_degree: usize) -> Vec<RatFun> {
    assert_eq!(series.first(), Some(&RatFun::zero()));
    assert_eq!(series.get(1), Some(&RatFun::one()));
    let mut inverse = vec![RatFun::zero(); max_degree + 1];
    if max_degree >= 1 {
        inverse[1] = RatFun::one();
    }
    for degree in 2..=max_degree {
        let mut trial = inverse.clone();
        trial[degree] = RatFun::one();
        let contribution = compose_plain_series_ratfun(series, &trial, max_degree)[degree].clone();
        let mut baseline = inverse.clone();
        baseline[degree] = RatFun::zero();
        let current = compose_plain_series_ratfun(series, &baseline, max_degree)[degree].clone();
        let sensitivity = contribution - current.clone();
        inverse[degree] = -current / sensitivity;
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

fn compose_plain_series_ratfun(
    series: &[RatFun],
    input: &[RatFun],
    max_degree: usize,
) -> Vec<RatFun> {
    let mut out = vec![RatFun::zero(); max_degree + 1];
    let mut power = vec![RatFun::zero(); max_degree + 1];
    power[0] = RatFun::one();
    for degree in 0..=max_degree {
        let coefficient = series.get(degree).cloned().unwrap_or_else(RatFun::zero);
        if !coefficient.is_zero() {
            for idx in 0..=max_degree {
                out[idx] = out[idx].clone() + coefficient.clone() * power[idx].clone();
            }
        }
        power = mul_plain_series_ratfun(&power, input, max_degree);
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

fn mul_plain_series_ratfun(left: &[RatFun], right: &[RatFun], max_degree: usize) -> Vec<RatFun> {
    let mut out = vec![RatFun::zero(); max_degree + 1];
    for left_degree in 0..=max_degree {
        if left[left_degree].is_zero() {
            continue;
        }
        for right_degree in 0..=max_degree - left_degree {
            if right[right_degree].is_zero() {
                continue;
            }
            out[left_degree + right_degree] = out[left_degree + right_degree].clone()
                + left[left_degree].clone() * right[right_degree].clone();
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

fn multiply_h_laurent_q_series_mod_relation_ratfun(
    left: &[HRatFunLaurentSeries],
    right: &[HRatFunLaurentSeries],
    max_degree: usize,
    h_power_relation: &[RatFun],
) -> Vec<HRatFunLaurentSeries> {
    let max_h_power = left
        .first()
        .or_else(|| right.first())
        .map(HRatFunLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HRatFunLaurentSeries::zero(max_h_power); max_degree + 1];
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

fn full_vector_mirror_gauge_coefficients_ratfun(
    n: usize,
    i_coefficients: &[HRatFunLaurentSeries],
    max_degree: usize,
    h_power_relation: &[RatFun],
) -> Vec<HRatFunLaurentSeries> {
    let mut exponent = vec![HRatFunLaurentSeries::zero(n); max_degree + 1];
    let mut gauge = vec![HRatFunLaurentSeries::zero(n); max_degree + 1];
    gauge[0] = HRatFunLaurentSeries::one(n);

    for degree in 1..=max_degree {
        let mut known_gauge = HRatFunLaurentSeries::zero(n);
        for split in 1..degree {
            if exponent[split].is_empty() {
                continue;
            }
            let term = exponent[split]
                .multiply_mod_relation(&gauge[degree - split], h_power_relation)
                .scale(RatFun::from(split));
            known_gauge = known_gauge.add(&term);
        }
        known_gauge = known_gauge.scale(RatFun::one() / RatFun::from(degree));
        gauge[degree] = known_gauge;

        let mut gauged_degree = HRatFunLaurentSeries::zero(n);
        for split in 0..=degree {
            let term = gauge[split]
                .multiply_mod_relation(&i_coefficients[degree - split], h_power_relation);
            gauged_degree = gauged_degree.add(&term);
        }
        let tau = z_power_part_ratfun(&gauged_degree, -1);
        exponent[degree] = tau
            .shift_z(-1)
            .scale(RatFun::from_rational(-Rational::one()));
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

fn z_power_part_ratfun(series: &HRatFunLaurentSeries, z_power: i32) -> HRatFunLaurentSeries {
    let mut out = HRatFunLaurentSeries::zero(series.max_h_power());
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

fn compose_h_laurent_q_series_ratfun(
    series: &[HRatFunLaurentSeries],
    input: &[RatFun],
    max_degree: usize,
) -> Vec<HRatFunLaurentSeries> {
    let max_h_power = series
        .first()
        .map(HRatFunLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HRatFunLaurentSeries::zero(max_h_power); max_degree + 1];
    let mut power = vec![RatFun::zero(); max_degree + 1];
    power[0] = RatFun::one();
    for source_degree in 0..=max_degree {
        for target_degree in 0..=max_degree {
            if power[target_degree].is_zero() {
                continue;
            }
            let term = series[source_degree].scale(power[target_degree].clone());
            out[target_degree] = out[target_degree].add(&term);
        }
        power = mul_plain_series_ratfun(&power, input, max_degree);
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

fn quantum_derivative_h_laurent_q_series_mod_relation_ratfun(
    series: &[HRatFunLaurentSeries],
    h_power_relation: &[RatFun],
) -> Vec<HRatFunLaurentSeries> {
    let max_degree = series.len().saturating_sub(1);
    let max_h_power = series
        .first()
        .map(HRatFunLaurentSeries::max_h_power)
        .unwrap_or(0);
    let mut out = vec![HRatFunLaurentSeries::zero(max_h_power); max_degree + 1];
    for degree in 0..=max_degree {
        out[degree] = out[degree].add(&series[degree].multiply_by_affine_mod_relation(
            RatFun::one(),
            RatFun::zero(),
            RatFun::zero(),
            h_power_relation,
        ));
        if degree > 0 {
            let derivative_term = series[degree].shift_z(1).scale(RatFun::from(degree));
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

fn h_ratfun_laurent_columns_to_laurent_matrix(
    n: usize,
    q_degree: usize,
    columns: &[Vec<HRatFunLaurentSeries>],
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
                    entries[h_power][col][degree] =
                        entries[h_power][col][degree].clone() + coeff.clone();
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

type RatFunMatrix = Vec<Vec<RatFun>>;
type LaurentRatFunMatrix = BTreeMap<i32, RatFunMatrix>;
type QDegreeLaurentFactor = Vec<LaurentRatFunMatrix>;

fn birkhoff_factor_by_q_degree(
    size: usize,
    q_degree: usize,
    matrix: &BTreeMap<i32, SeriesMatrix>,
) -> Result<(QDegreeLaurentFactor, QDegreeLaurentFactor), GwError> {
    validate_identity_at_q_zero(size, matrix)?;
    let mut positive = vec![BTreeMap::new(); q_degree + 1];
    let mut negative = vec![BTreeMap::new(); q_degree + 1];
    positive[0].insert(0, identity_ratfun_matrix(size));
    negative[0].insert(0, identity_ratfun_matrix(size));

    for degree in 1..=q_degree {
        let mut raw = q_degree_slice(matrix, degree, size);
        let known = multiply_laurent_matrix_q_slices(&negative, &positive, degree, size);
        subtract_laurent_matrix(&mut raw, &known);
        for (z_power, coeff) in raw {
            if matrix_is_zero(&coeff) {
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

fn validate_identity_at_q_zero(
    size: usize,
    matrix: &BTreeMap<i32, SeriesMatrix>,
) -> Result<(), GwError> {
    for (z_power, coefficient) in matrix {
        let q0 = matrix_q_coefficient(coefficient, 0);
        let expected = if *z_power == 0 {
            identity_ratfun_matrix(size)
        } else {
            zero_ratfun_matrix(size)
        };
        if q0 != expected {
            return Err(GwError::ConventionMismatch(format!(
                "Birkhoff input must be identity at q=0; z^{z_power} coefficient is nonstandard"
            )));
        }
    }
    Ok(())
}

fn q_degree_slice(
    matrix: &BTreeMap<i32, SeriesMatrix>,
    degree: usize,
    size: usize,
) -> LaurentRatFunMatrix {
    let mut out = BTreeMap::new();
    for (z_power, coefficient) in matrix {
        let q_coeff = matrix_q_coefficient(coefficient, degree);
        if !matrix_is_zero(&q_coeff) {
            out.insert(*z_power, q_coeff);
        }
    }
    if out.is_empty() {
        out.insert(0, zero_ratfun_matrix(size));
    }
    out
}

fn matrix_q_coefficient(matrix: &SeriesMatrix, degree: usize) -> RatFunMatrix {
    matrix
        .entries()
        .iter()
        .map(|row| {
            row.iter()
                .map(|entry| entry.coeff(degree).cloned().unwrap_or_else(RatFun::zero))
                .collect()
        })
        .collect()
}

fn multiply_laurent_matrix_q_slices(
    left: &[LaurentRatFunMatrix],
    right: &[LaurentRatFunMatrix],
    degree: usize,
    size: usize,
) -> LaurentRatFunMatrix {
    let mut out = BTreeMap::new();
    for split in 1..degree {
        for (left_z, left_matrix) in &left[split] {
            for (right_z, right_matrix) in &right[degree - split] {
                let product = multiply_ratfun_matrix(left_matrix, right_matrix, size);
                add_matrix_to_laurent(&mut out, left_z + right_z, product);
            }
        }
    }
    out
}

fn subtract_laurent_matrix(target: &mut LaurentRatFunMatrix, rhs: &LaurentRatFunMatrix) {
    for (z_power, matrix) in rhs {
        add_matrix_to_laurent(target, *z_power, neg_ratfun_matrix(matrix));
    }
}

fn add_matrix_to_laurent(target: &mut LaurentRatFunMatrix, z_power: i32, matrix: RatFunMatrix) {
    if matrix_is_zero(&matrix) {
        return;
    }
    let size = matrix.len();
    let entry = target
        .entry(z_power)
        .or_insert_with(|| zero_ratfun_matrix(size));
    for row in 0..size {
        for col in 0..size {
            entry[row][col] = entry[row][col].clone() + matrix[row][col].clone();
        }
    }
    if matrix_is_zero(entry) {
        target.remove(&z_power);
    }
}

fn negative_factor_to_s_coefficients(
    size: usize,
    q_degree: usize,
    z_order: usize,
    negative: &[LaurentRatFunMatrix],
) -> Vec<SeriesMatrix> {
    let mut coefficients = Vec::with_capacity(z_order + 1);
    for order in 0..=z_order {
        let mut entries = vec![vec![vec![RatFun::zero(); q_degree + 1]; size]; size];
        if order == 0 {
            for idx in 0..size {
                entries[idx][idx][0] = RatFun::one();
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

fn multiply_ratfun_matrix(left: &RatFunMatrix, right: &RatFunMatrix, size: usize) -> RatFunMatrix {
    let mut out = zero_ratfun_matrix(size);
    for row in 0..size {
        for col in 0..size {
            let mut total = RatFun::zero();
            for mid in 0..size {
                total = total + left[row][mid].clone() * right[mid][col].clone();
            }
            out[row][col] = total;
        }
    }
    out
}

fn identity_ratfun_matrix(size: usize) -> RatFunMatrix {
    let mut out = zero_ratfun_matrix(size);
    for idx in 0..size {
        out[idx][idx] = RatFun::one();
    }
    out
}

fn zero_ratfun_matrix(size: usize) -> RatFunMatrix {
    vec![vec![RatFun::zero(); size]; size]
}

fn neg_ratfun_matrix(matrix: &RatFunMatrix) -> RatFunMatrix {
    matrix
        .iter()
        .map(|row| row.iter().cloned().map(|entry| -entry).collect())
        .collect()
}

fn matrix_is_zero(matrix: &RatFunMatrix) -> bool {
    matrix.iter().all(|row| row.iter().all(RatFun::is_zero))
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

fn inverse_affine_z_laurent_ratfun(
    max_h_power: usize,
    h_coeff: RatFun,
    constant: RatFun,
    z_coeff: RatFun,
    min_z_power: i32,
    h_power_relation: Option<&[RatFun]>,
) -> Result<HRatFunLaurentSeries, GwError> {
    if z_coeff.is_zero() {
        return Err(GwError::AlgebraFailure(
            "cannot expand affine inverse at z=infinity with zero z coefficient".to_string(),
        ));
    }
    if min_z_power >= 0 {
        return Ok(HRatFunLaurentSeries::zero(max_h_power));
    }

    let mut out = HRatFunLaurentSeries::zero(max_h_power);
    let max_k = (-min_z_power - 1) as usize;
    for k in 0..=max_k {
        let sign = if k % 2 == 0 {
            RatFun::one()
        } else {
            RatFun::from_rational(-Rational::one())
        };
        let denominator = z_coeff.pow_usize(k + 1);
        if let Some(relation) = h_power_relation {
            let affine_power = h_affine_power_mod_relation_ratfun(
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
                let binom = RatFun::from_rational(binomial_rational(k, h_power));
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
pub struct SpecializedTwistedBirkhoffCanonicalData {
    pub roots: Vec<QSeries>,
    pub metric_norms: Vec<QSeries>,
    pub inverse_metric_norms: Vec<QSeries>,
    pub transition_to_flat: Vec<Vec<QSeries>>,
    pub quantum_h: SeriesMatrix,
}

pub fn specialized_twisted_birkhoff_canonical_data(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    max_q_degree: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SpecializedTwistedBirkhoffCanonicalData, GwError> {
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
        for col in 0..=n {
            if row != col && !canonical_metric.entry(row, col).is_zero() {
                return Err(GwError::ValidationFailure(
                    "Birkhoff idempotents do not diagonalize the twisted flat pairing".to_string(),
                ));
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

fn specialized_twisted_birkhoff_canonical_data_ratfun(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    max_q_degree: usize,
    base_weight_factors: &[Rational],
    fiber_weight_factors: &[Rational],
) -> Result<SpecializedTwistedBirkhoffCanonicalData, GwError> {
    let model = NegativeSplitLineHypergeometricModel::new(
        n,
        twist.clone(),
        max_q_degree,
        1,
        base_weight_factors,
        fiber_weight_factors,
    )?;
    let descendant_s = model.birkhoff_descendant_s_matrix(1)?;
    let classical_h =
        twisted_classical_h_multiplication_matrix_ratfun(n, max_q_degree, &model.base_weights)?;
    let quantum_h = twisted_quantum_multiplication_from_s(
        &descendant_s,
        &classical_h,
        &TwistedCalibrationMode::InverseEuler,
    )?;
    let flat_metric = twisted_inverse_euler_flat_metric_matrix_ratfun(
        n,
        max_q_degree,
        twist,
        &model.base_weights,
        &model.fiber_weights,
    )?;

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

    let charpoly = charpoly_qseries_coefficients(&quantum_h)?;
    let roots = (0..=n)
        .map(|branch| {
            root_series_from_charpoly_ratfun(
                &charpoly,
                model.base_weights[branch].clone(),
                max_q_degree,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transition = spectral_transition_matrix_from_roots(&quantum_h, &roots)?;
    let transition_to_flat = transition.entries().to_vec();
    let canonical_metric = transition.transpose().mul(&flat_metric).mul(&transition);

    let mut metric_norms = Vec::with_capacity(n + 1);
    let mut inverse_metric_norms = Vec::with_capacity(n + 1);
    for row in 0..=n {
        for col in 0..=n {
            if row != col && !canonical_metric.entry(row, col).is_zero() {
                return Err(GwError::ValidationFailure(
                    "Birkhoff idempotents do not diagonalize the twisted lambda-line pairing"
                        .to_string(),
                ));
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
    let transition_inverse = invert_series_matrix(&transition)?;
    let psi = transition.mul(&relative_scale);
    let psi_inverse = relative_scale_inv.mul(&transition_inverse);
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

pub fn negative_split_twisted_birkhoff_calibration_candidate(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weights: &[Rational],
    fiber_weights: &[Rational],
) -> Result<SemisimpleCalibration, GwError> {
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
    let mut calibration = negative_split_twisted_birkhoff_calibration_skeleton_with_mode(
        n,
        twist,
        q_degree,
        z_order,
        base_weights,
        fiber_weights,
        mode.clone(),
    )?;
    let canonical = specialized_twisted_birkhoff_canonical_data_with_mode(
        n,
        twist,
        q_degree,
        base_weights,
        fiber_weights,
        mode.clone(),
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
    calibration.r_matrix.check_unitarity(&calibration.metric)?;
    Ok(calibration)
}

fn negative_split_twisted_birkhoff_calibration_candidate_ratfun(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
    z_order: usize,
    base_weight_factors: &[Rational],
    fiber_weight_factors: &[Rational],
) -> Result<SemisimpleCalibration, GwError> {
    let canonical = specialized_twisted_birkhoff_canonical_data_ratfun(
        n,
        twist,
        q_degree,
        base_weight_factors,
        fiber_weight_factors,
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
    let transition_inverse = invert_series_matrix(&transition)?;
    let psi = transition.mul(&relative_scale);
    let psi_inverse = relative_scale_inv.mul(&transition_inverse);
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

    let lambda = crate::algebra::lambda(0);
    let base_weights = base_weight_factors
        .iter()
        .map(|weight| lambda.clone() * RatFun::from_rational(weight.clone()))
        .collect::<Vec<_>>();
    let fiber_weights = fiber_weight_factors
        .iter()
        .map(|weight| lambda.clone() * RatFun::from_rational(weight.clone()))
        .collect::<Vec<_>>();
    let classical_diagonal = twisted_classical_limit_diagonal_coefficients_ratfun(
        n,
        twist,
        z_order,
        &base_weights,
        &fiber_weights,
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
        CalibrationId("negative-split-lambda-line-birkhoff-qrr-candidate".to_string()),
        CanonicalFrameConvention::RelativeNormalizedCanonicalIdempotents,
    )?;
    r_matrix.check_unitarity(&metric)?;

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

fn twisted_base_polynomial_coefficients_ratfun(
    n: usize,
    q_degree: usize,
    base_weights: &[RatFun],
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
            &QSeries::constant(-weight.clone(), q_degree),
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
    let mut entries = vec![vec![QSeries::zero(q_degree); n + 1]; n + 1];
    for a in 0..=n {
        for b in 0..=n {
            let mut value = RatFun::zero();
            for branch in 0..=n {
                let lambda = base_weights[branch].clone();
                let mut tangent = RatFun::one();
                for (other, weight) in base_weights.iter().enumerate() {
                    if other != branch {
                        tangent = tangent * (lambda.clone() - weight.clone());
                    }
                }
                if tangent.is_zero() {
                    return Err(GwError::NonSemisimplePoint);
                }
                let fiber =
                    twisted_fiber_euler_at_fixed_point_ratfun(twist, fiber_weights, &lambda);
                if fiber.is_zero() {
                    return Err(GwError::NonSemisimplePoint);
                }
                value = value + lambda.pow_usize(a + b) / (tangent * fiber);
            }
            entries[a][b] = QSeries::constant(value, q_degree);
        }
    }
    Ok(SeriesMatrix::from_entries(entries))
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

fn twisted_fiber_euler_at_fixed_point_ratfun(
    twist: &NegativeSplitBundleTwist,
    fiber_weights: &[RatFun],
    lambda: &RatFun,
) -> RatFun {
    twist
        .degrees()
        .iter()
        .zip(fiber_weights)
        .fold(RatFun::one(), |acc, (degree, weight)| {
            acc * (weight.clone() - RatFun::from(*degree) * lambda.clone())
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

fn twisted_classical_h_multiplication_matrix_ratfun(
    n: usize,
    q_degree: usize,
    base_weights: &[RatFun],
) -> Result<SeriesMatrix, GwError> {
    let coefficients = twisted_base_polynomial_coefficients_ratfun(n, q_degree, base_weights)?;
    companion_multiplication_matrix_from_monic_polynomial(n + 1, &coefficients)
}

fn companion_multiplication_matrix_from_monic_polynomial(
    size: usize,
    coefficients: &[QSeries],
) -> Result<SeriesMatrix, GwError> {
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
    let mut entries = vec![vec![QSeries::zero(q_degree); size]; size];
    for col in 0..size.saturating_sub(1) {
        entries[col + 1][col] = QSeries::one(q_degree);
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
    if matrix.rows() != matrix.cols() {
        return Err(GwError::ConventionMismatch(
            "characteristic polynomial requires a square matrix".to_string(),
        ));
    }
    let size = matrix.rows();
    let q_degree = matrix.max_degree();
    let mut polynomial_matrix = vec![vec![vec![QSeries::zero(q_degree)]; size]; size];
    for (row, out_row) in polynomial_matrix.iter_mut().enumerate() {
        for (col, entry) in out_row.iter_mut().enumerate() {
            let mut poly = vec![matrix.entry(row, col).neg()];
            if row == col {
                poly.push(QSeries::one(q_degree));
            }
            *entry = poly;
        }
    }
    let mut charpoly = determinant_qseries_polynomial_matrix(&polynomial_matrix, q_degree);
    charpoly.resize(size + 1, QSeries::zero(q_degree));
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

fn root_series_from_charpoly_ratfun(
    coefficients: &[QSeries],
    branch_root: RatFun,
    max_q_degree: usize,
) -> Result<QSeries, GwError> {
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
    let size = roots.len();
    let q_degree = multiplication.max_degree();
    let identity = SeriesMatrix::identity(size, q_degree);
    let mut columns = vec![vec![QSeries::zero(q_degree); size]; size];

    for branch in 0..size {
        let mut projector = SeriesMatrix::identity(size, q_degree);
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

fn determinant_qseries_polynomial_matrix(
    matrix: &[Vec<Vec<QSeries>>],
    q_degree: usize,
) -> Vec<QSeries> {
    let size = matrix.len();
    if size == 0 {
        return vec![QSeries::one(q_degree)];
    }
    if size == 1 {
        return matrix[0][0].clone();
    }

    let mut total = vec![QSeries::zero(q_degree)];
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

fn qseries_polynomial_add(left: &[QSeries], right: &[QSeries], q_degree: usize) -> Vec<QSeries> {
    let size = left.len().max(right.len());
    let mut out = vec![QSeries::zero(q_degree); size];
    for degree in 0..size {
        let left_coeff = left
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::zero(q_degree));
        let right_coeff = right
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::zero(q_degree));
        out[degree] = left_coeff.add(&right_coeff);
    }
    out
}

fn qseries_polynomial_sub(left: &[QSeries], right: &[QSeries], q_degree: usize) -> Vec<QSeries> {
    let size = left.len().max(right.len());
    let mut out = vec![QSeries::zero(q_degree); size];
    for degree in 0..size {
        let left_coeff = left
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::zero(q_degree));
        let right_coeff = right
            .get(degree)
            .cloned()
            .unwrap_or_else(|| QSeries::zero(q_degree));
        out[degree] = left_coeff.sub(&right_coeff);
    }
    out
}

fn qseries_polynomial_mul(left: &[QSeries], right: &[QSeries], q_degree: usize) -> Vec<QSeries> {
    let mut out = vec![QSeries::zero(q_degree); left.len() + right.len() - 1];
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

fn series_matrix_scale(matrix: &SeriesMatrix, scalar: &QSeries) -> SeriesMatrix {
    SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| row.iter().map(|entry| entry.mul(scalar)).collect())
            .collect(),
    )
}

fn invert_series_matrix(matrix: &SeriesMatrix) -> Result<SeriesMatrix, GwError> {
    if matrix.rows() != matrix.cols() {
        return Err(GwError::ConventionMismatch(
            "matrix inversion requires a square matrix".to_string(),
        ));
    }
    let size = matrix.rows();
    let q_degree = matrix.max_degree();
    let mut augmented = vec![vec![QSeries::zero(q_degree); 2 * size]; size];
    for (row, augmented_row) in augmented.iter_mut().enumerate() {
        for col in 0..size {
            augmented_row[col] = matrix.entry(row, col).clone();
        }
        augmented_row[size + row] = QSeries::one(q_degree);
    }

    for col in 0..size {
        let pivot = (col..size)
            .find(|row| qseries_has_invertible_constant(&augmented[*row][col]))
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

fn qseries_has_invertible_constant(series: &QSeries) -> bool {
    series.coeff(0).is_some_and(|constant| !constant.is_zero())
}

fn derivative_qseries_polynomial_coefficients(
    coefficients: &[QSeries],
    q_degree: usize,
) -> Vec<QSeries> {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coeff)| coeff.scale(&RatFun::from(power)))
        .chain(std::iter::once(QSeries::zero(q_degree)))
        .collect::<Vec<_>>()
}

fn evaluate_qseries_polynomial(coefficients: &[QSeries], x: &QSeries) -> QSeries {
    let q_degree = x.max_degree();
    let mut out = QSeries::zero(q_degree);
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
    let delta0 = delta
        .coeff(0)
        .ok_or_else(|| GwError::AlgebraFailure("empty twisted Delta series".to_string()))?;
    if delta0.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    let inv_delta0 = &RatFun::one() / delta0;
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

fn twisted_classical_limit_diagonal_coefficients_ratfun(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    z_order: usize,
    base_weights: &[RatFun],
    fiber_weights: &[RatFun],
) -> Result<Vec<Vec<RatFun>>, GwError> {
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
            twisted_classical_limit_diagonal_coefficients_for_branch_ratfun(
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

fn twisted_classical_limit_diagonal_coefficients_for_branch_ratfun(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    branch: usize,
    z_order: usize,
    base_weights: &[RatFun],
    fiber_weights: &[RatFun],
) -> Result<Vec<RatFun>, GwError> {
    let mut exponent = vec![RatFun::zero(); z_order + 1];
    for r in 1..=((z_order + 1) / 2) {
        let order = 2 * r - 1;
        let coefficient = RatFun::from_rational(
            bernoulli_number_local(2 * r) / (Rational::from(2 * r) * Rational::from(2 * r - 1)),
        );
        let mut weight_sum = RatFun::zero();
        for other in 0..=n {
            if other == branch {
                continue;
            }
            let difference = base_weights[other].clone() - base_weights[branch].clone();
            if difference.is_zero() {
                return Err(GwError::NonSemisimplePoint);
            }
            weight_sum = weight_sum + RatFun::one() / difference.pow_usize(order);
        }
        for (bundle_degree, fiber_weight) in twist.degrees().iter().zip(fiber_weights) {
            let fiber_root =
                fiber_weight.clone() - RatFun::from(*bundle_degree) * base_weights[branch].clone();
            if fiber_root.is_zero() {
                return Err(GwError::NonSemisimplePoint);
            }
            weight_sum = weight_sum - RatFun::one() / fiber_root.pow_usize(order);
        }
        exponent[order] = coefficient * weight_sum;
    }
    Ok(exp_scalar_z_series_local(&exponent))
}

fn solve_twisted_r_coefficients(
    roots: &[QSeries],
    connection: &SeriesMatrix,
    classical_diagonal: &[Vec<RatFun>],
    q_degree: usize,
    z_order: usize,
) -> Result<Vec<SeriesMatrix>, GwError> {
    let size = roots.len();
    let mut coefficients = Vec::with_capacity(z_order + 1);
    coefficients.push(SeriesMatrix::identity(size, q_degree));

    for order in 1..=z_order {
        let previous = &coefficients[order - 1];
        let recursion_source = previous.q_derivative().add(&connection.mul(previous));
        let mut entries = vec![vec![QSeries::zero(q_degree); size]; size];

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
            entries[branch][branch] = solve_twisted_r_diagonal_from_flatness(
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

fn solve_twisted_r_diagonal_from_flatness(
    connection: &SeriesMatrix,
    entries: &[Vec<QSeries>],
    branch: usize,
    constant: RatFun,
    q_degree: usize,
) -> QSeries {
    let mut known = QSeries::zero(q_degree);
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
        .unwrap_or_else(RatFun::zero);

    let mut coeffs = vec![RatFun::zero(); q_degree + 1];
    coeffs[0] = constant;
    for degree in 1..=q_degree {
        let mut numerator = target.coeff(degree).cloned().unwrap_or_else(RatFun::zero);
        for connection_degree in 1..=degree {
            let term = diagonal_connection
                .coeff(connection_degree)
                .cloned()
                .unwrap_or_else(RatFun::zero)
                * coeffs[degree - connection_degree].clone();
            numerator = numerator - term;
        }
        let denominator = RatFun::from(degree) + a0.clone();
        coeffs[degree] = numerator / denominator;
    }
    QSeries::from_coeffs(coeffs)
}

fn exp_scalar_z_series_local(exponent: &[RatFun]) -> Vec<RatFun> {
    let z_order = exponent.len().saturating_sub(1);
    let mut out = vec![RatFun::zero(); z_order + 1];
    out[0] = RatFun::one();
    for degree in 1..=z_order {
        let mut total = RatFun::zero();
        for part in 1..=degree {
            if exponent[part].is_zero() {
                continue;
            }
            total =
                total + RatFun::from(part) * exponent[part].clone() * out[degree - part].clone();
        }
        out[degree] = total / RatFun::from(degree);
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

fn multiply_polynomial_by_linear_series(
    poly: &[QSeries],
    constant: &QSeries,
    max_q_degree: usize,
) -> Vec<QSeries> {
    multiply_polynomial_by_affine_h_series(
        poly,
        constant,
        &QSeries::one(max_q_degree),
        max_q_degree,
    )
}

fn multiply_polynomial_by_affine_h_series(
    poly: &[QSeries],
    constant: &QSeries,
    h_coeff: &QSeries,
    max_q_degree: usize,
) -> Vec<QSeries> {
    let mut out = vec![QSeries::zero(max_q_degree); poly.len() + 1];
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
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TwistedLineMode {
    EarlyRational,
    SymbolicLimit,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TwistedCalibrationMode {
    InverseEuler,
    InverseEulerFiberPlus,
    Euler,
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
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
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
    if req.equivariant {
        return Err(GwError::UnsupportedInvariant(
            "symbolic equivariant negative split-bundle output is not implemented yet; the current twisted calibration uses a generic rational lambda line"
                .to_string(),
        ));
    }
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider =
        TwistedProjectiveSpaceProvider::new(req.n, req.twist.degrees().to_vec(), req.equivariant)?;
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

    let raw = crate::givental::compute_semisimple_graph_value(
        &provider,
        req.genus,
        req.degree,
        &req.insertions,
        req.truncation.as_ref(),
    )?;
    let value = match raw.as_rational() {
        Some(value) => RatFun::from_rational(value),
        None => RatFun::from_rational(raw.nonequivariant_limit_line(0, &[Rational::one()])?),
    };
    Ok(InvariantResult {
        value,
        engine: "twisted-negative-split-givental-birkhoff-early-line",
        notes: vec![
            "computed by early rational one-parameter lambda-line hypergeometric/Birkhoff S and QRR R stable-graph expansion; no local oracle shortcut is used; fast validation currently covers resolved conifold genus 2 degree 1"
                .to_string(),
        ],
    })
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
        })
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

    fn fiber_weights(&self) -> Vec<Rational> {
        if let Some(weights) = &self.custom_fiber_weights {
            return weights.clone();
        }
        let start = (1usize << (self.base.n + 1)).saturating_sub(1);
        (0..self.twist.rank())
            .map(|idx| Rational::from(start + 2 * idx) * self.fiber_weight_scale.clone())
            .collect()
    }
}

fn twisted_default_base_weights(n: usize) -> Vec<Rational> {
    (0..=n).map(|idx| Rational::from(1usize << idx)).collect()
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
        let fiber_weights = self.fiber_weights();
        let s_matrix = match self.line_mode {
            TwistedLineMode::EarlyRational => {
                NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
                    self.base.n,
                    self.twist.clone(),
                    q_degree,
                    z_order,
                    self.base_weights().to_vec(),
                    fiber_weights,
                )?
                .birkhoff_descendant_s_matrix(z_order)
            }
            TwistedLineMode::SymbolicLimit => NegativeSplitLineHypergeometricModel::new(
                self.base.n,
                self.twist.clone(),
                q_degree,
                z_order,
                self.base_weights(),
                &fiber_weights,
            )?
            .birkhoff_descendant_s_matrix(z_order),
        }?;
        invert_descendant_s_matrix(s_matrix)
    }

    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        static CACHE: OnceLock<
            Mutex<HashMap<TwistedGraphKernelCacheKey, Arc<GiventalGraphKernel>>>,
        > = OnceLock::new();
        let fiber_weights = self.fiber_weights();
        let key = TwistedGraphKernelCacheKey {
            n: self.base.n,
            twist_degrees: self.twist.degrees().to_vec(),
            q_degree,
            r_order,
            graph_dimension,
            base_weights: self.base_weights().to_vec(),
            fiber_weights: fiber_weights.clone(),
            line_mode: self.line_mode.clone(),
            calibration_mode: self.calibration_mode.clone(),
        };
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(kernel) = cache.lock().unwrap().get(&key).cloned() {
            return Ok(kernel);
        }

        let calibration = match self.line_mode {
            TwistedLineMode::EarlyRational => {
                negative_split_twisted_birkhoff_calibration_candidate_with_mode(
                    self.base.n,
                    &self.twist,
                    q_degree,
                    r_order,
                    self.base_weights(),
                    &fiber_weights,
                    self.calibration_mode.clone(),
                )?
            }
            TwistedLineMode::SymbolicLimit => {
                negative_split_twisted_birkhoff_calibration_candidate_ratfun(
                    self.base.n,
                    &self.twist,
                    q_degree,
                    r_order,
                    self.base_weights(),
                    &fiber_weights,
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

    fn scalar_fallback_value(
        &self,
        _genus: usize,
        _degree: usize,
        _insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        Ok(None)
    }
}

fn invert_descendant_s_matrix(s_matrix: SeriesSMatrix) -> Result<SeriesSMatrix, GwError> {
    let size = s_matrix.size();
    let q_degree = s_matrix.q_degree();
    let z_order = s_matrix.z_order();
    let mut inverse = Vec::with_capacity(z_order + 1);
    inverse.push(SeriesMatrix::identity(size, q_degree));
    for order in 1..=z_order {
        let mut total = SeriesMatrix::zero(size, size, q_degree);
        for left in 1..=order {
            total = total.add(
                &s_matrix
                    .coefficient(left)
                    .unwrap()
                    .mul(&inverse[order - left]),
            );
        }
        inverse.push(total.neg());
    }
    SeriesSMatrix::from_coefficients(
        size,
        q_degree,
        z_order,
        inverse,
        CalibrationId(format!("{}-inverse-insertion", s_matrix.calibration().0)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Rational;
    use crate::geometry::CohomologyClass;
    use crate::tau;

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
                identity_ratfun_matrix(3)
            } else {
                zero_ratfun_matrix(3)
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
            &CalibrationId("negative-split-equivariant-hypergeometric-birkhoff".to_string())
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
                    crate::local_oracle::resolved_conifold_gw(genus, degree).unwrap()
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
    fn early_rational_twisted_graph_value_matches_lambda_line_limit() {
        let provider = TwistedProjectiveSpaceProvider::new(1, vec![1, 1], false).unwrap();
        let raw =
            crate::givental::compute_semisimple_graph_value(&provider, 2, 1, &[], None).unwrap();
        let oracle =
            RatFun::from_rational(crate::local_oracle::resolved_conifold_gw(2, 1).unwrap());

        assert_eq!(raw, oracle);
    }

    #[test]
    fn symbolic_raw_twisted_graph_value_needs_lambda_line_limit() {
        let provider =
            TwistedProjectiveSpaceProvider::symbolic_lambda_line(1, vec![1, 1], false).unwrap();
        let raw =
            crate::givental::compute_semisimple_graph_value(&provider, 2, 1, &[], None).unwrap();
        let oracle =
            RatFun::from_rational(crate::local_oracle::resolved_conifold_gw(2, 1).unwrap());
        let limit = RatFun::from_rational(
            raw.nonequivariant_limit_line(0, &[Rational::one()])
                .unwrap(),
        );

        assert_ne!(raw, oracle);
        assert_eq!(limit, oracle);
    }

    #[test]
    fn negative_split_compute_matches_local_p2_degree_one() {
        let req = TwistedInvariantRequest::new(2, vec![3], 2, 1, Vec::new()).unwrap();
        let result = compute_negative_split_twisted(&req).unwrap();
        assert_eq!(
            result.value,
            RatFun::from_rational(crate::local_oracle::local_p2_gw(2, 1).unwrap())
        );
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
