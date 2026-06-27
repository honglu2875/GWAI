//! Factored rational expressions for symbolic coefficient arithmetic.
//!
//! `RatFun` is intentionally simple: every denominator is an expanded sparse
//! polynomial.  That is a good default for small exact computations, but it is
//! the wrong representation for equivariant twisted graph sums where many
//! factors such as `mu_j - a_j lambda_i` are multiplied repeatedly.  This module
//! keeps denominators as factor lists and only expands them when explicitly
//! converting back to `RatFun`.

use crate::algebra::{Coeff, RatFun, Rational, SparsePoly};
use crate::error::GwError;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoredRatFun {
    terms: BTreeMap<Vec<SparsePoly>, SparsePoly>,
}

impl FactoredRatFun {
    pub fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
        }
    }

    pub fn one() -> Self {
        Self::from_rational(Rational::one())
    }

    pub fn from_rational(value: Rational) -> Self {
        Self::from_polynomial(SparsePoly::constant(value))
    }

    pub fn variable(name: impl Into<String>) -> Self {
        Self::from_polynomial(SparsePoly::variable(name))
    }

    pub fn from_polynomial(poly: SparsePoly) -> Self {
        let mut out = Self::zero();
        out.add_term(Vec::new(), poly);
        out
    }

    pub fn from_sparse_fraction(num: SparsePoly, den: SparsePoly) -> Self {
        Self::from_sparse_fraction_factors(num, vec![den])
    }

    pub fn from_ratfun(value: RatFun) -> Self {
        Self::from_sparse_fraction(value.num, value.den)
    }

    pub fn from_sparse_fraction_factors(num: SparsePoly, factors: Vec<SparsePoly>) -> Self {
        if num.is_zero() {
            return Self::zero();
        }
        let mut scalar = Rational::one();
        let mut nonconstant = Vec::new();
        for factor in factors {
            assert!(!factor.is_zero(), "division by zero factored denominator");
            if factor.is_one() {
                continue;
            }
            if let Some(value) = factor.constant_term() {
                scalar = scalar / value;
            } else {
                nonconstant.push(factor);
            }
        }
        let mut out = Self::zero();
        out.add_term(normalize_factors(nonconstant), scale_poly(&num, scalar));
        out
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.terms.len() == 1
            && self
                .terms
                .get(&Vec::<SparsePoly>::new())
                .is_some_and(SparsePoly::is_one)
    }

    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    pub fn total_denominator_factor_count(&self) -> usize {
        self.terms.keys().map(Vec::len).sum()
    }

    pub fn max_denominator_factor_count(&self) -> usize {
        self.terms.keys().map(Vec::len).max().unwrap_or(0)
    }

    pub fn expanded_denominator_term_count_upper_bound(&self) -> usize {
        self.terms
            .keys()
            .map(|factors| {
                factors
                    .iter()
                    .map(|factor| factor.term_count().max(1))
                    .product::<usize>()
            })
            .max()
            .unwrap_or(0)
    }

    pub fn as_rational(&self) -> Option<Rational> {
        let mut total = Rational::zero();
        for (factors, numerator) in &self.terms {
            if !factors.is_empty() {
                return None;
            }
            total += numerator.constant_term()?;
        }
        Some(total)
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut out = Self::one();
        for _ in 0..exp {
            out = &out * self;
        }
        out
    }

    pub fn evaluate_variables(
        &self,
        values: &BTreeMap<String, Rational>,
    ) -> Result<Rational, GwError> {
        let mut total = Rational::zero();
        for (factors, numerator) in &self.terms {
            let mut denominator = Rational::one();
            for factor in factors {
                denominator = denominator * factor.evaluate_variables(values)?;
            }
            if denominator.is_zero() {
                return Err(GwError::AlgebraFailure(
                    "zero denominator after factored variable evaluation".to_string(),
                ));
            }
            total += numerator.evaluate_variables(values)? / denominator;
        }
        Ok(total)
    }

    pub fn to_ratfun(&self) -> RatFun {
        let mut total = RatFun::zero();
        for (factors, numerator) in &self.terms {
            let denominator = multiply_factors(factors);
            total = total
                + RatFun {
                    num: numerator.clone(),
                    den: denominator,
                };
        }
        total
    }

    fn add_term(&mut self, factors: Vec<SparsePoly>, numerator: SparsePoly) {
        if numerator.is_zero() {
            return;
        }
        let factors = normalize_factors(factors);
        let next = self
            .terms
            .get(&factors)
            .map(|existing| existing + &numerator)
            .unwrap_or(numerator);
        if next.is_zero() {
            self.terms.remove(&factors);
        } else {
            self.terms.insert(factors, next);
        }
    }
}

fn normalize_factors(mut factors: Vec<SparsePoly>) -> Vec<SparsePoly> {
    factors.retain(|factor| !factor.is_one());
    factors.sort();
    factors
}

fn scale_poly(poly: &SparsePoly, scalar: Rational) -> SparsePoly {
    if scalar.is_zero() || poly.is_zero() {
        SparsePoly::zero()
    } else {
        poly * &SparsePoly::constant(scalar)
    }
}

fn multiply_factors(factors: &[SparsePoly]) -> SparsePoly {
    let mut out = SparsePoly::one();
    for factor in factors {
        out = &out * factor;
    }
    out
}

impl Coeff for FactoredRatFun {
    fn zero() -> Self {
        Self::zero()
    }

    fn one() -> Self {
        Self::one()
    }

    fn from_rational(value: Rational) -> Self {
        Self::from_rational(value)
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn neg(&self) -> Self {
        -self.clone()
    }

    fn add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    fn sub(&self, rhs: &Self) -> Self {
        self - rhs
    }

    fn mul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn div(&self, rhs: &Self) -> Self {
        self / rhs
    }
}

impl From<Rational> for FactoredRatFun {
    fn from(value: Rational) -> Self {
        Self::from_rational(value)
    }
}

impl From<usize> for FactoredRatFun {
    fn from(value: usize) -> Self {
        Self::from_rational(Rational::from(value))
    }
}

impl From<i32> for FactoredRatFun {
    fn from(value: i32) -> Self {
        Self::from_rational(Rational::from(value))
    }
}

impl From<RatFun> for FactoredRatFun {
    fn from(value: RatFun) -> Self {
        Self::from_ratfun(value)
    }
}

impl<'a, 'b> Add<&'b FactoredRatFun> for &'a FactoredRatFun {
    type Output = FactoredRatFun;

    fn add(self, rhs: &'b FactoredRatFun) -> Self::Output {
        let mut out = self.clone();
        for (factors, numerator) in &rhs.terms {
            out.add_term(factors.clone(), numerator.clone());
        }
        out
    }
}

impl<'a, 'b> Sub<&'b FactoredRatFun> for &'a FactoredRatFun {
    type Output = FactoredRatFun;

    fn sub(self, rhs: &'b FactoredRatFun) -> Self::Output {
        self + &(-rhs.clone())
    }
}

impl<'a, 'b> Mul<&'b FactoredRatFun> for &'a FactoredRatFun {
    type Output = FactoredRatFun;

    fn mul(self, rhs: &'b FactoredRatFun) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return FactoredRatFun::zero();
        }
        let mut out = FactoredRatFun::zero();
        for (left_factors, left_num) in &self.terms {
            for (right_factors, right_num) in &rhs.terms {
                let mut factors = left_factors.clone();
                factors.extend(right_factors.iter().cloned());
                out.add_term(factors, left_num * right_num);
            }
        }
        out
    }
}

impl<'a, 'b> Div<&'b FactoredRatFun> for &'a FactoredRatFun {
    type Output = FactoredRatFun;

    fn div(self, rhs: &'b FactoredRatFun) -> Self::Output {
        assert!(
            !rhs.is_zero(),
            "division by zero factored rational function"
        );
        if rhs.terms.len() == 1 {
            let (rhs_factors, rhs_num) = rhs.terms.iter().next().unwrap();
            let mut out = FactoredRatFun::zero();
            for (left_factors, left_num) in &self.terms {
                let mut numerator = left_num.clone();
                for factor in rhs_factors {
                    numerator = &numerator * factor;
                }
                let mut factors = left_factors.clone();
                factors.push(rhs_num.clone());
                out.add_term(factors, numerator);
            }
            return out;
        }

        let expanded = rhs.to_ratfun();
        self * &FactoredRatFun::from_sparse_fraction(expanded.den, expanded.num)
    }
}

impl Neg for FactoredRatFun {
    type Output = FactoredRatFun;

    fn neg(self) -> Self::Output {
        let terms = self
            .terms
            .into_iter()
            .map(|(factors, numerator)| (factors, -numerator))
            .collect();
        FactoredRatFun { terms }
    }
}

impl Add for FactoredRatFun {
    type Output = FactoredRatFun;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Sub for FactoredRatFun {
    type Output = FactoredRatFun;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Mul for FactoredRatFun {
    type Output = FactoredRatFun;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Div for FactoredRatFun {
    type Output = FactoredRatFun;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl fmt::Display for FactoredRatFun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (factors, numerator) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            if factors.is_empty() {
                write!(f, "{numerator}")?;
            } else {
                write!(f, "({numerator})/(")?;
                for (idx, factor) in factors.iter().enumerate() {
                    if idx > 0 {
                        write!(f, " * ")?;
                    }
                    write!(f, "{factor}")?;
                }
                write!(f, ")")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mu_shift(shift: i32) -> SparsePoly {
        &SparsePoly::variable("mu_0") + &SparsePoly::constant(Rational::from(shift))
    }

    #[test]
    fn multiplication_keeps_repeated_denominator_factors_unexpanded() {
        let factor = mu_shift(-3);
        let term = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), factor);
        let product = term.pow_usize(8);

        assert_eq!(product.term_count(), 1);
        assert_eq!(product.max_denominator_factor_count(), 8);
        assert_eq!(product.expanded_denominator_term_count_upper_bound(), 256);
        let rendered = product.to_string();
        assert!(rendered.contains("mu_0"));
    }

    #[test]
    fn evaluation_matches_expanded_ratfun() {
        let left =
            FactoredRatFun::from_sparse_fraction(SparsePoly::one(), mu_shift(-3)).pow_usize(2);
        let right = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), mu_shift(-5));
        let expr = &left + &right;
        let mut values = BTreeMap::new();
        values.insert("mu_0".to_string(), Rational::from(7usize));

        assert_eq!(
            expr.evaluate_variables(&values).unwrap(),
            expr.to_ratfun().evaluate_variables(&values).unwrap()
        );
    }

    #[test]
    fn expanded_ratfun_round_trips_through_factored_bridge() {
        let mu = RatFun::variable("mu_0");
        let expanded = &(&mu + &RatFun::from(1usize)) / &(&mu - &RatFun::from(3usize));
        let factored = FactoredRatFun::from_ratfun(expanded.clone());
        let mut values = BTreeMap::new();
        values.insert("mu_0".to_string(), Rational::from(7usize));

        assert_eq!(
            factored.evaluate_variables(&values).unwrap(),
            expanded.evaluate_variables(&values).unwrap()
        );
        assert_eq!(factored.to_ratfun(), expanded);
    }

    #[test]
    fn qseries_can_use_factored_coefficients() {
        let factor = mu_shift(-3);
        let coeff = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), factor);
        let series = crate::series::QSeries::constant(coeff.clone(), 2)
            .mul(&crate::series::QSeries::constant(coeff, 2));
        assert_eq!(series.coeff(0).unwrap().max_denominator_factor_count(), 2);
    }

    #[test]
    fn semisimple_calibration_can_use_factored_coefficients() {
        let q_degree = 1;
        let z_order = 1;
        let size = 1;
        let matrix = crate::series::SeriesMatrix::<FactoredRatFun>::identity(size, q_degree);
        let scalar = crate::series::QSeries::<FactoredRatFun>::one(q_degree);
        let calibration = crate::givental::SemisimpleCalibration {
            r_matrix: crate::givental::SeriesRMatrix::identity(
                size,
                q_degree,
                z_order,
                crate::givental::CanonicalFrameConvention::NormalizedCanonicalIdempotents,
            ),
            metric: matrix.clone(),
            psi: matrix.clone(),
            psi_inverse: matrix.clone(),
            connection: matrix,
            delta: vec![scalar.clone()],
            inverse_delta: vec![scalar.clone()],
            relative_sqrt_delta: vec![scalar.clone()],
            relative_sqrt_delta_inverse: vec![scalar],
        };

        assert_eq!(calibration.r_matrix.size(), size);
        assert_eq!(calibration.r_matrix.q_degree(), q_degree);
    }
}
