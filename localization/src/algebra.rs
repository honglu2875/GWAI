use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::error::GwError;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rational(BigRational);

impl Rational {
    pub fn zero() -> Self {
        Rational(BigRational::zero())
    }

    pub fn one() -> Self {
        Rational(BigRational::one())
    }

    pub fn new(num: i128, den: i128) -> Self {
        assert!(den != 0, "zero denominator");
        Rational(BigRational::new(BigInt::from(num), BigInt::from(den)))
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    pub fn is_negative(&self) -> bool {
        self.0.is_negative()
    }

    pub fn abs(&self) -> Self {
        Rational(self.0.abs())
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut out = Rational::one();
        for _ in 0..exp {
            out = out * self.clone();
        }
        out
    }
}

impl From<i128> for Rational {
    fn from(value: i128) -> Self {
        Rational::new(value, 1)
    }
}

impl From<i64> for Rational {
    fn from(value: i64) -> Self {
        Rational::new(value as i128, 1)
    }
}

impl From<i32> for Rational {
    fn from(value: i32) -> Self {
        Rational::new(value as i128, 1)
    }
}

impl From<usize> for Rational {
    fn from(value: usize) -> Self {
        Rational::new(value as i128, 1)
    }
}

impl From<isize> for Rational {
    fn from(value: isize) -> Self {
        Rational::new(value as i128, 1)
    }
}

impl Add for Rational {
    type Output = Rational;

    fn add(self, rhs: Self) -> Self::Output {
        Rational(self.0 + rhs.0)
    }
}

impl AddAssign for Rational {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl Sub for Rational {
    type Output = Rational;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul for Rational {
    type Output = Rational;

    fn mul(self, rhs: Self) -> Self::Output {
        Rational(self.0 * rhs.0)
    }
}

impl Div for Rational {
    type Output = Rational;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(!rhs.is_zero(), "division by zero");
        Rational(self.0 / rhs.0)
    }
}

impl Neg for Rational {
    type Output = Rational;

    fn neg(self) -> Self::Output {
        Rational(-self.0)
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.denom().is_one() {
            write!(f, "{}", self.0.numer())
        } else {
            write!(f, "{}/{}", self.0.numer(), self.0.denom())
        }
    }
}

/// Arithmetic interface used by performance-sensitive series containers.
///
/// The existing symbolic engine uses [`RatFun`], but many non-equivariant and
/// early-specialized computations only need exact rational coefficients.  This
/// trait lets those two cases share `q`-series and matrix code while keeping the
/// coefficient representation swappable.
pub trait Coeff: Clone + PartialEq + Eq + fmt::Debug {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_rational(value: Rational) -> Self;
    fn is_zero(&self) -> bool;
    fn neg(&self) -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;

    fn from_usize(value: usize) -> Self {
        Self::from_rational(Rational::from(value))
    }

    fn pow_usize(&self, exp: usize) -> Self {
        let mut out = Self::one();
        for _ in 0..exp {
            out = out.mul(self);
        }
        out
    }
}

impl Coeff for Rational {
    fn zero() -> Self {
        Rational::zero()
    }

    fn one() -> Self {
        Rational::one()
    }

    fn from_rational(value: Rational) -> Self {
        value
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn neg(&self) -> Self {
        -self.clone()
    }

    fn add(&self, rhs: &Self) -> Self {
        self.clone() + rhs.clone()
    }

    fn sub(&self, rhs: &Self) -> Self {
        self.clone() - rhs.clone()
    }

    fn mul(&self, rhs: &Self) -> Self {
        self.clone() * rhs.clone()
    }

    fn div(&self, rhs: &Self) -> Self {
        self.clone() / rhs.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Monomial(Vec<(String, i32)>);

impl Monomial {
    pub fn one() -> Self {
        Self(Vec::new())
    }

    pub fn variable(name: impl Into<String>) -> Self {
        Self(vec![(name.into(), 1)])
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let mut exps = BTreeMap::<String, i32>::new();
        for (name, exp) in self.0.iter().chain(rhs.0.iter()) {
            *exps.entry(name.clone()).or_default() += *exp;
        }
        Self(
            exps.into_iter()
                .filter(|(_, exp)| *exp != 0)
                .collect::<Vec<_>>(),
        )
    }

    fn lambda_line_weighted_degree(&self, weights: &[Rational]) -> Option<(i32, Rational)> {
        let mut degree = 0i32;
        let mut coefficient = Rational::one();
        for (name, exp) in &self.0 {
            if let Some(idx) = lambda_index(name) {
                if idx >= weights.len() {
                    return None;
                }
                degree += *exp;
                if *exp >= 0 {
                    coefficient = coefficient * weights[idx].pow_usize(*exp as usize);
                } else {
                    let reciprocal = Rational::one() / weights[idx].pow_usize((-*exp) as usize);
                    coefficient = coefficient * reciprocal;
                }
            } else {
                return None;
            }
        }
        Some((degree, coefficient))
    }

    fn evaluate_lambda_weights(&self, weights: &[Rational]) -> Option<Rational> {
        let mut coefficient = Rational::one();
        for (name, exp) in &self.0 {
            let idx = lambda_index(name)?;
            if idx >= weights.len() {
                return None;
            }
            if *exp >= 0 {
                coefficient = coefficient * weights[idx].pow_usize(*exp as usize);
            } else {
                coefficient = coefficient / weights[idx].pow_usize((-*exp) as usize);
            }
        }
        Some(coefficient)
    }
}

impl Ord for Monomial {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Monomial {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "1");
        }
        for (idx, (name, exp)) in self.0.iter().enumerate() {
            if idx > 0 {
                write!(f, "*")?;
            }
            if *exp == 1 {
                write!(f, "{name}")?;
            } else {
                write!(f, "{name}^{exp}")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsePoly {
    terms: BTreeMap<Monomial, Rational>,
}

impl SparsePoly {
    pub fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
        }
    }

    pub fn one() -> Self {
        Self::constant(Rational::one())
    }

    pub fn constant(c: Rational) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            let mut terms = BTreeMap::new();
            terms.insert(Monomial::one(), c);
            Self { terms }
        }
    }

    pub fn variable(name: impl Into<String>) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(Monomial::variable(name), Rational::one());
        Self { terms }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.terms.len() == 1 && self.terms.get(&Monomial::one()) == Some(&Rational::one())
    }

    pub fn constant_term(&self) -> Option<Rational> {
        if self.terms.is_empty() {
            Some(Rational::zero())
        } else if self.terms.len() == 1 {
            self.terms.get(&Monomial::one()).cloned()
        } else {
            None
        }
    }

    fn add_term(&mut self, monomial: Monomial, coeff: Rational) {
        if coeff.is_zero() {
            return;
        }
        let new_coeff = self
            .terms
            .get(&monomial)
            .cloned()
            .unwrap_or_else(Rational::zero)
            + coeff;
        if new_coeff.is_zero() {
            self.terms.remove(&monomial);
        } else {
            self.terms.insert(monomial, new_coeff);
        }
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut out = SparsePoly::one();
        for _ in 0..exp {
            out = &out * self;
        }
        out
    }

    fn substitute_lambda_line(
        &self,
        weights: &[Rational],
    ) -> Result<BTreeMap<i32, Rational>, GwError> {
        let mut out = BTreeMap::<i32, Rational>::new();
        for (monomial, coeff) in &self.terms {
            let Some((degree, monomial_coeff)) = monomial.lambda_line_weighted_degree(weights)
            else {
                return Err(GwError::AlgebraFailure(format!(
                    "lambda-line limit only supports lambda_i variables, found `{monomial}`"
                )));
            };
            let term = coeff.clone() * monomial_coeff;
            let next = out.get(&degree).cloned().unwrap_or_else(Rational::zero) + term;
            if next.is_zero() {
                out.remove(&degree);
            } else {
                out.insert(degree, next);
            }
        }
        Ok(out)
    }

    fn evaluate_lambda_weights(&self, weights: &[Rational]) -> Result<Rational, GwError> {
        let mut total = Rational::zero();
        for (monomial, coeff) in &self.terms {
            let Some(monomial_value) = monomial.evaluate_lambda_weights(weights) else {
                return Err(GwError::AlgebraFailure(format!(
                    "lambda evaluation only supports lambda_i variables, found `{monomial}`"
                )));
            };
            total += coeff.clone() * monomial_value;
        }
        Ok(total)
    }
}

impl<'a, 'b> Add<&'b SparsePoly> for &'a SparsePoly {
    type Output = SparsePoly;

    fn add(self, rhs: &'b SparsePoly) -> Self::Output {
        let mut out = self.clone();
        for (monomial, coeff) in &rhs.terms {
            out.add_term(monomial.clone(), coeff.clone());
        }
        out
    }
}

impl<'a, 'b> Sub<&'b SparsePoly> for &'a SparsePoly {
    type Output = SparsePoly;

    fn sub(self, rhs: &'b SparsePoly) -> Self::Output {
        let mut out = self.clone();
        for (monomial, coeff) in &rhs.terms {
            out.add_term(monomial.clone(), -coeff.clone());
        }
        out
    }
}

impl<'a, 'b> Mul<&'b SparsePoly> for &'a SparsePoly {
    type Output = SparsePoly;

    fn mul(self, rhs: &'b SparsePoly) -> Self::Output {
        let mut out = SparsePoly::zero();
        for (m1, c1) in &self.terms {
            for (m2, c2) in &rhs.terms {
                out.add_term(m1.mul(m2), c1.clone() * c2.clone());
            }
        }
        out
    }
}

impl Neg for SparsePoly {
    type Output = SparsePoly;

    fn neg(self) -> Self::Output {
        let mut out = SparsePoly::zero();
        for (monomial, coeff) in self.terms {
            out.add_term(monomial, -coeff);
        }
        out
    }
}

impl fmt::Display for SparsePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (monomial, coeff) in &self.terms {
            if !first {
                if coeff.is_negative() {
                    write!(f, " - ")?;
                } else {
                    write!(f, " + ")?;
                }
            } else if coeff.is_negative() {
                write!(f, "-")?;
            }
            first = false;

            let abs_coeff = coeff.abs();
            let monomial_is_one = monomial.0.is_empty();
            if monomial_is_one {
                write!(f, "{abs_coeff}")?;
            } else if abs_coeff == Rational::one() {
                write!(f, "{monomial}")?;
            } else {
                write!(f, "{abs_coeff}*{monomial}")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RatFun {
    pub num: SparsePoly,
    pub den: SparsePoly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaurentSeries {
    coeffs: BTreeMap<i32, Rational>,
}

impl LaurentSeries {
    pub fn zero() -> Self {
        Self {
            coeffs: BTreeMap::new(),
        }
    }

    pub fn constant(value: Rational) -> Self {
        if value.is_zero() {
            Self::zero()
        } else {
            let mut coeffs = BTreeMap::new();
            coeffs.insert(0, value);
            Self { coeffs }
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = self.clone();
        for (order, coeff) in &rhs.coeffs {
            let next = out
                .coeffs
                .get(order)
                .cloned()
                .unwrap_or_else(Rational::zero)
                + coeff.clone();
            if next.is_zero() {
                out.coeffs.remove(order);
            } else {
                out.coeffs.insert(*order, next);
            }
        }
        out
    }

    pub fn coefficient(&self, order: i32) -> Rational {
        self.coeffs
            .get(&order)
            .cloned()
            .unwrap_or_else(Rational::zero)
    }

    pub fn has_negative_terms(&self) -> bool {
        self.coeffs
            .iter()
            .any(|(order, coeff)| *order < 0 && !coeff.is_zero())
    }

    pub fn finite_limit(&self) -> Result<Rational, GwError> {
        if self.has_negative_terms() {
            return Err(GwError::NonFiniteLimit(
                "negative Laurent terms remain after summation".to_string(),
            ));
        }
        Ok(self.coefficient(0))
    }

    pub fn coeffs(&self) -> &BTreeMap<i32, Rational> {
        &self.coeffs
    }
}

impl RatFun {
    pub fn zero() -> Self {
        Self::from_rational(Rational::zero())
    }

    pub fn one() -> Self {
        Self::from_rational(Rational::one())
    }

    pub fn from_rational(c: Rational) -> Self {
        Self {
            num: SparsePoly::constant(c),
            den: SparsePoly::one(),
        }
    }

    pub fn variable(name: impl Into<String>) -> Self {
        Self {
            num: SparsePoly::variable(name),
            den: SparsePoly::one(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    pub fn is_one(&self) -> bool {
        self.num.is_one() && self.den.is_one()
    }

    pub fn as_rational(&self) -> Option<Rational> {
        let num = self.num.constant_term()?;
        let den = self.den.constant_term()?;
        if den.is_zero() {
            None
        } else {
            Some(num / den)
        }
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut out = RatFun::one();
        for _ in 0..exp {
            out = &out * self;
        }
        out
    }

    pub fn normalize_light(mut self) -> Self {
        if self.num.is_zero() {
            self.den = SparsePoly::one();
            return self;
        }
        if self.num == self.den {
            return RatFun::one();
        }
        if let (Some(n), Some(d)) = (self.num.constant_term(), self.den.constant_term()) {
            return RatFun::from_rational(n / d);
        }
        self
    }

    pub fn nonequivariant_limit_line(
        &self,
        target_n: usize,
        weights: &[Rational],
    ) -> Result<Rational, GwError> {
        self.lambda_line_laurent_series(target_n, weights, 0)?
            .finite_limit()
    }

    pub fn lambda_line_laurent_series(
        &self,
        target_n: usize,
        weights: &[Rational],
        max_order: i32,
    ) -> Result<LaurentSeries, GwError> {
        if weights.len() != target_n + 1 {
            return Err(GwError::AlgebraFailure(format!(
                "expected {} lambda-line weights, got {}",
                target_n + 1,
                weights.len()
            )));
        }
        let num = self.num.substitute_lambda_line(weights)?;
        let den = self.den.substitute_lambda_line(weights)?;
        ratio_laurent_series(&num, &den, max_order)
    }

    pub fn evaluate_lambda_weights(
        &self,
        target_n: usize,
        weights: &[Rational],
    ) -> Result<Rational, GwError> {
        if weights.len() != target_n + 1 {
            return Err(GwError::AlgebraFailure(format!(
                "expected {} lambda weights, got {}",
                target_n + 1,
                weights.len()
            )));
        }
        let num = self.num.evaluate_lambda_weights(weights)?;
        let den = self.den.evaluate_lambda_weights(weights)?;
        if den.is_zero() {
            return Err(GwError::AlgebraFailure(
                "zero denominator after lambda evaluation".to_string(),
            ));
        }
        Ok(num / den)
    }
}

impl Coeff for RatFun {
    fn zero() -> Self {
        RatFun::zero()
    }

    fn one() -> Self {
        RatFun::one()
    }

    fn from_rational(value: Rational) -> Self {
        RatFun::from_rational(value)
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

fn leading_term(poly: &BTreeMap<i32, Rational>) -> Option<(i32, Rational)> {
    poly.iter()
        .find(|(_, coeff)| !coeff.is_zero())
        .map(|(degree, coeff)| (*degree, coeff.clone()))
}

fn ratio_laurent_series(
    num: &BTreeMap<i32, Rational>,
    den: &BTreeMap<i32, Rational>,
    max_order: i32,
) -> Result<LaurentSeries, GwError> {
    let Some((num_order, _)) = leading_term(num) else {
        return Ok(LaurentSeries::zero());
    };
    let Some((den_order, den_lead)) = leading_term(den) else {
        return Err(GwError::AlgebraFailure(
            "zero denominator after lambda-line substitution".to_string(),
        ));
    };
    let min_order = num_order - den_order;
    if min_order > max_order {
        return Ok(LaurentSeries::zero());
    }
    let inner_max = (max_order - min_order) as usize;

    let mut den_unit = vec![Rational::zero(); inner_max + 1];
    den_unit[0] = Rational::one();
    for (order, coeff) in den {
        let shifted = *order - den_order;
        if shifted > 0 && shifted as usize <= inner_max {
            den_unit[shifted as usize] = coeff.clone() / den_lead.clone();
        }
    }

    let mut inv = vec![Rational::zero(); inner_max + 1];
    inv[0] = Rational::one();
    for degree in 1..=inner_max {
        let mut sum = Rational::zero();
        for k in 1..=degree {
            sum += den_unit[k].clone() * inv[degree - k].clone();
        }
        inv[degree] = -sum;
    }

    let mut num_shift = vec![Rational::zero(); inner_max + 1];
    for (order, coeff) in num {
        let shifted = *order - num_order;
        if shifted >= 0 && shifted as usize <= inner_max {
            num_shift[shifted as usize] = coeff.clone();
        }
    }

    let mut coeffs = BTreeMap::new();
    for degree in 0..=inner_max {
        let mut coeff = Rational::zero();
        for k in 0..=degree {
            coeff += num_shift[k].clone() * inv[degree - k].clone();
        }
        coeff = coeff / den_lead.clone();
        if !coeff.is_zero() {
            coeffs.insert(min_order + degree as i32, coeff);
        }
    }

    Ok(LaurentSeries { coeffs })
}

fn lambda_index(name: &str) -> Option<usize> {
    name.strip_prefix("lambda_")?.parse().ok()
}

impl From<Rational> for RatFun {
    fn from(value: Rational) -> Self {
        Self::from_rational(value)
    }
}

impl From<i32> for RatFun {
    fn from(value: i32) -> Self {
        Self::from_rational(Rational::from(value))
    }
}

impl From<i128> for RatFun {
    fn from(value: i128) -> Self {
        Self::from_rational(Rational::from(value))
    }
}

impl From<usize> for RatFun {
    fn from(value: usize) -> Self {
        Self::from_rational(Rational::from(value))
    }
}

impl<'a, 'b> Add<&'b RatFun> for &'a RatFun {
    type Output = RatFun;

    fn add(self, rhs: &'b RatFun) -> Self::Output {
        if let (Some(left), Some(right)) = (self.as_rational(), rhs.as_rational()) {
            return RatFun::from_rational(left + right);
        }
        let left = &self.num * &rhs.den;
        let right = &rhs.num * &self.den;
        RatFun {
            num: &left + &right,
            den: &self.den * &rhs.den,
        }
        .normalize_light()
    }
}

impl<'a, 'b> Sub<&'b RatFun> for &'a RatFun {
    type Output = RatFun;

    fn sub(self, rhs: &'b RatFun) -> Self::Output {
        if let (Some(left), Some(right)) = (self.as_rational(), rhs.as_rational()) {
            return RatFun::from_rational(left - right);
        }
        let left = &self.num * &rhs.den;
        let right = &rhs.num * &self.den;
        RatFun {
            num: &left - &right,
            den: &self.den * &rhs.den,
        }
        .normalize_light()
    }
}

impl<'a, 'b> Mul<&'b RatFun> for &'a RatFun {
    type Output = RatFun;

    fn mul(self, rhs: &'b RatFun) -> Self::Output {
        if let (Some(left), Some(right)) = (self.as_rational(), rhs.as_rational()) {
            return RatFun::from_rational(left * right);
        }
        RatFun {
            num: &self.num * &rhs.num,
            den: &self.den * &rhs.den,
        }
        .normalize_light()
    }
}

impl<'a, 'b> Div<&'b RatFun> for &'a RatFun {
    type Output = RatFun;

    fn div(self, rhs: &'b RatFun) -> Self::Output {
        assert!(!rhs.num.is_zero(), "division by zero rational function");
        if let (Some(left), Some(right)) = (self.as_rational(), rhs.as_rational()) {
            return RatFun::from_rational(left / right);
        }
        RatFun {
            num: &self.num * &rhs.den,
            den: &self.den * &rhs.num,
        }
        .normalize_light()
    }
}

impl Neg for RatFun {
    type Output = RatFun;

    fn neg(self) -> Self::Output {
        RatFun {
            num: -self.num,
            den: self.den,
        }
    }
}

impl Add for RatFun {
    type Output = RatFun;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Sub for RatFun {
    type Output = RatFun;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Mul for RatFun {
    type Output = RatFun;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Div for RatFun {
    type Output = RatFun;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl fmt::Display for RatFun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den.is_one() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "({})/({})", self.num, self.den)
        }
    }
}

pub fn lambda(i: usize) -> RatFun {
    RatFun::variable(format!("lambda_{i}"))
}

pub fn q() -> RatFun {
    RatFun::variable("q")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_reduces_signs() {
        assert_eq!(Rational::new(2, -4).to_string(), "-1/2");
        assert_eq!(Rational::new(-6, -8).to_string(), "3/4");
    }

    #[test]
    fn rational_function_constant_arithmetic() {
        let a = RatFun::from_rational(Rational::new(1, 2));
        let b = RatFun::from_rational(Rational::new(1, 3));
        assert_eq!((&a + &b).to_string(), "5/6");
        assert_eq!((&a * &b).to_string(), "1/6");
    }

    #[test]
    fn sparse_polynomial_collects_terms() {
        let x = RatFun::variable("x");
        let expr = &(&x + &RatFun::one()) * &(&x - &RatFun::one());
        assert_eq!(expr.to_string(), "-1 + x^2");
    }

    #[test]
    fn nonequivariant_line_limit_extracts_constant_term() {
        let l0 = lambda(0);
        let l1 = lambda(1);
        let expr = &(&l0 * &l0) / &(&l0 * &l1);
        let limit = expr
            .nonequivariant_limit_line(1, &[Rational::from(2), Rational::from(3)])
            .unwrap();
        assert_eq!(limit, Rational::new(2, 3));
    }

    #[test]
    fn nonequivariant_line_limit_detects_poles() {
        let expr = &RatFun::one() / &lambda(0);
        assert!(matches!(
            expr.nonequivariant_limit_line(0, &[Rational::one()]),
            Err(GwError::NonFiniteLimit(_))
        ));
    }

    #[test]
    fn laurent_series_keeps_negative_terms_for_later_cancellation() {
        let expr = &(&lambda(0) + &lambda(0).pow_usize(2)) / &lambda(0);
        let series = expr
            .lambda_line_laurent_series(0, &[Rational::from(3)], 0)
            .unwrap();
        assert_eq!(series.coefficient(0), Rational::one());
        assert_eq!(series.coeffs().len(), 1);

        let pole = (&RatFun::one() / &lambda(0))
            .lambda_line_laurent_series(0, &[Rational::from(2)], 0)
            .unwrap();
        assert_eq!(pole.coefficient(-1), Rational::new(1, 2));
    }

    #[test]
    fn lambda_weight_evaluation_is_exact() {
        let expr = &(&lambda(0) * &lambda(0)) / &(&lambda(1) - &lambda(0));
        let value = expr
            .evaluate_lambda_weights(1, &[Rational::from(2), Rational::from(5)])
            .unwrap();
        assert_eq!(value, Rational::new(4, 3));
    }
}
