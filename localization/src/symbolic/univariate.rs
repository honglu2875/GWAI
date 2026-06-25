use std::fmt;

use crate::algebra::{RatFun, Rational};
use crate::error::GwError;

/// Sparse enough for graph formulas, simple enough to audit.
///
/// Coefficients are stored in ascending powers of the generator `x`, and each
/// coefficient is a [`RatFun`] in the external parameters (`q`, lambdas,
/// insertion variables, and later descendants).  This is not meant to be a
/// general-purpose CAS polynomial type; it is the one-variable layer needed for
/// root and quotient reductions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniPoly {
    coeffs: Vec<RatFun>,
}

impl UniPoly {
    pub fn new(mut coeffs: Vec<RatFun>) -> Self {
        trim_coeffs(&mut coeffs);
        Self { coeffs }
    }

    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    pub fn one() -> Self {
        Self::constant(RatFun::one())
    }

    pub fn constant(value: RatFun) -> Self {
        if value.is_zero() {
            Self::zero()
        } else {
            Self {
                coeffs: vec![value],
            }
        }
    }

    pub fn monomial(degree: usize, coeff: RatFun) -> Self {
        if coeff.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![RatFun::zero(); degree + 1];
        coeffs[degree] = coeff;
        Self { coeffs }
    }

    pub fn variable() -> Self {
        Self::monomial(1, RatFun::one())
    }

    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }

    pub fn coeff(&self, degree: usize) -> RatFun {
        self.coeffs
            .get(degree)
            .cloned()
            .unwrap_or_else(RatFun::zero)
    }

    pub fn coeffs(&self) -> &[RatFun] {
        &self.coeffs
    }

    pub fn leading_coeff(&self) -> Option<RatFun> {
        self.coeffs.last().cloned()
    }

    pub fn as_constant(&self) -> Option<RatFun> {
        match self.degree() {
            None => Some(RatFun::zero()),
            Some(0) => Some(self.coeffs[0].clone()),
            Some(_) => None,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_one()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for degree in 0..len {
            let left = self.coeff(degree);
            let right = rhs.coeff(degree);
            coeffs.push(&left + &right);
        }
        Self::new(coeffs)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for degree in 0..len {
            let left = self.coeff(degree);
            let right = rhs.coeff(degree);
            coeffs.push(&left - &right);
        }
        Self::new(coeffs)
    }

    pub fn neg(&self) -> Self {
        Self::new(
            self.coeffs
                .iter()
                .cloned()
                .map(std::ops::Neg::neg)
                .collect(),
        )
    }

    pub fn scale(&self, scalar: &RatFun) -> Self {
        if scalar.is_zero() || self.is_zero() {
            return Self::zero();
        }
        Self::new(self.coeffs.iter().map(|coeff| coeff * scalar).collect())
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![RatFun::zero(); self.coeffs.len() + rhs.coeffs.len() - 1];
        for (left_degree, left) in self.coeffs.iter().enumerate() {
            for (right_degree, right) in rhs.coeffs.iter().enumerate() {
                let product = left * right;
                let next = &coeffs[left_degree + right_degree] + &product;
                coeffs[left_degree + right_degree] = next;
            }
        }
        Self::new(coeffs)
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut out = Self::one();
        for _ in 0..exp {
            out = out.mul(self);
        }
        out
    }

    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let mut coeffs = Vec::with_capacity(self.coeffs.len() - 1);
        for degree in 1..self.coeffs.len() {
            let factor = RatFun::from_rational(Rational::from(degree));
            coeffs.push(&self.coeffs[degree] * &factor);
        }
        Self::new(coeffs)
    }

    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), GwError> {
        let Some(divisor_degree) = divisor.degree() else {
            return Err(GwError::AlgebraFailure(
                "division by zero univariate polynomial".to_string(),
            ));
        };
        let divisor_lead = divisor.leading_coeff().expect("degree checked above");
        if divisor_lead.is_zero() {
            return Err(GwError::AlgebraFailure(
                "division by zero leading coefficient".to_string(),
            ));
        }
        if self.degree().is_none_or(|degree| degree < divisor_degree) {
            return Ok((Self::zero(), self.clone()));
        }

        let mut rem = self.coeffs.clone();
        let quotient_len = self.degree().unwrap() - divisor_degree + 1;
        let mut quotient = vec![RatFun::zero(); quotient_len];

        while rem.len() >= divisor.coeffs.len() && !rem.is_empty() {
            let rem_degree = rem.len() - 1;
            let shift = rem_degree - divisor_degree;
            let factor = &rem[rem_degree] / &divisor_lead;
            quotient[shift] = &quotient[shift] + &factor;
            for offset in 0..=divisor_degree {
                let subtraction = &factor * &divisor.coeffs[offset];
                rem[shift + offset] = &rem[shift + offset] - &subtraction;
            }
            trim_coeffs(&mut rem);
        }

        Ok((Self::new(quotient), Self::new(rem)))
    }
}

fn trim_coeffs(coeffs: &mut Vec<RatFun>) {
    while coeffs.last().is_some_and(RatFun::is_zero) {
        coeffs.pop();
    }
}

impl fmt::Display for UniPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (degree, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            match degree {
                0 => write!(f, "{coeff}")?,
                1 if coeff.is_one() => write!(f, "x")?,
                1 => write!(f, "({coeff})*x")?,
                _ if coeff.is_one() => write!(f, "x^{degree}")?,
                _ => write!(f, "({coeff})*x^{degree}")?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::q;

    #[test]
    fn univariate_arithmetic_collects_terms() {
        let x = UniPoly::variable();
        let one = UniPoly::one();
        let product = x.add(&one).mul(&x.sub(&one));
        assert_eq!(product, x.pow_usize(2).sub(&one));
    }

    #[test]
    fn polynomial_division_returns_quotient_and_remainder() {
        let x = UniPoly::variable();
        let divisor = x.sub(&UniPoly::one());
        let dividend = x.pow_usize(3).sub(&UniPoly::one());
        let (quotient, remainder) = dividend.div_rem(&divisor).unwrap();
        assert!(remainder.is_zero());
        assert_eq!(quotient, x.pow_usize(2).add(&x).add(&UniPoly::one()));
    }

    #[test]
    fn derivative_uses_ratfun_coefficients() {
        let x = UniPoly::variable();
        let poly = x.pow_usize(3).scale(&q());
        let expected = x
            .pow_usize(2)
            .scale(&(&RatFun::from_rational(Rational::from(3usize)) * &q()));
        assert_eq!(poly.derivative(), expected);
    }
}
