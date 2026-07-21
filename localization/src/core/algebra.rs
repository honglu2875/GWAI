//! Exact coefficient rings and sparse symbolic rational-function algebra.

use std::cmp::Ordering;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::sync::{OnceLock, RwLock};

use super::error::GwError;
#[cfg(not(feature = "gmp-rational"))]
use num_bigint::BigInt;
#[cfg(not(feature = "gmp-rational"))]
use num_rational::BigRational;
#[cfg(not(feature = "gmp-rational"))]
use num_traits::{One, Signed, Zero};
#[cfg(feature = "gmp-rational")]
use rug::Rational as BackendRational;

#[cfg(not(feature = "gmp-rational"))]
type BackendRational = BigRational;

/// Global variable-name interner.
///
/// Symbolic variables register once and are identified by dense `u32` ids
/// afterwards, so polynomial arithmetic never allocates, clones, or compares
/// name strings.  The table is append-only; display and name-based evaluation
/// resolve back through it, and `lambda_i` indices are parsed once at intern
/// time.  Multiplication and comparison of monomials never touch the table.
#[derive(Default)]
struct SymbolTable {
    /// `id -> (name, lambda index if the name is lambda_i)`.
    symbols: Vec<(String, Option<usize>)>,
    ids: HashMap<String, u32>,
}

fn symbol_table() -> &'static RwLock<SymbolTable> {
    static TABLE: OnceLock<RwLock<SymbolTable>> = OnceLock::new();
    TABLE.get_or_init(RwLock::default)
}

fn intern_symbol(name: &str) -> u32 {
    if let Some(&id) = symbol_table().read().unwrap().ids.get(name) {
        return id;
    }
    let mut table = symbol_table().write().unwrap();
    if let Some(&id) = table.ids.get(name) {
        return id;
    }
    let id = u32::try_from(table.symbols.len()).expect("symbol table overflow");
    let lambda = name
        .strip_prefix("lambda_")
        .and_then(|raw| raw.parse().ok());
    table.symbols.push((name.to_string(), lambda));
    table.ids.insert(name.to_string(), id);
    id
}

fn symbol_name(id: u32) -> String {
    symbol_table().read().unwrap().symbols[id as usize]
        .0
        .clone()
}

fn symbol_lambda_index(id: u32) -> Option<usize> {
    symbol_table().read().unwrap().symbols[id as usize].1
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rational(BackendRational);

impl Rational {
    pub fn zero() -> Self {
        rational_zero()
    }

    pub fn one() -> Self {
        rational_one()
    }

    pub fn new(num: i128, den: i128) -> Self {
        assert!(den != 0, "zero denominator");
        rational_new(num, den)
    }

    pub fn is_zero(&self) -> bool {
        rational_is_zero(&self.0)
    }

    pub fn is_negative(&self) -> bool {
        rational_is_negative(&self.0)
    }

    pub fn abs(&self) -> Self {
        if self.is_negative() {
            -self.clone()
        } else {
            self.clone()
        }
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut base = self.clone();
        let mut exp = exp;
        let mut out = Rational::one();
        while exp > 0 {
            if exp & 1 == 1 {
                out = Rational(out.0 * &base.0);
            }
            exp >>= 1;
            if exp > 0 {
                base = Rational(rational_mul_backend(&base.0, &base.0));
            }
        }
        out
    }
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_zero() -> Rational {
    Rational(BackendRational::zero())
}

#[cfg(feature = "gmp-rational")]
fn rational_zero() -> Rational {
    Rational(BackendRational::from(0))
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_one() -> Rational {
    Rational(BackendRational::one())
}

#[cfg(feature = "gmp-rational")]
fn rational_one() -> Rational {
    Rational(BackendRational::from(1))
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_new(num: i128, den: i128) -> Rational {
    Rational(BackendRational::new(BigInt::from(num), BigInt::from(den)))
}

#[cfg(feature = "gmp-rational")]
fn rational_new(num: i128, den: i128) -> Rational {
    Rational(BackendRational::from((num, den)))
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_is_zero(value: &BackendRational) -> bool {
    value.is_zero()
}

#[cfg(feature = "gmp-rational")]
fn rational_is_zero(value: &BackendRational) -> bool {
    value.is_zero()
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_is_negative(value: &BackendRational) -> bool {
    value.is_negative()
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_neg_backend(value: &BackendRational) -> BackendRational {
    -value
}

#[cfg(feature = "gmp-rational")]
fn rational_neg_backend(value: &BackendRational) -> BackendRational {
    (-value).into()
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_add_backend(left: &BackendRational, right: &BackendRational) -> BackendRational {
    left + right
}

#[cfg(feature = "gmp-rational")]
fn rational_add_backend(left: &BackendRational, right: &BackendRational) -> BackendRational {
    (left + right).into()
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_sub_backend(left: &BackendRational, right: &BackendRational) -> BackendRational {
    left - right
}

#[cfg(feature = "gmp-rational")]
fn rational_sub_backend(left: &BackendRational, right: &BackendRational) -> BackendRational {
    (left - right).into()
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_mul_backend(left: &BackendRational, right: &BackendRational) -> BackendRational {
    left * right
}

#[cfg(feature = "gmp-rational")]
fn rational_mul_backend(left: &BackendRational, right: &BackendRational) -> BackendRational {
    (left * right).into()
}

#[cfg(not(feature = "gmp-rational"))]
fn rational_div_backend(left: &BackendRational, right: &BackendRational) -> BackendRational {
    left / right
}

#[cfg(feature = "gmp-rational")]
fn rational_div_backend(left: &BackendRational, right: &BackendRational) -> BackendRational {
    (left / right).into()
}

#[cfg(feature = "gmp-rational")]
fn rational_is_negative(value: &BackendRational) -> bool {
    value.is_negative()
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
        self.0 += rhs.0;
    }
}

impl Sub for Rational {
    type Output = Rational;

    fn sub(self, rhs: Self) -> Self::Output {
        Rational(self.0 - rhs.0)
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
        #[cfg(feature = "gmp-rational")]
        {
            write!(f, "{}", self.0)
        }
        #[cfg(not(feature = "gmp-rational"))]
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
pub trait Coeff: Clone + PartialEq + Eq + fmt::Debug + Sized {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_rational(value: Rational) -> Self;
    fn is_zero(&self) -> bool;
    fn is_structurally_zero(&self) -> bool {
        self.is_zero()
    }
    fn is_one(&self) -> bool {
        self == &Self::one()
    }
    fn is_structurally_one(&self) -> bool {
        self.is_one()
    }
    fn neg(&self) -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;

    fn add_assign(&mut self, rhs: &Self) {
        *self = self.add(rhs);
    }

    fn add_product_assign(&mut self, left: &Self, right: &Self) {
        if left.is_structurally_zero() || right.is_structurally_zero() {
            return;
        }
        let product = left.mul(right);
        self.add_assign(&product);
    }

    fn from_usize(value: usize) -> Self {
        Self::from_rational(Rational::from(value))
    }

    fn pow_usize(&self, exp: usize) -> Self {
        let mut base = self.clone();
        let mut exp = exp;
        let mut out = Self::one();
        while exp > 0 {
            if exp & 1 == 1 {
                out = out.mul(&base);
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(&base);
            }
        }
        out
    }

    fn complexity_terms(&self) -> usize {
        usize::from(!self.is_structurally_zero())
    }

    fn complexity_denominator_factors(&self) -> usize {
        0
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
        Rational(rational_neg_backend(&self.0))
    }

    fn add(&self, rhs: &Self) -> Self {
        Rational(rational_add_backend(&self.0, &rhs.0))
    }

    fn add_assign(&mut self, rhs: &Self) {
        self.0 += rhs.0.clone();
    }

    fn sub(&self, rhs: &Self) -> Self {
        Rational(rational_sub_backend(&self.0, &rhs.0))
    }

    fn mul(&self, rhs: &Self) -> Self {
        Rational(rational_mul_backend(&self.0, &rhs.0))
    }

    fn add_product_assign(&mut self, left: &Self, right: &Self) {
        if left.is_zero() || right.is_zero() {
            return;
        }
        self.0 += rational_mul_backend(&left.0, &right.0);
    }

    fn div(&self, rhs: &Self) -> Self {
        assert!(!rhs.is_zero(), "division by zero");
        Rational(rational_div_backend(&self.0, &rhs.0))
    }
}

/// Product of interned variables with integer exponents, kept sorted by
/// variable id.
///
/// Arithmetic (multiplication, quotient splitting) works purely on the ids by
/// linear merges; only display and name-based evaluation resolve names through
/// the symbol table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Monomial(Vec<(u32, i32)>);

impl Monomial {
    pub fn one() -> Self {
        Self(Vec::new())
    }

    pub fn variable(name: impl Into<String>) -> Self {
        Self(vec![(intern_symbol(&name.into()), 1)])
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = Vec::with_capacity(self.0.len() + rhs.0.len());
        let mut left = 0usize;
        let mut right = 0usize;
        while left < self.0.len() && right < rhs.0.len() {
            match self.0[left].0.cmp(&rhs.0[right].0) {
                Ordering::Less => {
                    out.push(self.0[left]);
                    left += 1;
                }
                Ordering::Greater => {
                    out.push(rhs.0[right]);
                    right += 1;
                }
                Ordering::Equal => {
                    let exp = self.0[left].1 + rhs.0[right].1;
                    if exp != 0 {
                        out.push((self.0[left].0, exp));
                    }
                    left += 1;
                    right += 1;
                }
            }
        }
        out.extend_from_slice(&self.0[left..]);
        out.extend_from_slice(&rhs.0[right..]);
        Self(out)
    }

    fn split_quotient_monomials(&self, rhs: &Self) -> (Self, Self) {
        let mut numerator = Vec::new();
        let mut denominator = Vec::new();
        let mut push = |id: u32, exp: i32| {
            if exp > 0 {
                numerator.push((id, exp));
            } else if exp < 0 {
                denominator.push((id, -exp));
            }
        };
        let mut left = 0usize;
        let mut right = 0usize;
        while left < self.0.len() && right < rhs.0.len() {
            match self.0[left].0.cmp(&rhs.0[right].0) {
                Ordering::Less => {
                    push(self.0[left].0, self.0[left].1);
                    left += 1;
                }
                Ordering::Greater => {
                    push(rhs.0[right].0, -rhs.0[right].1);
                    right += 1;
                }
                Ordering::Equal => {
                    push(self.0[left].0, self.0[left].1 - rhs.0[right].1);
                    left += 1;
                    right += 1;
                }
            }
        }
        for &(id, exp) in &self.0[left..] {
            push(id, exp);
        }
        for &(id, exp) in &rhs.0[right..] {
            push(id, -exp);
        }
        (Self(numerator), Self(denominator))
    }

    fn lambda_line_weighted_degree(&self, weights: &[Rational]) -> Option<(i32, Rational)> {
        let mut degree = 0i32;
        let mut coefficient = Rational::one();
        for (id, exp) in &self.0 {
            let idx = symbol_lambda_index(*id)?;
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
        }
        Some((degree, coefficient))
    }

    fn split_lambda_line_weighted_degree(
        &self,
        weights: &[Rational],
    ) -> Option<(i32, Rational, Monomial)> {
        let mut degree = 0i32;
        let mut coefficient = Rational::one();
        let mut residual = Vec::new();
        for (id, exp) in &self.0 {
            if let Some(idx) = symbol_lambda_index(*id) {
                if idx >= weights.len() {
                    return None;
                }
                degree += *exp;
                if *exp >= 0 {
                    coefficient = coefficient * weights[idx].pow_usize(*exp as usize);
                } else {
                    coefficient = coefficient / weights[idx].pow_usize((-*exp) as usize);
                }
            } else {
                residual.push((*id, *exp));
            }
        }
        Some((degree, coefficient, Monomial(residual)))
    }

    fn evaluate_lambda_weights(&self, weights: &[Rational]) -> Option<Rational> {
        let mut coefficient = Rational::one();
        for (id, exp) in &self.0 {
            let idx = symbol_lambda_index(*id)?;
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

    fn evaluate_variables(&self, values: &BTreeMap<String, Rational>) -> Result<Rational, GwError> {
        let mut coefficient = Rational::one();
        for (id, exp) in &self.0 {
            let name = symbol_name(*id);
            let value = values.get(&name).ok_or_else(|| {
                GwError::AlgebraFailure(format!(
                    "missing value for symbolic variable `{name}` in monomial `{self}`"
                ))
            })?;
            if *exp >= 0 {
                coefficient = coefficient * value.pow_usize(*exp as usize);
            } else {
                coefficient = coefficient / value.pow_usize((-*exp) as usize);
            }
        }
        Ok(coefficient)
    }

    /// Factors as `(name, exponent)` pairs sorted by name, for display paths
    /// that must not depend on interning order.
    fn named_factors(&self) -> Vec<(String, i32)> {
        let mut named = self
            .0
            .iter()
            .map(|&(id, exp)| (symbol_name(id), exp))
            .collect::<Vec<_>>();
        named.sort();
        named
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "1");
        }
        for (idx, (name, exp)) in self.named_factors().iter().enumerate() {
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

    pub fn term_count(&self) -> usize {
        self.terms.len()
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

    /// Splits a nonzero polynomial into a rational scalar and a canonical
    /// monic polynomial.
    ///
    /// `SparsePoly` has a deterministic monomial order, so dividing by the
    /// coefficient of its first term gives the same representative for every
    /// nonzero rational multiple of a polynomial.  Factored rational
    /// functions use this to canonicalize denominator factors without
    /// expanding their product.
    pub(crate) fn rational_scalar_and_monic(&self) -> Option<(Rational, Self)> {
        let scalar = self.terms.values().next()?.clone();
        debug_assert!(!scalar.is_zero());
        let inverse = Rational::one() / scalar.clone();
        let terms = self
            .terms
            .iter()
            .map(|(monomial, coefficient)| {
                (monomial.clone(), coefficient.clone() * inverse.clone())
            })
            .collect();
        Some((scalar, Self { terms }))
    }

    fn from_monomial_coeff(monomial: Monomial, coeff: Rational) -> Self {
        let mut out = Self::zero();
        out.add_term(monomial, coeff);
        out
    }

    fn single_term(&self) -> Option<(Monomial, Rational)> {
        if self.terms.len() == 1 {
            self.terms
                .iter()
                .next()
                .map(|(monomial, coeff)| (monomial.clone(), coeff.clone()))
        } else {
            None
        }
    }

    fn add_term(&mut self, monomial: Monomial, coeff: Rational) {
        if coeff.is_zero() {
            return;
        }
        match self.terms.entry(monomial) {
            Entry::Vacant(entry) => {
                entry.insert(coeff);
            }
            Entry::Occupied(mut entry) => {
                *entry.get_mut() += coeff;
                if entry.get().is_zero() {
                    entry.remove();
                }
            }
        }
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut base = self.clone();
        let mut exp = exp;
        let mut out = SparsePoly::one();
        while exp > 0 {
            if exp & 1 == 1 {
                out = &out * &base;
            }
            exp >>= 1;
            if exp > 0 {
                base = &base * &base;
            }
        }
        out
    }

    pub fn evaluate_variables(
        &self,
        values: &BTreeMap<String, Rational>,
    ) -> Result<Rational, GwError> {
        let mut total = Rational::zero();
        for (monomial, coeff) in &self.terms {
            total += coeff.clone() * monomial.evaluate_variables(values)?;
        }
        Ok(total)
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
            match out.entry(degree) {
                Entry::Vacant(entry) => {
                    entry.insert(term);
                }
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() += term;
                    if entry.get().is_zero() {
                        entry.remove();
                    }
                }
            }
        }
        Ok(out)
    }

    /// Write this polynomial after `lambda_i = weights[i] * lambda_0` as a
    /// polynomial in the single lambda-line parameter.  The map key is its
    /// power and the value is the coefficient in all remaining variables.
    ///
    /// This is crate-visible so factored rational functions can take the
    /// lambda-line limit one denominator factor at a time, without first
    /// multiplying every factor into one expanded polynomial.
    pub(crate) fn lambda_line_coefficients_preserving_variables(
        &self,
        weights: &[Rational],
    ) -> Result<BTreeMap<i32, SparsePoly>, GwError> {
        let mut out = BTreeMap::<i32, SparsePoly>::new();
        for (monomial, coeff) in &self.terms {
            let Some((degree, monomial_coeff, residual)) =
                monomial.split_lambda_line_weighted_degree(weights)
            else {
                return Err(GwError::AlgebraFailure(format!(
                    "lambda-line limit found lambda_i index out of range in `{monomial}`"
                )));
            };
            let mut residual_poly = SparsePoly::zero();
            residual_poly.add_term(residual, coeff.clone() * monomial_coeff);
            if residual_poly.is_zero() {
                continue;
            }
            match out.entry(degree) {
                Entry::Vacant(entry) => {
                    entry.insert(residual_poly);
                }
                Entry::Occupied(mut entry) => {
                    let next = entry.get() + &residual_poly;
                    if next.is_zero() {
                        entry.remove();
                    } else {
                        *entry.get_mut() = next;
                    }
                }
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

impl<'b> Add<&'b SparsePoly> for &SparsePoly {
    type Output = SparsePoly;

    fn add(self, rhs: &'b SparsePoly) -> Self::Output {
        let mut out = self.clone();
        for (monomial, coeff) in &rhs.terms {
            out.add_term(monomial.clone(), coeff.clone());
        }
        out
    }
}

impl<'b> Sub<&'b SparsePoly> for &SparsePoly {
    type Output = SparsePoly;

    fn sub(self, rhs: &'b SparsePoly) -> Self::Output {
        let mut out = self.clone();
        for (monomial, coeff) in &rhs.terms {
            out.add_term(monomial.clone(), -coeff.clone());
        }
        out
    }
}

impl<'b> Mul<&'b SparsePoly> for &SparsePoly {
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
        // Terms are stored in interning-id order; print in name order so the
        // rendered polynomial does not depend on which variables were interned
        // first in this process.
        let mut terms = self
            .terms
            .iter()
            .map(|(monomial, coeff)| (monomial.named_factors(), monomial, coeff))
            .collect::<Vec<_>>();
        terms.sort_by(|left, right| left.0.cmp(&right.0));

        let mut first = true;
        for (named, monomial, coeff) in terms {
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
            if named.is_empty() {
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
            match out.coeffs.entry(*order) {
                Entry::Vacant(entry) => {
                    entry.insert(coeff.clone());
                }
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() += coeff.clone();
                    if entry.get().is_zero() {
                        entry.remove();
                    }
                }
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
        !self.den.is_zero() && self.num.is_zero()
    }

    pub fn is_one(&self) -> bool {
        !self.den.is_zero() && self.num == self.den
    }

    pub fn is_structurally_one(&self) -> bool {
        self.num.is_one() && self.den.is_one()
    }

    /// Whether two stored fractions define the same rational function.
    ///
    /// [`PartialEq`] deliberately remains a cheap structural comparison: it
    /// is useful when coefficient containers and caches need to distinguish
    /// their exact stored forms.  Validation should use this method instead,
    /// because light normalization does not cancel general polynomial factors.
    pub fn equivalent(&self, rhs: &Self) -> bool {
        !self.den.is_zero()
            && !rhs.den.is_zero()
            && (self == rhs || &self.num * &rhs.den == &rhs.num * &self.den)
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
        let mut base = self.clone();
        let mut exp = exp;
        let mut out = RatFun::one();
        while exp > 0 {
            if exp & 1 == 1 {
                out = &out * &base;
            }
            exp >>= 1;
            if exp > 0 {
                base = &base * &base;
            }
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
        if let (Some((num_monomial, num_coeff)), Some((den_monomial, den_coeff))) =
            (self.num.single_term(), self.den.single_term())
        {
            let (num_residual, den_residual) = num_monomial.split_quotient_monomials(&den_monomial);
            return RatFun {
                num: SparsePoly::from_monomial_coeff(num_residual, num_coeff / den_coeff),
                den: SparsePoly::from_monomial_coeff(den_residual, Rational::one()),
            };
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

    pub fn lambda_line_limit_preserving_variables(
        &self,
        target_n: usize,
        weights: &[Rational],
    ) -> Result<RatFun, GwError> {
        if weights.len() != target_n + 1 {
            return Err(GwError::AlgebraFailure(format!(
                "expected {} lambda-line weights, got {}",
                target_n + 1,
                weights.len()
            )));
        }
        let num = self
            .num
            .lambda_line_coefficients_preserving_variables(weights)?
            .into_iter()
            .map(|(degree, coefficient)| {
                (
                    degree,
                    RatFun {
                        num: coefficient,
                        den: SparsePoly::one(),
                    },
                )
            })
            .collect();
        let den = self
            .den
            .lambda_line_coefficients_preserving_variables(weights)?
            .into_iter()
            .map(|(degree, coefficient)| {
                (
                    degree,
                    RatFun {
                        num: coefficient,
                        den: SparsePoly::one(),
                    },
                )
            })
            .collect();
        let coeffs = ratio_laurent_series_coeff(&num, &den, 0)?;
        if coeffs
            .iter()
            .any(|(order, coeff)| *order < 0 && !coeff.is_zero())
        {
            return Err(GwError::NonFiniteLimit(
                "negative Laurent terms remain after base lambda-line summation".to_string(),
            ));
        }
        Ok(coeffs
            .get(&0)
            .cloned()
            .unwrap_or_else(RatFun::zero)
            .normalize_light())
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

    pub fn evaluate_variables(
        &self,
        values: &BTreeMap<String, Rational>,
    ) -> Result<Rational, GwError> {
        let num = self.num.evaluate_variables(values)?;
        let den = self.den.evaluate_variables(values)?;
        if den.is_zero() {
            return Err(GwError::AlgebraFailure(
                "zero denominator after symbolic variable evaluation".to_string(),
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

    fn is_one(&self) -> bool {
        RatFun::is_one(self)
    }

    fn is_structurally_one(&self) -> bool {
        RatFun::is_structurally_one(self)
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

fn leading_term_coeff<C: Coeff>(poly: &BTreeMap<i32, C>) -> Option<(i32, C)> {
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

fn ratio_laurent_series_coeff<C: Coeff>(
    num: &BTreeMap<i32, C>,
    den: &BTreeMap<i32, C>,
    max_order: i32,
) -> Result<BTreeMap<i32, C>, GwError> {
    let Some((num_order, _)) = leading_term_coeff(num) else {
        return Ok(BTreeMap::new());
    };
    let Some((den_order, den_lead)) = leading_term_coeff(den) else {
        return Err(GwError::AlgebraFailure(
            "zero denominator after lambda-line substitution".to_string(),
        ));
    };
    let min_order = num_order - den_order;
    if min_order > max_order {
        return Ok(BTreeMap::new());
    }
    let inner_max = (max_order - min_order) as usize;

    let mut den_unit = vec![C::zero(); inner_max + 1];
    den_unit[0] = C::one();
    for (order, coeff) in den {
        let shifted = *order - den_order;
        if shifted > 0 && shifted as usize <= inner_max {
            den_unit[shifted as usize] = coeff.div(&den_lead);
        }
    }

    let mut inv = vec![C::zero(); inner_max + 1];
    inv[0] = C::one();
    for degree in 1..=inner_max {
        let mut sum = C::zero();
        for k in 1..=degree {
            sum = sum.add(&den_unit[k].mul(&inv[degree - k]));
        }
        inv[degree] = sum.neg();
    }

    let mut num_shift = vec![C::zero(); inner_max + 1];
    for (order, coeff) in num {
        let shifted = *order - num_order;
        if shifted >= 0 && shifted as usize <= inner_max {
            num_shift[shifted as usize] = coeff.clone();
        }
    }

    let mut coeffs = BTreeMap::new();
    for degree in 0..=inner_max {
        let mut coeff = C::zero();
        for k in 0..=degree {
            coeff = coeff.add(&num_shift[k].mul(&inv[degree - k]));
        }
        coeff = coeff.div(&den_lead);
        if !coeff.is_zero() {
            coeffs.insert(min_order + degree as i32, coeff);
        }
    }
    Ok(coeffs)
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

impl<'b> Add<&'b RatFun> for &RatFun {
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

impl<'b> Sub<&'b RatFun> for &RatFun {
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

impl<'b> Mul<&'b RatFun> for &RatFun {
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

impl<'b> Div<&'b RatFun> for &RatFun {
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
    fn lambda_line_limit_can_preserve_fiber_variables() {
        let mu = RatFun::variable("mu_0");
        let expr = &(&lambda(0) * &mu) / &lambda(0);
        let limit = expr
            .lambda_line_limit_preserving_variables(0, &[Rational::from(3)])
            .unwrap();

        assert_eq!(limit, mu);
    }

    #[test]
    fn light_normalization_cancels_single_monomial_quotients() {
        let mu = RatFun::variable("mu_0");
        let expr = &mu.pow_usize(10) / &mu.pow_usize(9);

        assert_eq!(expr, mu);
    }

    #[test]
    fn rational_function_equivalence_cancels_polynomial_factors() {
        let x = RatFun::variable("x");
        let y = RatFun::variable("y");
        let difference_of_squares = &x.pow_usize(2) - &y.pow_usize(2);
        let difference = &x - &y;
        let quotient = &difference_of_squares / &difference;
        let sum = &x + &y;

        assert_ne!(
            quotient, sum,
            "light normalization should remain structural"
        );
        assert!(quotient.equivalent(&sum));
        assert!(!quotient.equivalent(&x));
    }

    #[test]
    fn rational_function_one_check_is_semantic() {
        let x = RatFun::variable("x");
        let y = RatFun::variable("y");
        let sum = &x + &y;
        let quotient = RatFun {
            num: sum.num.clone(),
            den: sum.num,
        };
        assert!(!quotient.is_structurally_one());
        assert!(quotient.is_one());
    }

    #[test]
    fn semantic_predicates_reject_publicly_constructed_zero_denominators() {
        let invalid = RatFun {
            num: SparsePoly::zero(),
            den: SparsePoly::zero(),
        };
        assert!(!invalid.is_zero());
        assert!(!invalid.is_one());
        assert!(!invalid.equivalent(&RatFun::zero()));
        assert!(!invalid.equivalent(&invalid));
    }

    #[test]
    fn lambda_weight_evaluation_is_exact() {
        let expr = &(&lambda(0) * &lambda(0)) / &(&lambda(1) - &lambda(0));
        let value = expr
            .evaluate_lambda_weights(1, &[Rational::from(2), Rational::from(5)])
            .unwrap();
        assert_eq!(value, Rational::new(4, 3));
    }

    #[test]
    fn variable_evaluation_replaces_named_parameters() {
        let mu = RatFun::variable("mu_0");
        let expr = &(&mu + &RatFun::from(1usize)) / &(&mu - &RatFun::from(2usize));
        let mut values = BTreeMap::new();
        values.insert("mu_0".to_string(), Rational::from(5usize));
        assert_eq!(
            expr.evaluate_variables(&values).unwrap(),
            Rational::from(2usize)
        );
    }
}
