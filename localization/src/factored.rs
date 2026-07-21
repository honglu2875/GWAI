//! Factored rational expressions for symbolic coefficient arithmetic.
//!
//! `RatFun` is intentionally simple: every denominator is an expanded sparse
//! polynomial.  That is a good default for small exact computations, but it is
//! the wrong representation for equivariant twisted graph sums where many
//! factors such as `mu_j - a_j lambda_i` are multiplied repeatedly.  This module
//! keeps denominators as factor lists and only expands them when explicitly
//! converting back to `RatFun`.

use crate::core::algebra::{Coeff, RatFun, Rational, SparsePoly};
use crate::core::error::GwError;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

const SCALAR_DISPLAY_EXPANSION_TERM_LIMIT: usize = 1024;
const ZERO_DISPLAY_EXPANSION_TERM_LIMIT: usize = 16;

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
        let (factors, numerator_scale) = normalize_factors(factors);
        let mut out = Self::zero();
        out.add_normalized_term(factors, scale_poly(&num, numerator_scale));
        out
    }

    pub fn is_zero(&self) -> bool {
        if self.terms.is_empty() {
            return true;
        }
        if self.terms.len() == 1 {
            return self.terms.values().next().is_some_and(SparsePoly::is_zero);
        }
        self.to_ratfun().is_zero()
    }

    pub fn is_structurally_zero(&self) -> bool {
        self.terms.is_empty()
            || (self.terms.len() == 1
                && self.terms.values().next().is_some_and(SparsePoly::is_zero))
    }

    pub fn is_one(&self) -> bool {
        if self.is_structurally_one() {
            return true;
        }
        self.to_ratfun().is_one()
    }

    pub fn is_structurally_one(&self) -> bool {
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
                return self
                    .can_expand_for_scalar_display()
                    .then(|| self.to_ratfun().as_rational())
                    .flatten();
            }
            total += numerator.constant_term()?;
        }
        Some(total)
    }

    pub fn as_structural_rational(&self) -> Option<Rational> {
        let mut total = Rational::zero();
        for (factors, numerator) in &self.terms {
            if !factors.is_empty() {
                return None;
            }
            total += numerator.constant_term()?;
        }
        Some(total)
    }

    fn can_expand_for_scalar_display(&self) -> bool {
        self.terms.len() <= SCALAR_DISPLAY_EXPANSION_TERM_LIMIT
            && self.expanded_denominator_term_count_upper_bound()
                <= SCALAR_DISPLAY_EXPANSION_TERM_LIMIT
    }

    fn can_expand_for_zero_display(&self) -> bool {
        self.terms.len() <= ZERO_DISPLAY_EXPANSION_TERM_LIMIT
            && self.expanded_denominator_term_count_upper_bound()
                <= ZERO_DISPLAY_EXPANSION_TERM_LIMIT
            && self.total_denominator_factor_count() <= 2 * ZERO_DISPLAY_EXPANSION_TERM_LIMIT
            && self
                .terms
                .values()
                .map(SparsePoly::term_count)
                .sum::<usize>()
                <= 4 * ZERO_DISPLAY_EXPANSION_TERM_LIMIT
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut base = self.clone();
        let mut exp = exp;
        let mut out = Self::one();
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

    /// Take the exact limit along `lambda_i = weights[i] * lambda_0` while
    /// preserving every non-lambda variable.
    ///
    /// Each denominator factor is expanded only as a *truncated* power series
    /// in the single lambda-line parameter.  In particular, this never forms
    /// the product of all equivariant denominator factors before taking the
    /// limit.  Laurent coefficients are summed across terms before poles are
    /// diagnosed, so cancellations between different factorizations remain
    /// mathematically visible.
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

        let profile = crate::env_flag("GW_PROFILE");
        let expansion_started = std::time::Instant::now();
        let mut coefficients = BTreeMap::<i32, FactoredRatFun>::new();
        for (factors, numerator) in &self.terms {
            let Some((minimum_order, term_coefficients)) =
                factored_term_lambda_line_series_through_zero(numerator, factors, weights)?
            else {
                continue;
            };
            for (relative_order, coefficient) in term_coefficients.into_iter().enumerate() {
                if coefficient.is_structurally_zero() {
                    continue;
                }
                let relative_order = i32::try_from(relative_order).map_err(|_| {
                    GwError::AlgebraFailure(
                        "lambda-line Laurent expansion order exceeds i32".to_string(),
                    )
                })?;
                let order = minimum_order.checked_add(relative_order).ok_or_else(|| {
                    GwError::AlgebraFailure(
                        "lambda-line Laurent expansion order overflow".to_string(),
                    )
                })?;
                match coefficients.entry(order) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(coefficient);
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        <FactoredRatFun as Coeff>::add_assign(entry.get_mut(), &coefficient);
                        if entry.get().is_structurally_zero() {
                            entry.remove();
                        }
                    }
                }
            }
        }
        if profile {
            eprintln!(
                "GW_PROFILE factored_lambda_limit_series={:.3}s input_terms={} input_factors={} orders={} output_terms={} output_factors={}",
                expansion_started.elapsed().as_secs_f64(),
                self.term_count(),
                self.total_denominator_factor_count(),
                coefficients.len(),
                coefficients.values().map(FactoredRatFun::term_count).sum::<usize>(),
                coefficients
                    .values()
                    .map(FactoredRatFun::total_denominator_factor_count)
                    .sum::<usize>()
            );
        }

        let pole_check_started = std::time::Instant::now();
        for (order, coefficient) in coefficients.range(..0) {
            if !coefficient.is_zero() {
                return Err(GwError::NonFiniteLimit(format!(
                    "negative Laurent term of order {order} remains after base lambda-line summation"
                )));
            }
        }
        if profile {
            eprintln!(
                "GW_PROFILE factored_lambda_limit_pole_check={:.3}s",
                pole_check_started.elapsed().as_secs_f64()
            );
        }
        let conversion_started = std::time::Instant::now();
        let result = coefficients
            .get(&0)
            .map(FactoredRatFun::to_ratfun)
            .unwrap_or_else(RatFun::zero)
            .normalize_light();
        if profile {
            eprintln!(
                "GW_PROFILE factored_lambda_limit_conversion={:.3}s result_num_terms={} result_den_terms={}",
                conversion_started.elapsed().as_secs_f64(),
                result.num.term_count(),
                result.den.term_count()
            );
        }
        Ok(result)
    }

    pub fn to_ratfun(&self) -> RatFun {
        if self.terms.is_empty() {
            return RatFun::zero();
        }

        // Use the least common multiple of the *stored formal factors* rather
        // than multiplying every term denominator into the running result.
        // General polynomial gcds are intentionally out of scope, but exact
        // repeated factors are already canonical keys, and exploiting them
        // prevents denominator powers from growing with the number of terms.
        let mut maximum_multiplicities = BTreeMap::<SparsePoly, usize>::new();
        for factors in self.terms.keys() {
            let mut multiplicities = BTreeMap::<&SparsePoly, usize>::new();
            for factor in factors {
                *multiplicities.entry(factor).or_default() += 1;
            }
            for (factor, multiplicity) in multiplicities {
                maximum_multiplicities
                    .entry(factor.clone())
                    .and_modify(|maximum| *maximum = (*maximum).max(multiplicity))
                    .or_insert(multiplicity);
            }
        }
        let common_factors = maximum_multiplicities
            .into_iter()
            .flat_map(|(factor, multiplicity)| std::iter::repeat_n(factor, multiplicity))
            .collect::<Vec<_>>();

        let mut total_numerator = SparsePoly::zero();
        for (factors, numerator) in &self.terms {
            let missing = missing_normalized_factors(&common_factors, factors);
            total_numerator = &total_numerator + &(numerator * &multiply_factors(&missing));
        }
        RatFun {
            num: total_numerator,
            den: multiply_factors(&common_factors),
        }
        .normalize_light()
    }

    fn add_term(&mut self, factors: Vec<SparsePoly>, numerator: SparsePoly) {
        let (factors, numerator_scale) = normalize_factors(factors);
        self.add_normalized_term(factors, scale_poly(&numerator, numerator_scale));
    }

    fn add_normalized_term(&mut self, factors: Vec<SparsePoly>, numerator: SparsePoly) {
        if numerator.is_zero() {
            return;
        }
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

    /// Divide every summand by the same list of polynomial factors in one
    /// pass.  This is the bulk counterpart of repeated one-term division and
    /// avoids cloning a successively growing factor list for every factor.
    fn divide_by_polynomial_factors(&self, factors: &[SparsePoly]) -> Self {
        if self.is_structurally_zero() || factors.is_empty() {
            return self.clone();
        }
        let (factors, numerator_scale) = normalize_factors(factors.to_vec());
        let mut out = Self::zero();
        for (existing_factors, numerator) in &self.terms {
            out.add_normalized_term(
                merge_normalized_factors(existing_factors, &factors),
                scale_poly(numerator, numerator_scale.clone()),
            );
        }
        out
    }
}

/// Canonicalizes denominator factors up to rational units.
///
/// If `factor = scalar * monic`, then dividing by `factor` multiplies the
/// numerator by `scalar^{-1}`.  Extracting that unit makes scalar multiples
/// (including opposite signs) share the same factor key, which lets additions
/// combine like denominators without expanding them.
fn normalize_factors(factors: Vec<SparsePoly>) -> (Vec<SparsePoly>, Rational) {
    let mut normalized = Vec::with_capacity(factors.len());
    let mut numerator_scale = Rational::one();
    for factor in factors {
        assert!(!factor.is_zero(), "division by zero factored denominator");
        let (scalar, monic) = factor
            .rational_scalar_and_monic()
            .expect("nonzero denominator factor has a leading coefficient");
        numerator_scale = numerator_scale / scalar;
        if !monic.is_one() {
            normalized.push(monic);
        }
    }
    normalized.sort();
    (normalized, numerator_scale)
}

/// Merges two already-normalized denominator lists without sorting their
/// concatenation again.  Factor lists are the keys of every stored term, so
/// multiplication reaches this path much more often than construction from an
/// arbitrary external list.
fn merge_normalized_factors(left: &[SparsePoly], right: &[SparsePoly]) -> Vec<SparsePoly> {
    let mut factors = Vec::with_capacity(left.len() + right.len());
    let (mut left_idx, mut right_idx) = (0, 0);
    while left_idx < left.len() && right_idx < right.len() {
        if left[left_idx] <= right[right_idx] {
            factors.push(left[left_idx].clone());
            left_idx += 1;
        } else {
            factors.push(right[right_idx].clone());
            right_idx += 1;
        }
    }
    factors.extend_from_slice(&left[left_idx..]);
    factors.extend_from_slice(&right[right_idx..]);
    factors
}

fn insert_normalized_factor(
    mut factors: Vec<SparsePoly>,
    factor: SparsePoly,
) -> (Vec<SparsePoly>, Rational) {
    assert!(!factor.is_zero(), "division by zero factored denominator");
    let (scalar, monic) = factor
        .rational_scalar_and_monic()
        .expect("nonzero denominator factor has a leading coefficient");
    if !monic.is_one() {
        let index = factors.binary_search(&monic).unwrap_or_else(|index| index);
        factors.insert(index, monic);
    }
    (factors, Rational::one() / scalar)
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

/// Multiset difference `common - present` for sorted normalized factor lists.
fn missing_normalized_factors(common: &[SparsePoly], present: &[SparsePoly]) -> Vec<SparsePoly> {
    let mut missing = Vec::with_capacity(common.len().saturating_sub(present.len()));
    let mut present_index = 0usize;
    for factor in common {
        if present.get(present_index) == Some(factor) {
            present_index += 1;
        } else {
            missing.push(factor.clone());
        }
    }
    debug_assert_eq!(present_index, present.len());
    missing
}

/// Laurent-expand one stored summand through absolute order zero.
///
/// Successive formal division by each factor is exact to the requested order:
/// the coefficient of relative order `r` only depends on numerator and factor
/// coefficients of relative order at most `r`.  Thus truncating at the order
/// needed to reach lambda degree zero loses no information about the limit or
/// its pole part.
fn factored_term_lambda_line_series_through_zero(
    numerator: &SparsePoly,
    factors: &[SparsePoly],
    weights: &[Rational],
) -> Result<Option<(i32, Vec<FactoredRatFun>)>, GwError> {
    let numerator_coefficients =
        numerator.lambda_line_coefficients_preserving_variables(weights)?;
    let Some((&numerator_order, _)) = numerator_coefficients.first_key_value() else {
        return Ok(None);
    };

    let mut denominator_order = 0i32;
    let mut factor_coefficients = Vec::with_capacity(factors.len());
    for factor in factors {
        let coefficients = factor.lambda_line_coefficients_preserving_variables(weights)?;
        let Some((&factor_order, _)) = coefficients.first_key_value() else {
            return Err(GwError::AlgebraFailure(
                "zero denominator factor after lambda-line substitution".to_string(),
            ));
        };
        denominator_order = denominator_order.checked_add(factor_order).ok_or_else(|| {
            GwError::AlgebraFailure("lambda-line denominator order overflow".to_string())
        })?;
        factor_coefficients.push((factor_order, coefficients));
    }

    let minimum_order = numerator_order
        .checked_sub(denominator_order)
        .ok_or_else(|| GwError::AlgebraFailure("lambda-line order overflow".to_string()))?;
    if minimum_order > 0 {
        return Ok(None);
    }

    // The overwhelmingly common localization case has valuation zero.  Its
    // constant term depends only on the leading coefficient of the numerator
    // and of each denominator factor.  Build that single factored fraction in
    // one pass; successively dividing by the factors would repeatedly clone a
    // growing factor list and is quadratic in their number.
    if minimum_order == 0 {
        let leading_factors = factor_coefficients
            .iter()
            .map(|(factor_order, coefficients)| {
                coefficients
                    .get(factor_order)
                    .expect("recorded leading factor coefficient must exist")
                    .clone()
            })
            .collect();
        let leading_numerator = numerator_coefficients
            .get(&numerator_order)
            .expect("recorded leading numerator coefficient must exist")
            .clone();
        return Ok(Some((
            0,
            vec![FactoredRatFun::from_sparse_fraction_factors(
                leading_numerator,
                leading_factors,
            )],
        )));
    }

    let maximum_relative_order = usize::try_from(-(minimum_order as i64)).map_err(|_| {
        GwError::AlgebraFailure("lambda-line expansion order exceeds usize".to_string())
    })?;

    let mut series = (0..=maximum_relative_order)
        .map(|relative_order| {
            let relative_order = i32::try_from(relative_order).ok()?;
            let order = numerator_order.checked_add(relative_order)?;
            numerator_coefficients
                .get(&order)
                .cloned()
                .map(FactoredRatFun::from_polynomial)
        })
        .map(|coefficient| coefficient.unwrap_or_else(FactoredRatFun::zero))
        .collect::<Vec<_>>();

    let leading_factors = factor_coefficients
        .iter()
        .map(|(factor_order, coefficients)| {
            coefficients
                .get(factor_order)
                .expect("recorded leading factor coefficient must exist")
                .clone()
        })
        .collect::<Vec<_>>();
    for (factor_order, coefficients) in factor_coefficients {
        let leading = FactoredRatFun::from_polynomial(
            coefficients
                .get(&factor_order)
                .expect("recorded leading factor coefficient must exist")
                .clone(),
        );
        let mut quotient = vec![FactoredRatFun::zero(); maximum_relative_order + 1];
        for relative_order in 0..=maximum_relative_order {
            let mut remainder = series[relative_order].clone();
            for factor_relative_order in 1..=relative_order {
                let Some(factor_relative_order_i32) = i32::try_from(factor_relative_order).ok()
                else {
                    return Err(GwError::AlgebraFailure(
                        "lambda-line factor expansion order exceeds i32".to_string(),
                    ));
                };
                let Some(factor_absolute_order) =
                    factor_order.checked_add(factor_relative_order_i32)
                else {
                    return Err(GwError::AlgebraFailure(
                        "lambda-line factor expansion order overflow".to_string(),
                    ));
                };
                let Some(factor_coefficient) = coefficients.get(&factor_absolute_order) else {
                    continue;
                };
                let normalized_factor_coefficient =
                    &FactoredRatFun::from_polynomial(factor_coefficient.clone()) / &leading;
                let product = &normalized_factor_coefficient
                    * &quotient[relative_order - factor_relative_order];
                remainder = &remainder - &product;
            }
            quotient[relative_order] = remainder;
        }
        series = quotient;
    }
    for coefficient in &mut series {
        *coefficient = coefficient.divide_by_polynomial_factors(&leading_factors);
    }

    Ok(Some((minimum_order, series)))
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

    fn is_structurally_zero(&self) -> bool {
        self.is_structurally_zero()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }

    fn is_structurally_one(&self) -> bool {
        self.is_structurally_one()
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

    fn add_assign(&mut self, rhs: &Self) {
        for (factors, numerator) in &rhs.terms {
            self.add_normalized_term(factors.clone(), numerator.clone());
        }
    }

    fn add_product_assign(&mut self, left: &Self, right: &Self) {
        if left.is_structurally_zero() || right.is_structurally_zero() {
            return;
        }
        for (left_factors, left_num) in &left.terms {
            for (right_factors, right_num) in &right.terms {
                self.add_normalized_term(
                    merge_normalized_factors(left_factors, right_factors),
                    left_num * right_num,
                );
            }
        }
    }

    fn complexity_terms(&self) -> usize {
        self.term_count()
    }

    fn complexity_denominator_factors(&self) -> usize {
        self.total_denominator_factor_count()
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

impl<'b> Add<&'b FactoredRatFun> for &FactoredRatFun {
    type Output = FactoredRatFun;

    fn add(self, rhs: &'b FactoredRatFun) -> Self::Output {
        let mut out = self.clone();
        <FactoredRatFun as Coeff>::add_assign(&mut out, rhs);
        out
    }
}

impl<'b> Sub<&'b FactoredRatFun> for &FactoredRatFun {
    type Output = FactoredRatFun;

    fn sub(self, rhs: &'b FactoredRatFun) -> Self::Output {
        self + &(-rhs.clone())
    }
}

impl<'b> Mul<&'b FactoredRatFun> for &FactoredRatFun {
    type Output = FactoredRatFun;

    fn mul(self, rhs: &'b FactoredRatFun) -> Self::Output {
        if self.is_structurally_zero() || rhs.is_structurally_zero() {
            return FactoredRatFun::zero();
        }
        let mut out = FactoredRatFun::zero();
        for (left_factors, left_num) in &self.terms {
            for (right_factors, right_num) in &rhs.terms {
                out.add_normalized_term(
                    merge_normalized_factors(left_factors, right_factors),
                    left_num * right_num,
                );
            }
        }
        out
    }
}

impl<'b> Div<&'b FactoredRatFun> for &FactoredRatFun {
    type Output = FactoredRatFun;

    fn div(self, rhs: &'b FactoredRatFun) -> Self::Output {
        // Only the structural check is affordable here: `is_zero` falls back
        // to a full denominator expansion for multi-term values, which would
        // run on every division.  A multi-term rhs that cancels to zero is
        // still caught below, where the expansion happens anyway.
        assert!(
            !rhs.is_structurally_zero(),
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
                let (factors, numerator_scale) =
                    insert_normalized_factor(left_factors.clone(), rhs_num.clone());
                out.add_normalized_term(factors, scale_poly(&numerator, numerator_scale));
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
        if self.is_structurally_zero()
            || (self.terms.len() > 1
                && self.can_expand_for_zero_display()
                && self.to_ratfun().is_zero())
        {
            return write!(f, "0");
        }
        if let Some(value) = self.as_structural_rational() {
            return write!(f, "{value}");
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
                    if factor.term_count() > 1 {
                        write!(f, "({factor})")?;
                    } else {
                        write!(f, "{factor}")?;
                    }
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
    fn scalar_multiple_denominators_combine_under_one_canonical_key() {
        let x = SparsePoly::variable("factor_x");
        let two_x = &SparsePoly::constant(Rational::from(2)) * &x;
        let half = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), two_x);
        let whole = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), x.clone());
        let sum = &half + &whole;

        assert_eq!(sum.term_count(), 1);
        assert_eq!(
            sum.to_ratfun(),
            FactoredRatFun::from_sparse_fraction(SparsePoly::constant(Rational::new(3, 2)), x,)
                .to_ratfun()
        );
    }

    #[test]
    fn opposite_denominator_factors_cancel_structurally() {
        let x = SparsePoly::variable("factor_sign_x");
        let y = SparsePoly::variable("factor_sign_y");
        let x_minus_y = &x - &y;
        let y_minus_x = &y - &x;
        let left = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), x_minus_y);
        let right = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), y_minus_x);

        assert!((&left + &right).is_structurally_zero());
    }

    #[test]
    fn repeated_scaled_factors_transfer_units_with_multiplicity() {
        let x = SparsePoly::variable("factor_repeat_x");
        let factors = vec![
            &SparsePoly::constant(Rational::from(2)) * &x,
            &SparsePoly::constant(Rational::from(-3)) * &x,
            x.clone(),
        ];
        let scaled = FactoredRatFun::from_sparse_fraction_factors(SparsePoly::one(), factors);
        let canonical = FactoredRatFun::from_sparse_fraction_factors(
            SparsePoly::constant(Rational::new(-1, 6)),
            vec![x.clone(), x.clone(), x],
        );

        assert_eq!(scaled.term_count(), 1);
        assert_eq!(scaled.max_denominator_factor_count(), 3);
        assert_eq!(scaled, canonical);
        assert_eq!(scaled.to_ratfun(), canonical.to_ratfun());
    }

    #[test]
    fn factor_normalization_preserves_direct_fraction_semantics() {
        let x = SparsePoly::variable("factor_semantic_x");
        let y = SparsePoly::variable("factor_semantic_y");
        let first = &SparsePoly::constant(Rational::from(-2)) * &(&x + &y);
        let second = &SparsePoly::constant(Rational::from(5)) * &(&x - &y);
        let numerator = &x + &SparsePoly::constant(Rational::from(7));
        let normalized = FactoredRatFun::from_sparse_fraction_factors(
            numerator.clone(),
            vec![first.clone(), second.clone()],
        );
        let direct = RatFun {
            num: numerator,
            den: &first * &second,
        };

        assert!(normalized.to_ratfun().equivalent(&direct));
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
    fn common_denominator_conversion_uses_formal_factor_lcm() {
        let x = SparsePoly::variable("factored_lcm_x");
        let inverse = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), x.clone());
        let inverse_square = FactoredRatFun::from_sparse_fraction_factors(
            SparsePoly::one(),
            vec![x.clone(), x.clone()],
        );
        let converted = (&inverse + &inverse_square).to_ratfun();
        let expected = RatFun {
            num: &SparsePoly::one() + &x,
            den: x.pow_usize(2),
        };

        assert_eq!(converted, expected);
    }

    #[test]
    fn direct_lambda_line_limit_matches_expansion_after_pole_cancellation() {
        let lambda = SparsePoly::variable("lambda_0");
        let one_plus_lambda = &SparsePoly::one() + &lambda;
        let pole = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), lambda.clone());
        let shifted_pole = FactoredRatFun::from_sparse_fraction_factors(
            -SparsePoly::one(),
            vec![lambda, one_plus_lambda],
        );
        let expression = &pole + &shifted_pole;
        let weights = [Rational::from(2)];

        let direct = expression
            .lambda_line_limit_preserving_variables(0, &weights)
            .unwrap();
        let expanded = expression
            .to_ratfun()
            .lambda_line_limit_preserving_variables(0, &weights)
            .unwrap();

        assert_eq!(direct, RatFun::one());
        assert!(direct.equivalent(&expanded));
    }

    #[test]
    fn direct_lambda_line_limit_matches_expansion_through_a_double_pole() {
        let lambda = SparsePoly::variable("lambda_0");
        let one_plus_lambda = &SparsePoly::one() + &lambda;
        let double_pole = FactoredRatFun::from_sparse_fraction_factors(
            SparsePoly::one(),
            vec![lambda.clone(), lambda.clone()],
        );
        let shifted_double_pole = FactoredRatFun::from_sparse_fraction_factors(
            -SparsePoly::one(),
            vec![lambda.clone(), lambda.clone(), one_plus_lambda],
        );
        let simple_pole = FactoredRatFun::from_sparse_fraction(-SparsePoly::one(), lambda);
        let expression = &(&double_pole + &shifted_double_pole) + &simple_pole;
        let weights = [Rational::from(2)];

        let direct = expression
            .lambda_line_limit_preserving_variables(0, &weights)
            .unwrap();
        let expanded = expression
            .to_ratfun()
            .lambda_line_limit_preserving_variables(0, &weights)
            .unwrap();

        assert_eq!(direct, RatFun::from(-1));
        assert!(direct.equivalent(&expanded));
    }

    #[test]
    fn direct_lambda_line_limit_preserves_variables_and_cancels_residual_poles() {
        let lambda = SparsePoly::variable("lambda_0");
        let mu = SparsePoly::variable("limit_mu");
        let mu_plus_lambda = &mu + &lambda;
        let first = FactoredRatFun::from_sparse_fraction_factors(
            SparsePoly::one(),
            vec![lambda.clone(), mu.clone()],
        );
        let second = FactoredRatFun::from_sparse_fraction_factors(
            -SparsePoly::one(),
            vec![lambda, mu_plus_lambda],
        );
        let expression = &first + &second;
        let weights = [Rational::from(3)];

        let direct = expression
            .lambda_line_limit_preserving_variables(0, &weights)
            .unwrap();
        let expanded = expression
            .to_ratfun()
            .lambda_line_limit_preserving_variables(0, &weights)
            .unwrap();
        let expected = &RatFun::one() / &RatFun::variable("limit_mu").pow_usize(2);

        assert!(direct.equivalent(&expected));
        assert!(direct.equivalent(&expanded));
    }

    #[test]
    fn direct_lambda_line_limit_rejects_an_uncancelled_pole() {
        let expression = FactoredRatFun::from_sparse_fraction(
            SparsePoly::one(),
            SparsePoly::variable("lambda_0"),
        );

        assert!(matches!(
            expression.lambda_line_limit_preserving_variables(0, &[Rational::one()]),
            Err(GwError::NonFiniteLimit(_))
        ));
    }

    #[test]
    fn direct_lambda_line_limit_omits_terms_killed_by_a_zero_weight() {
        let expression = FactoredRatFun::from_polynomial(SparsePoly::variable("lambda_0"));

        assert_eq!(
            expression
                .lambda_line_limit_preserving_variables(0, &[Rational::zero()])
                .unwrap(),
            RatFun::zero()
        );
    }

    #[test]
    fn direct_lambda_line_limit_rejects_denominator_killed_by_a_zero_weight() {
        let expression = FactoredRatFun::from_sparse_fraction(
            SparsePoly::one(),
            SparsePoly::variable("lambda_0"),
        );

        assert!(matches!(
            expression.lambda_line_limit_preserving_variables(0, &[Rational::zero()]),
            Err(GwError::AlgebraFailure(_))
        ));
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
        assert!(factored.to_ratfun().equivalent(&expanded));
    }

    #[test]
    fn cross_denominator_cancellation_is_zero() {
        let mu = FactoredRatFun::variable("mu_0");
        let expr = &(&mu / &mu) - &FactoredRatFun::one();

        assert!(expr.is_zero());
        assert_eq!(expr.to_ratfun(), RatFun::zero());
        assert_eq!(expr.to_string(), "0");
        assert_eq!((&mu / &mu).as_rational(), Some(Rational::one()));
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn division_by_hidden_zero_panics() {
        // Multi-term rhs whose terms cancel to zero: the cheap structural
        // check passes, but the expansion in the multi-term division path
        // must still refuse the zero denominator.
        let mu = FactoredRatFun::variable("mu_0");
        let hidden_zero = &(&mu / &mu) - &FactoredRatFun::one();
        let _ = &FactoredRatFun::one() / &hidden_zero;
    }

    #[test]
    fn quotient_with_matching_factor_is_one() {
        let factor = mu_shift(1);
        let quotient = FactoredRatFun::from_sparse_fraction(factor.clone(), factor);

        assert!(quotient.is_one());
    }

    #[test]
    fn qseries_can_use_factored_coefficients() {
        let factor = mu_shift(-3);
        let coeff = FactoredRatFun::from_sparse_fraction(SparsePoly::one(), factor);
        let series = crate::core::series::QSeries::constant(coeff.clone(), 2)
            .mul(&crate::core::series::QSeries::constant(coeff, 2));
        assert_eq!(series.coeff(0).unwrap().max_denominator_factor_count(), 2);
    }

    #[test]
    fn semisimple_calibration_can_use_factored_coefficients() {
        let q_degree = 1;
        let z_order = 1;
        let size = 1;
        let matrix = crate::core::series::SeriesMatrix::<FactoredRatFun>::identity(size, q_degree);
        let scalar = crate::core::series::QSeries::<FactoredRatFun>::one(q_degree);
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

        let kernel =
            crate::givental::GiventalGraphKernel::from_calibration(calibration, 1).unwrap();
        assert_eq!(kernel.inverse_r().len(), z_order + 1);
        assert_eq!(kernel.translation().len(), size);
    }
}
