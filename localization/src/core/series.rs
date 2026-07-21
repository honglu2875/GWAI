//! Coefficient-generic truncated Novikov series and series matrices.

use super::algebra::{Coeff, RatFun, Rational};
use super::error::GwError;
use super::fused;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QSeries<C = RatFun> {
    coeffs: Vec<C>,
}

pub type RationalQSeries = QSeries<Rational>;

impl<C: Coeff> QSeries<C> {
    pub fn from_coeffs(coeffs: Vec<C>) -> Self {
        assert!(
            !coeffs.is_empty(),
            "q-series must contain at least its constant coefficient"
        );
        Self { coeffs }
    }

    pub fn zero(max_degree: usize) -> Self {
        Self {
            coeffs: vec![C::zero(); max_degree + 1],
        }
    }

    pub fn one(max_degree: usize) -> Self {
        Self::constant(C::one(), max_degree)
    }

    pub fn constant(value: C, max_degree: usize) -> Self {
        let mut out = Self::zero(max_degree);
        out.coeffs[0] = value;
        out
    }

    pub fn q(max_degree: usize) -> Self {
        let mut out = Self::zero(max_degree);
        if max_degree >= 1 {
            out.coeffs[1] = C::one();
        }
        out
    }

    pub fn max_degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    pub fn coeff(&self, degree: usize) -> Option<&C> {
        self.coeffs.get(degree)
    }

    pub fn coeffs(&self) -> &[C] {
        &self.coeffs
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(Coeff::is_zero)
    }

    pub fn is_structurally_zero(&self) -> bool {
        self.coeffs.iter().all(Coeff::is_structurally_zero)
    }

    pub fn is_structurally_one(&self) -> bool {
        self.coeffs.first().is_some_and(Coeff::is_structurally_one)
            && self.coeffs.iter().skip(1).all(Coeff::is_structurally_zero)
    }

    pub fn complexity_terms(&self) -> usize {
        self.coeffs.iter().map(Coeff::complexity_terms).sum()
    }

    pub fn complexity_denominator_factors(&self) -> usize {
        self.coeffs
            .iter()
            .map(Coeff::complexity_denominator_factors)
            .sum()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = self.clone();
        out.add_assign(rhs);
        out
    }

    pub fn add_assign(&mut self, rhs: &Self) {
        self.assert_same_truncation(rhs);
        for (left, right) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            fused::add_assign(left, right);
        }
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.assert_same_truncation(rhs);
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(rhs.coeffs.iter())
                .map(|(a, b)| a.sub(b))
                .collect(),
        }
    }

    pub fn neg(&self) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(Coeff::neg).collect(),
        }
    }

    pub fn scale(&self, scalar: &C) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|coeff| coeff.mul(scalar)).collect(),
        }
    }

    pub fn q_derivative(&self) -> Self {
        let coeffs = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(degree, coeff)| {
                if degree == 0 {
                    C::zero()
                } else {
                    coeff.mul(&C::from_usize(degree))
                }
            })
            .collect();
        Self { coeffs }
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        self.assert_same_truncation(rhs);
        if self.is_structurally_zero() || rhs.is_structurally_zero() {
            return Self::zero(self.max_degree());
        }
        if self.is_structurally_one() {
            return rhs.clone();
        }
        if rhs.is_structurally_one() {
            return self.clone();
        }
        let max_degree = self.max_degree();
        let mut out = vec![C::zero(); max_degree + 1];
        for i in 0..=max_degree {
            if self.coeffs[i].is_structurally_zero() {
                continue;
            }
            for j in 0..=max_degree - i {
                if rhs.coeffs[j].is_structurally_zero() {
                    continue;
                }
                fused::add_product_assign(&mut out[i + j], &self.coeffs[i], &rhs.coeffs[j]);
            }
        }
        Self { coeffs: out }
    }

    pub fn add_product_assign(&mut self, left: &Self, right: &Self) {
        left.assert_same_truncation(right);
        self.assert_same_truncation(left);
        if left.is_structurally_zero() || right.is_structurally_zero() {
            return;
        }
        if left.is_structurally_one() {
            self.add_assign(right);
            return;
        }
        if right.is_structurally_one() {
            self.add_assign(left);
            return;
        }
        let max_degree = self.max_degree();
        for i in 0..=max_degree {
            if left.coeffs[i].is_structurally_zero() {
                continue;
            }
            for j in 0..=max_degree - i {
                if right.coeffs[j].is_structurally_zero() {
                    continue;
                }
                fused::add_product_assign(
                    &mut self.coeffs[i + j],
                    &left.coeffs[i],
                    &right.coeffs[j],
                );
            }
        }
    }

    pub fn inverse(&self) -> Result<Self, GwError> {
        let max_degree = self.max_degree();
        let constant = self
            .coeffs
            .first()
            .ok_or_else(|| GwError::AlgebraFailure("empty q-series".to_string()))?;
        if constant.is_zero() {
            return Err(GwError::AlgebraFailure(
                "cannot invert q-series with zero constant term".to_string(),
            ));
        }

        let mut out = vec![C::zero(); max_degree + 1];
        out[0] = C::one().div(constant);
        for degree in 1..=max_degree {
            let mut sum = C::zero();
            for k in 1..=degree {
                fused::add_product_assign(&mut sum, &self.coeffs[k], &out[degree - k]);
            }
            out[degree] = sum.div(constant).neg();
        }
        Ok(Self { coeffs: out })
    }

    pub fn div(&self, rhs: &Self) -> Result<Self, GwError> {
        Ok(self.mul(&rhs.inverse()?))
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut base = self.clone();
        let mut exp = exp;
        let mut out = QSeries::<C>::one(self.max_degree());
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

    pub fn sqrt_with_constant_one(&self) -> Result<Self, GwError> {
        let max_degree = self.max_degree();
        if !self.coeff(0).is_some_and(Coeff::is_one) {
            return Err(GwError::AlgebraFailure(
                "sqrt_with_constant_one requires constant coefficient 1".to_string(),
            ));
        }

        let mut out = vec![C::zero(); max_degree + 1];
        out[0] = C::one();
        for degree in 1..=max_degree {
            let mut quadratic_terms = C::zero();
            for left in 1..degree {
                fused::add_product_assign(&mut quadratic_terms, &out[left], &out[degree - left]);
            }
            let numerator = self.coeffs[degree].sub(&quadratic_terms);
            out[degree] = numerator.div(&C::from_usize(2));
        }
        Ok(Self { coeffs: out })
    }

    fn assert_same_truncation(&self, rhs: &Self) {
        assert_eq!(
            self.max_degree(),
            rhs.max_degree(),
            "q-series truncation mismatch"
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesMatrix<C = RatFun> {
    rows: usize,
    cols: usize,
    entries: Vec<Vec<QSeries<C>>>,
}

pub type RationalSeriesMatrix = SeriesMatrix<Rational>;

impl<C: Coeff> SeriesMatrix<C> {
    pub fn zero(rows: usize, cols: usize, max_degree: usize) -> Self {
        Self {
            rows,
            cols,
            entries: vec![vec![QSeries::<C>::zero(max_degree); cols]; rows],
        }
    }

    pub fn identity(size: usize, max_degree: usize) -> Self {
        let mut out = Self::zero(size, size, max_degree);
        for i in 0..size {
            out.entries[i][i] = QSeries::<C>::one(max_degree);
        }
        out
    }

    pub fn from_entries(entries: Vec<Vec<QSeries<C>>>) -> Self {
        let rows = entries.len();
        let cols = entries.first().map(Vec::len).unwrap_or_default();
        assert!(
            entries.iter().all(|row| row.len() == cols),
            "series-matrix rows must have equal lengths"
        );
        if let Some(max_degree) = entries
            .iter()
            .flat_map(|row| row.iter())
            .next()
            .map(QSeries::<C>::max_degree)
        {
            assert!(
                entries
                    .iter()
                    .flat_map(|row| row.iter())
                    .all(|entry| entry.max_degree() == max_degree),
                "series-matrix entries must have equal q-series truncations"
            );
        }
        Self {
            rows,
            cols,
            entries,
        }
    }

    pub fn constant(entries: Vec<Vec<C>>, max_degree: usize) -> Self {
        Self::from_entries(
            entries
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| QSeries::<C>::constant(entry, max_degree))
                        .collect()
                })
                .collect(),
        )
    }

    pub fn diagonal(diagonal: Vec<QSeries<C>>) -> Self {
        let size = diagonal.len();
        let max_degree = diagonal
            .first()
            .map(QSeries::<C>::max_degree)
            .unwrap_or_default();
        assert!(
            diagonal
                .iter()
                .all(|entry| entry.max_degree() == max_degree),
            "series-matrix diagonal entries must have equal q-series truncations"
        );
        let mut out = Self::zero(size, size, max_degree);
        for (idx, value) in diagonal.into_iter().enumerate() {
            out.entries[idx][idx] = value;
        }
        out
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn entry(&self, row: usize, col: usize) -> &QSeries<C> {
        &self.entries[row][col]
    }

    pub fn entries(&self) -> &[Vec<QSeries<C>>] {
        &self.entries
    }

    pub fn transpose(&self) -> Self {
        let max_degree = self.max_degree();
        let mut out = Self::zero(self.cols, self.rows, max_degree);
        for row in 0..self.rows {
            for col in 0..self.cols {
                out.entries[col][row] = self.entries[row][col].clone();
            }
        }
        out
    }

    pub fn q_derivative(&self) -> Self {
        Self::from_entries(
            self.entries
                .iter()
                .map(|row| row.iter().map(QSeries::<C>::q_derivative).collect())
                .collect(),
        )
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = self.clone();
        out.add_assign(rhs);
        out
    }

    pub fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        for (left_row, right_row) in self.entries.iter_mut().zip(rhs.entries.iter()) {
            for (left, right) in left_row.iter_mut().zip(right_row.iter()) {
                left.add_assign(right);
            }
        }
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        Self::from_entries(
            self.entries
                .iter()
                .zip(rhs.entries.iter())
                .map(|(left_row, right_row)| {
                    left_row
                        .iter()
                        .zip(right_row.iter())
                        .map(|(left, right)| left.sub(right))
                        .collect()
                })
                .collect(),
        )
    }

    pub fn neg(&self) -> Self {
        Self::from_entries(
            self.entries
                .iter()
                .map(|row| row.iter().map(QSeries::<C>::neg).collect())
                .collect(),
        )
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows);
        let max_degree = self.max_degree();
        assert_eq!(max_degree, rhs.max_degree());
        let mut out = Self::zero(self.rows, rhs.cols, max_degree);
        for row in 0..self.rows {
            for col in 0..rhs.cols {
                let mut total = QSeries::<C>::zero(max_degree);
                for k in 0..self.cols {
                    if self.entries[row][k].is_structurally_zero()
                        || rhs.entries[k][col].is_structurally_zero()
                    {
                        continue;
                    }
                    total.add_product_assign(&self.entries[row][k], &rhs.entries[k][col]);
                }
                out.entries[row][col] = total;
            }
        }
        out
    }

    pub fn max_degree(&self) -> usize {
        self.entries
            .first()
            .and_then(|row| row.first())
            .map(QSeries::<C>::max_degree)
            .unwrap_or_default()
    }

    pub fn is_zero(&self) -> bool {
        self.entries
            .iter()
            .all(|row| row.iter().all(QSeries::<C>::is_zero))
    }

    pub fn is_structurally_zero(&self) -> bool {
        self.entries
            .iter()
            .all(|row| row.iter().all(QSeries::<C>::is_structurally_zero))
    }
}

impl<C: Coeff + fmt::Display> fmt::Display for QSeries<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        for (degree, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            if degree == 0 {
                parts.push(format!("{coeff}"));
            } else if degree == 1 {
                parts.push(format!("({coeff})*q"));
            } else {
                parts.push(format!("({coeff})*q^{degree}"));
            }
        }
        if parts.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", parts.join(" + "))
        }
    }
}

/// Formally integrate `q d/dq` of a power series, taking the integration
/// constant to be zero.
///
/// This inverts the `q`-derivation on the positive-degree part; it errors if the
/// constant term is nonzero, since no zero integration constant can absorb it.
/// Shared by the ordinary and twisted calibration paths.
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

/// Apply [`integrate_q_derivative_zero_constant`] entrywise to a series matrix.
pub(crate) fn integrate_q_derivative_zero_constant_matrix(
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

// --- Generic formal power-series utilities over coefficient lists `Vec<C>` ---
//
// These are theory-agnostic operations on truncated power series represented as
// coefficient vectors indexed by degree. They are generic over any [`Coeff`] and
// are shared by the calibration paths.

/// Formal exponential of a power series, assuming a zero constant term in the
/// exponent (so the result has constant term one).
pub(crate) fn exp_series<C: Coeff>(series: &[C], max_degree: usize) -> Vec<C> {
    let mut out = vec![C::zero(); max_degree + 1];
    out[0] = C::one();
    for degree in 1..=max_degree {
        let mut sum = C::zero();
        for split in 1..=degree {
            let coeff = series.get(split).cloned().unwrap_or_else(C::zero);
            let scaled_coeff = C::from_usize(split).mul(&coeff);
            fused::add_product_assign(&mut sum, &scaled_coeff, &out[degree - split]);
        }
        out[degree] = sum.div(&C::from_usize(degree));
    }
    out
}

/// Truncated product of two power series up to `max_degree`.
pub(crate) fn mul_plain_series<C: Coeff>(left: &[C], right: &[C], max_degree: usize) -> Vec<C> {
    let mut out = vec![C::zero(); max_degree + 1];
    for left_degree in 0..=max_degree {
        if left[left_degree].is_zero() {
            continue;
        }
        for right_degree in 0..=max_degree - left_degree {
            if right[right_degree].is_zero() {
                continue;
            }
            fused::add_product_assign(
                &mut out[left_degree + right_degree],
                &left[left_degree],
                &right[right_degree],
            );
        }
    }
    out
}

/// Truncated composition `series(input)` up to `max_degree`, where `input` has
/// zero constant term.
pub(crate) fn compose_plain_series<C: Coeff>(
    series: &[C],
    input: &[C],
    max_degree: usize,
) -> Vec<C> {
    let mut out = vec![C::zero(); max_degree + 1];
    let mut power = vec![C::zero(); max_degree + 1];
    power[0] = C::one();
    for degree in 0..=max_degree {
        let coefficient = series.get(degree).cloned().unwrap_or_else(C::zero);
        if !coefficient.is_zero() {
            for idx in 0..=max_degree {
                fused::add_product_assign(&mut out[idx], &coefficient, &power[idx]);
            }
        }
        power = mul_plain_series(&power, input, max_degree);
    }
    out
}

/// Compositional inverse of a series with zero constant term and linear
/// coefficient one.
pub(crate) fn invert_series_with_linear_term_one<C: Coeff>(
    series: &[C],
    max_degree: usize,
) -> Vec<C> {
    assert_eq!(series.first(), Some(&C::zero()));
    if max_degree == 0 {
        return vec![C::zero()];
    }
    assert_eq!(series.get(1), Some(&C::one()));
    let mut inverse = vec![C::zero(); max_degree + 1];
    inverse[1] = C::one();
    for degree in 2..=max_degree {
        // With series = z + O(z^2) and input = z + ..., the coefficient of
        // z^degree in series(input) depends on input[degree] with derivative
        // exactly one: the linear term contributes input[degree] directly, and
        // every higher power of the input reaches z^degree only through
        // lower-order input coefficients.  So the still-unset inverse[degree]
        // must cancel the currently composed coefficient.
        let current = compose_plain_series(series, &inverse, degree)[degree].clone();
        inverse[degree] = current.neg();
    }
    inverse
}

/// Invert a mirror map `q(Q) = Q * exp(mirror(Q))` to recover `Q(q)`.
pub(crate) fn invert_mirror_map<C: Coeff>(mirror: &[C], q_degree: usize) -> Vec<C> {
    let exp_mirror = exp_series(mirror, q_degree);
    let mut q_of_q = vec![C::zero(); q_degree + 1];
    if q_degree >= 1 {
        q_of_q[1] = C::one();
    }
    let target = mul_plain_series(&q_of_q, &exp_mirror, q_degree);
    invert_series_with_linear_term_one(&target, q_degree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algebra::Rational;

    #[test]
    #[should_panic(expected = "q-series must contain at least its constant coefficient")]
    fn q_series_rejects_empty_coefficient_storage() {
        let _ = QSeries::<RatFun>::from_coeffs(Vec::new());
    }

    #[test]
    #[should_panic(expected = "series-matrix entries must have equal q-series truncations")]
    fn series_matrix_rejects_mixed_entry_truncations() {
        let _ =
            SeriesMatrix::<RatFun>::from_entries(vec![vec![QSeries::zero(0), QSeries::zero(1)]]);
    }

    #[test]
    #[should_panic(
        expected = "series-matrix diagonal entries must have equal q-series truncations"
    )]
    fn diagonal_series_matrix_rejects_mixed_truncations() {
        let _ = SeriesMatrix::<RatFun>::diagonal(vec![QSeries::zero(0), QSeries::zero(1)]);
    }

    #[test]
    fn inverse_multiplies_to_one() {
        let max_degree = 3;
        let one = QSeries::one(max_degree);
        let s = one.add(&QSeries::q(max_degree));
        let inv = s.inverse().unwrap();
        let product = s.mul(&inv);
        assert_eq!(product.coeff(0), Some(&RatFun::one()));
        assert_eq!(product.coeff(1), Some(&RatFun::zero()));
        assert_eq!(product.coeff(2), Some(&RatFun::zero()));
        assert_eq!(product.coeff(3), Some(&RatFun::zero()));
    }

    #[test]
    fn multiplication_truncates() {
        let q = QSeries::q(2);
        let two = QSeries::constant(RatFun::from_rational(Rational::from(2)), 2);
        let s = q.add(&two);
        let square = s.mul(&s);
        assert_eq!(square.coeff(0).unwrap().to_string(), "4");
        assert_eq!(square.coeff(1).unwrap().to_string(), "4");
        assert_eq!(square.coeff(2).unwrap().to_string(), "1");
    }

    #[test]
    fn sqrt_and_q_derivative_work_for_unit_series() {
        let max_degree = 2;
        let one_plus_q = QSeries::one(max_degree).add(&QSeries::q(max_degree));
        let sqrt = one_plus_q.sqrt_with_constant_one().unwrap();
        assert_eq!(sqrt.coeff(0), Some(&RatFun::one()));
        assert_eq!(sqrt.coeff(1).unwrap().to_string(), "1/2");
        assert_eq!(sqrt.coeff(2).unwrap().to_string(), "-1/8");
        assert_eq!(one_plus_q.q_derivative().coeff(1), Some(&RatFun::one()));
    }

    #[test]
    fn rational_q_series_uses_plain_rational_coefficients() {
        let max_degree = 3;
        let one = RationalQSeries::one(max_degree);
        let q = RationalQSeries::q(max_degree);
        let two = RationalQSeries::constant(Rational::from(2usize), max_degree);
        let series = one.add(&q).mul(&two);

        assert_eq!(series.coeff(0), Some(&Rational::from(2usize)));
        assert_eq!(series.coeff(1), Some(&Rational::from(2usize)));
        assert_eq!(series.coeff(2), Some(&Rational::zero()));

        let inv = one.add(&q).inverse().unwrap();
        assert_eq!(inv.coeff(0), Some(&Rational::one()));
        assert_eq!(inv.coeff(1), Some(&Rational::from(-1)));
        assert_eq!(inv.coeff(2), Some(&Rational::one()));
        assert_eq!(inv.coeff(3), Some(&Rational::from(-1)));
    }

    #[test]
    fn series_inversion_round_trips_through_composition() {
        let max_degree = 6;
        let mut series = vec![RatFun::zero(); max_degree + 1];
        for (degree, coeff) in series.iter_mut().enumerate().skip(1) {
            *coeff = RatFun::from(degree);
        }
        let inverse = invert_series_with_linear_term_one(&series, max_degree);
        let composed = compose_plain_series(&series, &inverse, max_degree);
        for (degree, coeff) in composed.iter().enumerate() {
            let expected = if degree == 1 {
                RatFun::one()
            } else {
                RatFun::zero()
            };
            assert_eq!(*coeff, expected, "composition defect at degree {degree}");
        }
    }

    #[test]
    fn series_inversion_accepts_degree_zero_truncation() {
        assert_eq!(
            invert_series_with_linear_term_one(&[RatFun::zero()], 0),
            vec![RatFun::zero()]
        );
        assert_eq!(
            invert_mirror_map(&[RatFun::zero()], 0),
            vec![RatFun::zero()]
        );
    }

    #[test]
    fn matrix_multiplication_works_over_q_series() {
        let q = QSeries::<RatFun>::q(1);
        let one = QSeries::<RatFun>::one(1);
        let a = SeriesMatrix::from_entries(vec![
            vec![one.clone(), q.clone()],
            vec![QSeries::<RatFun>::zero(1), one.clone()],
        ]);
        let product = a.mul(&SeriesMatrix::identity(2, 1));
        assert_eq!(product, a);
        assert_eq!(a.transpose().rows(), 2);
        assert!(SeriesMatrix::<RatFun>::zero(2, 2, 1).is_zero());
        assert_eq!(a.add(&a.neg()), SeriesMatrix::<RatFun>::zero(2, 2, 1));
    }

    #[test]
    fn rational_series_matrix_multiplication_works() {
        let one = RationalQSeries::one(1);
        let q = RationalQSeries::q(1);
        let matrix = RationalSeriesMatrix::from_entries(vec![
            vec![one.clone(), q.clone()],
            vec![RationalQSeries::zero(1), one],
        ]);
        let product = matrix.mul(&RationalSeriesMatrix::identity(2, 1));
        assert_eq!(product, matrix);
    }
}
