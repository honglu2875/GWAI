use crate::algebra::{Coeff, RatFun, Rational};
use crate::error::GwError;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QSeries<C = RatFun> {
    coeffs: Vec<C>,
}

pub type RationalQSeries = QSeries<Rational>;

impl<C: Coeff> QSeries<C> {
    pub fn from_coeffs(coeffs: Vec<C>) -> Self {
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
        self.assert_same_truncation(rhs);
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(rhs.coeffs.iter())
                .map(|(a, b)| a.add(b))
                .collect(),
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
                let term = self.coeffs[i].mul(&rhs.coeffs[j]);
                out[i + j] = out[i + j].add(&term);
            }
        }
        Self { coeffs: out }
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
                let term = self.coeffs[k].mul(&out[degree - k]);
                sum = sum.add(&term);
            }
            out[degree] = sum.div(constant).neg();
        }
        Ok(Self { coeffs: out })
    }

    pub fn div(&self, rhs: &Self) -> Result<Self, GwError> {
        Ok(self.mul(&rhs.inverse()?))
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut out = QSeries::<C>::one(self.max_degree());
        for _ in 0..exp {
            out = out.mul(self);
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
                let term = out[left].mul(&out[degree - left]);
                quadratic_terms = quadratic_terms.add(&term);
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
        assert!(entries.iter().all(|row| row.len() == cols));
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
                        .map(|(left, right)| left.add(right))
                        .collect()
                })
                .collect(),
        )
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
                    total = total.add(&self.entries[row][k].mul(&rhs.entries[k][col]));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Rational;

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
