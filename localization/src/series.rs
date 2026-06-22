use crate::algebra::RatFun;
use crate::error::GwError;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QSeries {
    coeffs: Vec<RatFun>,
}

impl QSeries {
    pub fn from_coeffs(coeffs: Vec<RatFun>) -> Self {
        Self { coeffs }
    }

    pub fn zero(max_degree: usize) -> Self {
        Self {
            coeffs: vec![RatFun::zero(); max_degree + 1],
        }
    }

    pub fn one(max_degree: usize) -> Self {
        Self::constant(RatFun::one(), max_degree)
    }

    pub fn constant(value: RatFun, max_degree: usize) -> Self {
        let mut out = Self::zero(max_degree);
        out.coeffs[0] = value;
        out
    }

    pub fn q(max_degree: usize) -> Self {
        let mut out = Self::zero(max_degree);
        if max_degree >= 1 {
            out.coeffs[1] = RatFun::one();
        }
        out
    }

    pub fn max_degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    pub fn coeff(&self, degree: usize) -> Option<&RatFun> {
        self.coeffs.get(degree)
    }

    pub fn coeffs(&self) -> &[RatFun] {
        &self.coeffs
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(RatFun::is_zero)
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.assert_same_truncation(rhs);
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(rhs.coeffs.iter())
                .map(|(a, b)| a + b)
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
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    pub fn neg(&self) -> Self {
        Self {
            coeffs: self.coeffs.iter().cloned().map(|c| -c).collect(),
        }
    }

    pub fn scale(&self, scalar: &RatFun) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|coeff| coeff * scalar).collect(),
        }
    }

    pub fn q_derivative(&self) -> Self {
        let coeffs = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(degree, coeff)| {
                if degree == 0 {
                    RatFun::zero()
                } else {
                    coeff * &RatFun::from(degree)
                }
            })
            .collect();
        Self { coeffs }
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        self.assert_same_truncation(rhs);
        let max_degree = self.max_degree();
        let mut out = vec![RatFun::zero(); max_degree + 1];
        for i in 0..=max_degree {
            if self.coeffs[i].is_zero() {
                continue;
            }
            for j in 0..=max_degree - i {
                if rhs.coeffs[j].is_zero() {
                    continue;
                }
                let term = &self.coeffs[i] * &rhs.coeffs[j];
                out[i + j] = &out[i + j] + &term;
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

        let mut out = vec![RatFun::zero(); max_degree + 1];
        out[0] = &RatFun::one() / constant;
        for degree in 1..=max_degree {
            let mut sum = RatFun::zero();
            for k in 1..=degree {
                let term = &self.coeffs[k] * &out[degree - k];
                sum = &sum + &term;
            }
            out[degree] = -(&sum / constant);
        }
        Ok(Self { coeffs: out })
    }

    pub fn div(&self, rhs: &Self) -> Result<Self, GwError> {
        Ok(self.mul(&rhs.inverse()?))
    }

    pub fn pow_usize(&self, exp: usize) -> Self {
        let mut out = QSeries::one(self.max_degree());
        for _ in 0..exp {
            out = out.mul(self);
        }
        out
    }

    pub fn sqrt_with_constant_one(&self) -> Result<Self, GwError> {
        let max_degree = self.max_degree();
        if self.coeff(0) != Some(&RatFun::one()) {
            return Err(GwError::AlgebraFailure(
                "sqrt_with_constant_one requires constant coefficient 1".to_string(),
            ));
        }

        let mut out = vec![RatFun::zero(); max_degree + 1];
        out[0] = RatFun::one();
        for degree in 1..=max_degree {
            let mut quadratic_terms = RatFun::zero();
            for left in 1..degree {
                let term = &out[left] * &out[degree - left];
                quadratic_terms = &quadratic_terms + &term;
            }
            let numerator = self.coeffs[degree].clone() - quadratic_terms;
            out[degree] = &numerator / &RatFun::from(2usize);
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
pub struct SeriesMatrix {
    rows: usize,
    cols: usize,
    entries: Vec<Vec<QSeries>>,
}

impl SeriesMatrix {
    pub fn zero(rows: usize, cols: usize, max_degree: usize) -> Self {
        Self {
            rows,
            cols,
            entries: vec![vec![QSeries::zero(max_degree); cols]; rows],
        }
    }

    pub fn identity(size: usize, max_degree: usize) -> Self {
        let mut out = Self::zero(size, size, max_degree);
        for i in 0..size {
            out.entries[i][i] = QSeries::one(max_degree);
        }
        out
    }

    pub fn from_entries(entries: Vec<Vec<QSeries>>) -> Self {
        let rows = entries.len();
        let cols = entries.first().map(Vec::len).unwrap_or_default();
        assert!(entries.iter().all(|row| row.len() == cols));
        Self {
            rows,
            cols,
            entries,
        }
    }

    pub fn constant(entries: Vec<Vec<RatFun>>, max_degree: usize) -> Self {
        Self::from_entries(
            entries
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| QSeries::constant(entry, max_degree))
                        .collect()
                })
                .collect(),
        )
    }

    pub fn diagonal(diagonal: Vec<QSeries>) -> Self {
        let size = diagonal.len();
        let max_degree = diagonal
            .first()
            .map(QSeries::max_degree)
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

    pub fn entry(&self, row: usize, col: usize) -> &QSeries {
        &self.entries[row][col]
    }

    pub fn entries(&self) -> &[Vec<QSeries>] {
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
                .map(|row| row.iter().map(QSeries::q_derivative).collect())
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
                .map(|row| row.iter().map(QSeries::neg).collect())
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
                let mut total = QSeries::zero(max_degree);
                for k in 0..self.cols {
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
            .map(QSeries::max_degree)
            .unwrap_or_default()
    }

    pub fn is_zero(&self) -> bool {
        self.entries
            .iter()
            .all(|row| row.iter().all(QSeries::is_zero))
    }
}

impl fmt::Display for QSeries {
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
    fn matrix_multiplication_works_over_q_series() {
        let q = QSeries::q(1);
        let one = QSeries::one(1);
        let a = SeriesMatrix::from_entries(vec![
            vec![one.clone(), q.clone()],
            vec![QSeries::zero(1), one.clone()],
        ]);
        let product = a.mul(&SeriesMatrix::identity(2, 1));
        assert_eq!(product, a);
        assert_eq!(a.transpose().rows(), 2);
        assert!(SeriesMatrix::zero(2, 2, 1).is_zero());
        assert_eq!(a.add(&a.neg()), SeriesMatrix::zero(2, 2, 1));
    }
}
