//! Givental matrix container types: the constant R-matrix and the truncated
//! series R- and S-matrices over a coefficient ring.

use super::*;

/// `q`-series valued `R(z) = 1 + R_1 z + ...` in the canonical frame.
///
/// In Givental-Teleman reconstruction this is the upper-triangular symplectic
/// loop-group calibration.  It transforms the product-of-point-theories TFT
/// into the target CohFT after the descendant/ancestor calibration has been
/// applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesRMatrix<C = RatFun> {
    pub(crate) size: usize,
    pub(crate) q_degree: usize,
    pub(crate) z_order: usize,
    pub(crate) coefficients: Vec<SeriesMatrix<C>>,
    pub(crate) calibration: CalibrationId,
    pub(crate) convention: CanonicalFrameConvention,
}

impl<C: Coeff> SeriesRMatrix<C> {
    pub fn from_coefficients(
        size: usize,
        q_degree: usize,
        z_order: usize,
        coefficients: Vec<SeriesMatrix<C>>,
        calibration: CalibrationId,
        convention: CanonicalFrameConvention,
    ) -> Result<Self, GwError> {
        if coefficients.len() != z_order + 1 {
            return Err(GwError::ConventionMismatch(format!(
                "R-matrix has {} coefficient(s), expected {}",
                coefficients.len(),
                z_order + 1
            )));
        }
        for coefficient in &coefficients {
            if coefficient.rows() != size
                || coefficient.cols() != size
                || coefficient.max_degree() != q_degree
            {
                return Err(GwError::ConventionMismatch(
                    "R-matrix coefficient shape/truncation mismatch".to_string(),
                ));
            }
        }
        Ok(Self {
            size,
            q_degree,
            z_order,
            coefficients,
            calibration,
            convention,
        })
    }

    pub fn identity(
        size: usize,
        q_degree: usize,
        z_order: usize,
        convention: CanonicalFrameConvention,
    ) -> Self {
        let mut coefficients = Vec::with_capacity(z_order + 1);
        coefficients.push(SeriesMatrix::identity(size, q_degree));
        for _ in 0..z_order {
            coefficients.push(SeriesMatrix::zero(size, size, q_degree));
        }
        Self {
            size,
            q_degree,
            z_order,
            coefficients,
            calibration: CalibrationId("identity".to_string()),
            convention,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn q_degree(&self) -> usize {
        self.q_degree
    }

    pub fn z_order(&self) -> usize {
        self.z_order
    }

    pub fn calibration(&self) -> &CalibrationId {
        &self.calibration
    }

    pub fn convention(&self) -> CanonicalFrameConvention {
        self.convention
    }

    pub fn coefficient(&self, order: usize) -> Option<&SeriesMatrix<C>> {
        self.coefficients.get(order)
    }

    pub fn coefficients(&self) -> &[SeriesMatrix<C>] {
        &self.coefficients
    }

    pub fn entry(&self, z_order: usize, row: usize, col: usize) -> Option<&QSeries<C>> {
        self.coefficient(z_order)
            .and_then(|matrix| matrix.entries().get(row))
            .and_then(|row_values| row_values.get(col))
    }

    pub fn check_identity_calibration(&self) -> Result<(), GwError> {
        if self.calibration.0 != "identity" {
            return Err(GwError::ConventionMismatch(
                "only identity calibration has a built-in coefficient check".to_string(),
            ));
        }
        for order in 0..=self.z_order {
            let expected = if order == 0 {
                SeriesMatrix::identity(self.size, self.q_degree)
            } else {
                SeriesMatrix::zero(self.size, self.size, self.q_degree)
            };
            if self.coefficients[order] != expected {
                return Err(GwError::ValidationFailure(format!(
                    "identity R-matrix has a nonstandard coefficient at z^{order}"
                )));
            }
        }
        Ok(())
    }

    /// Checks the symplectic condition `R(-z)^T eta R(z) = eta`.
    ///
    /// This is the most useful local sanity check for an `R`-matrix: if it
    /// fails, the edge propagator would not define a CohFT-compatible graph
    /// sum.
    pub fn check_unitarity(&self, metric: &SeriesMatrix<C>) -> Result<(), GwError> {
        if metric.rows() != self.size || metric.cols() != self.size {
            return Err(GwError::ConventionMismatch(format!(
                "metric shape {}x{} does not match R-matrix size {}",
                metric.rows(),
                metric.cols(),
                self.size
            )));
        }
        if metric.max_degree() != self.q_degree {
            return Err(GwError::ConventionMismatch(format!(
                "metric q-degree {} does not match R-matrix q-degree {}",
                metric.max_degree(),
                self.q_degree
            )));
        }

        for z_degree in 0..=self.z_order {
            let mut total = SeriesMatrix::zero(self.size, self.size, self.q_degree);
            for left_order in 0..=z_degree {
                let right_order = z_degree - left_order;
                let term = self.coefficients[left_order]
                    .transpose()
                    .mul(metric)
                    .mul(&self.coefficients[right_order]);
                total = if left_order % 2 == 0 {
                    total.add(&term)
                } else {
                    total.sub(&term)
                };
            }
            let expected = if z_degree == 0 {
                metric.clone()
            } else {
                SeriesMatrix::zero(self.size, self.size, self.q_degree)
            };
            if total != expected {
                return Err(GwError::ValidationFailure(format!(
                    "R(-z)^T eta R(z) failed at z^{z_degree}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesSMatrix<C = RatFun> {
    pub(crate) size: usize,
    pub(crate) q_degree: usize,
    pub(crate) z_order: usize,
    pub(crate) coefficients: Vec<SeriesMatrix<C>>,
    pub(crate) calibration: CalibrationId,
}

impl<C: Coeff> SeriesSMatrix<C> {
    pub fn from_coefficients(
        size: usize,
        q_degree: usize,
        z_order: usize,
        coefficients: Vec<SeriesMatrix<C>>,
        calibration: CalibrationId,
    ) -> Result<Self, GwError> {
        if coefficients.len() != z_order + 1 {
            return Err(GwError::ConventionMismatch(format!(
                "S-matrix has {} coefficient(s), expected {}",
                coefficients.len(),
                z_order + 1
            )));
        }
        for coefficient in &coefficients {
            if coefficient.rows() != size
                || coefficient.cols() != size
                || coefficient.max_degree() != q_degree
            {
                return Err(GwError::ConventionMismatch(
                    "S-matrix coefficient shape/truncation mismatch".to_string(),
                ));
            }
        }
        Ok(Self {
            size,
            q_degree,
            z_order,
            coefficients,
            calibration,
        })
    }

    pub fn identity(size: usize, q_degree: usize, z_order: usize) -> Self {
        let mut coefficients = Vec::with_capacity(z_order + 1);
        coefficients.push(SeriesMatrix::identity(size, q_degree));
        for _ in 0..z_order {
            coefficients.push(SeriesMatrix::zero(size, size, q_degree));
        }
        Self {
            size,
            q_degree,
            z_order,
            coefficients,
            calibration: CalibrationId("identity".to_string()),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn q_degree(&self) -> usize {
        self.q_degree
    }

    pub fn z_order(&self) -> usize {
        self.z_order
    }

    pub fn calibration(&self) -> &CalibrationId {
        &self.calibration
    }

    pub fn coefficient(&self, order: usize) -> Option<&SeriesMatrix<C>> {
        self.coefficients.get(order)
    }

    pub fn coefficients(&self) -> &[SeriesMatrix<C>] {
        &self.coefficients
    }

    pub(crate) fn truncated(&self, z_order: usize) -> Self {
        debug_assert!(z_order <= self.z_order);
        Self {
            size: self.size,
            q_degree: self.q_degree,
            z_order,
            coefficients: self.coefficients[..=z_order].to_vec(),
            calibration: self.calibration.clone(),
        }
    }
}
