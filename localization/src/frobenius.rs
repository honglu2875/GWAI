//! Frobenius algebra data for equivariant quantum cohomology of `P^n`.
//!
//! This module implements the target-specific input used by the projective
//! Givental calibration.  The core algebra is
//! `QH_T(P^n) = Q(lambda)[[q]][H] / (prod_i(H-lambda_i)-q)`.
//! Canonical coordinates are the roots of `prod_i(x-lambda_i)=q`; idempotents
//! are the corresponding Lagrange interpolation projectors.

use crate::algebra::{lambda, RatFun};
use crate::error::GwError;
use crate::geometry::{CohomologyClass, EquivariantProjectiveSpace};
use crate::series::{QSeries, SeriesMatrix};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductKind {
    ClassicalEquivariant,
    QuantumEquivariant,
}

/// Classical or small-quantum Frobenius algebra of equivariant `P^n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrobeniusData {
    pub target: EquivariantProjectiveSpace,
    pub product: ProductKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassicalCanonicalData {
    pub idempotents: Vec<CohomologyClass>,
    pub metric_norms: Vec<RatFun>,
    pub inverse_metric_norms: Vec<RatFun>,
    /// Columns are unnormalized fixed-point idempotents in the hyperplane basis.
    pub transition_to_flat: Vec<Vec<RatFun>>,
}

/// A cohomology class with each hyperplane-basis coefficient expanded as a
/// truncated Novikov `q`-series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesCohomologyClass {
    pub n: usize,
    pub coeffs: Vec<QSeries>,
}

impl SeriesCohomologyClass {
    pub fn zero(n: usize, max_q_degree: usize) -> Self {
        Self {
            n,
            coeffs: vec![QSeries::zero(max_q_degree); n + 1],
        }
    }

    pub fn one(n: usize, max_q_degree: usize) -> Self {
        let mut out = Self::zero(n, max_q_degree);
        out.coeffs[0] = QSeries::one(max_q_degree);
        out
    }

    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.n, rhs.n);
        Self {
            n: self.n,
            coeffs: self
                .coeffs
                .iter()
                .zip(rhs.coeffs.iter())
                .map(|(a, b)| a.add(b))
                .collect(),
        }
    }

    pub fn multiply_quantum(&self, rhs: &Self) -> Self {
        // Multiply in Q[H][[q]] and reduce by the quantum relation
        // prod_i(H-lambda_i)=q.
        assert_eq!(self.n, rhs.n);
        let max_q_degree = self.max_q_degree();
        let mut product = vec![QSeries::zero(max_q_degree); 2 * self.n + 1];
        for (left_power, left_coeff) in self.coeffs.iter().enumerate() {
            if left_coeff.is_zero() {
                continue;
            }
            for (right_power, right_coeff) in rhs.coeffs.iter().enumerate() {
                if right_coeff.is_zero() {
                    continue;
                }
                let term = left_coeff.mul(right_coeff);
                product[left_power + right_power] = product[left_power + right_power].add(&term);
            }
        }
        let coeffs = reduce_series_h_polynomial(self.n, product);
        Self { n: self.n, coeffs }
    }

    pub fn max_q_degree(&self) -> usize {
        self.coeffs
            .first()
            .map(QSeries::max_degree)
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantumCanonicalData {
    pub roots: Vec<QSeries>,
    pub idempotents: Vec<SeriesCohomologyClass>,
    pub metric_norms: Vec<QSeries>,
    pub inverse_metric_norms: Vec<QSeries>,
    /// Columns are unnormalized quantum idempotents in the hyperplane basis.
    pub transition_to_flat: Vec<Vec<QSeries>>,
}

impl FrobeniusData {
    pub fn classical(n: usize) -> Self {
        Self {
            target: EquivariantProjectiveSpace::new(n),
            product: ProductKind::ClassicalEquivariant,
        }
    }

    pub fn quantum(n: usize) -> Self {
        Self {
            target: EquivariantProjectiveSpace::new(n),
            product: ProductKind::QuantumEquivariant,
        }
    }

    pub fn multiply(&self, left: &CohomologyClass, right: &CohomologyClass) -> CohomologyClass {
        match self.product {
            ProductKind::ClassicalEquivariant => left.multiply_classical_equivariant(right),
            ProductKind::QuantumEquivariant => left.multiply_quantum_equivariant(right),
        }
    }

    pub fn multiplication_by_h_matrix(&self) -> Vec<Vec<RatFun>> {
        // Companion matrix for multiplication by H in the hyperplane basis.
        // The last column is exactly the reduction of H^{n+1}.
        let n = self.target.n;
        let mut matrix = vec![vec![RatFun::zero(); n + 1]; n + 1];
        for col in 0..n {
            matrix[col + 1][col] = RatFun::one();
        }
        let relation = match self.product {
            ProductKind::ClassicalEquivariant => self.target.classical_relation_rhs(),
            ProductKind::QuantumEquivariant => self.target.quantum_relation_rhs(),
        };
        for row in 0..=n {
            matrix[row][n] = relation[row].clone();
        }
        matrix
    }

    pub fn flat_metric_matrix(&self) -> Vec<Vec<RatFun>> {
        let size = self.target.n + 1;
        let mut matrix = vec![vec![RatFun::zero(); size]; size];
        for row in 0..size {
            for col in 0..size {
                matrix[row][col] = self.target.pairing(
                    &CohomologyClass::h_power(self.target.n, row),
                    &CohomologyClass::h_power(self.target.n, col),
                );
            }
        }
        matrix
    }

    pub fn identity_decomposition_at_q_zero(&self) -> CohomologyClass {
        let mut total = CohomologyClass::zero(self.target.n);
        for fixed_point in 0..=self.target.n {
            total = total.add(&self.target.fixed_point_idempotent(fixed_point));
        }
        total
    }

    pub fn canonical_root_series(&self, max_q_degree: usize) -> Result<Vec<QSeries>, GwError> {
        (0..=self.target.n)
            .map(|branch| canonical_root_series(self.target.n, branch, max_q_degree))
            .collect()
    }

    pub fn classical_canonical_data(&self) -> ClassicalCanonicalData {
        let idempotents = (0..=self.target.n)
            .map(|fixed_point| self.target.fixed_point_idempotent(fixed_point))
            .collect::<Vec<_>>();
        let metric_norms = idempotents
            .iter()
            .map(|idempotent| self.target.pairing(idempotent, idempotent))
            .collect::<Vec<_>>();
        let inverse_metric_norms = (0..=self.target.n)
            .map(|fixed_point| self.target.fixed_point_euler(fixed_point))
            .collect::<Vec<_>>();
        let mut transition_to_flat =
            vec![vec![RatFun::zero(); self.target.n + 1]; self.target.n + 1];
        for (col, idempotent) in idempotents.iter().enumerate() {
            for (row, coeff) in idempotent.coeffs().iter().enumerate() {
                transition_to_flat[row][col] = coeff.clone();
            }
        }
        ClassicalCanonicalData {
            idempotents,
            metric_norms,
            inverse_metric_norms,
            transition_to_flat,
        }
    }

    pub fn quantum_canonical_data(
        &self,
        max_q_degree: usize,
    ) -> Result<QuantumCanonicalData, GwError> {
        // For each branch u_i(q), build the idempotent
        // prod_{j != i} (H-u_j)/(u_i-u_j).  The denominator is P'(u_i), which
        // is also the inverse metric norm in the unnormalized canonical frame.
        let roots = self.canonical_root_series(max_q_degree)?;
        let mut idempotents = Vec::with_capacity(self.target.n + 1);
        let mut inverse_metric_norms = Vec::with_capacity(self.target.n + 1);
        let mut metric_norms = Vec::with_capacity(self.target.n + 1);

        for branch in 0..=self.target.n {
            let mut numerator = vec![QSeries::one(max_q_degree)];
            let mut denominator = QSeries::one(max_q_degree);
            for other in 0..=self.target.n {
                if other == branch {
                    continue;
                }
                numerator = multiply_series_polynomial_by_linear(
                    &numerator,
                    &roots[other].neg(),
                    max_q_degree,
                );
                denominator = denominator.mul(&roots[branch].sub(&roots[other]));
            }
            let denominator_inv = denominator.inverse()?;
            let coeffs = numerator
                .into_iter()
                .map(|coeff| coeff.mul(&denominator_inv))
                .collect::<Vec<_>>();
            idempotents.push(SeriesCohomologyClass {
                n: self.target.n,
                coeffs,
            });
            metric_norms.push(denominator.inverse()?);
            inverse_metric_norms.push(denominator);
        }

        let mut transition_to_flat =
            vec![vec![QSeries::zero(max_q_degree); self.target.n + 1]; self.target.n + 1];
        for (col, idempotent) in idempotents.iter().enumerate() {
            for (row, coeff) in idempotent.coeffs.iter().enumerate() {
                transition_to_flat[row][col] = coeff.clone();
            }
        }

        Ok(QuantumCanonicalData {
            roots,
            idempotents,
            metric_norms,
            inverse_metric_norms,
            transition_to_flat,
        })
    }
}

impl QuantumCanonicalData {
    pub fn transition_matrix(&self) -> SeriesMatrix {
        SeriesMatrix::from_entries(self.transition_to_flat.clone())
    }

    pub fn metric_norm_matrix(&self) -> SeriesMatrix {
        SeriesMatrix::diagonal(self.metric_norms.clone())
    }

    pub fn canonical_metric_from_transition(&self, flat_metric: &SeriesMatrix) -> SeriesMatrix {
        let transition = self.transition_matrix();
        transition.transpose().mul(flat_metric).mul(&transition)
    }
}

fn multiply_series_polynomial_by_linear(
    poly: &[QSeries],
    constant: &QSeries,
    max_q_degree: usize,
) -> Vec<QSeries> {
    let mut out = vec![QSeries::zero(max_q_degree); poly.len() + 1];
    for (degree, coeff) in poly.iter().enumerate() {
        out[degree] = out[degree].add(&coeff.mul(constant));
        out[degree + 1] = out[degree + 1].add(coeff);
    }
    out
}

fn reduce_series_h_polynomial(n: usize, mut coeffs: Vec<QSeries>) -> Vec<QSeries> {
    // Reduces a polynomial in H to the basis 1,H,...,H^n using the quantum
    // relation.  This is the series analogue of geometry::reduce_h_polynomial.
    let max_q_degree = coeffs.first().map(QSeries::max_degree).unwrap_or_default();
    let relation = series_h_power_relation_rhs(n, max_q_degree);
    while coeffs.len() > n + 1 {
        let degree = coeffs.len() - 1;
        let leading = coeffs.pop().unwrap();
        if leading.is_zero() {
            continue;
        }
        let shift = degree - (n + 1);
        for (power, relation_coeff) in relation.iter().enumerate() {
            if relation_coeff.is_zero() {
                continue;
            }
            let term = leading.mul(relation_coeff);
            coeffs[shift + power] = coeffs[shift + power].add(&term);
        }
    }
    coeffs.resize(n + 1, QSeries::zero(max_q_degree));
    coeffs
}

fn series_h_power_relation_rhs(n: usize, max_q_degree: usize) -> Vec<QSeries> {
    let mut elementary = vec![RatFun::zero(); n + 2];
    elementary[0] = RatFun::one();
    for i in 0..=n {
        let li = lambda(i);
        for k in (1..=i + 1).rev() {
            let term = &elementary[k - 1] * &li;
            elementary[k] = &elementary[k] + &term;
        }
    }

    let mut rhs = vec![QSeries::zero(max_q_degree); n + 1];
    for k in 1..=n + 1 {
        let power = n + 1 - k;
        let signed = if k % 2 == 1 {
            elementary[k].clone()
        } else {
            -elementary[k].clone()
        };
        rhs[power] = rhs[power].add(&QSeries::constant(signed, max_q_degree));
    }
    rhs[0] = rhs[0].add(&QSeries::q(max_q_degree));
    rhs
}

pub fn canonical_root_series(
    n: usize,
    branch: usize,
    max_q_degree: usize,
) -> Result<QSeries, GwError> {
    // Newton iteration in the complete local ring at the classical root
    // lambda_branch.  The roots are formal series u_i(q) satisfying P(u_i)=q.
    if branch > n {
        return Err(GwError::AlgebraFailure(format!(
            "root branch {branch} out of range for P^{n}"
        )));
    }
    let mut root = QSeries::constant(lambda(branch), max_q_degree);
    for _ in 0..=max_q_degree {
        let p = characteristic_series(n, &root).sub(&QSeries::q(max_q_degree));
        if p.coeffs().iter().all(|coeff| coeff.is_zero()) {
            break;
        }
        let dp = characteristic_derivative_series(n, &root);
        root = root.sub(&p.div(&dp)?);
    }
    Ok(root)
}

pub fn characteristic_series(n: usize, x: &QSeries) -> QSeries {
    let max_q_degree = x.max_degree();
    let mut product = QSeries::one(max_q_degree);
    for j in 0..=n {
        product = product.mul(&x.sub(&QSeries::constant(lambda(j), max_q_degree)));
    }
    product
}

pub fn characteristic_derivative_series(n: usize, x: &QSeries) -> QSeries {
    let max_q_degree = x.max_degree();
    let mut total = QSeries::zero(max_q_degree);
    for omitted in 0..=n {
        let mut product = QSeries::one(max_q_degree);
        for j in 0..=n {
            if j == omitted {
                continue;
            }
            product = product.mul(&x.sub(&QSeries::constant(lambda(j), max_q_degree)));
        }
        total = total.add(&product);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::q;

    #[test]
    fn companion_matrix_has_shift_columns() {
        let frob = FrobeniusData::quantum(2);
        let matrix = frob.multiplication_by_h_matrix();
        assert_eq!(matrix[1][0], RatFun::one());
        assert_eq!(matrix[2][1], RatFun::one());
        assert_eq!(matrix[0][2].to_string(), "lambda_0*lambda_1*lambda_2 + q");
    }

    #[test]
    fn quantum_h_product_matches_relation() {
        let frob = FrobeniusData::quantum(1);
        let h = CohomologyClass::h_power(1, 1);
        let h2 = frob.multiply(&h, &h);
        assert_eq!(h2.coeffs()[0].to_string(), "-lambda_0*lambda_1 + q");
        assert_eq!(h2.coeffs()[1].to_string(), "lambda_0 + lambda_1");
        assert!(h2.restrict_to_fixed_point(0) != q());
    }

    #[test]
    fn fixed_point_idempotents_sum_to_identity() {
        let frob = FrobeniusData::classical(2);
        assert_eq!(
            frob.identity_decomposition_at_q_zero(),
            CohomologyClass::one(2)
        );
    }

    #[test]
    fn root_series_first_coefficient_is_inverse_euler_weight() {
        let frob = FrobeniusData::quantum(2);
        let roots = frob.canonical_root_series(2).unwrap();
        for (branch, root) in roots.iter().enumerate() {
            assert_eq!(root.coeff(0), Some(&lambda(branch)));
            let expected = &RatFun::one() / &frob.target.fixed_point_euler(branch);
            assert_eq!(root.coeff(1), Some(&expected));
        }
    }

    #[test]
    fn root_series_solves_characteristic_equation_to_truncation() {
        let root = canonical_root_series(1, 0, 3).unwrap();
        let residual = characteristic_series(1, &root).sub(&QSeries::q(3));
        assert!(residual.coeffs().iter().all(|coeff| coeff.is_zero()));
    }

    #[test]
    fn classical_canonical_data_has_diagonal_norms() {
        let frob = FrobeniusData::classical(2);
        let data = frob.classical_canonical_data();
        assert_eq!(data.idempotents.len(), 3);
        for i in 0..=2 {
            assert_eq!(
                data.metric_norms[i],
                &RatFun::one() / &frob.target.fixed_point_euler(i)
            );
            assert_eq!(
                data.inverse_metric_norms[i],
                frob.target.fixed_point_euler(i)
            );
            for j in 0..=2 {
                let expected = if i == j {
                    RatFun::one()
                } else {
                    RatFun::zero()
                };
                assert_eq!(data.idempotents[i].restrict_to_fixed_point(j), expected);
            }
        }
    }

    #[test]
    fn classical_transition_columns_are_idempotent_coefficients() {
        let frob = FrobeniusData::classical(1);
        let data = frob.classical_canonical_data();
        for col in 0..=1 {
            for row in 0..=1 {
                assert_eq!(
                    data.transition_to_flat[row][col],
                    data.idempotents[col].coeffs()[row]
                );
            }
        }
    }

    #[test]
    fn quantum_idempotents_sum_to_identity() {
        let frob = FrobeniusData::quantum(1);
        let data = frob.quantum_canonical_data(1).unwrap();
        let sum = data
            .idempotents
            .iter()
            .fold(SeriesCohomologyClass::zero(1, 1), |acc, idempotent| {
                acc.add(idempotent)
            });
        assert_eq!(sum, SeriesCohomologyClass::one(1, 1));
    }

    #[test]
    fn quantum_idempotents_multiply_diagonally() {
        let frob = FrobeniusData::quantum(1);
        let data = frob.quantum_canonical_data(1).unwrap();
        for i in 0..=1 {
            for j in 0..=1 {
                let product = data.idempotents[i].multiply_quantum(&data.idempotents[j]);
                if i == j {
                    assert_series_class_equal_after_lambda_eval(
                        &product,
                        &data.idempotents[i],
                        &[RatFun::from(2usize), RatFun::from(5usize)],
                    );
                } else {
                    assert_series_class_equal_after_lambda_eval(
                        &product,
                        &SeriesCohomologyClass::zero(1, 1),
                        &[RatFun::from(2usize), RatFun::from(5usize)],
                    );
                }
            }
        }
    }

    #[test]
    fn quantum_delta_specializes_to_classical_euler_weight() {
        let frob = FrobeniusData::quantum(1);
        let data = frob.quantum_canonical_data(1).unwrap();
        for i in 0..=1 {
            assert_eq!(
                data.inverse_metric_norms[i].coeff(0),
                Some(&frob.target.fixed_point_euler(i))
            );
            assert_eq!(
                data.metric_norms[i].coeff(0),
                Some(&(&RatFun::one() / &frob.target.fixed_point_euler(i)))
            );
        }
    }

    #[test]
    fn quantum_transition_q_zero_matches_classical_transition() {
        let frob = FrobeniusData::quantum(1);
        let quantum = frob.quantum_canonical_data(2).unwrap();
        let classical = FrobeniusData::classical(1).classical_canonical_data();
        for row in 0..=1 {
            for col in 0..=1 {
                assert_eq!(
                    quantum.transition_to_flat[row][col].coeff(0),
                    Some(&classical.transition_to_flat[row][col])
                );
            }
        }
    }

    #[test]
    fn quantum_transition_diagonalizes_flat_metric() {
        let frob = FrobeniusData::quantum(1);
        let data = frob.quantum_canonical_data(1).unwrap();
        let flat_metric = SeriesMatrix::constant(frob.flat_metric_matrix(), 1);
        let actual = data.canonical_metric_from_transition(&flat_metric);
        let expected = data.metric_norm_matrix();
        assert_series_matrix_equal_after_lambda_eval(
            &actual,
            &expected,
            1,
            &[RatFun::from(2usize), RatFun::from(5usize)],
        );
    }

    fn assert_series_class_equal_after_lambda_eval(
        left: &SeriesCohomologyClass,
        right: &SeriesCohomologyClass,
        weights: &[RatFun],
    ) {
        assert_eq!(left.n, right.n);
        let rational_weights = weights
            .iter()
            .map(|weight| weight.as_rational().expect("test weights are rational"))
            .collect::<Vec<_>>();
        for (left_series, right_series) in left.coeffs.iter().zip(right.coeffs.iter()) {
            for degree in 0..=left_series.max_degree() {
                let diff = left_series.coeff(degree).unwrap() - right_series.coeff(degree).unwrap();
                let value = diff
                    .evaluate_lambda_weights(left.n, &rational_weights)
                    .expect("test specialization should avoid poles");
                assert_eq!(value, crate::algebra::Rational::zero());
            }
        }
    }

    fn assert_series_matrix_equal_after_lambda_eval(
        left: &SeriesMatrix,
        right: &SeriesMatrix,
        target_n: usize,
        weights: &[RatFun],
    ) {
        assert_eq!(left.rows(), right.rows());
        assert_eq!(left.cols(), right.cols());
        let rational_weights = weights
            .iter()
            .map(|weight| weight.as_rational().expect("test weights are rational"))
            .collect::<Vec<_>>();
        for row in 0..left.rows() {
            for col in 0..left.cols() {
                let left_series = left.entry(row, col);
                let right_series = right.entry(row, col);
                assert_eq!(left_series.max_degree(), right_series.max_degree());
                for degree in 0..=left_series.max_degree() {
                    let diff =
                        left_series.coeff(degree).unwrap() - right_series.coeff(degree).unwrap();
                    let value = diff
                        .evaluate_lambda_weights(target_n, &rational_weights)
                        .expect("test specialization should avoid poles");
                    assert_eq!(value, crate::algebra::Rational::zero());
                }
            }
        }
    }
}
