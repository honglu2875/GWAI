use crate::algebra::{lambda, RatFun, Rational};
use crate::error::GwError;
use crate::frobenius::FrobeniusData;
use crate::geometry::elementary_symmetric_weights;
use crate::graphs::stable_graphs;
use crate::series::{QSeries, SeriesMatrix};
use crate::tautological::{TautologicalOracle, WittenKontsevich};
use crate::validation;
use crate::{Insertion, InvariantRequest, InvariantResult};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaConvention {
    MetricNorm,
    InverseMetricNorm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalFrameConvention {
    FlatBasis,
    UnnormalizedCanonicalIdempotents,
    RelativeNormalizedCanonicalIdempotents,
    NormalizedCanonicalIdempotents,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RMatrix {
    size: usize,
    z_order: usize,
    coefficients: Vec<Vec<Vec<RatFun>>>,
    calibration: CalibrationId,
}

impl RMatrix {
    pub fn identity(size: usize, z_order: usize) -> Self {
        let mut coefficients = vec![vec![vec![RatFun::zero(); size]; size]; z_order + 1];
        for i in 0..size {
            coefficients[0][i][i] = RatFun::one();
        }
        Self {
            size,
            z_order,
            coefficients,
            calibration: CalibrationId("identity".to_string()),
        }
    }

    pub fn coefficient(&self, order: usize, row: usize, col: usize) -> Option<&RatFun> {
        self.coefficients
            .get(order)
            .and_then(|matrix| matrix.get(row))
            .and_then(|row_values| row_values.get(col))
    }

    pub fn check_unitarity_identity_case(&self) -> Result<(), GwError> {
        if self.calibration.0 != "identity" {
            return Err(GwError::ConventionMismatch(
                "only identity calibration has a built-in check in the current scaffold"
                    .to_string(),
            ));
        }
        for order in 0..=self.z_order {
            for row in 0..self.size {
                for col in 0..self.size {
                    let expected = if order == 0 && row == col {
                        RatFun::one()
                    } else {
                        RatFun::zero()
                    };
                    if self.coefficients[order][row][col] != expected {
                        return Err(GwError::ValidationFailure(format!(
                            "identity R-matrix coefficient ({order},{row},{col}) is {}",
                            self.coefficients[order][row][col]
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesRMatrix {
    size: usize,
    q_degree: usize,
    z_order: usize,
    coefficients: Vec<SeriesMatrix>,
    calibration: CalibrationId,
    convention: CanonicalFrameConvention,
}

impl SeriesRMatrix {
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

    pub fn coefficient(&self, order: usize) -> Option<&SeriesMatrix> {
        self.coefficients.get(order)
    }

    pub fn coefficients(&self) -> &[SeriesMatrix] {
        &self.coefficients
    }

    pub fn entry(&self, z_order: usize, row: usize, col: usize) -> Option<&crate::series::QSeries> {
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

    pub fn check_unitarity(&self, metric: &SeriesMatrix) -> Result<(), GwError> {
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
pub struct SeriesSMatrix {
    size: usize,
    q_degree: usize,
    z_order: usize,
    coefficients: Vec<SeriesMatrix>,
    calibration: CalibrationId,
}

impl SeriesSMatrix {
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

    pub fn coefficient(&self, order: usize) -> Option<&SeriesMatrix> {
        self.coefficients.get(order)
    }

    pub fn coefficients(&self) -> &[SeriesMatrix] {
        &self.coefficients
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectiveSpaceJCalibration {
    pub r_matrix: SeriesRMatrix,
    pub metric: SeriesMatrix,
    pub psi: SeriesMatrix,
    pub psi_inverse: SeriesMatrix,
    pub connection: SeriesMatrix,
    pub delta: Vec<QSeries>,
    pub inverse_delta: Vec<QSeries>,
    pub relative_sqrt_delta: Vec<QSeries>,
    pub relative_sqrt_delta_inverse: Vec<QSeries>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CalibrationCacheKey {
    n: usize,
    q_degree: usize,
    z_order: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LambdaCalibrationCacheKey {
    n: usize,
    q_degree: usize,
    z_order: usize,
    weights: Vec<Rational>,
}

pub fn projective_space_j_calibration(
    n: usize,
    q_degree: usize,
    z_order: usize,
) -> Result<ProjectiveSpaceJCalibration, GwError> {
    static CACHE: OnceLock<Mutex<HashMap<CalibrationCacheKey, ProjectiveSpaceJCalibration>>> =
        OnceLock::new();
    let key = CalibrationCacheKey {
        n,
        q_degree,
        z_order,
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(calibration) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(calibration);
    }

    let frobenius = FrobeniusData::quantum(n);
    let canonical = frobenius.quantum_canonical_data(q_degree)?;
    let size = n + 1;

    let transition = canonical.transition_matrix();
    let relative_sqrt_delta = canonical
        .inverse_metric_norms
        .iter()
        .map(relative_sqrt_delta_series)
        .collect::<Result<Vec<_>, _>>()?;
    let relative_sqrt_delta_inv = relative_sqrt_delta
        .iter()
        .map(QSeries::inverse)
        .collect::<Result<Vec<_>, _>>()?;

    let relative_scale = SeriesMatrix::diagonal(relative_sqrt_delta.clone());
    let relative_scale_inv = SeriesMatrix::diagonal(relative_sqrt_delta_inv.clone());
    let evaluation = canonical_evaluation_matrix(&canonical.roots);
    let psi = transition.mul(&relative_scale);
    let psi_inverse = relative_scale_inv.mul(&evaluation);
    let connection = psi_inverse.mul(&psi.q_derivative());

    let metric = SeriesMatrix::diagonal(
        canonical
            .metric_norms
            .iter()
            .map(|norm| {
                QSeries::constant(
                    norm.coeff(0).cloned().unwrap_or_else(RatFun::zero),
                    q_degree,
                )
            })
            .collect(),
    );
    let classical_diagonal = classical_limit_diagonal_coefficients(n, z_order);
    let coefficients = solve_projective_r_coefficients(
        &canonical.roots,
        &connection,
        &metric,
        &classical_diagonal,
        q_degree,
        z_order,
    )?;

    let r_matrix = SeriesRMatrix {
        size,
        q_degree,
        z_order,
        coefficients,
        calibration: CalibrationId("projective-space-j".to_string()),
        convention: CanonicalFrameConvention::RelativeNormalizedCanonicalIdempotents,
    };

    let calibration = ProjectiveSpaceJCalibration {
        r_matrix,
        metric,
        psi,
        psi_inverse,
        connection,
        delta: canonical.inverse_metric_norms,
        inverse_delta: canonical.metric_norms,
        relative_sqrt_delta,
        relative_sqrt_delta_inverse: relative_sqrt_delta_inv,
    };
    cache.lock().unwrap().insert(key, calibration.clone());
    Ok(calibration)
}

fn projective_space_j_calibration_at_lambda_weights(
    n: usize,
    q_degree: usize,
    z_order: usize,
    weights: &[Rational],
) -> Result<ProjectiveSpaceJCalibration, GwError> {
    static CACHE: OnceLock<Mutex<HashMap<LambdaCalibrationCacheKey, ProjectiveSpaceJCalibration>>> =
        OnceLock::new();
    let key = LambdaCalibrationCacheKey {
        n,
        q_degree,
        z_order,
        weights: weights.to_vec(),
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(calibration) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(calibration);
    }

    let canonical = specialized_quantum_canonical_data(n, q_degree, weights)?;
    let size = n + 1;

    let transition = SeriesMatrix::from_entries(canonical.transition_to_flat.clone());

    let relative_sqrt_delta = canonical
        .inverse_metric_norms
        .iter()
        .map(relative_sqrt_delta_series)
        .collect::<Result<Vec<_>, _>>()?;
    let relative_sqrt_delta_inv = relative_sqrt_delta
        .iter()
        .map(QSeries::inverse)
        .collect::<Result<Vec<_>, _>>()?;

    let relative_scale = SeriesMatrix::diagonal(relative_sqrt_delta.clone());
    let relative_scale_inv = SeriesMatrix::diagonal(relative_sqrt_delta_inv.clone());
    let evaluation = canonical_evaluation_matrix(&canonical.roots);
    let psi = transition.mul(&relative_scale);
    let psi_inverse = relative_scale_inv.mul(&evaluation);
    let connection = psi_inverse.mul(&psi.q_derivative());

    let metric = SeriesMatrix::diagonal(
        canonical
            .metric_norms
            .iter()
            .map(|norm| {
                QSeries::constant(
                    norm.coeff(0).cloned().unwrap_or_else(RatFun::zero),
                    q_degree,
                )
            })
            .collect(),
    );
    let classical_diagonal =
        classical_limit_diagonal_coefficients_at_lambda_weights(n, z_order, weights);
    let coefficients = solve_projective_r_coefficients(
        &canonical.roots,
        &connection,
        &metric,
        &classical_diagonal,
        q_degree,
        z_order,
    )?;

    let r_matrix = SeriesRMatrix {
        size,
        q_degree,
        z_order,
        coefficients,
        calibration: CalibrationId("projective-space-j-lambda-line".to_string()),
        convention: CanonicalFrameConvention::RelativeNormalizedCanonicalIdempotents,
    };

    let calibration = ProjectiveSpaceJCalibration {
        r_matrix,
        metric,
        psi,
        psi_inverse,
        connection,
        delta: canonical.inverse_metric_norms,
        inverse_delta: canonical.metric_norms,
        relative_sqrt_delta,
        relative_sqrt_delta_inverse: relative_sqrt_delta_inv,
    };
    cache.lock().unwrap().insert(key, calibration.clone());
    Ok(calibration)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpecializedQuantumCanonicalData {
    roots: Vec<QSeries>,
    metric_norms: Vec<QSeries>,
    inverse_metric_norms: Vec<QSeries>,
    transition_to_flat: Vec<Vec<QSeries>>,
}

fn specialized_quantum_canonical_data(
    n: usize,
    max_q_degree: usize,
    weights: &[Rational],
) -> Result<SpecializedQuantumCanonicalData, GwError> {
    if weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} lambda weights, got {}",
            n + 1,
            weights.len()
        )));
    }

    let roots = (0..=n)
        .map(|branch| canonical_root_series_at_lambda_weights(n, branch, max_q_degree, weights))
        .collect::<Result<Vec<_>, _>>()?;
    let mut inverse_metric_norms = Vec::with_capacity(n + 1);
    let mut metric_norms = Vec::with_capacity(n + 1);
    let mut transition_to_flat = vec![vec![QSeries::zero(max_q_degree); n + 1]; n + 1];

    for branch in 0..=n {
        let mut numerator = vec![QSeries::one(max_q_degree)];
        let mut denominator = QSeries::one(max_q_degree);
        for other in 0..=n {
            if other == branch {
                continue;
            }
            numerator = multiply_qseries_polynomial_by_linear(
                &numerator,
                &roots[other].neg(),
                max_q_degree,
            );
            denominator = denominator.mul(&roots[branch].sub(&roots[other]));
        }
        let denominator_inv = denominator.inverse()?;
        for (row, coeff) in numerator.into_iter().enumerate() {
            transition_to_flat[row][branch] = coeff.mul(&denominator_inv);
        }
        metric_norms.push(denominator.inverse()?);
        inverse_metric_norms.push(denominator);
    }

    Ok(SpecializedQuantumCanonicalData {
        roots,
        metric_norms,
        inverse_metric_norms,
        transition_to_flat,
    })
}

fn canonical_root_series_at_lambda_weights(
    n: usize,
    branch: usize,
    max_q_degree: usize,
    weights: &[Rational],
) -> Result<QSeries, GwError> {
    let mut root = QSeries::constant(RatFun::from_rational(weights[branch].clone()), max_q_degree);
    for _ in 0..=max_q_degree {
        let p = characteristic_series_at_lambda_weights(n, &root, weights)
            .sub(&QSeries::q(max_q_degree));
        if p.coeffs().iter().all(RatFun::is_zero) {
            break;
        }
        let dp = characteristic_derivative_series_at_lambda_weights(n, &root, weights);
        root = root.sub(&p.div(&dp)?);
    }
    Ok(root)
}

fn characteristic_series_at_lambda_weights(n: usize, x: &QSeries, weights: &[Rational]) -> QSeries {
    let max_q_degree = x.max_degree();
    let mut product = QSeries::one(max_q_degree);
    for weight in weights.iter().take(n + 1) {
        product = product.mul(&x.sub(&QSeries::constant(
            RatFun::from_rational(weight.clone()),
            max_q_degree,
        )));
    }
    product
}

fn characteristic_derivative_series_at_lambda_weights(
    n: usize,
    x: &QSeries,
    weights: &[Rational],
) -> QSeries {
    let max_q_degree = x.max_degree();
    let mut total = QSeries::zero(max_q_degree);
    for omitted in 0..=n {
        let mut product = QSeries::one(max_q_degree);
        for (idx, weight) in weights.iter().enumerate().take(n + 1) {
            if idx == omitted {
                continue;
            }
            product = product.mul(&x.sub(&QSeries::constant(
                RatFun::from_rational(weight.clone()),
                max_q_degree,
            )));
        }
        total = total.add(&product);
    }
    total
}

fn multiply_qseries_polynomial_by_linear(
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

pub fn projective_space_descendant_s_matrix(
    n: usize,
    q_degree: usize,
    z_order: usize,
) -> Result<SeriesSMatrix, GwError> {
    static CACHE: OnceLock<Mutex<HashMap<CalibrationCacheKey, SeriesSMatrix>>> = OnceLock::new();
    let key = CalibrationCacheKey {
        n,
        q_degree,
        z_order,
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(descendant_s) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(descendant_s);
    }

    let size = n + 1;
    let quantum_h = series_h_multiplication_matrix(n, q_degree, true);
    let classical_h = series_h_multiplication_matrix(n, q_degree, false);
    let mut coefficients = Vec::with_capacity(z_order + 1);
    coefficients.push(SeriesMatrix::identity(size, q_degree));

    for order in 1..=z_order {
        let previous = &coefficients[order - 1];
        let source = quantum_h.mul(previous).sub(&previous.mul(&classical_h));
        coefficients.push(integrate_q_derivative_zero_constant_matrix(&source)?);
    }

    let descendant_s = SeriesSMatrix {
        size,
        q_degree,
        z_order,
        coefficients,
        calibration: CalibrationId("projective-space-small-j".to_string()),
    };
    cache.lock().unwrap().insert(key, descendant_s.clone());
    Ok(descendant_s)
}

fn projective_space_descendant_s_matrix_at_lambda_weights(
    n: usize,
    q_degree: usize,
    z_order: usize,
    weights: &[Rational],
) -> Result<SeriesSMatrix, GwError> {
    static CACHE: OnceLock<Mutex<HashMap<LambdaCalibrationCacheKey, SeriesSMatrix>>> =
        OnceLock::new();
    let key = LambdaCalibrationCacheKey {
        n,
        q_degree,
        z_order,
        weights: weights.to_vec(),
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(descendant_s) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(descendant_s);
    }

    let size = n + 1;
    let quantum_h = series_h_multiplication_matrix_at_lambda_weights(n, q_degree, true, weights)?;
    let classical_h =
        series_h_multiplication_matrix_at_lambda_weights(n, q_degree, false, weights)?;
    let mut coefficients = Vec::with_capacity(z_order + 1);
    coefficients.push(SeriesMatrix::identity(size, q_degree));

    for order in 1..=z_order {
        let previous = &coefficients[order - 1];
        let source = quantum_h.mul(previous).sub(&previous.mul(&classical_h));
        coefficients.push(integrate_q_derivative_zero_constant_matrix(&source)?);
    }

    let descendant_s = SeriesSMatrix {
        size,
        q_degree,
        z_order,
        coefficients,
        calibration: CalibrationId("projective-space-small-j-lambda-line".to_string()),
    };
    cache.lock().unwrap().insert(key, descendant_s.clone());
    Ok(descendant_s)
}

fn series_h_multiplication_matrix_at_lambda_weights(
    n: usize,
    q_degree: usize,
    quantum: bool,
    weights: &[Rational],
) -> Result<SeriesMatrix, GwError> {
    if weights.len() != n + 1 {
        return Err(GwError::AlgebraFailure(format!(
            "expected {} lambda weights, got {}",
            n + 1,
            weights.len()
        )));
    }

    let size = n + 1;
    let mut matrix = vec![vec![QSeries::zero(q_degree); size]; size];
    for col in 0..n {
        matrix[col + 1][col] = QSeries::one(q_degree);
    }
    let relation = h_power_relation_series_at_lambda_weights(n, q_degree, quantum, weights);
    for row in 0..=n {
        matrix[row][n] = relation[row].clone();
    }
    Ok(SeriesMatrix::from_entries(matrix))
}

fn series_h_multiplication_matrix(n: usize, q_degree: usize, quantum: bool) -> SeriesMatrix {
    let size = n + 1;
    let mut matrix = vec![vec![QSeries::zero(q_degree); size]; size];
    for col in 0..n {
        matrix[col + 1][col] = QSeries::one(q_degree);
    }
    let relation = h_power_relation_series(n, q_degree, quantum);
    for row in 0..=n {
        matrix[row][n] = relation[row].clone();
    }
    SeriesMatrix::from_entries(matrix)
}

fn h_power_relation_series(n: usize, q_degree: usize, quantum: bool) -> Vec<QSeries> {
    let elementary = elementary_symmetric_weights(n);
    let mut rhs = vec![QSeries::zero(q_degree); n + 1];
    for k in 1..=n + 1 {
        let power = n + 1 - k;
        let signed = if k % 2 == 1 {
            elementary[k].clone()
        } else {
            -elementary[k].clone()
        };
        rhs[power] = rhs[power].add(&QSeries::constant(signed, q_degree));
    }
    if quantum {
        rhs[0] = rhs[0].add(&QSeries::q(q_degree));
    }
    rhs
}

fn h_power_relation_series_at_lambda_weights(
    n: usize,
    q_degree: usize,
    quantum: bool,
    weights: &[Rational],
) -> Vec<QSeries> {
    let elementary = elementary_symmetric_rational(weights);
    let mut rhs = vec![QSeries::zero(q_degree); n + 1];
    for k in 1..=n + 1 {
        let power = n + 1 - k;
        let signed = if k % 2 == 1 {
            elementary[k].clone()
        } else {
            -elementary[k].clone()
        };
        rhs[power] = rhs[power].add(&QSeries::constant(RatFun::from_rational(signed), q_degree));
    }
    if quantum {
        rhs[0] = rhs[0].add(&QSeries::q(q_degree));
    }
    rhs
}

fn elementary_symmetric_rational(weights: &[Rational]) -> Vec<Rational> {
    let mut elementary = vec![Rational::zero(); weights.len() + 1];
    elementary[0] = Rational::one();
    for (idx, weight) in weights.iter().enumerate() {
        for k in (1..=idx + 1).rev() {
            elementary[k] = elementary[k].clone() + elementary[k - 1].clone() * weight.clone();
        }
    }
    elementary
}

fn integrate_q_derivative_zero_constant_matrix(
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

fn integrate_q_derivative_zero_constant(series: &QSeries) -> Result<QSeries, GwError> {
    let max_degree = series.max_degree();
    if series.coeff(0).is_some_and(|constant| !constant.is_zero()) {
        return Err(GwError::AlgebraFailure(
            "cannot integrate q d/dq with nonzero constant term and zero integration constant"
                .to_string(),
        ));
    }

    let mut coeffs = vec![RatFun::zero(); max_degree + 1];
    for degree in 1..=max_degree {
        let denominator = RatFun::from(degree);
        coeffs[degree] = series.coeff(degree).cloned().unwrap_or_else(RatFun::zero) / denominator;
    }
    Ok(QSeries::from_coeffs(coeffs))
}

fn relative_sqrt_delta_series(delta: &QSeries) -> Result<QSeries, GwError> {
    let delta0 = delta
        .coeff(0)
        .ok_or_else(|| GwError::AlgebraFailure("empty Delta series".to_string()))?;
    if delta0.is_zero() {
        return Err(GwError::NonSemisimplePoint);
    }
    let inv_delta0 = &RatFun::one() / delta0;
    delta.scale(&inv_delta0).sqrt_with_constant_one()
}

fn canonical_evaluation_matrix(roots: &[QSeries]) -> SeriesMatrix {
    SeriesMatrix::from_entries(
        roots
            .iter()
            .map(|root| {
                (0..roots.len())
                    .map(|power| root.pow_usize(power))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
}

fn solve_projective_r_coefficients(
    roots: &[QSeries],
    connection: &SeriesMatrix,
    _metric: &SeriesMatrix,
    classical_diagonal: &[Vec<RatFun>],
    q_degree: usize,
    z_order: usize,
) -> Result<Vec<SeriesMatrix>, GwError> {
    let size = roots.len();
    let mut coefficients = Vec::with_capacity(z_order + 1);
    coefficients.push(SeriesMatrix::identity(size, q_degree));

    for order in 1..=z_order {
        let previous = &coefficients[order - 1];
        let recursion_source = previous.q_derivative().add(&connection.mul(previous));
        let mut entries = vec![vec![QSeries::zero(q_degree); size]; size];

        for row in 0..size {
            for col in 0..size {
                if row == col {
                    continue;
                }
                let root_difference = roots[col].sub(&roots[row]);
                entries[row][col] = recursion_source
                    .entry(row, col)
                    .neg()
                    .div(&root_difference)?;
            }
        }

        for branch in 0..size {
            entries[branch][branch] = solve_r_diagonal_from_flatness(
                connection,
                &entries,
                branch,
                classical_diagonal[branch][order].clone(),
                q_degree,
            );
        }

        let next = SeriesMatrix::from_entries(entries);
        coefficients.push(next);
    }

    Ok(coefficients)
}

fn solve_r_diagonal_from_flatness(
    connection: &SeriesMatrix,
    entries: &[Vec<QSeries>],
    branch: usize,
    constant: RatFun,
    q_degree: usize,
) -> QSeries {
    let mut known = QSeries::zero(q_degree);
    for (source, row) in entries.iter().enumerate() {
        if source == branch {
            continue;
        }
        known = known.add(&connection.entry(branch, source).mul(&row[branch]));
    }
    let target = known.neg();
    let diagonal_connection = connection.entry(branch, branch);
    let a0 = diagonal_connection
        .coeff(0)
        .cloned()
        .unwrap_or_else(RatFun::zero);

    let mut coeffs = vec![RatFun::zero(); q_degree + 1];
    coeffs[0] = constant;
    for degree in 1..=q_degree {
        let mut numerator = target.coeff(degree).cloned().unwrap_or_else(RatFun::zero);
        for connection_degree in 1..=degree {
            let term = diagonal_connection
                .coeff(connection_degree)
                .cloned()
                .unwrap_or_else(RatFun::zero)
                * coeffs[degree - connection_degree].clone();
            numerator = numerator - term;
        }
        let denominator = RatFun::from(degree) + a0.clone();
        coeffs[degree] = numerator / denominator;
    }
    QSeries::from_coeffs(coeffs)
}

fn classical_limit_diagonal_coefficients(n: usize, z_order: usize) -> Vec<Vec<RatFun>> {
    (0..=n)
        .map(|branch| classical_limit_diagonal_coefficients_for_branch(n, branch, z_order))
        .collect()
}

fn classical_limit_diagonal_coefficients_at_lambda_weights(
    n: usize,
    z_order: usize,
    weights: &[Rational],
) -> Vec<Vec<RatFun>> {
    (0..=n)
        .map(|branch| {
            classical_limit_diagonal_coefficients_for_branch_at_lambda_weights(
                n, branch, z_order, weights,
            )
        })
        .collect()
}

fn classical_limit_diagonal_coefficients_for_branch(
    n: usize,
    branch: usize,
    z_order: usize,
) -> Vec<RatFun> {
    let mut exponent = vec![RatFun::zero(); z_order + 1];
    for r in 1..=((z_order + 1) / 2) {
        let order = 2 * r - 1;
        let coefficient =
            bernoulli_number(2 * r) / (Rational::from(2 * r) * Rational::from(2 * r - 1));
        let coefficient = RatFun::from_rational(coefficient);
        let mut weight_sum = RatFun::zero();
        for other in 0..=n {
            if other == branch {
                continue;
            }
            let difference = &lambda(other) - &lambda(branch);
            let term = &RatFun::one() / &difference.pow_usize(order);
            weight_sum = &weight_sum + &term;
        }
        exponent[order] = &coefficient * &weight_sum;
    }
    exp_scalar_z_series(&exponent)
}

fn classical_limit_diagonal_coefficients_for_branch_at_lambda_weights(
    n: usize,
    branch: usize,
    z_order: usize,
    weights: &[Rational],
) -> Vec<RatFun> {
    let mut exponent = vec![RatFun::zero(); z_order + 1];
    for r in 1..=((z_order + 1) / 2) {
        let order = 2 * r - 1;
        let coefficient =
            bernoulli_number(2 * r) / (Rational::from(2 * r) * Rational::from(2 * r - 1));
        let mut weight_sum = Rational::zero();
        for other in 0..=n {
            if other == branch {
                continue;
            }
            let difference = weights[other].clone() - weights[branch].clone();
            weight_sum += Rational::one() / difference.pow_usize(order);
        }
        exponent[order] = RatFun::from_rational(coefficient * weight_sum);
    }
    exp_scalar_z_series(&exponent)
}

fn exp_scalar_z_series(exponent: &[RatFun]) -> Vec<RatFun> {
    let z_order = exponent.len().saturating_sub(1);
    let mut out = vec![RatFun::zero(); z_order + 1];
    out[0] = RatFun::one();
    for degree in 1..=z_order {
        let mut total = RatFun::zero();
        for part in 1..=degree {
            if exponent[part].is_zero() {
                continue;
            }
            let scaled = &exponent[part] * &RatFun::from(part);
            let term = &scaled * &out[degree - part];
            total = &total + &term;
        }
        out[degree] = &total / &RatFun::from(degree);
    }
    out
}

fn bernoulli_number(n: usize) -> Rational {
    let mut bernoulli = vec![Rational::zero(); n + 1];
    bernoulli[0] = Rational::one();
    for degree in 1..=n {
        let mut sum = Rational::zero();
        for idx in 0..degree {
            sum += Rational::from(binomial(degree + 1, idx)) * bernoulli[idx].clone();
        }
        bernoulli[degree] = -sum / Rational::from(degree + 1);
    }
    bernoulli[n].clone()
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut out = 1usize;
    for step in 1..=k {
        out = out * (n + 1 - step) / step;
    }
    out
}

pub fn compute(req: &InvariantRequest) -> Result<InvariantResult, GwError> {
    match compute_by_givental_graphs(req) {
        Ok(result) => Ok(result),
        Err(GwError::UnsupportedInvariant(message)) => {
            if message.contains("full quantized S-action") {
                Err(GwError::UnsupportedInvariant(message))
            } else {
                validation::seed_compute(req, "givental-seed")
            }
        }
        Err(err) => Err(err),
    }
}

pub fn product_of_point_theories(
    colors: usize,
    genus: usize,
    color: usize,
    descendant_powers: &[usize],
) -> Result<RatFun, GwError> {
    if color >= colors {
        return Err(GwError::AlgebraFailure(format!(
            "color {color} out of range for {colors} colors"
        )));
    }
    let value = WittenKontsevich::new().psi_integral(genus, descendant_powers);
    Ok(RatFun::from_rational(value))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AncestorLegTerm {
    base_power: usize,
    vector: Vec<QSeries>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LegFactorOption {
    power: usize,
    coefficient: QSeries,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EdgeFactorOption {
    left_power: usize,
    right_power: usize,
    coefficient: QSeries,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VertexContributionKey {
    genus: usize,
    color: usize,
    powers: Vec<usize>,
}

pub fn compute_by_givental_graphs(req: &InvariantRequest) -> Result<InvariantResult, GwError> {
    let max_descendant_power = req
        .insertions
        .iter()
        .map(|insertion| insertion.descendant_power)
        .max()
        .unwrap_or(0);

    if !is_stable_cohft_range(req.genus, req.insertions.len()) {
        return Err(GwError::UnsupportedInvariant(
            "Givental graph expansion is implemented for stable (g,n) CohFT ranges only"
                .to_string(),
        ));
    }

    if let Some(total_degree) = req.insertion_degree() {
        let virtual_dimension = req.virtual_dimension();
        if virtual_dimension >= 0 && total_degree as isize != virtual_dimension {
            return Ok(InvariantResult {
                value: RatFun::zero(),
                engine: "givental-r-graph",
                notes: vec![format!(
                    "dimension mismatch gives zero: virtual dimension {virtual_dimension}, insertion degree {total_degree}"
                )],
            });
        }
    }

    let graph_dimension = 3 * req.genus + req.insertions.len() - 3;
    let needed_r_order = graph_dimension + 1;
    let needed_s_order = max_descendant_power;
    let needed_z_order = needed_r_order.max(needed_s_order);
    let q_degree = req.degree;
    let z_order = req
        .truncation
        .as_ref()
        .map(|truncation| truncation.z_order)
        .unwrap_or(needed_z_order);
    if z_order < needed_z_order {
        return Err(GwError::TruncationTooLow);
    }

    let weights = (1..=req.n + 1).map(Rational::from).collect::<Vec<_>>();
    let (calibration, descendant_s, specialized_nonequivariant) = if req.equivariant {
        (
            projective_space_j_calibration(req.n, q_degree, needed_r_order)?,
            projective_space_descendant_s_matrix(req.n, q_degree, needed_s_order)?,
            false,
        )
    } else {
        (
            projective_space_j_calibration_at_lambda_weights(
                req.n,
                q_degree,
                needed_r_order,
                &weights,
            )?,
            projective_space_descendant_s_matrix_at_lambda_weights(
                req.n,
                q_degree,
                needed_s_order,
                &weights,
            )?,
            true,
        )
    };
    let inverse_r = inverse_r_coefficients(calibration.r_matrix.coefficients());
    let unit = calibration.relative_sqrt_delta_inverse.clone();
    let translation = translation_coefficients(&inverse_r, &unit, q_degree);
    let edge_coefficients =
        edge_propagator_coefficients(&inverse_r, &calibration.metric, graph_dimension, q_degree)?;
    let insertion_terms = ancestor_insertion_terms(
        req.n,
        &req.insertions,
        &descendant_s,
        &calibration.psi_inverse,
        q_degree,
        graph_dimension,
    );
    let leg_options = leg_options_by_marking_color(
        &insertion_terms,
        &inverse_r,
        q_degree,
        graph_dimension,
        req.n + 1,
    );
    let edge_options = edge_options_by_color(&edge_coefficients);

    let mut total = QSeries::zero(q_degree);
    let graphs = stable_graphs(req.genus, req.insertions.len());
    let oracle = WittenKontsevich::new();
    let mut vertex_cache = HashMap::<VertexContributionKey, QSeries>::new();
    let mut coloring_cache = HashMap::<usize, Vec<Vec<usize>>>::new();

    for graph in &graphs {
        let automorphism_factor = &RatFun::one() / &RatFun::from(graph.automorphism_order());
        let colorings = coloring_cache
            .entry(graph.vertices.len())
            .or_insert_with(|| vertex_colorings(graph.vertices.len(), req.n + 1));
        for colors in colorings.iter() {
            let mut graph_total = QSeries::zero(q_degree);
            let mut base_powers = vec![Vec::<usize>::new(); graph.vertices.len()];
            accumulate_graph_factors(
                graph,
                &colors,
                &leg_options,
                &edge_options,
                &calibration,
                &translation,
                &oracle,
                &mut vertex_cache,
                q_degree,
                graph_dimension,
                0,
                0,
                QSeries::one(q_degree),
                &mut base_powers,
                &mut graph_total,
            );

            total = total.add(&graph_total.scale(&automorphism_factor));
        }
    }

    let value = total
        .coeff(req.degree)
        .cloned()
        .unwrap_or_else(RatFun::zero);
    if req.equivariant {
        return Ok(InvariantResult {
            value,
            engine: "givental-r-graph",
            notes: vec![
                "computed by truncated J-calibrated R-matrix stable-graph expansion; result remains equivariant"
                    .to_string(),
            ],
        });
    }

    if specialized_nonequivariant {
        return Ok(InvariantResult {
            value,
            engine: "givental-r-graph-lambda-line",
            notes: vec![
                "computed by J-calibrated S/R stable-graph expansion after early generic lambda-line specialization"
                    .to_string(),
            ],
        });
    }

    let limit = value.nonequivariant_limit_line(req.n, &weights)?;
    Ok(InvariantResult {
        value: RatFun::from_rational(limit),
        engine: "givental-r-graph-limit",
        notes: vec![
            "computed by truncated J-calibrated R-matrix stable-graph expansion and lambda-line nonequivariant limit"
                .to_string(),
        ],
    })
}

fn is_stable_cohft_range(genus: usize, markings: usize) -> bool {
    2 * genus + markings > 2
}

fn ancestor_insertion_terms(
    n: usize,
    insertions: &[Insertion],
    descendant_s: &SeriesSMatrix,
    psi_inverse: &SeriesMatrix,
    q_degree: usize,
    max_power: usize,
) -> Vec<Vec<AncestorLegTerm>> {
    insertions
        .iter()
        .map(|insertion| {
            let max_order = insertion
                .descendant_power
                .min(descendant_s.coefficients().len().saturating_sub(1));
            let mut terms = Vec::new();
            for s_order in 0..=max_order {
                let base_power = insertion.descendant_power - s_order;
                if base_power > max_power {
                    continue;
                }
                let flat_vector =
                    apply_s_coefficient_to_insertion(n, descendant_s, s_order, insertion, q_degree);
                if flat_vector.iter().all(QSeries::is_zero) {
                    continue;
                }
                let canonical_vector = apply_matrix_to_vector(psi_inverse, &flat_vector, q_degree);
                if canonical_vector.iter().all(QSeries::is_zero) {
                    continue;
                }
                terms.push(AncestorLegTerm {
                    base_power,
                    vector: canonical_vector,
                });
            }
            terms
        })
        .collect()
}

fn apply_s_coefficient_to_insertion(
    n: usize,
    descendant_s: &SeriesSMatrix,
    s_order: usize,
    insertion: &Insertion,
    q_degree: usize,
) -> Vec<QSeries> {
    let matrix = descendant_s
        .coefficient(s_order)
        .expect("S coefficient order was bounded before access");
    let class_vector = insertion
        .class
        .coeffs()
        .iter()
        .map(|coeff| QSeries::constant(coeff.clone(), q_degree))
        .collect::<Vec<_>>();
    debug_assert_eq!(class_vector.len(), n + 1);
    apply_matrix_to_vector(matrix, &class_vector, q_degree)
}

fn apply_matrix_to_vector(
    matrix: &SeriesMatrix,
    vector: &[QSeries],
    q_degree: usize,
) -> Vec<QSeries> {
    debug_assert_eq!(matrix.cols(), vector.len());
    (0..matrix.rows())
        .map(|row| {
            let mut total = QSeries::zero(q_degree);
            for (col, entry) in vector.iter().enumerate() {
                total = total.add(&matrix.entry(row, col).mul(entry));
            }
            total
        })
        .collect()
}

fn inverse_r_coefficients(coefficients: &[SeriesMatrix]) -> Vec<SeriesMatrix> {
    let size = coefficients[0].rows();
    let q_degree = coefficients[0].max_degree();
    let mut inverse = Vec::with_capacity(coefficients.len());
    inverse.push(SeriesMatrix::identity(size, q_degree));
    for order in 1..coefficients.len() {
        let mut total = SeriesMatrix::zero(size, size, q_degree);
        for left in 1..=order {
            total = total.add(&coefficients[left].mul(&inverse[order - left]));
        }
        inverse.push(total.neg());
    }
    inverse
}

fn translation_coefficients(
    inverse_r: &[SeriesMatrix],
    unit: &[QSeries],
    q_degree: usize,
) -> Vec<Vec<QSeries>> {
    let size = unit.len();
    let mut out = vec![vec![QSeries::zero(q_degree); inverse_r.len() + 1]; size];
    for power in 2..=inverse_r.len() {
        let r_order = power - 1;
        for color in 0..size {
            let mut total = QSeries::zero(q_degree);
            for (source, unit_coeff) in unit.iter().enumerate() {
                total = total.add(&inverse_r[r_order].entry(color, source).mul(unit_coeff));
            }
            out[color][power] = total.neg();
        }
    }
    out
}

fn edge_propagator_coefficients(
    inverse_r: &[SeriesMatrix],
    metric: &SeriesMatrix,
    max_power: usize,
    q_degree: usize,
) -> Result<Vec<Vec<Vec<Vec<QSeries>>>>, GwError> {
    let size = metric.rows();
    let mut metric_inverse = Vec::with_capacity(size);
    for color in 0..size {
        metric_inverse.push(metric.entry(color, color).inverse()?);
    }

    let mut out =
        vec![vec![vec![vec![QSeries::zero(q_degree); max_power + 1]; max_power + 1]; size]; size];
    for left_color in 0..size {
        for right_color in 0..size {
            for left_power in 0..=max_power {
                for right_power in 0..=max_power {
                    let mut coefficient = QSeries::zero(q_degree);
                    for shift in 0..=right_power {
                        let numerator = edge_numerator_coefficient(
                            inverse_r,
                            &metric_inverse,
                            left_color,
                            right_color,
                            left_power + 1 + shift,
                            right_power - shift,
                            q_degree,
                        );
                        coefficient = if shift % 2 == 0 {
                            coefficient.sub(&numerator)
                        } else {
                            coefficient.add(&numerator)
                        };
                    }
                    out[left_color][right_color][left_power][right_power] = coefficient;
                }
            }
        }
    }
    Ok(out)
}

fn edge_numerator_coefficient(
    inverse_r: &[SeriesMatrix],
    metric_inverse: &[QSeries],
    left_color: usize,
    right_color: usize,
    left_power: usize,
    right_power: usize,
    q_degree: usize,
) -> QSeries {
    if left_power >= inverse_r.len() || right_power >= inverse_r.len() {
        return QSeries::zero(q_degree);
    }
    let mut total = QSeries::zero(q_degree);
    for source in 0..metric_inverse.len() {
        let term = inverse_r[left_power]
            .entry(left_color, source)
            .mul(&metric_inverse[source])
            .mul(inverse_r[right_power].entry(right_color, source));
        total = total.add(&term);
    }
    total
}

fn accumulate_graph_factors(
    graph: &crate::graphs::StableGraph,
    colors: &[usize],
    leg_options: &[Vec<Vec<LegFactorOption>>],
    edge_options: &[Vec<Vec<EdgeFactorOption>>],
    calibration: &ProjectiveSpaceJCalibration,
    translation: &[Vec<QSeries>],
    oracle: &WittenKontsevich,
    vertex_cache: &mut HashMap<VertexContributionKey, QSeries>,
    q_degree: usize,
    max_power: usize,
    factor_index: usize,
    current_power_sum: usize,
    coefficient: QSeries,
    base_powers: &mut [Vec<usize>],
    total: &mut QSeries,
) {
    if coefficient.is_zero() || current_power_sum > max_power {
        return;
    }

    let leg_count = graph.legs.len();
    let edge_count = graph.edges.len();
    if factor_index < leg_count {
        let marking = factor_index;
        let vertex = graph.legs[marking];
        let color = colors[vertex];
        for option in &leg_options[marking][color] {
            let next_power_sum = current_power_sum + option.power;
            if next_power_sum > max_power {
                continue;
            }
            let next_coefficient = coefficient.mul(&option.coefficient);
            if next_coefficient.is_zero() {
                continue;
            }
            base_powers[vertex].push(option.power);
            accumulate_graph_factors(
                graph,
                colors,
                leg_options,
                edge_options,
                calibration,
                translation,
                oracle,
                vertex_cache,
                q_degree,
                max_power,
                factor_index + 1,
                next_power_sum,
                next_coefficient,
                base_powers,
                total,
            );
            base_powers[vertex].pop();
        }
        return;
    }

    if factor_index < leg_count + edge_count {
        let edge_index = factor_index - leg_count;
        let edge = &graph.edges[edge_index];
        let left_color = colors[edge.a];
        let right_color = colors[edge.b];
        for option in &edge_options[left_color][right_color] {
            let next_power_sum = current_power_sum + option.left_power + option.right_power;
            if next_power_sum > max_power {
                continue;
            }
            let next_coefficient = coefficient.mul(&option.coefficient);
            if next_coefficient.is_zero() {
                continue;
            }
            base_powers[edge.a].push(option.left_power);
            base_powers[edge.b].push(option.right_power);
            accumulate_graph_factors(
                graph,
                colors,
                leg_options,
                edge_options,
                calibration,
                translation,
                oracle,
                vertex_cache,
                q_degree,
                max_power,
                factor_index + 1,
                next_power_sum,
                next_coefficient,
                base_powers,
                total,
            );
            base_powers[edge.b].pop();
            base_powers[edge.a].pop();
        }
        return;
    }

    let mut vertex_product = QSeries::one(q_degree);
    for (vertex, powers) in base_powers.iter().enumerate() {
        let vertex_sum = vertex_contribution_with_translations(
            graph.vertices[vertex].genus,
            colors[vertex],
            powers,
            calibration,
            translation,
            oracle,
            vertex_cache,
            q_degree,
        );
        vertex_product = vertex_product.mul(&vertex_sum);
        if vertex_product.is_zero() {
            return;
        }
    }
    *total = total.add(&coefficient.mul(&vertex_product));
}

fn leg_options_by_marking_color(
    insertion_terms: &[Vec<AncestorLegTerm>],
    inverse_r: &[SeriesMatrix],
    q_degree: usize,
    max_power: usize,
    colors: usize,
) -> Vec<Vec<Vec<LegFactorOption>>> {
    insertion_terms
        .iter()
        .map(|terms| {
            (0..colors)
                .map(|color| leg_options_for_color(color, terms, inverse_r, q_degree, max_power))
                .collect()
        })
        .collect()
}

fn leg_options_for_color(
    color: usize,
    insertion_terms: &[AncestorLegTerm],
    inverse_r: &[SeriesMatrix],
    q_degree: usize,
    max_power: usize,
) -> Vec<LegFactorOption> {
    let mut out = Vec::new();
    for term in insertion_terms {
        for (order, matrix) in inverse_r.iter().enumerate() {
            let power = term.base_power + order;
            if power > max_power {
                continue;
            }
            let mut coefficient = QSeries::zero(q_degree);
            for (source, source_coeff) in term.vector.iter().enumerate() {
                coefficient = coefficient.add(&matrix.entry(color, source).mul(source_coeff));
            }
            if !coefficient.is_zero() {
                out.push(LegFactorOption { power, coefficient });
            }
        }
    }
    out
}

fn edge_options_by_color(
    edge_coefficients: &[Vec<Vec<Vec<QSeries>>>],
) -> Vec<Vec<Vec<EdgeFactorOption>>> {
    let colors = edge_coefficients.len();
    (0..colors)
        .map(|left_color| {
            (0..colors)
                .map(|right_color| {
                    edge_options_for_colors(left_color, right_color, edge_coefficients)
                })
                .collect()
        })
        .collect()
}

fn edge_options_for_colors(
    left_color: usize,
    right_color: usize,
    edge_coefficients: &[Vec<Vec<Vec<QSeries>>>],
) -> Vec<EdgeFactorOption> {
    let max_power = edge_coefficients[left_color][right_color]
        .len()
        .saturating_sub(1);
    let mut out = Vec::new();
    for left_power in 0..edge_coefficients[left_color][right_color].len() {
        for right_power in 0..edge_coefficients[left_color][right_color][left_power].len() {
            if left_power + right_power > max_power {
                continue;
            }
            let coefficient =
                edge_coefficients[left_color][right_color][left_power][right_power].clone();
            if !coefficient.is_zero() {
                out.push(EdgeFactorOption {
                    left_power,
                    right_power,
                    coefficient,
                });
            }
        }
    }
    out
}

fn vertex_contribution_with_translations(
    genus: usize,
    color: usize,
    base_powers: &[usize],
    calibration: &ProjectiveSpaceJCalibration,
    translation: &[Vec<QSeries>],
    oracle: &WittenKontsevich,
    vertex_cache: &mut HashMap<VertexContributionKey, QSeries>,
    q_degree: usize,
) -> QSeries {
    let mut sorted_powers = base_powers.to_vec();
    sorted_powers.sort_unstable();
    let key = VertexContributionKey {
        genus,
        color,
        powers: sorted_powers,
    };
    if let Some(cached) = vertex_cache.get(&key) {
        return cached.clone();
    }

    let base_dimension = 3isize * genus as isize - 3 + base_powers.len() as isize;
    let base_power_sum = base_powers.iter().sum::<usize>() as isize;
    let translation_excess = base_dimension - base_power_sum;
    if translation_excess < 0 {
        let zero = QSeries::zero(q_degree);
        vertex_cache.insert(key, zero.clone());
        return zero;
    }

    let mut total = QSeries::zero(q_degree);
    let max_translations = translation_excess as usize;
    for translation_count in 0..=max_translations {
        if translation_count == 0 {
            if translation_excess == 0 {
                let vertex_factor = vertex_tft_factor(genus, base_powers.len(), color, calibration);
                total = total.add(&vertex_factor.scale(&RatFun::from_rational(
                    oracle.psi_integral(genus, base_powers),
                )));
            }
            continue;
        }

        for composition in positive_compositions(translation_excess as usize, translation_count) {
            let mut powers = base_powers.to_vec();
            let mut coefficient = QSeries::one(q_degree);
            for excess in composition {
                let power = excess + 1;
                if power >= translation[color].len() {
                    coefficient = QSeries::zero(q_degree);
                    break;
                }
                coefficient = coefficient.mul(&translation[color][power]);
                powers.push(power);
            }
            if coefficient.is_zero() {
                continue;
            }
            let vertex_factor = vertex_tft_factor(genus, powers.len(), color, calibration);
            let psi = RatFun::from_rational(oracle.psi_integral(genus, &powers));
            let symmetry = RatFun::from(factorial(translation_count));
            let term = coefficient.mul(&vertex_factor).scale(&(&psi / &symmetry));
            total = total.add(&term);
        }
    }
    vertex_cache.insert(key, total.clone());
    total
}

fn vertex_tft_factor(
    genus: usize,
    valence: usize,
    color: usize,
    calibration: &ProjectiveSpaceJCalibration,
) -> QSeries {
    let genus_factor = if genus == 0 {
        calibration.inverse_delta[color].clone()
    } else {
        calibration.delta[color].pow_usize(genus - 1)
    };
    genus_factor.mul(&calibration.relative_sqrt_delta[color].pow_usize(valence))
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

fn positive_compositions(total: usize, parts: usize) -> Vec<Vec<usize>> {
    fn rec(total: usize, parts: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() + 1 == parts {
            if total > 0 {
                current.push(total);
                out.push(current.clone());
                current.pop();
            }
            return;
        }
        let remaining_slots = parts - current.len() - 1;
        for value in 1..=total.saturating_sub(remaining_slots) {
            current.push(value);
            rec(total - value, parts, current, out);
            current.pop();
        }
    }

    if parts == 0 {
        return if total == 0 {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
    }
    if total < parts {
        return Vec::new();
    }
    let mut out = Vec::new();
    rec(total, parts, &mut Vec::new(), &mut out);
    out
}

fn vertex_colorings(vertices: usize, colors: usize) -> Vec<Vec<usize>> {
    fn rec(vertices: usize, colors: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == vertices {
            out.push(current.clone());
            return;
        }
        for color in 0..colors {
            current.push(color);
            rec(vertices, colors, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    rec(vertices, colors, &mut Vec::new(), &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CohomologyClass;
    use crate::{tau, ComputeMode, InvariantRequest};

    #[test]
    fn identity_r_matrix_has_expected_coefficients() {
        let r = RMatrix::identity(3, 4);
        r.check_unitarity_identity_case().unwrap();
        assert_eq!(r.coefficient(0, 2, 2), Some(&RatFun::one()));
        assert_eq!(r.coefficient(1, 2, 2), Some(&RatFun::zero()));
    }

    #[test]
    fn series_identity_r_matrix_is_unitary_for_any_metric() {
        let metric = SeriesMatrix::from_entries(vec![
            vec![
                crate::series::QSeries::constant(RatFun::from(2usize), 1),
                crate::series::QSeries::q(1),
            ],
            vec![
                crate::series::QSeries::q(1),
                crate::series::QSeries::constant(RatFun::from(5usize), 1),
            ],
        ]);
        let r = SeriesRMatrix::identity(
            2,
            1,
            3,
            CanonicalFrameConvention::UnnormalizedCanonicalIdempotents,
        );
        r.check_identity_calibration().unwrap();
        r.check_unitarity(&metric).unwrap();
        assert_eq!(r.size(), 2);
        assert_eq!(r.q_degree(), 1);
        assert_eq!(r.z_order(), 3);
        assert_eq!(
            r.convention(),
            CanonicalFrameConvention::UnnormalizedCanonicalIdempotents
        );
    }

    #[test]
    fn projective_j_calibration_matches_p1_classical_limit() {
        let calibration = projective_space_j_calibration(1, 1, 2).unwrap();
        let weights = [
            crate::algebra::Rational::from(2),
            crate::algebra::Rational::from(5),
        ];
        let r1 = calibration.r_matrix.coefficient(1).unwrap();
        let r2 = calibration.r_matrix.coefficient(2).unwrap();

        assert_eq!(
            r1.entry(0, 0)
                .coeff(0)
                .unwrap()
                .evaluate_lambda_weights(1, &weights)
                .unwrap(),
            crate::algebra::Rational::new(1, 36)
        );
        assert_eq!(
            r1.entry(1, 1)
                .coeff(0)
                .unwrap()
                .evaluate_lambda_weights(1, &weights)
                .unwrap(),
            crate::algebra::Rational::new(-1, 36)
        );
        assert_eq!(
            r2.entry(0, 0)
                .coeff(0)
                .unwrap()
                .evaluate_lambda_weights(1, &weights)
                .unwrap(),
            crate::algebra::Rational::new(1, 2592)
        );
        assert_eq!(
            r1.entry(0, 1)
                .coeff(0)
                .unwrap()
                .evaluate_lambda_weights(1, &weights)
                .unwrap(),
            crate::algebra::Rational::zero()
        );
    }

    #[test]
    fn projective_j_calibration_relative_frame_inverts_for_p1() {
        let calibration = projective_space_j_calibration(1, 1, 2).unwrap();
        let product = calibration.psi_inverse.mul(&calibration.psi);
        assert_series_matrix_equal_after_lambda_eval(
            &product,
            &SeriesMatrix::identity(2, 1),
            1,
            &[
                crate::algebra::Rational::from(2),
                crate::algebra::Rational::from(5),
            ],
        );
    }

    #[test]
    fn projective_j_calibration_is_unitary_for_p1_low_order() {
        let calibration = projective_space_j_calibration(1, 1, 2).unwrap();
        assert_r_matrix_unitary_after_lambda_eval(
            &calibration.r_matrix,
            &calibration.metric,
            1,
            &[
                crate::algebra::Rational::from(2),
                crate::algebra::Rational::from(5),
            ],
        );
    }

    #[test]
    fn projective_j_calibration_r_coefficients_satisfy_diagonal_flatness() {
        let weights = [
            crate::algebra::Rational::from(2),
            crate::algebra::Rational::from(5),
            crate::algebra::Rational::from(11),
        ];
        let calibration =
            projective_space_j_calibration_at_lambda_weights(2, 3, 5, &weights).unwrap();

        for order in 0..=calibration.r_matrix.z_order() {
            let coefficient = calibration.r_matrix.coefficient(order).unwrap();
            let source = coefficient
                .q_derivative()
                .add(&calibration.connection.mul(coefficient));
            for branch in 0..calibration.r_matrix.size() {
                let diagonal = source.entry(branch, branch);
                for degree in 0..=diagonal.max_degree() {
                    assert_eq!(
                        diagonal.coeff(degree),
                        Some(&RatFun::zero()),
                        "nonzero diagonal flatness source at z^{order}, branch {branch}, q^{degree}"
                    );
                }
            }
        }
    }

    #[test]
    fn projective_descendant_s_matrix_matches_p1_low_order() {
        let descendant_s = projective_space_descendant_s_matrix(1, 1, 2).unwrap();
        let weights = [
            crate::algebra::Rational::from(2),
            crate::algebra::Rational::from(5),
        ];
        assert_eq!(
            descendant_s
                .coefficient(1)
                .unwrap()
                .entry(0, 1)
                .coeff(1)
                .unwrap()
                .evaluate_lambda_weights(1, &weights)
                .unwrap(),
            crate::algebra::Rational::one()
        );
        assert_eq!(
            descendant_s
                .coefficient(2)
                .unwrap()
                .entry(1, 1)
                .coeff(1)
                .unwrap()
                .evaluate_lambda_weights(1, &weights)
                .unwrap(),
            crate::algebra::Rational::one()
        );
    }

    #[test]
    fn product_of_point_theories_uses_psi_oracle() {
        let value = product_of_point_theories(2, 1, 0, &[1]).unwrap();
        assert_eq!(value.to_string(), "1/24");
    }

    #[test]
    fn givental_graph_reproduces_degree_zero_classical_product() {
        let req = InvariantRequest {
            mode: ComputeMode::Givental,
            ..InvariantRequest::new(
                2,
                0,
                0,
                vec![
                    tau(0, CohomologyClass::h_power(2, 1)),
                    tau(0, CohomologyClass::h_power(2, 1)),
                    tau(0, CohomologyClass::one(2)),
                ],
            )
        };
        let result = compute_by_givental_graphs(&req).unwrap();
        assert_eq!(result.value, RatFun::one());
    }

    #[test]
    fn givental_graph_reproduces_genus_one_degree_zero_obstruction_values() {
        let cases = [
            (
                InvariantRequest {
                    mode: ComputeMode::Givental,
                    ..InvariantRequest::new(1, 1, 0, vec![tau(0, CohomologyClass::h_power(1, 1))])
                },
                crate::algebra::Rational::new(-1, 24),
            ),
            (
                InvariantRequest {
                    mode: ComputeMode::Givental,
                    ..InvariantRequest::new(2, 1, 0, vec![tau(0, CohomologyClass::h_power(2, 1))])
                },
                crate::algebra::Rational::new(-1, 8),
            ),
            (
                InvariantRequest {
                    mode: ComputeMode::Givental,
                    ..InvariantRequest::new(2, 1, 0, vec![tau(1, CohomologyClass::one(2))])
                },
                crate::algebra::Rational::new(1, 8),
            ),
        ];

        for (req, expected) in cases {
            let graph = compute_by_givental_graphs(&req).unwrap();
            let obstruction =
                crate::validation::genus_one_degree_zero_one_point_obstruction(&req, "test")
                    .unwrap();
            assert_eq!(graph.value, RatFun::from_rational(expected));
            assert_eq!(graph.value, obstruction.value);
            assert_ne!(graph.engine, "givental-seed");
        }
    }

    #[test]
    fn givental_graph_reproduces_p1_degree_one_three_point() {
        let req = InvariantRequest {
            mode: ComputeMode::Givental,
            ..InvariantRequest::new(
                1,
                0,
                1,
                vec![
                    tau(0, CohomologyClass::h_power(1, 1)),
                    tau(0, CohomologyClass::h_power(1, 1)),
                    tau(0, CohomologyClass::h_power(1, 1)),
                ],
            )
        };
        let result = compute_by_givental_graphs(&req).unwrap();
        assert_eq!(result.value, RatFun::one());
    }

    #[test]
    fn givental_graph_reproduces_p1_degree_two_stationary_descendant() {
        let req = InvariantRequest {
            mode: ComputeMode::Givental,
            ..InvariantRequest::new(
                1,
                0,
                2,
                vec![
                    tau(2, CohomologyClass::h_power(1, 1)),
                    tau(0, CohomologyClass::h_power(1, 1)),
                    tau(0, CohomologyClass::h_power(1, 1)),
                ],
            )
        };

        let result = compute_by_givental_graphs(&req).unwrap();
        assert_eq!(result.value, RatFun::one());
    }

    #[test]
    fn givental_graph_reproduces_p1_higher_degree_stationary_descendants() {
        let degree_three = InvariantRequest {
            mode: ComputeMode::Givental,
            ..InvariantRequest::new(
                1,
                0,
                3,
                vec![
                    tau(4, CohomologyClass::h_power(1, 1)),
                    tau(0, CohomologyClass::h_power(1, 1)),
                    tau(0, CohomologyClass::h_power(1, 1)),
                ],
            )
        };
        let result = compute_by_givental_graphs(&degree_three).unwrap();
        assert_eq!(
            result.value,
            RatFun::from_rational(crate::algebra::Rational::new(1, 4))
        );

        let degree_four = InvariantRequest {
            mode: ComputeMode::Givental,
            ..InvariantRequest::new(
                1,
                0,
                4,
                vec![
                    tau(6, CohomologyClass::h_power(1, 1)),
                    tau(0, CohomologyClass::h_power(1, 1)),
                    tau(0, CohomologyClass::h_power(1, 1)),
                ],
            )
        };
        let result = compute_by_givental_graphs(&degree_four).unwrap();
        assert_eq!(
            result.value,
            RatFun::from_rational(crate::algebra::Rational::new(1, 36))
        );
    }

    fn assert_r_matrix_unitary_after_lambda_eval(
        r: &SeriesRMatrix,
        metric: &SeriesMatrix,
        target_n: usize,
        weights: &[crate::algebra::Rational],
    ) {
        for z_degree in 0..=r.z_order() {
            let mut total = SeriesMatrix::zero(r.size(), r.size(), r.q_degree());
            for left_order in 0..=z_degree {
                let right_order = z_degree - left_order;
                let term = r
                    .coefficient(left_order)
                    .unwrap()
                    .transpose()
                    .mul(metric)
                    .mul(r.coefficient(right_order).unwrap());
                total = if left_order % 2 == 0 {
                    total.add(&term)
                } else {
                    total.sub(&term)
                };
            }
            let expected = if z_degree == 0 {
                metric.clone()
            } else {
                SeriesMatrix::zero(r.size(), r.size(), r.q_degree())
            };
            assert_series_matrix_equal_after_lambda_eval(&total, &expected, target_n, weights);
        }
    }

    fn assert_series_matrix_equal_after_lambda_eval(
        left: &SeriesMatrix,
        right: &SeriesMatrix,
        target_n: usize,
        weights: &[crate::algebra::Rational],
    ) {
        assert_eq!(left.rows(), right.rows());
        assert_eq!(left.cols(), right.cols());
        for row in 0..left.rows() {
            for col in 0..left.cols() {
                let left_series = left.entry(row, col);
                let right_series = right.entry(row, col);
                assert_eq!(left_series.max_degree(), right_series.max_degree());
                for degree in 0..=left_series.max_degree() {
                    let diff =
                        left_series.coeff(degree).unwrap() - right_series.coeff(degree).unwrap();
                    let value = diff
                        .evaluate_lambda_weights(target_n, weights)
                        .expect("test specialization should avoid poles");
                    assert_eq!(value, crate::algebra::Rational::zero());
                }
            }
        }
    }
}
