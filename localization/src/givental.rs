use crate::algebra::{lambda, RatFun, Rational};
use crate::error::GwError;
use crate::frobenius::FrobeniusData;
use crate::geometry::elementary_symmetric_weights;
use crate::graphs::{stable_graphs, StableGraph};
use crate::series::{QSeries, SeriesMatrix};
use crate::tautological::{TautologicalOracle, WittenKontsevich};
use crate::validation;
use crate::{
    ComputeMode, Insertion, InvariantRequest, InvariantResult, SeriesCoefficient, SeriesRequest,
    SeriesResult, Truncation,
};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

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

    fn truncated(&self, z_order: usize) -> Self {
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
    {
        let cache = cache.lock().unwrap();
        if let Some(descendant_s) = cache.get(&key).cloned() {
            return Ok(descendant_s);
        }
        if let Some(descendant_s) = cache
            .iter()
            .find(|(cached_key, _)| {
                cached_key.n == n
                    && cached_key.q_degree == q_degree
                    && cached_key.z_order >= z_order
            })
            .map(|(_, descendant_s)| descendant_s.truncated(z_order))
        {
            return Ok(descendant_s);
        }
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
    {
        let cache = cache.lock().unwrap();
        if let Some(descendant_s) = cache.get(&key).cloned() {
            return Ok(descendant_s);
        }
        if let Some(descendant_s) = cache
            .iter()
            .find(|(cached_key, _)| {
                cached_key.n == n
                    && cached_key.q_degree == q_degree
                    && cached_key.weights == weights
                    && cached_key.z_order >= z_order
            })
            .map(|(_, descendant_s)| descendant_s.truncated(z_order))
        {
            return Ok(descendant_s);
        }
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

#[derive(Debug)]
struct GraphEvalProfile {
    enabled: bool,
    started: Instant,
    calibration_elapsed: Duration,
    option_elapsed: Duration,
    graph_elapsed: Duration,
    stable_graphs: usize,
    colorings: usize,
    recursion_calls: usize,
    leaves: usize,
    vertex_cache_hits: usize,
    vertex_cache_misses: usize,
    translation_terms: usize,
    leg_options: usize,
    edge_options: usize,
}

impl GraphEvalProfile {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("GW_PROFILE").is_some(),
            started: Instant::now(),
            calibration_elapsed: Duration::ZERO,
            option_elapsed: Duration::ZERO,
            graph_elapsed: Duration::ZERO,
            stable_graphs: 0,
            colorings: 0,
            recursion_calls: 0,
            leaves: 0,
            vertex_cache_hits: 0,
            vertex_cache_misses: 0,
            translation_terms: 0,
            leg_options: 0,
            edge_options: 0,
        }
    }

    fn add_calibration_elapsed(&mut self, elapsed: Duration) {
        if self.enabled {
            self.calibration_elapsed += elapsed;
        }
    }

    fn add_option_elapsed(&mut self, elapsed: Duration) {
        if self.enabled {
            self.option_elapsed += elapsed;
        }
    }

    fn add_graph_elapsed(&mut self, elapsed: Duration) {
        if self.enabled {
            self.graph_elapsed += elapsed;
        }
    }

    fn finish(&self) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "GW_PROFILE total={:.3}s calibration={:.3}s options={:.3}s graphs={:.3}s stable_graphs={} colorings={} recursion_calls={} leaves={} vertex_cache_hits={} vertex_cache_misses={} translation_terms={} leg_options={} edge_options={}",
            self.started.elapsed().as_secs_f64(),
            self.calibration_elapsed.as_secs_f64(),
            self.option_elapsed.as_secs_f64(),
            self.graph_elapsed.as_secs_f64(),
            self.stable_graphs,
            self.colorings,
            self.recursion_calls,
            self.leaves,
            self.vertex_cache_hits,
            self.vertex_cache_misses,
            self.translation_terms,
            self.leg_options,
            self.edge_options,
        );
    }

    fn absorb_graph_counts(&mut self, other: &Self) {
        self.colorings += other.colorings;
        self.recursion_calls += other.recursion_calls;
        self.leaves += other.leaves;
        self.vertex_cache_hits += other.vertex_cache_hits;
        self.vertex_cache_misses += other.vertex_cache_misses;
        self.translation_terms += other.translation_terms;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GraphKernelCacheKey {
    n: usize,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
    equivariant: bool,
}

#[derive(Debug)]
struct GraphKernel {
    calibration: ProjectiveSpaceJCalibration,
    inverse_r: Vec<SeriesMatrix>,
    translation: Vec<Vec<QSeries>>,
    edge_options: Vec<Vec<Vec<EdgeFactorOption>>>,
    vertex_cache: Mutex<HashMap<VertexContributionKey, QSeries>>,
}

fn projective_space_graph_kernel(
    n: usize,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
    equivariant: bool,
    weights: &[Rational],
) -> Result<Arc<GraphKernel>, GwError> {
    static CACHE: OnceLock<Mutex<HashMap<GraphKernelCacheKey, Arc<GraphKernel>>>> = OnceLock::new();
    let key = GraphKernelCacheKey {
        n,
        q_degree,
        r_order,
        graph_dimension,
        equivariant,
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(kernel) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(kernel);
    }

    let calibration = if equivariant {
        projective_space_j_calibration(n, q_degree, r_order)?
    } else {
        projective_space_j_calibration_at_lambda_weights(n, q_degree, r_order, weights)?
    };
    let inverse_r = inverse_r_coefficients(calibration.r_matrix.coefficients());
    let unit = calibration.relative_sqrt_delta_inverse.clone();
    let translation = translation_coefficients(&inverse_r, &unit, q_degree);
    let edge_coefficients =
        edge_propagator_coefficients(&inverse_r, &calibration.metric, graph_dimension, q_degree)?;
    let edge_options = edge_options_by_color(&edge_coefficients);

    let kernel = Arc::new(GraphKernel {
        calibration,
        inverse_r,
        translation,
        edge_options,
        vertex_cache: Mutex::new(HashMap::new()),
    });
    cache.lock().unwrap().insert(key, kernel.clone());
    Ok(kernel)
}

pub fn compute_by_givental_graphs(req: &InvariantRequest) -> Result<InvariantResult, GwError> {
    let mut profile = GraphEvalProfile::new();
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
    let calibration_started = Instant::now();
    let descendant_s = if req.equivariant {
        projective_space_descendant_s_matrix(req.n, q_degree, needed_s_order)?
    } else {
        projective_space_descendant_s_matrix_at_lambda_weights(
            req.n,
            q_degree,
            needed_s_order,
            &weights,
        )?
    };
    let kernel = projective_space_graph_kernel(
        req.n,
        q_degree,
        needed_r_order,
        graph_dimension,
        req.equivariant,
        &weights,
    )?;
    let specialized_nonequivariant = !req.equivariant;
    profile.add_calibration_elapsed(calibration_started.elapsed());

    let options_started = Instant::now();
    let insertion_terms = ancestor_insertion_terms(
        req.n,
        &req.insertions,
        &descendant_s,
        &kernel.calibration.psi_inverse,
        q_degree,
        graph_dimension,
    );
    let leg_options = leg_options_by_marking_color(
        &insertion_terms,
        &kernel.inverse_r,
        q_degree,
        graph_dimension,
        req.n + 1,
    );
    profile.leg_options = leg_options
        .iter()
        .flat_map(|by_color| by_color.iter())
        .map(Vec::len)
        .sum();
    profile.edge_options = kernel
        .edge_options
        .iter()
        .flat_map(|row| row.iter())
        .map(Vec::len)
        .sum();
    profile.add_option_elapsed(options_started.elapsed());

    let graphs = stable_graphs(req.genus, req.insertions.len());
    profile.stable_graphs = graphs.len();

    let graphs_started = Instant::now();
    let total = evaluate_scalar_graphs_parallel(
        &graphs,
        req.n + 1,
        &leg_options,
        &kernel,
        q_degree,
        graph_dimension,
        &mut profile,
    );
    profile.add_graph_elapsed(graphs_started.elapsed());

    let value = total
        .coeff(req.degree)
        .cloned()
        .unwrap_or_else(RatFun::zero);
    profile.finish();
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

const MASTER_SHARED_KERNEL_MAX_MARKINGS: usize = 2;
const MASTER_DEFAULT_MAX_WORKERS: usize = 8;
const MASTER_MIN_SHARED_KERNEL_TASKS: usize = 8;
const MASTER_MIN_RESTRICTED_KERNEL_TASKS: usize = 6;

pub fn compute_series_master(req: &SeriesRequest) -> Result<Option<SeriesResult>, GwError> {
    if req.mode != ComputeMode::Givental {
        return Ok(None);
    }

    let mut prepared_coefficients = Vec::new();
    let mut contraction_tasks = Vec::new();
    let mut restricted_contraction_tasks = (0..=req.max_markings)
        .map(|_| Vec::<MasterContractionTask>::new())
        .collect::<Vec<_>>();
    let mut notes = vec![
        "series enumerates a bounded sparse descendant potential; unsupported dimension-valid coefficients are skipped"
            .to_string(),
    ];
    let basis = crate::insertion_basis(req.n, req.max_descendant_power);
    let mut candidates_by_degree = vec![Vec::<Vec<Insertion>>::new(); req.degree_max + 1];

    for markings in 0..=req.max_markings {
        for insertions in crate::insertion_monomials(&basis, markings) {
            let probe_req = req.coefficient_request(0, insertions.clone());
            if probe_req.insertion_degree().is_some() {
                if let Some(expected_degree) = probe_req.expected_degree_from_dimension() {
                    if expected_degree <= req.degree_max {
                        candidates_by_degree[expected_degree].push(insertions);
                    }
                }
            } else {
                for bucket in &mut candidates_by_degree {
                    bucket.push(insertions.clone());
                }
            }
        }
    }

    let shared_kernel_task_counts = shared_kernel_task_counts(req, &candidates_by_degree);
    let mut evaluator = GiventalMasterEvaluator::new(req);
    evaluator.set_shared_kernel_task_counts(shared_kernel_task_counts.clone());
    for (markings, count) in shared_kernel_task_counts.iter().copied().enumerate() {
        if count >= MASTER_MIN_RESTRICTED_KERNEL_TASKS && count < MASTER_MIN_SHARED_KERNEL_TASKS {
            notes.push(format!(
                "using restricted sparse S/R graph kernel for {count} coefficient(s) with {markings} marking(s); shared kernel threshold is {MASTER_MIN_SHARED_KERNEL_TASKS}"
            ));
        } else if count > 0 && count < MASTER_MIN_RESTRICTED_KERNEL_TASKS {
            notes.push(format!(
                "using scalar S/R graph evaluation for {count} coefficient(s) with {markings} marking(s); restricted kernel threshold is {MASTER_MIN_RESTRICTED_KERNEL_TASKS}"
            ));
        }
    }

    let mut ordinal = 0usize;
    for (degree, candidates) in candidates_by_degree.into_iter().enumerate() {
        let mut candidates = candidates;
        candidates.sort_by_key(|insertions| {
            std::cmp::Reverse(
                insertions
                    .iter()
                    .map(|insertion| insertion.descendant_power)
                    .max()
                    .unwrap_or(0),
            )
        });

        for insertions in candidates {
            let current_ordinal = ordinal;
            ordinal += 1;
            let coefficient_req = req.coefficient_request(degree, insertions.clone());
            if coefficient_req
                .insertion_degree()
                .is_some_and(|actual| actual as isize != coefficient_req.virtual_dimension())
            {
                continue;
            }

            if evaluator.can_use_external_leg_kernel(&insertions) {
                match evaluator.prepare_contraction_task(current_ordinal, degree, &insertions) {
                    Ok(task) => contraction_tasks.push(task),
                    Err(GwError::UnsupportedInvariant(msg)) => {
                        notes.push(format!(
                            "skipped q^{degree} {}: {msg}",
                            crate::insertion_monomial_label(&insertions)
                        ));
                    }
                    Err(err) => return Err(err),
                }
                continue;
            }

            if evaluator.can_use_restricted_external_leg_kernel(&insertions) {
                match evaluator.prepare_sparse_contraction_task(
                    current_ordinal,
                    degree,
                    &insertions,
                ) {
                    Ok(task) => restricted_contraction_tasks[insertions.len()].push(task),
                    Err(GwError::UnsupportedInvariant(msg)) => {
                        notes.push(format!(
                            "skipped q^{degree} {}: {msg}",
                            crate::insertion_monomial_label(&insertions)
                        ));
                    }
                    Err(err) => return Err(err),
                }
                continue;
            }

            match evaluator.compute_coefficient(degree, &insertions) {
                Ok(value) => {
                    if req.include_zero || !value.is_zero() {
                        prepared_coefficients.push(OrderedSeriesCoefficient {
                            ordinal: current_ordinal,
                            coefficient: SeriesCoefficient {
                                degree,
                                insertions,
                                value,
                            },
                        });
                    }
                }
                Err(GwError::UnsupportedInvariant(msg)) => {
                    notes.push(format!(
                        "skipped q^{degree} {}: {msg}",
                        crate::insertion_monomial_label(&insertions)
                    ));
                }
                Err(err) => return Err(err),
            }
        }
    }

    prepared_coefficients
        .extend(evaluator.contract_tasks_parallel(&contraction_tasks, req.include_zero));
    for (markings, tasks) in restricted_contraction_tasks.iter().enumerate() {
        if tasks.is_empty() {
            continue;
        }
        prepared_coefficients.extend(evaluator.contract_restricted_tasks(
            markings,
            tasks,
            req.include_zero,
        )?);
    }
    prepared_coefficients.sort_by_key(|entry| entry.ordinal);
    let coefficients = prepared_coefficients
        .into_iter()
        .map(|entry| entry.coefficient)
        .collect();

    Ok(Some(SeriesResult {
        coefficients,
        engine: "givental-master-series",
        notes,
    }))
}

fn shared_kernel_task_counts(
    req: &SeriesRequest,
    candidates_by_degree: &[Vec<Vec<Insertion>>],
) -> Vec<usize> {
    let mut counts = vec![0usize; req.max_markings + 1];
    for (degree, candidates) in candidates_by_degree.iter().enumerate() {
        for insertions in candidates {
            let coefficient_req = req.coefficient_request(degree, insertions.clone());
            if coefficient_req
                .insertion_degree()
                .is_some_and(|actual| actual as isize != coefficient_req.virtual_dimension())
            {
                continue;
            }
            let markings = insertions.len();
            if markings > 0
                && markings <= MASTER_SHARED_KERNEL_MAX_MARKINGS
                && is_stable_cohft_range(req.genus, markings)
            {
                counts[markings] += 1;
            }
        }
    }
    counts
}

#[derive(Debug)]
struct GiventalMasterEvaluator {
    n: usize,
    genus: usize,
    degree_max: usize,
    max_descendant_power: usize,
    equivariant: bool,
    truncation: Option<Truncation>,
    weights: Vec<Rational>,
    descendant_s_cache: HashMap<usize, SeriesSMatrix>,
    external_kernel_cache: HashMap<usize, ExternalLegKernel>,
    leg_options_cache: HashMap<MasterLegOptionsKey, Vec<Vec<LegFactorOption>>>,
    shared_kernel_task_counts: Option<Vec<usize>>,
}

impl GiventalMasterEvaluator {
    fn new(req: &SeriesRequest) -> Self {
        Self {
            n: req.n,
            genus: req.genus,
            degree_max: req.degree_max,
            max_descendant_power: req.max_descendant_power,
            equivariant: req.equivariant,
            truncation: req.truncation.clone(),
            weights: (1..=req.n + 1).map(Rational::from).collect(),
            descendant_s_cache: HashMap::new(),
            external_kernel_cache: HashMap::new(),
            leg_options_cache: HashMap::new(),
            shared_kernel_task_counts: None,
        }
    }

    fn set_shared_kernel_task_counts(&mut self, counts: Vec<usize>) {
        self.shared_kernel_task_counts = Some(counts);
    }

    fn compute_coefficient(
        &mut self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<RatFun, GwError> {
        let coefficient_req = self.coefficient_request(degree, insertions.to_vec());
        if coefficient_req
            .insertion_degree()
            .is_some_and(|actual| actual as isize != coefficient_req.virtual_dimension())
        {
            return Ok(RatFun::zero());
        }

        let markings = insertions.len();
        if !self.can_use_external_leg_kernel(insertions) {
            return Ok(crate::compute(coefficient_req)?.value);
        }

        self.validate_truncation(markings, insertions)?;
        let mut leg_options = Vec::with_capacity(markings);
        for insertion in insertions {
            leg_options.push(self.leg_options_for_insertion(markings, insertion)?);
        }
        let kernel = self.external_leg_kernel(markings)?;
        let contracted = contract_external_leg_kernel(kernel, &leg_options);
        Ok(contracted
            .coeff(degree)
            .cloned()
            .unwrap_or_else(RatFun::zero))
    }

    fn can_use_external_leg_kernel(&self, insertions: &[Insertion]) -> bool {
        let markings = insertions.len();
        if !(markings > 0
            && markings <= MASTER_SHARED_KERNEL_MAX_MARKINGS
            && is_stable_cohft_range(self.genus, markings))
        {
            return false;
        }
        self.shared_kernel_task_counts
            .as_ref()
            .and_then(|counts| counts.get(markings))
            .is_none_or(|count| *count >= MASTER_MIN_SHARED_KERNEL_TASKS)
    }

    fn can_use_restricted_external_leg_kernel(&self, insertions: &[Insertion]) -> bool {
        let markings = insertions.len();
        if !(markings > 0
            && markings <= MASTER_SHARED_KERNEL_MAX_MARKINGS
            && is_stable_cohft_range(self.genus, markings))
        {
            return false;
        }
        self.shared_kernel_task_counts
            .as_ref()
            .and_then(|counts| counts.get(markings))
            .is_some_and(|count| {
                *count >= MASTER_MIN_RESTRICTED_KERNEL_TASKS
                    && *count < MASTER_MIN_SHARED_KERNEL_TASKS
            })
    }

    fn prepare_contraction_task(
        &mut self,
        ordinal: usize,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<MasterContractionTask, GwError> {
        if !self.can_use_external_leg_kernel(insertions) {
            return Err(GwError::UnsupportedInvariant(
                "external-leg master kernel is not available for this marking count".to_string(),
            ));
        }

        let markings = insertions.len();
        self.validate_truncation(markings, insertions)?;
        let mut leg_options = Vec::with_capacity(markings);
        for insertion in insertions {
            leg_options.push(self.leg_options_for_insertion(markings, insertion)?);
        }
        self.external_leg_kernel(markings)?;
        Ok(MasterContractionTask {
            ordinal,
            degree,
            insertions: insertions.to_vec(),
            markings,
            leg_options,
        })
    }

    fn prepare_sparse_contraction_task(
        &mut self,
        ordinal: usize,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<MasterContractionTask, GwError> {
        if !self.can_use_restricted_external_leg_kernel(insertions) {
            return Err(GwError::UnsupportedInvariant(
                "restricted external-leg kernel is not available for this marking count"
                    .to_string(),
            ));
        }

        let markings = insertions.len();
        self.validate_truncation(markings, insertions)?;
        let mut leg_options = Vec::with_capacity(markings);
        for insertion in insertions {
            leg_options.push(self.leg_options_for_insertion(markings, insertion)?);
        }
        Ok(MasterContractionTask {
            ordinal,
            degree,
            insertions: insertions.to_vec(),
            markings,
            leg_options,
        })
    }

    fn contract_tasks_parallel(
        &self,
        tasks: &[MasterContractionTask],
        include_zero: bool,
    ) -> Vec<OrderedSeriesCoefficient> {
        let worker_count = master_worker_count(tasks.len());
        if worker_count <= 1 {
            return contract_task_chunk(&self.external_kernel_cache, tasks, include_zero);
        }

        let chunk_size = tasks.len().div_ceil(worker_count);
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in tasks.chunks(chunk_size) {
                let kernels = &self.external_kernel_cache;
                handles
                    .push(scope.spawn(move || contract_task_chunk(kernels, chunk, include_zero)));
            }

            let mut out = Vec::new();
            for handle in handles {
                out.extend(
                    handle
                        .join()
                        .expect("master series contraction worker panicked"),
                );
            }
            out
        })
    }

    fn contract_restricted_tasks(
        &self,
        markings: usize,
        tasks: &[MasterContractionTask],
        include_zero: bool,
    ) -> Result<Vec<OrderedSeriesCoefficient>, GwError> {
        let kernel = self.build_restricted_external_leg_kernel(markings, tasks)?;
        Ok(contract_restricted_tasks_parallel(
            &kernel,
            tasks,
            include_zero,
        ))
    }

    fn build_restricted_external_leg_kernel(
        &self,
        markings: usize,
        tasks: &[MasterContractionTask],
    ) -> Result<RestrictedExternalLegKernel, GwError> {
        let graph_dimension = self.graph_dimension(markings);
        let graph_kernel = self.graph_kernel_for_markings(markings)?;
        let template = RestrictedExternalLegKernel::from_tasks(
            markings,
            self.n + 1,
            graph_dimension,
            self.degree_max,
            tasks,
        );
        let graphs = stable_graphs(self.genus, markings);
        let mut profile = GraphEvalProfile::new();
        profile.stable_graphs = graphs.len();
        profile.edge_options = graph_kernel
            .edge_options
            .iter()
            .flat_map(|row| row.iter())
            .map(Vec::len)
            .sum();

        let graphs_started = Instant::now();
        let total = evaluate_restricted_external_graphs_parallel(
            &graphs,
            &template,
            &graph_kernel,
            self.degree_max,
            graph_dimension,
            &mut profile,
        );
        profile.add_graph_elapsed(graphs_started.elapsed());
        profile.finish();
        Ok(total)
    }

    fn coefficient_request(&self, degree: usize, insertions: Vec<Insertion>) -> InvariantRequest {
        InvariantRequest {
            n: self.n,
            genus: self.genus,
            degree,
            insertions,
            equivariant: self.equivariant,
            mode: ComputeMode::Givental,
            truncation: self.truncation.clone(),
        }
    }

    fn validate_truncation(
        &self,
        markings: usize,
        insertions: &[Insertion],
    ) -> Result<(), GwError> {
        let graph_dimension = self.graph_dimension(markings);
        let needed_r_order = graph_dimension + 1;
        let needed_s_order = insertions
            .iter()
            .map(|insertion| insertion.descendant_power)
            .max()
            .unwrap_or(0);
        let needed_z_order = needed_r_order.max(needed_s_order);
        if self
            .truncation
            .as_ref()
            .is_some_and(|truncation| truncation.z_order < needed_z_order)
        {
            return Err(GwError::TruncationTooLow);
        }
        Ok(())
    }

    fn graph_dimension(&self, markings: usize) -> usize {
        3 * self.genus + markings - 3
    }

    fn descendant_s(&mut self, z_order: usize) -> Result<&SeriesSMatrix, GwError> {
        if !self.descendant_s_cache.contains_key(&z_order) {
            let descendant_s = if self.equivariant {
                projective_space_descendant_s_matrix(self.n, self.degree_max, z_order)?
            } else {
                projective_space_descendant_s_matrix_at_lambda_weights(
                    self.n,
                    self.degree_max,
                    z_order,
                    &self.weights,
                )?
            };
            self.descendant_s_cache.insert(z_order, descendant_s);
        }
        Ok(self
            .descendant_s_cache
            .get(&z_order)
            .expect("descendant S-matrix cache populated before access"))
    }

    fn graph_kernel_for_markings(&self, markings: usize) -> Result<Arc<GraphKernel>, GwError> {
        let graph_dimension = self.graph_dimension(markings);
        projective_space_graph_kernel(
            self.n,
            self.degree_max,
            graph_dimension + 1,
            graph_dimension,
            self.equivariant,
            &self.weights,
        )
    }

    fn external_leg_kernel(&mut self, markings: usize) -> Result<&ExternalLegKernel, GwError> {
        if !self.external_kernel_cache.contains_key(&markings) {
            let kernel = self.build_external_leg_kernel(markings)?;
            self.external_kernel_cache.insert(markings, kernel);
        }
        Ok(self
            .external_kernel_cache
            .get(&markings)
            .expect("external leg kernel cache populated before access"))
    }

    fn build_external_leg_kernel(&self, markings: usize) -> Result<ExternalLegKernel, GwError> {
        let graph_dimension = self.graph_dimension(markings);
        let graph_kernel = self.graph_kernel_for_markings(markings)?;
        let graphs = stable_graphs(self.genus, markings);
        let mut profile = GraphEvalProfile::new();
        profile.stable_graphs = graphs.len();
        profile.edge_options = graph_kernel
            .edge_options
            .iter()
            .flat_map(|row| row.iter())
            .map(Vec::len)
            .sum();

        let graphs_started = Instant::now();
        let total = evaluate_external_graphs_parallel(
            &graphs,
            markings,
            self.n + 1,
            &graph_kernel,
            self.degree_max,
            graph_dimension,
            &mut profile,
        );
        profile.add_graph_elapsed(graphs_started.elapsed());
        profile.finish();

        Ok(total)
    }

    fn leg_options_for_insertion(
        &mut self,
        markings: usize,
        insertion: &Insertion,
    ) -> Result<Vec<Vec<LegFactorOption>>, GwError> {
        let cache_key = insertion
            .class
            .pure_power()
            .map(|class_power| MasterLegOptionsKey {
                markings,
                descendant_power: insertion.descendant_power,
                class_power,
            });
        if let Some(key) = cache_key {
            if let Some(cached) = self.leg_options_cache.get(&key) {
                return Ok(cached.clone());
            }
        }

        let graph_dimension = self.graph_dimension(markings);
        let s_order = self.max_descendant_power.max(insertion.descendant_power);
        let graph_kernel = self.graph_kernel_for_markings(markings)?;
        let n = self.n;
        let q_degree = self.degree_max;
        let colors = self.n + 1;
        let psi_inverse = graph_kernel.calibration.psi_inverse.clone();
        let inverse_r = graph_kernel.inverse_r.clone();
        let insertion_terms = {
            let descendant_s = self.descendant_s(s_order)?;
            ancestor_insertion_terms(
                n,
                std::slice::from_ref(insertion),
                descendant_s,
                &psi_inverse,
                q_degree,
                graph_dimension,
            )
        };
        let mut options = leg_options_by_marking_color(
            &insertion_terms,
            &inverse_r,
            q_degree,
            graph_dimension,
            colors,
        );
        let by_color = options.pop().unwrap_or_else(|| vec![Vec::new(); colors]);
        if let Some(key) = cache_key {
            self.leg_options_cache.insert(key, by_color.clone());
        }
        Ok(by_color)
    }
}

#[derive(Debug)]
struct MasterContractionTask {
    ordinal: usize,
    degree: usize,
    insertions: Vec<Insertion>,
    markings: usize,
    leg_options: Vec<Vec<Vec<LegFactorOption>>>,
}

#[derive(Debug)]
struct OrderedSeriesCoefficient {
    ordinal: usize,
    coefficient: SeriesCoefficient,
}

fn master_worker_count(work_items: usize) -> usize {
    if work_items < 8 {
        return 1;
    }
    let available = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let requested = std::env::var("GW_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or_else(|| available.min(MASTER_DEFAULT_MAX_WORKERS));
    requested.min(work_items).max(1)
}

fn contract_task_chunk(
    kernels: &HashMap<usize, ExternalLegKernel>,
    tasks: &[MasterContractionTask],
    include_zero: bool,
) -> Vec<OrderedSeriesCoefficient> {
    let mut out = Vec::new();
    for task in tasks {
        let kernel = kernels
            .get(&task.markings)
            .expect("prepared master task must have a cached external-leg kernel");
        let contracted = contract_external_leg_kernel(kernel, &task.leg_options);
        let value = contracted
            .coeff(task.degree)
            .cloned()
            .unwrap_or_else(RatFun::zero);
        if include_zero || !value.is_zero() {
            out.push(OrderedSeriesCoefficient {
                ordinal: task.ordinal,
                coefficient: SeriesCoefficient {
                    degree: task.degree,
                    insertions: task.insertions.clone(),
                    value,
                },
            });
        }
    }
    out
}

fn contract_restricted_tasks_parallel(
    kernel: &RestrictedExternalLegKernel,
    tasks: &[MasterContractionTask],
    include_zero: bool,
) -> Vec<OrderedSeriesCoefficient> {
    let worker_count = master_worker_count(tasks.len());
    if worker_count <= 1 {
        return contract_restricted_task_chunk(kernel, tasks, include_zero);
    }

    let chunk_size = tasks.len().div_ceil(worker_count);
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in tasks.chunks(chunk_size) {
            handles.push(
                scope.spawn(move || contract_restricted_task_chunk(kernel, chunk, include_zero)),
            );
        }

        let mut out = Vec::new();
        for handle in handles {
            out.extend(
                handle
                    .join()
                    .expect("restricted master series contraction worker panicked"),
            );
        }
        out
    })
}

fn contract_restricted_task_chunk(
    kernel: &RestrictedExternalLegKernel,
    tasks: &[MasterContractionTask],
    include_zero: bool,
) -> Vec<OrderedSeriesCoefficient> {
    let mut out = Vec::new();
    for task in tasks {
        let contracted = contract_restricted_external_leg_kernel(kernel, &task.leg_options);
        let value = contracted
            .coeff(task.degree)
            .cloned()
            .unwrap_or_else(RatFun::zero);
        if include_zero || !value.is_zero() {
            out.push(OrderedSeriesCoefficient {
                ordinal: task.ordinal,
                coefficient: SeriesCoefficient {
                    degree: task.degree,
                    insertions: task.insertions.clone(),
                    value,
                },
            });
        }
    }
    out
}

struct ScalarGraphChunkResult {
    total: QSeries,
    profile: GraphEvalProfile,
    vertex_cache: HashMap<VertexContributionKey, QSeries>,
}

fn evaluate_scalar_graphs_parallel(
    graphs: &[StableGraph],
    colors: usize,
    leg_options: &[Vec<Vec<LegFactorOption>>],
    kernel: &Arc<GraphKernel>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> QSeries {
    let worker_count = master_worker_count(graphs.len());
    let initial_vertex_cache = kernel.vertex_cache.lock().unwrap().clone();
    let results = if worker_count <= 1 {
        vec![evaluate_scalar_graph_chunk(
            graphs,
            colors,
            leg_options,
            kernel,
            q_degree,
            graph_dimension,
            initial_vertex_cache,
        )]
    } else {
        let chunk_size = graphs.len().div_ceil(worker_count);
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in graphs.chunks(chunk_size) {
                let local_vertex_cache = initial_vertex_cache.clone();
                handles.push(scope.spawn(move || {
                    evaluate_scalar_graph_chunk(
                        chunk,
                        colors,
                        leg_options,
                        kernel,
                        q_degree,
                        graph_dimension,
                        local_vertex_cache,
                    )
                }));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .expect("scalar graph evaluation worker panicked")
                })
                .collect::<Vec<_>>()
        })
    };

    let mut total = QSeries::zero(q_degree);
    let mut shared_vertex_cache = kernel.vertex_cache.lock().unwrap();
    for result in results {
        profile.absorb_graph_counts(&result.profile);
        total = total.add(&result.total);
        for (key, value) in result.vertex_cache {
            shared_vertex_cache.entry(key).or_insert(value);
        }
    }
    total
}

fn evaluate_scalar_graph_chunk(
    graphs: &[StableGraph],
    colors: usize,
    leg_options: &[Vec<Vec<LegFactorOption>>],
    kernel: &GraphKernel,
    q_degree: usize,
    graph_dimension: usize,
    mut vertex_cache: HashMap<VertexContributionKey, QSeries>,
) -> ScalarGraphChunkResult {
    let oracle = WittenKontsevich::new();
    let mut profile = GraphEvalProfile::new();
    let mut total = QSeries::zero(q_degree);
    for graph in graphs {
        let automorphism_factor = &RatFun::one() / &RatFun::from(graph.automorphism_order());
        let colorings = vertex_coloring_orbits(graph, colors);
        profile.colorings += colorings.len();
        let vertex_power_caps = graph
            .vertices
            .iter()
            .enumerate()
            .map(|(vertex, stable_vertex)| 3 * stable_vertex.genus + graph.valence(vertex) - 3)
            .collect::<Vec<_>>();
        for coloring in colorings.iter() {
            let coloring_factor = &automorphism_factor * &RatFun::from(coloring.multiplicity);
            let mut graph_total = QSeries::zero(q_degree);
            let mut base_powers = vec![Vec::<usize>::new(); graph.vertices.len()];
            let mut vertex_power_sums = vec![0usize; graph.vertices.len()];
            accumulate_graph_factors(
                graph,
                &coloring.colors,
                leg_options,
                &kernel.edge_options,
                &kernel.calibration,
                &kernel.translation,
                &oracle,
                &mut vertex_cache,
                q_degree,
                graph_dimension,
                0,
                0,
                QSeries::one(q_degree),
                &mut base_powers,
                &mut vertex_power_sums,
                &vertex_power_caps,
                &mut graph_total,
                &mut profile,
            );

            total = total.add(&graph_total.scale(&coloring_factor));
        }
    }
    ScalarGraphChunkResult {
        total,
        profile,
        vertex_cache,
    }
}

struct ExternalGraphChunkResult {
    total: ExternalLegKernel,
    profile: GraphEvalProfile,
    vertex_cache: HashMap<VertexContributionKey, QSeries>,
}

fn evaluate_external_graphs_parallel(
    graphs: &[StableGraph],
    markings: usize,
    colors: usize,
    kernel: &Arc<GraphKernel>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> ExternalLegKernel {
    let worker_count = master_worker_count(graphs.len());
    let initial_vertex_cache = kernel.vertex_cache.lock().unwrap().clone();
    let results = if worker_count <= 1 {
        vec![evaluate_external_graph_chunk(
            graphs,
            markings,
            colors,
            kernel,
            q_degree,
            graph_dimension,
            initial_vertex_cache,
        )]
    } else {
        let chunk_size = graphs.len().div_ceil(worker_count);
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in graphs.chunks(chunk_size) {
                let local_vertex_cache = initial_vertex_cache.clone();
                handles.push(scope.spawn(move || {
                    evaluate_external_graph_chunk(
                        chunk,
                        markings,
                        colors,
                        kernel,
                        q_degree,
                        graph_dimension,
                        local_vertex_cache,
                    )
                }));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .expect("external graph kernel worker panicked")
                })
                .collect::<Vec<_>>()
        })
    };

    let mut total = ExternalLegKernel::zero(markings, colors, graph_dimension, q_degree);
    let mut shared_vertex_cache = kernel.vertex_cache.lock().unwrap();
    for result in results {
        profile.absorb_graph_counts(&result.profile);
        total.add_assign(&result.total);
        for (key, value) in result.vertex_cache {
            shared_vertex_cache.entry(key).or_insert(value);
        }
    }
    total
}

fn evaluate_external_graph_chunk(
    graphs: &[StableGraph],
    markings: usize,
    colors: usize,
    kernel: &GraphKernel,
    q_degree: usize,
    graph_dimension: usize,
    mut vertex_cache: HashMap<VertexContributionKey, QSeries>,
) -> ExternalGraphChunkResult {
    let oracle = WittenKontsevich::new();
    let mut profile = GraphEvalProfile::new();
    let mut total = ExternalLegKernel::zero(markings, colors, graph_dimension, q_degree);
    for graph in graphs {
        let automorphism_factor = &RatFun::one() / &RatFun::from(graph.automorphism_order());
        let colorings = vertex_coloring_orbits(graph, colors);
        profile.colorings += colorings.len();
        let vertex_power_caps = graph
            .vertices
            .iter()
            .enumerate()
            .map(|(vertex, stable_vertex)| 3 * stable_vertex.genus + graph.valence(vertex) - 3)
            .collect::<Vec<_>>();

        for coloring in colorings.iter() {
            let coloring_factor = &automorphism_factor * &RatFun::from(coloring.multiplicity);
            let mut base_powers = vec![Vec::<usize>::new(); graph.vertices.len()];
            let mut vertex_power_sums = vec![0usize; graph.vertices.len()];
            let mut external_states = Vec::with_capacity(markings);
            accumulate_external_leg_graph_factors(
                graph,
                &coloring.colors,
                &kernel.edge_options,
                &kernel.calibration,
                &kernel.translation,
                &oracle,
                &mut vertex_cache,
                q_degree,
                graph_dimension,
                0,
                0,
                QSeries::one(q_degree).scale(&coloring_factor),
                &mut base_powers,
                &mut vertex_power_sums,
                &vertex_power_caps,
                &mut external_states,
                &mut total,
                &mut profile,
            );
        }
    }
    ExternalGraphChunkResult {
        total,
        profile,
        vertex_cache,
    }
}

struct RestrictedExternalGraphChunkResult {
    total: RestrictedExternalLegKernel,
    profile: GraphEvalProfile,
    vertex_cache: HashMap<VertexContributionKey, QSeries>,
}

fn evaluate_restricted_external_graphs_parallel(
    graphs: &[StableGraph],
    template: &RestrictedExternalLegKernel,
    kernel: &Arc<GraphKernel>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> RestrictedExternalLegKernel {
    let worker_count = master_worker_count(graphs.len());
    let initial_vertex_cache = kernel.vertex_cache.lock().unwrap().clone();
    let results = if worker_count <= 1 {
        vec![evaluate_restricted_external_graph_chunk(
            graphs,
            template,
            kernel,
            q_degree,
            graph_dimension,
            initial_vertex_cache,
        )]
    } else {
        let chunk_size = graphs.len().div_ceil(worker_count);
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in graphs.chunks(chunk_size) {
                let local_vertex_cache = initial_vertex_cache.clone();
                handles.push(scope.spawn(move || {
                    evaluate_restricted_external_graph_chunk(
                        chunk,
                        template,
                        kernel,
                        q_degree,
                        graph_dimension,
                        local_vertex_cache,
                    )
                }));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .expect("restricted external graph kernel worker panicked")
                })
                .collect::<Vec<_>>()
        })
    };

    let mut total = template.zero_like();
    let mut shared_vertex_cache = kernel.vertex_cache.lock().unwrap();
    for result in results {
        profile.absorb_graph_counts(&result.profile);
        total.add_assign(&result.total);
        for (key, value) in result.vertex_cache {
            shared_vertex_cache.entry(key).or_insert(value);
        }
    }
    total
}

fn evaluate_restricted_external_graph_chunk(
    graphs: &[StableGraph],
    template: &RestrictedExternalLegKernel,
    kernel: &GraphKernel,
    q_degree: usize,
    graph_dimension: usize,
    mut vertex_cache: HashMap<VertexContributionKey, QSeries>,
) -> RestrictedExternalGraphChunkResult {
    let oracle = WittenKontsevich::new();
    let mut profile = GraphEvalProfile::new();
    let mut total = template.zero_like();
    for graph in graphs {
        let automorphism_factor = &RatFun::one() / &RatFun::from(graph.automorphism_order());
        let colorings = vertex_coloring_orbits(graph, template.colors);
        profile.colorings += colorings.len();
        let vertex_power_caps = graph
            .vertices
            .iter()
            .enumerate()
            .map(|(vertex, stable_vertex)| 3 * stable_vertex.genus + graph.valence(vertex) - 3)
            .collect::<Vec<_>>();

        for coloring in colorings.iter() {
            let coloring_factor = &automorphism_factor * &RatFun::from(coloring.multiplicity);
            let mut base_powers = vec![Vec::<usize>::new(); graph.vertices.len()];
            let mut vertex_power_sums = vec![0usize; graph.vertices.len()];
            let mut external_state_indices = Vec::with_capacity(template.markings);
            accumulate_restricted_external_leg_graph_factors(
                graph,
                &coloring.colors,
                &kernel.edge_options,
                &kernel.calibration,
                &kernel.translation,
                &oracle,
                &mut vertex_cache,
                q_degree,
                graph_dimension,
                0,
                0,
                QSeries::one(q_degree).scale(&coloring_factor),
                &mut base_powers,
                &mut vertex_power_sums,
                &vertex_power_caps,
                &mut external_state_indices,
                &template.states_by_marking_color,
                &mut total,
                &mut profile,
            );
        }
    }
    RestrictedExternalGraphChunkResult {
        total,
        profile,
        vertex_cache,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MasterLegOptionsKey {
    markings: usize,
    descendant_power: usize,
    class_power: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExternalLegState {
    color: usize,
    power: usize,
}

#[derive(Debug, Clone)]
struct ExternalLegKernel {
    markings: usize,
    colors: usize,
    max_power: usize,
    q_degree: usize,
    state_count: usize,
    entries: Vec<QSeries>,
}

impl ExternalLegKernel {
    fn zero(markings: usize, colors: usize, max_power: usize, q_degree: usize) -> Self {
        let state_count = colors * (max_power + 1);
        let entries = vec![QSeries::zero(q_degree); state_count.pow(markings as u32)];
        Self {
            markings,
            colors,
            max_power,
            q_degree,
            state_count,
            entries,
        }
    }

    fn state_index(&self, color: usize, power: usize) -> usize {
        debug_assert!(color < self.colors);
        debug_assert!(power <= self.max_power);
        color * (self.max_power + 1) + power
    }

    fn tensor_index(&self, states: &[ExternalLegState]) -> usize {
        debug_assert_eq!(states.len(), self.markings);
        let mut index = 0usize;
        for state in states {
            index = index * self.state_count + self.state_index(state.color, state.power);
        }
        index
    }

    fn tensor_index_from_state_indices(&self, state_indices: &[usize]) -> usize {
        debug_assert_eq!(state_indices.len(), self.markings);
        let mut index = 0usize;
        for state_index in state_indices {
            index = index * self.state_count + state_index;
        }
        index
    }

    fn add_term(&mut self, states: &[ExternalLegState], value: &QSeries) {
        if value.is_zero() {
            return;
        }
        let index = self.tensor_index(states);
        self.entries[index] = self.entries[index].add(value);
    }

    fn add_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(self.markings, rhs.markings);
        debug_assert_eq!(self.colors, rhs.colors);
        debug_assert_eq!(self.max_power, rhs.max_power);
        debug_assert_eq!(self.q_degree, rhs.q_degree);
        for (left, right) in self.entries.iter_mut().zip(rhs.entries.iter()) {
            if !right.is_zero() {
                *left = left.add(right);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RestrictedLegState {
    power: usize,
    state_index: usize,
}

#[derive(Debug, Clone)]
struct RestrictedExternalLegKernel {
    markings: usize,
    colors: usize,
    max_power: usize,
    q_degree: usize,
    states_by_marking_color: Vec<Vec<Vec<RestrictedLegState>>>,
    state_index_by_marking_color_power: Vec<Vec<Vec<Option<usize>>>>,
    state_counts: Vec<usize>,
    strides: Vec<usize>,
    entries: Vec<QSeries>,
}

impl RestrictedExternalLegKernel {
    fn from_tasks(
        markings: usize,
        colors: usize,
        max_power: usize,
        q_degree: usize,
        tasks: &[MasterContractionTask],
    ) -> Self {
        let mut powers_by_marking_color = vec![vec![BTreeSet::<usize>::new(); colors]; markings];
        for task in tasks {
            debug_assert_eq!(task.markings, markings);
            for (marking, by_color) in task.leg_options.iter().enumerate() {
                for (color, options) in by_color.iter().enumerate() {
                    for option in options {
                        if option.power <= max_power {
                            powers_by_marking_color[marking][color].insert(option.power);
                        }
                    }
                }
            }
        }

        let mut states_by_marking_color = vec![vec![Vec::new(); colors]; markings];
        let mut state_index_by_marking_color_power =
            vec![vec![vec![None; max_power + 1]; colors]; markings];
        let mut state_counts = vec![0usize; markings];
        for marking in 0..markings {
            let mut state_index = 0usize;
            for color in 0..colors {
                for power in &powers_by_marking_color[marking][color] {
                    states_by_marking_color[marking][color].push(RestrictedLegState {
                        power: *power,
                        state_index,
                    });
                    state_index_by_marking_color_power[marking][color][*power] = Some(state_index);
                    state_index += 1;
                }
            }
            state_counts[marking] = state_index;
        }

        let mut strides = vec![1usize; markings];
        let mut entries_len = 1usize;
        for marking in (0..markings).rev() {
            strides[marking] = entries_len;
            entries_len = entries_len.saturating_mul(state_counts[marking]);
        }
        let entries = vec![QSeries::zero(q_degree); entries_len];

        Self {
            markings,
            colors,
            max_power,
            q_degree,
            states_by_marking_color,
            state_index_by_marking_color_power,
            state_counts,
            strides,
            entries,
        }
    }

    fn zero_like(&self) -> Self {
        let mut out = self.clone();
        out.entries = vec![QSeries::zero(self.q_degree); self.entries.len()];
        out
    }

    fn tensor_index_from_state_indices(&self, state_indices: &[usize]) -> usize {
        debug_assert_eq!(state_indices.len(), self.markings);
        state_indices
            .iter()
            .zip(self.strides.iter())
            .map(|(state_index, stride)| state_index * stride)
            .sum()
    }

    fn add_term(&mut self, state_indices: &[usize], value: &QSeries) {
        if value.is_zero() || self.entries.is_empty() {
            return;
        }
        let index = self.tensor_index_from_state_indices(state_indices);
        self.entries[index] = self.entries[index].add(value);
    }

    fn add_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(self.markings, rhs.markings);
        debug_assert_eq!(self.colors, rhs.colors);
        debug_assert_eq!(self.max_power, rhs.max_power);
        debug_assert_eq!(self.q_degree, rhs.q_degree);
        debug_assert_eq!(self.state_counts, rhs.state_counts);
        for (left, right) in self.entries.iter_mut().zip(rhs.entries.iter()) {
            if !right.is_zero() {
                *left = left.add(right);
            }
        }
    }
}

fn contract_external_leg_kernel(
    kernel: &ExternalLegKernel,
    leg_options: &[Vec<Vec<LegFactorOption>>],
) -> QSeries {
    debug_assert_eq!(kernel.markings, leg_options.len());
    let mut total = QSeries::zero(kernel.q_degree);
    let mut state_indices = vec![0usize; kernel.markings];
    contract_external_leg_kernel_rec(
        kernel,
        leg_options,
        0,
        QSeries::one(kernel.q_degree),
        &mut state_indices,
        &mut total,
    );
    total
}

fn contract_external_leg_kernel_rec(
    kernel: &ExternalLegKernel,
    leg_options: &[Vec<Vec<LegFactorOption>>],
    marking: usize,
    coefficient: QSeries,
    state_indices: &mut [usize],
    total: &mut QSeries,
) {
    if coefficient.is_zero() {
        return;
    }
    if marking == kernel.markings {
        let index = kernel.tensor_index_from_state_indices(state_indices);
        if kernel.entries[index].is_zero() {
            return;
        }
        let term = coefficient.mul(&kernel.entries[index]);
        *total = total.add(&term);
        return;
    }

    for color in 0..kernel.colors {
        for option in &leg_options[marking][color] {
            if option.power > kernel.max_power {
                continue;
            }
            let next_coefficient = coefficient.mul(&option.coefficient);
            if next_coefficient.is_zero() {
                continue;
            }
            state_indices[marking] = kernel.state_index(color, option.power);
            contract_external_leg_kernel_rec(
                kernel,
                leg_options,
                marking + 1,
                next_coefficient,
                state_indices,
                total,
            );
        }
    }
}

fn contract_restricted_external_leg_kernel(
    kernel: &RestrictedExternalLegKernel,
    leg_options: &[Vec<Vec<LegFactorOption>>],
) -> QSeries {
    debug_assert_eq!(kernel.markings, leg_options.len());
    if kernel.entries.is_empty() {
        return QSeries::zero(kernel.q_degree);
    }
    let mut total = QSeries::zero(kernel.q_degree);
    let mut state_indices = vec![0usize; kernel.markings];
    contract_restricted_external_leg_kernel_rec(
        kernel,
        leg_options,
        0,
        QSeries::one(kernel.q_degree),
        &mut state_indices,
        &mut total,
    );
    total
}

fn contract_restricted_external_leg_kernel_rec(
    kernel: &RestrictedExternalLegKernel,
    leg_options: &[Vec<Vec<LegFactorOption>>],
    marking: usize,
    coefficient: QSeries,
    state_indices: &mut [usize],
    total: &mut QSeries,
) {
    if coefficient.is_zero() {
        return;
    }
    if marking == kernel.markings {
        let index = kernel.tensor_index_from_state_indices(state_indices);
        if kernel.entries[index].is_zero() {
            return;
        }
        let term = coefficient.mul(&kernel.entries[index]);
        *total = total.add(&term);
        return;
    }

    for color in 0..kernel.colors {
        for option in &leg_options[marking][color] {
            if option.power > kernel.max_power {
                continue;
            }
            let Some(state_index) =
                kernel.state_index_by_marking_color_power[marking][color][option.power]
            else {
                continue;
            };
            let next_coefficient = coefficient.mul(&option.coefficient);
            if next_coefficient.is_zero() {
                continue;
            }
            state_indices[marking] = state_index;
            contract_restricted_external_leg_kernel_rec(
                kernel,
                leg_options,
                marking + 1,
                next_coefficient,
                state_indices,
                total,
            );
        }
    }
}

fn accumulate_external_leg_graph_factors(
    graph: &crate::graphs::StableGraph,
    colors: &[usize],
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
    vertex_power_sums: &mut [usize],
    vertex_power_caps: &[usize],
    external_states: &mut Vec<ExternalLegState>,
    total: &mut ExternalLegKernel,
    profile: &mut GraphEvalProfile,
) {
    if profile.enabled {
        profile.recursion_calls += 1;
    }
    if coefficient.is_zero() || current_power_sum > max_power {
        return;
    }

    let leg_count = graph.legs.len();
    let edge_count = graph.edges.len();
    if factor_index < leg_count {
        let marking = factor_index;
        let vertex = graph.legs[marking];
        let color = colors[vertex];
        for power in 0..=max_power - current_power_sum {
            let next_vertex_power = vertex_power_sums[vertex] + power;
            if next_vertex_power > vertex_power_caps[vertex] {
                continue;
            }
            vertex_power_sums[vertex] = next_vertex_power;
            base_powers[vertex].push(power);
            external_states.push(ExternalLegState { color, power });
            accumulate_external_leg_graph_factors(
                graph,
                colors,
                edge_options,
                calibration,
                translation,
                oracle,
                vertex_cache,
                q_degree,
                max_power,
                factor_index + 1,
                current_power_sum + power,
                coefficient.clone(),
                base_powers,
                vertex_power_sums,
                vertex_power_caps,
                external_states,
                total,
                profile,
            );
            external_states.pop();
            base_powers[vertex].pop();
            vertex_power_sums[vertex] -= power;
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
            if edge.a == edge.b {
                let next_vertex_power =
                    vertex_power_sums[edge.a] + option.left_power + option.right_power;
                if next_vertex_power > vertex_power_caps[edge.a] {
                    continue;
                }
                vertex_power_sums[edge.a] = next_vertex_power;
            } else {
                let next_left_power = vertex_power_sums[edge.a] + option.left_power;
                let next_right_power = vertex_power_sums[edge.b] + option.right_power;
                if next_left_power > vertex_power_caps[edge.a]
                    || next_right_power > vertex_power_caps[edge.b]
                {
                    continue;
                }
                vertex_power_sums[edge.a] = next_left_power;
                vertex_power_sums[edge.b] = next_right_power;
            }
            let next_coefficient = coefficient.mul(&option.coefficient);
            if next_coefficient.is_zero() {
                if edge.a == edge.b {
                    vertex_power_sums[edge.a] -= option.left_power + option.right_power;
                } else {
                    vertex_power_sums[edge.b] -= option.right_power;
                    vertex_power_sums[edge.a] -= option.left_power;
                }
                continue;
            }
            base_powers[edge.a].push(option.left_power);
            base_powers[edge.b].push(option.right_power);
            accumulate_external_leg_graph_factors(
                graph,
                colors,
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
                vertex_power_sums,
                vertex_power_caps,
                external_states,
                total,
                profile,
            );
            base_powers[edge.b].pop();
            base_powers[edge.a].pop();
            if edge.a == edge.b {
                vertex_power_sums[edge.a] -= option.left_power + option.right_power;
            } else {
                vertex_power_sums[edge.b] -= option.right_power;
                vertex_power_sums[edge.a] -= option.left_power;
            }
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
            profile,
        );
        vertex_product = vertex_product.mul(&vertex_sum);
        if vertex_product.is_zero() {
            return;
        }
    }
    if profile.enabled {
        profile.leaves += 1;
    }
    total.add_term(external_states, &coefficient.mul(&vertex_product));
}

fn accumulate_restricted_external_leg_graph_factors(
    graph: &crate::graphs::StableGraph,
    colors: &[usize],
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
    vertex_power_sums: &mut [usize],
    vertex_power_caps: &[usize],
    external_state_indices: &mut Vec<usize>,
    states_by_marking_color: &[Vec<Vec<RestrictedLegState>>],
    total: &mut RestrictedExternalLegKernel,
    profile: &mut GraphEvalProfile,
) {
    if profile.enabled {
        profile.recursion_calls += 1;
    }
    if coefficient.is_zero() || current_power_sum > max_power {
        return;
    }

    let leg_count = graph.legs.len();
    let edge_count = graph.edges.len();
    if factor_index < leg_count {
        let marking = factor_index;
        let vertex = graph.legs[marking];
        let color = colors[vertex];
        for state in &states_by_marking_color[marking][color] {
            let power = state.power;
            let next_power_sum = current_power_sum + power;
            if next_power_sum > max_power {
                continue;
            }
            let next_vertex_power = vertex_power_sums[vertex] + power;
            if next_vertex_power > vertex_power_caps[vertex] {
                continue;
            }
            vertex_power_sums[vertex] = next_vertex_power;
            base_powers[vertex].push(power);
            external_state_indices.push(state.state_index);
            accumulate_restricted_external_leg_graph_factors(
                graph,
                colors,
                edge_options,
                calibration,
                translation,
                oracle,
                vertex_cache,
                q_degree,
                max_power,
                factor_index + 1,
                next_power_sum,
                coefficient.clone(),
                base_powers,
                vertex_power_sums,
                vertex_power_caps,
                external_state_indices,
                states_by_marking_color,
                total,
                profile,
            );
            external_state_indices.pop();
            base_powers[vertex].pop();
            vertex_power_sums[vertex] -= power;
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
            if edge.a == edge.b {
                let next_vertex_power =
                    vertex_power_sums[edge.a] + option.left_power + option.right_power;
                if next_vertex_power > vertex_power_caps[edge.a] {
                    continue;
                }
                vertex_power_sums[edge.a] = next_vertex_power;
            } else {
                let next_left_power = vertex_power_sums[edge.a] + option.left_power;
                let next_right_power = vertex_power_sums[edge.b] + option.right_power;
                if next_left_power > vertex_power_caps[edge.a]
                    || next_right_power > vertex_power_caps[edge.b]
                {
                    continue;
                }
                vertex_power_sums[edge.a] = next_left_power;
                vertex_power_sums[edge.b] = next_right_power;
            }
            let next_coefficient = coefficient.mul(&option.coefficient);
            if next_coefficient.is_zero() {
                if edge.a == edge.b {
                    vertex_power_sums[edge.a] -= option.left_power + option.right_power;
                } else {
                    vertex_power_sums[edge.b] -= option.right_power;
                    vertex_power_sums[edge.a] -= option.left_power;
                }
                continue;
            }
            base_powers[edge.a].push(option.left_power);
            base_powers[edge.b].push(option.right_power);
            accumulate_restricted_external_leg_graph_factors(
                graph,
                colors,
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
                vertex_power_sums,
                vertex_power_caps,
                external_state_indices,
                states_by_marking_color,
                total,
                profile,
            );
            base_powers[edge.b].pop();
            base_powers[edge.a].pop();
            if edge.a == edge.b {
                vertex_power_sums[edge.a] -= option.left_power + option.right_power;
            } else {
                vertex_power_sums[edge.b] -= option.right_power;
                vertex_power_sums[edge.a] -= option.left_power;
            }
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
            profile,
        );
        vertex_product = vertex_product.mul(&vertex_sum);
        if vertex_product.is_zero() {
            return;
        }
    }
    if profile.enabled {
        profile.leaves += 1;
    }
    total.add_term(external_state_indices, &coefficient.mul(&vertex_product));
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
    vertex_power_sums: &mut [usize],
    vertex_power_caps: &[usize],
    total: &mut QSeries,
    profile: &mut GraphEvalProfile,
) {
    if profile.enabled {
        profile.recursion_calls += 1;
    }
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
            let next_vertex_power = vertex_power_sums[vertex] + option.power;
            if next_vertex_power > vertex_power_caps[vertex] {
                continue;
            }
            let next_coefficient = coefficient.mul(&option.coefficient);
            if next_coefficient.is_zero() {
                continue;
            }
            vertex_power_sums[vertex] = next_vertex_power;
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
                vertex_power_sums,
                vertex_power_caps,
                total,
                profile,
            );
            base_powers[vertex].pop();
            vertex_power_sums[vertex] -= option.power;
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
            if edge.a == edge.b {
                let next_vertex_power =
                    vertex_power_sums[edge.a] + option.left_power + option.right_power;
                if next_vertex_power > vertex_power_caps[edge.a] {
                    continue;
                }
                vertex_power_sums[edge.a] = next_vertex_power;
            } else {
                let next_left_power = vertex_power_sums[edge.a] + option.left_power;
                let next_right_power = vertex_power_sums[edge.b] + option.right_power;
                if next_left_power > vertex_power_caps[edge.a]
                    || next_right_power > vertex_power_caps[edge.b]
                {
                    continue;
                }
                vertex_power_sums[edge.a] = next_left_power;
                vertex_power_sums[edge.b] = next_right_power;
            }
            let next_coefficient = coefficient.mul(&option.coefficient);
            if next_coefficient.is_zero() {
                if edge.a == edge.b {
                    vertex_power_sums[edge.a] -= option.left_power + option.right_power;
                } else {
                    vertex_power_sums[edge.b] -= option.right_power;
                    vertex_power_sums[edge.a] -= option.left_power;
                }
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
                vertex_power_sums,
                vertex_power_caps,
                total,
                profile,
            );
            base_powers[edge.b].pop();
            base_powers[edge.a].pop();
            if edge.a == edge.b {
                vertex_power_sums[edge.a] -= option.left_power + option.right_power;
            } else {
                vertex_power_sums[edge.b] -= option.right_power;
                vertex_power_sums[edge.a] -= option.left_power;
            }
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
            profile,
        );
        vertex_product = vertex_product.mul(&vertex_sum);
        if vertex_product.is_zero() {
            return;
        }
    }
    if profile.enabled {
        profile.leaves += 1;
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
    let mut by_power = vec![QSeries::zero(q_degree); max_power + 1];
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
                by_power[power] = by_power[power].add(&coefficient);
            }
        }
    }
    by_power
        .into_iter()
        .enumerate()
        .filter_map(|(power, coefficient)| {
            (!coefficient.is_zero()).then_some(LegFactorOption { power, coefficient })
        })
        .collect()
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
    profile: &mut GraphEvalProfile,
) -> QSeries {
    let mut sorted_powers = base_powers.to_vec();
    sorted_powers.sort_unstable();
    let key = VertexContributionKey {
        genus,
        color,
        powers: sorted_powers,
    };
    if let Some(cached) = vertex_cache.get(&key) {
        if profile.enabled {
            profile.vertex_cache_hits += 1;
        }
        return cached.clone();
    }
    if profile.enabled {
        profile.vertex_cache_misses += 1;
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
            if profile.enabled {
                profile.translation_terms += 1;
            }
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ColoringOrbit {
    colors: Vec<usize>,
    multiplicity: usize,
}

fn vertex_coloring_orbits(graph: &StableGraph, colors: usize) -> Arc<Vec<ColoringOrbit>> {
    static CACHE: OnceLock<Mutex<HashMap<(StableGraph, usize), Arc<Vec<ColoringOrbit>>>>> =
        OnceLock::new();
    let key = (graph.clone(), colors);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = cache.lock().unwrap().get(&key).cloned() {
        return cached;
    }

    let automorphisms = graph.vertex_automorphism_permutations();
    let all_colorings = vertex_colorings(graph.vertices.len(), colors);
    let orbits = if automorphisms.len() <= 1 {
        all_colorings
            .into_iter()
            .map(|colors| ColoringOrbit {
                colors,
                multiplicity: 1,
            })
            .collect()
    } else {
        let mut visited = HashSet::<Vec<usize>>::new();
        let mut out = Vec::new();
        for coloring in all_colorings {
            if visited.contains(&coloring) {
                continue;
            }
            let mut orbit = HashSet::<Vec<usize>>::new();
            for permutation in &automorphisms {
                orbit.insert(permute_coloring(&coloring, permutation));
            }
            for member in &orbit {
                visited.insert(member.clone());
            }
            out.push(ColoringOrbit {
                colors: coloring,
                multiplicity: orbit.len(),
            });
        }
        out
    };

    let orbits = Arc::new(orbits);
    cache.lock().unwrap().insert(key, orbits.clone());
    orbits
}

fn permute_coloring(coloring: &[usize], permutation: &[usize]) -> Vec<usize> {
    permutation.iter().map(|&old| coloring[old]).collect()
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

    #[test]
    fn master_evaluator_matches_scalar_one_point_graph() {
        let series_req = crate::SeriesRequest {
            n: 1,
            genus: 1,
            degree_max: 1,
            max_markings: 1,
            max_descendant_power: 2,
            include_zero: true,
            equivariant: false,
            mode: ComputeMode::Givental,
            truncation: None,
        };
        let insertions = vec![tau(2, CohomologyClass::h_power(1, 1))];
        let mut evaluator = GiventalMasterEvaluator::new(&series_req);
        let master = evaluator.compute_coefficient(1, &insertions).unwrap();
        let scalar = compute_by_givental_graphs(&InvariantRequest {
            mode: ComputeMode::Givental,
            ..InvariantRequest::new(1, 1, 1, insertions)
        })
        .unwrap()
        .value;
        assert_eq!(master, scalar);
    }

    #[test]
    fn master_evaluator_matches_scalar_two_point_graph() {
        let series_req = crate::SeriesRequest {
            n: 1,
            genus: 1,
            degree_max: 1,
            max_markings: 2,
            max_descendant_power: 1,
            include_zero: true,
            equivariant: false,
            mode: ComputeMode::Givental,
            truncation: None,
        };
        let insertions = vec![
            tau(1, CohomologyClass::h_power(1, 1)),
            tau(1, CohomologyClass::h_power(1, 1)),
        ];
        let mut evaluator = GiventalMasterEvaluator::new(&series_req);
        let master = evaluator.compute_coefficient(1, &insertions).unwrap();
        let scalar = compute_by_givental_graphs(&InvariantRequest {
            mode: ComputeMode::Givental,
            ..InvariantRequest::new(1, 1, 1, insertions)
        })
        .unwrap()
        .value;
        assert_eq!(master, scalar);
    }

    #[test]
    fn coloring_orbits_reduce_vertex_automorphism_symmetry() {
        let graph = crate::graphs::StableGraph {
            vertices: vec![
                crate::graphs::StableVertex { genus: 1 },
                crate::graphs::StableVertex { genus: 1 },
            ],
            edges: vec![crate::graphs::StableEdge::new(0, 1)],
            legs: Vec::new(),
        };
        let orbits = vertex_coloring_orbits(&graph, 3);
        assert_eq!(orbits.len(), 6);
        assert_eq!(
            orbits.iter().map(|orbit| orbit.multiplicity).sum::<usize>(),
            9
        );
    }

    #[test]
    fn sparse_series_planner_skips_shared_kernel_for_few_one_point_tasks() {
        let req = crate::SeriesRequest {
            n: 2,
            genus: 3,
            degree_max: 2,
            max_markings: 1,
            max_descendant_power: 5,
            include_zero: false,
            equivariant: false,
            mode: ComputeMode::Givental,
            truncation: None,
        };
        let basis = crate::insertion_basis(req.n, req.max_descendant_power);
        let mut candidates_by_degree = vec![Vec::<Vec<Insertion>>::new(); req.degree_max + 1];
        for markings in 0..=req.max_markings {
            for insertions in crate::insertion_monomials(&basis, markings) {
                let probe_req = req.coefficient_request(0, insertions.clone());
                if let Some(expected_degree) = probe_req.expected_degree_from_dimension() {
                    if expected_degree <= req.degree_max {
                        candidates_by_degree[expected_degree].push(insertions);
                    }
                }
            }
        }

        let counts = shared_kernel_task_counts(&req, &candidates_by_degree);
        assert_eq!(counts[1], 5);
        assert!(counts[1] < MASTER_MIN_SHARED_KERNEL_TASKS);
        let mut evaluator = GiventalMasterEvaluator::new(&req);
        evaluator.set_shared_kernel_task_counts(counts);
        assert!(!evaluator.can_use_external_leg_kernel(&[tau(3, CohomologyClass::one(2))]));
        assert!(
            !evaluator.can_use_restricted_external_leg_kernel(&[tau(3, CohomologyClass::one(2))])
        );
    }

    #[test]
    fn restricted_sparse_kernel_matches_scalar_graph_evaluator() {
        let req = crate::SeriesRequest {
            n: 1,
            genus: 1,
            degree_max: 1,
            max_markings: 1,
            max_descendant_power: 2,
            include_zero: true,
            equivariant: false,
            mode: ComputeMode::Givental,
            truncation: None,
        };
        let cases = vec![
            (0, vec![tau(1, CohomologyClass::one(1))]),
            (0, vec![tau(0, CohomologyClass::h_power(1, 1))]),
            (1, vec![tau(2, CohomologyClass::h_power(1, 1))]),
        ];
        let mut evaluator = GiventalMasterEvaluator::new(&req);
        evaluator.set_shared_kernel_task_counts(vec![0, MASTER_MIN_RESTRICTED_KERNEL_TASKS]);
        let tasks = cases
            .iter()
            .enumerate()
            .map(|(ordinal, (degree, insertions))| {
                evaluator
                    .prepare_sparse_contraction_task(ordinal, *degree, insertions)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let restricted = evaluator
            .contract_restricted_tasks(1, &tasks, true)
            .unwrap();
        for result in restricted {
            let scalar = compute_by_givental_graphs(&InvariantRequest {
                mode: ComputeMode::Givental,
                ..InvariantRequest::new(
                    req.n,
                    req.genus,
                    result.coefficient.degree,
                    result.coefficient.insertions.clone(),
                )
            })
            .unwrap();
            assert_eq!(result.coefficient.value, scalar.value);
        }
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
