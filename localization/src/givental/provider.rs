//! The semisimple-CohFT calibration data, provider traits, and the
//! projective-space provider with its J-function calibration, descendant
//! S-matrix, and H-multiplication construction.

use super::*;
use crate::factored::FactoredRatFun;

/// Semisimple calibration data in a canonical idempotent frame.
///
/// The graph evaluator below only depends on this package of data, not on how
/// it was produced.  For projective space it comes from the small J-function;
/// for twisted theories, equivariant theories, r-spin, or other semisimple
/// CohFTs a provider can supply a different calibration with the same shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemisimpleCalibration<C = RatFun> {
    pub r_matrix: SeriesRMatrix<C>,
    pub metric: SeriesMatrix<C>,
    pub psi: SeriesMatrix<C>,
    pub psi_inverse: SeriesMatrix<C>,
    pub connection: SeriesMatrix<C>,
    pub delta: Vec<QSeries<C>>,
    pub inverse_delta: Vec<QSeries<C>>,
    pub relative_sqrt_delta: Vec<QSeries<C>>,
    pub relative_sqrt_delta_inverse: Vec<QSeries<C>>,
}

pub type ProjectiveSpaceJCalibration = SemisimpleCalibration;

/// Source of the semisimple data needed by the Givental-Teleman graph engine.
///
/// The current coefficient ring is `RatFun` over one Novikov variable through
/// `QSeries`.  That is enough for projective space and split-bundle twists over
/// projective space; genuinely multi-parameter theories should eventually
/// replace `QSeries` behind this boundary rather than modifying graph
/// contraction.
pub trait SemisimpleCohftProvider {
    type Insertion;

    /// Number of canonical idempotents, also the number of colors in the graph
    /// sum.
    fn colors(&self) -> usize;

    /// Descendant exponent `k` in an insertion `tau_k(gamma)`.
    fn descendant_power(&self, insertion: &Self::Insertion) -> usize;

    /// Cohomological degree of the whole insertion monomial, when it is known
    /// from the target basis.
    fn insertion_degree(&self, _insertions: &[Self::Insertion]) -> Option<usize> {
        None
    }

    /// Virtual dimension in the target theory.  The graph engine uses this only
    /// for pruning; the actual Givental sum is independent of this helper.
    fn virtual_dimension(&self, _genus: usize, _degree: usize, _markings: usize) -> Option<isize> {
        None
    }

    fn expected_degree_from_dimension(
        &self,
        _genus: usize,
        _insertions: &[Self::Insertion],
    ) -> Option<usize> {
        None
    }

    fn candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        if self.insertion_degree(insertions).is_some() {
            self.expected_degree_from_dimension(genus, insertions)
                .filter(|degree| *degree <= degree_max)
                .into_iter()
                .collect()
        } else {
            (0..=degree_max).collect()
        }
    }

    /// Descendant-to-ancestor calibration.
    ///
    /// Algebraically, this expands each descendant insertion into ancestor
    /// powers before the `R`-matrix graph action is applied.
    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError>;

    /// Complete reusable graph kernel for a fixed target and truncation.
    ///
    /// The kernel contains `R`, `R^{-1}`, the edge propagator, and translation
    /// coefficients.  It is cached aggressively because those objects dominate
    /// repeated series computations.
    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError>;

    /// Flat-basis vector for a cohomology insertion.
    ///
    /// The graph evaluator immediately applies `S` and `Psi^{-1}` after this
    /// conversion, so provider implementations should return coefficients in
    /// the same flat basis used by their `S`-matrix.
    fn insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError>;

    fn direct_value(
        &self,
        _genus: usize,
        _degree: usize,
        _insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        Ok(None)
    }

    /// Optional scalar fallback for intentionally small seed cases.
    ///
    /// This is used only after the graph path reports that an unstable range or
    /// missing truncation is outside the implemented graph evaluator.
    fn scalar_fallback_value(
        &self,
        _genus: usize,
        _degree: usize,
        _insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        Ok(None)
    }
}

/// Coefficient-generic provider boundary for semisimple graph reconstruction.
///
/// This is the extension point for alternate algebra engines such as
/// `FactoredRatFun`.  The older [`SemisimpleCohftProvider`] remains the public
/// `RatFun` API; this trait is deliberately parallel rather than replacing it,
/// so existing projective and twisted providers do not see method-resolution
/// churn.
pub trait CoefficientSemisimpleCohftProvider<C: Coeff> {
    type Insertion;

    fn coeff_colors(&self) -> usize;

    fn coeff_descendant_power(&self, insertion: &Self::Insertion) -> usize;

    fn coeff_insertion_degree(&self, _insertions: &[Self::Insertion]) -> Option<usize> {
        None
    }

    fn coeff_virtual_dimension(
        &self,
        _genus: usize,
        _degree: usize,
        _markings: usize,
    ) -> Option<isize> {
        None
    }

    fn coeff_expected_degree_from_dimension(
        &self,
        _genus: usize,
        _insertions: &[Self::Insertion],
    ) -> Option<usize> {
        None
    }

    fn coeff_candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        if self.coeff_insertion_degree(insertions).is_some() {
            self.coeff_expected_degree_from_dimension(genus, insertions)
                .filter(|degree| *degree <= degree_max)
                .into_iter()
                .collect()
        } else {
            (0..=degree_max).collect()
        }
    }

    fn coeff_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<C>, GwError>;

    fn coeff_graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel<C>>, GwError>;

    fn coeff_insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries<C>>, GwError>;

    fn coeff_direct_value(
        &self,
        _genus: usize,
        _degree: usize,
        _insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<C>, GwError> {
        Ok(None)
    }

    fn coeff_scalar_fallback_value(
        &self,
        _genus: usize,
        _degree: usize,
        _insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<C>, GwError> {
        Ok(None)
    }
}

impl<P> CoefficientSemisimpleCohftProvider<RatFun> for P
where
    P: SemisimpleCohftProvider,
{
    type Insertion = P::Insertion;

    fn coeff_colors(&self) -> usize {
        SemisimpleCohftProvider::colors(self)
    }

    fn coeff_descendant_power(&self, insertion: &Self::Insertion) -> usize {
        SemisimpleCohftProvider::descendant_power(self, insertion)
    }

    fn coeff_insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        SemisimpleCohftProvider::insertion_degree(self, insertions)
    }

    fn coeff_virtual_dimension(
        &self,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Option<isize> {
        SemisimpleCohftProvider::virtual_dimension(self, genus, degree, markings)
    }

    fn coeff_expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        SemisimpleCohftProvider::expected_degree_from_dimension(self, genus, insertions)
    }

    fn coeff_candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        SemisimpleCohftProvider::candidate_degrees_from_dimension(
            self, genus, degree_max, insertions,
        )
    }

    fn coeff_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        SemisimpleCohftProvider::descendant_s_matrix(self, q_degree, z_order)
    }

    fn coeff_graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        SemisimpleCohftProvider::graph_kernel(self, q_degree, r_order, graph_dimension)
    }

    fn coeff_insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError> {
        SemisimpleCohftProvider::insertion_vector(self, insertion, q_degree)
    }

    fn coeff_direct_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        SemisimpleCohftProvider::direct_value(self, genus, degree, insertions, truncation)
    }

    fn coeff_scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        SemisimpleCohftProvider::scalar_fallback_value(self, genus, degree, insertions, truncation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectiveSpaceProvider {
    pub n: usize,
    /// `true` keeps the symbolic equivariant lambda parameters.  `false` uses
    /// the current fast non-equivariant path, namely early specialization to a
    /// generic lambda line.
    pub equivariant: bool,
    pub weights: Vec<Rational>,
}

impl ProjectiveSpaceProvider {
    pub fn new(n: usize, equivariant: bool) -> Self {
        Self {
            n,
            equivariant,
            weights: (1..=n + 1).map(Rational::from).collect(),
        }
    }

    pub fn symbolic_equivariant(n: usize) -> Self {
        Self::new(n, true)
    }

    pub fn lambda_line_nonequivariant(n: usize) -> Self {
        Self::new(n, false)
    }

    pub(crate) fn specialized_nonequivariant(&self) -> bool {
        !self.equivariant
    }
}

impl SemisimpleCohftProvider for ProjectiveSpaceProvider {
    type Insertion = Insertion;

    fn colors(&self) -> usize {
        self.n + 1
    }

    fn descendant_power(&self, insertion: &Self::Insertion) -> usize {
        insertion.descendant_power
    }

    fn insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        let mut total = 0usize;
        for insertion in insertions {
            total = total.checked_add(insertion.descendant_power)?;
            total = total.checked_add(insertion.class.pure_power()?)?;
        }
        Some(total)
    }

    fn virtual_dimension(&self, genus: usize, degree: usize, markings: usize) -> Option<isize> {
        Some(
            (1 - genus as isize) * (self.n as isize - 3)
                + (self.n + 1) as isize * degree as isize
                + markings as isize,
        )
    }

    fn expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        let insertion_degree = self.insertion_degree(insertions)? as isize;
        let dimension_without_degree =
            (1 - genus as isize) * (self.n as isize - 3) + insertions.len() as isize;
        let numerator = insertion_degree - dimension_without_degree;
        let denominator = (self.n + 1) as isize;
        if numerator < 0 || numerator % denominator != 0 {
            return None;
        }
        Some((numerator / denominator) as usize)
    }

    fn candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        self.expected_degree_from_dimension(genus, insertions)
            .filter(|degree| *degree <= degree_max)
            .into_iter()
            .collect()
    }

    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        if self.equivariant {
            projective_space_descendant_s_matrix(self.n, q_degree, z_order)
        } else {
            projective_space_descendant_s_matrix_at_lambda_weights(
                self.n,
                q_degree,
                z_order,
                &self.weights,
            )
        }
    }

    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        projective_space_graph_kernel(
            self.n,
            q_degree,
            r_order,
            graph_dimension,
            self.equivariant,
            &self.weights,
        )
    }

    fn insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError> {
        let coeffs = insertion.class.coeffs();
        if coeffs.len() != self.colors() {
            return Err(GwError::ConventionMismatch(format!(
                "P^{} insertion has {} coefficients, expected {}",
                self.n,
                coeffs.len(),
                self.colors()
            )));
        }
        Ok(coeffs
            .iter()
            .map(|coeff| QSeries::constant(coeff.clone(), q_degree))
            .collect())
    }

    fn scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        let req = InvariantRequest {
            n: self.n,
            genus,
            degree,
            insertions: insertions.to_vec(),
            equivariant: self.equivariant,
            mode: ComputeMode::Givental,
            truncation: truncation.cloned(),
        };
        match validation::seed_compute(&req, "givental-seed") {
            Ok(result) => Ok(Some(result.value)),
            Err(GwError::UnsupportedInvariant(_)) => Ok(None),
            Err(err) => Err(err),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CalibrationCacheKey {
    n: usize,
    q_degree: usize,
    z_order: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct LambdaCalibrationCacheKey {
    n: usize,
    q_degree: usize,
    z_order: usize,
    weights: Vec<Rational>,
}

/// Builds the projective-space calibration from small quantum cohomology.
///
/// This is the `P^n` specialization of the general reconstruction input:
///
/// 1. solve the quantum relation `prod(H-lambda_i)=q` for canonical roots;
/// 2. form unnormalized idempotents and the flat-to-canonical matrix `Psi`;
/// 3. compute the Dubrovin connection `Psi^{-1} q d(Psi)/dq`;
/// 4. solve the `R`-matrix flatness recursion with the Bernoulli classical
///    asymptotic as the integration constant.
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
    let frame = CanonicalFrame {
        flat_to_canonical: canonical_evaluation_matrix(&canonical.roots),
        transition_to_flat: canonical.transition_matrix(),
        roots: canonical.roots,
        metric_norms: canonical.metric_norms,
        inverse_metric_norms: canonical.inverse_metric_norms,
    };
    let classical_diagonal = classical_limit_diagonal_coefficients(n, z_order);
    let calibration = calibration_from_canonical_frame(
        &frame,
        &classical_diagonal,
        q_degree,
        z_order,
        CalibrationId("projective-space-j".to_string()),
    )?;
    cache.lock().unwrap().insert(key, calibration.clone());
    Ok(calibration)
}

pub(crate) fn projective_space_j_calibration_at_lambda_weights(
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
    let frame = CanonicalFrame {
        flat_to_canonical: canonical_evaluation_matrix(&canonical.roots),
        transition_to_flat: SeriesMatrix::from_entries(canonical.transition_to_flat),
        roots: canonical.roots,
        metric_norms: canonical.metric_norms,
        inverse_metric_norms: canonical.inverse_metric_norms,
    };
    let classical_diagonal =
        classical_limit_diagonal_coefficients_at_lambda_weights(n, z_order, weights);
    let calibration = calibration_from_canonical_frame(
        &frame,
        &classical_diagonal,
        q_degree,
        z_order,
        CalibrationId("projective-space-j-lambda-line".to_string()),
    )?;
    cache.lock().unwrap().insert(key, calibration.clone());
    Ok(calibration)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpecializedQuantumCanonicalData {
    roots: Vec<QSeries>,
    metric_norms: Vec<QSeries>,
    inverse_metric_norms: Vec<QSeries>,
    transition_to_flat: Vec<Vec<QSeries>>,
}

pub(crate) fn specialized_quantum_canonical_data(
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

pub(crate) fn canonical_root_series_at_lambda_weights(
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

pub(crate) fn characteristic_series_at_lambda_weights(
    n: usize,
    x: &QSeries,
    weights: &[Rational],
) -> QSeries {
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

pub(crate) fn characteristic_derivative_series_at_lambda_weights(
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

pub(crate) fn multiply_qseries_polynomial_by_linear(
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

    let quantum_h = series_h_multiplication_matrix(n, q_degree, true);
    let classical_h = series_h_multiplication_matrix(n, q_degree, false);
    let descendant_s = descendant_s_from_divisor_qde(
        &quantum_h,
        &classical_h,
        z_order,
        CalibrationId("projective-space-small-j".to_string()),
    )?;
    cache.lock().unwrap().insert(key, descendant_s.clone());
    Ok(descendant_s)
}

pub(crate) fn projective_space_descendant_s_matrix_at_lambda_weights(
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

    let quantum_h = series_h_multiplication_matrix_at_lambda_weights(n, q_degree, true, weights)?;
    let classical_h =
        series_h_multiplication_matrix_at_lambda_weights(n, q_degree, false, weights)?;
    let descendant_s = descendant_s_from_divisor_qde(
        &quantum_h,
        &classical_h,
        z_order,
        CalibrationId("projective-space-small-j-lambda-line".to_string()),
    )?;
    cache.lock().unwrap().insert(key, descendant_s.clone());
    Ok(descendant_s)
}

pub(crate) fn series_h_multiplication_matrix_at_lambda_weights(
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

pub(crate) fn series_h_multiplication_matrix(
    n: usize,
    q_degree: usize,
    quantum: bool,
) -> SeriesMatrix {
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

pub(crate) fn h_power_relation_series(n: usize, q_degree: usize, quantum: bool) -> Vec<QSeries> {
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

pub(crate) fn h_power_relation_series_at_lambda_weights(
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

/// Equivariant projective-space provider over factored coefficients.
///
/// The J-calibration itself is cheap to build in expanded `RatFun` (small,
/// low-degree entries), but constructing `R^{-1}` and the edge propagators
/// from it — and then contracting graphs — multiplies those entries many
/// times, which is exactly where expanded denominators blow up.  This wrapper
/// converts the calibration once and lets everything downstream run over
/// [`FactoredRatFun`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoredProjectiveSpaceProvider(pub ProjectiveSpaceProvider);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FactoredKernelCacheKey {
    n: usize,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
}

pub(crate) fn projective_space_factored_graph_kernel(
    n: usize,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
) -> Result<Arc<GiventalGraphKernel<FactoredRatFun>>, GwError> {
    static CACHE: OnceLock<
        Mutex<HashMap<FactoredKernelCacheKey, Arc<GiventalGraphKernel<FactoredRatFun>>>>,
    > = OnceLock::new();
    let key = FactoredKernelCacheKey {
        n,
        q_degree,
        r_order,
        graph_dimension,
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(kernel) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(kernel);
    }

    let calibration = projective_space_j_calibration(n, q_degree, r_order)?;
    let kernel = Arc::new(GiventalGraphKernel::from_calibration(
        calibration_to_factored(&calibration),
        graph_dimension,
    )?);
    cache.lock().unwrap().insert(key, kernel.clone());
    Ok(kernel)
}

impl CoefficientSemisimpleCohftProvider<FactoredRatFun> for FactoredProjectiveSpaceProvider {
    type Insertion = Insertion;

    fn coeff_colors(&self) -> usize {
        self.0.colors()
    }

    fn coeff_descendant_power(&self, insertion: &Self::Insertion) -> usize {
        insertion.descendant_power
    }

    fn coeff_insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        self.0.insertion_degree(insertions)
    }

    fn coeff_virtual_dimension(
        &self,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Option<isize> {
        self.0.virtual_dimension(genus, degree, markings)
    }

    fn coeff_expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        self.0.expected_degree_from_dimension(genus, insertions)
    }

    fn coeff_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
        series_s_matrix_to_factored(&self.0.descendant_s_matrix(q_degree, z_order)?)
    }

    fn coeff_graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel<FactoredRatFun>>, GwError> {
        projective_space_factored_graph_kernel(self.0.n, q_degree, r_order, graph_dimension)
    }

    fn coeff_insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries<FactoredRatFun>>, GwError> {
        Ok(self
            .0
            .insertion_vector(insertion, q_degree)?
            .iter()
            .map(qseries_to_factored)
            .collect())
    }

    fn coeff_scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<Option<FactoredRatFun>, GwError> {
        Ok(self
            .0
            .scalar_fallback_value(genus, degree, insertions, truncation)?
            .map(FactoredRatFun::from_ratfun))
    }
}
