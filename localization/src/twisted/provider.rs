//! The twisted semisimple CohFT providers (rational and factored), the
//! TwistedInvariantRequest, the public compute entry points, and the graph
//! evaluator wiring.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwistedProjectiveSpaceProvider {
    base: ProjectiveSpaceProvider,
    twist: NegativeSplitBundleTwist,
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
    fiber_weight_scale: Rational,
    custom_fiber_weights: Option<Vec<Rational>>,
    fiber_parameter_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum TwistedLineMode {
    EarlyRational,
    SymbolicLimit,
    FiberEquivariant,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TwistedCalibrationMode {
    InverseEuler,
    InverseEulerFiberPlus,
    Euler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TwistedCalibrationValidation {
    /// Skip diagnostic identities that are expensive and do not change the
    /// produced calibration.  This is the default in the graph-evaluation path.
    Fast,
    /// Run full self-adjointness, diagonalization, and R-unitarity checks.
    Full,
}

impl TwistedCalibrationValidation {
    pub(crate) fn runs_expensive_checks(self) -> bool {
        matches!(self, Self::Full)
    }
}

pub(crate) fn twisted_calibration_validation_from_env() -> TwistedCalibrationValidation {
    // Expensive identities are off in the hot graph-evaluation path by default.
    // Set either variable to 1/true/yes/on/full when debugging a calibration
    // change and wanting self-adjointness, diagonalization, and unitarity checks
    // to run before caching the graph kernel.
    if env_flag_enabled("GWAI_VALIDATE_TWISTED_CALIBRATION")
        || env_flag_enabled("GW_VALIDATE_CALIBRATION")
    {
        TwistedCalibrationValidation::Full
    } else {
        TwistedCalibrationValidation::Fast
    }
}

pub(crate) fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "full"
            )
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TwistedGraphKernelCacheKey {
    n: usize,
    twist_degrees: Vec<usize>,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
    base_weights: Vec<Rational>,
    fiber_weights: Vec<Rational>,
    fiber_parameter_names: Vec<String>,
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
    validation: TwistedCalibrationValidation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwistedInvariantRequest {
    pub n: usize,
    pub twist: NegativeSplitBundleTwist,
    pub genus: usize,
    pub degree: usize,
    pub insertions: Vec<Insertion>,
    pub equivariant: bool,
    pub truncation: Option<Truncation>,
}

impl TwistedInvariantRequest {
    pub fn new(
        n: usize,
        degrees: Vec<usize>,
        genus: usize,
        degree: usize,
        insertions: Vec<Insertion>,
    ) -> Result<Self, GwError> {
        Ok(Self {
            n,
            twist: NegativeSplitBundleTwist::new(degrees)?,
            genus,
            degree,
            insertions,
            equivariant: false,
            truncation: None,
        })
    }
}

pub fn compute_negative_split_twisted(
    req: &TwistedInvariantRequest,
) -> Result<InvariantResult, GwError> {
    // Public local/twisted path: construct a semisimple provider from the
    // hypergeometric/Birkhoff/QRR calibration, run the generic Givental graph
    // evaluator, and either take the one-parameter lambda-line limit or keep the
    // fiber equivariant parameters as symbolic rational variables.
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = if req.equivariant {
        TwistedProjectiveSpaceProvider::fiber_equivariant(req.n, req.twist.degrees().to_vec())?
    } else {
        TwistedProjectiveSpaceProvider::new(req.n, req.twist.degrees().to_vec(), false)?
    };
    if !req.equivariant {
        if let Some((virtual_dimension, total_degree)) =
            twisted_dimension_mismatch(&provider, req.genus, req.degree, &req.insertions)
        {
            return Ok(InvariantResult {
                value: RatFun::zero(),
                engine: "twisted-negative-split-dimension",
                notes: vec![format!(
                    "dimension mismatch gives zero: virtual dimension {virtual_dimension}, insertion degree {total_degree}"
                )],
            });
        }
    }

    let unstable_two_point = req.genus == 0 && req.insertions.len() == 2;
    let primary_three_point = req.genus == 0 && genus_zero_three_primary_layout(&req.insertions);
    let raw = crate::givental::compute_semisimple_graph_value(
        &provider,
        req.genus,
        req.degree,
        &req.insertions,
        req.truncation.as_ref(),
    )?;
    let value = if req.equivariant {
        raw.lambda_line_limit_preserving_variables(req.n, provider.base_weights())?
    } else {
        match raw.as_rational() {
            Some(value) => RatFun::from_rational(value),
            None => RatFun::from_rational(raw.nonequivariant_limit_line(0, &[Rational::one()])?),
        }
    };
    Ok(InvariantResult {
        value,
        engine: if req.equivariant {
            "twisted-negative-split-fiber-equivariant-givental-birkhoff"
        } else {
            "twisted-negative-split-givental-birkhoff-early-line"
        },
        notes: vec![if unstable_two_point {
            if req.equivariant {
                "computed by the genus-zero two-point unstable S-matrix convention from the fiber-equivariant hypergeometric/Birkhoff calibration; base weights are early-specialized and fiber weights are symbolic mu_i"
                    .to_string()
            } else {
                "computed by the genus-zero two-point unstable S-matrix convention from the same early rational one-parameter lambda-line hypergeometric/Birkhoff calibration; no local oracle shortcut is used"
                    .to_string()
            }
        } else if primary_three_point {
            if req.equivariant {
                "computed by the fiber-equivariant twisted genus-zero Frobenius quantum product from the same hypergeometric/Birkhoff calibration; base weights are early-specialized and fiber weights are symbolic mu_i"
                    .to_string()
            } else {
                "computed by the twisted genus-zero Frobenius quantum product from the same early rational hypergeometric/Birkhoff calibration"
                    .to_string()
            }
        } else if req.equivariant {
            "computed by fiber-equivariant hypergeometric/Birkhoff S and QRR R stable-graph expansion; base weights are early-specialized and fiber weights are symbolic mu_i"
                .to_string()
        } else {
            "computed by early rational one-parameter lambda-line hypergeometric/Birkhoff S and QRR R stable-graph expansion; no local oracle shortcut is used; fast validation currently covers resolved conifold genus 2 degree 1"
                .to_string()
        }],
    })
}

pub fn compute_negative_split_twisted_factored(
    req: &TwistedInvariantRequest,
) -> Result<FactoredRatFun, GwError> {
    if !req.equivariant {
        return Err(GwError::UnsupportedInvariant(
            "factored twisted mode is currently for fiber-equivariant computations; pass --equivariant"
                .to_string(),
        ));
    }
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(
        req.n,
        req.twist.degrees().to_vec(),
    )?;
    if twisted_dimension_mismatch(provider.inner(), req.genus, req.degree, &req.insertions)
        .is_some()
    {
        return compute_negative_split_twisted(req)
            .map(|result| FactoredRatFun::from_ratfun(result.value));
    }
    compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
        &provider,
        req.genus,
        req.degree,
        &req.insertions,
        req.truncation.as_ref(),
    )
}

pub fn compute_negative_split_twisted_resolvent_packed(
    target_n: usize,
    degrees: Vec<usize>,
    req: &ResolventRequest,
    equivariant: bool,
) -> Result<ResolventResult, GwError> {
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = if equivariant {
        TwistedProjectiveSpaceProvider::fiber_equivariant(target_n, degrees)?
    } else {
        TwistedProjectiveSpaceProvider::new(target_n, degrees, false)?
    };
    crate::givental::compute_packed_resolvent_with_provider(
        req,
        provider,
        if equivariant {
            "twisted-negative-split-fiber-equivariant-packed-resolvent"
        } else {
            "twisted-negative-split-packed-resolvent"
        },
        if equivariant {
            "computed by packed fiber-equivariant twisted S/R external-leg graph kernel; base weights are early-specialized and fiber weights are symbolic mu_i"
        } else {
            "computed by packed twisted S/R external-leg graph kernel; all resolvent coefficients share one stable-graph contraction"
        },
        move |raw| {
            if equivariant {
                return Ok(raw);
            }
            match raw.as_rational() {
                Some(value) => Ok(RatFun::from_rational(value)),
                None => Ok(RatFun::from_rational(
                    raw.nonequivariant_limit_line(0, &[Rational::one()])?,
                )),
            }
        },
    )
}

pub fn compute_negative_split_twisted_resolvent_packed_factored(
    target_n: usize,
    degrees: Vec<usize>,
    req: &ResolventRequest,
) -> Result<ResolventResult<FactoredRatFun>, GwError> {
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(target_n, degrees)?;
    crate::givental::compute_packed_resolvent_with_coeff_provider(
        req,
        provider,
        "twisted-negative-split-fiber-equivariant-factored-packed-resolvent",
        "computed by packed fiber-equivariant twisted S/R external-leg graph kernel with factored symbolic coefficients; base weights are early-specialized and fiber weights are symbolic mu_i",
        Ok::<FactoredRatFun, GwError>,
    )
}

pub(crate) fn twisted_dimension_mismatch(
    provider: &TwistedProjectiveSpaceProvider,
    genus: usize,
    degree: usize,
    insertions: &[Insertion],
) -> Option<(isize, usize)> {
    let total_degree = provider.insertion_degree(insertions)?;
    let virtual_dimension = provider.virtual_dimension(genus, degree, insertions.len())?;
    (virtual_dimension >= 0 && total_degree as isize != virtual_dimension)
        .then_some((virtual_dimension, total_degree))
}

impl TwistedProjectiveSpaceProvider {
    pub fn new(n: usize, degrees: Vec<usize>, equivariant: bool) -> Result<Self, GwError> {
        let mut base = ProjectiveSpaceProvider::new(n, equivariant);
        base.weights = twisted_default_base_weights(n);
        Ok(Self {
            base,
            twist: NegativeSplitBundleTwist::new(degrees)?,
            line_mode: TwistedLineMode::EarlyRational,
            calibration_mode: TwistedCalibrationMode::InverseEuler,
            fiber_weight_scale: Rational::one(),
            custom_fiber_weights: None,
            fiber_parameter_names: Vec::new(),
        })
    }

    pub fn fiber_equivariant(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.line_mode = TwistedLineMode::FiberEquivariant;
        out.fiber_parameter_names = default_fiber_parameter_names(out.twist.rank());
        Ok(out)
    }

    pub fn symbolic_lambda_line(
        n: usize,
        degrees: Vec<usize>,
        equivariant: bool,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, equivariant)?;
        out.line_mode = TwistedLineMode::SymbolicLimit;
        Ok(out)
    }

    pub fn euler_twist(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.calibration_mode = TwistedCalibrationMode::Euler;
        Ok(out)
    }

    pub fn inverse_euler_with_positive_fiber_qrr(
        n: usize,
        degrees: Vec<usize>,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.calibration_mode = TwistedCalibrationMode::InverseEulerFiberPlus;
        Ok(out)
    }

    fn flat_metric_matrix(&self, q_degree: usize) -> Result<SeriesMatrix, GwError> {
        let fiber_weights = self.rational_fiber_weights();
        match self.line_mode {
            TwistedLineMode::EarlyRational => twisted_inverse_euler_flat_metric_matrix(
                self.base.n,
                q_degree,
                &self.twist,
                self.base_weights(),
                &fiber_weights,
            ),
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                twisted_inverse_euler_flat_metric_matrix_ratfun(
                    self.base.n,
                    q_degree,
                    &self.twist,
                    &base_weights,
                    &fiber_weights,
                )
            }
        }
    }

    fn genus_zero_two_point_fallback(
        &self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<Option<RatFun>, GwError> {
        let Some((descendant_idx, primary_idx, s_order)) =
            genus_zero_two_point_descendant_layout(insertions)
        else {
            return Ok(None);
        };
        let s_matrix = self.descendant_s_matrix(degree, s_order)?;
        let metric = self.flat_metric_matrix(degree)?;
        let descendant = self.insertion_vector(&insertions[descendant_idx], degree)?;
        let primary = self.insertion_vector(&insertions[primary_idx], degree)?;
        genus_zero_two_point_s_matrix_pairing_coeff(
            self.colors(),
            degree,
            s_order,
            &s_matrix,
            &metric,
            &descendant,
            &primary,
        )
        .map(Some)
    }

    pub fn rational_lambda_line_with_scale(
        n: usize,
        degrees: Vec<usize>,
        scale: Rational,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        for weight in &mut out.base.weights {
            *weight = weight.clone() * scale.clone();
        }
        out.fiber_weight_scale = scale;
        Ok(out)
    }

    pub fn rational_lambda_line_with_weights(
        n: usize,
        degrees: Vec<usize>,
        base_weights: Vec<Rational>,
        fiber_weights: Vec<Rational>,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        validate_twisted_weights(n, &out.twist, &base_weights, &fiber_weights)?;
        out.base.weights = base_weights;
        out.fiber_weight_scale = Rational::one();
        out.custom_fiber_weights = Some(fiber_weights);
        Ok(out)
    }

    pub fn n(&self) -> usize {
        self.base.n
    }

    pub fn twist(&self) -> &NegativeSplitBundleTwist {
        &self.twist
    }

    fn base_weights(&self) -> &[Rational] {
        &self.base.weights
    }

    fn rational_fiber_weights(&self) -> Vec<Rational> {
        if let Some(weights) = &self.custom_fiber_weights {
            return weights.clone();
        }
        let start = (1usize << (self.base.n + 1)).saturating_sub(1);
        (0..self.twist.rank())
            .map(|idx| Rational::from(start + 2 * idx) * self.fiber_weight_scale.clone())
            .collect()
    }

    fn ratfun_base_weights(&self) -> Vec<RatFun> {
        match self.line_mode {
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let lambda = crate::algebra::lambda(0);
                self.base_weights()
                    .iter()
                    .map(|weight| lambda.clone() * RatFun::from_rational(weight.clone()))
                    .collect()
            }
            TwistedLineMode::EarlyRational => self
                .base_weights()
                .iter()
                .cloned()
                .map(RatFun::from_rational)
                .collect(),
        }
    }

    fn ratfun_fiber_weights(&self) -> Vec<RatFun> {
        match self.line_mode {
            TwistedLineMode::SymbolicLimit => {
                let lambda = crate::algebra::lambda(0);
                self.rational_fiber_weights()
                    .into_iter()
                    .map(|weight| lambda.clone() * RatFun::from_rational(weight))
                    .collect()
            }
            TwistedLineMode::FiberEquivariant => self
                .fiber_parameter_names
                .iter()
                .cloned()
                .map(RatFun::variable)
                .collect(),
            TwistedLineMode::EarlyRational => self
                .rational_fiber_weights()
                .into_iter()
                .map(RatFun::from_rational)
                .collect(),
        }
    }
}

pub(crate) fn genus_zero_two_point_descendant_layout(
    insertions: &[Insertion],
) -> Option<(usize, usize, usize)> {
    if insertions.len() != 2 {
        return None;
    }
    let descendant_positions = insertions
        .iter()
        .enumerate()
        .filter_map(|(idx, insertion)| (insertion.descendant_power > 0).then_some(idx))
        .collect::<Vec<_>>();
    if descendant_positions.len() != 1 {
        return None;
    }

    let descendant_idx = descendant_positions[0];
    let primary_idx = 1 - descendant_idx;
    let s_order = insertions[descendant_idx].descendant_power + 1;
    Some((descendant_idx, primary_idx, s_order))
}

pub(crate) fn genus_zero_two_point_s_matrix_pairing_coeff<C: Coeff>(
    colors: usize,
    degree: usize,
    s_order: usize,
    s_matrix: &SeriesSMatrix<C>,
    metric: &SeriesMatrix<C>,
    descendant: &[QSeries<C>],
    primary: &[QSeries<C>],
) -> Result<C, GwError> {
    let s_coeff = s_matrix
        .coefficient(s_order)
        .ok_or(GwError::TruncationTooLow)?;

    let mut transformed = vec![QSeries::<C>::zero(degree); colors];
    for (row, target) in transformed.iter_mut().enumerate() {
        let mut total = QSeries::<C>::zero(degree);
        for (col, class_coeff) in descendant.iter().enumerate() {
            total = total.add(&s_coeff.entry(row, col).mul(class_coeff));
        }
        *target = total;
    }

    let mut paired = QSeries::<C>::zero(degree);
    for (left, transformed_coeff) in transformed.iter().enumerate() {
        for (right, primary_coeff) in primary.iter().enumerate() {
            let term = transformed_coeff
                .mul(metric.entry(left, right))
                .mul(primary_coeff);
            paired = paired.add(&term);
        }
    }
    Ok(paired.coeff(degree).cloned().unwrap_or_else(C::zero))
}

pub(crate) fn genus_zero_three_primary_layout(insertions: &[Insertion]) -> bool {
    insertions.len() == 3
        && insertions
            .iter()
            .all(|insertion| insertion.descendant_power == 0)
}

pub(crate) fn twisted_genus_zero_three_primary_value_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    calibration_mode: &TwistedCalibrationMode,
    base_weights: &[C],
    fiber_weights: &[C],
    insertions: &[Vec<QSeries<C>>],
) -> Result<C, GwError> {
    debug_assert_eq!(insertions.len(), 3);
    let model = NegativeSplitLineHypergeometricModel::<C>::from_coeff_weights(
        n,
        twist.clone(),
        degree,
        1,
        base_weights.to_vec(),
        fiber_weights,
    )?;
    let descendant_s = model.birkhoff_descendant_s_matrix(1)?;
    let classical_h = twisted_classical_h_multiplication_matrix_coeff(n, degree, base_weights)?;
    let quantum_h =
        twisted_quantum_multiplication_from_s_coeff(&descendant_s, &classical_h, calibration_mode)?;
    let metric = twisted_inverse_euler_flat_metric_matrix_coeff(
        n,
        degree,
        twist,
        base_weights,
        fiber_weights,
    )?;
    let product = quantum_product_vectors_coeff(&quantum_h, &insertions[0], &insertions[1], degree);
    pair_vectors_coeff(&metric, &product, &insertions[2], degree)
}

pub(crate) fn quantum_product_vectors_coeff<C: Coeff>(
    quantum_h: &SeriesMatrix<C>,
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let colors = quantum_h.rows();
    debug_assert_eq!(left.len(), colors);
    debug_assert_eq!(right.len(), colors);
    let mut result = vec![QSeries::<C>::zero(q_degree); colors];
    let mut h_power_right = right.to_vec();
    for (power, left_coeff) in left.iter().enumerate() {
        if !left_coeff.is_structurally_zero() {
            for row in 0..colors {
                result[row] = result[row].add(&h_power_right[row].mul(left_coeff));
            }
        }
        if power + 1 < left.len() {
            h_power_right =
                apply_series_matrix_to_vector_coeff(quantum_h, &h_power_right, q_degree);
        }
    }
    result
}

pub(crate) fn apply_series_matrix_to_vector_coeff<C: Coeff>(
    matrix: &SeriesMatrix<C>,
    vector: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    debug_assert_eq!(matrix.cols(), vector.len());
    (0..matrix.rows())
        .map(|row| {
            let mut total = QSeries::<C>::zero(q_degree);
            for (col, vector_coeff) in vector.iter().enumerate() {
                if vector_coeff.is_structurally_zero()
                    || matrix.entry(row, col).is_structurally_zero()
                {
                    continue;
                }
                total = total.add(&matrix.entry(row, col).mul(vector_coeff));
            }
            total
        })
        .collect()
}

pub(crate) fn pair_vectors_coeff<C: Coeff>(
    metric: &SeriesMatrix<C>,
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    degree: usize,
) -> Result<C, GwError> {
    let colors = metric.rows();
    debug_assert_eq!(metric.cols(), colors);
    debug_assert_eq!(left.len(), colors);
    debug_assert_eq!(right.len(), colors);
    let mut paired = QSeries::<C>::zero(degree);
    for (left_idx, left_coeff) in left.iter().enumerate() {
        if left_coeff.is_structurally_zero() {
            continue;
        }
        for (right_idx, right_coeff) in right.iter().enumerate() {
            if right_coeff.is_structurally_zero()
                || metric.entry(left_idx, right_idx).is_structurally_zero()
            {
                continue;
            }
            paired = paired.add(
                &left_coeff
                    .mul(metric.entry(left_idx, right_idx))
                    .mul(right_coeff),
            );
        }
    }
    Ok(paired.coeff(degree).cloned().unwrap_or_else(C::zero))
}

pub(crate) fn twisted_default_base_weights(n: usize) -> Vec<Rational> {
    (0..=n).map(|idx| Rational::from(1usize << idx)).collect()
}

pub(crate) fn default_fiber_parameter_names(rank: usize) -> Vec<String> {
    (0..rank).map(|idx| format!("mu_{idx}")).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoredTwistedProjectiveSpaceProvider {
    inner: TwistedProjectiveSpaceProvider,
}

impl FactoredTwistedProjectiveSpaceProvider {
    pub fn fiber_equivariant(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        Ok(Self {
            inner: TwistedProjectiveSpaceProvider::fiber_equivariant(n, degrees)?,
        })
    }

    pub fn inner(&self) -> &TwistedProjectiveSpaceProvider {
        &self.inner
    }

    fn factored_base_weights(&self) -> Vec<FactoredRatFun> {
        self.inner
            .base_weights()
            .iter()
            .cloned()
            .map(FactoredRatFun::from_rational)
            .collect()
    }

    fn factored_fiber_weights(&self) -> Vec<FactoredRatFun> {
        self.inner
            .fiber_parameter_names
            .iter()
            .cloned()
            .map(FactoredRatFun::variable)
            .collect()
    }

    fn factored_flat_metric_matrix(
        &self,
        q_degree: usize,
    ) -> Result<SeriesMatrix<FactoredRatFun>, GwError> {
        let (metric, _) = twisted_inverse_euler_flat_metric_pair_from_rational_base(
            self.inner.base.n,
            q_degree,
            &self.inner.twist,
            self.inner.base_weights(),
            &self.factored_fiber_weights(),
        )?;
        Ok(metric)
    }

    fn factored_genus_zero_two_point_fallback(
        &self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<Option<FactoredRatFun>, GwError> {
        let Some((descendant_idx, primary_idx, s_order)) =
            genus_zero_two_point_descendant_layout(insertions)
        else {
            return Ok(None);
        };
        let s_matrix = self.coeff_descendant_s_matrix(degree, s_order)?;
        let metric = self.factored_flat_metric_matrix(degree)?;
        let descendant = self.coeff_insertion_vector(&insertions[descendant_idx], degree)?;
        let primary = self.coeff_insertion_vector(&insertions[primary_idx], degree)?;
        genus_zero_two_point_s_matrix_pairing_coeff(
            self.coeff_colors(),
            degree,
            s_order,
            &s_matrix,
            &metric,
            &descendant,
            &primary,
        )
        .map(Some)
    }
}

impl CoefficientSemisimpleCohftProvider<FactoredRatFun> for FactoredTwistedProjectiveSpaceProvider {
    type Insertion = Insertion;

    fn coeff_colors(&self) -> usize {
        self.inner.colors()
    }

    fn coeff_descendant_power(&self, insertion: &Self::Insertion) -> usize {
        self.inner.descendant_power(insertion)
    }

    fn coeff_insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        self.inner.insertion_degree(insertions)
    }

    fn coeff_virtual_dimension(
        &self,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Option<isize> {
        self.inner.virtual_dimension(genus, degree, markings)
    }

    fn coeff_expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        self.inner.expected_degree_from_dimension(genus, insertions)
    }

    fn coeff_candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        self.inner
            .candidate_degrees_from_dimension(genus, degree_max, insertions)
    }

    fn coeff_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
        let base_weights = self.factored_base_weights();
        let fiber_weights = self.factored_fiber_weights();
        let s_matrix = NegativeSplitLineHypergeometricModel::<FactoredRatFun>::from_coeff_weights(
            self.inner.base.n,
            self.inner.twist.clone(),
            q_degree,
            z_order,
            base_weights.clone(),
            &fiber_weights,
        )?
        .birkhoff_descendant_s_matrix(z_order)?;
        let (flat_metric, flat_metric_inverse) =
            twisted_inverse_euler_flat_metric_pair_from_rational_base(
                self.inner.base.n,
                q_degree,
                &self.inner.twist,
                self.inner.base_weights(),
                &fiber_weights,
            )?;
        metric_adjoint_descendant_s_matrix_with_inverse_coeff(
            s_matrix,
            &flat_metric,
            &flat_metric_inverse,
        )
    }

    fn coeff_graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel<FactoredRatFun>>, GwError> {
        let profile_enabled = std::env::var_os("GW_PROFILE").is_some();
        let started = std::time::Instant::now();
        let base_weights = self.factored_base_weights();
        let fiber_weights = self.factored_fiber_weights();
        let calibration =
            negative_split_twisted_birkhoff_calibration_candidate_for_coeff_weights_with_validation(
                self.inner.base.n,
                &self.inner.twist,
                q_degree,
                r_order,
                &base_weights,
                &fiber_weights,
                twisted_calibration_validation_from_env(),
            )?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE factored_twisted_calibration={:.3}s",
                started.elapsed().as_secs_f64()
            );
        }
        let kernel_started = std::time::Instant::now();
        let kernel = Arc::new(GiventalGraphKernel::from_calibration(
            calibration,
            graph_dimension,
        )?);
        if profile_enabled {
            eprintln!(
                "GW_PROFILE factored_graph_kernel={:.3}s",
                kernel_started.elapsed().as_secs_f64()
            );
        }
        Ok(kernel)
    }

    fn coeff_insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries<FactoredRatFun>>, GwError> {
        Ok(self
            .inner
            .insertion_vector(insertion, q_degree)?
            .iter()
            .map(qseries_to_factored)
            .collect())
    }

    fn coeff_direct_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<FactoredRatFun>, GwError> {
        if genus != 0 || !genus_zero_three_primary_layout(insertions) {
            return Ok(None);
        }
        let base_weights = self.factored_base_weights();
        let fiber_weights = self.factored_fiber_weights();
        let insertion_vectors = insertions
            .iter()
            .map(|insertion| self.coeff_insertion_vector(insertion, degree))
            .collect::<Result<Vec<_>, _>>()?;
        twisted_genus_zero_three_primary_value_coeff(
            self.inner.base.n,
            &self.inner.twist,
            degree,
            &self.inner.calibration_mode,
            &base_weights,
            &fiber_weights,
            &insertion_vectors,
        )
        .map(Some)
    }

    fn coeff_scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<Option<FactoredRatFun>, GwError> {
        if genus == 0 {
            return self.factored_genus_zero_two_point_fallback(degree, insertions);
        }
        Ok(self
            .inner
            .scalar_fallback_value(genus, degree, insertions, truncation)?
            .map(FactoredRatFun::from_ratfun))
    }
}

#[cfg(test)]
pub(crate) fn series_s_matrix_to_factored(
    matrix: &SeriesSMatrix,
) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
    SeriesSMatrix::from_coefficients(
        matrix.size(),
        matrix.q_degree(),
        matrix.z_order(),
        matrix
            .coefficients()
            .iter()
            .map(series_matrix_to_factored)
            .collect(),
        matrix.calibration().clone(),
    )
}

#[cfg(test)]
pub(crate) fn series_matrix_to_factored(matrix: &SeriesMatrix) -> SeriesMatrix<FactoredRatFun> {
    SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| row.iter().map(qseries_to_factored).collect())
            .collect(),
    )
}

pub(crate) fn qseries_to_factored(series: &QSeries) -> QSeries<FactoredRatFun> {
    QSeries::from_coeffs(
        series
            .coeffs()
            .iter()
            .cloned()
            .map(FactoredRatFun::from_ratfun)
            .collect(),
    )
}

impl SemisimpleCohftProvider for TwistedProjectiveSpaceProvider {
    type Insertion = Insertion;

    fn colors(&self) -> usize {
        self.base.colors()
    }

    fn descendant_power(&self, insertion: &Self::Insertion) -> usize {
        self.base.descendant_power(insertion)
    }

    fn insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        self.base.insertion_degree(insertions)
    }

    fn virtual_dimension(&self, genus: usize, degree: usize, markings: usize) -> Option<isize> {
        Some(
            self.twist
                .virtual_dimension(self.base.n, genus, degree, markings),
        )
    }

    fn expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        if self.line_mode == TwistedLineMode::FiberEquivariant {
            return None;
        }
        let insertion_degree = self.insertion_degree(insertions)? as isize;
        let constant_dimension = (1 - genus as isize)
            * (self.twist.total_space_dimension(self.base.n) as isize - 3)
            + insertions.len() as isize;
        let slope = self.base.n as isize + 1 - self.twist.degree_sum() as isize;
        if slope == 0 {
            return None;
        }
        let numerator = insertion_degree - constant_dimension;
        if numerator < 0 || numerator % slope != 0 {
            return None;
        }
        Some((numerator / slope) as usize)
    }

    fn candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        if self.line_mode == TwistedLineMode::FiberEquivariant {
            return (0..=degree_max).collect();
        }
        self.twist.candidate_degrees(
            self.base.n,
            genus,
            degree_max,
            insertions.len(),
            self.insertion_degree(insertions),
        )
    }

    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        // The Birkhoff factor produces the fundamental-solution convention.
        // The graph evaluator expects the descendant insertion operator, which
        // is the adjoint with respect to the twisted flat metric.
        let rational_fiber_weights = self.rational_fiber_weights();
        let (s_matrix, flat_metric) = match self.line_mode {
            TwistedLineMode::EarlyRational => {
                let s_matrix =
                    NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
                        self.base.n,
                        self.twist.clone(),
                        q_degree,
                        z_order,
                        self.base_weights().to_vec(),
                        rational_fiber_weights.clone(),
                    )?
                    .birkhoff_descendant_s_matrix(z_order)?;
                let flat_metric = twisted_inverse_euler_flat_metric_matrix(
                    self.base.n,
                    q_degree,
                    &self.twist,
                    self.base_weights(),
                    &rational_fiber_weights,
                )?;
                (s_matrix, flat_metric)
            }
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                let s_matrix = NegativeSplitLineHypergeometricModel::from_ratfun_weights(
                    self.base.n,
                    self.twist.clone(),
                    q_degree,
                    z_order,
                    base_weights.clone(),
                    &fiber_weights,
                )?
                .birkhoff_descendant_s_matrix(z_order)?;
                let flat_metric = twisted_inverse_euler_flat_metric_matrix_ratfun(
                    self.base.n,
                    q_degree,
                    &self.twist,
                    &base_weights,
                    &fiber_weights,
                )?;
                (s_matrix, flat_metric)
            }
        };
        metric_adjoint_descendant_s_matrix(s_matrix, &flat_metric)
    }

    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        // This is where the twisted target becomes a generic semisimple CohFT:
        // Birkhoff canonical data + QRR R-recursion are packaged into the same
        // kernel interface used by ordinary P^n.
        static CACHE: OnceLock<
            Mutex<HashMap<TwistedGraphKernelCacheKey, Arc<GiventalGraphKernel>>>,
        > = OnceLock::new();
        let rational_fiber_weights = self.rational_fiber_weights();
        let validation = twisted_calibration_validation_from_env();
        let key = TwistedGraphKernelCacheKey {
            n: self.base.n,
            twist_degrees: self.twist.degrees().to_vec(),
            q_degree,
            r_order,
            graph_dimension,
            base_weights: self.base_weights().to_vec(),
            fiber_weights: rational_fiber_weights.clone(),
            fiber_parameter_names: self.fiber_parameter_names.clone(),
            line_mode: self.line_mode.clone(),
            calibration_mode: self.calibration_mode.clone(),
            validation,
        };
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(kernel) = cache.lock().unwrap().get(&key).cloned() {
            return Ok(kernel);
        }

        let calibration = match self.line_mode {
            TwistedLineMode::EarlyRational => {
                negative_split_twisted_birkhoff_calibration_candidate_with_mode_and_validation(
                    self.base.n,
                    &self.twist,
                    q_degree,
                    r_order,
                    self.base_weights(),
                    &rational_fiber_weights,
                    self.calibration_mode.clone(),
                    validation,
                )?
            }
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                negative_split_twisted_birkhoff_calibration_candidate_for_ratfun_weights_with_validation(
                    self.base.n,
                    &self.twist,
                    q_degree,
                    r_order,
                    &base_weights,
                    &fiber_weights,
                    validation,
                )?
            }
        };
        let kernel = Arc::new(GiventalGraphKernel::from_calibration(
            calibration,
            graph_dimension,
        )?);
        cache.lock().unwrap().insert(key, kernel.clone());
        Ok(kernel)
    }

    fn insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError> {
        self.base.insertion_vector(insertion, q_degree)
    }

    fn direct_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        if genus != 0 || !genus_zero_three_primary_layout(insertions) {
            return Ok(None);
        }
        let base_weights = self.ratfun_base_weights();
        let fiber_weights = self.ratfun_fiber_weights();
        let insertion_vectors = insertions
            .iter()
            .map(|insertion| self.insertion_vector(insertion, degree))
            .collect::<Result<Vec<_>, _>>()?;
        twisted_genus_zero_three_primary_value_coeff(
            self.base.n,
            &self.twist,
            degree,
            &self.calibration_mode,
            &base_weights,
            &fiber_weights,
            &insertion_vectors,
        )
        .map(Some)
    }

    fn scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        if genus == 0 {
            return self.genus_zero_two_point_fallback(degree, insertions);
        }
        Ok(None)
    }
}

pub(crate) fn metric_adjoint_descendant_s_matrix(
    s_matrix: SeriesSMatrix,
    flat_metric: &SeriesMatrix,
) -> Result<SeriesSMatrix, GwError> {
    metric_adjoint_descendant_s_matrix_coeff(s_matrix, flat_metric)
}

pub(crate) fn metric_adjoint_descendant_s_matrix_coeff<C: Coeff>(
    s_matrix: SeriesSMatrix<C>,
    flat_metric: &SeriesMatrix<C>,
) -> Result<SeriesSMatrix<C>, GwError> {
    // Converts S to its metric adjoint: S^* = eta^{-1} S^T eta.  This is the
    // correct action on covector insertions in the graph formula, and is
    // especially important after inverse-Euler twisting changes the pairing.
    let metric_inverse = invert_series_matrix_coeff(flat_metric)?;
    metric_adjoint_descendant_s_matrix_with_inverse_coeff(s_matrix, flat_metric, &metric_inverse)
}

pub(crate) fn metric_adjoint_descendant_s_matrix_with_inverse_coeff<C: Coeff>(
    s_matrix: SeriesSMatrix<C>,
    flat_metric: &SeriesMatrix<C>,
    metric_inverse: &SeriesMatrix<C>,
) -> Result<SeriesSMatrix<C>, GwError> {
    // Converts S to its metric adjoint: S^* = eta^{-1} S^T eta.  This overload
    // accepts eta^{-1} directly, which is important for factored twisted
    // theories where eta has a closed Vandermonde inverse and Gaussian
    // elimination would expand symbolic rational sums.
    let coefficients = s_matrix
        .coefficients()
        .iter()
        .map(|matrix| metric_inverse.mul(&matrix.transpose()).mul(flat_metric))
        .collect::<Vec<_>>();
    SeriesSMatrix::from_coefficients(
        s_matrix.size(),
        s_matrix.q_degree(),
        s_matrix.z_order(),
        coefficients,
        CalibrationId(format!("{}-metric-adjoint", s_matrix.calibration().0)),
    )
}
