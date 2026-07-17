//! The Givental stable-graph evaluation engine: graph kernels, the master
//! evaluator and its parallel contraction workers, external-leg kernels, and
//! the public compute / series / packed-resolvent entry points.

// The coefficient-generic entry points retain the prefixed compatibility
// trait until their public signatures migrate to the canonical generic
// provider in a later stage.
#![allow(deprecated)]

use super::*;
use crate::bounded_cache::{BoundedCache, TARGET_RECONSTRUCTION_CACHE_CAPACITY};
use crate::factored::FactoredRatFun;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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
        Err(limit @ GwError::ResourceLimit { .. }) => {
            // A graph-enumeration envelope must not hide an exact closed-form
            // seed (notably point theory). If no seed covers the request,
            // preserve the structured limit instead of replacing it with the
            // seed backend's broad UnsupportedInvariant error.
            match validation::seed_compute(req, "givental-seed") {
                Ok(result) => Ok(result),
                Err(GwError::UnsupportedInvariant(_)) => Err(limit),
                Err(error) => Err(error),
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
    let value = WittenKontsevich::shared().psi_integral(genus, descendant_powers);
    Ok(RatFun::from_rational(value))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AncestorLegTerm<C = RatFun> {
    base_power: usize,
    vector: Vec<QSeries<C>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LegFactorOption<C = RatFun> {
    power: usize,
    coefficient: QSeries<C>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EdgeFactorOption<C = RatFun> {
    left_power: usize,
    right_power: usize,
    coefficient: QSeries<C>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct VertexContributionKey {
    genus: usize,
    color: usize,
    powers: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct GraphEvalProfile {
    enabled: bool,
    started: Instant,
    calibration_elapsed: Duration,
    option_elapsed: Duration,
    stable_graph_elapsed: Duration,
    graph_elapsed: Duration,
    prepared_stable_graphs: usize,
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
            enabled: crate::env_flag("GW_PROFILE"),
            started: Instant::now(),
            calibration_elapsed: Duration::ZERO,
            option_elapsed: Duration::ZERO,
            stable_graph_elapsed: Duration::ZERO,
            graph_elapsed: Duration::ZERO,
            prepared_stable_graphs: 0,
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

    fn add_stable_graph_elapsed(&mut self, elapsed: Duration) {
        if self.enabled {
            self.stable_graph_elapsed += elapsed;
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
            "GW_PROFILE total={:.3}s calibration={:.3}s options={:.3}s stable_graph_generation={:.3}s graphs={:.3}s prepared_stable_graphs={} stable_graphs={} colorings={} recursion_calls={} leaves={} vertex_cache_hits={} vertex_cache_misses={} translation_terms={} leg_options={} edge_options={}",
            self.started.elapsed().as_secs_f64(),
            self.calibration_elapsed.as_secs_f64(),
            self.option_elapsed.as_secs_f64(),
            self.stable_graph_elapsed.as_secs_f64(),
            self.graph_elapsed.as_secs_f64(),
            self.prepared_stable_graphs,
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
pub(crate) struct GraphKernelCacheKey {
    n: usize,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
    equivariant: bool,
    weights: Vec<Rational>,
}

#[derive(Debug)]
pub struct GiventalGraphKernel<C = RatFun> {
    calibration: SemisimpleCalibration<C>,
    inverse_r: Vec<SeriesMatrix<C>>,
    translation: Vec<Vec<QSeries<C>>>,
    edge_options: Vec<Vec<Vec<EdgeFactorOption<C>>>>,
    vertex_cache: Mutex<HashMap<VertexContributionKey, Arc<QSeries<C>>>>,
}

impl<C: Coeff> GiventalGraphKernel<C> {
    /// Builds the Feynman-rule kernel from a semisimple calibration.
    ///
    /// This performs the universal part of quantization:
    /// `R -> R^{-1}`, `R^{-1}1 -> T`, and `R^{-1},eta^{-1} ->` edge
    /// propagators.  It does not inspect target geometry.
    pub fn from_calibration(
        calibration: SemisimpleCalibration<C>,
        graph_dimension: usize,
    ) -> Result<Self, GwError> {
        let q_degree = calibration.r_matrix.q_degree();
        let inverse_r = inverse_r_coefficients(calibration.r_matrix.coefficients());
        let unit = calibration.relative_sqrt_delta_inverse.clone();
        let translation = translation_coefficients(&inverse_r, &unit, q_degree);
        Self::from_parts(calibration, inverse_r, translation, graph_dimension)
    }

    /// Builds a graph kernel when a caller has already supplied `R^{-1}` and
    /// translation coefficients.
    ///
    /// Twisted-theory experiments use this to test alternate QRR/Birkhoff
    /// calibrations without reusing the default projective-space construction.
    pub fn from_parts(
        calibration: SemisimpleCalibration<C>,
        inverse_r: Vec<SeriesMatrix<C>>,
        translation: Vec<Vec<QSeries<C>>>,
        graph_dimension: usize,
    ) -> Result<Self, GwError> {
        let q_degree = calibration.r_matrix.q_degree();
        let edge_coefficients = edge_propagator_coefficients(
            &inverse_r,
            &calibration.metric,
            graph_dimension,
            q_degree,
        )?;
        let edge_options = edge_options_by_color(&edge_coefficients);
        Ok(Self {
            calibration,
            inverse_r,
            translation,
            edge_options,
            vertex_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn calibration(&self) -> &SemisimpleCalibration<C> {
        &self.calibration
    }

    pub fn inverse_r(&self) -> &[SeriesMatrix<C>] {
        &self.inverse_r
    }

    pub fn translation(&self) -> &[Vec<QSeries<C>>] {
        &self.translation
    }
}

/// Materializes all graph-local data derived from a calibration.
///
/// The mathematical content is the conversion from a global `R`-matrix to the
/// Feynman rules used on a stable graph: `R^{-1}` on legs, the symplectic edge
/// propagator, and the translation vector.
pub(crate) fn projective_space_graph_kernel(
    n: usize,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
    equivariant: bool,
    weights: &[Rational],
) -> Result<Arc<GiventalGraphKernel>, GwError> {
    static CACHE: OnceLock<Mutex<BoundedCache<GraphKernelCacheKey, Arc<GiventalGraphKernel>>>> =
        OnceLock::new();
    let key = GraphKernelCacheKey {
        n,
        q_degree,
        r_order,
        graph_dimension,
        equivariant,
        weights: if equivariant {
            Vec::new()
        } else {
            weights.to_vec()
        },
    };
    let cache =
        CACHE.get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
    if let Some(kernel) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(kernel);
    }

    let calibration = if equivariant {
        projective_space_j_calibration(n, q_degree, r_order)?
    } else {
        projective_space_j_calibration_at_lambda_weights(n, q_degree, r_order, weights)?
    };
    let kernel = Arc::new(GiventalGraphKernel::from_calibration(
        calibration,
        graph_dimension,
    )?);
    cache.lock().unwrap().insert(key, kernel.clone());
    Ok(kernel)
}

/// Public ordinary `P^n` Gromov-Witten computation by the Givental graph sum.
///
/// This is the production path for projective-space invariants.  It wraps the
/// generic semisimple evaluator with projective-space dimension checks and
/// result labels.
pub fn compute_by_givental_graphs(req: &InvariantRequest) -> Result<InvariantResult, GwError> {
    let provider = ProjectiveSpaceProvider::new(req.n, req.equivariant);
    provider.validate_insertions(&req.insertions)?;

    if !provider.degree_is_effective(req.degree) {
        return Ok(InvariantResult {
            value: RatFun::zero(),
            engine: "givental-effective-degree",
            notes: vec![format!(
                "P^{} has no effective curve class of degree {}",
                req.n, req.degree
            )],
        });
    }

    if let Some((virtual_dimension, total_degree)) =
        dimension_mismatch(&provider, req.genus, req.degree, &req.insertions)
    {
        return Ok(InvariantResult {
            value: RatFun::zero(),
            engine: "givental-r-graph",
            notes: vec![format!(
                "dimension mismatch gives zero: virtual dimension {virtual_dimension}, insertion degree {total_degree}"
            )],
        });
    }

    if !is_stable_cohft_range(req.genus, req.insertions.len()) {
        return Err(GwError::UnsupportedInvariant(
            "Givental graph expansion is implemented for stable (g,n) CohFT ranges only"
                .to_string(),
        ));
    }

    if req.equivariant {
        // Symbolic lambda parameters: build the kernel and contract over
        // factored coefficients from the start, expanding only the final
        // value.  Building R^{-1} and the edge propagators in expanded RatFun
        // costs orders of magnitude more than the calibration itself.
        let value = compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
            &FactoredProjectiveSpaceProvider(provider),
            req.genus,
            req.degree,
            &req.insertions,
            req.truncation.as_ref(),
        )?;
        return Ok(InvariantResult {
            value: value.to_ratfun(),
            engine: "givental-r-graph",
            notes: vec![
                "computed by truncated J-calibrated R-matrix stable-graph expansion; result remains equivariant"
                    .to_string(),
            ],
        });
    }

    let value = compute_semisimple_graph_value(
        &provider,
        req.genus,
        req.degree,
        &req.insertions,
        req.truncation.as_ref(),
    )?;

    if provider.specialized_nonequivariant() {
        return Ok(InvariantResult {
            value,
            engine: "givental-r-graph-lambda-line",
            notes: vec![
                "computed by J-calibrated S/R stable-graph expansion after early generic lambda-line specialization"
                    .to_string(),
            ],
        });
    }

    let limit = value.nonequivariant_limit_line(req.n, &provider.weights)?;
    Ok(InvariantResult {
        value: RatFun::from_rational(limit),
        engine: "givental-r-graph-limit",
        notes: vec![
            "computed by truncated J-calibrated R-matrix stable-graph expansion and lambda-line nonequivariant limit"
                .to_string(),
        ],
    })
}

pub(crate) fn dimension_mismatch<P>(
    provider: &P,
    genus: usize,
    degree: usize,
    insertions: &[P::Insertion],
) -> Option<(isize, usize)>
where
    P: SemisimpleCohftProvider,
{
    let total_degree = provider.insertion_degree(insertions)?;
    let virtual_dimension = provider.virtual_dimension(genus, degree, insertions.len())?;
    provider
        .vanishes_by_dimension(virtual_dimension, total_degree)
        .then_some((virtual_dimension, total_degree))
}

fn exact_dimension_mismatch<P>(
    provider: &P,
    genus: usize,
    degree: usize,
    insertions: &[P::Insertion],
) -> Option<(isize, usize)>
where
    P: SemisimpleCohftProvider,
{
    let total_degree = provider.insertion_degree(insertions)?;
    let virtual_dimension = provider.virtual_dimension(genus, degree, insertions.len())?;
    (usize::try_from(virtual_dimension).ok() != Some(total_degree))
        .then_some((virtual_dimension, total_degree))
}

/// Computes a single coefficient using the generic semisimple Givental graph
/// engine.
///
/// Public `P^n` requests still go through `compute_by_givental_graphs` for
/// projective-space dimension checks and result labeling.  Extension providers
/// can call this directly once they can supply S/R calibration data and flat
/// insertion vectors.
pub fn compute_semisimple_graph_value<P>(
    provider: &P,
    genus: usize,
    degree: usize,
    insertions: &[P::Insertion],
    truncation: Option<&Truncation>,
) -> Result<RatFun, GwError>
where
    P: SemisimpleCohftProvider,
{
    if !provider.degree_is_effective(degree) {
        return Ok(RatFun::zero());
    }
    if let Some(value) = provider.direct_value(genus, degree, insertions, truncation)? {
        return Ok(value);
    }
    if !is_stable_cohft_range(genus, insertions.len()) {
        if let Some(value) =
            provider.scalar_fallback_value(genus, degree, insertions, truncation)?
        {
            return Ok(value);
        }
    }
    let total = compute_semisimple_graph_series(provider, genus, degree, insertions, truncation)?;
    Ok(total.coeff(degree).cloned().unwrap_or_else(RatFun::zero))
}

pub fn compute_semisimple_graph_value_with_coeff<C, P>(
    provider: &P,
    genus: usize,
    degree: usize,
    insertions: &[P::Insertion],
    truncation: Option<&Truncation>,
) -> Result<C, GwError>
where
    C: Coeff + Send + Sync,
    P: CoefficientSemisimpleCohftProvider<C>,
{
    if !provider.coeff_degree_is_effective(degree) {
        return Ok(C::zero());
    }
    if let Some(value) = provider.coeff_direct_value(genus, degree, insertions, truncation)? {
        return Ok(value);
    }
    if !is_stable_cohft_range(genus, insertions.len()) {
        if let Some(value) =
            provider.coeff_scalar_fallback_value(genus, degree, insertions, truncation)?
        {
            return Ok(value);
        }
    }
    let total = compute_semisimple_graph_series_with_coeff(
        provider, genus, degree, insertions, truncation,
    )?;
    Ok(total.coeff(degree).cloned().unwrap_or_else(C::zero))
}

/// Computes all coefficients `q^0, ..., q^degree_max` for one fixed insertion
/// list with a single graph-kernel construction and a single stable-graph sum.
///
/// This is useful for validation rows such as local Calabi-Yau no-insertion
/// tables: repeated calls to [`compute_semisimple_graph_value`] rebuild and
/// reevaluate the same truncated graph problem at every degree, while this
/// helper evaluates once at the largest requested degree and then extracts the
/// coefficients.
pub fn compute_semisimple_graph_coefficients<P>(
    provider: &P,
    genus: usize,
    degree_max: usize,
    insertions: &[P::Insertion],
    truncation: Option<&Truncation>,
) -> Result<Vec<RatFun>, GwError>
where
    P: SemisimpleCohftProvider,
{
    compute_semisimple_graph_coefficient_range(
        provider, genus, 0, degree_max, insertions, truncation,
    )
}

/// Computes a contiguous coefficient range from one truncated graph sum.
///
/// Unlike [`compute_semisimple_graph_coefficients`], this avoids cloning
/// coefficients below `degree_min`, which matters for diagnostic rows whose
/// intermediate rational functions can be much larger than the final
/// non-equivariant numbers.
pub fn compute_semisimple_graph_coefficient_range<P>(
    provider: &P,
    genus: usize,
    degree_min: usize,
    degree_max: usize,
    insertions: &[P::Insertion],
    truncation: Option<&Truncation>,
) -> Result<Vec<RatFun>, GwError>
where
    P: SemisimpleCohftProvider,
{
    if degree_min > degree_max {
        return Ok(Vec::new());
    }
    let total =
        compute_semisimple_graph_series(provider, genus, degree_max, insertions, truncation)?;
    Ok((degree_min..=degree_max)
        .map(|degree| total.coeff(degree).cloned().unwrap_or_else(RatFun::zero))
        .collect())
}

/// Computes one stable graph's contribution to the bounded descendant
/// potential of ordinary `P^n`.
///
/// This is the formula renderer's graph-local rational path.  It uses the same
/// external-leg kernel as `series`: each marking is left as an open
/// `(canonical color, psi power)` state while the selected stable graph is
/// fully contracted over colors, edge propagators, translations, and
/// point-theory vertex integrals.  The final loop then attaches bounded flat
/// insertions through the calibrated `S`, `Psi^{-1}`, and `R^{-1}` leg options.
pub fn projective_graph_bounded_potential_coefficients(
    n: usize,
    genus: usize,
    markings: usize,
    graph_index: usize,
    degree_max: usize,
    max_descendant_power: usize,
    equivariant: bool,
) -> Result<Vec<SeriesCoefficient>, GwError> {
    if !is_stable_cohft_range(genus, markings) {
        return Err(GwError::UnsupportedInvariant(
            "graph-local rational potential is implemented for stable (g,m) CohFT ranges only"
                .to_string(),
        ));
    }

    let provider = ProjectiveSpaceProvider::try_new(n, equivariant)?;
    let colors = provider.colors();
    let graph_dimension = checked_stable_graph_work_dimension(genus, markings)?;
    let needed_r_order = graph_dimension.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("stable-graph R-order overflow".to_string())
    })?;
    let graph_kernel = provider.graph_kernel(degree_max, needed_r_order, graph_dimension)?;
    let mut profile = GraphEvalProfile::new();
    let graphs = profiled_prepared_stable_graphs(genus, markings, colors, &mut profile)?;
    let prepared = graphs.get(graph_index).ok_or_else(|| {
        GwError::UnsupportedInvariant(format!(
            "stable graph index {graph_index} is out of range for (g,m)=({genus},{markings})"
        ))
    })?;

    profile.stable_graphs = 1;
    profile.edge_options = graph_kernel
        .edge_options
        .iter()
        .flat_map(|row| row.iter())
        .map(Vec::len)
        .sum();
    let external_kernel = evaluate_external_graphs_auto(
        std::slice::from_ref(prepared),
        markings,
        colors,
        &graph_kernel,
        degree_max,
        graph_dimension,
        &mut profile,
    );

    let descendant_s = provider.descendant_s_matrix(degree_max, max_descendant_power)?;
    let basis = crate::insertion_basis(n, max_descendant_power);
    let mut out = Vec::new();
    for insertions in crate::insertion_monomials(&basis, markings) {
        let insertion_terms = ancestor_insertion_terms_from_provider(
            &provider,
            &insertions,
            &descendant_s,
            &graph_kernel.calibration.psi_inverse,
            degree_max,
            graph_dimension,
        )?;
        let leg_options = leg_options_by_marking_color(
            &insertion_terms,
            &graph_kernel.inverse_r,
            degree_max,
            graph_dimension,
            colors,
        );

        for degree in provider.candidate_degrees_from_dimension(genus, degree_max, &insertions) {
            if dimension_mismatch(&provider, genus, degree, &insertions).is_some() {
                continue;
            }
            let value = contract_external_leg_kernel_coeff(&external_kernel, &leg_options, degree);
            if !value.is_zero() {
                out.push(SeriesCoefficient {
                    degree,
                    insertions: insertions.clone(),
                    value,
                });
            }
        }
    }
    Ok(out)
}

/// Computes the full truncated q-series produced by the generic semisimple
/// Givental graph engine for the given insertions.
pub fn compute_semisimple_graph_series<P>(
    provider: &P,
    genus: usize,
    q_degree: usize,
    insertions: &[P::Insertion],
    truncation: Option<&Truncation>,
) -> Result<QSeries, GwError>
where
    P: SemisimpleCohftProvider,
{
    let mut profile = GraphEvalProfile::new();
    let max_descendant_power = insertions
        .iter()
        .map(|insertion| provider.descendant_power(insertion))
        .max()
        .unwrap_or(0);

    if !is_stable_cohft_range(genus, insertions.len()) {
        return Err(GwError::UnsupportedInvariant(
            "Givental graph expansion is implemented for stable (g,n) CohFT ranges only"
                .to_string(),
        ));
    }

    // The largest total psi degree on a stable-curve vertex is the dimension
    // of Mbar_{g,n}.  This one number bounds the necessary `R`, edge, and
    // translation powers for the whole graph sum.
    let graph_dimension = checked_stable_graph_work_dimension(genus, insertions.len())?;
    let needed_r_order = graph_dimension.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("stable-graph R-order overflow".to_string())
    })?;
    let needed_s_order = max_descendant_power;
    let needed_z_order = needed_r_order.max(needed_s_order);
    let z_order = truncation
        .map(|truncation| truncation.z_order)
        .unwrap_or(needed_z_order);
    if z_order < needed_z_order {
        return Err(GwError::TruncationTooLow);
    }

    let calibration_started = Instant::now();
    let kernel = provider.graph_kernel(q_degree, needed_r_order, graph_dimension)?;
    profile.add_calibration_elapsed(calibration_started.elapsed());

    let options_started = Instant::now();
    let leg_options = if insertions.is_empty() {
        Vec::new()
    } else {
        // Descendant insertions first become ancestor insertions via `S`, then
        // move from the flat basis to the canonical basis via `Psi^{-1}`, then
        // receive the graph-leg action of `R^{-1}`.
        let descendant_s = provider.descendant_s_matrix(q_degree, needed_s_order)?;
        let insertion_terms = ancestor_insertion_terms_from_provider(
            provider,
            insertions,
            &descendant_s,
            &kernel.calibration.psi_inverse,
            q_degree,
            graph_dimension,
        )?;
        leg_options_by_marking_color(
            &insertion_terms,
            &kernel.inverse_r,
            q_degree,
            graph_dimension,
            provider.colors(),
        )
    };
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
    if profile.enabled {
        eprintln!(
            "GW_PROFILE coeff_options_total={:.3}s leg_options={} edge_options={}",
            options_started.elapsed().as_secs_f64(),
            profile.leg_options,
            profile.edge_options
        );
    }

    let graphs =
        profiled_prepared_stable_graphs(genus, insertions.len(), provider.colors(), &mut profile)?;
    profile.stable_graphs = graphs.len();

    let graphs_started = Instant::now();
    // When the whole problem is numeric — early lambda-line specialization
    // leaves only constant coefficients — evaluate over plain rationals
    // instead of dragging constant RatFuns through every convolution.
    let total = if crate::env_flag("GWAI_DISABLE_RATIONAL_GRAPH") {
        None
    } else {
        evaluate_rational_graphs_if_possible(
            graphs.as_ref(),
            &leg_options,
            &kernel,
            q_degree,
            graph_dimension,
            &mut profile,
        )
    }
    .or_else(|| {
        // Genuinely symbolic coefficients (equivariant lambda parameters):
        // contract with factored denominators so products of linear factors
        // never expand mid-computation, and expand once at the very end.
        if crate::env_flag("GWAI_DISABLE_FACTORED_GRAPH") {
            None
        } else {
            Some(evaluate_factored_graphs(
                graphs.as_ref(),
                &leg_options,
                &kernel,
                q_degree,
                graph_dimension,
                &mut profile,
            ))
        }
    })
    .unwrap_or_else(|| {
        evaluate_scalar_graphs_parallel(
            graphs.as_ref(),
            &leg_options,
            &kernel,
            q_degree,
            graph_dimension,
            &mut profile,
        )
    });
    profile.add_graph_elapsed(graphs_started.elapsed());
    profile.finish();
    Ok(QSeries::from_coeffs(
        total
            .coeffs()
            .iter()
            .enumerate()
            .map(|(degree, coefficient)| {
                if provider.degree_is_effective(degree) {
                    coefficient.clone()
                } else {
                    RatFun::zero()
                }
            })
            .collect(),
    ))
}

pub fn compute_semisimple_graph_series_with_coeff<C, P>(
    provider: &P,
    genus: usize,
    q_degree: usize,
    insertions: &[P::Insertion],
    truncation: Option<&Truncation>,
) -> Result<QSeries<C>, GwError>
where
    C: Coeff + Send + Sync,
    P: CoefficientSemisimpleCohftProvider<C>,
{
    let mut profile = GraphEvalProfile::new();
    let max_descendant_power = insertions
        .iter()
        .map(|insertion| provider.coeff_descendant_power(insertion))
        .max()
        .unwrap_or(0);

    if !is_stable_cohft_range(genus, insertions.len()) {
        return Err(GwError::UnsupportedInvariant(
            "Givental graph expansion is implemented for stable (g,n) CohFT ranges only"
                .to_string(),
        ));
    }

    let graph_dimension = checked_stable_graph_work_dimension(genus, insertions.len())?;
    let needed_r_order = graph_dimension.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("stable-graph R-order overflow".to_string())
    })?;
    let needed_s_order = max_descendant_power;
    let needed_z_order = needed_r_order.max(needed_s_order);
    let z_order = truncation
        .map(|truncation| truncation.z_order)
        .unwrap_or(needed_z_order);
    if z_order < needed_z_order {
        return Err(GwError::TruncationTooLow);
    }

    let calibration_started = Instant::now();
    let kernel = provider.coeff_graph_kernel(q_degree, needed_r_order, graph_dimension)?;
    profile.add_calibration_elapsed(calibration_started.elapsed());
    if profile.enabled {
        eprintln!(
            "GW_PROFILE coeff_kernel_total={:.3}s",
            calibration_started.elapsed().as_secs_f64()
        );
    }

    let options_started = Instant::now();
    let leg_options = if insertions.is_empty() {
        Vec::new()
    } else {
        let descendant_started = Instant::now();
        let descendant_s = provider.coeff_descendant_s_matrix(q_degree, needed_s_order)?;
        if profile.enabled {
            eprintln!(
                "GW_PROFILE coeff_descendant_s={:.3}s",
                descendant_started.elapsed().as_secs_f64()
            );
        }
        let insertion_started = Instant::now();
        let insertion_terms = ancestor_insertion_terms_from_provider(
            provider,
            insertions,
            &descendant_s,
            &kernel.calibration.psi_inverse,
            q_degree,
            graph_dimension,
        )?;
        if profile.enabled {
            eprintln!(
                "GW_PROFILE coeff_ancestor_terms={:.3}s",
                insertion_started.elapsed().as_secs_f64()
            );
        }
        let leg_started = Instant::now();
        let leg_options = leg_options_by_marking_color(
            &insertion_terms,
            &kernel.inverse_r,
            q_degree,
            graph_dimension,
            provider.coeff_colors(),
        );
        if profile.enabled {
            eprintln!(
                "GW_PROFILE coeff_leg_options={:.3}s",
                leg_started.elapsed().as_secs_f64()
            );
        }
        leg_options
    };
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

    let graphs = profiled_prepared_stable_graphs(
        genus,
        insertions.len(),
        provider.coeff_colors(),
        &mut profile,
    )?;
    profile.stable_graphs = graphs.len();

    let graphs_started = Instant::now();
    let total = evaluate_scalar_graphs_parallel(
        graphs.as_ref(),
        &leg_options,
        &kernel,
        q_degree,
        graph_dimension,
        &mut profile,
    );
    profile.add_graph_elapsed(graphs_started.elapsed());
    if profile.enabled {
        eprintln!(
            "GW_PROFILE coeff_graphs_total={:.3}s",
            graphs_started.elapsed().as_secs_f64()
        );
    }
    profile.finish();
    Ok(QSeries::from_coeffs(
        total
            .coeffs()
            .iter()
            .enumerate()
            .map(|(degree, coefficient)| {
                if provider.coeff_degree_is_effective(degree) {
                    coefficient.clone()
                } else {
                    C::zero()
                }
            })
            .collect(),
    ))
}

const MASTER_SHARED_KERNEL_MAX_MARKINGS: usize = 2;
const MASTER_MIN_SHARED_KERNEL_TASKS: usize = 8;
const MASTER_MIN_RESTRICTED_KERNEL_TASKS: usize = 2;

/// Batched sparse potential evaluator for many coefficients at once.
///
/// Mathematically this computes the same coefficients as repeated calls to
/// `compute_semisimple_graph_value`.  The reorganization is purely algorithmic:
/// it shares graph kernels and, for small marking counts, precontracts the
/// entire stable-graph sum into an external-leg tensor.
pub fn compute_series_master(req: &SeriesRequest) -> Result<Option<SeriesResult>, GwError> {
    req.validate()?;
    if req.mode != ComputeMode::Givental {
        return Ok(None);
    }

    let provider = ProjectiveSpaceProvider::try_new(req.n, req.equivariant)?;
    compute_series_master_with_provider(req, provider)
}

pub fn compute_series_master_with_provider<P>(
    req: &SeriesRequest,
    provider: P,
) -> Result<Option<SeriesResult>, GwError>
where
    P: SemisimpleCohftProvider<Insertion = Insertion>,
{
    req.validate()?;
    if req.mode != ComputeMode::Givental {
        return Ok(None);
    }

    let mut prepared_coefficients = Vec::new();
    let mut contraction_tasks = Vec::new();
    let mut restricted_contraction_tasks = (0..=req.max_markings)
        .map(|_| {
            (0..=req.degree_max)
                .map(|_| Vec::<MasterContractionTask>::new())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut notes = vec![
        "series enumerates a bounded sparse descendant potential; unsupported coefficients are skipped and profiles forced to vanish by the provider's coefficient-ring grading are dimension-pruned"
            .to_string(),
    ];
    let mut complete = true;
    let basis = crate::insertion_basis(req.n, req.max_descendant_power);
    let mut candidates_by_degree = vec![Vec::<Vec<Insertion>>::new(); req.degree_max + 1];

    for markings in 0..=req.max_markings {
        for insertions in crate::insertion_monomials(&basis, markings) {
            for degree in
                provider.candidate_degrees_from_dimension(req.genus, req.degree_max, &insertions)
            {
                candidates_by_degree[degree].push(insertions.clone());
            }
        }
    }

    let shared_kernel_task_counts =
        shared_kernel_task_counts(req, &provider, &candidates_by_degree);
    let mut evaluator = GiventalMasterEvaluator::with_provider(req, provider);
    evaluator.set_shared_kernel_task_counts(shared_kernel_task_counts.clone());
    for (markings, count) in shared_kernel_task_counts.iter().copied().enumerate() {
        if (MASTER_MIN_RESTRICTED_KERNEL_TASKS..MASTER_MIN_SHARED_KERNEL_TASKS).contains(&count) {
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
            if evaluator.is_dimension_mismatch(degree, &insertions) {
                continue;
            }

            if evaluator.can_use_external_leg_kernel(&insertions) {
                match evaluator.prepare_contraction_task(current_ordinal, degree, &insertions) {
                    Ok(task) => contraction_tasks.push(task),
                    Err(GwError::UnsupportedInvariant(msg)) => {
                        complete = false;
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
                    Ok(task) => restricted_contraction_tasks[insertions.len()][degree].push(task),
                    Err(GwError::UnsupportedInvariant(msg)) => {
                        complete = false;
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
                    complete = false;
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
    for (markings, by_degree) in restricted_contraction_tasks.iter().enumerate() {
        for (degree, tasks) in by_degree.iter().enumerate() {
            if tasks.is_empty() {
                continue;
            }
            prepared_coefficients.extend(evaluator.contract_restricted_tasks(
                markings,
                degree,
                tasks,
                req.include_zero,
            )?);
        }
    }
    prepared_coefficients.sort_by_key(|entry| entry.ordinal);
    let coefficients = prepared_coefficients
        .into_iter()
        .map(|entry| entry.coefficient)
        .collect();

    Ok(Some(SeriesResult {
        coefficients,
        engine: "givental-master-series",
        complete,
        notes,
    }))
}

/// Computes a fixed-degree labelled resolvent potential by sharing the
/// Givental graph contraction across all descendant/cohomology coefficients.
///
/// The coefficient-wise resolver calls the full invariant evaluator once for
/// every pair `(a_i,k_i)`.  This routine instead builds the stable-graph kernel
/// once for fixed `(g,d,m)`, precontracts the graph sum into a restricted
/// external-leg tensor, and then attaches every resolvent leg coefficient to
/// that tensor.  The output is still the finite Laurent polynomial in
/// `z_i^{-1}`; only the algorithm is reorganized.
pub fn compute_packed_resolvent_with_provider<P, N>(
    req: &ResolventRequest,
    provider: P,
    engine: &'static str,
    note: impl Into<String>,
    mut normalize: N,
) -> Result<ResolventResult, GwError>
where
    P: SemisimpleCohftProvider<Insertion = Insertion>,
    N: FnMut(RatFun) -> Result<RatFun, GwError>,
{
    validate_packed_resolvent_virtual_dimension(
        req,
        provider.virtual_dimension(req.genus, req.degree, req.markings),
    )?;

    if !provider.degree_is_effective(req.degree) {
        return Ok(ResolventResult {
            value: ResolventPolynomial::zero(),
            candidate_terms: 0,
            nonzero_terms: 0,
            engine: "packed-resolvent-ineffective-degree",
            notes: vec![format!(
                "the target has no effective curve class of degree {}",
                req.degree
            )],
        });
    }

    if req.virtual_dimension < 0 {
        return Ok(ResolventResult {
            value: ResolventPolynomial::zero(),
            candidate_terms: 0,
            nonzero_terms: 0,
            engine: "packed-resolvent-empty-dimension",
            notes: vec![format!(
                "virtual dimension {} is negative, so the packed resolvent generating function is zero",
                req.virtual_dimension
            )],
        });
    }

    if req.markings == 0 {
        return compute_no_marking_packed_resolvent(req, &provider, engine, note, normalize);
    }
    if !is_stable_cohft_range(req.genus, req.markings) {
        return Err(GwError::UnsupportedInvariant(
            "packed resolvent graph expansion is implemented for stable (g,n) CohFT ranges only"
                .to_string(),
        ));
    }

    let max_descendant_power = req.virtual_dimension as usize;
    let mut evaluator = GiventalMasterEvaluator::for_problem(
        provider,
        req.genus,
        req.degree,
        max_descendant_power,
        None,
    );
    let mut tasks = Vec::new();
    let mut task_indices = Vec::<ResolventIndex>::new();
    let candidate_terms = enumerate_resolvent_indices(req, |index| {
        let insertions = index.to_insertions(req.target_n);
        // A resolvent is intentionally the exact virtual-dimension slice,
        // even when its coefficients retain equivariant parameters.
        if exact_dimension_mismatch(&evaluator.provider, req.genus, req.degree, &insertions)
            .is_some()
        {
            return Ok(());
        }
        evaluator.validate_truncation(req.markings, &insertions)?;
        let mut leg_options = Vec::with_capacity(req.markings);
        for insertion in &insertions {
            leg_options.push(evaluator.leg_options_for_insertion_at_q(
                req.markings,
                insertion,
                req.degree,
            )?);
        }
        tasks.push(MasterContractionTask {
            ordinal: tasks.len(),
            degree: req.degree,
            insertions,
            markings: req.markings,
            leg_options,
        });
        task_indices.push(index);
        Ok(())
    })?;
    if tasks.is_empty() {
        return Ok(ResolventResult {
            value: ResolventPolynomial::zero(),
            candidate_terms,
            nonzero_terms: 0,
            engine,
            notes: vec![note.into()],
        });
    }

    let coefficients =
        evaluator.contract_restricted_tasks(req.markings, req.degree, &tasks, false)?;
    let mut value = ResolventPolynomial::zero();
    let mut nonzero_terms = 0usize;
    for coefficient in coefficients {
        let index = task_indices.get(coefficient.ordinal).ok_or_else(|| {
            GwError::AlgebraFailure(format!(
                "packed resolvent task ordinal {} is out of range",
                coefficient.ordinal
            ))
        })?;
        let normalized = normalize(coefficient.coefficient.value)?;
        if normalized.is_zero() {
            continue;
        }
        value.add_index_coefficient(index, normalized);
        nonzero_terms += 1;
    }

    Ok(ResolventResult {
        value,
        candidate_terms,
        nonzero_terms,
        engine,
        notes: vec![note.into()],
    })
}

pub fn compute_packed_resolvent_with_coeff_provider<C, P, N>(
    req: &ResolventRequest,
    provider: P,
    engine: &'static str,
    note: impl Into<String>,
    mut normalize: N,
) -> Result<ResolventResult<C>, GwError>
where
    C: Coeff + Send + Sync,
    P: CoefficientSemisimpleCohftProvider<C, Insertion = Insertion>,
    N: FnMut(C) -> Result<C, GwError>,
{
    validate_packed_resolvent_virtual_dimension(
        req,
        provider.coeff_virtual_dimension(req.genus, req.degree, req.markings),
    )?;

    if !provider.coeff_degree_is_effective(req.degree) {
        return Ok(ResolventResult {
            value: ResolventPolynomial::zero(),
            candidate_terms: 0,
            nonzero_terms: 0,
            engine: "packed-resolvent-ineffective-degree",
            notes: vec![format!(
                "the target has no effective curve class of degree {}",
                req.degree
            )],
        });
    }

    if req.virtual_dimension < 0 {
        return Ok(ResolventResult {
            value: ResolventPolynomial::zero(),
            candidate_terms: 0,
            nonzero_terms: 0,
            engine: "packed-resolvent-empty-dimension",
            notes: vec![format!(
                "virtual dimension {} is negative, so the packed resolvent generating function is zero",
                req.virtual_dimension
            )],
        });
    }

    if req.markings == 0 {
        let mut value = ResolventPolynomial::zero();
        let mut nonzero_terms = 0usize;
        let candidate_terms = usize::from(req.virtual_dimension == 0);
        let provider_dimension_matches = provider
            .coeff_virtual_dimension(req.genus, req.degree, 0)
            .is_none_or(|virtual_dimension| usize::try_from(virtual_dimension).ok() == Some(0));
        if req.virtual_dimension == 0 && provider_dimension_matches {
            let coefficient = compute_semisimple_graph_value_with_coeff::<C, _>(
                &provider,
                req.genus,
                req.degree,
                &[],
                None,
            )?;
            let normalized = normalize(coefficient)?;
            if !normalized.is_structurally_zero() {
                value.add_index_coefficient(&ResolventIndex::empty(), normalized);
                nonzero_terms = 1;
            }
        }
        return Ok(ResolventResult {
            value,
            candidate_terms,
            nonzero_terms,
            engine,
            notes: vec![note.into()],
        });
    }
    if !is_stable_cohft_range(req.genus, req.markings) {
        return Err(GwError::UnsupportedInvariant(
            "packed resolvent graph expansion is implemented for stable (g,n) CohFT ranges only"
                .to_string(),
        ));
    }

    let graph_dimension = checked_stable_graph_work_dimension(req.genus, req.markings)?;
    let needed_r_order = graph_dimension.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("stable-graph R-order overflow".to_string())
    })?;
    let needed_s_order = req.virtual_dimension as usize;
    let graph_kernel = provider.coeff_graph_kernel(req.degree, needed_r_order, graph_dimension)?;
    let descendant_s = provider.coeff_descendant_s_matrix(req.degree, needed_s_order)?;
    let mut tasks = Vec::<MasterContractionTask<C>>::new();
    let mut task_indices = Vec::<ResolventIndex>::new();
    let candidate_terms = enumerate_resolvent_indices(req, |index| {
        let insertions = index.to_insertions(req.target_n);
        if let (Some(total_degree), Some(virtual_dimension)) = (
            provider.coeff_insertion_degree(&insertions),
            provider.coeff_virtual_dimension(req.genus, req.degree, req.markings),
        ) {
            if usize::try_from(virtual_dimension).ok() != Some(total_degree) {
                return Ok(());
            }
        }

        let mut leg_options = Vec::with_capacity(req.markings);
        for insertion in &insertions {
            let insertion_terms = ancestor_insertion_terms_from_provider(
                &provider,
                std::slice::from_ref(insertion),
                &descendant_s,
                &graph_kernel.calibration.psi_inverse,
                req.degree,
                graph_dimension,
            )?;
            let mut options = leg_options_by_marking_color(
                &insertion_terms,
                &graph_kernel.inverse_r,
                req.degree,
                graph_dimension,
                provider.coeff_colors(),
            );
            leg_options.push(options.pop().unwrap_or_else(|| {
                vec![Vec::<LegFactorOption<C>>::new(); provider.coeff_colors()]
            }));
        }

        tasks.push(MasterContractionTask {
            ordinal: tasks.len(),
            degree: req.degree,
            insertions,
            markings: req.markings,
            leg_options,
        });
        task_indices.push(index);
        Ok(())
    })?;

    if tasks.is_empty() {
        return Ok(ResolventResult {
            value: ResolventPolynomial::zero(),
            candidate_terms,
            nonzero_terms: 0,
            engine,
            notes: vec![note.into()],
        });
    }

    let template = RestrictedExternalLegKernel::from_tasks(
        req.markings,
        provider.coeff_colors(),
        graph_dimension,
        req.degree,
        &tasks,
    );
    let mut profile = GraphEvalProfile::new();
    let graphs = profiled_prepared_stable_graphs(
        req.genus,
        req.markings,
        provider.coeff_colors(),
        &mut profile,
    )?;
    profile.stable_graphs = graphs.len();
    profile.edge_options = graph_kernel
        .edge_options
        .iter()
        .flat_map(|row| row.iter())
        .map(Vec::len)
        .sum();
    let graph_started = Instant::now();
    let restricted_kernel = evaluate_restricted_external_graphs_parallel(
        graphs.as_ref(),
        &template,
        &graph_kernel,
        req.degree,
        graph_dimension,
        &mut profile,
    );
    profile.add_graph_elapsed(graph_started.elapsed());
    profile.finish();

    let mut value = ResolventPolynomial::zero();
    let mut nonzero_terms = 0usize;
    for task in &tasks {
        let index = task_indices.get(task.ordinal).ok_or_else(|| {
            GwError::AlgebraFailure(format!(
                "packed resolvent task ordinal {} is out of range",
                task.ordinal
            ))
        })?;
        let coefficient = contract_restricted_external_leg_kernel_coeff_generic(
            &restricted_kernel,
            &task.leg_options,
            task.degree,
        );
        let normalized = normalize(coefficient)?;
        if normalized.is_structurally_zero() {
            continue;
        }
        value.add_index_coefficient(index, normalized);
        nonzero_terms += 1;
    }

    Ok(ResolventResult {
        value,
        candidate_terms,
        nonzero_terms,
        engine,
        notes: vec![note.into()],
    })
}

fn validate_packed_resolvent_virtual_dimension(
    req: &ResolventRequest,
    provider_virtual_dimension: Option<isize>,
) -> Result<(), GwError> {
    if let Some(actual) = provider_virtual_dimension {
        if req.virtual_dimension != actual {
            return Err(GwError::ConventionMismatch(format!(
                "packed resolvent virtual dimension {} does not match provider virtual dimension {actual} for (g,d,m)=({},{},{})",
                req.virtual_dimension, req.genus, req.degree, req.markings
            )));
        }
    }
    Ok(())
}

pub fn compute_projective_resolvent_packed(
    req: &ResolventRequest,
    equivariant: bool,
) -> Result<ResolventResult, GwError> {
    let provider = ProjectiveSpaceProvider::new(req.target_n, equivariant);
    let engine = if equivariant {
        "givental-packed-resolvent"
    } else {
        "givental-packed-resolvent-lambda-line"
    };
    compute_packed_resolvent_with_provider(
        req,
        provider,
        engine,
        "computed by packed S/R external-leg graph kernel; all resolvent coefficients share one stable-graph contraction",
        Ok::<RatFun, GwError>,
    )
}

pub(crate) fn compute_no_marking_packed_resolvent<P, N>(
    req: &ResolventRequest,
    provider: &P,
    engine: &'static str,
    note: impl Into<String>,
    mut normalize: N,
) -> Result<ResolventResult, GwError>
where
    P: SemisimpleCohftProvider<Insertion = Insertion>,
    N: FnMut(RatFun) -> Result<RatFun, GwError>,
{
    let mut value = ResolventPolynomial::zero();
    let mut nonzero_terms = 0usize;
    let candidate_terms = usize::from(req.virtual_dimension == 0);
    let provider_dimension_matches = provider
        .virtual_dimension(req.genus, req.degree, 0)
        .is_none_or(|virtual_dimension| usize::try_from(virtual_dimension).ok() == Some(0));
    if req.virtual_dimension == 0 && provider_dimension_matches {
        let coefficient =
            compute_semisimple_graph_value(provider, req.genus, req.degree, &[], None)?;
        let normalized = normalize(coefficient)?;
        if !normalized.is_zero() {
            value.add_index_coefficient(&ResolventIndex::empty(), normalized);
            nonzero_terms = 1;
        }
    }

    Ok(ResolventResult {
        value,
        candidate_terms,
        nonzero_terms,
        engine,
        notes: vec![note.into()],
    })
}

pub(crate) fn shared_kernel_task_counts(
    req: &SeriesRequest,
    provider: &impl SemisimpleCohftProvider<Insertion = Insertion>,
    candidates_by_degree: &[Vec<Vec<Insertion>>],
) -> Vec<usize> {
    let mut counts = vec![0usize; req.max_markings + 1];
    for (degree, candidates) in candidates_by_degree.iter().enumerate() {
        for insertions in candidates {
            if dimension_mismatch(provider, req.genus, degree, insertions).is_some() {
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
pub(crate) struct GiventalMasterEvaluator<P>
where
    P: SemisimpleCohftProvider<Insertion = Insertion>,
{
    provider: P,
    genus: usize,
    degree_max: usize,
    max_descendant_power: usize,
    truncation: Option<Truncation>,
    descendant_s_cache: HashMap<(usize, usize), SeriesSMatrix>,
    external_kernel_cache: HashMap<usize, ExternalLegKernel>,
    leg_options_cache: HashMap<MasterLegOptionsKey, Vec<Vec<LegFactorOption>>>,
    shared_kernel_task_counts: Option<Vec<usize>>,
}

#[cfg(test)]
impl GiventalMasterEvaluator<ProjectiveSpaceProvider> {
    fn new(req: &SeriesRequest) -> Self {
        Self::with_provider(req, ProjectiveSpaceProvider::new(req.n, req.equivariant))
    }
}

impl<P> GiventalMasterEvaluator<P>
where
    P: SemisimpleCohftProvider<Insertion = Insertion>,
{
    fn with_provider(req: &SeriesRequest, provider: P) -> Self {
        Self::for_problem(
            provider,
            req.genus,
            req.degree_max,
            req.max_descendant_power,
            req.truncation.clone(),
        )
    }

    fn for_problem(
        provider: P,
        genus: usize,
        degree_max: usize,
        max_descendant_power: usize,
        truncation: Option<Truncation>,
    ) -> Self {
        Self {
            provider,
            genus,
            degree_max,
            max_descendant_power,
            truncation,
            descendant_s_cache: HashMap::new(),
            external_kernel_cache: HashMap::new(),
            leg_options_cache: HashMap::new(),
            shared_kernel_task_counts: None,
        }
    }

    fn set_shared_kernel_task_counts(&mut self, counts: Vec<usize>) {
        self.shared_kernel_task_counts = Some(counts);
    }

    fn colors(&self) -> usize {
        self.provider.colors()
    }

    fn is_dimension_mismatch(&self, degree: usize, insertions: &[Insertion]) -> bool {
        dimension_mismatch(&self.provider, self.genus, degree, insertions).is_some()
    }

    fn compute_coefficient(
        &mut self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<RatFun, GwError> {
        if self.is_dimension_mismatch(degree, insertions) {
            return Ok(RatFun::zero());
        }

        let markings = insertions.len();
        if !self.can_use_external_leg_kernel(insertions) {
            return match compute_semisimple_graph_value(
                &self.provider,
                self.genus,
                degree,
                insertions,
                self.truncation.as_ref(),
            ) {
                Ok(value) => Ok(value),
                Err(GwError::UnsupportedInvariant(msg)) => {
                    if let Some(value) = self.provider.scalar_fallback_value(
                        self.genus,
                        degree,
                        insertions,
                        self.truncation.as_ref(),
                    )? {
                        Ok(value)
                    } else {
                        Err(GwError::UnsupportedInvariant(msg))
                    }
                }
                Err(err) => Err(err),
            };
        }

        self.validate_truncation(markings, insertions)?;
        let mut leg_options = Vec::with_capacity(markings);
        for insertion in insertions {
            leg_options.push(self.leg_options_for_insertion(markings, insertion)?);
        }
        let kernel = self.external_leg_kernel(markings)?;
        Ok(contract_external_leg_kernel_coeff(
            kernel,
            &leg_options,
            degree,
        ))
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
            leg_options.push(self.leg_options_for_insertion_at_q(markings, insertion, degree)?);
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
        q_degree: usize,
        tasks: &[MasterContractionTask],
        include_zero: bool,
    ) -> Result<Vec<OrderedSeriesCoefficient>, GwError> {
        let kernel = self.build_restricted_external_leg_kernel(markings, q_degree, tasks)?;
        Ok(contract_restricted_tasks_parallel(
            &kernel,
            tasks,
            include_zero,
        ))
    }

    fn build_restricted_external_leg_kernel(
        &self,
        markings: usize,
        q_degree: usize,
        tasks: &[MasterContractionTask],
    ) -> Result<RestrictedExternalLegKernel, GwError> {
        let graph_dimension = self.graph_dimension(markings)?;
        let graph_kernel = self.graph_kernel_for_markings_at_q(markings, q_degree)?;
        let template = RestrictedExternalLegKernel::from_tasks(
            markings,
            self.colors(),
            graph_dimension,
            q_degree,
            tasks,
        );
        let mut profile = GraphEvalProfile::new();
        let graphs =
            profiled_prepared_stable_graphs(self.genus, markings, self.colors(), &mut profile)?;
        profile.stable_graphs = graphs.len();
        profile.edge_options = graph_kernel
            .edge_options
            .iter()
            .flat_map(|row| row.iter())
            .map(Vec::len)
            .sum();

        let graphs_started = Instant::now();
        let total = evaluate_restricted_external_graphs_auto(
            graphs.as_ref(),
            &template,
            &graph_kernel,
            q_degree,
            graph_dimension,
            &mut profile,
        );
        profile.add_graph_elapsed(graphs_started.elapsed());
        profile.finish();
        Ok(total)
    }

    fn validate_truncation(
        &self,
        markings: usize,
        insertions: &[Insertion],
    ) -> Result<(), GwError> {
        let graph_dimension = self.graph_dimension(markings)?;
        let needed_r_order = graph_dimension.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("stable-graph R-order overflow".to_string())
        })?;
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

    fn graph_dimension(&self, markings: usize) -> Result<usize, GwError> {
        checked_stable_graph_work_dimension(self.genus, markings)
    }

    fn descendant_s(&mut self, q_degree: usize, z_order: usize) -> Result<&SeriesSMatrix, GwError> {
        let key = (q_degree, z_order);
        if !self.descendant_s_cache.contains_key(&key) {
            let descendant_s = self.provider.descendant_s_matrix(q_degree, z_order)?;
            self.descendant_s_cache.insert(key, descendant_s);
        }
        Ok(self
            .descendant_s_cache
            .get(&key)
            .expect("descendant S-matrix cache populated before access"))
    }

    fn graph_kernel_for_markings(
        &self,
        markings: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        self.graph_kernel_for_markings_at_q(markings, self.degree_max)
    }

    fn graph_kernel_for_markings_at_q(
        &self,
        markings: usize,
        q_degree: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        let graph_dimension = self.graph_dimension(markings)?;
        let needed_r_order = graph_dimension.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("stable-graph R-order overflow".to_string())
        })?;
        self.provider
            .graph_kernel(q_degree, needed_r_order, graph_dimension)
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
        let graph_dimension = self.graph_dimension(markings)?;
        let graph_kernel = self.graph_kernel_for_markings(markings)?;
        let mut profile = GraphEvalProfile::new();
        let graphs =
            profiled_prepared_stable_graphs(self.genus, markings, self.colors(), &mut profile)?;
        profile.stable_graphs = graphs.len();
        profile.edge_options = graph_kernel
            .edge_options
            .iter()
            .flat_map(|row| row.iter())
            .map(Vec::len)
            .sum();

        let graphs_started = Instant::now();
        let total = evaluate_external_graphs_auto(
            graphs.as_ref(),
            markings,
            self.colors(),
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
        self.leg_options_for_insertion_at_q(markings, insertion, self.degree_max)
    }

    fn leg_options_for_insertion_at_q(
        &mut self,
        markings: usize,
        insertion: &Insertion,
        q_degree: usize,
    ) -> Result<Vec<Vec<LegFactorOption>>, GwError> {
        let cache_key = insertion
            .class
            .pure_power()
            .map(|class_power| MasterLegOptionsKey {
                q_degree,
                markings,
                descendant_power: insertion.descendant_power,
                class_power,
            });
        if let Some(key) = cache_key {
            if let Some(cached) = self.leg_options_cache.get(&key) {
                return Ok(cached.clone());
            }
        }

        let graph_dimension = self.graph_dimension(markings)?;
        let s_order = self.max_descendant_power.max(insertion.descendant_power);
        let graph_kernel = self.graph_kernel_for_markings_at_q(markings, q_degree)?;
        let colors = self.colors();
        let psi_inverse = graph_kernel.calibration.psi_inverse.clone();
        let inverse_r = graph_kernel.inverse_r.clone();
        let insertion_terms = {
            let descendant_s = self.descendant_s(q_degree, s_order)?.clone();
            ancestor_insertion_terms_from_provider(
                &self.provider,
                std::slice::from_ref(insertion),
                &descendant_s,
                &psi_inverse,
                q_degree,
                graph_dimension,
            )?
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
pub(crate) struct MasterContractionTask<C = RatFun> {
    ordinal: usize,
    degree: usize,
    insertions: Vec<Insertion>,
    markings: usize,
    leg_options: Vec<Vec<Vec<LegFactorOption<C>>>>,
}

#[derive(Debug)]
pub(crate) struct OrderedSeriesCoefficient {
    ordinal: usize,
    coefficient: SeriesCoefficient,
}

pub(crate) fn master_worker_count(work_items: usize) -> usize {
    if work_items <= 1 {
        return 1;
    }
    let available = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let requested = std::env::var("GW_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(available);
    requested.min(work_items).max(1)
}

pub(crate) fn contract_task_chunk(
    kernels: &HashMap<usize, ExternalLegKernel>,
    tasks: &[MasterContractionTask],
    include_zero: bool,
) -> Vec<OrderedSeriesCoefficient> {
    let mut out = Vec::new();
    for task in tasks {
        let kernel = kernels
            .get(&task.markings)
            .expect("prepared master task must have a cached external-leg kernel");
        let value = contract_external_leg_kernel_coeff(kernel, &task.leg_options, task.degree);
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

pub(crate) fn contract_restricted_tasks_parallel(
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

pub(crate) fn contract_restricted_task_chunk(
    kernel: &RestrictedExternalLegKernel,
    tasks: &[MasterContractionTask],
    include_zero: bool,
) -> Vec<OrderedSeriesCoefficient> {
    let mut out = Vec::new();
    for task in tasks {
        let value =
            contract_restricted_external_leg_kernel_coeff(kernel, &task.leg_options, task.degree);
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

pub(crate) struct ScalarGraphChunkResult<C = RatFun> {
    total: QSeries<C>,
    profile: GraphEvalProfile,
    vertex_cache: HashMap<VertexContributionKey, Arc<QSeries<C>>>,
}

pub(crate) fn evaluate_scalar_graphs_parallel<C>(
    graphs: &[PreparedStableGraph],
    leg_options: &[Vec<Vec<LegFactorOption<C>>>],
    kernel: &Arc<GiventalGraphKernel<C>>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> QSeries<C>
where
    C: Coeff + Send + Sync,
{
    let worker_count = master_worker_count(graphs.len());
    let initial_vertex_cache = kernel.vertex_cache.lock().unwrap().clone();
    let results = if worker_count <= 1 {
        vec![evaluate_scalar_graph_chunk(
            graphs,
            leg_options,
            kernel,
            q_degree,
            graph_dimension,
            initial_vertex_cache,
        )]
    } else {
        let next_graph = AtomicUsize::new(0);
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..worker_count {
                let next_graph = &next_graph;
                let local_vertex_cache = initial_vertex_cache.clone();
                handles.push(scope.spawn(move || {
                    let mut total = QSeries::<C>::zero(q_degree);
                    let mut profile = GraphEvalProfile::new();
                    let mut vertex_cache = local_vertex_cache;
                    loop {
                        let graph_index = next_graph.fetch_add(1, Ordering::Relaxed);
                        if graph_index >= graphs.len() {
                            break;
                        }
                        let result = evaluate_scalar_graph_chunk(
                            &graphs[graph_index..graph_index + 1],
                            leg_options,
                            kernel,
                            q_degree,
                            graph_dimension,
                            vertex_cache,
                        );
                        total.add_assign(&result.total);
                        profile.absorb_graph_counts(&result.profile);
                        vertex_cache = result.vertex_cache;
                    }
                    ScalarGraphChunkResult {
                        total,
                        profile,
                        vertex_cache,
                    }
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

    let mut total = QSeries::<C>::zero(q_degree);
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

pub(crate) fn evaluate_scalar_graph_chunk<C>(
    graphs: &[PreparedStableGraph],
    leg_options: &[Vec<Vec<LegFactorOption<C>>>],
    kernel: &GiventalGraphKernel<C>,
    q_degree: usize,
    graph_dimension: usize,
    mut vertex_cache: HashMap<VertexContributionKey, Arc<QSeries<C>>>,
) -> ScalarGraphChunkResult<C>
where
    C: Coeff,
{
    let mut profile = GraphEvalProfile::new();
    let mut total = QSeries::<C>::zero(q_degree);
    for prepared in graphs {
        for coloring_index in 0..prepared.colorings.len() {
            accumulate_scalar_coloring(
                prepared,
                coloring_index,
                leg_options,
                kernel,
                q_degree,
                graph_dimension,
                &mut vertex_cache,
                &mut total,
                &mut profile,
            );
        }
    }
    ScalarGraphChunkResult {
        total,
        profile,
        vertex_cache,
    }
}

fn accumulate_scalar_coloring<C>(
    prepared: &PreparedStableGraph,
    coloring_index: usize,
    leg_options: &[Vec<Vec<LegFactorOption<C>>>],
    kernel: &GiventalGraphKernel<C>,
    q_degree: usize,
    graph_dimension: usize,
    vertex_cache: &mut HashMap<VertexContributionKey, Arc<QSeries<C>>>,
    total: &mut QSeries<C>,
    profile: &mut GraphEvalProfile,
) where
    C: Coeff,
{
    let oracle = WittenKontsevich::shared();
    let graph = &prepared.graph;
    let coloring = &prepared.colorings[coloring_index];
    profile.colorings += 1;
    let mut graph_total = QSeries::<C>::zero(q_degree);
    let mut base_powers = vec![Vec::<usize>::new(); graph.vertices.len()];
    let mut vertex_power_sums = vec![0usize; graph.vertices.len()];
    let coloring_factor = C::from_rational(coloring.factor.clone());
    accumulate_graph_factors(
        graph,
        &coloring.colors,
        leg_options,
        &kernel.edge_options,
        &kernel.calibration,
        &kernel.translation,
        oracle,
        vertex_cache,
        q_degree,
        graph_dimension,
        0,
        0,
        QSeries::<C>::one(q_degree),
        &mut base_powers,
        &mut vertex_power_sums,
        &prepared.vertex_power_caps,
        &mut graph_total,
        profile,
    );

    total.add_assign(&graph_total.scale(&coloring_factor));
}

pub(crate) fn qseries_slice_to_rational(series: &[QSeries]) -> Option<Vec<RationalQSeries>> {
    series.iter().map(qseries_to_rational).collect()
}

pub(crate) fn qseries_to_rational(series: &QSeries) -> Option<RationalQSeries> {
    Some(RationalQSeries::from_coeffs(
        series
            .coeffs()
            .iter()
            .map(RatFun::as_rational)
            .collect::<Option<Vec<_>>>()?,
    ))
}

pub(crate) fn rational_qseries_to_ratfun(series: &RationalQSeries) -> QSeries {
    QSeries::from_coeffs(
        series
            .coeffs()
            .iter()
            .cloned()
            .map(RatFun::from_rational)
            .collect(),
    )
}

pub(crate) fn series_matrix_to_rational(matrix: &SeriesMatrix) -> Option<SeriesMatrix<Rational>> {
    Some(SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| {
                row.iter()
                    .map(qseries_to_rational)
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?,
    ))
}

pub(crate) fn series_r_matrix_to_rational(
    r_matrix: &SeriesRMatrix,
) -> Option<SeriesRMatrix<Rational>> {
    Some(SeriesRMatrix {
        size: r_matrix.size,
        q_degree: r_matrix.q_degree,
        z_order: r_matrix.z_order,
        coefficients: r_matrix
            .coefficients
            .iter()
            .map(series_matrix_to_rational)
            .collect::<Option<Vec<_>>>()?,
        calibration: r_matrix.calibration.clone(),
        convention: r_matrix.convention,
    })
}

/// Rational twin of a symbolic graph kernel, available exactly when every
/// coefficient is a constant (the early lambda-line-specialized case).
pub(crate) fn kernel_to_rational(
    kernel: &GiventalGraphKernel,
) -> Option<GiventalGraphKernel<Rational>> {
    let calibration = &kernel.calibration;
    Some(GiventalGraphKernel {
        calibration: SemisimpleCalibration {
            r_matrix: series_r_matrix_to_rational(&calibration.r_matrix)?,
            metric: series_matrix_to_rational(&calibration.metric)?,
            psi: series_matrix_to_rational(&calibration.psi)?,
            psi_inverse: series_matrix_to_rational(&calibration.psi_inverse)?,
            connection: series_matrix_to_rational(&calibration.connection)?,
            delta: qseries_slice_to_rational(&calibration.delta)?,
            inverse_delta: qseries_slice_to_rational(&calibration.inverse_delta)?,
            relative_sqrt_delta: qseries_slice_to_rational(&calibration.relative_sqrt_delta)?,
            relative_sqrt_delta_inverse: qseries_slice_to_rational(
                &calibration.relative_sqrt_delta_inverse,
            )?,
        },
        inverse_r: kernel
            .inverse_r
            .iter()
            .map(series_matrix_to_rational)
            .collect::<Option<Vec<_>>>()?,
        translation: kernel
            .translation
            .iter()
            .map(|row| qseries_slice_to_rational(row))
            .collect::<Option<Vec<_>>>()?,
        edge_options: kernel
            .edge_options
            .iter()
            .map(|row| {
                row.iter()
                    .map(|options| {
                        options
                            .iter()
                            .map(|option| {
                                Some(EdgeFactorOption {
                                    left_power: option.left_power,
                                    right_power: option.right_power,
                                    coefficient: qseries_to_rational(&option.coefficient)?,
                                })
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?,
        vertex_cache: Mutex::new(HashMap::new()),
    })
}

pub(crate) fn leg_options_to_rational(
    leg_options: &[Vec<Vec<LegFactorOption>>],
) -> Option<Vec<Vec<Vec<LegFactorOption<Rational>>>>> {
    leg_options
        .iter()
        .map(|by_color| {
            by_color
                .iter()
                .map(|options| {
                    options
                        .iter()
                        .map(|option| {
                            Some(LegFactorOption {
                                power: option.power,
                                coefficient: qseries_to_rational(&option.coefficient)?,
                            })
                        })
                        .collect::<Option<Vec<_>>>()
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect()
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

pub(crate) fn qseries_slice_to_factored(series: &[QSeries]) -> Vec<QSeries<FactoredRatFun>> {
    series.iter().map(qseries_to_factored).collect()
}

pub(crate) fn series_matrix_to_factored(matrix: &SeriesMatrix) -> SeriesMatrix<FactoredRatFun> {
    SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| row.iter().map(qseries_to_factored).collect())
            .collect(),
    )
}

pub(crate) fn series_r_matrix_to_factored(
    r_matrix: &SeriesRMatrix,
) -> SeriesRMatrix<FactoredRatFun> {
    SeriesRMatrix {
        size: r_matrix.size,
        q_degree: r_matrix.q_degree,
        z_order: r_matrix.z_order,
        coefficients: r_matrix
            .coefficients
            .iter()
            .map(series_matrix_to_factored)
            .collect(),
        calibration: r_matrix.calibration.clone(),
        convention: r_matrix.convention,
    }
}

pub(crate) fn calibration_to_factored(
    calibration: &SemisimpleCalibration,
) -> SemisimpleCalibration<FactoredRatFun> {
    SemisimpleCalibration {
        r_matrix: series_r_matrix_to_factored(&calibration.r_matrix),
        metric: series_matrix_to_factored(&calibration.metric),
        psi: series_matrix_to_factored(&calibration.psi),
        psi_inverse: series_matrix_to_factored(&calibration.psi_inverse),
        connection: series_matrix_to_factored(&calibration.connection),
        delta: qseries_slice_to_factored(&calibration.delta),
        inverse_delta: qseries_slice_to_factored(&calibration.inverse_delta),
        relative_sqrt_delta: qseries_slice_to_factored(&calibration.relative_sqrt_delta),
        relative_sqrt_delta_inverse: qseries_slice_to_factored(
            &calibration.relative_sqrt_delta_inverse,
        ),
    }
}

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

/// Factored-denominator twin of a symbolic graph kernel.
pub(crate) fn kernel_to_factored(
    kernel: &GiventalGraphKernel,
) -> GiventalGraphKernel<FactoredRatFun> {
    let calibration = &kernel.calibration;
    GiventalGraphKernel {
        calibration: calibration_to_factored(calibration),
        inverse_r: kernel
            .inverse_r
            .iter()
            .map(series_matrix_to_factored)
            .collect(),
        translation: kernel
            .translation
            .iter()
            .map(|row| qseries_slice_to_factored(row))
            .collect(),
        edge_options: kernel
            .edge_options
            .iter()
            .map(|row| {
                row.iter()
                    .map(|options| {
                        options
                            .iter()
                            .map(|option| EdgeFactorOption {
                                left_power: option.left_power,
                                right_power: option.right_power,
                                coefficient: qseries_to_factored(&option.coefficient),
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect(),
        vertex_cache: Mutex::new(HashMap::new()),
    }
}

pub(crate) fn leg_options_to_factored(
    leg_options: &[Vec<Vec<LegFactorOption>>],
) -> Vec<Vec<Vec<LegFactorOption<FactoredRatFun>>>> {
    leg_options
        .iter()
        .map(|by_color| {
            by_color
                .iter()
                .map(|options| {
                    options
                        .iter()
                        .map(|option| LegFactorOption {
                            power: option.power,
                            coefficient: qseries_to_factored(&option.coefficient),
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Runs the generic graph evaluator over factored rational functions and
/// expands the result once at the end.
pub(crate) fn evaluate_factored_graphs(
    graphs: &[PreparedStableGraph],
    leg_options: &[Vec<Vec<LegFactorOption>>],
    kernel: &Arc<GiventalGraphKernel>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> QSeries {
    let factored_kernel = Arc::new(kernel_to_factored(kernel));
    let factored_leg_options = leg_options_to_factored(leg_options);
    let total = evaluate_scalar_graphs_parallel(
        graphs,
        &factored_leg_options,
        &factored_kernel,
        q_degree,
        graph_dimension,
        profile,
    );
    QSeries::from_coeffs(
        total
            .coeffs()
            .iter()
            .map(FactoredRatFun::to_ratfun)
            .collect(),
    )
}

fn external_kernel_to_ratfun(kernel: ExternalLegKernel<Rational>) -> ExternalLegKernel {
    ExternalLegKernel {
        markings: kernel.markings,
        colors: kernel.colors,
        max_power: kernel.max_power,
        q_degree: kernel.q_degree,
        state_count: kernel.state_count,
        entries: kernel
            .entries
            .iter()
            .map(rational_qseries_to_ratfun)
            .collect(),
    }
}

fn restricted_template_to_rational(
    template: &RestrictedExternalLegKernel,
) -> RestrictedExternalLegKernel<Rational> {
    RestrictedExternalLegKernel {
        markings: template.markings,
        colors: template.colors,
        max_power: template.max_power,
        q_degree: template.q_degree,
        states_by_marking_color: template.states_by_marking_color.clone(),
        state_index_by_marking_color_power: template.state_index_by_marking_color_power.clone(),
        state_counts: template.state_counts.clone(),
        strides: template.strides.clone(),
        entries: vec![QSeries::<Rational>::zero(template.q_degree); template.entries.len()],
    }
}

fn restricted_kernel_to_ratfun(
    template: &RestrictedExternalLegKernel,
    kernel: RestrictedExternalLegKernel<Rational>,
) -> RestrictedExternalLegKernel {
    let mut out = template.zero_like();
    out.entries = kernel
        .entries
        .iter()
        .map(rational_qseries_to_ratfun)
        .collect();
    out
}

/// [`evaluate_external_graphs_parallel`], but over plain rationals whenever
/// the kernel is constant.
pub(crate) fn evaluate_external_graphs_auto(
    graphs: &[PreparedStableGraph],
    markings: usize,
    colors: usize,
    kernel: &Arc<GiventalGraphKernel>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> ExternalLegKernel {
    if !crate::env_flag("GWAI_DISABLE_RATIONAL_GRAPH") {
        if let Some(rational_kernel) = kernel_to_rational(kernel) {
            let total = evaluate_external_graphs_parallel(
                graphs,
                markings,
                colors,
                &Arc::new(rational_kernel),
                q_degree,
                graph_dimension,
                profile,
            );
            return external_kernel_to_ratfun(total);
        }
    }
    evaluate_external_graphs_parallel(
        graphs,
        markings,
        colors,
        kernel,
        q_degree,
        graph_dimension,
        profile,
    )
}

/// [`evaluate_restricted_external_graphs_parallel`], but over plain rationals
/// whenever the kernel is constant.
pub(crate) fn evaluate_restricted_external_graphs_auto(
    graphs: &[PreparedStableGraph],
    template: &RestrictedExternalLegKernel,
    kernel: &Arc<GiventalGraphKernel>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> RestrictedExternalLegKernel {
    if !crate::env_flag("GWAI_DISABLE_RATIONAL_GRAPH") {
        if let Some(rational_kernel) = kernel_to_rational(kernel) {
            let total = evaluate_restricted_external_graphs_parallel(
                graphs,
                &restricted_template_to_rational(template),
                &Arc::new(rational_kernel),
                q_degree,
                graph_dimension,
                profile,
            );
            return restricted_kernel_to_ratfun(template, total);
        }
    }
    evaluate_restricted_external_graphs_parallel(
        graphs,
        template,
        kernel,
        q_degree,
        graph_dimension,
        profile,
    )
}

/// Runs the generic graph evaluator over plain rational coefficients when the
/// kernel and all leg options are constant, converting the result back.
pub(crate) fn evaluate_rational_graphs_if_possible(
    graphs: &[PreparedStableGraph],
    leg_options: &[Vec<Vec<LegFactorOption>>],
    kernel: &Arc<GiventalGraphKernel>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> Option<QSeries> {
    let rational_kernel = Arc::new(kernel_to_rational(kernel)?);
    let rational_leg_options = leg_options_to_rational(leg_options)?;
    let total = evaluate_scalar_graphs_parallel(
        graphs,
        &rational_leg_options,
        &rational_kernel,
        q_degree,
        graph_dimension,
        profile,
    );
    Some(rational_qseries_to_ratfun(&total))
}

pub(crate) struct ExternalGraphChunkResult<C = RatFun> {
    total: ExternalLegKernel<C>,
    profile: GraphEvalProfile,
    vertex_cache: HashMap<VertexContributionKey, Arc<QSeries<C>>>,
}

pub(crate) fn evaluate_external_graphs_parallel<C>(
    graphs: &[PreparedStableGraph],
    markings: usize,
    colors: usize,
    kernel: &Arc<GiventalGraphKernel<C>>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> ExternalLegKernel<C>
where
    C: Coeff + Send + Sync,
{
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
        let next_graph = AtomicUsize::new(0);
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..worker_count {
                let next_graph = &next_graph;
                let local_vertex_cache = initial_vertex_cache.clone();
                handles.push(scope.spawn(move || {
                    let mut total =
                        ExternalLegKernel::<C>::zero(markings, colors, graph_dimension, q_degree);
                    let mut profile = GraphEvalProfile::new();
                    let mut vertex_cache = local_vertex_cache;
                    loop {
                        let graph_index = next_graph.fetch_add(1, Ordering::Relaxed);
                        if graph_index >= graphs.len() {
                            break;
                        }
                        let result = evaluate_external_graph_chunk(
                            &graphs[graph_index..graph_index + 1],
                            markings,
                            colors,
                            kernel,
                            q_degree,
                            graph_dimension,
                            vertex_cache,
                        );
                        total.add_assign(&result.total);
                        profile.absorb_graph_counts(&result.profile);
                        vertex_cache = result.vertex_cache;
                    }
                    ExternalGraphChunkResult {
                        total,
                        profile,
                        vertex_cache,
                    }
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

    let mut total = ExternalLegKernel::<C>::zero(markings, colors, graph_dimension, q_degree);
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

pub(crate) fn evaluate_external_graph_chunk<C>(
    graphs: &[PreparedStableGraph],
    markings: usize,
    colors: usize,
    kernel: &GiventalGraphKernel<C>,
    q_degree: usize,
    graph_dimension: usize,
    mut vertex_cache: HashMap<VertexContributionKey, Arc<QSeries<C>>>,
) -> ExternalGraphChunkResult<C>
where
    C: Coeff,
{
    let mut profile = GraphEvalProfile::new();
    let mut total = ExternalLegKernel::<C>::zero(markings, colors, graph_dimension, q_degree);
    for prepared in graphs {
        for coloring_index in 0..prepared.colorings.len() {
            accumulate_external_coloring(
                prepared,
                coloring_index,
                kernel,
                q_degree,
                graph_dimension,
                &mut vertex_cache,
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

fn accumulate_external_coloring<C>(
    prepared: &PreparedStableGraph,
    coloring_index: usize,
    kernel: &GiventalGraphKernel<C>,
    q_degree: usize,
    graph_dimension: usize,
    vertex_cache: &mut HashMap<VertexContributionKey, Arc<QSeries<C>>>,
    total: &mut ExternalLegKernel<C>,
    profile: &mut GraphEvalProfile,
) where
    C: Coeff,
{
    let oracle = WittenKontsevich::shared();
    let graph = &prepared.graph;
    let coloring = &prepared.colorings[coloring_index];
    profile.colorings += 1;
    let mut base_powers = vec![Vec::<usize>::new(); graph.vertices.len()];
    let mut vertex_power_sums = vec![0usize; graph.vertices.len()];
    let mut external_states = Vec::with_capacity(graph.legs.len());
    let coloring_factor = C::from_rational(coloring.factor.clone());
    accumulate_external_leg_graph_factors(
        graph,
        &coloring.colors,
        &kernel.edge_options,
        &kernel.calibration,
        &kernel.translation,
        oracle,
        vertex_cache,
        q_degree,
        graph_dimension,
        0,
        0,
        QSeries::<C>::one(q_degree).scale(&coloring_factor),
        &mut base_powers,
        &mut vertex_power_sums,
        &prepared.vertex_power_caps,
        &mut external_states,
        total,
        profile,
    );
}

pub(crate) struct RestrictedExternalGraphChunkResult<C = RatFun> {
    total: RestrictedExternalLegKernel<C>,
    profile: GraphEvalProfile,
    vertex_cache: HashMap<VertexContributionKey, Arc<QSeries<C>>>,
}

pub(crate) fn evaluate_restricted_external_graphs_parallel<C>(
    graphs: &[PreparedStableGraph],
    template: &RestrictedExternalLegKernel<C>,
    kernel: &Arc<GiventalGraphKernel<C>>,
    q_degree: usize,
    graph_dimension: usize,
    profile: &mut GraphEvalProfile,
) -> RestrictedExternalLegKernel<C>
where
    C: Coeff + Send + Sync,
{
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
        let next_graph = AtomicUsize::new(0);
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..worker_count {
                let next_graph = &next_graph;
                let local_vertex_cache = initial_vertex_cache.clone();
                handles.push(scope.spawn(move || {
                    let mut total = template.zero_like();
                    let mut profile = GraphEvalProfile::new();
                    let mut vertex_cache = local_vertex_cache;
                    loop {
                        let graph_index = next_graph.fetch_add(1, Ordering::Relaxed);
                        if graph_index >= graphs.len() {
                            break;
                        }
                        let result = evaluate_restricted_external_graph_chunk(
                            &graphs[graph_index..graph_index + 1],
                            template,
                            kernel,
                            q_degree,
                            graph_dimension,
                            vertex_cache,
                        );
                        total.add_assign(&result.total);
                        profile.absorb_graph_counts(&result.profile);
                        vertex_cache = result.vertex_cache;
                    }
                    RestrictedExternalGraphChunkResult {
                        total,
                        profile,
                        vertex_cache,
                    }
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

pub(crate) fn evaluate_restricted_external_graph_chunk<C>(
    graphs: &[PreparedStableGraph],
    template: &RestrictedExternalLegKernel<C>,
    kernel: &GiventalGraphKernel<C>,
    q_degree: usize,
    graph_dimension: usize,
    mut vertex_cache: HashMap<VertexContributionKey, Arc<QSeries<C>>>,
) -> RestrictedExternalGraphChunkResult<C>
where
    C: Coeff,
{
    let mut profile = GraphEvalProfile::new();
    let mut total = template.zero_like();
    for prepared in graphs {
        for coloring_index in 0..prepared.colorings.len() {
            accumulate_restricted_external_coloring(
                prepared,
                coloring_index,
                template,
                kernel,
                q_degree,
                graph_dimension,
                &mut vertex_cache,
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

fn accumulate_restricted_external_coloring<C>(
    prepared: &PreparedStableGraph,
    coloring_index: usize,
    template: &RestrictedExternalLegKernel<C>,
    kernel: &GiventalGraphKernel<C>,
    q_degree: usize,
    graph_dimension: usize,
    vertex_cache: &mut HashMap<VertexContributionKey, Arc<QSeries<C>>>,
    total: &mut RestrictedExternalLegKernel<C>,
    profile: &mut GraphEvalProfile,
) where
    C: Coeff,
{
    let oracle = WittenKontsevich::shared();
    let graph = &prepared.graph;
    let coloring = &prepared.colorings[coloring_index];
    profile.colorings += 1;
    let mut base_powers = vec![Vec::<usize>::new(); graph.vertices.len()];
    let mut vertex_power_sums = vec![0usize; graph.vertices.len()];
    let mut external_state_indices = Vec::with_capacity(template.markings);
    let coloring_factor = C::from_rational(coloring.factor.clone());
    accumulate_restricted_external_leg_graph_factors(
        graph,
        &coloring.colors,
        &kernel.edge_options,
        &kernel.calibration,
        &kernel.translation,
        oracle,
        vertex_cache,
        q_degree,
        graph_dimension,
        0,
        0,
        QSeries::<C>::one(q_degree).scale(&coloring_factor),
        &mut base_powers,
        &mut vertex_power_sums,
        &prepared.vertex_power_caps,
        &mut external_state_indices,
        &template.states_by_marking_color,
        total,
        profile,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MasterLegOptionsKey {
    q_degree: usize,
    markings: usize,
    descendant_power: usize,
    class_power: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExternalLegState {
    color: usize,
    power: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ExternalLegKernel<C = RatFun> {
    markings: usize,
    colors: usize,
    max_power: usize,
    q_degree: usize,
    state_count: usize,
    entries: Vec<QSeries<C>>,
}

impl<C: Coeff> ExternalLegKernel<C> {
    fn zero(markings: usize, colors: usize, max_power: usize, q_degree: usize) -> Self {
        let state_count = colors * (max_power + 1);
        let entries = vec![QSeries::<C>::zero(q_degree); state_count.pow(markings as u32)];
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

    fn add_term(&mut self, states: &[ExternalLegState], value: &QSeries<C>) {
        if value.is_structurally_zero() {
            return;
        }
        let index = self.tensor_index(states);
        self.entries[index].add_assign(value);
    }

    fn add_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(self.markings, rhs.markings);
        debug_assert_eq!(self.colors, rhs.colors);
        debug_assert_eq!(self.max_power, rhs.max_power);
        debug_assert_eq!(self.q_degree, rhs.q_degree);
        for (left, right) in self.entries.iter_mut().zip(rhs.entries.iter()) {
            if !right.is_structurally_zero() {
                left.add_assign(right);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestrictedLegState {
    power: usize,
    state_index: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct RestrictedExternalLegKernel<C = RatFun> {
    markings: usize,
    colors: usize,
    max_power: usize,
    q_degree: usize,
    states_by_marking_color: Vec<Vec<Vec<RestrictedLegState>>>,
    state_index_by_marking_color_power: Vec<Vec<Vec<Option<usize>>>>,
    state_counts: Vec<usize>,
    strides: Vec<usize>,
    entries: Vec<QSeries<C>>,
}

impl<C: Coeff> RestrictedExternalLegKernel<C> {
    fn from_tasks(
        markings: usize,
        colors: usize,
        max_power: usize,
        q_degree: usize,
        tasks: &[MasterContractionTask<C>],
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
        let entries = vec![QSeries::<C>::zero(q_degree); entries_len];

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
        out.entries = vec![QSeries::<C>::zero(self.q_degree); self.entries.len()];
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

    fn add_term(&mut self, state_indices: &[usize], value: &QSeries<C>) {
        if value.is_structurally_zero() || self.entries.is_empty() {
            return;
        }
        let index = self.tensor_index_from_state_indices(state_indices);
        self.entries[index].add_assign(value);
    }

    fn add_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(self.markings, rhs.markings);
        debug_assert_eq!(self.colors, rhs.colors);
        debug_assert_eq!(self.max_power, rhs.max_power);
        debug_assert_eq!(self.q_degree, rhs.q_degree);
        debug_assert_eq!(self.state_counts, rhs.state_counts);
        for (left, right) in self.entries.iter_mut().zip(rhs.entries.iter()) {
            if !right.is_structurally_zero() {
                left.add_assign(right);
            }
        }
    }
}

pub(crate) fn contract_external_leg_kernel_coeff(
    kernel: &ExternalLegKernel,
    leg_options: &[Vec<Vec<LegFactorOption>>],
    degree: usize,
) -> RatFun {
    contract_external_leg_kernel_coeff_generic(kernel, leg_options, degree)
}

pub(crate) fn contract_external_leg_kernel_coeff_generic<C: Coeff>(
    kernel: &ExternalLegKernel<C>,
    leg_options: &[Vec<Vec<LegFactorOption<C>>>],
    degree: usize,
) -> C {
    debug_assert_eq!(kernel.markings, leg_options.len());
    let mut total = C::zero();
    let mut state_indices = vec![0usize; kernel.markings];
    contract_external_leg_kernel_coeff_generic_rec(
        kernel,
        leg_options,
        degree,
        0,
        QSeries::<C>::one(kernel.q_degree),
        &mut state_indices,
        &mut total,
    );
    total
}

pub(crate) fn contract_external_leg_kernel_coeff_generic_rec<C: Coeff>(
    kernel: &ExternalLegKernel<C>,
    leg_options: &[Vec<Vec<LegFactorOption<C>>>],
    degree: usize,
    marking: usize,
    coefficient: QSeries<C>,
    state_indices: &mut [usize],
    total: &mut C,
) {
    if coefficient.is_structurally_zero() {
        return;
    }
    if marking == kernel.markings {
        let index = kernel.tensor_index_from_state_indices(state_indices);
        if kernel.entries[index].is_structurally_zero() {
            return;
        }
        *total = total.add(&qseries_mul_coeff_generic(
            &coefficient,
            &kernel.entries[index],
            degree,
        ));
        return;
    }

    for color in 0..kernel.colors {
        for option in &leg_options[marking][color] {
            if option.power > kernel.max_power {
                continue;
            }
            let next_coefficient = coefficient.mul(&option.coefficient);
            if next_coefficient.is_structurally_zero() {
                continue;
            }
            state_indices[marking] = kernel.state_index(color, option.power);
            contract_external_leg_kernel_coeff_generic_rec(
                kernel,
                leg_options,
                degree,
                marking + 1,
                next_coefficient,
                state_indices,
                total,
            );
        }
    }
}

pub(crate) fn contract_restricted_external_leg_kernel_coeff(
    kernel: &RestrictedExternalLegKernel,
    leg_options: &[Vec<Vec<LegFactorOption>>],
    degree: usize,
) -> RatFun {
    contract_restricted_external_leg_kernel_coeff_generic(kernel, leg_options, degree)
}

pub(crate) fn contract_restricted_external_leg_kernel_coeff_generic<C: Coeff>(
    kernel: &RestrictedExternalLegKernel<C>,
    leg_options: &[Vec<Vec<LegFactorOption<C>>>],
    degree: usize,
) -> C {
    debug_assert_eq!(kernel.markings, leg_options.len());
    if kernel.entries.is_empty() {
        return C::zero();
    }
    let mut total = C::zero();
    let mut state_indices = vec![0usize; kernel.markings];
    contract_restricted_external_leg_kernel_coeff_generic_rec(
        kernel,
        leg_options,
        degree,
        0,
        QSeries::<C>::one(kernel.q_degree),
        &mut state_indices,
        &mut total,
    );
    total
}

pub(crate) fn contract_restricted_external_leg_kernel_coeff_generic_rec<C: Coeff>(
    kernel: &RestrictedExternalLegKernel<C>,
    leg_options: &[Vec<Vec<LegFactorOption<C>>>],
    degree: usize,
    marking: usize,
    coefficient: QSeries<C>,
    state_indices: &mut [usize],
    total: &mut C,
) {
    if coefficient.is_structurally_zero() {
        return;
    }
    if marking == kernel.markings {
        let index = kernel.tensor_index_from_state_indices(state_indices);
        if kernel.entries[index].is_structurally_zero() {
            return;
        }
        *total = total.add(&qseries_mul_coeff_generic(
            &coefficient,
            &kernel.entries[index],
            degree,
        ));
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
            if next_coefficient.is_structurally_zero() {
                continue;
            }
            state_indices[marking] = state_index;
            contract_restricted_external_leg_kernel_coeff_generic_rec(
                kernel,
                leg_options,
                degree,
                marking + 1,
                next_coefficient,
                state_indices,
                total,
            );
        }
    }
}

pub(crate) fn qseries_mul_coeff_generic<C: Coeff>(
    left: &QSeries<C>,
    right: &QSeries<C>,
    degree: usize,
) -> C {
    let max_left = left.max_degree().min(degree);
    let mut total = C::zero();
    for left_degree in 0..=max_left {
        let right_degree = degree - left_degree;
        if right_degree > right.max_degree() {
            continue;
        }
        let left_coeff = left
            .coeff(left_degree)
            .expect("left q-series degree is bounded");
        if left_coeff.is_structurally_zero() {
            continue;
        }
        let right_coeff = right
            .coeff(right_degree)
            .expect("right q-series degree is bounded");
        if right_coeff.is_structurally_zero() {
            continue;
        }
        crate::fused::add_product_assign(&mut total, left_coeff, right_coeff);
    }
    total
}

pub(crate) fn accumulate_external_leg_graph_factors<C>(
    graph: &crate::graphs::StableGraph,
    colors: &[usize],
    edge_options: &[Vec<Vec<EdgeFactorOption<C>>>],
    calibration: &SemisimpleCalibration<C>,
    translation: &[Vec<QSeries<C>>],
    oracle: &WittenKontsevich,
    vertex_cache: &mut HashMap<VertexContributionKey, Arc<QSeries<C>>>,
    q_degree: usize,
    max_power: usize,
    factor_index: usize,
    current_power_sum: usize,
    coefficient: QSeries<C>,
    base_powers: &mut [Vec<usize>],
    vertex_power_sums: &mut [usize],
    vertex_power_caps: &[usize],
    external_states: &mut Vec<ExternalLegState>,
    total: &mut ExternalLegKernel<C>,
    profile: &mut GraphEvalProfile,
) where
    C: Coeff,
{
    if profile.enabled {
        profile.recursion_calls += 1;
    }
    if coefficient.is_structurally_zero() || current_power_sum > max_power {
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
            if next_coefficient.is_structurally_zero() {
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

    let mut vertex_product = QSeries::<C>::one(q_degree);
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
        if vertex_product.is_structurally_zero() {
            return;
        }
    }
    if profile.enabled {
        profile.leaves += 1;
    }
    total.add_term(external_states, &coefficient.mul(&vertex_product));
}

pub(crate) fn accumulate_restricted_external_leg_graph_factors<C>(
    graph: &crate::graphs::StableGraph,
    colors: &[usize],
    edge_options: &[Vec<Vec<EdgeFactorOption<C>>>],
    calibration: &SemisimpleCalibration<C>,
    translation: &[Vec<QSeries<C>>],
    oracle: &WittenKontsevich,
    vertex_cache: &mut HashMap<VertexContributionKey, Arc<QSeries<C>>>,
    q_degree: usize,
    max_power: usize,
    factor_index: usize,
    current_power_sum: usize,
    coefficient: QSeries<C>,
    base_powers: &mut [Vec<usize>],
    vertex_power_sums: &mut [usize],
    vertex_power_caps: &[usize],
    external_state_indices: &mut Vec<usize>,
    states_by_marking_color: &[Vec<Vec<RestrictedLegState>>],
    total: &mut RestrictedExternalLegKernel<C>,
    profile: &mut GraphEvalProfile,
) where
    C: Coeff,
{
    if profile.enabled {
        profile.recursion_calls += 1;
    }
    if coefficient.is_structurally_zero() || current_power_sum > max_power {
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
            if next_coefficient.is_structurally_zero() {
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

    let mut vertex_product = QSeries::<C>::one(q_degree);
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
        if vertex_product.is_structurally_zero() {
            return;
        }
    }
    if profile.enabled {
        profile.leaves += 1;
    }
    total.add_term(external_state_indices, &coefficient.mul(&vertex_product));
}

pub(crate) fn is_stable_cohft_range(genus: usize, markings: usize) -> bool {
    crate::graphs::is_stable_moduli_range(genus, markings)
}

fn checked_stable_graph_work_dimension(genus: usize, markings: usize) -> Result<usize, GwError> {
    crate::graphs::stable_graph_generation_bounds(genus, markings)?;
    crate::graphs::stable_graph_dimension(genus, markings)
}

pub(crate) fn ancestor_insertion_terms_from_provider<C, P>(
    provider: &P,
    insertions: &[P::Insertion],
    descendant_s: &SeriesSMatrix<C>,
    psi_inverse: &SeriesMatrix<C>,
    q_degree: usize,
    max_power: usize,
) -> Result<Vec<Vec<AncestorLegTerm<C>>>, GwError>
where
    C: Coeff,
    P: CoefficientSemisimpleCohftProvider<C>,
{
    // For tau_k(gamma), the coefficient of z^{-s} in S contributes an ancestor
    // insertion psi^{k-s}.  Applying Psi^{-1} then expresses the flat class in
    // the canonical idempotent basis used by the graph colors.
    let profile_enabled = crate::env_flag("GW_PROFILE");
    insertions
        .iter()
        .enumerate()
        .map(|(idx, insertion)| {
            let insertion_started = Instant::now();
            let descendant_power = provider.coeff_descendant_power(insertion);
            let flat_class_vector = provider.coeff_insertion_vector(insertion, q_degree)?;
            if profile_enabled {
                eprintln!(
                    "GW_PROFILE ancestor_insertion_vector[{idx}]={:.3}s",
                    insertion_started.elapsed().as_secs_f64()
                );
            }
            let max_order =
                descendant_power.min(descendant_s.coefficients().len().saturating_sub(1));
            let mut terms = Vec::new();
            for s_order in 0..=max_order {
                let base_power = descendant_power - s_order;
                if base_power > max_power {
                    continue;
                }
                let s_started = Instant::now();
                let flat_vector = apply_s_coefficient_to_vector(
                    descendant_s,
                    s_order,
                    &flat_class_vector,
                    q_degree,
                );
                if profile_enabled {
                    let (terms, factors) = qseries_vector_complexity(&flat_vector);
                    eprintln!(
                        "GW_PROFILE ancestor_apply_s[{idx},{s_order}]={:.3}s terms={} factors={}",
                        s_started.elapsed().as_secs_f64(),
                        terms,
                        factors
                    );
                }
                if flat_vector.iter().all(QSeries::is_structurally_zero) {
                    continue;
                }
                let psi_started = Instant::now();
                let canonical_vector = apply_matrix_to_vector(psi_inverse, &flat_vector, q_degree);
                if profile_enabled {
                    let (terms, factors) = qseries_vector_complexity(&canonical_vector);
                    eprintln!(
                        "GW_PROFILE ancestor_apply_psi_inverse[{idx},{s_order}]={:.3}s terms={} factors={}",
                        psi_started.elapsed().as_secs_f64(),
                        terms,
                        factors
                    );
                }
                if canonical_vector.iter().all(QSeries::is_structurally_zero) {
                    continue;
                }
                terms.push(AncestorLegTerm {
                    base_power,
                    vector: canonical_vector,
                });
            }
            Ok(terms)
        })
        .collect()
}

pub(crate) fn apply_s_coefficient_to_vector<C>(
    descendant_s: &SeriesSMatrix<C>,
    s_order: usize,
    class_vector: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>>
where
    C: Coeff,
{
    let matrix = descendant_s
        .coefficient(s_order)
        .expect("S coefficient order was bounded before access");
    apply_matrix_to_vector(matrix, class_vector, q_degree)
}

pub(crate) fn apply_matrix_to_vector<C>(
    matrix: &SeriesMatrix<C>,
    vector: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>>
where
    C: Coeff,
{
    debug_assert_eq!(matrix.cols(), vector.len());
    (0..matrix.rows())
        .map(|row| {
            let mut total = QSeries::<C>::zero(q_degree);
            for (col, entry) in vector.iter().enumerate() {
                total = total.add(&matrix.entry(row, col).mul(entry));
            }
            total
        })
        .collect()
}

pub(crate) fn qseries_vector_complexity<C: Coeff>(vector: &[QSeries<C>]) -> (usize, usize) {
    (
        vector.iter().map(QSeries::complexity_terms).sum(),
        vector
            .iter()
            .map(QSeries::complexity_denominator_factors)
            .sum(),
    )
}

pub(crate) fn inverse_r_coefficients<C: Coeff>(
    coefficients: &[SeriesMatrix<C>],
) -> Vec<SeriesMatrix<C>> {
    // Formal inverse of R(z) with R_0 = 1.  The recurrence is the coefficient
    // extraction of R(z) R(z)^{-1} = 1.
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

pub(crate) fn translation_coefficients<C>(
    inverse_r: &[SeriesMatrix<C>],
    unit: &[QSeries<C>],
    q_degree: usize,
) -> Vec<Vec<QSeries<C>>>
where
    C: Coeff,
{
    // Givental's translation is T(psi)=psi(1-R^{-1})1.  Since R^{-1}_0=1, the
    // first nonzero coefficient appears at psi^2.
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

pub(crate) fn edge_propagator_coefficients<C>(
    inverse_r: &[SeriesMatrix<C>],
    metric: &SeriesMatrix<C>,
    max_power: usize,
    q_degree: usize,
) -> Result<Vec<Vec<Vec<Vec<QSeries<C>>>>>, GwError>
where
    C: Coeff,
{
    // The edge term is the regular part of
    //   (eta^{-1} - R^{-1}(psi_1) eta^{-1} R^{-1}(-psi_2)^T)
    //       / (psi_1 + psi_2).
    // This expands that quotient into coefficients of psi_1^a psi_2^b for
    // every pair of endpoint colors.
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

pub(crate) fn edge_numerator_coefficient<C>(
    inverse_r: &[SeriesMatrix<C>],
    metric_inverse: &[QSeries<C>],
    left_color: usize,
    right_color: usize,
    left_power: usize,
    right_power: usize,
    q_degree: usize,
) -> QSeries<C>
where
    C: Coeff,
{
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

pub(crate) fn accumulate_graph_factors<C>(
    graph: &crate::graphs::StableGraph,
    colors: &[usize],
    leg_options: &[Vec<Vec<LegFactorOption<C>>>],
    edge_options: &[Vec<Vec<EdgeFactorOption<C>>>],
    calibration: &SemisimpleCalibration<C>,
    translation: &[Vec<QSeries<C>>],
    oracle: &WittenKontsevich,
    vertex_cache: &mut HashMap<VertexContributionKey, Arc<QSeries<C>>>,
    q_degree: usize,
    max_power: usize,
    factor_index: usize,
    current_power_sum: usize,
    coefficient: QSeries<C>,
    base_powers: &mut [Vec<usize>],
    vertex_power_sums: &mut [usize],
    vertex_power_caps: &[usize],
    total: &mut QSeries<C>,
    profile: &mut GraphEvalProfile,
) where
    C: Coeff,
{
    // Recursively chooses one leg/edge factor option for every half-edge in a
    // fixed colored stable graph.  At the leaves, the collected psi powers are
    // integrated over each vertex Mbar_{g(v),val(v)} with translation
    // insertions included by `vertex_contribution_with_translations`.
    if profile.enabled {
        profile.recursion_calls += 1;
    }
    if coefficient.is_structurally_zero() || current_power_sum > max_power {
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
            if next_coefficient.is_structurally_zero() {
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
            if next_coefficient.is_structurally_zero() {
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
        if vertex_product.is_structurally_zero() {
            return;
        }
    }
    if profile.enabled {
        profile.leaves += 1;
    }
    *total = total.add(&coefficient.mul(&vertex_product));
}

pub(crate) fn leg_options_by_marking_color<C: Coeff>(
    insertion_terms: &[Vec<AncestorLegTerm<C>>],
    inverse_r: &[SeriesMatrix<C>],
    q_degree: usize,
    max_power: usize,
    colors: usize,
) -> Vec<Vec<Vec<LegFactorOption<C>>>> {
    insertion_terms
        .iter()
        .map(|terms| {
            (0..colors)
                .map(|color| leg_options_for_color(color, terms, inverse_r, q_degree, max_power))
                .collect()
        })
        .collect()
}

pub(crate) fn leg_options_for_color<C: Coeff>(
    color: usize,
    insertion_terms: &[AncestorLegTerm<C>],
    inverse_r: &[SeriesMatrix<C>],
    q_degree: usize,
    max_power: usize,
) -> Vec<LegFactorOption<C>> {
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
            if !coefficient.is_structurally_zero() {
                by_power[power] = by_power[power].add(&coefficient);
            }
        }
    }
    by_power
        .into_iter()
        .enumerate()
        .filter_map(|(power, coefficient)| {
            (!coefficient.is_structurally_zero()).then_some(LegFactorOption { power, coefficient })
        })
        .collect()
}

pub(crate) fn edge_options_by_color<C: Coeff>(
    edge_coefficients: &[Vec<Vec<Vec<QSeries<C>>>>],
) -> Vec<Vec<Vec<EdgeFactorOption<C>>>> {
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

pub(crate) fn edge_options_for_colors<C>(
    left_color: usize,
    right_color: usize,
    edge_coefficients: &[Vec<Vec<Vec<QSeries<C>>>>],
) -> Vec<EdgeFactorOption<C>>
where
    C: Coeff,
{
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
            if !coefficient.is_structurally_zero() {
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

pub(crate) fn vertex_contribution_with_translations<C>(
    genus: usize,
    color: usize,
    base_powers: &[usize],
    calibration: &SemisimpleCalibration<C>,
    translation: &[Vec<QSeries<C>>],
    oracle: &WittenKontsevich,
    vertex_cache: &mut HashMap<VertexContributionKey, Arc<QSeries<C>>>,
    q_degree: usize,
    profile: &mut GraphEvalProfile,
) -> QSeries<C>
where
    C: Coeff,
{
    // A vertex is the diagonal TFT factor times a point-theory psi integral.
    // If the chosen half-edge powers do not fill the vertex dimension, the
    // missing degree is supplied by any number of translation insertions
    // T(psi), divided by their factorial symmetry.
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
        return cached.as_ref().clone();
    }
    if profile.enabled {
        profile.vertex_cache_misses += 1;
    }

    let base_dimension = match crate::graphs::stable_graph_dimension(genus, base_powers.len()) {
        Ok(dimension) => dimension,
        Err(_) => {
            let zero = QSeries::<C>::zero(q_degree);
            vertex_cache.insert(key, Arc::new(zero.clone()));
            return zero;
        }
    };
    let Some(base_power_sum) = base_powers
        .iter()
        .try_fold(0usize, |sum, power| sum.checked_add(*power))
    else {
        let zero = QSeries::<C>::zero(q_degree);
        vertex_cache.insert(key, Arc::new(zero.clone()));
        return zero;
    };
    let Some(translation_excess) = base_dimension.checked_sub(base_power_sum) else {
        let zero = QSeries::<C>::zero(q_degree);
        vertex_cache.insert(key, Arc::new(zero.clone()));
        return zero;
    };

    let mut total = QSeries::<C>::zero(q_degree);
    if translation_excess == 0 {
        let vertex_factor = vertex_tft_factor(genus, base_powers.len(), color, calibration);
        total = total
            .add(&vertex_factor.scale(&C::from_rational(oracle.psi_integral(genus, base_powers))));
    }

    for partition in translation_excess_partitions(translation_excess) {
        if profile.enabled {
            profile.translation_terms += 1;
        }

        let translation_count = partition
            .iter()
            .map(|(_, multiplicity)| *multiplicity)
            .sum::<usize>();
        let mut powers = Vec::with_capacity(base_powers.len() + translation_count);
        powers.extend_from_slice(base_powers);

        let mut coefficient = QSeries::<C>::one(q_degree);
        let mut symmetry = C::one();
        for (excess, multiplicity) in partition {
            let power = excess + 1;
            if power >= translation[color].len() {
                coefficient = QSeries::zero(q_degree);
                break;
            }

            coefficient = coefficient.mul(&translation[color][power].pow_usize(multiplicity));
            if coefficient.is_structurally_zero() {
                break;
            }
            powers.extend(std::iter::repeat_n(power, multiplicity));

            let multiplicity_factor = C::from_rational(factorial(multiplicity));
            symmetry = symmetry.mul(&multiplicity_factor);
        }
        if coefficient.is_structurally_zero() {
            continue;
        }

        let vertex_factor = vertex_tft_factor(genus, powers.len(), color, calibration);
        let psi = C::from_rational(oracle.psi_integral(genus, &powers));
        let term = coefficient.mul(&vertex_factor).scale(&psi.div(&symmetry));
        total = total.add(&term);
    }
    vertex_cache.insert(key, Arc::new(total.clone()));
    total
}

pub(crate) fn vertex_tft_factor<C>(
    genus: usize,
    valence: usize,
    color: usize,
    calibration: &SemisimpleCalibration<C>,
) -> QSeries<C>
where
    C: Coeff,
{
    // In an unnormalized/relative-normalized canonical frame the TFT vertex is
    // diagonal.  The powers below are the usual product-of-point-theories
    // factors rewritten in the frame stored by `SemisimpleCalibration`.
    let genus_factor = if genus == 0 {
        calibration.inverse_delta[color].clone()
    } else {
        calibration.delta[color].pow_usize(genus - 1)
    };
    genus_factor.mul(&calibration.relative_sqrt_delta[color].pow_usize(valence))
}

pub(crate) fn factorial(n: usize) -> Rational {
    // Exact big-rational accumulation: usize factorials overflow at 21!,
    // which translation multiplicities can reach at high vertex dimension.
    let mut out = Rational::one();
    for k in 2..=n {
        out = out * Rational::from(k);
    }
    out
}

/// Unordered translation excess profiles.
///
/// A translation insertion with psi power `power` contributes one new marked
/// point and consumes `power - 1` units of the vertex dimension excess.  The
/// older ordered-composition expansion of `exp(T)` produced many identical
/// point-theory terms and then divided by the total number of translations
/// factorial.  Grouping by multiplicity profiles leaves exactly one term per
/// partition with symmetry factor `prod_e c_e!`, where `c_e` is the
/// multiplicity of excess `e`.
pub(crate) fn translation_excess_partitions(total: usize) -> Vec<Vec<(usize, usize)>> {
    fn rec(
        next_excess: usize,
        remaining: usize,
        current: &mut Vec<(usize, usize)>,
        out: &mut Vec<Vec<(usize, usize)>>,
    ) {
        if remaining == 0 {
            out.push(current.clone());
            return;
        }

        for excess in next_excess..=remaining {
            let max_multiplicity = remaining / excess;
            for multiplicity in 1..=max_multiplicity {
                current.push((excess, multiplicity));
                rec(excess + 1, remaining - excess * multiplicity, current, out);
                current.pop();
            }
        }
    }

    if total == 0 {
        return Vec::new();
    }

    let mut out = Vec::new();
    rec(1, total, &mut Vec::new(), &mut out);
    out
}

/// Conservative retained/transient-memory envelope for materialized color
/// assignments across one prepared stable-graph table.  The estimator charges
/// each coloring for its `Vec` header and `usize` payload, then multiplies by
/// eight for the raw list, orbit/visited hash sets, prepared copy, allocator
/// metadata, and the short overlap between cached representations.
pub(crate) const MAX_STABLE_GRAPH_COLORING_BYTES: usize = 64 * 1024 * 1024;
const COLORING_STORAGE_AMPLIFICATION: usize = 8;
const PREPARED_COLORING_CACHE_CAPACITY: usize = 8;

fn coloring_resource_limit(operation: &str, requested: usize) -> GwError {
    GwError::ResourceLimit {
        operation: operation.to_string(),
        requested,
        limit: MAX_STABLE_GRAPH_COLORING_BYTES,
    }
}

fn vertex_coloring_count(vertices: usize, colors: usize) -> Result<usize, GwError> {
    let mut count = 1usize;
    for _ in 0..vertices {
        count = count.checked_mul(colors).ok_or_else(|| {
            coloring_resource_limit("estimated vertex-coloring storage", usize::MAX)
        })?;
    }
    Ok(count)
}

fn estimated_vertex_coloring_storage(
    vertices: usize,
    count: usize,
    operation: &str,
) -> Result<usize, GwError> {
    let payload_bytes = vertices
        .checked_mul(std::mem::size_of::<usize>())
        .ok_or_else(|| coloring_resource_limit(operation, usize::MAX))?;
    std::mem::size_of::<Vec<usize>>()
        .checked_add(payload_bytes)
        .and_then(|per_coloring| per_coloring.checked_mul(count))
        .and_then(|bytes| bytes.checked_mul(COLORING_STORAGE_AMPLIFICATION))
        .ok_or_else(|| coloring_resource_limit(operation, usize::MAX))
}

pub(crate) fn checked_vertex_coloring_count(
    vertices: usize,
    colors: usize,
) -> Result<usize, GwError> {
    let count = vertex_coloring_count(vertices, colors)?;
    let requested =
        estimated_vertex_coloring_storage(vertices, count, "estimated vertex-coloring storage")?;
    if requested > MAX_STABLE_GRAPH_COLORING_BYTES {
        return Err(coloring_resource_limit(
            "estimated vertex-coloring storage",
            requested,
        ));
    }
    Ok(count)
}

fn checked_prepared_coloring_storage(
    graphs: &[StableGraph],
    colors: usize,
) -> Result<usize, GwError> {
    let mut total = 0usize;
    for graph in graphs {
        let count = vertex_coloring_count(graph.vertices.len(), colors)?;
        let requested = estimated_vertex_coloring_storage(
            graph.vertices.len(),
            count,
            "estimated prepared stable-graph coloring storage",
        )?;
        total = total.checked_add(requested).ok_or_else(|| {
            coloring_resource_limit(
                "estimated prepared stable-graph coloring storage",
                usize::MAX,
            )
        })?;
    }
    if total > MAX_STABLE_GRAPH_COLORING_BYTES {
        return Err(coloring_resource_limit(
            "estimated prepared stable-graph coloring storage",
            total,
        ));
    }
    Ok(total)
}

pub(crate) fn vertex_colorings(vertices: usize, colors: usize) -> Result<Vec<Vec<usize>>, GwError> {
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
    let count = checked_vertex_coloring_count(vertices, colors)?;
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|_| {
        GwError::UnsupportedInvariant(format!(
            "cannot allocate {count} stable-graph vertex colorings"
        ))
    })?;
    rec(vertices, colors, &mut Vec::new(), &mut out);
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ColoringOrbit {
    colors: Vec<usize>,
    multiplicity: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedColoringOrbit {
    colors: Vec<usize>,
    factor: Rational,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedStableGraph {
    graph: StableGraph,
    vertex_power_caps: Vec<usize>,
    colorings: Arc<Vec<PreparedColoringOrbit>>,
}

pub(crate) fn prepared_stable_graphs(
    genus: usize,
    markings: usize,
    colors: usize,
) -> Result<Arc<Vec<PreparedStableGraph>>, GwError> {
    // Stable graphs and color orbits depend only on (g,n,number of idempotents),
    // not on insertions or degree.  Precomputing automorphism factors and vertex
    // dimension caps avoids repeating graph-theoretic work in series mode.
    static CACHE: OnceLock<
        Mutex<BoundedCache<(usize, usize, usize), Arc<Vec<PreparedStableGraph>>>>,
    > = OnceLock::new();
    let key = (genus, markings, colors);
    let cache =
        CACHE.get_or_init(|| Mutex::new(BoundedCache::new(PREPARED_COLORING_CACHE_CAPACITY)));
    if let Some(cached) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(cached);
    }

    let stable_graphs = try_stable_graphs(genus, markings)?;
    // Bound the whole table, not only each graph: thousands of individually
    // modest coloring sets can otherwise retain an unbounded aggregate.
    checked_prepared_coloring_storage(&stable_graphs, colors)?;
    let graphs = stable_graphs
        .into_iter()
        .map(|graph| -> Result<PreparedStableGraph, GwError> {
            let automorphism_factor = Rational::one() / Rational::from(graph.automorphism_order());
            let colorings = vertex_coloring_orbits(&graph, colors)?
                .iter()
                .map(|orbit| PreparedColoringOrbit {
                    colors: orbit.colors.clone(),
                    factor: automorphism_factor.clone() * Rational::from(orbit.multiplicity),
                })
                .collect::<Vec<_>>();
            let vertex_power_caps = graph
                .vertices
                .iter()
                .enumerate()
                .map(|(vertex, stable_vertex)| {
                    crate::graphs::stable_graph_dimension(
                        stable_vertex.genus,
                        graph.valence(vertex),
                    )
                    .expect("stable-graph generation produced only stable vertices")
                })
                .collect::<Vec<_>>();
            Ok(PreparedStableGraph {
                graph,
                vertex_power_caps,
                colorings: Arc::new(colorings),
            })
        })
        .collect::<Result<Vec<_>, GwError>>()?;
    let graphs = Arc::new(graphs);
    cache.lock().unwrap().insert(key, graphs.clone());
    Ok(graphs)
}

fn profiled_prepared_stable_graphs(
    genus: usize,
    markings: usize,
    colors: usize,
    profile: &mut GraphEvalProfile,
) -> Result<Arc<Vec<PreparedStableGraph>>, GwError> {
    let started = Instant::now();
    let graphs = prepared_stable_graphs(genus, markings, colors);
    let elapsed = started.elapsed();
    profile.add_stable_graph_elapsed(elapsed);
    let graphs = graphs?;
    profile.prepared_stable_graphs = graphs.len();
    if profile.enabled {
        eprintln!(
            "GW_PROFILE stable_graph_generation={:.3}s genus={} markings={} colors={} prepared_stable_graphs={}",
            elapsed.as_secs_f64(),
            genus,
            markings,
            colors,
            graphs.len()
        );
    }
    Ok(graphs)
}

pub(crate) fn vertex_coloring_orbits(
    graph: &StableGraph,
    colors: usize,
) -> Result<Arc<Vec<ColoringOrbit>>, GwError> {
    static CACHE: OnceLock<Mutex<BoundedCache<(StableGraph, usize), Arc<Vec<ColoringOrbit>>>>> =
        OnceLock::new();
    let key = (graph.clone(), colors);
    let cache =
        CACHE.get_or_init(|| Mutex::new(BoundedCache::new(PREPARED_COLORING_CACHE_CAPACITY)));
    if let Some(cached) = cache.lock().unwrap().get(&key).cloned() {
        return Ok(cached);
    }

    let automorphisms = graph.vertex_automorphism_permutations();
    let all_colorings = vertex_colorings(graph.vertices.len(), colors)?;
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
    Ok(orbits)
}

pub(crate) fn permute_coloring(coloring: &[usize], permutation: &[usize]) -> Vec<usize> {
    permutation.iter().map(|&old| coloring[old]).collect()
}

#[cfg(test)]
mod tests;
