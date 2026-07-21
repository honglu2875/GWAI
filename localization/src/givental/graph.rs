//! The Givental stable-graph evaluation engine: graph kernels, parallel
//! contraction workers, external-leg tensors, and provider-generic scalar and
//! series entry points.

use super::{
    SemisimpleCalibration, SemisimpleCohftProvider, SeriesRMatrix, SeriesSMatrix, Truncation,
};
use crate::core::algebra::{Coeff, RatFun, Rational};
use crate::core::bounded_cache::BoundedCache;
use crate::core::error::GwError;
use crate::core::series::{QSeries, RationalQSeries, SeriesMatrix};
use crate::factored::FactoredRatFun;
use crate::graphs::{try_stable_graphs, StableGraph};
#[cfg(test)]
use crate::spaces::projective_space::provider::{
    projective_space_graph_kernel, projective_space_j_calibration_at_lambda_weights,
};
use crate::tautological::{TautologicalOracle, WittenKontsevich};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

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

#[derive(Debug)]
pub struct GiventalGraphKernel<C = RatFun> {
    calibration: SemisimpleCalibration<C>,
    inverse_r: Vec<SeriesMatrix<C>>,
    translation: Vec<Vec<QSeries<C>>>,
    edge_options: Vec<Vec<Vec<EdgeFactorOption<C>>>>,
    vertex_cache: Mutex<HashMap<VertexContributionKey, Arc<QSeries<C>>>>,
    // Symbolic graph contraction uses factored denominators.  Keeping the
    // converted twin here is important when one provider evaluates a family
    // of correlators (for example a Virasoro dependency closure): rebuilding
    // it for every correlator also discarded its reusable vertex cache.
    // The field is present on every coefficient specialization so the generic
    // kernel remains one type; only the RatFun entry point initializes it.
    factored_twin: OnceLock<Arc<GiventalGraphKernel<FactoredRatFun>>>,
}

impl<C: Coeff> GiventalGraphKernel<C> {
    /// Builds the Feynman-rule kernel from a semisimple calibration.
    ///
    /// This performs the universal part of quantization:
    /// `R -> R^{-1}`, `R^{-1}1 -> T`, and `R^{-1},eta^{-1} ->` edge
    /// propagators.  The first step uses the `SeriesRMatrix` symplectic
    /// contract
    ///
    /// `R(-z)^T eta R(z) = eta`,
    ///
    /// hence `(R^{-1})_k = (-1)^k eta^{-1} R_k^T eta`.  Because `eta` is the
    /// diagonal canonical metric, this is implemented by row/column scaling
    /// rather than dense matrix products.  It does not inspect target
    /// geometry.
    pub fn from_calibration(
        calibration: SemisimpleCalibration<C>,
        graph_dimension: usize,
    ) -> Result<Self, GwError> {
        let profile = crate::env_flag("GW_PROFILE");
        let started = std::time::Instant::now();
        let q_degree = calibration.r_matrix.q_degree();
        let stage = std::time::Instant::now();
        let inverse_metric = constant_inverse_metric_diagonal(&calibration, q_degree)?;
        let inverse_r = symplectic_inverse_r_coefficients(
            &calibration.r_matrix,
            &calibration.metric,
            &inverse_metric,
        )?;
        if profile {
            eprintln!(
                "GW_PROFILE graph_kernel_inverse_r={:.3}s q_degree={} r_order={} colors={}",
                stage.elapsed().as_secs_f64(),
                q_degree,
                calibration.r_matrix.z_order(),
                calibration.r_matrix.size()
            );
        }
        let unit = calibration.relative_sqrt_delta_inverse.clone();
        let stage = std::time::Instant::now();
        let translation = translation_coefficients(&inverse_r, &unit, q_degree);
        if profile {
            eprintln!(
                "GW_PROFILE graph_kernel_translation={:.3}s",
                stage.elapsed().as_secs_f64()
            );
        }
        let stage = std::time::Instant::now();
        let kernel = Self::from_parts_with_inverse_metric(
            calibration,
            inverse_r,
            translation,
            inverse_metric,
            graph_dimension,
        )?;
        if profile {
            eprintln!(
                "GW_PROFILE graph_kernel_edges={:.3}s graph_dimension={} total={:.3}s",
                stage.elapsed().as_secs_f64(),
                graph_dimension,
                started.elapsed().as_secs_f64()
            );
        }
        Ok(kernel)
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
        let inverse_metric = constant_inverse_metric_diagonal(&calibration, q_degree)?;
        validate_canonical_metric_inverse(
            &calibration.metric,
            &inverse_metric,
            calibration.r_matrix.size(),
            q_degree,
        )?;
        Self::from_parts_with_inverse_metric(
            calibration,
            inverse_r,
            translation,
            inverse_metric,
            graph_dimension,
        )
    }

    fn from_parts_with_inverse_metric(
        calibration: SemisimpleCalibration<C>,
        inverse_r: Vec<SeriesMatrix<C>>,
        translation: Vec<Vec<QSeries<C>>>,
        inverse_metric: Vec<QSeries<C>>,
        graph_dimension: usize,
    ) -> Result<Self, GwError> {
        let q_degree = calibration.r_matrix.q_degree();
        let edge_coefficients =
            edge_propagator_coefficients(&inverse_r, &inverse_metric, graph_dimension, q_degree)?;
        let edge_options = edge_options_by_color(&edge_coefficients);
        Ok(Self {
            calibration,
            inverse_r,
            translation,
            edge_options,
            vertex_cache: Mutex::new(HashMap::new()),
            factored_twin: OnceLock::new(),
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

/// Extracts the inverse of the constant canonical metric without performing a
/// second symbolic inversion.
///
/// `SemisimpleCalibration::metric` stores the constant canonical metric, while
/// `delta` stores the inverse metric norm as a Novikov series for the TFT
/// vertex factors.  Its constant coefficient is therefore exactly the
/// diagonal inverse needed by the edge propagator.  Re-inverting `metric`
/// here is mathematically redundant and is particularly costly for coefficient
/// rings which retain a sum of factored fractions.
fn constant_inverse_metric_diagonal<C: Coeff>(
    calibration: &SemisimpleCalibration<C>,
    q_degree: usize,
) -> Result<Vec<QSeries<C>>, GwError> {
    let colors = calibration.metric.rows();
    if calibration.metric.cols() != colors || calibration.delta.len() != colors {
        return Err(GwError::ConventionMismatch(format!(
            "canonical metric/delta shape mismatch: metric is {}x{}, delta has {} entries",
            calibration.metric.rows(),
            calibration.metric.cols(),
            calibration.delta.len()
        )));
    }
    calibration
        .delta
        .iter()
        .map(|delta| {
            delta
                .coeff(0)
                .cloned()
                .map(|constant| QSeries::constant(constant, q_degree))
                .ok_or_else(|| {
                    GwError::ConventionMismatch(
                        "canonical inverse-metric series has no constant coefficient".to_string(),
                    )
                })
        })
        .collect()
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

pub(crate) fn exact_dimension_mismatch<P>(
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
    P: SemisimpleCohftProvider<C>,
{
    if !provider.degree_is_effective(degree) {
        return Ok(C::zero());
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

/// Computes one stable graph's contribution to a bounded descendant
/// potential using a caller-supplied provider and finite insertion profiles.
///
/// This is the formula renderer's graph-local rational path.  It uses the same
/// external-leg kernel as `series`: each marking is left as an open
/// `(canonical color, psi power)` state while the selected stable graph is
/// fully contracted over colors, edge propagators, translations, and
/// point-theory vertex integrals.  The final loop then attaches bounded flat
/// insertions through the calibrated `S`, `Psi^{-1}`, and `R^{-1}` leg options.
pub(crate) fn graph_bounded_potential_coefficients_with_provider<P>(
    provider: &P,
    genus: usize,
    markings: usize,
    graph_index: usize,
    degree_max: usize,
    max_descendant_power: usize,
    insertion_profiles: impl IntoIterator<Item = Vec<P::Insertion>>,
) -> Result<Vec<(usize, Vec<P::Insertion>, RatFun)>, GwError>
where
    P: SemisimpleCohftProvider,
    P::Insertion: Clone,
{
    if !is_stable_cohft_range(genus, markings) {
        return Err(GwError::UnsupportedInvariant(
            "graph-local rational potential is implemented for stable (g,m) CohFT ranges only"
                .to_string(),
        ));
    }

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
    let mut out = Vec::new();
    for insertions in insertion_profiles {
        let insertion_terms = ancestor_insertion_terms_from_provider(
            provider,
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
            if dimension_mismatch(provider, genus, degree, &insertions).is_some() {
                continue;
            }
            let value = contract_external_leg_kernel_coeff(&external_kernel, &leg_options, degree);
            if !value.is_zero() {
                out.push((degree, insertions.clone(), value));
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
    P: SemisimpleCohftProvider<C>,
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
        let descendant_s = provider.descendant_s_matrix(q_degree, needed_s_order)?;
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
            provider.colors(),
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

    let graphs =
        profiled_prepared_stable_graphs(genus, insertions.len(), provider.colors(), &mut profile)?;
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
                if provider.degree_is_effective(degree) {
                    coefficient.clone()
                } else {
                    C::zero()
                }
            })
            .collect(),
    ))
}

pub(crate) fn graph_worker_count(work_items: usize) -> usize {
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
    let worker_count = graph_worker_count(graphs.len());
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
    for (prepared_index, prepared) in graphs.iter().enumerate() {
        let graph_started = Instant::now();
        // Keep one graph's colorings together before merging it into the
        // worker total. Besides making profile complexity attributable, this
        // also lets coefficient rings combine like denominators locally.
        let mut prepared_total = QSeries::<C>::zero(q_degree);
        for coloring_index in 0..prepared.colorings.len() {
            accumulate_scalar_coloring(
                prepared,
                coloring_index,
                leg_options,
                kernel,
                q_degree,
                graph_dimension,
                &mut vertex_cache,
                &mut prepared_total,
                &mut profile,
            );
        }
        if profile.enabled {
            eprintln!(
                "GW_PROFILE scalar_graph[{prepared_index}]={:.3}s vertices={} edges={} colorings={} terms={} factors={}",
                graph_started.elapsed().as_secs_f64(),
                prepared.graph.vertices.len(),
                prepared.graph.edges.len(),
                prepared.colorings.len(),
                prepared_total.complexity_terms(),
                prepared_total.complexity_denominator_factors(),
            );
        }
        total.add_assign(&prepared_total);
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
        factored_twin: OnceLock::new(),
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
        factored_twin: OnceLock::new(),
    }
}

fn cached_factored_kernel(
    kernel: &GiventalGraphKernel,
) -> Arc<GiventalGraphKernel<FactoredRatFun>> {
    kernel
        .factored_twin
        .get_or_init(|| Arc::new(kernel_to_factored(kernel)))
        .clone()
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
    let factored_kernel = cached_factored_kernel(kernel);
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

pub(crate) fn build_external_leg_kernel_for_problem(
    genus: usize,
    markings: usize,
    colors: usize,
    kernel: &Arc<GiventalGraphKernel>,
    q_degree: usize,
    graph_dimension: usize,
) -> Result<ExternalLegKernel, GwError> {
    let mut profile = GraphEvalProfile::new();
    let graphs = profiled_prepared_stable_graphs(genus, markings, colors, &mut profile)?;
    profile.stable_graphs = graphs.len();
    profile.edge_options = kernel
        .edge_options
        .iter()
        .flat_map(|row| row.iter())
        .map(Vec::len)
        .sum();
    let started = Instant::now();
    let result = evaluate_external_graphs_auto(
        graphs.as_ref(),
        markings,
        colors,
        kernel,
        q_degree,
        graph_dimension,
        &mut profile,
    );
    profile.add_graph_elapsed(started.elapsed());
    profile.finish();
    Ok(result)
}

pub(crate) fn build_restricted_external_leg_kernel_for_problem<'a>(
    genus: usize,
    markings: usize,
    colors: usize,
    kernel: &Arc<GiventalGraphKernel>,
    q_degree: usize,
    graph_dimension: usize,
    leg_options: impl IntoIterator<Item = &'a [Vec<Vec<LegFactorOption>>]>,
) -> Result<RestrictedExternalLegKernel, GwError> {
    let template = RestrictedExternalLegKernel::from_leg_options(
        markings,
        colors,
        graph_dimension,
        q_degree,
        leg_options,
    );
    let mut profile = GraphEvalProfile::new();
    let graphs = profiled_prepared_stable_graphs(genus, markings, colors, &mut profile)?;
    profile.stable_graphs = graphs.len();
    profile.edge_options = kernel
        .edge_options
        .iter()
        .flat_map(|row| row.iter())
        .map(Vec::len)
        .sum();
    let started = Instant::now();
    let result = evaluate_restricted_external_graphs_auto(
        graphs.as_ref(),
        &template,
        kernel,
        q_degree,
        graph_dimension,
        &mut profile,
    );
    profile.add_graph_elapsed(started.elapsed());
    profile.finish();
    Ok(result)
}

pub(crate) fn build_restricted_external_leg_kernel_with_coeff_for_problem<'a, C>(
    genus: usize,
    markings: usize,
    colors: usize,
    kernel: &Arc<GiventalGraphKernel<C>>,
    q_degree: usize,
    graph_dimension: usize,
    leg_options: impl IntoIterator<Item = &'a [Vec<Vec<LegFactorOption<C>>>]>,
) -> Result<RestrictedExternalLegKernel<C>, GwError>
where
    C: Coeff + Send + Sync + 'a,
{
    let template = RestrictedExternalLegKernel::from_leg_options(
        markings,
        colors,
        graph_dimension,
        q_degree,
        leg_options,
    );
    let mut profile = GraphEvalProfile::new();
    let graphs = profiled_prepared_stable_graphs(genus, markings, colors, &mut profile)?;
    profile.stable_graphs = graphs.len();
    profile.edge_options = kernel
        .edge_options
        .iter()
        .flat_map(|row| row.iter())
        .map(Vec::len)
        .sum();
    let started = Instant::now();
    let result = evaluate_restricted_external_graphs_parallel(
        graphs.as_ref(),
        &template,
        kernel,
        q_degree,
        graph_dimension,
        &mut profile,
    );
    profile.add_graph_elapsed(started.elapsed());
    profile.finish();
    Ok(result)
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
    let worker_count = graph_worker_count(graphs.len());
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
    let worker_count = graph_worker_count(graphs.len());
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
    fn from_leg_options<'a>(
        markings: usize,
        colors: usize,
        max_power: usize,
        q_degree: usize,
        tasks: impl IntoIterator<Item = &'a [Vec<Vec<LegFactorOption<C>>>]>,
    ) -> Self
    where
        C: 'a,
    {
        let mut powers_by_marking_color = vec![vec![BTreeSet::<usize>::new(); colors]; markings];
        for leg_options in tasks {
            debug_assert_eq!(leg_options.len(), markings);
            for (marking, by_color) in leg_options.iter().enumerate() {
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
        crate::core::fused::add_product_assign(&mut total, left_coeff, right_coeff);
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
    crate::core::moduli::pointed_curve_is_stable(genus, markings)
}

pub(crate) fn checked_stable_graph_work_dimension(
    genus: usize,
    markings: usize,
) -> Result<usize, GwError> {
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
    P: SemisimpleCohftProvider<C>,
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
            let descendant_power = provider.descendant_power(insertion);
            let flat_class_vector = provider.insertion_vector(insertion, q_degree)?;
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

#[cfg(test)]
pub(crate) fn inverse_r_coefficients<C: Coeff>(
    coefficients: &[SeriesMatrix<C>],
) -> Vec<SeriesMatrix<C>> {
    // Reference construction of the formal inverse of R(z) with R_0 = 1.
    // The graph-kernel hot path instead uses the equivalent symplectic adjoint
    // below.  Retaining this convolution gives tests an implementation which
    // is independent of the symplectic identity.
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

/// Computes `R(z)^{-1}` from the symplectic adjoint of a canonical-frame
/// `R`-matrix.
///
/// The [`SeriesRMatrix`] contract is
/// `R(-z)^T eta R(z) = eta`, so
///
/// `(R^{-1})_k = (-1)^k eta^{-1} R_k^T eta`.
///
/// The canonical metric is diagonal.  Thus entry `(i,j)` only needs the two
/// scalar series multiplications
/// `eta_i^{-1} (R_k)_{j,i} eta_j`, avoiding dense products.  Unitarity itself
/// is the provider's calibration contract and can be audited with
/// [`SeriesRMatrix::check_unitarity`]; this routine validates every structural
/// precondition and exact metric/inverse-metric coherence before indexing so
/// malformed calibrations return an error rather than panicking.
pub(crate) fn symplectic_inverse_r_coefficients<C: Coeff>(
    r_matrix: &SeriesRMatrix<C>,
    metric: &SeriesMatrix<C>,
    inverse_metric_diagonal: &[QSeries<C>],
) -> Result<Vec<SeriesMatrix<C>>, GwError> {
    let size = r_matrix.size();
    let q_degree = r_matrix.q_degree();
    if size == 0
        || r_matrix.coefficients().len() != r_matrix.z_order() + 1
        || r_matrix.coefficients().iter().any(|coefficient| {
            coefficient.rows() != size
                || coefficient.cols() != size
                || coefficient.max_degree() != q_degree
        })
    {
        return Err(GwError::ConventionMismatch(
            "R-matrix shape/truncation is invalid for symplectic inversion".to_string(),
        ));
    }
    validate_canonical_metric_inverse(metric, inverse_metric_diagonal, size, q_degree)?;

    if (0..size).any(|row| {
        (0..size).any(|col| {
            let entry = r_matrix.coefficients()[0].entry(row, col);
            if row == col {
                !entry.is_structurally_one()
            } else {
                !entry.is_structurally_zero()
            }
        })
    }) {
        return Err(GwError::ConventionMismatch(
            "SeriesRMatrix must have R_0 = identity".to_string(),
        ));
    }

    Ok(r_matrix
        .coefficients()
        .iter()
        .enumerate()
        .map(|(order, coefficient)| {
            let entries = (0..size)
                .map(|row| {
                    (0..size)
                        .map(|col| {
                            let entry = inverse_metric_diagonal[row]
                                .mul(coefficient.entry(col, row))
                                .mul(metric.entry(col, col));
                            if order % 2 == 0 {
                                entry
                            } else {
                                entry.neg()
                            }
                        })
                        .collect()
                })
                .collect();
            SeriesMatrix::from_entries(entries)
        })
        .collect())
}

fn validate_canonical_metric_inverse<C: Coeff>(
    metric: &SeriesMatrix<C>,
    inverse_metric_diagonal: &[QSeries<C>],
    size: usize,
    q_degree: usize,
) -> Result<(), GwError> {
    if metric.rows() != size
        || metric.cols() != size
        || metric.max_degree() != q_degree
        || inverse_metric_diagonal.len() != size
        || inverse_metric_diagonal
            .iter()
            .any(|entry| entry.max_degree() != q_degree)
    {
        return Err(GwError::ConventionMismatch(
            "canonical metric/inverse-metric shape or q-truncation mismatch".to_string(),
        ));
    }

    for row in 0..size {
        for col in 0..size {
            if row != col && !metric.entry(row, col).is_structurally_zero() {
                return Err(GwError::ConventionMismatch(
                    "symplectic fast inverse requires a diagonal canonical metric".to_string(),
                ));
            }
        }
    }

    // The optimized adjoint and edge formulas receive the inverse metric
    // through the constant terms of `delta`.  `SemisimpleCalibration` is
    // publicly constructible, so reject incoherent inputs at both public
    // kernel constructors instead of silently computing with a non-inverse.
    // The structural branch is the allocation-free path used by normal
    // providers; the exact fallback also accepts coefficient types with
    // noncanonical representations of one.
    let unit = QSeries::<C>::one(q_degree);
    for color in 0..size {
        let product = inverse_metric_diagonal[color].mul(metric.entry(color, color));
        if !product.is_structurally_one() && !product.sub(&unit).is_zero() {
            return Err(GwError::ConventionMismatch(
                "the canonical metric and inverse-metric norm must be exact inverses".to_string(),
            ));
        }
    }
    Ok(())
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
    metric_inverse: &[QSeries<C>],
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
    let size = metric_inverse.len();
    if inverse_r.first().is_none_or(|matrix| {
        matrix.rows() != size || matrix.cols() != size || matrix.max_degree() != q_degree
    }) || metric_inverse
        .iter()
        .any(|entry| entry.max_degree() != q_degree)
    {
        return Err(GwError::ConventionMismatch(
            "inverse metric/R-matrix shape or q-truncation mismatch".to_string(),
        ));
    }

    let profile = crate::env_flag("GW_PROFILE");
    let coefficients_started = std::time::Instant::now();
    let mut out =
        vec![vec![vec![vec![QSeries::zero(q_degree); max_power + 1]; max_power + 1]; size]; size];
    for left_color in 0..size {
        for right_color in 0..size {
            // Every numerator coefficient on the diagonal p + q <= D + 1
            // occurs in several quotient coefficients.  Computing it inside
            // the alternating sums below repeats the two matrix-entry
            // products O(D) times; that is especially costly for factored
            // coefficient rings.  Cache the triangular numerator table once
            // for this ordered pair of endpoint colors.
            let numerator_width = max_power + 2;
            let mut numerators =
                vec![vec![QSeries::zero(q_degree); numerator_width]; numerator_width];
            for left_order in 1..numerator_width {
                for right_order in 0..(numerator_width - left_order) {
                    numerators[left_order][right_order] = edge_numerator_coefficient(
                        inverse_r,
                        metric_inverse,
                        left_color,
                        right_color,
                        left_order,
                        right_order,
                        q_degree,
                    );
                }
            }

            for left_power in 0..=max_power {
                // A stable-graph contraction of dimension D never consumes an
                // edge monomial psi_1^a psi_2^b with a + b > D.  Leave that
                // unused half of the public rectangular table at zero.
                for right_power in 0..=(max_power - left_power) {
                    let mut coefficient = QSeries::zero(q_degree);
                    for shift in 0..=right_power {
                        let numerator = &numerators[left_power + 1 + shift][right_power - shift];
                        coefficient = if shift % 2 == 0 {
                            coefficient.sub(numerator)
                        } else {
                            coefficient.add(numerator)
                        };
                    }
                    out[left_color][right_color][left_power][right_power] = coefficient;
                }
            }
        }
    }
    if profile {
        eprintln!(
            "GW_PROFILE graph_kernel_edge_coefficients={:.3}s colors={} max_power={}",
            coefficients_started.elapsed().as_secs_f64(),
            size,
            max_power
        );
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
