//! Projective-space descendant-potential batching and packed resolvents.
//!
//! These adapters enumerate the cyclic H-power insertion basis and assemble
//! projective request/result types around the target-neutral Givental graph
//! kernels. Negative split twists over projective space reuse the same batch
//! orchestration through their own provider.

use super::api::*;
use super::resolvent::*;
use crate::core::algebra::{Coeff, RatFun};
use crate::core::error::GwError;
use crate::givental::{
    ancestor_insertion_terms_from_provider, build_external_leg_kernel_for_problem,
    build_restricted_external_leg_kernel_for_problem,
    build_restricted_external_leg_kernel_with_coeff_for_problem,
    checked_stable_graph_work_dimension, compute_semisimple_graph_value,
    compute_semisimple_graph_value_with_coeff, contract_external_leg_kernel_coeff,
    contract_restricted_external_leg_kernel_coeff,
    contract_restricted_external_leg_kernel_coeff_generic, dimension_mismatch,
    exact_dimension_mismatch, graph_worker_count, is_stable_cohft_range,
    leg_options_by_marking_color, ExternalLegKernel, GiventalGraphKernel, LegFactorOption,
    RestrictedExternalLegKernel, SemisimpleCohftProvider, SeriesSMatrix,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MasterLegOptionsKey {
    q_degree: usize,
    markings: usize,
    descendant_power: usize,
    class_power: usize,
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
    let basis = insertion_basis(req.n, req.max_descendant_power);
    let mut candidates_by_degree = vec![Vec::<Vec<Insertion>>::new(); req.degree_max + 1];

    for markings in 0..=req.max_markings {
        for insertions in insertion_monomials(&basis, markings) {
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
                            insertion_monomial_label(&insertions)
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
                            insertion_monomial_label(&insertions)
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
                        insertion_monomial_label(&insertions)
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
    P: SemisimpleCohftProvider<C, Insertion = Insertion>,
    N: FnMut(C) -> Result<C, GwError>,
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
        let mut value = ResolventPolynomial::zero();
        let mut nonzero_terms = 0usize;
        let candidate_terms = usize::from(req.virtual_dimension == 0);
        let provider_dimension_matches = provider
            .virtual_dimension(req.genus, req.degree, 0)
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
    let graph_kernel = provider.graph_kernel(req.degree, needed_r_order, graph_dimension)?;
    let descendant_s = provider.descendant_s_matrix(req.degree, needed_s_order)?;
    let mut tasks = Vec::<MasterContractionTask<C>>::new();
    let mut task_indices = Vec::<ResolventIndex>::new();
    let candidate_terms = enumerate_resolvent_indices(req, |index| {
        let insertions = index.to_insertions(req.target_n);
        if let (Some(total_degree), Some(virtual_dimension)) = (
            provider.insertion_degree(&insertions),
            provider.virtual_dimension(req.genus, req.degree, req.markings),
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
                &graph_kernel.calibration().psi_inverse,
                req.degree,
                graph_dimension,
            )?;
            let mut options = leg_options_by_marking_color(
                &insertion_terms,
                graph_kernel.inverse_r(),
                req.degree,
                graph_dimension,
                provider.colors(),
            );
            leg_options.push(
                options
                    .pop()
                    .unwrap_or_else(|| vec![Vec::<LegFactorOption<C>>::new(); provider.colors()]),
            );
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

    let restricted_kernel = build_restricted_external_leg_kernel_with_coeff_for_problem(
        req.genus,
        req.markings,
        provider.colors(),
        &graph_kernel,
        req.degree,
        graph_dimension,
        tasks.iter().map(|task| task.leg_options.as_slice()),
    )?;

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

fn compute_no_marking_packed_resolvent<P, N>(
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

fn shared_kernel_task_counts(
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
struct GiventalMasterEvaluator<P>
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
        let worker_count = graph_worker_count(tasks.len());
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
        build_restricted_external_leg_kernel_for_problem(
            self.genus,
            markings,
            self.colors(),
            &graph_kernel,
            q_degree,
            graph_dimension,
            tasks.iter().map(|task| task.leg_options.as_slice()),
        )
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
        build_external_leg_kernel_for_problem(
            self.genus,
            markings,
            self.colors(),
            &graph_kernel,
            self.degree_max,
            graph_dimension,
        )
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
        let psi_inverse = graph_kernel.calibration().psi_inverse.clone();
        let inverse_r = graph_kernel.inverse_r().to_vec();
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
struct MasterContractionTask<C = RatFun> {
    ordinal: usize,
    degree: usize,
    insertions: Vec<Insertion>,
    markings: usize,
    leg_options: Vec<Vec<Vec<LegFactorOption<C>>>>,
}

#[derive(Debug)]
struct OrderedSeriesCoefficient {
    ordinal: usize,
    coefficient: SeriesCoefficient,
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

fn contract_restricted_tasks_parallel(
    kernel: &RestrictedExternalLegKernel,
    tasks: &[MasterContractionTask],
    include_zero: bool,
) -> Vec<OrderedSeriesCoefficient> {
    let worker_count = graph_worker_count(tasks.len());
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

#[cfg(test)]
mod tests;
