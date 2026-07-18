//! Public request and result API for ordinary projective space.

use crate::core::algebra::{RatFun, Rational};
use crate::core::error::GwError;
use crate::core::moduli::pointed_curve_is_stable;
use crate::core::theory::{CurveClass, CurveEffectivity, GwTheory};
use crate::factored::FactoredRatFun;
use crate::givental::SemisimpleCohftProvider;
pub use crate::givental::Truncation;
use crate::spaces::projective_space::resolvent::{ResolventRequest, ResolventResult};
use crate::{givental, graphs};

use super::seeds::seed_compute;
use super::{
    CohomologyClass, FactoredProjectiveSpaceProvider, ProjectiveSpaceProvider,
    ProjectiveSpaceTheory,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeMode {
    Givental,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Insertion {
    pub descendant_power: usize,
    pub class: CohomologyClass,
}

impl Insertion {
    pub fn new(descendant_power: usize, class: CohomologyClass) -> Self {
        Self {
            descendant_power,
            class,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvariantRequest {
    pub n: usize,
    pub genus: usize,
    pub degree: usize,
    pub insertions: Vec<Insertion>,
    pub equivariant: bool,
    pub mode: ComputeMode,
    pub truncation: Option<Truncation>,
}

impl InvariantRequest {
    pub fn new(n: usize, genus: usize, degree: usize, insertions: Vec<Insertion>) -> Self {
        Self {
            n,
            genus,
            degree,
            insertions,
            equivariant: false,
            mode: ComputeMode::Givental,
            truncation: None,
        }
    }

    /// Canonical ordinary projective-space theory underlying this request.
    pub fn canonical_theory(&self) -> ProjectiveSpaceTheory {
        self.try_canonical_theory()
            .expect("projective-space request dimension must be representable")
    }

    /// Fallible canonical theory construction for untrusted dimensions.
    pub fn try_canonical_theory(&self) -> Result<ProjectiveSpaceTheory, GwError> {
        ProjectiveSpaceTheory::try_new(self.n)
    }

    /// Checked virtual dimension derived from the canonical target theory.
    pub fn try_virtual_dimension(&self) -> Result<isize, GwError> {
        let degree = i64::try_from(self.degree)
            .map_err(|_| GwError::AlgebraFailure("curve degree does not fit in i64".to_string()))?;
        self.try_canonical_theory()?.virtual_dimension(
            self.genus,
            &CurveClass::new(vec![degree]),
            self.insertions.len(),
        )
    }

    pub fn virtual_dimension(&self) -> isize {
        self.try_virtual_dimension()
            .expect("validated projective-space request has a representable virtual dimension")
    }

    /// Validate target-dependent request data before mathematical shortcuts
    /// such as dimension pruning are applied.
    pub fn validate(&self) -> Result<(), GwError> {
        for (index, insertion) in self.insertions.iter().enumerate() {
            if insertion.class.n() != self.n {
                return Err(GwError::ConventionMismatch(format!(
                    "P^{} request insertion {index} belongs to P^{}",
                    self.n,
                    insertion.class.n()
                )));
            }
        }
        self.try_virtual_dimension()?;
        Ok(())
    }

    /// Checked degree-zero part of the virtual dimension.
    pub fn try_dimension_without_degree(&self) -> Result<isize, GwError> {
        self.try_canonical_theory()?.virtual_dimension(
            self.genus,
            &CurveClass::new(vec![0]),
            self.insertions.len(),
        )
    }

    pub fn dimension_without_degree(&self) -> isize {
        self.try_dimension_without_degree()
            .expect("validated projective-space request has a representable virtual dimension")
    }

    pub fn insertion_degree(&self) -> Option<usize> {
        let mut total = 0usize;
        for insertion in &self.insertions {
            total = total.checked_add(insertion.descendant_power)?;
            total = total.checked_add(insertion.class.pure_power()?)?;
        }
        Some(total)
    }

    pub fn expected_degree_from_dimension(&self) -> Option<usize> {
        let theory = self.try_canonical_theory().ok()?;
        let insertion_degree = isize::try_from(self.insertion_degree()?).ok()?;
        let dimension_without_degree = theory
            .virtual_dimension(self.genus, &CurveClass::new(vec![0]), self.insertions.len())
            .ok()?;
        let numerator = insertion_degree.checked_sub(dimension_without_degree)?;
        let denominator =
            isize::try_from(theory.c1_pairing(&CurveClass::new(vec![1])).ok()?).ok()?;
        if denominator <= 0 || numerator < 0 || numerator % denominator != 0 {
            return None;
        }
        let degree = usize::try_from(numerator / denominator).ok()?;
        let curve_degree = i64::try_from(degree).ok()?;
        (theory
            .effectivity(&CurveClass::new(vec![curve_degree]))
            .ok()?
            == CurveEffectivity::Effective)
            .then_some(degree)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesRequest {
    pub n: usize,
    pub genus: usize,
    pub degree_max: usize,
    pub max_markings: usize,
    pub max_descendant_power: usize,
    pub include_zero: bool,
    pub equivariant: bool,
    pub mode: ComputeMode,
    pub truncation: Option<Truncation>,
}

/// Finite work envelope for sparse descendant-potential enumeration.
pub const MAX_SERIES_STATE_SPACE_RANK: usize = 64;
pub const MAX_SERIES_DEGREE: usize = 64;
pub const MAX_SERIES_DESCENDANT_POWER: usize = 64;
pub const MAX_SERIES_CANDIDATE_COEFFICIENTS: usize = 100_000;
/// Work guard for the branching descendant divisor recursion used only when
/// the underlying pointed curve is unstable.  Stable graph evaluation keeps
/// the larger series descendant boundary above.
pub use crate::core::moduli::MAX_UNSTABLE_DIVISOR_RECURSION_TOTAL_PSI;

impl SeriesRequest {
    pub fn new(n: usize, genus: usize, degree_max: usize, max_markings: usize) -> Self {
        Self {
            n,
            genus,
            degree_max,
            max_markings,
            max_descendant_power: 0,
            include_zero: false,
            equivariant: false,
            mode: ComputeMode::Givental,
            truncation: None,
        }
    }

    pub fn coefficient_request(
        &self,
        degree: usize,
        insertions: Vec<Insertion>,
    ) -> InvariantRequest {
        InvariantRequest {
            n: self.n,
            genus: self.genus,
            degree,
            insertions,
            equivariant: self.equivariant,
            mode: self.mode,
            truncation: self.truncation.clone(),
        }
    }

    /// Validate all public bounds before candidate vectors or graph kernels
    /// are allocated.
    pub fn validate(&self) -> Result<(), GwError> {
        let state_space_rank = self.n.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("series target dimension overflow".to_string())
        })?;
        if state_space_rank > MAX_SERIES_STATE_SPACE_RANK {
            return Err(GwError::ResourceLimit {
                operation: "series state-space rank".to_string(),
                requested: state_space_rank,
                limit: MAX_SERIES_STATE_SPACE_RANK,
            });
        }
        if self.degree_max > MAX_SERIES_DEGREE {
            return Err(GwError::ResourceLimit {
                operation: "series degree bound".to_string(),
                requested: self.degree_max,
                limit: MAX_SERIES_DEGREE,
            });
        }
        if self.max_markings > graphs::MAX_STABLE_GRAPH_MARKINGS {
            return Err(GwError::ResourceLimit {
                operation: "series marking bound".to_string(),
                requested: self.max_markings,
                limit: graphs::MAX_STABLE_GRAPH_MARKINGS,
            });
        }
        if self.max_descendant_power > MAX_SERIES_DESCENDANT_POWER {
            return Err(GwError::ResourceLimit {
                operation: "series descendant bound".to_string(),
                requested: self.max_descendant_power,
                limit: MAX_SERIES_DESCENDANT_POWER,
            });
        }
        for markings in 0..=self.max_markings {
            if pointed_curve_is_stable(self.genus, markings) {
                graphs::stable_graph_generation_bounds(self.genus, markings)?;
            }
        }

        let descendant_count = self.max_descendant_power.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("series descendant basis overflow".to_string())
        })?;
        let basis_count = state_space_rank
            .checked_mul(descendant_count)
            .ok_or_else(|| GwError::UnsupportedInvariant("series basis overflow".to_string()))?;
        let mut profiles = 1usize;
        let mut exact_marking_profiles = 1usize;
        for markings in 1..=self.max_markings {
            let numerator = basis_count.checked_add(markings - 1).ok_or_else(|| {
                GwError::UnsupportedInvariant("series profile count overflow".to_string())
            })?;
            exact_marking_profiles = exact_marking_profiles
                .checked_mul(numerator)
                .and_then(|value| value.checked_div(markings))
                .ok_or_else(|| {
                    GwError::UnsupportedInvariant("series profile count overflow".to_string())
                })?;
            profiles = profiles
                .checked_add(exact_marking_profiles)
                .ok_or_else(|| {
                    GwError::UnsupportedInvariant("series profile count overflow".to_string())
                })?;
            if profiles > MAX_SERIES_CANDIDATE_COEFFICIENTS {
                return Err(GwError::ResourceLimit {
                    operation: "series insertion profiles".to_string(),
                    requested: profiles,
                    limit: MAX_SERIES_CANDIDATE_COEFFICIENTS,
                });
            }
        }
        let degree_count = self.degree_max.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("series degree count overflow".to_string())
        })?;
        let coefficient_bound = profiles.checked_mul(degree_count).ok_or_else(|| {
            GwError::UnsupportedInvariant("series coefficient count overflow".to_string())
        })?;
        if coefficient_bound > MAX_SERIES_CANDIDATE_COEFFICIENTS {
            return Err(GwError::ResourceLimit {
                operation: "series candidate coefficients".to_string(),
                requested: coefficient_bound,
                limit: MAX_SERIES_CANDIDATE_COEFFICIENTS,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvariantResult {
    pub value: RatFun,
    pub engine: &'static str,
    pub notes: Vec<String>,
}

impl InvariantResult {
    pub fn rational(value: impl Into<Rational>, engine: &'static str) -> Self {
        Self {
            value: RatFun::from_rational(value.into()),
            engine,
            notes: Vec::new(),
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn nonequivariant_limit_line(
        &self,
        target_n: usize,
        weights: &[Rational],
    ) -> Result<Rational, GwError> {
        self.value.nonequivariant_limit_line(target_n, weights)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesCoefficient {
    pub degree: usize,
    pub insertions: Vec<Insertion>,
    pub value: RatFun,
}

impl SeriesCoefficient {
    pub fn insertion_label(&self) -> String {
        if self.insertions.is_empty() {
            return "1".to_string();
        }
        self.insertions
            .iter()
            .map(insertion_label)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesResult {
    pub coefficients: Vec<SeriesCoefficient>,
    pub engine: &'static str,
    /// False when at least one requested coefficient was unsupported and
    /// therefore omitted from this result.
    pub complete: bool,
    pub notes: Vec<String>,
}

impl SeriesResult {
    pub fn is_complete(&self) -> bool {
        self.complete
    }
}

pub fn tau(descendant_power: usize, class: CohomologyClass) -> Insertion {
    Insertion::new(descendant_power, class)
}

pub fn compute(req: InvariantRequest) -> Result<InvariantResult, GwError> {
    req.validate()?;
    match req.mode {
        ComputeMode::Givental => compute_givental(&req),
    }
}

/// Historical `givental::compute` entry point for an already borrowed
/// projective-space request.
pub fn compute_givental(req: &InvariantRequest) -> Result<InvariantResult, GwError> {
    match compute_by_givental_graphs(req) {
        Ok(result) => Ok(result),
        Err(GwError::UnsupportedInvariant(message)) => {
            if message.contains("full quantized S-action") {
                Err(GwError::UnsupportedInvariant(message))
            } else {
                seed_compute(req, "givental-seed")
            }
        }
        Err(limit @ GwError::ResourceLimit { .. }) => match seed_compute(req, "givental-seed") {
            Ok(result) => Ok(result),
            Err(GwError::UnsupportedInvariant(_)) => Err(limit),
            Err(error) => Err(error),
        },
        Err(error) => Err(error),
    }
}

/// Ordinary `P^n` computation through the generic semisimple graph engine.
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
        givental::dimension_mismatch(&provider, req.genus, req.degree, &req.insertions)
    {
        return Ok(InvariantResult {
            value: RatFun::zero(),
            engine: "givental-r-graph",
            notes: vec![format!(
                "dimension mismatch gives zero: virtual dimension {virtual_dimension}, insertion degree {total_degree}"
            )],
        });
    }

    if !givental::is_stable_cohft_range(req.genus, req.insertions.len()) {
        return Err(GwError::UnsupportedInvariant(
            "Givental graph expansion is implemented for stable (g,n) CohFT ranges only"
                .to_string(),
        ));
    }

    if req.equivariant {
        let value = givental::compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
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

    let value = givental::compute_semisimple_graph_value(
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

    let limit = value.nonequivariant_limit_line(req.n, provider.weights())?;
    Ok(InvariantResult {
        value: RatFun::from_rational(limit),
        engine: "givental-r-graph-limit",
        notes: vec![
            "computed by truncated J-calibrated R-matrix stable-graph expansion and lambda-line nonequivariant limit"
                .to_string(),
        ],
    })
}

/// Compute one stable graph's contribution to a bounded ordinary-projective
/// descendant potential.
pub fn projective_graph_bounded_potential_coefficients(
    n: usize,
    genus: usize,
    markings: usize,
    graph_index: usize,
    degree_max: usize,
    max_descendant_power: usize,
    equivariant: bool,
) -> Result<Vec<SeriesCoefficient>, GwError> {
    let provider = ProjectiveSpaceProvider::try_new(n, equivariant)?;
    let basis = insertion_basis(n, max_descendant_power);
    let profiles = insertion_monomials(&basis, markings);
    givental::graph_bounded_potential_coefficients_with_provider(
        &provider,
        genus,
        markings,
        graph_index,
        degree_max,
        max_descendant_power,
        profiles,
    )
    .map(|coefficients| {
        coefficients
            .into_iter()
            .map(|(degree, insertions, value)| SeriesCoefficient {
                degree,
                insertions,
                value,
            })
            .collect()
    })
}

/// Projective-space wrapper around the generic batched series evaluator.
pub fn compute_series_master(req: &SeriesRequest) -> Result<Option<SeriesResult>, GwError> {
    req.validate()?;
    if req.mode != ComputeMode::Givental {
        return Ok(None);
    }
    let provider = ProjectiveSpaceProvider::try_new(req.n, req.equivariant)?;
    super::batch::compute_series_master_with_provider(req, provider)
}

pub fn compute_series(req: SeriesRequest) -> Result<SeriesResult, GwError> {
    req.validate()?;
    if req.mode == ComputeMode::Givental {
        if let Some(result) = compute_series_master(&req)? {
            return Ok(result);
        }
    }

    let mut coefficients = Vec::new();
    let mut notes = vec![
        "series enumerates a bounded sparse descendant potential; unsupported coefficients are skipped and profiles forced to vanish by the provider's coefficient-ring grading are dimension-pruned"
            .to_string(),
    ];
    let mut complete = true;
    let mut engine = "series";
    let basis = insertion_basis(req.n, req.max_descendant_power);
    let mut candidates_by_degree = vec![Vec::<Vec<Insertion>>::new(); req.degree_max + 1];

    for markings in 0..=req.max_markings {
        for insertions in insertion_monomials(&basis, markings) {
            if req.equivariant {
                for bucket in &mut candidates_by_degree {
                    bucket.push(insertions.clone());
                }
                continue;
            }
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
            let coefficient_req = req.coefficient_request(degree, insertions.clone());
            if !req.equivariant
                && coefficient_req.insertion_degree().is_some_and(|actual| {
                    usize::try_from(coefficient_req.virtual_dimension()).ok() != Some(actual)
                })
            {
                continue;
            }
            match compute(coefficient_req) {
                Ok(result) => {
                    engine = result.engine;
                    if req.include_zero || !result.value.is_zero() {
                        coefficients.push(SeriesCoefficient {
                            degree,
                            insertions,
                            value: result.value,
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

    Ok(SeriesResult {
        coefficients,
        engine,
        complete,
        notes,
    })
}

/// Projective-space wrapper around the provider-generic packed resolvent.
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
    super::batch::compute_packed_resolvent_with_provider(
        req,
        provider,
        engine,
        "computed by packed S/R external-leg graph kernel; all resolvent coefficients share one stable-graph contraction",
        Ok::<RatFun, GwError>,
    )
}

pub(crate) fn insertion_basis(n: usize, max_descendant_power: usize) -> Vec<Insertion> {
    let mut basis = Vec::new();
    for descendant_power in 0..=max_descendant_power {
        for h_power in 0..=n {
            basis.push(tau(descendant_power, CohomologyClass::h_power(n, h_power)));
        }
    }
    basis
}

pub(crate) fn insertion_monomials(basis: &[Insertion], markings: usize) -> Vec<Vec<Insertion>> {
    fn rec(
        basis: &[Insertion],
        markings: usize,
        start: usize,
        current: &mut Vec<Insertion>,
        out: &mut Vec<Vec<Insertion>>,
    ) {
        if current.len() == markings {
            out.push(current.clone());
            return;
        }
        for idx in start..basis.len() {
            current.push(basis[idx].clone());
            rec(basis, markings, idx, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(basis, markings, 0, &mut Vec::new(), &mut out);
    out
}

pub(crate) fn insertion_monomial_label(insertions: &[Insertion]) -> String {
    if insertions.is_empty() {
        "1".to_string()
    } else {
        insertions
            .iter()
            .map(insertion_label)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn insertion_label(insertion: &Insertion) -> String {
    let class = match insertion.class.pure_power() {
        Some(0) => "1".to_string(),
        Some(1) => "H".to_string(),
        Some(power) => format!("H^{power}"),
        None => "class".to_string(),
    };
    format!("tau{}({class})", insertion.descendant_power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p2_line_through_two_points_seed() {
        let insertions = vec![
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ];
        let req = InvariantRequest::new(2, 0, 1, insertions);
        let result = compute(req).unwrap();
        assert_eq!(result.value, RatFun::one());
    }

    #[test]
    fn degree_zero_classical_intersection_seed() {
        let insertions = vec![
            tau(0, CohomologyClass::h_power(2, 1)),
            tau(0, CohomologyClass::h_power(2, 1)),
            tau(0, CohomologyClass::one(2)),
        ];
        let req = InvariantRequest::new(2, 0, 0, insertions);
        let result = compute(req).unwrap();
        assert_eq!(result.value, RatFun::one());
    }

    #[test]
    fn expected_degree_is_for_fixed_homogeneous_insertions() {
        let insertions = vec![
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 2)),
            tau(0, CohomologyClass::h_power(2, 1)),
        ];
        let req = InvariantRequest::new(2, 0, 99, insertions);
        assert_eq!(req.expected_degree_from_dimension(), Some(1));
    }

    #[test]
    fn point_target_degree_inference_never_returns_positive_degree() {
        let degree_zero = InvariantRequest::new(
            0,
            0,
            99,
            vec![
                tau(0, CohomologyClass::one(0)),
                tau(0, CohomologyClass::one(0)),
                tau(0, CohomologyClass::one(0)),
            ],
        );
        assert_eq!(degree_zero.expected_degree_from_dimension(), Some(0));

        let formally_degree_one = InvariantRequest::new(
            0,
            0,
            99,
            vec![
                tau(1, CohomologyClass::one(0)),
                tau(0, CohomologyClass::one(0)),
                tau(0, CohomologyClass::one(0)),
            ],
        );
        assert_eq!(formally_degree_one.expected_degree_from_dimension(), None);
    }

    #[test]
    fn extreme_projective_request_is_rejected_without_panicking() {
        let request = InvariantRequest::new(usize::MAX, 0, 0, Vec::new());
        assert!(matches!(
            request.validate(),
            Err(GwError::UnsupportedInvariant(_))
        ));
        assert!(matches!(
            request.try_virtual_dimension(),
            Err(GwError::UnsupportedInvariant(_))
        ));
        assert_eq!(request.expected_degree_from_dimension(), None);
    }

    #[test]
    fn sparse_series_rejects_unbounded_public_bounds_before_allocation() {
        assert!(matches!(
            SeriesRequest::new(1, 0, usize::MAX, 0).validate(),
            Err(GwError::ResourceLimit {
                operation,
                requested: usize::MAX,
                limit: MAX_SERIES_DEGREE,
            }) if operation == "series degree bound"
        ));
        assert!(matches!(
            SeriesRequest::new(1, 0, 0, usize::MAX).validate(),
            Err(GwError::ResourceLimit {
                operation,
                requested: usize::MAX,
                limit: graphs::MAX_STABLE_GRAPH_MARKINGS,
            }) if operation == "series marking bound"
        ));
        assert!(matches!(
            SeriesRequest::new(usize::MAX, 0, 0, 0).validate(),
            Err(GwError::UnsupportedInvariant(_))
        ));

        let mut descendants = SeriesRequest::new(1, 0, 0, 0);
        descendants.max_descendant_power = usize::MAX;
        assert!(matches!(
            descendants.validate(),
            Err(GwError::ResourceLimit {
                operation,
                requested: usize::MAX,
                limit: MAX_SERIES_DESCENDANT_POWER,
            }) if operation == "series descendant bound"
        ));

        let mut combinatorial = SeriesRequest::new(2, 0, 0, 8);
        combinatorial.max_descendant_power = 3;
        assert!(matches!(
            compute_series(combinatorial),
            Err(GwError::ResourceLimit { .. })
        ));
    }

    #[test]
    fn primary_potential_series_finds_degree_one_plane_line_coefficient() {
        let series = compute_series(SeriesRequest::new(2, 0, 1, 3)).unwrap();
        assert!(series.coefficients.iter().any(|coefficient| {
            coefficient.degree == 1
                && coefficient.value == RatFun::one()
                && coefficient.insertion_label() == "tau0(H) tau0(H^2) tau0(H^2)"
        }));
    }

    #[test]
    fn descendant_potential_series_buckets_by_expected_degree() {
        let mut req = SeriesRequest::new(1, 0, 4, 1);
        req.max_descendant_power = 6;
        let series = compute_series(req).unwrap();
        assert!(series.coefficients.iter().any(|coefficient| {
            coefficient.degree == 4
                && coefficient.value == RatFun::from_rational(Rational::new(1, 576))
                && coefficient.insertion_label() == "tau6(H)"
        }));
    }

    #[test]
    fn partial_series_reports_incomplete_status() {
        let mut req = SeriesRequest::new(1, 1, 0, 0);
        req.include_zero = true;
        let series = compute_series(req).unwrap();
        assert!(!series.is_complete());
        assert!(series.notes.iter().any(|note| note.contains("skipped q^0")));
    }

    #[test]
    fn point_target_series_never_emits_positive_degree() {
        let mut req = SeriesRequest::new(0, 0, 2, 3);
        req.max_descendant_power = 1;
        let series = compute_series(req).unwrap();
        assert!(series.coefficients.iter().any(|coefficient| {
            coefficient.degree == 0
                && coefficient.insertion_label() == "tau0(1) tau0(1) tau0(1)"
                && coefficient.value == RatFun::one()
        }));
        assert!(series
            .coefficients
            .iter()
            .all(|coefficient| coefficient.degree == 0));
    }

    #[test]
    fn equivariant_series_keeps_excess_degree_coefficients() {
        let mut req = SeriesRequest::new(1, 0, 0, 3);
        req.equivariant = true;
        let series = compute_series(req).unwrap();
        let coefficient = series
            .coefficients
            .iter()
            .find(|coefficient| {
                coefficient.degree == 0
                    && coefficient.insertion_label() == "tau0(1) tau0(H) tau0(H)"
            })
            .expect("equivariant degree-zero H^2 coefficient should be retained");
        assert_eq!(
            coefficient
                .value
                .evaluate_lambda_weights(1, &[Rational::from(2), Rational::from(5)],)
                .unwrap(),
            Rational::from(7)
        );
    }
}
