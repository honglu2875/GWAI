//! Experimental exact Gromov--Witten computations for projective spaces,
//! products, projective bundles, and negative split-bundle twists, together
//! with symbolic Virasoro generation and exact compact-theory audits.
//!
//! The crate is intentionally staged.  The public computation path is the
//! Givental/S/R graph pipeline, while validation-only backends preserve older
//! convention checks and independent oracle comparisons.
//!
//! The `formula` command is the human-facing explanation path.  It renders the
//! stable-graph formula in text or TeX.  The default raw basis adds the current
//! backend-specific symbolic calibration dictionary:
//!
//! ```text
//! gw-pn formula --n 2 --g 2 --markings 1 --format tex-fragment
//! gw-pn formula --n 2 --g 2 --markings 1 --basis raw --format tex
//! gw-pn formula --n 2 --g 2 --markings 1 --twist -3 --basis raw --format tex
//! ```
//!
//! The `factored` module keeps denominator factors unexpanded and is the
//! default coefficient engine for symbolic equivariant graph contraction; the
//! expanded `RatFun` engine remains available as a fallback and validation
//! target (`GWAI_DISABLE_FACTORED_GRAPH`).

pub mod algebra;
pub mod constraints;
pub mod error;
pub mod factored;
pub mod formula;
pub mod frobenius;
mod fused;
pub mod geometry;
pub mod givental;
pub mod graphs;
pub(crate) mod reconstruction;
pub mod resolvent;
pub mod series;
pub mod symbolic;
pub mod tautological;
pub mod testsuite;
pub mod theory;
pub mod twisted;
pub mod validation;
#[doc(hidden)]
pub mod validation_backends;

use algebra::RatFun;
use error::GwError;
use geometry::CohomologyClass;
use theory::{CurveClass, CurveEffectivity, GwTheory, ProjectiveSpaceTheory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeMode {
    Givental,
}

/// Optional caller-imposed cap on the `z`-order of the `S`/`R` calibration.
///
/// The graph engine derives the order it needs from each request and returns
/// [`GwError::TruncationTooLow`] when this cap is below that.  Only the
/// `z`-order is configurable; earlier revisions carried additional fields
/// (`q_degree`, `descendant_degree`, `genus`) that were never consulted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncation {
    pub z_order: usize,
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
pub const MAX_UNSTABLE_DIVISOR_RECURSION_TOTAL_PSI: usize = 8;

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
            return Err(GwError::UnsupportedInvariant(format!(
                "series state-space rank {state_space_rank} exceeds the explicit limit {MAX_SERIES_STATE_SPACE_RANK}"
            )));
        }
        if self.degree_max > MAX_SERIES_DEGREE {
            return Err(GwError::UnsupportedInvariant(format!(
                "series degree bound {} exceeds the explicit limit {MAX_SERIES_DEGREE}",
                self.degree_max
            )));
        }
        if self.max_markings > graphs::MAX_STABLE_GRAPH_MARKINGS {
            return Err(GwError::UnsupportedInvariant(format!(
                "series marking bound {} exceeds the explicit limit {}",
                self.max_markings,
                graphs::MAX_STABLE_GRAPH_MARKINGS
            )));
        }
        if self.max_descendant_power > MAX_SERIES_DESCENDANT_POWER {
            return Err(GwError::UnsupportedInvariant(format!(
                "series descendant bound {} exceeds the explicit limit {MAX_SERIES_DESCENDANT_POWER}",
                self.max_descendant_power
            )));
        }
        for markings in 0..=self.max_markings {
            if graphs::is_stable_moduli_range(self.genus, markings) {
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
                return Err(GwError::UnsupportedInvariant(format!(
                    "series insertion-profile count exceeds the explicit coefficient limit {MAX_SERIES_CANDIDATE_COEFFICIENTS}"
                )));
            }
        }
        let degree_count = self.degree_max.checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant("series degree count overflow".to_string())
        })?;
        let coefficient_bound = profiles.checked_mul(degree_count).ok_or_else(|| {
            GwError::UnsupportedInvariant("series coefficient count overflow".to_string())
        })?;
        if coefficient_bound > MAX_SERIES_CANDIDATE_COEFFICIENTS {
            return Err(GwError::UnsupportedInvariant(format!(
                "series candidate coefficient bound {coefficient_bound} exceeds the explicit limit {MAX_SERIES_CANDIDATE_COEFFICIENTS}"
            )));
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
    pub fn rational(value: impl Into<algebra::Rational>, engine: &'static str) -> Self {
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
        weights: &[algebra::Rational],
    ) -> Result<algebra::Rational, GwError> {
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
        ComputeMode::Givental => givental::compute(&req),
    }
}

pub fn compute_series(req: SeriesRequest) -> Result<SeriesResult, GwError> {
    req.validate()?;
    if req.mode == ComputeMode::Givental {
        if let Some(result) = givental::compute_series_master(&req)? {
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

/// Crate-wide boolean environment flag.
///
/// Enabled by `1`, `true`, `yes`, `on`, or `full` (case-insensitive); unset,
/// empty, or any other value — including `0` — disables.  Every debug/tuning
/// flag in the crate goes through this helper so that `FLAG=0` never means
/// "on".
pub(crate) fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "full"
            )
        })
        .unwrap_or(false)
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
        for request in [
            SeriesRequest::new(1, 0, usize::MAX, 0),
            SeriesRequest::new(1, 0, 0, usize::MAX),
            SeriesRequest::new(usize::MAX, 0, 0, 0),
        ] {
            assert!(matches!(
                request.validate(),
                Err(GwError::UnsupportedInvariant(_))
            ));
        }

        let mut descendants = SeriesRequest::new(1, 0, 0, 0);
        descendants.max_descendant_power = usize::MAX;
        assert!(matches!(
            descendants.validate(),
            Err(GwError::UnsupportedInvariant(_))
        ));

        let mut combinatorial = SeriesRequest::new(2, 0, 0, 8);
        combinatorial.max_descendant_power = 3;
        assert!(matches!(
            compute_series(combinatorial),
            Err(GwError::UnsupportedInvariant(_))
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
                && coefficient.value == RatFun::from_rational(algebra::Rational::new(1, 576))
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
                .evaluate_lambda_weights(
                    1,
                    &[algebra::Rational::from(2), algebra::Rational::from(5)],
                )
                .unwrap(),
            algebra::Rational::from(7)
        );
    }
}
