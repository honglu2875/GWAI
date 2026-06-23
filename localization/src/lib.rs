//! Experimental exact computations for Gromov-Witten invariants of projective
//! space.
//!
//! The crate is intentionally staged.  The public computation path is the
//! Givental/S/R graph pipeline, while validation-only backends preserve older
//! convention checks and independent oracle comparisons.

pub mod algebra;
pub mod error;
pub mod frobenius;
pub mod geometry;
pub mod givental;
pub mod graphs;
pub mod series;
pub mod tautological;
pub mod testsuite;
pub mod twisted;
pub mod validation;
#[doc(hidden)]
pub mod validation_backends;

use algebra::RatFun;
use error::GwError;
use geometry::CohomologyClass;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeMode {
    Givental,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncation {
    pub q_degree: usize,
    pub z_order: usize,
    pub descendant_degree: usize,
    pub genus: usize,
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

    pub fn virtual_dimension(&self) -> isize {
        (1 - self.genus as isize) * (self.n as isize - 3)
            + (self.n + 1) as isize * self.degree as isize
            + self.insertions.len() as isize
    }

    pub fn dimension_without_degree(&self) -> isize {
        (1 - self.genus as isize) * (self.n as isize - 3) + self.insertions.len() as isize
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
        let insertion_degree = self.insertion_degree()? as isize;
        let numerator = insertion_degree - self.dimension_without_degree();
        let denominator = (self.n + 1) as isize;
        if numerator < 0 || numerator % denominator != 0 {
            return None;
        }
        Some((numerator / denominator) as usize)
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
    pub notes: Vec<String>,
}

pub fn tau(descendant_power: usize, class: CohomologyClass) -> Insertion {
    Insertion::new(descendant_power, class)
}

pub fn compute(req: InvariantRequest) -> Result<InvariantResult, GwError> {
    match req.mode {
        ComputeMode::Givental => givental::compute(&req),
    }
}

pub fn compute_series(req: SeriesRequest) -> Result<SeriesResult, GwError> {
    if req.mode == ComputeMode::Givental {
        if let Some(result) = givental::compute_series_master(&req)? {
            return Ok(result);
        }
    }

    let mut coefficients = Vec::new();
    let mut notes = vec![
        "series enumerates a bounded sparse descendant potential; unsupported dimension-valid coefficients are skipped"
            .to_string(),
    ];
    let mut engine = "series";
    let basis = insertion_basis(req.n, req.max_descendant_power);
    let mut candidates_by_degree = vec![Vec::<Vec<Insertion>>::new(); req.degree_max + 1];

    for markings in 0..=req.max_markings {
        for insertions in insertion_monomials(&basis, markings) {
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
            if coefficient_req
                .insertion_degree()
                .is_some_and(|actual| actual as isize != coefficient_req.virtual_dimension())
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
        notes,
    })
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
}
