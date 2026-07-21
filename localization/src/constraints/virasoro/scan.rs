use super::{
    evaluate_constraint_with_bounds, generate_constraint_with_term_limit,
    CanonicalCorrelatorEvaluator, CanonicalVirasoroConstraint, CorrelatorEvaluationBounds,
    Descendant, ResidualReport, ResidualStatus, TimeMonomial,
};
use crate::core::algebra::RatFun;
use crate::core::error::GwError;
use crate::core::theory::{BasisId, CurveClass};
use std::collections::BTreeSet;

/// Maximum number of external markings enumerated in a bounded scan.
///
/// A scan materializes descendant profiles and, for nonlinear constraints,
/// labelled marking partitions.  Single-equation generation has a separate,
/// larger payload cap.
pub const MAX_VIRASORO_SCAN_MARKINGS: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirasoroScanBounds {
    pub operator_min: i32,
    pub operator_max: i32,
    pub genus_max: usize,
    /// Bound in the canonical theory's effective/admissible grading.
    pub degree_grading_max: usize,
    pub markings_max: usize,
    pub descendant_max: usize,
    /// Hard guard against an accidentally enormous Cartesian product.
    pub equation_limit: usize,
    /// Per-equation upper bound checked before labelled partitions or matrix
    /// powers are materialized.
    pub generated_term_limit: usize,
    /// Aggregate number of retained AST terms across the complete scan.
    pub total_generated_term_limit: usize,
    /// Bounds for the unique correlator dependency closure of each equation.
    pub correlator_bounds: CorrelatorEvaluationBounds,
}

impl VirasoroScanBounds {
    pub fn small() -> Self {
        Self {
            operator_min: -1,
            operator_max: 1,
            genus_max: 1,
            degree_grading_max: 1,
            markings_max: 2,
            descendant_max: 1,
            equation_limit: 10_000,
            generated_term_limit: 1_000_000,
            total_generated_term_limit: 1_000_000,
            correlator_bounds: CorrelatorEvaluationBounds {
                max_genus: Some(1),
                max_markings: Some(4),
                max_descendant_power: Some(2),
                dependency_limit: 100_000,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirasoroScanEntry {
    pub constraint: CanonicalVirasoroConstraint,
    pub report: ResidualReport<CurveClass, BasisId, RatFun>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirasoroScanReport {
    bounds: VirasoroScanBounds,
    entries: Vec<VirasoroScanEntry>,
}

impl VirasoroScanReport {
    pub fn is_success(&self) -> bool {
        !self.entries.is_empty()
            && self
                .entries
                .iter()
                .all(|entry| entry.report.status() == ResidualStatus::VerifiedZero)
    }

    pub fn total(&self) -> usize {
        self.entries.len()
    }

    pub fn generated_term_count(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.constraint.terms.len())
            .sum()
    }

    pub fn bounds(&self) -> &VirasoroScanBounds {
        &self.bounds
    }

    pub fn entries(&self) -> &[VirasoroScanEntry] {
        &self.entries
    }

    pub fn verified_zero_count(&self) -> usize {
        self.status_count(ResidualStatus::VerifiedZero)
    }

    pub fn nonzero_count(&self) -> usize {
        self.status_count(ResidualStatus::Nonzero)
    }

    pub fn incomplete_count(&self) -> usize {
        self.status_count(ResidualStatus::Incomplete)
    }

    /// Equations with no terms after exact symbolic aggregation.
    pub fn vacuous_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.constraint.terms.is_empty())
            .count()
    }

    /// Non-vacuous equations closed without a backend call (only constants
    /// and/or canonical-theory-certified structural zeros).
    pub fn structural_only_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| {
                !entry.constraint.terms.is_empty()
                    && entry.report.backend_correlator_count() == 0
                    && entry.report.status() != ResidualStatus::Incomplete
            })
            .count()
    }

    /// Equations that exercised at least one backend correlator.
    pub fn backend_exercised_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.report.backend_correlator_count() > 0)
            .count()
    }

    /// Non-vacuous equations with missing dependencies and no backend value.
    pub fn unresolved_only_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| {
                !entry.constraint.terms.is_empty()
                    && entry.report.backend_correlator_count() == 0
                    && entry.report.status() == ResidualStatus::Incomplete
            })
            .count()
    }

    fn status_count(&self, status: ResidualStatus) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.report.status() == status)
            .count()
    }
}

pub fn scan_constraints(
    evaluator: &dyn CanonicalCorrelatorEvaluator,
    bounds: VirasoroScanBounds,
) -> Result<VirasoroScanReport, GwError> {
    if bounds.operator_min < -1 || bounds.operator_max < bounds.operator_min {
        return Err(GwError::ConventionMismatch(
            "Virasoro scan requires -1 <= operator_min <= operator_max".to_string(),
        ));
    }
    if bounds.markings_max > MAX_VIRASORO_SCAN_MARKINGS {
        return Err(GwError::ResourceLimit {
            operation: "Virasoro scan markings".to_string(),
            requested: bounds.markings_max,
            limit: MAX_VIRASORO_SCAN_MARKINGS,
        });
    }
    let theory = evaluator.theory();
    let operator_count_i64 = i64::from(bounds.operator_max)
        .checked_sub(i64::from(bounds.operator_min))
        .and_then(|difference| difference.checked_add(1))
        .ok_or_else(|| GwError::AlgebraFailure("operator count overflow".to_string()))?;
    let operator_count = usize::try_from(operator_count_i64)
        .map_err(|_| GwError::AlgebraFailure("operator count overflow".to_string()))?;
    let descendant_levels = bounds.descendant_max.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("descendant scan bound is too large".to_string())
    })?;
    let atom_count = descendant_levels
        .checked_mul(theory.state_space().basis.len())
        .ok_or_else(|| {
            GwError::UnsupportedInvariant("descendant atom count overflow".to_string())
        })?;
    let profile_count = multiset_profile_count(atom_count, bounds.markings_max)?;
    // The canonical theory owns its admissible cone.  Ask only for its cardinality
    // until the full Cartesian-product budget has been accepted; otherwise a
    // small equation limit would not actually protect this allocation.
    let curve_count = theory.bounded_admissible_class_count(bounds.degree_grading_max)?;
    if curve_count == 0 {
        return Err(GwError::ValidationFailure(format!(
            "theory {} omitted the degree-zero curve class from its bounded cone",
            theory.theory_id()
        )));
    }
    let projected = operator_count
        .checked_mul(
            bounds
                .genus_max
                .checked_add(1)
                .ok_or_else(|| GwError::AlgebraFailure("genus scan bound overflow".to_string()))?,
        )
        .and_then(|count| count.checked_mul(curve_count))
        .and_then(|count| count.checked_mul(profile_count))
        .ok_or_else(|| GwError::AlgebraFailure("Virasoro scan size overflow".to_string()))?;
    if projected > bounds.equation_limit {
        return Err(GwError::ResourceLimit {
            operation: "Virasoro scan equations".to_string(),
            requested: projected,
            limit: bounds.equation_limit,
        });
    }
    let curves = theory.bounded_admissible_classes(bounds.degree_grading_max)?;
    if curves.len() != curve_count {
        return Err(GwError::ValidationFailure(format!(
            "theory {} reported {curve_count} bounded curve classes but produced {}",
            theory.theory_id(),
            curves.len()
        )));
    }
    let mut unique_curves = BTreeSet::new();
    for curve in &curves {
        theory.curve_class_space().validate(curve)?;
        if !unique_curves.insert(curve.clone()) {
            return Err(GwError::ValidationFailure(format!(
                "theory {} returned a duplicate bounded curve class {curve}",
                theory.theory_id()
            )));
        }
    }
    let zero_curve = CurveClass::zero(theory.curve_class_space().rank());
    if !unique_curves.contains(&zero_curve) {
        return Err(GwError::ValidationFailure(format!(
            "theory {} omitted the degree-zero curve class from its bounded cone",
            theory.theory_id()
        )));
    }
    let profiles = if bounds.markings_max == 0 {
        vec![TimeMonomial::one()]
    } else {
        let mut atoms = Vec::new();
        atoms.try_reserve_exact(atom_count).map_err(|_| {
            GwError::UnsupportedInvariant(format!(
                "cannot allocate {atom_count} descendant variables"
            ))
        })?;
        for psi_power in 0..=bounds.descendant_max {
            for basis in 0..theory.state_space().basis.len() {
                atoms.push(Descendant::new(psi_power, BasisId(basis)));
            }
        }
        descendant_profiles(&atoms, bounds.markings_max, profile_count)?
    };
    if profiles.len() != profile_count {
        return Err(GwError::ValidationFailure(format!(
            "profile counter predicted {profile_count} monomials but generated {}",
            profiles.len()
        )));
    }

    let mut entries = Vec::new();
    entries.try_reserve_exact(projected).map_err(|_| {
        GwError::UnsupportedInvariant(format!("cannot allocate {projected} Virasoro scan entries"))
    })?;
    let mut retained_terms = 0usize;
    for operator in bounds.operator_min..=bounds.operator_max {
        for genus in 0..=bounds.genus_max {
            for curve in &curves {
                for profile in &profiles {
                    let constraint = generate_constraint_with_term_limit(
                        theory,
                        operator,
                        genus,
                        curve.clone(),
                        profile.clone(),
                        bounds.generated_term_limit,
                    )?;
                    retained_terms = retained_terms
                        .checked_add(constraint.terms.len())
                        .ok_or_else(|| {
                            GwError::UnsupportedInvariant(
                                "aggregate Virasoro scan term count overflow".to_string(),
                            )
                        })?;
                    if retained_terms > bounds.total_generated_term_limit {
                        return Err(GwError::ResourceLimit {
                            operation: "retained Virasoro scan terms".to_string(),
                            requested: retained_terms,
                            limit: bounds.total_generated_term_limit,
                        });
                    }
                    let report = evaluate_constraint_with_bounds(
                        evaluator,
                        &constraint,
                        bounds.correlator_bounds,
                    );
                    entries.push(VirasoroScanEntry { constraint, report });
                }
            }
        }
    }
    Ok(VirasoroScanReport { bounds, entries })
}

fn multiset_profile_count(atom_count: usize, markings_max: usize) -> Result<usize, GwError> {
    let mut total = 1usize;
    let mut exact = 1usize;
    for markings in 1..=markings_max {
        let numerator = atom_count
            .checked_add(markings - 1)
            .ok_or_else(|| GwError::UnsupportedInvariant("profile count overflow".to_string()))?;
        exact = exact
            .checked_mul(numerator)
            .ok_or_else(|| GwError::UnsupportedInvariant("profile count overflow".to_string()))?
            / markings;
        total = total
            .checked_add(exact)
            .ok_or_else(|| GwError::UnsupportedInvariant("profile count overflow".to_string()))?;
    }
    Ok(total)
}

fn descendant_profiles(
    atoms: &[Descendant<BasisId>],
    markings_max: usize,
    expected_count: usize,
) -> Result<Vec<TimeMonomial<BasisId>>, GwError> {
    fn collect(
        atoms: &[Descendant<BasisId>],
        remaining: usize,
        minimum: usize,
        current: &mut Vec<Descendant<BasisId>>,
        out: &mut Vec<TimeMonomial<BasisId>>,
        expected_count: usize,
    ) -> Result<(), GwError> {
        if remaining == 0 {
            if out.len() >= expected_count {
                return Err(GwError::ValidationFailure(
                    "descendant-profile generation exceeded its checked count".to_string(),
                ));
            }
            out.push(TimeMonomial::from_descendants(current.clone()));
            return Ok(());
        }
        for index in minimum..atoms.len() {
            current.push(atoms[index].clone());
            collect(atoms, remaining - 1, index, current, out, expected_count)?;
            current.pop();
        }
        Ok(())
    }

    let mut out = Vec::new();
    out.try_reserve_exact(expected_count).map_err(|_| {
        GwError::UnsupportedInvariant(format!(
            "cannot allocate {expected_count} descendant profiles"
        ))
    })?;
    out.push(TimeMonomial::one());
    for markings in 1..=markings_max {
        collect(
            atoms,
            markings,
            0,
            &mut Vec::new(),
            &mut out,
            expected_count,
        )?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::virasoro::{ProductProjectiveEvaluator, ProjectiveSpaceEvaluator};

    #[test]
    fn an_empty_scan_report_is_never_a_success() {
        let report = VirasoroScanReport {
            bounds: VirasoroScanBounds::small(),
            entries: Vec::new(),
        };
        assert!(!report.is_success());
    }

    #[test]
    fn bounded_point_scan_is_complete_and_exact() {
        let evaluator = ProjectiveSpaceEvaluator::new(0);
        let report = scan_constraints(
            &evaluator,
            VirasoroScanBounds {
                operator_min: -1,
                operator_max: 1,
                genus_max: 1,
                degree_grading_max: 0,
                markings_max: 4,
                descendant_max: 0,
                equation_limit: 100,
                generated_term_limit: 10_000,
                total_generated_term_limit: 100_000,
                correlator_bounds: CorrelatorEvaluationBounds::unbounded(),
            },
        )
        .unwrap();
        assert_eq!(report.total(), 30);
        assert_eq!(report.verified_zero_count(), 30);
        assert_eq!(report.nonzero_count(), 0);
        assert_eq!(report.incomplete_count(), 0);
        // Quadratic terms with a certified structural-zero factor no longer
        // force evaluation of their irrelevant second factor.  The scan still
        // verifies the same 30 equations exactly, with five genuinely needed
        // backend rows and the remaining 25 closed structurally.
        assert_eq!(report.backend_exercised_count(), 5);
        assert_eq!(report.structural_only_count(), 25);
        assert_eq!(report.vacuous_count(), 0);
        assert_eq!(
            report.vacuous_count()
                + report.structural_only_count()
                + report.backend_exercised_count()
                + report.unresolved_only_count(),
            report.total()
        );
        assert!(report.is_success());
    }

    #[test]
    fn equation_limit_is_checked_before_generation() {
        let evaluator = ProjectiveSpaceEvaluator::new(2);
        let mut bounds = VirasoroScanBounds::small();
        bounds.equation_limit = 1;
        let error = scan_constraints(&evaluator, bounds).unwrap_err();
        assert!(matches!(error, GwError::ResourceLimit { limit: 1, .. }));
    }

    #[test]
    fn public_scan_marking_cap_is_enforced() {
        let evaluator = ProjectiveSpaceEvaluator::new(0);
        let mut bounds = VirasoroScanBounds::small();
        bounds.markings_max = MAX_VIRASORO_SCAN_MARKINGS + 1;
        let error = scan_constraints(&evaluator, bounds).unwrap_err();
        assert!(matches!(
            error,
            GwError::ResourceLimit {
                requested,
                limit: MAX_VIRASORO_SCAN_MARKINGS,
                ..
            } if requested == MAX_VIRASORO_SCAN_MARKINGS + 1
        ));
    }

    #[test]
    fn aggregate_term_limit_bounds_retained_scan_state() {
        let evaluator = ProjectiveSpaceEvaluator::new(0);
        let mut bounds = VirasoroScanBounds::small();
        bounds.degree_grading_max = 0;
        bounds.total_generated_term_limit = 0;
        let error = scan_constraints(&evaluator, bounds).unwrap_err();
        assert!(matches!(error, GwError::ResourceLimit { limit: 0, .. }));
    }

    #[test]
    fn curve_cone_is_counted_before_a_large_product_allocation() {
        let evaluator = ProductProjectiveEvaluator::new(1, 1).unwrap();
        let error = scan_constraints(
            &evaluator,
            VirasoroScanBounds {
                operator_min: -1,
                operator_max: -1,
                genus_max: 0,
                degree_grading_max: 10_000,
                markings_max: 0,
                descendant_max: 0,
                equation_limit: 1,
                generated_term_limit: 100,
                total_generated_term_limit: 100,
                correlator_bounds: CorrelatorEvaluationBounds::unbounded(),
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GwError::ResourceLimit {
                requested: 50_015_001,
                limit: 1,
                ..
            }
        ));
    }

    #[test]
    fn irrelevant_large_degree_bound_is_valid_for_a_point() {
        let evaluator = ProjectiveSpaceEvaluator::new(0);
        let report = scan_constraints(
            &evaluator,
            VirasoroScanBounds {
                operator_min: -1,
                operator_max: -1,
                genus_max: 0,
                degree_grading_max: usize::MAX,
                markings_max: 0,
                descendant_max: 1_000_000,
                equation_limit: 1,
                generated_term_limit: 100,
                total_generated_term_limit: 100,
                correlator_bounds: CorrelatorEvaluationBounds::unbounded(),
            },
        )
        .unwrap();
        assert_eq!(report.total(), 1);
    }

    #[test]
    fn dependency_limit_is_fail_closed_and_visible_in_coverage() {
        let evaluator = ProjectiveSpaceEvaluator::new(0);
        let report = scan_constraints(
            &evaluator,
            VirasoroScanBounds {
                operator_min: 0,
                operator_max: 0,
                genus_max: 1,
                degree_grading_max: 0,
                markings_max: 0,
                descendant_max: 0,
                equation_limit: 2,
                generated_term_limit: 100,
                total_generated_term_limit: 100,
                correlator_bounds: CorrelatorEvaluationBounds {
                    dependency_limit: 0,
                    ..CorrelatorEvaluationBounds::unbounded()
                },
            },
        )
        .unwrap();
        assert!(report.incomplete_count() > 0);
        assert!(report.unresolved_only_count() > 0);
        assert!(!report.is_success());
    }
}
