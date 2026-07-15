use super::{
    CanonicalVirasoroConstraint, ConstraintTerm, CorrelatorKey, EvaluatedTerm, IncompleteReason,
    MissingCorrelator, ResidualReport,
};
use crate::algebra::{RatFun, Rational};
use crate::error::GwError;
use crate::geometry::CohomologyClass;
use crate::givental::{
    compute_semisimple_graph_value, reconstruct_bidegree_invariants_in_theory,
    reconstruct_bundle_invariants_in_theory, BundleInsertion, ProductInsertion,
    ProjectiveSpaceProvider,
};
use crate::tau;
use crate::theory::{
    BasisId, CurveClass, CurveEffectivity, GwTheory, ProductProjectiveTheory,
    ProjectiveBundleTheory, ProjectiveSpaceTheory,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

pub trait CanonicalCorrelatorEvaluator: Send + Sync {
    fn theory(&self) -> &dyn GwTheory;

    /// Evaluate a correlator that has already passed the canonical structural
    /// zero checks.  Unsupported coefficients must return
    /// [`GwError::UnsupportedInvariant`], never a placeholder zero.
    fn evaluate_backend(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<RatFun, GwError>;
}

/// Hard limits on the correlator dependency closure evaluated for one
/// constraint.
///
/// Bounds are checked on canonical unique dependencies before the selected
/// dependency is sent to a backend.  Dependencies are ordered by
/// [`CorrelatorKey`], so a finite `dependency_limit` deterministically retains
/// the smallest keys.  If the closure is larger, one canonical omitted witness
/// is reported as [`IncompleteReason::OutsideBounds`] and the report is marked
/// as truncated, making the residual incomplete without retaining the full
/// excluded closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorrelatorEvaluationBounds {
    pub max_genus: Option<usize>,
    pub max_markings: Option<usize>,
    /// Maximum individual psi power in a dependency.
    pub max_descendant_power: Option<usize>,
    /// Maximum number of canonical unique dependencies considered.  Use
    /// [`usize::MAX`] for no finite dependency cap.
    pub dependency_limit: usize,
}

impl CorrelatorEvaluationBounds {
    pub const fn unbounded() -> Self {
        Self {
            max_genus: None,
            max_markings: None,
            max_descendant_power: None,
            dependency_limit: usize::MAX,
        }
    }

    fn contains<D, B>(&self, correlator: &CorrelatorKey<D, B>) -> bool {
        if self
            .max_genus
            .is_some_and(|max_genus| correlator.genus > max_genus)
        {
            return false;
        }
        if self
            .max_markings
            .is_some_and(|max_markings| correlator.insertions().len() > max_markings)
        {
            return false;
        }
        if self.max_descendant_power.is_some_and(|maximum| {
            correlator
                .insertions()
                .iter()
                .any(|insertion| insertion.psi_power > maximum)
        }) {
            return false;
        }
        true
    }
}

impl Default for CorrelatorEvaluationBounds {
    fn default() -> Self {
        Self::unbounded()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CorrelatorResolution {
    Backend(RatFun),
    StructuralZero(RatFun),
}

impl CorrelatorResolution {
    fn value(&self) -> &RatFun {
        match self {
            Self::Backend(value) | Self::StructuralZero(value) => value,
        }
    }
}

/// Substitute exact backend values into a generated constraint.
///
/// Every unique dependency is resolved once.  Ineffective, dimension-mismatched,
/// and degree-zero unstable correlators are certified as structural zeros;
/// unknown effectivity is queried.  Any unsupported or failed dependency makes
/// the report `Incomplete` even when the exact partial sum happens to vanish.
pub fn evaluate_constraint(
    evaluator: &dyn CanonicalCorrelatorEvaluator,
    constraint: &CanonicalVirasoroConstraint,
) -> ResidualReport<CurveClass, BasisId, RatFun> {
    evaluate_constraint_with_bounds(
        evaluator,
        constraint,
        CorrelatorEvaluationBounds::unbounded(),
    )
}

/// Substitute exact values while enforcing an explicit dependency envelope.
///
/// This function is fail-closed: one dependency outside the envelope is
/// enough to make the report incomplete, even if every evaluated contribution
/// and the resulting partial sum vanish exactly.
pub fn evaluate_constraint_with_bounds(
    evaluator: &dyn CanonicalCorrelatorEvaluator,
    constraint: &CanonicalVirasoroConstraint,
    bounds: CorrelatorEvaluationBounds,
) -> ResidualReport<CurveClass, BasisId, RatFun> {
    let theory = evaluator.theory();
    let BoundedDependencyClosure {
        retained,
        omitted_witness,
    } = bounded_dependencies(constraint, bounds.dependency_limit);
    let dependency_closure_truncated = omitted_witness.is_some();
    if constraint.theory_fingerprint != theory.theory_fingerprint() {
        let mismatch_message = "constraint and evaluator refer to different theories";
        let mut missing_correlators = retained
            .into_iter()
            .map(|correlator| MissingCorrelator {
                correlator,
                reason: IncompleteReason::EvaluationError(mismatch_message.to_string()),
            })
            .collect::<Vec<_>>();
        if let Some(correlator) = omitted_witness {
            missing_correlators.push(MissingCorrelator {
                correlator,
                reason: IncompleteReason::OutsideBounds,
            });
        }
        let report = ResidualReport::incomplete(
            None,
            constraint.terms.len(),
            Vec::new(),
            missing_correlators,
        )
        .with_note(mismatch_message);
        return if dependency_closure_truncated {
            report.with_truncated_dependency_closure(bounds.dependency_limit)
        } else {
            report
        };
    }

    let mut values = BTreeMap::new();
    let mut backend_correlators = Vec::new();
    let mut structural_zero_correlators = Vec::new();
    for correlator in retained {
        let resolution = if bounds.contains(&correlator) {
            resolve_correlator(theory, evaluator, &correlator)
        } else {
            Err(IncompleteReason::OutsideBounds)
        };
        match &resolution {
            Ok(CorrelatorResolution::Backend(_)) => backend_correlators.push(correlator.clone()),
            Ok(CorrelatorResolution::StructuralZero(_)) => {
                structural_zero_correlators.push(correlator.clone())
            }
            Err(_) => {}
        }
        values.insert(correlator, resolution);
    }

    let mut residual = RatFun::zero();
    let mut evaluated_terms = Vec::new();
    for (term_index, term) in constraint.terms.iter().enumerate() {
        let contribution = match term {
            ConstraintTerm::Constant { coefficient, .. } => {
                Some(RatFun::from_rational(coefficient.clone()))
            }
            ConstraintTerm::Linear(term) if term.coefficient.is_zero() => Some(RatFun::zero()),
            ConstraintTerm::Linear(term) => values
                .get(&term.correlator)
                .and_then(|value| value.as_ref().ok())
                .map(|value| &RatFun::from_rational(term.coefficient.clone()) * value.value()),
            ConstraintTerm::Quadratic(term) if term.coefficient.is_zero() => Some(RatFun::zero()),
            ConstraintTerm::Quadratic(term) => values
                .get(&term.left)
                .and_then(|value| value.as_ref().ok())
                .zip(
                    values
                        .get(&term.right)
                        .and_then(|value| value.as_ref().ok()),
                )
                .map(|(left, right)| {
                    let product = left.value() * right.value();
                    &RatFun::from_rational(term.coefficient.clone()) * &product
                }),
        };
        if let Some(contribution) = contribution {
            residual = &residual + &contribution;
            evaluated_terms.push(EvaluatedTerm {
                term_index,
                exact_contribution: contribution,
            });
        }
    }

    let mut missing_correlators = values
        .into_iter()
        .filter_map(|(correlator, value)| {
            value
                .err()
                .map(|reason| MissingCorrelator { correlator, reason })
        })
        .collect::<Vec<_>>();
    if let Some(correlator) = omitted_witness {
        missing_correlators.push(MissingCorrelator {
            correlator,
            reason: IncompleteReason::OutsideBounds,
        });
    }
    let report = if !missing_correlators.is_empty() {
        ResidualReport::incomplete(
            Some(residual),
            constraint.terms.len(),
            evaluated_terms,
            missing_correlators,
        )
    } else if residual.equivalent(&RatFun::zero()) {
        ResidualReport::verified_zero(residual, constraint.terms.len(), evaluated_terms)
    } else {
        ResidualReport::nonzero(residual, constraint.terms.len(), evaluated_terms)
    }
    .with_dependency_coverage(backend_correlators, structural_zero_correlators);
    if dependency_closure_truncated {
        report.with_truncated_dependency_closure(bounds.dependency_limit)
    } else {
        report
    }
}

struct BoundedDependencyClosure {
    retained: Vec<CorrelatorKey<CurveClass, BasisId>>,
    omitted_witness: Option<CorrelatorKey<CurveClass, BasisId>>,
}

fn dependency_refs<'a>(
    constraint: &'a CanonicalVirasoroConstraint,
) -> impl Iterator<Item = &'a CorrelatorKey<CurveClass, BasisId>> + 'a {
    constraint.terms.iter().flat_map(|term| {
        let dependencies = match term {
            ConstraintTerm::Constant { .. } => [None, None],
            ConstraintTerm::Linear(term) if term.coefficient.is_zero() => [None, None],
            ConstraintTerm::Linear(term) => [Some(&term.correlator), None],
            ConstraintTerm::Quadratic(term) if term.coefficient.is_zero() => [None, None],
            ConstraintTerm::Quadratic(term) => [Some(&term.left), Some(&term.right)],
        };
        dependencies.into_iter().flatten()
    })
}

/// Retain exactly the `limit` smallest unique canonical keys, then make a
/// second pass to retain the smallest key that was actually omitted.  The
/// extraction result therefore owns at most `limit + 1` dependency keys
/// regardless of the full closure size (with the usual unbounded behavior at
/// `usize::MAX`).
fn bounded_dependencies(
    constraint: &CanonicalVirasoroConstraint,
    limit: usize,
) -> BoundedDependencyClosure {
    let mut retained = BTreeSet::new();
    for dependency in dependency_refs(constraint) {
        if retained.contains(dependency) {
            continue;
        }
        if retained.len() < limit {
            retained.insert(dependency.clone());
            continue;
        }
        if retained.last().is_some_and(|largest| dependency < largest) {
            retained.pop_last();
            retained.insert(dependency.clone());
        }
    }
    let omitted_witness = dependency_refs(constraint)
        .filter(|dependency| !retained.contains(*dependency))
        .min()
        .cloned();
    BoundedDependencyClosure {
        retained: retained.into_iter().collect(),
        omitted_witness,
    }
}

fn resolve_correlator(
    theory: &dyn GwTheory,
    evaluator: &dyn CanonicalCorrelatorEvaluator,
    correlator: &CorrelatorKey<CurveClass, BasisId>,
) -> Result<CorrelatorResolution, IncompleteReason> {
    theory
        .curve_class_space()
        .validate(&correlator.degree)
        .map_err(|error| IncompleteReason::EvaluationError(error.to_string()))?;
    // Validate every public key before any mathematical zero shortcut.  A
    // malformed class must never pass merely because its curve is ineffective
    // or its constant-map sector is unstable.
    let mut insertion_degree = 0usize;
    for insertion in correlator.insertions() {
        let basis = theory
            .state_space()
            .element(insertion.class)
            .ok_or_else(|| {
                IncompleteReason::EvaluationError(format!(
                    "unknown basis id {} in correlator",
                    insertion.class.0
                ))
            })?;
        insertion_degree = insertion_degree
            .checked_add(insertion.psi_power)
            .and_then(|degree| degree.checked_add(basis.complex_codimension))
            .ok_or_else(|| {
                IncompleteReason::EvaluationError("insertion degree overflow".to_string())
            })?;
    }
    match theory
        .effectivity(&correlator.degree)
        .map_err(|error| IncompleteReason::EvaluationError(error.to_string()))?
    {
        CurveEffectivity::Ineffective => {
            return Ok(CorrelatorResolution::StructuralZero(RatFun::zero()))
        }
        CurveEffectivity::Effective | CurveEffectivity::Unknown => {}
    }

    // The connected potential excludes unstable constant maps.  Their
    // contribution is present only through explicit operator corrections.
    if correlator.degree.is_zero()
        && correlator
            .genus
            .checked_mul(2)
            .and_then(|twice_genus| twice_genus.checked_add(correlator.insertions().len()))
            .is_some_and(|stability| stability <= 2)
    {
        return Ok(CorrelatorResolution::StructuralZero(RatFun::zero()));
    }

    let virtual_dimension = theory
        .virtual_dimension(
            correlator.genus,
            &correlator.degree,
            correlator.insertions().len(),
        )
        .map_err(|error| IncompleteReason::EvaluationError(error.to_string()))?;
    if virtual_dimension < 0 || usize::try_from(virtual_dimension).ok() != Some(insertion_degree) {
        return Ok(CorrelatorResolution::StructuralZero(RatFun::zero()));
    }

    evaluator
        .evaluate_backend(correlator)
        .map(CorrelatorResolution::Backend)
        .map_err(|error| match error {
            GwError::UnsupportedInvariant(message) => IncompleteReason::Unsupported(message),
            other => IncompleteReason::EvaluationError(other.to_string()),
        })
}

pub struct ProjectiveSpaceEvaluator {
    provider: ProjectiveSpaceProvider,
    cache: Mutex<BTreeMap<CorrelatorKey<CurveClass, BasisId>, RatFun>>,
}

impl ProjectiveSpaceEvaluator {
    pub fn new(n: usize) -> Self {
        Self::try_new(n).expect("projective-space evaluator construction failed")
    }

    pub fn try_new(n: usize) -> Result<Self, GwError> {
        Ok(Self {
            provider: ProjectiveSpaceProvider::try_new(n, false)?,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn projective_theory(&self) -> &ProjectiveSpaceTheory {
        self.provider.canonical_theory()
    }

    pub fn provider(&self) -> &ProjectiveSpaceProvider {
        &self.provider
    }
}

impl CanonicalCorrelatorEvaluator for ProjectiveSpaceEvaluator {
    fn theory(&self) -> &dyn GwTheory {
        self.projective_theory()
    }

    fn evaluate_backend(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<RatFun, GwError> {
        if let Some(value) = self.cache.lock().unwrap().get(correlator).cloned() {
            return Ok(value);
        }
        let theory = self.projective_theory();
        let degree = theory.degree(&correlator.degree).ok_or_else(|| {
            GwError::ConventionMismatch("invalid projective curve class".to_string())
        })?;
        let insertions = correlator
            .insertions()
            .iter()
            .map(|insertion| {
                CohomologyClass::try_h_power(theory.n(), insertion.class.0)
                    .map(|class| tau(insertion.psi_power, class))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.provider.validate_insertions(&insertions)?;
        let value = compute_semisimple_graph_value(
            &self.provider,
            correlator.genus,
            degree,
            &insertions,
            None,
        )?;
        self.cache
            .lock()
            .unwrap()
            .insert(correlator.clone(), value.clone());
        Ok(value)
    }
}

pub struct ProductProjectiveEvaluator {
    theory: ProductProjectiveTheory,
    weights_x: Vec<Rational>,
    weights_y: Vec<Rational>,
    cache: Mutex<BTreeMap<CorrelatorKey<CurveClass, BasisId>, RatFun>>,
}

impl ProductProjectiveEvaluator {
    pub fn new(n: usize, m: usize) -> Result<Self, GwError> {
        let theory = ProductProjectiveTheory::new(n, m)?;
        let weights_x = (0..=n).map(|index| Rational::from(index + 1)).collect();
        let weight_stride = n.checked_add(2).ok_or_else(|| {
            GwError::UnsupportedInvariant("product default weight overflow".to_string())
        })?;
        let weights_y = (0..=m)
            .map(|index| {
                weight_stride
                    .checked_mul(index + 1)
                    .map(Rational::from)
                    .ok_or_else(|| {
                        GwError::UnsupportedInvariant("product default weight overflow".to_string())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::with_weights(theory, weights_x, weights_y)
    }

    pub fn with_weights(
        theory: ProductProjectiveTheory,
        weights_x: Vec<Rational>,
        weights_y: Vec<Rational>,
    ) -> Result<Self, GwError> {
        let (n, m) = theory.dimensions();
        if weights_x.len() != n + 1 || weights_y.len() != m + 1 {
            return Err(GwError::ConventionMismatch(
                "product evaluator weights do not match its canonical theory".to_string(),
            ));
        }
        Ok(Self {
            theory,
            weights_x,
            weights_y,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn product_theory(&self) -> &ProductProjectiveTheory {
        &self.theory
    }
}

impl CanonicalCorrelatorEvaluator for ProductProjectiveEvaluator {
    fn theory(&self) -> &dyn GwTheory {
        &self.theory
    }

    fn evaluate_backend(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<RatFun, GwError> {
        if let Some(value) = self.cache.lock().unwrap().get(correlator).cloned() {
            return Ok(value);
        }
        let (d1, d2) = self
            .theory
            .bidegree(&correlator.degree)
            .ok_or_else(|| GwError::ConventionMismatch("invalid product bidegree".to_string()))?;
        let insertions = correlator
            .insertions()
            .iter()
            .map(|insertion| {
                let (h1, h2) = self.theory.basis_powers(insertion.class).ok_or_else(|| {
                    GwError::ConventionMismatch("invalid product basis id".to_string())
                })?;
                Ok(ProductInsertion::new(insertion.psi_power, h1, h2))
            })
            .collect::<Result<Vec<_>, GwError>>()?;
        let values = reconstruct_bidegree_invariants_in_theory(
            &self.theory,
            &self.weights_x,
            &self.weights_y,
            correlator.genus,
            d1.checked_add(d2)
                .ok_or_else(|| GwError::AlgebraFailure("total bidegree overflow".to_string()))?,
            &insertions,
        )?;
        let value = values.get(d2).cloned().ok_or_else(|| {
            GwError::AlgebraFailure("product reconstruction omitted requested bidegree".to_string())
        })?;
        let value = RatFun::from_rational(value);
        self.cache
            .lock()
            .unwrap()
            .insert(correlator.clone(), value.clone());
        Ok(value)
    }
}

pub struct ProjectiveBundleEvaluator {
    theory: ProjectiveBundleTheory,
    weights_base: Vec<Rational>,
    weights_fiber: Vec<Rational>,
    cache: Mutex<BTreeMap<CorrelatorKey<CurveClass, BasisId>, RatFun>>,
}

impl ProjectiveBundleEvaluator {
    pub fn new(theory: ProjectiveBundleTheory) -> Result<Self, GwError> {
        let n = theory.base_dimension();
        let weights_base = (0..=n).map(|index| Rational::from(index + 1)).collect();
        let max_twist = *theory.twists().iter().max().expect("nonempty twists");
        // Put the grading seeds belonging to distinct bundle summands in
        // disjoint intervals.  This remains generic after any permutation of
        // the normalized twists, unlike a stride that mixes the twist and
        // summand indices additively.
        let fiber_stride = max_twist
            .checked_add(1)
            .and_then(|value| value.checked_mul(n.checked_add(1)?))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                GwError::UnsupportedInvariant("bundle default weight overflow".to_string())
            })?;
        let weights_fiber = theory
            .twists()
            .iter()
            .enumerate()
            .map(|(index, _)| {
                fiber_stride
                    .checked_mul(index)
                    .map(Rational::from)
                    .ok_or_else(|| {
                        GwError::UnsupportedInvariant("bundle default weight overflow".to_string())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::with_weights(theory, weights_base, weights_fiber)
    }

    pub fn with_weights(
        theory: ProjectiveBundleTheory,
        weights_base: Vec<Rational>,
        weights_fiber: Vec<Rational>,
    ) -> Result<Self, GwError> {
        if weights_base.len() != theory.base_dimension() + 1 || weights_fiber.len() != theory.rank()
        {
            return Err(GwError::ConventionMismatch(
                "bundle evaluator weights do not match its canonical theory".to_string(),
            ));
        }
        Ok(Self {
            theory,
            weights_base,
            weights_fiber,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn bundle_theory(&self) -> &ProjectiveBundleTheory {
        &self.theory
    }
}

impl CanonicalCorrelatorEvaluator for ProjectiveBundleEvaluator {
    fn theory(&self) -> &dyn GwTheory {
        &self.theory
    }

    fn evaluate_backend(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<RatFun, GwError> {
        if let Some(value) = self.cache.lock().unwrap().get(correlator).cloned() {
            return Ok(value);
        }
        let (d1, d2) = self
            .theory
            .bidegree(&correlator.degree)
            .ok_or_else(|| GwError::ConventionMismatch("invalid bundle bidegree".to_string()))?;
        let (_, shifted) = self
            .theory
            .shifted_bidegree(&correlator.degree)
            .ok_or_else(|| {
                GwError::ConventionMismatch(
                    "bundle bidegree is outside the shifted cone".to_string(),
                )
            })?;
        let total = d1
            .checked_add(shifted)
            .ok_or_else(|| GwError::AlgebraFailure("bundle total degree overflow".to_string()))?;
        let insertions = correlator
            .insertions()
            .iter()
            .map(|insertion| {
                let (h, xi) = self.theory.basis_powers(insertion.class).ok_or_else(|| {
                    GwError::ConventionMismatch("invalid bundle basis id".to_string())
                })?;
                Ok(BundleInsertion::new(insertion.psi_power, h, xi))
            })
            .collect::<Result<Vec<_>, GwError>>()?;
        let values = reconstruct_bundle_invariants_in_theory(
            &self.theory,
            &self.weights_base,
            &self.weights_fiber,
            correlator.genus,
            total,
            &insertions,
        )?;
        let requested_d2 = isize::try_from(d2).map_err(|_| {
            GwError::AlgebraFailure("bundle fiber degree does not fit in isize".to_string())
        })?;
        let value = values
            .into_iter()
            .find(|(candidate_d1, candidate_d2, _)| {
                *candidate_d1 == d1 && *candidate_d2 == requested_d2
            })
            .map(|(_, _, value)| value)
            .ok_or_else(|| {
                GwError::AlgebraFailure(
                    "bundle reconstruction omitted requested bidegree".to_string(),
                )
            })?;
        let value = RatFun::from_rational(value);
        self.cache
            .lock()
            .unwrap()
            .insert(correlator.clone(), value.clone());
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::virasoro::{
        generate_constraint, Descendant, LinearTerm, ResidualStatus, TermOrigin, TimeMonomial,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn point_constraint(
        operator: i32,
        genus: usize,
        insertions: impl IntoIterator<Item = Descendant<BasisId>>,
    ) -> (ProjectiveSpaceEvaluator, CanonicalVirasoroConstraint) {
        let evaluator = ProjectiveSpaceEvaluator::new(0);
        let constraint = generate_constraint(
            evaluator.projective_theory(),
            operator,
            genus,
            evaluator.projective_theory().curve(0),
            TimeMonomial::from_descendants(insertions),
        )
        .unwrap();
        (evaluator, constraint)
    }

    #[test]
    fn point_string_constraint_is_verified_exactly() {
        let (evaluator, constraint) = point_constraint(
            -1,
            0,
            [
                Descendant::new(0, BasisId(0)),
                Descendant::new(0, BasisId(0)),
            ],
        );
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::VerifiedZero);
    }

    #[test]
    fn projective_evaluator_reuses_the_provider_theory() {
        let evaluator = ProjectiveSpaceEvaluator::new(2);
        assert!(std::ptr::eq(
            evaluator.projective_theory(),
            evaluator.provider().canonical_theory()
        ));
        assert_eq!(
            evaluator.theory().theory_fingerprint(),
            evaluator.provider().canonical_theory().theory_fingerprint()
        );
    }

    #[test]
    fn point_l0_anomaly_constraint_is_verified_exactly() {
        let (evaluator, constraint) = point_constraint(0, 1, []);
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::VerifiedZero);
        assert_eq!(report.backend_correlator_count(), 1);
        assert_eq!(report.structural_zero_correlator_count(), 0);
    }

    #[test]
    fn unbounded_entry_points_are_equivalent() {
        let (evaluator, constraint) = point_constraint(0, 1, []);
        assert_eq!(
            evaluate_constraint(&evaluator, &constraint),
            evaluate_constraint_with_bounds(
                &evaluator,
                &constraint,
                CorrelatorEvaluationBounds::default(),
            )
        );
    }

    #[test]
    fn report_distinguishes_backend_values_from_structural_zeros() {
        let (evaluator, mut constraint) = point_constraint(0, 1, []);
        let unstable = CorrelatorKey::new(0, evaluator.projective_theory().curve(0), Vec::new());
        constraint
            .terms
            .push(ConstraintTerm::Linear(LinearTerm::new(
                Rational::one(),
                unstable.clone(),
                TermOrigin::Other("coverage test".to_string()),
            )));

        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::VerifiedZero);
        assert_eq!(report.backend_correlator_count(), 1);
        assert_eq!(report.structural_zero_correlators(), &[unstable]);
        assert_eq!(report.resolved_correlator_count(), 2);
    }

    #[test]
    fn zero_coefficient_terms_do_not_create_dependencies() {
        let (evaluator, mut constraint) = point_constraint(0, 1, []);
        let ignored = CorrelatorKey::new(
            usize::MAX,
            evaluator.projective_theory().curve(0),
            vec![Descendant::new(usize::MAX, BasisId(0))],
        );
        constraint
            .terms
            .push(ConstraintTerm::Linear(LinearTerm::new(
                Rational::zero(),
                ignored.clone(),
                TermOrigin::Other("zero coefficient".to_string()),
            )));
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::VerifiedZero);
        assert!(!report.backend_correlators().contains(&ignored));
        assert!(!report.structural_zero_correlators().contains(&ignored));
        assert_eq!(report.evaluated_term_count(), report.total_term_count());
    }

    struct CountingEvaluator {
        theory: ProjectiveSpaceTheory,
        backend_calls: AtomicUsize,
    }

    impl CanonicalCorrelatorEvaluator for CountingEvaluator {
        fn theory(&self) -> &dyn GwTheory {
            &self.theory
        }

        fn evaluate_backend(
            &self,
            _correlator: &CorrelatorKey<CurveClass, BasisId>,
        ) -> Result<RatFun, GwError> {
            self.backend_calls.fetch_add(1, Ordering::SeqCst);
            Ok(RatFun::zero())
        }
    }

    fn point_constraint_with_dependencies(
        theory: &ProjectiveSpaceTheory,
        genera: impl IntoIterator<Item = usize>,
    ) -> CanonicalVirasoroConstraint {
        let mut constraint =
            generate_constraint(theory, 0, 1, theory.curve(0), TimeMonomial::one()).unwrap();
        constraint.terms = genera
            .into_iter()
            .map(|genus| {
                let psi_power = genus
                    .checked_mul(3)
                    .and_then(|value| value.checked_sub(2))
                    .expect("test genus has a stable one-point dimension");
                ConstraintTerm::Linear(LinearTerm::new(
                    Rational::one(),
                    CorrelatorKey::new(
                        genus,
                        theory.curve(0),
                        vec![Descendant::new(psi_power, BasisId(0))],
                    ),
                    TermOrigin::Other("bounded dependency test".to_string()),
                ))
            })
            .collect();
        constraint
    }

    #[test]
    fn genus_bound_is_enforced_before_backend_calls() {
        let evaluator = CountingEvaluator {
            theory: ProjectiveSpaceTheory::new(0),
            backend_calls: AtomicUsize::new(0),
        };
        let constraint = generate_constraint(
            &evaluator.theory,
            0,
            1,
            evaluator.theory.curve(0),
            TimeMonomial::one(),
        )
        .unwrap();
        let report = evaluate_constraint_with_bounds(
            &evaluator,
            &constraint,
            CorrelatorEvaluationBounds {
                max_genus: Some(0),
                ..CorrelatorEvaluationBounds::unbounded()
            },
        );

        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert_eq!(evaluator.backend_calls.load(Ordering::SeqCst), 0);
        assert_eq!(report.backend_correlator_count(), 0);
        assert_eq!(report.structural_zero_correlator_count(), 0);
        assert_eq!(report.missing_correlator_count(), 1);
        assert_eq!(
            report.missing_correlators()[0].reason,
            IncompleteReason::OutsideBounds
        );
        assert_eq!(report.exact_residual(), None);
    }

    #[test]
    fn marking_and_descendant_bounds_block_only_excluded_backend_keys() {
        let evaluator = CountingEvaluator {
            theory: ProjectiveSpaceTheory::new(0),
            backend_calls: AtomicUsize::new(0),
        };
        let mut constraint = generate_constraint(
            &evaluator.theory,
            0,
            1,
            evaluator.theory.curve(0),
            TimeMonomial::one(),
        )
        .unwrap();
        let three_primaries = CorrelatorKey::new(
            0,
            evaluator.theory.curve(0),
            vec![Descendant::new(0, BasisId(0)); 3],
        );
        let mut four_with_descendant = vec![Descendant::new(0, BasisId(0)); 3];
        four_with_descendant.push(Descendant::new(1, BasisId(0)));
        let four_with_descendant =
            CorrelatorKey::new(0, evaluator.theory.curve(0), four_with_descendant);
        for correlator in [&three_primaries, &four_with_descendant] {
            constraint
                .terms
                .push(ConstraintTerm::Linear(LinearTerm::new(
                    Rational::one(),
                    correlator.clone(),
                    TermOrigin::Other("per-dependency bound test".to_string()),
                )));
        }

        let marking_bounded = evaluate_constraint_with_bounds(
            &evaluator,
            &constraint,
            CorrelatorEvaluationBounds {
                max_markings: Some(2),
                ..CorrelatorEvaluationBounds::unbounded()
            },
        );
        assert_eq!(evaluator.backend_calls.load(Ordering::SeqCst), 1);
        assert_eq!(marking_bounded.missing_correlator_count(), 2);
        assert!(marking_bounded
            .missing_correlators()
            .iter()
            .all(|missing| missing.reason == IncompleteReason::OutsideBounds));

        evaluator.backend_calls.store(0, Ordering::SeqCst);
        let descendant_bounded = evaluate_constraint_with_bounds(
            &evaluator,
            &constraint,
            CorrelatorEvaluationBounds {
                max_descendant_power: Some(0),
                ..CorrelatorEvaluationBounds::unbounded()
            },
        );
        assert_eq!(
            evaluator.backend_calls.load(Ordering::SeqCst),
            descendant_bounded.backend_correlator_count()
        );
        assert!(descendant_bounded
            .backend_correlators()
            .contains(&three_primaries));
        assert!(!descendant_bounded
            .backend_correlators()
            .contains(&four_with_descendant));
        assert_eq!(descendant_bounded.missing_correlator_count(), 2);
        assert!(descendant_bounded
            .missing_correlators()
            .iter()
            .any(|missing| {
                missing.correlator == four_with_descendant
                    && missing.reason == IncompleteReason::OutsideBounds
            }));
        assert!(descendant_bounded
            .missing_correlators()
            .iter()
            .all(|missing| missing.reason == IncompleteReason::OutsideBounds));
    }

    #[test]
    fn zero_dependency_limit_is_explicitly_incomplete() {
        let evaluator = CountingEvaluator {
            theory: ProjectiveSpaceTheory::new(0),
            backend_calls: AtomicUsize::new(0),
        };
        let constraint = generate_constraint(
            &evaluator.theory,
            0,
            1,
            evaluator.theory.curve(0),
            TimeMonomial::one(),
        )
        .unwrap();
        let report = evaluate_constraint_with_bounds(
            &evaluator,
            &constraint,
            CorrelatorEvaluationBounds {
                dependency_limit: 0,
                ..CorrelatorEvaluationBounds::unbounded()
            },
        );

        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert_eq!(evaluator.backend_calls.load(Ordering::SeqCst), 0);
        assert!(report.dependency_closure_truncated());
        assert_eq!(report.missing_correlator_count(), 1);
        assert_eq!(report.dependency_count(), 1);
        assert!(report
            .missing_correlators()
            .iter()
            .all(|missing| missing.reason == IncompleteReason::OutsideBounds));
        assert!(report
            .notes()
            .iter()
            .any(|note| note.contains("dependency closure truncated at 0")));
    }

    #[test]
    fn zero_dependency_limit_does_not_truncate_a_constant_only_equation() {
        let (evaluator, mut constraint) = point_constraint(0, 1, []);
        constraint.terms = vec![ConstraintTerm::Constant {
            coefficient: Rational::one(),
            origin: TermOrigin::Other("constant-only dependency test".to_string()),
        }];

        let report = evaluate_constraint_with_bounds(
            &evaluator,
            &constraint,
            CorrelatorEvaluationBounds {
                dependency_limit: 0,
                ..CorrelatorEvaluationBounds::unbounded()
            },
        );

        assert_eq!(report.status(), ResidualStatus::Nonzero);
        assert_eq!(report.evaluated_term_count(), 1);
        assert_eq!(report.dependency_count(), 0);
        assert_eq!(report.missing_correlator_count(), 0);
        assert!(!report.dependency_closure_truncated());
    }

    #[test]
    fn zero_dependency_limit_keeps_one_witness_for_a_large_closure() {
        const DEPENDENCY_COUNT: usize = 20_000;
        let evaluator = CountingEvaluator {
            theory: ProjectiveSpaceTheory::new(0),
            backend_calls: AtomicUsize::new(0),
        };
        let constraint =
            point_constraint_with_dependencies(&evaluator.theory, (1..=DEPENDENCY_COUNT).rev());

        let report = evaluate_constraint_with_bounds(
            &evaluator,
            &constraint,
            CorrelatorEvaluationBounds {
                dependency_limit: 0,
                ..CorrelatorEvaluationBounds::unbounded()
            },
        );

        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert_eq!(evaluator.backend_calls.load(Ordering::SeqCst), 0);
        assert_eq!(report.evaluated_term_count(), 0);
        assert_eq!(report.total_term_count(), DEPENDENCY_COUNT);
        assert_eq!(report.missing_correlator_count(), 1);
        assert_eq!(report.dependency_count(), 1);
        assert!(report.dependency_closure_truncated());
        let witness = &report.missing_correlators()[0];
        assert_eq!(witness.reason, IncompleteReason::OutsideBounds);
        assert_eq!(witness.correlator.genus, 1);
    }

    #[test]
    fn dependency_limit_evaluates_the_smallest_keys_and_keeps_the_next_witness() {
        let evaluator = CountingEvaluator {
            theory: ProjectiveSpaceTheory::new(0),
            backend_calls: AtomicUsize::new(0),
        };
        let constraint = point_constraint_with_dependencies(&evaluator.theory, [4, 2, 3, 1, 2]);

        let report = evaluate_constraint_with_bounds(
            &evaluator,
            &constraint,
            CorrelatorEvaluationBounds {
                dependency_limit: 2,
                ..CorrelatorEvaluationBounds::unbounded()
            },
        );

        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert_eq!(evaluator.backend_calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            report
                .backend_correlators()
                .iter()
                .map(|correlator| correlator.genus)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(report.missing_correlator_count(), 1);
        assert_eq!(report.missing_correlators()[0].correlator.genus, 3);
        assert_eq!(
            report.missing_correlators()[0].reason,
            IncompleteReason::OutsideBounds
        );
        assert!(report.dependency_closure_truncated());
        assert_eq!(report.dependency_count(), 3);
    }

    #[test]
    fn fingerprint_mismatch_uses_the_same_bounded_dependency_diagnostics() {
        let evaluator = CountingEvaluator {
            theory: ProjectiveSpaceTheory::new(0),
            backend_calls: AtomicUsize::new(0),
        };
        let mut constraint = point_constraint_with_dependencies(&evaluator.theory, [4, 2, 3, 1]);
        constraint.theory_fingerprint = "deliberately different theory".to_string();

        let report = evaluate_constraint_with_bounds(
            &evaluator,
            &constraint,
            CorrelatorEvaluationBounds {
                dependency_limit: 2,
                ..CorrelatorEvaluationBounds::unbounded()
            },
        );

        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert_eq!(evaluator.backend_calls.load(Ordering::SeqCst), 0);
        assert_eq!(report.missing_correlator_count(), 3);
        assert_eq!(
            report
                .missing_correlators()
                .iter()
                .map(|missing| missing.correlator.genus)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(report.missing_correlators()[..2]
            .iter()
            .all(|missing| { matches!(missing.reason, IncompleteReason::EvaluationError(_)) }));
        assert_eq!(
            report.missing_correlators()[2].reason,
            IncompleteReason::OutsideBounds
        );
        assert!(report.dependency_closure_truncated());
        assert!(report
            .notes()
            .iter()
            .any(|note| note.contains("different theories")));
    }

    #[test]
    fn outside_bounds_takes_precedence_over_structural_zero_proofs() {
        let (evaluator, mut constraint) = point_constraint(0, 1, []);
        // This genus-two vacuum has virtual dimension three and is therefore
        // a structural zero when evaluated without bounds.
        let high_genus_zero =
            CorrelatorKey::new(2, evaluator.projective_theory().curve(0), Vec::new());
        constraint
            .terms
            .push(ConstraintTerm::Linear(LinearTerm::new(
                Rational::one(),
                high_genus_zero.clone(),
                TermOrigin::Other("bounded structural-zero test".to_string()),
            )));
        let unbounded = evaluate_constraint(&evaluator, &constraint);
        assert!(unbounded
            .structural_zero_correlators()
            .contains(&high_genus_zero));

        let bounded = evaluate_constraint_with_bounds(
            &evaluator,
            &constraint,
            CorrelatorEvaluationBounds {
                max_genus: Some(1),
                ..CorrelatorEvaluationBounds::unbounded()
            },
        );
        assert_eq!(bounded.status(), ResidualStatus::Incomplete);
        assert!(!bounded
            .structural_zero_correlators()
            .contains(&high_genus_zero));
        assert!(bounded.missing_correlators().iter().any(|missing| {
            missing.correlator == high_genus_zero
                && missing.reason == IncompleteReason::OutsideBounds
        }));
    }

    #[test]
    fn point_l1_checks_labelled_partitions_and_genus_reduction() {
        let four_primaries = std::iter::repeat_n(Descendant::new(0, BasisId(0)), 4);
        let (evaluator, constraint) = point_constraint(1, 0, four_primaries);
        assert_eq!(
            evaluate_constraint(&evaluator, &constraint).status(),
            ResidualStatus::VerifiedZero
        );

        let (evaluator, constraint) = point_constraint(1, 1, [Descendant::new(0, BasisId(0))]);
        assert_eq!(
            evaluate_constraint(&evaluator, &constraint).status(),
            ResidualStatus::VerifiedZero
        );
    }

    #[test]
    fn p1_genus_one_l2_uses_the_grading_before_metric_index_raising() {
        let evaluator = ProjectiveSpaceEvaluator::new(1);
        let constraint = generate_constraint(
            evaluator.projective_theory(),
            2,
            1,
            evaluator.projective_theory().curve(0),
            TimeMonomial::from_descendants([
                Descendant::new(0, BasisId(0)),
                Descendant::new(0, BasisId(0)),
            ]),
        )
        .unwrap();
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::VerifiedZero);
        assert_eq!(report.exact_residual(), Some(&RatFun::zero()));
        assert_eq!(report.missing_correlator_count(), 0);
    }

    #[test]
    fn p2_genus_one_l2_uses_the_grading_before_metric_index_raising() {
        let evaluator = ProjectiveSpaceEvaluator::new(2);
        let constraint = generate_constraint(
            evaluator.projective_theory(),
            2,
            1,
            evaluator.projective_theory().curve(0),
            TimeMonomial::from_descendants([
                Descendant::new(0, BasisId(0)),
                Descendant::new(0, BasisId(0)),
            ]),
        )
        .unwrap();
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::VerifiedZero);
        assert_eq!(report.exact_residual(), Some(&RatFun::zero()));
        assert_eq!(report.missing_correlator_count(), 0);
    }

    #[test]
    fn p1_degree_one_l1_exposes_backend_gaps_without_false_pass() {
        let evaluator = ProjectiveSpaceEvaluator::new(1);
        let constraint = generate_constraint(
            evaluator.projective_theory(),
            1,
            0,
            evaluator.projective_theory().curve(1),
            TimeMonomial::from_descendants([Descendant::new(0, BasisId(0))]),
        )
        .unwrap();
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert_eq!(report.missing_correlator_count(), 2);

        struct GoldenP1 {
            theory: ProjectiveSpaceTheory,
        }
        impl CanonicalCorrelatorEvaluator for GoldenP1 {
            fn theory(&self) -> &dyn GwTheory {
                &self.theory
            }

            fn evaluate_backend(
                &self,
                correlator: &CorrelatorKey<CurveClass, BasisId>,
            ) -> Result<RatFun, GwError> {
                let insertions = correlator.insertions();
                let value = match insertions {
                    [Descendant {
                        psi_power: 0,
                        class: BasisId(0),
                    }, Descendant {
                        psi_power: 2,
                        class: BasisId(0),
                    }] => Rational::from(-2),
                    [Descendant {
                        psi_power: 0,
                        class: BasisId(0),
                    }, Descendant {
                        psi_power: 1,
                        class: BasisId(1),
                    }] => Rational::one(),
                    [Descendant {
                        psi_power: 0,
                        class: BasisId(1),
                    }] => Rational::one(),
                    _ => {
                        return Err(GwError::UnsupportedInvariant(
                            "not in the P1 golden oracle".to_string(),
                        ))
                    }
                };
                Ok(RatFun::from_rational(value))
            }
        }
        let golden = GoldenP1 {
            theory: ProjectiveSpaceTheory::new(1),
        };
        assert_eq!(
            evaluate_constraint(&golden, &constraint).status(),
            ResidualStatus::VerifiedZero
        );
    }

    struct UnsupportedEvaluator {
        theory: ProjectiveSpaceTheory,
    }

    impl CanonicalCorrelatorEvaluator for UnsupportedEvaluator {
        fn theory(&self) -> &dyn GwTheory {
            &self.theory
        }

        fn evaluate_backend(
            &self,
            _correlator: &CorrelatorKey<CurveClass, BasisId>,
        ) -> Result<RatFun, GwError> {
            Err(GwError::UnsupportedInvariant("deliberate gap".to_string()))
        }
    }

    #[test]
    fn unsupported_dependencies_can_never_pass() {
        let evaluator = UnsupportedEvaluator {
            theory: ProjectiveSpaceTheory::new(0),
        };
        let constraint = generate_constraint(
            &evaluator.theory,
            0,
            1,
            evaluator.theory.curve(0),
            TimeMonomial::one(),
        )
        .unwrap();
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::Incomplete);
        assert_eq!(report.missing_correlator_count(), 1);
    }

    #[test]
    fn product_and_bundle_backends_use_the_same_geometric_string_equation() {
        let product = ProductProjectiveEvaluator::new(1, 1).unwrap();
        let product_theory = product.product_theory();
        let product_time = TimeMonomial::from_descendants([
            Descendant::new(0, product_theory.basis_id(1, 1).unwrap()),
            Descendant::new(0, product_theory.basis_id(0, 0).unwrap()),
        ]);
        let product_constraint = generate_constraint(
            product_theory,
            -1,
            0,
            product_theory.curve(0, 0),
            product_time,
        )
        .unwrap();
        assert_eq!(
            evaluate_constraint(&product, &product_constraint).status(),
            ResidualStatus::VerifiedZero
        );

        let bundle_theory = ProjectiveBundleTheory::new(1, vec![0, 0]).unwrap();
        let bundle = ProjectiveBundleEvaluator::new(bundle_theory).unwrap();
        let bundle_theory = bundle.bundle_theory();
        let bundle_time = TimeMonomial::from_descendants([
            Descendant::new(0, bundle_theory.basis_id(1, 1).unwrap()),
            Descendant::new(0, bundle_theory.basis_id(0, 0).unwrap()),
        ]);
        let bundle_constraint =
            generate_constraint(bundle_theory, -1, 0, bundle_theory.curve(0, 0), bundle_time)
                .unwrap();
        assert_eq!(
            evaluate_constraint(&bundle, &bundle_constraint).status(),
            ResidualStatus::VerifiedZero
        );
    }

    #[test]
    fn positive_degree_product_and_twisted_bundle_l0_relations_exercise_backends() {
        let product = ProductProjectiveEvaluator::new(1, 1).unwrap();
        let product_theory = product.product_theory();
        let point_class = product_theory.basis_id(1, 1).unwrap();
        let product_constraint = generate_constraint(
            product_theory,
            0,
            0,
            product_theory.curve(1, 1),
            TimeMonomial::from_descendants(vec![Descendant::new(0, point_class); 3]),
        )
        .unwrap();
        let product_report = evaluate_constraint(&product, &product_constraint);
        assert_eq!(product_report.status(), ResidualStatus::VerifiedZero);
        assert_eq!(product_report.total_term_count(), 4);
        assert_eq!(product_report.backend_correlator_count(), 4);
        assert_eq!(product_report.structural_zero_correlator_count(), 0);
        for key in product_report.backend_correlators() {
            assert_eq!(product.evaluate_backend(key).unwrap(), RatFun::one());
        }

        // F_1 in its exceptional class (H.beta, xi.beta)=(1,-1).  The
        // negative geometric fiber coordinate exercises the theory-owned
        // shifted cone, while L_0 also exercises the twist-dependent c1
        // action from the canonical theory.
        let bundle_theory = ProjectiveBundleTheory::new(1, vec![0, 1]).unwrap();
        let bundle = ProjectiveBundleEvaluator::new(bundle_theory).unwrap();
        let bundle_theory = bundle.bundle_theory();
        let h = bundle_theory.basis_id(1, 0).unwrap();
        let bundle_constraint = generate_constraint(
            bundle_theory,
            0,
            0,
            bundle_theory.curve(1, -1),
            TimeMonomial::from_descendants(vec![Descendant::new(0, h); 3]),
        )
        .unwrap();
        let bundle_report = evaluate_constraint(&bundle, &bundle_constraint);
        assert_eq!(bundle_report.status(), ResidualStatus::VerifiedZero);
        assert_eq!(bundle_report.total_term_count(), 4);
        assert_eq!(bundle_report.backend_correlator_count(), 4);
        assert_eq!(bundle_report.structural_zero_correlator_count(), 0);
        let values = bundle_report
            .backend_correlators()
            .iter()
            .map(|key| bundle.evaluate_backend(key).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            values
                .iter()
                .filter(|value| **value == RatFun::from_rational(Rational::from(-1)))
                .count(),
            1
        );
        assert_eq!(
            values
                .iter()
                .filter(|value| **value == RatFun::one())
                .count(),
            3
        );
    }
}
