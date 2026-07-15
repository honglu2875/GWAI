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
    BasisId, CurveClass, CurveEffectivity, GwTheory, NegativeSplitProjectiveCompletion,
    ProductProjectiveTheory, ProjectiveBundleTheory, ProjectiveSpaceTheory,
};
use crate::twisted::TwistedProjectiveSpaceProvider;
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
        let value = evaluate_product_with_divisor_recursion(
            &self.theory,
            &self.weights_x,
            &self.weights_y,
            correlator.genus,
            d1,
            d2,
            insertions,
        )?;
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
        let value = evaluate_bundle_with_divisor_recursion(
            &self.theory,
            &self.weights_base,
            &self.weights_fiber,
            correlator.genus,
            d1,
            d2,
            total,
            insertions,
        )?;
        let value = RatFun::from_rational(value);
        self.cache
            .lock()
            .unwrap()
            .insert(correlator.clone(), value.clone());
        Ok(value)
    }
}

/// Audit a negative split-bundle theory through its compact projective
/// completion.
///
/// The generated constraint is the ordinary compact Virasoro equation for
/// `P(O + V)`.  Positive-degree dependencies in the distinguished section are
/// evaluated by the local inverse-Euler provider after the exact restriction
/// `H^h xi^j -> (-A)^j H^(h+j)`.  Degree-zero dependencies are deliberately
/// evaluated by the compact bundle backend: the local/section identification
/// is a positive-degree concavity statement and does not identify the
/// degree-zero theories.
///
/// A positive class outside the distinguished section is rejected rather than
/// assigned a placeholder zero.  Thus this adapter is suitable for
/// section-sector constraints, not for arbitrary compact-bundle invariants.
pub struct NegativeSplitCompletionEvaluator {
    completion: NegativeSplitProjectiveCompletion,
    compact_evaluator: ProjectiveBundleEvaluator,
    local_provider: TwistedProjectiveSpaceProvider,
    cache: Mutex<BTreeMap<CorrelatorKey<CurveClass, BasisId>, RatFun>>,
}

impl NegativeSplitCompletionEvaluator {
    pub fn new(base_n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        let local_provider = TwistedProjectiveSpaceProvider::new(base_n, degrees, false)?;
        Self::from_provider(local_provider)
    }

    pub fn from_provider(local_provider: TwistedProjectiveSpaceProvider) -> Result<Self, GwError> {
        local_provider.validate_compact_completion_audit()?;
        let completion =
            NegativeSplitProjectiveCompletion::new(local_provider.canonical_theory()?.clone())?;
        let compact_evaluator =
            ProjectiveBundleEvaluator::new(completion.compact_theory().clone())?;
        Ok(Self {
            completion,
            compact_evaluator,
            local_provider,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn completion(&self) -> &NegativeSplitProjectiveCompletion {
        &self.completion
    }

    pub fn compact_theory(&self) -> &ProjectiveBundleTheory {
        self.completion.compact_theory()
    }

    pub fn local_provider(&self) -> &TwistedProjectiveSpaceProvider {
        &self.local_provider
    }
}

impl CanonicalCorrelatorEvaluator for NegativeSplitCompletionEvaluator {
    fn theory(&self) -> &dyn GwTheory {
        self.compact_theory()
    }

    fn evaluate_backend(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<RatFun, GwError> {
        if let Some(value) = self.cache.lock().unwrap().get(correlator).cloned() {
            return Ok(value);
        }

        let value = if correlator.degree.is_zero() {
            self.compact_evaluator.evaluate_backend(correlator)?
        } else {
            let degree = self
                .completion
                .section_degree(&correlator.degree)
                .filter(|degree| *degree > 0)
                .ok_or_else(|| {
                    GwError::UnsupportedInvariant(format!(
                        "compact-section evaluator only supports positive section classes, not {:?}",
                        correlator.degree.coordinates()
                    ))
                })?;

            let mut coefficient = Rational::one();
            let mut insertions = Vec::new();
            insertions
                .try_reserve_exact(correlator.insertions().len())
                .map_err(|_| {
                    GwError::UnsupportedInvariant(
                        "cannot allocate compact-section insertion restrictions".to_string(),
                    )
                })?;
            for insertion in correlator.insertions() {
                let Some((local_basis, scalar)) =
                    self.completion.restrict_basis_to_section(insertion.class)?
                else {
                    let zero = RatFun::zero();
                    self.cache
                        .lock()
                        .unwrap()
                        .insert(correlator.clone(), zero.clone());
                    return Ok(zero);
                };
                coefficient = coefficient * scalar;
                let class = CohomologyClass::try_h_power(
                    self.completion.local_theory().base_dimension(),
                    local_basis.0,
                )?;
                insertions.push(tau(insertion.psi_power, class));
            }
            let local_value = self
                .local_provider
                .evaluate_nonequivariant_positive_degree(correlator.genus, degree, &insertions)?;
            &RatFun::from_rational(coefficient) * &local_value
        };

        self.cache
            .lock()
            .unwrap()
            .insert(correlator.clone(), value.clone());
        Ok(value)
    }
}

fn pointed_curve_is_stable(genus: usize, markings: usize) -> bool {
    match genus {
        0 => markings >= 3,
        1 => markings >= 1,
        _ => true,
    }
}

fn check_divisor_recursion_depth(
    genus: usize,
    markings: usize,
    mut descendant_powers: impl Iterator<Item = usize>,
    target: &str,
) -> Result<(), GwError> {
    if pointed_curve_is_stable(genus, markings) {
        return Ok(());
    }
    let total_descendant_power = descendant_powers.try_fold(0usize, |total, power| {
        total.checked_add(power).ok_or_else(|| {
            GwError::UnsupportedInvariant(format!(
                "{target} divisor-recursion descendant degree overflow"
            ))
        })
    })?;
    // With two unstable descendants the correction branches form a lattice,
    // so a stack-depth bound alone is not a work bound.  Keep a deliberately
    // smaller aggregate psi envelope for this adapter-only recursion; stable
    // graph evaluation retains its independent, larger descendant boundary.
    let maximum = crate::MAX_UNSTABLE_DIVISOR_RECURSION_TOTAL_PSI;
    if total_descendant_power > maximum {
        return Err(GwError::UnsupportedInvariant(format!(
            "{target} unstable descendant degree {total_descendant_power} exceeds the divisor-recursion implementation bound {maximum}"
        )));
    }
    Ok(())
}

fn reconstruct_requested_product_value(
    theory: &ProductProjectiveTheory,
    weights_x: &[Rational],
    weights_y: &[Rational],
    genus: usize,
    d1: usize,
    d2: usize,
    insertions: &[ProductInsertion],
) -> Result<Rational, GwError> {
    let total_degree = d1
        .checked_add(d2)
        .ok_or_else(|| GwError::AlgebraFailure("total bidegree overflow".to_string()))?;
    let values = reconstruct_bidegree_invariants_in_theory(
        theory,
        weights_x,
        weights_y,
        genus,
        total_degree,
        insertions,
    )?;
    values.get(d2).cloned().ok_or_else(|| {
        GwError::AlgebraFailure("product reconstruction omitted requested bidegree".to_string())
    })
}

/// Evaluate a positive-degree correlator even when its underlying pointed
/// curve is unstable.  Adding a divisor `D` and solving
///
/// `<D, prod tau_ai(gamma_i)> = (D.beta)<prod ...>
///    + sum_i <tau_(ai-1)(D cup gamma_i), ...>`
///
/// recursively includes the descendant corrections that a simple division by
/// `(D.beta)` would miss.
fn evaluate_product_with_divisor_recursion(
    theory: &ProductProjectiveTheory,
    weights_x: &[Rational],
    weights_y: &[Rational],
    genus: usize,
    d1: usize,
    d2: usize,
    insertions: Vec<ProductInsertion>,
) -> Result<Rational, GwError> {
    if pointed_curve_is_stable(genus, insertions.len()) {
        return reconstruct_requested_product_value(
            theory,
            weights_x,
            weights_y,
            genus,
            d1,
            d2,
            &insertions,
        );
    }
    check_divisor_recursion_depth(
        genus,
        insertions.len(),
        insertions
            .iter()
            .map(|insertion| insertion.descendant_power),
        "product",
    )?;
    let (use_first_factor, divisor, intersection) = if d1 > 0 {
        (true, ProductInsertion::new(0, 1, 0), Rational::from(d1))
    } else if d2 > 0 {
        (false, ProductInsertion::new(0, 0, 1), Rational::from(d2))
    } else {
        return Err(GwError::ConventionMismatch(
            "degree-zero unstable correlators must be removed before product reconstruction"
                .to_string(),
        ));
    };
    let mut with_divisor = insertions.clone();
    with_divisor.push(divisor);
    let mut numerator = evaluate_product_with_divisor_recursion(
        theory,
        weights_x,
        weights_y,
        genus,
        d1,
        d2,
        with_divisor,
    )?;
    let (n, m) = theory.dimensions();
    for index in 0..insertions.len() {
        if insertions[index].descendant_power == 0 {
            continue;
        }
        let mut correction = insertions.clone();
        correction[index].descendant_power -= 1;
        if use_first_factor {
            if correction[index].h1_power == n {
                continue;
            }
            correction[index].h1_power += 1;
        } else {
            if correction[index].h2_power == m {
                continue;
            }
            correction[index].h2_power += 1;
        }
        numerator = numerator
            - evaluate_product_with_divisor_recursion(
                theory, weights_x, weights_y, genus, d1, d2, correction,
            )?;
    }
    Ok(numerator / intersection)
}

fn reconstruct_requested_bundle_value(
    theory: &ProjectiveBundleTheory,
    weights_base: &[Rational],
    weights_fiber: &[Rational],
    genus: usize,
    d1: usize,
    d2: i64,
    total_degree: usize,
    insertions: &[BundleInsertion],
) -> Result<Rational, GwError> {
    let values = reconstruct_bundle_invariants_in_theory(
        theory,
        weights_base,
        weights_fiber,
        genus,
        total_degree,
        insertions,
    )?;
    let requested_d2 = isize::try_from(d2).map_err(|_| {
        GwError::AlgebraFailure("bundle fiber degree does not fit in isize".to_string())
    })?;
    values
        .into_iter()
        .find(|(candidate_d1, candidate_d2, _)| {
            *candidate_d1 == d1 && *candidate_d2 == requested_d2
        })
        .map(|(_, _, value)| value)
        .ok_or_else(|| {
            GwError::AlgebraFailure("bundle reconstruction omitted requested bidegree".to_string())
        })
}

fn evaluate_bundle_with_divisor_recursion(
    theory: &ProjectiveBundleTheory,
    weights_base: &[Rational],
    weights_fiber: &[Rational],
    genus: usize,
    d1: usize,
    d2: i64,
    total_degree: usize,
    insertions: Vec<BundleInsertion>,
) -> Result<Rational, GwError> {
    if pointed_curve_is_stable(genus, insertions.len()) {
        return reconstruct_requested_bundle_value(
            theory,
            weights_base,
            weights_fiber,
            genus,
            d1,
            d2,
            total_degree,
            &insertions,
        );
    }
    check_divisor_recursion_depth(
        genus,
        insertions.len(),
        insertions
            .iter()
            .map(|insertion| insertion.descendant_power),
        "projective-bundle",
    )?;
    let (use_base_divisor, divisor, intersection) = if d1 > 0 {
        (true, BundleInsertion::new(0, 1, 0), Rational::from(d1))
    } else if d2 > 0 {
        (false, BundleInsertion::new(0, 0, 1), Rational::from(d2))
    } else {
        return Err(GwError::ConventionMismatch(
            "degree-zero unstable correlators must be removed before projective-bundle reconstruction"
                .to_string(),
        ));
    };
    if !use_base_divisor
        && insertions
            .iter()
            .any(|insertion| insertion.descendant_power > 0)
    {
        return Err(GwError::UnsupportedInvariant(
            "fiber-only projective-bundle descendant stabilization requires classical xi multiplication across the bundle relation"
                .to_string(),
        ));
    }
    let mut with_divisor = insertions.clone();
    with_divisor.push(divisor);
    let mut numerator = evaluate_bundle_with_divisor_recursion(
        theory,
        weights_base,
        weights_fiber,
        genus,
        d1,
        d2,
        total_degree,
        with_divisor,
    )?;
    if use_base_divisor {
        let n = theory.base_dimension();
        for index in 0..insertions.len() {
            if insertions[index].descendant_power == 0 || insertions[index].h_power == n {
                continue;
            }
            let mut correction = insertions.clone();
            correction[index].descendant_power -= 1;
            correction[index].h_power += 1;
            numerator = numerator
                - evaluate_bundle_with_divisor_recursion(
                    theory,
                    weights_base,
                    weights_fiber,
                    genus,
                    d1,
                    d2,
                    total_degree,
                    correction,
                )?;
        }
    }
    Ok(numerator / intersection)
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

    fn has_nonzero_contribution_from(
        constraint: &CanonicalVirasoroConstraint,
        report: &ResidualReport<CurveClass, BasisId, RatFun>,
        origin: TermOrigin,
    ) -> bool {
        report.evaluated_terms().iter().any(|evaluated| {
            !evaluated.exact_contribution.is_zero()
                && constraint.terms[evaluated.term_index].origin() == &origin
        })
    }

    struct PerturbedEvaluator<'a> {
        inner: &'a dyn CanonicalCorrelatorEvaluator,
        target: CorrelatorKey<CurveClass, BasisId>,
    }

    impl CanonicalCorrelatorEvaluator for PerturbedEvaluator<'_> {
        fn theory(&self) -> &dyn GwTheory {
            self.inner.theory()
        }

        fn evaluate_backend(
            &self,
            correlator: &CorrelatorKey<CurveClass, BasisId>,
        ) -> Result<RatFun, GwError> {
            let value = self.inner.evaluate_backend(correlator)?;
            if correlator == &self.target {
                Ok(&value + &RatFun::one())
            } else {
                Ok(value)
            }
        }
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
    fn bundle_primary_divisor_stabilization_closes_positive_unstable_ranges() {
        let theory = ProjectiveBundleTheory::new(1, vec![0, 1]).unwrap();
        let evaluator = ProjectiveBundleEvaluator::new(theory).unwrap();
        let theory = evaluator.bundle_theory();
        let unit = theory.basis_id(0, 0).unwrap();
        let h = theory.basis_id(1, 0).unwrap();
        let curve = theory.curve(1, -1);

        let one_point = CorrelatorKey::new(0, curve.clone(), vec![Descendant::new(0, unit)]);
        assert!(evaluator.evaluate_backend(&one_point).is_ok());

        let two_point = CorrelatorKey::new(
            0,
            curve,
            vec![Descendant::new(0, unit), Descendant::new(0, h)],
        );
        assert!(evaluator.evaluate_backend(&two_point).is_ok());
    }

    #[test]
    fn bundle_descendant_divisor_recursion_recovers_the_unstable_dilaton_value() {
        let theory = ProjectiveBundleTheory::new(1, vec![0, 1]).unwrap();
        let evaluator = ProjectiveBundleEvaluator::new(theory).unwrap();
        let theory = evaluator.bundle_theory();
        let correlator = CorrelatorKey::new(
            0,
            theory.curve(1, -1),
            vec![Descendant::new(1, theory.basis_id(0, 0).unwrap())],
        );

        assert_eq!(
            evaluator.evaluate_backend(&correlator).unwrap(),
            RatFun::from_rational(Rational::from(-2))
        );
    }

    #[test]
    fn product_descendant_divisor_recursion_recovers_the_unstable_dilaton_value() {
        let evaluator = ProductProjectiveEvaluator::new(1, 1).unwrap();
        let theory = evaluator.product_theory();
        let correlator = CorrelatorKey::new(
            0,
            theory.curve(1, 0),
            vec![
                Descendant::new(1, theory.basis_id(0, 0).unwrap()),
                Descendant::new(0, theory.basis_id(1, 1).unwrap()),
            ],
        );

        assert_eq!(
            evaluator.evaluate_backend(&correlator).unwrap(),
            RatFun::from_rational(Rational::from(-1))
        );
    }

    #[test]
    fn product_divisor_recursion_handles_two_descendant_corrections() {
        let evaluator = ProductProjectiveEvaluator::new(1, 1).unwrap();
        let theory = evaluator.product_theory();
        let curve = theory.curve(1, 0);
        let unit = theory.basis_id(0, 0).unwrap();
        let h2 = theory.basis_id(0, 1).unwrap();
        let one_descendant = CorrelatorKey::new(0, curve.clone(), vec![Descendant::new(1, h2)]);
        let two_descendants = CorrelatorKey::new(
            0,
            curve,
            vec![Descendant::new(1, unit), Descendant::new(1, h2)],
        );

        let reduced = evaluator.evaluate_backend(&one_descendant).unwrap();
        assert!(!reduced.is_zero());
        assert_eq!(
            evaluator.evaluate_backend(&two_descendants).unwrap(),
            -reduced,
            "the genus-zero dilaton equation has coefficient -1 with one other marking"
        );
    }

    #[test]
    fn divisor_recursion_rejects_adversarial_descendant_depth_before_recursing() {
        let evaluator = ProductProjectiveEvaluator::new(1, 1).unwrap();
        let theory = evaluator.product_theory();
        let correlator = CorrelatorKey::new(
            0,
            theory.curve(1, 0),
            vec![Descendant::new(usize::MAX, theory.basis_id(0, 0).unwrap())],
        );

        let error = evaluator.evaluate_backend(&correlator).unwrap_err();
        assert!(matches!(error, GwError::UnsupportedInvariant(_)));
        assert!(error
            .to_string()
            .contains("divisor-recursion implementation bound"));
    }

    #[test]
    fn f2_genus_two_exceptional_l1_relation_fails_closed_on_native_bundle_gap() {
        let theory = ProjectiveBundleTheory::new(1, vec![0, 2]).unwrap();
        let evaluator = ProjectiveBundleEvaluator::new(theory).unwrap();
        let theory = evaluator.bundle_theory();
        let constraint = generate_constraint(
            theory,
            1,
            2,
            theory.curve(1, -2),
            TimeMonomial::from_descendants([Descendant::new(0, theory.basis_id(1, 0).unwrap())]),
        )
        .unwrap();
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::Incomplete, "{report:?}");
        assert!(report.missing_correlators().iter().any(|missing| {
            missing.correlator.degree == theory.curve(1, -2)
                && matches!(
                    &missing.reason,
                    IncompleteReason::Unsupported(message)
                        if message.contains("deformation-negative")
                )
        }));
    }

    #[test]
    fn f1_genus_two_l2_relation_is_nonlinear_and_perturbation_sensitive() {
        let theory = ProjectiveBundleTheory::new(1, vec![0, 1]).unwrap();
        let evaluator = ProjectiveBundleEvaluator::new(theory).unwrap();
        let theory = evaluator.bundle_theory();
        let curve = theory.curve(1, -1);
        let constraint =
            generate_constraint(theory, 2, 2, curve.clone(), TimeMonomial::one()).unwrap();
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::VerifiedZero, "{report:?}");
        assert!(report.total_term_count() >= 50);
        assert!(report.backend_correlator_count() >= 20);

        let high_genus_target = CorrelatorKey::new(
            2,
            curve,
            vec![Descendant::new(3, theory.basis_id(0, 0).unwrap())],
        );
        assert!(report.backend_correlators().contains(&high_genus_target));
        assert!(!evaluator
            .evaluate_backend(&high_genus_target)
            .unwrap()
            .is_zero());
        assert!(has_nonzero_contribution_from(
            &constraint,
            &report,
            TermOrigin::GenusReduction
        ));
        assert!(has_nonzero_contribution_from(
            &constraint,
            &report,
            TermOrigin::DegreeSplitting
        ));

        let perturbed = PerturbedEvaluator {
            inner: &evaluator,
            target: high_genus_target,
        };
        assert_eq!(
            evaluate_constraint(&perturbed, &constraint).status(),
            ResidualStatus::Nonzero,
            "the relation must detect a corrupted native bundle invariant"
        );
    }

    #[test]
    fn conifold_completion_l1_uses_nonzero_genus_two_twisted_descendants() {
        let evaluator = NegativeSplitCompletionEvaluator::new(1, vec![1, 1]).unwrap();
        let theory = evaluator.compact_theory();
        let curve = evaluator.completion().section_curve(1).unwrap();
        let constraint = generate_constraint(
            theory,
            1,
            2,
            curve.clone(),
            TimeMonomial::from_descendants([Descendant::new(0, theory.state_space().unit)]),
        )
        .unwrap();
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::VerifiedZero, "{report:?}");
        assert!(report.backend_correlator_count() >= 7);
        assert!(report.backend_correlators().iter().any(|key| {
            key.genus == 2
                && key.degree == curve
                && evaluator
                    .evaluate_backend(key)
                    .is_ok_and(|value| !value.is_zero())
        }));
    }

    #[test]
    fn compact_completion_routes_only_positive_section_classes_to_local_provider() {
        let evaluator = NegativeSplitCompletionEvaluator::new(1, vec![1, 1]).unwrap();
        let theory = evaluator.compact_theory();
        assert_eq!(
            evaluator.completion().local_theory(),
            evaluator.local_provider().canonical_theory().unwrap(),
            "the completion must be derived from the provider-owned geometry"
        );

        let nonsection = CorrelatorKey::new(2, theory.curve(0, 1), Vec::new());
        let error = evaluator.evaluate_backend(&nonsection).unwrap_err();
        assert!(matches!(error, GwError::UnsupportedInvariant(_)));
        assert!(error
            .to_string()
            .contains("only supports positive section classes"));

        let invalid_basis = CorrelatorKey::new(
            2,
            evaluator.completion().section_curve(1).unwrap(),
            vec![Descendant::new(0, BasisId(usize::MAX))],
        );
        let error = evaluator.evaluate_backend(&invalid_basis).unwrap_err();
        assert!(matches!(error, GwError::ConventionMismatch(_)));

        let vanishing_restriction = CorrelatorKey::new(
            2,
            evaluator.completion().section_curve(1).unwrap(),
            vec![Descendant::new(0, theory.basis_id(1, 1).unwrap())],
        );
        assert!(evaluator
            .evaluate_backend(&vanishing_restriction)
            .unwrap()
            .is_zero());

        let degree_zero = CorrelatorKey::new(
            0,
            theory.curve(0, 0),
            vec![
                Descendant::new(0, theory.basis_id(0, 0).unwrap()),
                Descendant::new(0, theory.basis_id(0, 0).unwrap()),
                Descendant::new(0, theory.basis_id(1, 2).unwrap()),
            ],
        );
        let compact = ProjectiveBundleEvaluator::new(theory.clone()).unwrap();
        assert_eq!(
            evaluator.evaluate_backend(&degree_zero).unwrap(),
            compact.evaluate_backend(&degree_zero).unwrap()
        );
    }

    #[test]
    fn compact_completion_rejects_incompatible_twisted_calibrations() {
        let euler = TwistedProjectiveSpaceProvider::euler_twist(1, vec![1, 1]).unwrap();
        assert!(!euler.supports_compact_completion_audit());
        let error = NegativeSplitCompletionEvaluator::from_provider(euler)
            .err()
            .expect("Euler twist must be rejected");
        assert!(error.to_string().contains("nonequivariant inverse-Euler"));

        let base_equivariant = TwistedProjectiveSpaceProvider::new(1, vec![1, 1], true).unwrap();
        assert!(!base_equivariant.supports_compact_completion_audit());
        assert!(NegativeSplitCompletionEvaluator::from_provider(base_equivariant).is_err());
    }

    #[test]
    fn twisted_descendant_divisor_recursion_recovers_conifold_dilaton() {
        let evaluator = NegativeSplitCompletionEvaluator::new(1, vec![1, 1]).unwrap();
        let theory = evaluator.compact_theory();
        let curve = evaluator.completion().section_curve(1).unwrap();
        let correlator = CorrelatorKey::new(
            0,
            curve,
            vec![Descendant::new(1, theory.basis_id(0, 0).unwrap())],
        );

        assert_eq!(
            evaluator.evaluate_backend(&correlator).unwrap(),
            RatFun::from_rational(Rational::from(-2))
        );
    }

    #[test]
    fn non_calabi_twist_completion_l2_is_high_genus_and_perturbation_sensitive() {
        let evaluator = NegativeSplitCompletionEvaluator::new(2, vec![2]).unwrap();
        let theory = evaluator.compact_theory();
        let curve = evaluator.completion().section_curve(1).unwrap();
        let constraint = generate_constraint(
            theory,
            2,
            2,
            curve.clone(),
            TimeMonomial::from_descendants([Descendant::new(0, theory.state_space().unit)]),
        )
        .unwrap();
        let report = evaluate_constraint(&evaluator, &constraint);
        assert_eq!(report.status(), ResidualStatus::VerifiedZero, "{report:?}");
        assert!(report.total_term_count() >= 70);
        assert!(report.backend_correlator_count() >= 25);

        let high_genus_target = CorrelatorKey::new(
            2,
            curve.clone(),
            vec![
                Descendant::new(0, theory.state_space().unit),
                Descendant::new(1, theory.basis_id(1, 1).unwrap()),
            ],
        );
        assert!(report.backend_correlators().contains(&high_genus_target));
        assert!(!evaluator
            .evaluate_backend(&high_genus_target)
            .unwrap()
            .is_zero());
        assert!(constraint
            .terms
            .iter()
            .any(|term| term.origin() == &TermOrigin::GenusReduction));
        assert!(constraint
            .terms
            .iter()
            .any(|term| term.origin() == &TermOrigin::DegreeSplitting));

        let perturbed = PerturbedEvaluator {
            inner: &evaluator,
            target: high_genus_target,
        };
        let perturbed_report = evaluate_constraint(&perturbed, &constraint);
        assert_eq!(
            perturbed_report.status(),
            ResidualStatus::Nonzero,
            "the constraint must detect a corrupted high-genus twisted value"
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
