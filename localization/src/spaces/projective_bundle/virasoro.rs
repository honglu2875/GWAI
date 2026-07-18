//! Canonical-correlator adapter for Virasoro audits of split projective bundles.

use crate::constraints::virasoro::{
    evaluate_with_divisor_recursion, CanonicalCorrelatorEvaluator, CorrelatorKey, Descendant,
};
use crate::core::algebra::{RatFun, Rational};
use crate::core::error::GwError;
use crate::core::theory::{BasisId, CurveClass, GwTheory};
use std::collections::BTreeMap;
use std::sync::Mutex;

use super::{reconstruct_bundle_invariants_in_theory, BundleInsertion, ProjectiveBundleTheory};

/// Exact ray-reconstruction adapter used to substitute projective-bundle
/// invariants into canonical Virasoro constraints.
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
        // disjoint intervals. This remains generic after any permutation of
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
    let curve = theory.try_curve(d1, d2)?;
    let canonical = insertions
        .iter()
        .map(|insertion| {
            theory
                .basis_id(insertion.h_power, insertion.xi_power)
                .map(|basis| Descendant::new(insertion.descendant_power, basis))
                .ok_or_else(|| {
                    GwError::ConventionMismatch(
                        "projective-bundle insertion is outside the canonical basis".to_string(),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let evaluate_stable = |canonical: &[Descendant<BasisId>]| {
        let backend_insertions = canonical
            .iter()
            .map(|insertion| {
                theory
                    .basis_powers(insertion.class)
                    .map(|(h_power, xi_power)| {
                        BundleInsertion::new(insertion.psi_power, h_power, xi_power)
                    })
                    .ok_or_else(|| {
                        GwError::AlgebraFailure(
                            "projective-bundle cup product returned an invalid basis id"
                                .to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        reconstruct_requested_bundle_value(
            theory,
            weights_base,
            weights_fiber,
            genus,
            d1,
            d2,
            total_degree,
            &backend_insertions,
        )
    };
    evaluate_with_divisor_recursion(
        theory,
        genus,
        &curve,
        canonical,
        "projective-bundle",
        &evaluate_stable,
    )
}
