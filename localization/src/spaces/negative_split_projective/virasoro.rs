//! Compact-completion Virasoro adapter for negative split-bundle theories.

use crate::constraints::virasoro::{CanonicalCorrelatorEvaluator, CorrelatorKey};
use crate::core::algebra::{RatFun, Rational};
use crate::core::error::GwError;
use crate::core::theory::{BasisId, CurveClass, GwTheory};
use crate::spaces::projective_bundle::{ProjectiveBundleEvaluator, ProjectiveBundleTheory};
use crate::spaces::projective_space::{tau, CohomologyClass};
use std::collections::BTreeMap;
use std::sync::Mutex;

use super::{NegativeSplitProjectiveCompletion, TwistedProjectiveSpaceProvider};

/// Audit a negative split-bundle theory through its compact projective
/// completion.
///
/// The generated constraint is the ordinary compact Virasoro equation for
/// `P(O + V)`. Positive-degree dependencies in the distinguished section are
/// evaluated by the local inverse-Euler provider after the exact restriction
/// `H^h xi^j -> (-A)^j H^(h+j)`. Degree-zero dependencies are deliberately
/// evaluated by the compact bundle backend: the local/section identification
/// is a positive-degree concavity statement and does not identify the
/// degree-zero theories.
///
/// A positive class outside the distinguished section is rejected rather than
/// assigned a placeholder zero. Thus this adapter is suitable for
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
