//! Canonical-correlator adapter for Virasoro audits of ordinary projective space.

use crate::constraints::virasoro::{CanonicalCorrelatorEvaluator, CorrelatorKey};
use crate::core::algebra::RatFun;
use crate::core::error::GwError;
use crate::core::theory::{BasisId, CurveClass, GwTheory};
use crate::givental::compute_semisimple_graph_value;
use std::collections::BTreeMap;
use std::sync::Mutex;

use super::{tau, CohomologyClass, ProjectiveSpaceProvider, ProjectiveSpaceTheory};

/// Exact backend adapter used to substitute projective-space invariants into
/// canonical Virasoro constraints.
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
