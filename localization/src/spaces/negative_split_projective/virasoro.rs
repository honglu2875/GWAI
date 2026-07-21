//! Direct QRR and compact-completion Virasoro adapters for negative
//! split-bundle theories.

mod qrr;
pub use qrr::*;

use crate::constraints::virasoro::{
    specialize_symbolic_constraint_parameters, CanonicalCorrelatorEvaluator,
    CorrelatorDimensionPolicy, CorrelatorKey, SpecializedVirasoroConstraint,
    SymbolicVirasoroConstraint,
};
use crate::core::algebra::{RatFun, Rational};
use crate::core::error::GwError;
use crate::core::theory::{BasisId, CurveClass, GwTheory};
use crate::factored::FactoredRatFun;
use crate::givental::compute_semisimple_graph_value_with_coeff;
use crate::spaces::projective_bundle::{ProjectiveBundleEvaluator, ProjectiveBundleTheory};
use crate::spaces::projective_space::{tau, CohomologyClass};
use std::collections::BTreeMap;
use std::sync::Mutex;

use super::{
    FactoredTwistedProjectiveSpaceProvider, NegativeSplitProjectiveCompletion,
    TwistedProjectiveSpaceProvider,
};

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
                .evaluate_nonequivariant_positive_degree(
                    correlator.genus,
                    degree,
                    &insertions,
                    None,
                )?;
            &RatFun::from_rational(coefficient) * &local_value
        };

        self.cache
            .lock()
            .unwrap()
            .insert(correlator.clone(), value.clone());
        Ok(value)
    }
}

fn translate_qrr_insertions(
    provider: &TwistedProjectiveSpaceProvider,
    correlator: &CorrelatorKey<CurveClass, BasisId>,
) -> Result<Vec<crate::spaces::projective_space::Insertion>, GwError> {
    correlator
        .insertions()
        .iter()
        .map(|insertion| {
            Ok(tau(
                insertion.psi_power,
                CohomologyClass::try_h_power(provider.n(), insertion.class.0)?,
            ))
        })
        .collect()
}

fn qrr_degree(correlator: &CorrelatorKey<CurveClass, BasisId>) -> Result<usize, GwError> {
    if correlator.degree.rank() != 1 {
        return Err(GwError::ConventionMismatch(
            "negative-split QRR evaluation requires a rank-one curve class".to_string(),
        ));
    }
    usize::try_from(correlator.degree.coordinates()[0]).map_err(|_| {
        GwError::ConventionMismatch(
            "negative-split QRR evaluation requires a nonnegative degree".to_string(),
        )
    })
}

/// Evaluate an exact rational point of the fiber-equivariant inverse-Euler
/// theory used by the QRR-conjugated operator.
///
/// The same values must be substituted into the symbolic operator
/// coefficients.  [`Self::specialize_constraint`] performs that operation and
/// returns an artifact which records the assignments.  On the backend side,
/// the fiber parameters are fixed before calibration.  Native-factored graph
/// and divisor-recursion arithmetic then retains only the auxiliary base
/// parameter `lambda_0`; its limit is taken after each complete correlator has
/// been summed.
pub struct NegativeSplitFixedFiberQrrEvaluator {
    provider: FactoredTwistedProjectiveSpaceProvider,
    assignments: BTreeMap<String, Rational>,
    cache: Mutex<BTreeMap<CorrelatorKey<CurveClass, BasisId>, RatFun>>,
}

impl NegativeSplitFixedFiberQrrEvaluator {
    /// Construct from weights attached pairwise to the supplied negative line
    /// summands.  The canonical theory reorders both degrees and weights, so
    /// `mu_i` always refers to the same canonical summand as it does in
    /// [`InverseEulerQrrL0Operator`].
    pub fn new(
        base_n: usize,
        degrees: Vec<usize>,
        fiber_weights: Vec<Rational>,
    ) -> Result<Self, GwError> {
        let provider = FactoredTwistedProjectiveSpaceProvider::qrr_fixed_fiber_lambda_line(
            base_n,
            degrees,
            fiber_weights,
        )?;
        let assignments = provider.inner().fixed_fiber_parameter_assignments()?;
        Ok(Self {
            provider,
            assignments,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn provider(&self) -> &TwistedProjectiveSpaceProvider {
        self.provider.inner()
    }

    pub fn parameter_assignments(&self) -> &BTreeMap<String, Rational> {
        &self.assignments
    }

    /// Specialize a generated symbolic equation at exactly the coefficient
    /// point used by this evaluator.
    pub fn specialize_constraint(
        &self,
        constraint: &SymbolicVirasoroConstraint,
    ) -> Result<SpecializedVirasoroConstraint, GwError> {
        specialize_symbolic_constraint_parameters(constraint, &self.assignments)
    }
}

impl CanonicalCorrelatorEvaluator for NegativeSplitFixedFiberQrrEvaluator {
    fn theory(&self) -> &dyn GwTheory {
        self.provider
            .inner()
            .canonical_theory()
            .expect("fixed-fiber provider has canonical theory")
    }

    fn dimension_policy(&self) -> CorrelatorDimensionPolicy {
        // Fixing the coefficient-field parameters does not turn the twisted
        // equivariant theory into its homogeneous nonequivariant limit.
        CorrelatorDimensionPolicy::EquivariantWeights
    }

    fn certified_zero(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<bool, GwError> {
        let degree = qrr_degree(correlator)?;
        let insertions = translate_qrr_insertions(self.provider.inner(), correlator)?;
        Ok(self.provider.inner().vanishes_above_base_virtual_dimension(
            correlator.genus,
            degree,
            &insertions,
        ))
    }

    fn evaluate_backend(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<RatFun, GwError> {
        if let Some(value) = self.cache.lock().unwrap().get(correlator).cloned() {
            return Ok(value);
        }
        let degree = qrr_degree(correlator)?;
        let insertions = translate_qrr_insertions(self.provider.inner(), correlator)?;
        if self.provider.inner().vanishes_above_base_virtual_dimension(
            correlator.genus,
            degree,
            &insertions,
        ) {
            let zero = RatFun::zero();
            self.cache
                .lock()
                .unwrap()
                .insert(correlator.clone(), zero.clone());
            return Ok(zero);
        }

        let profile = crate::env_flag("GW_PROFILE");
        let started = std::time::Instant::now();
        if profile {
            eprintln!(
                "GW_PROFILE fixed_fiber_qrr_correlator=start genus={} degree={} insertions={:?}",
                correlator.genus,
                degree,
                correlator.insertions()
            );
        }
        let raw = if degree == 0 {
            // Canonical preflight excludes unstable constant maps.  Stable
            // degree-zero coefficients use the same specialized calibration
            // and must be present for 0+d QRR splittings.
            compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
                &self.provider,
                correlator.genus,
                0,
                &insertions,
                None,
            )?
        } else {
            self.provider.evaluate_qrr_fixed_fiber_positive_degree(
                correlator.genus,
                degree,
                &insertions,
                None,
            )?
        };
        // The fixed mu_i were substituted before calibration.  Remove only
        // the auxiliary base lambda after the complete graph/divisor sum.
        let value = self.provider.qrr_lambda_line_limit(&raw)?;
        if profile {
            eprintln!(
                "GW_PROFILE fixed_fiber_qrr_correlator=finish genus={} degree={} insertions={:?} elapsed={:.3}s",
                correlator.genus,
                degree,
                correlator.insertions(),
                started.elapsed().as_secs_f64()
            );
        }
        self.cache
            .lock()
            .unwrap()
            .insert(correlator.clone(), value.clone());
        Ok(value)
    }
}

/// Evaluate the complete fiber-equivariant inverse-Euler theory used by the
/// QRR-conjugated operator.
///
/// Unlike the compact-completion adapter, this evaluator includes stable
/// degree-zero twisted correlators.  Those terms are indispensable: every
/// positive QRR second-order mode produces `0+d` and `d+0` splittings.  The
/// evaluator keeps the independent fiber weights `mu_i` symbolic and removes
/// only the auxiliary base localization parameter.
pub struct NegativeSplitQrrEvaluator {
    provider: FactoredTwistedProjectiveSpaceProvider,
    cache: Mutex<BTreeMap<CorrelatorKey<CurveClass, BasisId>, RatFun>>,
}

impl NegativeSplitQrrEvaluator {
    pub fn new(base_n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        Ok(Self {
            provider: FactoredTwistedProjectiveSpaceProvider::qrr_fiber_equivariant(
                base_n, degrees,
            )?,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn provider(&self) -> &TwistedProjectiveSpaceProvider {
        self.provider.inner()
    }
}

impl CanonicalCorrelatorEvaluator for NegativeSplitQrrEvaluator {
    fn theory(&self) -> &dyn GwTheory {
        self.provider
            .inner()
            .canonical_theory()
            .expect("fiber-equivariant provider has canonical theory")
    }

    fn dimension_policy(&self) -> CorrelatorDimensionPolicy {
        CorrelatorDimensionPolicy::EquivariantWeights
    }

    fn certified_zero(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<bool, GwError> {
        let degree = qrr_degree(correlator)?;
        let insertions = translate_qrr_insertions(self.provider.inner(), correlator)?;
        Ok(self.provider.inner().vanishes_above_base_virtual_dimension(
            correlator.genus,
            degree,
            &insertions,
        ))
    }

    fn evaluate_backend(
        &self,
        correlator: &CorrelatorKey<CurveClass, BasisId>,
    ) -> Result<RatFun, GwError> {
        if let Some(value) = self.cache.lock().unwrap().get(correlator).cloned() {
            return Ok(value);
        }
        let degree = qrr_degree(correlator)?;
        let insertions = translate_qrr_insertions(self.provider.inner(), correlator)?;
        if self.provider.inner().vanishes_above_base_virtual_dimension(
            correlator.genus,
            degree,
            &insertions,
        ) {
            let zero = RatFun::zero();
            self.cache
                .lock()
                .unwrap()
                .insert(correlator.clone(), zero.clone());
            return Ok(zero);
        }
        let profile = crate::env_flag("GW_PROFILE");
        let started = std::time::Instant::now();
        if profile {
            eprintln!(
                "GW_PROFILE qrr_correlator=start genus={} degree={} insertions={:?}",
                correlator.genus,
                degree,
                correlator.insertions()
            );
        }
        let raw = if degree == 0 {
            // The canonical evaluator has already excluded unstable constant
            // maps, so this is a genuine stable degree-zero CohFT coefficient.
            compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
                &self.provider,
                correlator.genus,
                0,
                &insertions,
                None,
            )?
        } else {
            self.provider
                .evaluate_qrr_fiber_equivariant_positive_degree(
                    correlator.genus,
                    degree,
                    &insertions,
                    None,
                )?
        };
        // Expand and remove the auxiliary base-localization parameter once,
        // after all graph and divisor-equation cancellations have happened in
        // the factored coefficient ring.  The independent fiber weights mu_i
        // remain symbolic.
        let value = self.provider.qrr_lambda_line_limit(&raw)?;
        if profile {
            eprintln!(
                "GW_PROFILE qrr_correlator=finish genus={} degree={} insertions={:?} elapsed={:.3}s",
                correlator.genus,
                degree,
                correlator.insertions(),
                started.elapsed().as_secs_f64()
            );
        }
        self.cache
            .lock()
            .unwrap()
            .insert(correlator.clone(), value.clone());
        Ok(value)
    }
}

#[cfg(test)]
mod fixed_fiber_tests {
    use super::*;
    use crate::constraints::virasoro::Descendant;

    fn key(degree: usize, classes: &[usize]) -> CorrelatorKey<CurveClass, BasisId> {
        CorrelatorKey::new(
            0,
            CurveClass::new(vec![degree as i64]),
            classes
                .iter()
                .map(|&class| Descendant::new(0, BasisId(class)))
                .collect(),
        )
    }

    #[test]
    fn fixed_fiber_weights_follow_canonical_summand_order_and_reject_zero() {
        let evaluator = NegativeSplitFixedFiberQrrEvaluator::new(
            2,
            vec![2, 1],
            vec![Rational::from(5), Rational::from(3)],
        )
        .unwrap();
        assert_eq!(
            evaluator.parameter_assignments(),
            &BTreeMap::from([
                ("mu_0".to_string(), Rational::from(3)),
                ("mu_1".to_string(), Rational::from(5)),
            ])
        );
        assert!(matches!(
            NegativeSplitFixedFiberQrrEvaluator::new(
                1,
                vec![1],
                vec![Rational::zero()]
            ),
            Err(GwError::ConventionMismatch(message)) if message.contains("nonzero")
        ));
    }

    #[test]
    fn fixed_fiber_native_factored_path_matches_symbolic_specialization() {
        let symbolic = NegativeSplitQrrEvaluator::new(1, vec![2]).unwrap();
        let fixed =
            NegativeSplitFixedFiberQrrEvaluator::new(1, vec![2], vec![Rational::from(3)]).unwrap();

        // Stable degree zero is needed by QRR's 0+d splittings.  The second
        // row is a genuine positive-degree twisted coefficient with an
        // equivariant degree deficit.
        for correlator in [key(0, &[0, 0, 1]), key(1, &[0, 1, 1])] {
            let symbolic_value = symbolic.evaluate_backend(&correlator).unwrap();
            let expected = symbolic_value
                .evaluate_variables(fixed.parameter_assignments())
                .unwrap();
            let actual = fixed
                .evaluate_backend(&correlator)
                .unwrap()
                .as_rational()
                .expect("fixed-fiber lambda limit must be rational");
            assert_eq!(actual, expected, "correlator: {correlator:?}");
        }
    }
}
