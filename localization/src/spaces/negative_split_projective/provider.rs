//! The twisted semisimple CohFT providers (rational and factored), the
//! TwistedInvariantRequest, the public compute entry points, and the graph
//! evaluator wiring.

use super::calibration::{
    negative_split_twisted_birkhoff_calibration_candidate_for_coeff_weights_with_validation,
    negative_split_twisted_birkhoff_calibration_candidate_for_ratfun_weights_with_validation,
    negative_split_twisted_birkhoff_calibration_candidate_with_mode_and_validation,
    twisted_classical_h_multiplication_matrix_coeff, twisted_inverse_euler_flat_metric_matrix,
    twisted_inverse_euler_flat_metric_matrix_coeff,
    twisted_inverse_euler_flat_metric_matrix_ratfun,
    twisted_inverse_euler_flat_metric_pair_from_rational_base,
    twisted_quantum_multiplication_from_s_coeff, validate_twisted_weights,
};
use super::hypergeometric::{
    NegativeSplitEquivariantHypergeometricModel, NegativeSplitLineHypergeometricModel,
};
use super::theory::NegativeSplitTotalSpaceTheory;
use super::twist::NegativeSplitBundleTwist;
use crate::core::algebra::{Coeff, RatFun, Rational};
use crate::core::bounded_cache::{BoundedCache, TARGET_RECONSTRUCTION_CACHE_CAPACITY};
use crate::core::error::GwError;
use crate::core::moduli::{pointed_curve_is_stable, MAX_UNSTABLE_DIVISOR_RECURSION_TOTAL_PSI};
use crate::core::series::{QSeries, SeriesMatrix};
use crate::core::theory::{BasisId, GwTheory};
use crate::factored::FactoredRatFun;
use crate::givental::recipe::{
    metric_adjoint_descendant_s_matrix_coeff, metric_adjoint_descendant_s_matrix_with_inverse_coeff,
};
use crate::givental::{
    compute_semisimple_graph_value_with_coeff, GiventalGraphKernel, SemisimpleCohftProvider,
    SeriesSMatrix, Truncation,
};
use crate::reconstruction::{CyclicCoordinates, CyclicQuantumAlgebra};
use crate::spaces::projective_space::{
    Insertion, InvariantResult, ProjectiveSpaceProvider, ResolventRequest, ResolventResult,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwistedProjectiveSpaceProvider {
    base: ProjectiveSpaceProvider,
    // Immutable calibration recipe derived from `canonical_theory` during
    // construction. It is retained to preserve the public `twist()` view and
    // avoid rebuilding hypergeometric inputs; it is not a geometry authority.
    twist: NegativeSplitBundleTwist,
    canonical_theory: NegativeSplitTotalSpaceTheory,
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
    fiber_weight_scale: Rational,
    custom_fiber_weights: Option<Vec<Rational>>,
    fiber_parameter_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum TwistedLineMode {
    EarlyRational,
    SymbolicLimit,
    FiberEquivariant,
    /// Auxiliary base weights lie on a symbolic lambda line, while the
    /// independent inverse-Euler fiber parameters are fixed nonzero
    /// rationals.  This is a coefficient-field specialization of the
    /// fiber-equivariant theory, not the all-weights nonequivariant limit.
    FixedFiberLambdaLine,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TwistedCalibrationMode {
    InverseEuler,
    InverseEulerFiberPlus,
    Euler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TwistedCalibrationValidation {
    /// Skip diagnostic identities that are expensive and do not change the
    /// produced calibration.  This is the default in the graph-evaluation path.
    Fast,
    /// Run full self-adjointness, diagonalization, and R-unitarity checks.
    Full,
}

impl TwistedCalibrationValidation {
    pub(crate) fn runs_expensive_checks(self) -> bool {
        matches!(self, Self::Full)
    }
}

pub(crate) fn twisted_calibration_validation_from_env() -> TwistedCalibrationValidation {
    // Expensive identities are off in the hot graph-evaluation path by default.
    // Set either variable to 1/true/yes/on/full when debugging a calibration
    // change and wanting self-adjointness, diagonalization, and unitarity checks
    // to run before caching the graph kernel.
    if crate::env_flag("GWAI_VALIDATE_TWISTED_CALIBRATION")
        || crate::env_flag("GW_VALIDATE_CALIBRATION")
    {
        TwistedCalibrationValidation::Full
    } else {
        TwistedCalibrationValidation::Fast
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TwistedGraphKernelCacheKey {
    n: usize,
    twist_degrees: Vec<usize>,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
    base_weights: Vec<Rational>,
    fiber_weights: Vec<Rational>,
    fiber_parameter_names: Vec<String>,
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
    validation: TwistedCalibrationValidation,
}

/// Complete identity of one descendant-to-ancestor Birkhoff calibration.
///
/// This is intentionally separate from the graph-kernel cache key: descendant
/// requests have their own `z_order`, while graph kernels are keyed by an
/// `R`-order and stable-graph dimension.  Keeping every geometry and convention
/// input in the key prevents two provider modes from sharing a merely
/// shape-compatible matrix.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TwistedDescendantSCacheKey {
    n: usize,
    twist_degrees: Vec<usize>,
    q_degree: usize,
    z_order: usize,
    base_weights: Vec<Rational>,
    fiber_weights: Vec<Rational>,
    fiber_parameter_names: Vec<String>,
    base_equivariant: bool,
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
}

fn require_nonempty_negative_split_twist(degrees: &[usize]) -> Result<(), GwError> {
    if degrees.is_empty() {
        Err(GwError::ConventionMismatch(
            "a twisted negative-split provider requires at least one line-bundle summand"
                .to_string(),
        ))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwistedInvariantRequest {
    pub n: usize,
    pub twist: NegativeSplitBundleTwist,
    pub genus: usize,
    pub degree: usize,
    pub insertions: Vec<Insertion>,
    pub equivariant: bool,
    pub truncation: Option<Truncation>,
}

impl TwistedInvariantRequest {
    pub fn new(
        n: usize,
        degrees: Vec<usize>,
        genus: usize,
        degree: usize,
        insertions: Vec<Insertion>,
    ) -> Result<Self, GwError> {
        let request = Self {
            n,
            twist: NegativeSplitBundleTwist::new(degrees)?,
            genus,
            degree,
            insertions,
            equivariant: false,
            truncation: None,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), GwError> {
        require_nonempty_negative_split_twist(self.twist.degrees())?;
        for (index, insertion) in self.insertions.iter().enumerate() {
            if insertion.class.n() != self.n {
                return Err(GwError::ConventionMismatch(format!(
                    "twisted P^{} request insertion {index} belongs to P^{}",
                    self.n,
                    insertion.class.n()
                )));
            }
        }
        Ok(())
    }
}

pub fn compute_negative_split_twisted(
    req: &TwistedInvariantRequest,
) -> Result<InvariantResult, GwError> {
    req.validate()?;
    // Public local/twisted path: construct a semisimple provider from the
    // hypergeometric/Birkhoff/QRR calibration, run the generic Givental graph
    // evaluator, and either take the one-parameter lambda-line limit or keep the
    // fiber equivariant parameters as symbolic rational variables.
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }
    if req.n == 0 {
        return Ok(InvariantResult {
            value: RatFun::zero(),
            engine: "twisted-negative-split-effective-degree",
            notes: vec!["P^0 has no positive curve classes".to_string()],
        });
    }

    if req.equivariant {
        let provider =
            TwistedProjectiveSpaceProvider::fiber_equivariant(req.n, req.twist.degrees().to_vec())?;
        let unstable_pointed_curve = !pointed_curve_is_stable(req.genus, req.insertions.len());
        let primary_three_point =
            req.genus == 0 && genus_zero_three_primary_layout(&req.insertions);
        let value = provider.evaluate_fiber_equivariant_positive_degree(
            req.genus,
            req.degree,
            &req.insertions,
            req.truncation.as_ref(),
        )?;
        return Ok(InvariantResult {
            value,
            engine: "twisted-negative-split-fiber-equivariant-givental-birkhoff",
            notes: vec![if unstable_pointed_curve {
                "positive-degree unstable pointed correlator reconstructed by the full descendant divisor equation from the fiber-equivariant hypergeometric/Birkhoff calibration; base weights are early-specialized and fiber weights are symbolic mu_i"
                    .to_string()
            } else if primary_three_point {
                "computed by the fiber-equivariant twisted genus-zero Frobenius quantum product from the same hypergeometric/Birkhoff calibration; base weights are early-specialized and fiber weights are symbolic mu_i"
                    .to_string()
            } else {
                "computed by fiber-equivariant hypergeometric/Birkhoff S and QRR R stable-graph expansion; base weights are early-specialized and fiber weights are symbolic mu_i"
                    .to_string()
            }],
        });
    }

    let provider = TwistedProjectiveSpaceProvider::new(req.n, req.twist.degrees().to_vec(), false)?;
    if let Some((virtual_dimension, total_degree)) =
        twisted_dimension_mismatch(&provider, req.genus, req.degree, &req.insertions)
    {
        return Ok(InvariantResult {
            value: RatFun::zero(),
            engine: "twisted-negative-split-dimension",
            notes: vec![format!(
                "dimension mismatch gives zero: virtual dimension {virtual_dimension}, insertion degree {total_degree}"
            )],
        });
    }

    let unstable_pointed_curve = !pointed_curve_is_stable(req.genus, req.insertions.len());
    let primary_three_point = req.genus == 0 && genus_zero_three_primary_layout(&req.insertions);
    let value = provider.evaluate_nonequivariant_positive_degree(
        req.genus,
        req.degree,
        &req.insertions,
        req.truncation.as_ref(),
    )?;
    Ok(InvariantResult {
        value,
        engine: "twisted-negative-split-givental-birkhoff-early-line",
        notes: vec![if unstable_pointed_curve {
            "positive-degree unstable pointed correlator reconstructed by the full descendant divisor equation from the early rational one-parameter lambda-line hypergeometric/Birkhoff calibration; no local oracle shortcut is used"
                .to_string()
        } else if primary_three_point {
            "computed by the twisted genus-zero Frobenius quantum product from the same early rational hypergeometric/Birkhoff calibration"
                .to_string()
        } else {
            "computed by early rational one-parameter lambda-line hypergeometric/Birkhoff S and QRR R stable-graph expansion; no local oracle shortcut is used; fast validation currently covers resolved conifold genus 2 degree 1"
                .to_string()
        }],
    })
}

pub fn compute_negative_split_twisted_factored(
    req: &TwistedInvariantRequest,
) -> Result<FactoredRatFun, GwError> {
    req.validate()?;
    if !req.equivariant {
        return Err(GwError::UnsupportedInvariant(
            "factored twisted mode is currently for fiber-equivariant computations; pass --equivariant"
                .to_string(),
        ));
    }
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }
    let provider = FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(
        req.n,
        req.twist.degrees().to_vec(),
    )?;

    // The native factored provider has an exact unstable two-point S-matrix
    // convention.  Use it before the generic expanded recursion fallback:
    // equivariant coefficients need not satisfy nonequivariant dimension
    // equality, and expanding them here defeats the purpose of this API.
    if req.genus == 0 && req.insertions.len() == 2 {
        if let Some(value) =
            provider.factored_genus_zero_two_point_fallback(req.degree, &req.insertions)?
        {
            return Ok(value);
        }
    }
    if !pointed_curve_is_stable(req.genus, req.insertions.len()) {
        return compute_negative_split_twisted(req)
            .map(|result| FactoredRatFun::from_ratfun(result.value));
    }

    let dimension_matches =
        twisted_dimension_data(provider.inner(), req.genus, req.degree, &req.insertions)
            .is_some_and(|(virtual_dimension, total_degree)| {
                usize::try_from(virtual_dimension).ok() == Some(total_degree)
            });
    if !dimension_matches {
        return compute_negative_split_twisted(req)
            .map(|result| FactoredRatFun::from_ratfun(result.value));
    }
    compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
        &provider,
        req.genus,
        req.degree,
        &req.insertions,
        req.truncation.as_ref(),
    )
}

pub fn compute_negative_split_twisted_resolvent_packed(
    target_n: usize,
    degrees: Vec<usize>,
    req: &ResolventRequest,
    equivariant: bool,
) -> Result<ResolventResult, GwError> {
    if req.target_n != target_n {
        return Err(GwError::ConventionMismatch(format!(
            "twisted resolvent request targets P^{}, but the provider targets P^{target_n}",
            req.target_n
        )));
    }
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = if equivariant {
        TwistedProjectiveSpaceProvider::fiber_equivariant(target_n, degrees)?
    } else {
        TwistedProjectiveSpaceProvider::new(target_n, degrees, false)?
    };
    crate::spaces::projective_space::compute_packed_resolvent_with_provider(
        req,
        provider,
        if equivariant {
            "twisted-negative-split-fiber-equivariant-packed-resolvent"
        } else {
            "twisted-negative-split-packed-resolvent"
        },
        if equivariant {
            "computed by packed fiber-equivariant twisted S/R external-leg graph kernel; base weights are early-specialized and fiber weights are symbolic mu_i"
        } else {
            "computed by packed twisted S/R external-leg graph kernel; all resolvent coefficients share one stable-graph contraction"
        },
        move |raw| {
            if equivariant {
                return Ok(raw);
            }
            match raw.as_rational() {
                Some(value) => Ok(RatFun::from_rational(value)),
                None => Ok(RatFun::from_rational(
                    raw.nonequivariant_limit_line(0, &[Rational::one()])?,
                )),
            }
        },
    )
}

pub fn compute_negative_split_twisted_resolvent_packed_factored(
    target_n: usize,
    degrees: Vec<usize>,
    req: &ResolventRequest,
) -> Result<ResolventResult<FactoredRatFun>, GwError> {
    if req.target_n != target_n {
        return Err(GwError::ConventionMismatch(format!(
            "twisted resolvent request targets P^{}, but the provider targets P^{target_n}",
            req.target_n
        )));
    }
    if req.degree == 0 {
        return Err(GwError::UnsupportedInvariant(
            "degree-zero local invariants are not implemented in the negative split-bundle path"
                .to_string(),
        ));
    }

    let provider = FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(target_n, degrees)?;
    crate::spaces::projective_space::compute_packed_resolvent_with_coeff_provider(
        req,
        provider,
        "twisted-negative-split-fiber-equivariant-factored-packed-resolvent",
        "computed by packed fiber-equivariant twisted S/R external-leg graph kernel with factored symbolic coefficients; base weights are early-specialized and fiber weights are symbolic mu_i",
        Ok::<FactoredRatFun, GwError>,
    )
}

pub(crate) fn twisted_dimension_mismatch(
    provider: &TwistedProjectiveSpaceProvider,
    genus: usize,
    degree: usize,
    insertions: &[Insertion],
) -> Option<(isize, usize)> {
    let (virtual_dimension, total_degree) =
        twisted_dimension_data(provider, genus, degree, insertions)?;
    (usize::try_from(virtual_dimension).ok() != Some(total_degree))
        .then_some((virtual_dimension, total_degree))
}

fn twisted_dimension_data(
    provider: &TwistedProjectiveSpaceProvider,
    genus: usize,
    degree: usize,
    insertions: &[Insertion],
) -> Option<(isize, usize)> {
    let total_degree = provider.insertion_degree(insertions)?;
    let virtual_dimension = provider.virtual_dimension(genus, degree, insertions.len())?;
    Some((virtual_dimension, total_degree))
}

impl TwistedProjectiveSpaceProvider {
    pub fn new(n: usize, degrees: Vec<usize>, equivariant: bool) -> Result<Self, GwError> {
        require_nonempty_negative_split_twist(&degrees)?;
        // Twisted Euler factors need a more separated generic lambda line
        // than the ordinary provider's consecutive weights.  Preserve the
        // historical powers-of-two specialization through the checked public
        // calibration constructor; otherwise legitimate twists can acquire
        // accidental poles such as mu-3*lambda=0.
        let base = ProjectiveSpaceProvider::try_with_weights(
            n,
            equivariant,
            twisted_default_base_weights(n)?,
        )?;
        let canonical_theory = NegativeSplitTotalSpaceTheory::new(n, degrees)?;
        let twist = NegativeSplitBundleTwist::from_theory(&canonical_theory);
        Ok(Self {
            base,
            twist,
            canonical_theory,
            line_mode: TwistedLineMode::EarlyRational,
            calibration_mode: TwistedCalibrationMode::InverseEuler,
            fiber_weight_scale: Rational::one(),
            custom_fiber_weights: None,
            fiber_parameter_names: Vec::new(),
        })
    }

    /// Canonical geometry of the negative split-bundle total space.  This
    /// intentionally does not fabricate the twisted pairing or compact
    /// characteristic numbers.
    pub fn canonical_theory(&self) -> Result<&NegativeSplitTotalSpaceTheory, GwError> {
        Ok(&self.canonical_theory)
    }

    /// Whether this provider has exactly the calibration used by the compact
    /// positive-section identity.
    ///
    /// Euler twists, the alternate positive-fiber QRR convention, symbolic
    /// base-equivariant output, and fiber-equivariant output are distinct
    /// theories and must not be routed through the nonequivariant
    /// inverse-Euler completion adapter.
    pub fn supports_compact_completion_audit(&self) -> bool {
        self.line_mode == TwistedLineMode::EarlyRational
            && self.calibration_mode == TwistedCalibrationMode::InverseEuler
            && !self.base.is_equivariant()
    }

    pub(crate) fn validate_compact_completion_audit(&self) -> Result<(), GwError> {
        if self.supports_compact_completion_audit() {
            Ok(())
        } else {
            Err(GwError::ConventionMismatch(
                "compact-section audit requires the nonequivariant inverse-Euler calibration"
                    .to_string(),
            ))
        }
    }

    /// Evaluate one positive-degree, non-equivariant correlator using this
    /// provider's already constructed hypergeometric/Birkhoff/QRR
    /// calibration.
    ///
    /// This crate-visible entry point is used by independent audit adapters
    /// which must reuse the provider rather than silently reconstructing a
    /// second copy of its geometry and calibration.  Positive-degree stable
    /// maps whose underlying pointed curve is unstable are handled, for
    /// bounded descendants, by recursively adding divisor markings and
    /// applying the full descendant divisor equation.
    pub(crate) fn evaluate_nonequivariant_positive_degree(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<RatFun, GwError> {
        self.validate_compact_completion_audit()?;
        if degree == 0 {
            return Err(GwError::UnsupportedInvariant(
                "degree-zero local invariants are not supplied by the positive-section twisted evaluator"
                    .to_string(),
            ));
        }
        for (index, insertion) in insertions.iter().enumerate() {
            if insertion.class.n() != self.n() {
                return Err(GwError::ConventionMismatch(format!(
                    "twisted P^{} audit insertion {index} belongs to P^{}",
                    self.n(),
                    insertion.class.n()
                )));
            }
        }
        if twisted_dimension_mismatch(self, genus, degree, insertions).is_some() {
            return Ok(RatFun::zero());
        }

        self.evaluate_positive_degree_with_divisor_recursion(
            genus,
            degree,
            insertions.to_vec(),
            truncation,
        )
    }

    /// Evaluate the fixed-fiber-equivariant inverse-Euler theory, including
    /// every positive-degree unstable pointed range via descendant divisor
    /// recursion.  This is the provider-level implementation used by QRR
    /// audits; callers must not reproduce a partial stabilization policy.
    pub(crate) fn evaluate_fiber_equivariant_positive_degree(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<RatFun, GwError> {
        self.validate_fiber_equivariant_positive_degree_request(degree, insertions)?;
        if self.vanishes_above_base_virtual_dimension(genus, degree, insertions) {
            return Ok(RatFun::zero());
        }
        self.evaluate_positive_degree_with_divisor_recursion(
            genus,
            degree,
            insertions.to_vec(),
            truncation,
        )
    }

    /// Expanded fixed-fiber reference evaluator retained for backend
    /// comparison tests.
    ///
    /// Production QRR evaluation uses the native-factored provider below so
    /// the auxiliary base lambda remains unexpanded through the complete
    /// graph and divisor-recursion sum.
    #[cfg(test)]
    pub(crate) fn evaluate_qrr_fixed_fiber_positive_degree(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<RatFun, GwError> {
        self.validate_qrr_fixed_fiber_positive_degree_request(degree, insertions)?;
        if self.vanishes_above_base_virtual_dimension(genus, degree, insertions) {
            return Ok(RatFun::zero());
        }
        self.evaluate_positive_degree_with_divisor_recursion(
            genus,
            degree,
            insertions.to_vec(),
            truncation,
        )
    }

    /// Remove the auxiliary base-localization parameter from a complete
    /// fixed-fiber correlator.  The result is exact rational arithmetic;
    /// uncancelled poles fail through [`GwError::NonFiniteLimit`].
    pub(crate) fn qrr_fixed_fiber_lambda_line_limit(
        &self,
        value: &RatFun,
    ) -> Result<RatFun, GwError> {
        if self.line_mode != TwistedLineMode::FixedFiberLambdaLine {
            return Err(GwError::ConventionMismatch(
                "fixed-fiber lambda-line limit requires a fixed-fiber QRR provider".to_string(),
            ));
        }
        Ok(RatFun::from_rational(value.nonequivariant_limit_line(
            self.n(),
            self.base_weights(),
        )?))
    }

    fn validate_fiber_equivariant_positive_degree_request(
        &self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<(), GwError> {
        if self.line_mode != TwistedLineMode::FiberEquivariant
            || self.calibration_mode != TwistedCalibrationMode::InverseEuler
            || self.base.is_equivariant()
        {
            return Err(GwError::ConventionMismatch(
                "fiber-equivariant positive-degree evaluation requires the inverse-Euler calibration with only the fiber weights symbolic"
                    .to_string(),
            ));
        }
        if degree == 0 {
            return Err(GwError::UnsupportedInvariant(
                "the positive-degree fiber-equivariant evaluator does not supply constant maps"
                    .to_string(),
            ));
        }
        for (index, insertion) in insertions.iter().enumerate() {
            if insertion.class.n() != self.n() {
                return Err(GwError::ConventionMismatch(format!(
                    "twisted P^{} audit insertion {index} belongs to P^{}",
                    self.n(),
                    insertion.class.n()
                )));
            }
        }
        Ok(())
    }

    fn validate_qrr_fixed_fiber_positive_degree_request(
        &self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<(), GwError> {
        if self.line_mode != TwistedLineMode::FixedFiberLambdaLine
            || self.calibration_mode != TwistedCalibrationMode::InverseEuler
            || self.base.is_equivariant()
        {
            return Err(GwError::ConventionMismatch(
                "fixed-fiber QRR evaluation requires the inverse-Euler calibration with symbolic base lambda-line weights and rational fiber weights"
                    .to_string(),
            ));
        }
        if degree == 0 {
            return Err(GwError::UnsupportedInvariant(
                "the positive-degree fixed-fiber QRR evaluator does not supply constant maps"
                    .to_string(),
            ));
        }
        for (index, insertion) in insertions.iter().enumerate() {
            if insertion.class.n() != self.n() {
                return Err(GwError::ConventionMismatch(format!(
                    "twisted P^{} audit insertion {index} belongs to P^{}",
                    self.n(),
                    insertion.class.n()
                )));
            }
        }
        Ok(())
    }

    /// Exact upper-bound pruning for the fixed-fiber inverse-Euler theory
    /// after the auxiliary base lambda-line limit.
    ///
    /// A codimension-`j` term of `1/e(R pi_* f^* E)` has `j >= 0`; its
    /// remaining homogeneous degree is carried by a fiber-parameter
    /// coefficient of degree `-chi(E)-j`.  Therefore insertions may lie
    /// *below* the base stable-map virtual dimension, but never above it.  This
    /// is deliberately an inequality, not the nonequivariant dimension-equality
    /// rule.
    pub(crate) fn vanishes_above_base_virtual_dimension(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Insertion],
    ) -> bool {
        let Some(insertion_degree) =
            <ProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::insertion_degree(
                &self.base, insertions,
            )
        else {
            return false;
        };
        let Some(virtual_dimension) = <ProjectiveSpaceProvider as SemisimpleCohftProvider<
            RatFun,
        >>::virtual_dimension(
            &self.base, genus, degree, insertions.len()
        ) else {
            return false;
        };
        virtual_dimension < 0
            || usize::try_from(virtual_dimension)
                .is_ok_and(|dimension| insertion_degree > dimension)
    }

    fn evaluate_positive_degree_with_divisor_recursion(
        &self,
        genus: usize,
        degree: usize,
        insertions: Vec<Insertion>,
        truncation: Option<&Truncation>,
    ) -> Result<RatFun, GwError> {
        self.evaluate_positive_degree_with_divisor_recursion_coeff(
            genus,
            degree,
            insertions,
            &|stable_insertions| {
                let raw = crate::givental::compute_semisimple_graph_value(
                    self,
                    genus,
                    degree,
                    stable_insertions,
                    truncation,
                )?;
                match self.line_mode {
                    TwistedLineMode::EarlyRational => match raw.as_rational() {
                        Some(value) => Ok(RatFun::from_rational(value)),
                        None => Ok(RatFun::from_rational(
                            raw.nonequivariant_limit_line(0, &[Rational::one()])?,
                        )),
                    },
                    TwistedLineMode::FiberEquivariant => {
                        raw.lambda_line_limit_preserving_variables(self.n(), self.base_weights())
                    }
                    TwistedLineMode::FixedFiberLambdaLine => {
                        self.qrr_fixed_fiber_lambda_line_limit(&raw)
                    }
                    TwistedLineMode::SymbolicLimit => Err(GwError::ConventionMismatch(
                        "positive-degree divisor recursion has no specialization policy for the symbolic base lambda-line mode"
                            .to_string(),
                    )),
                }
            },
        )
    }

    /// Apply the descendant divisor equation in one coefficient ring.
    ///
    /// The stabilization policy belongs to the target provider, while the
    /// stable correlator itself may be evaluated by either the expanded or the
    /// native-factored semisimple engine.  Keeping the recursion here prevents
    /// those two arithmetic backends from acquiring independent mathematical
    /// conventions.
    ///
    /// In the stable range this also removes exact primary insertions of the
    /// provider's stabilizing divisor using the full descendant divisor
    /// equation.  The principal term is multiplied by the curve pairing, and
    /// every other `tau_k(gamma)` with `k > 0` contributes the positive
    /// correction `tau_{k-1}(H cup gamma)`.  We only remove a marking when the
    /// reduced pointed curve remains stable.  Every resulting recursive call
    /// has one fewer marking (and each correction also has one less total psi
    /// degree), while unstable reconstruction stops at a stable boundary from
    /// which removal is disallowed; the two directions can never cycle.
    fn evaluate_positive_degree_with_divisor_recursion_coeff<C, F>(
        &self,
        genus: usize,
        degree: usize,
        insertions: Vec<Insertion>,
        evaluate_stable: &F,
    ) -> Result<C, GwError>
    where
        C: Coeff,
        F: Fn(&[Insertion]) -> Result<C, GwError>,
    {
        if pointed_curve_is_stable(genus, insertions.len()) {
            if let Some((reduced, divisor_basis, divisor_pairing)) =
                self.stable_primary_divisor_reduction(genus, degree, &insertions)?
            {
                let reduced_value = self.evaluate_positive_degree_with_divisor_recursion_coeff(
                    genus,
                    degree,
                    reduced.clone(),
                    evaluate_stable,
                )?;
                let mut value =
                    reduced_value.mul(&C::from_rational(Rational::from(divisor_pairing)));
                for index in 0..reduced.len() {
                    if reduced[index].descendant_power == 0 {
                        continue;
                    }
                    let product =
                        self.classical_multiply_by_basis(divisor_basis, &reduced[index].class)?;
                    if product.coeffs().iter().all(RatFun::is_zero) {
                        continue;
                    }
                    let mut correction = reduced.clone();
                    correction[index].descendant_power -= 1;
                    correction[index].class = product;
                    let correction_value = self
                        .evaluate_positive_degree_with_divisor_recursion_coeff(
                            genus,
                            degree,
                            correction,
                            evaluate_stable,
                        )?;
                    value = value.add(&correction_value);
                }
                return Ok(value);
            }
            return evaluate_stable(&insertions);
        }

        let total_descendant_power = insertions.iter().try_fold(0usize, |total, insertion| {
            total
                .checked_add(insertion.descendant_power)
                .ok_or_else(|| {
                    GwError::UnsupportedInvariant(
                        "twisted divisor-recursion descendant degree overflow".to_string(),
                    )
                })
        })?;
        let maximum = MAX_UNSTABLE_DIVISOR_RECURSION_TOTAL_PSI;
        if total_descendant_power > maximum {
            return Err(GwError::UnsupportedInvariant(format!(
                "twisted unstable descendant degree {total_descendant_power} exceeds the divisor-recursion implementation bound {maximum}"
            )));
        }

        let curve = self.canonical_theory.try_curve(degree)?;
        let (divisor_basis, divisor_pairing) = self
            .canonical_theory
            .stabilizing_divisor(&curve)?
            .filter(|(_, pairing)| *pairing > 0)
            .ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "negative-split theory has no positive stabilizing divisor for this curve"
                        .to_string(),
                )
            })?;
        let divisor = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(
                self.n(),
                divisor_basis.0,
            )?,
        );
        let mut with_divisor = insertions.clone();
        with_divisor.push(divisor);
        let mut numerator = self.evaluate_positive_degree_with_divisor_recursion_coeff(
            genus,
            degree,
            with_divisor,
            evaluate_stable,
        )?;
        for index in 0..insertions.len() {
            if insertions[index].descendant_power == 0 {
                continue;
            }
            let product =
                self.classical_multiply_by_basis(divisor_basis, &insertions[index].class)?;
            if product.coeffs().iter().all(RatFun::is_zero) {
                continue;
            }
            let mut correction = insertions.clone();
            correction[index].descendant_power -= 1;
            correction[index].class = product;
            let correction_value = self.evaluate_positive_degree_with_divisor_recursion_coeff(
                genus,
                degree,
                correction,
                evaluate_stable,
            )?;
            numerator = numerator.sub(&correction_value);
        }
        Ok(numerator.div(&C::from_rational(Rational::from(divisor_pairing))))
    }

    /// Identify one exact, terminating primary-divisor reduction in the stable
    /// range.
    ///
    /// The inverse-Euler characteristic class is pulled back from the
    /// universal map, so it does not change the usual divisor equation.  If
    /// `H` is the canonical stabilizing divisor, then
    ///
    /// `<H, tau_{k_1}(gamma_1), ...>_{g,d} = (H.d)
    ///     <tau_{k_1}(gamma_1), ...>_{g,d}
    ///     + sum_i <..., tau_{k_i-1}(H cup gamma_i), ...>_{g,d}`.
    ///
    /// This helper only removes the distinguished primary `H`; its caller
    /// constructs every positive correction in the displayed formula.
    fn stable_primary_divisor_reduction(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<Option<(Vec<Insertion>, BasisId, i64)>, GwError> {
        let Some(reduced_markings) = insertions.len().checked_sub(1) else {
            return Ok(None);
        };
        if !pointed_curve_is_stable(genus, reduced_markings) {
            return Ok(None);
        }

        let curve = self.canonical_theory.try_curve(degree)?;
        let Some((divisor_basis, divisor_pairing)) = self
            .canonical_theory
            .stabilizing_divisor(&curve)?
            .filter(|(_, pairing)| *pairing > 0)
        else {
            return Ok(None);
        };
        let Some(index) = insertions.iter().position(|insertion| {
            insertion.descendant_power == 0 && insertion.class.pure_power() == Some(divisor_basis.0)
        }) else {
            return Ok(None);
        };

        let mut reduced = insertions.to_vec();
        reduced.remove(index);
        Ok(Some((reduced, divisor_basis, divisor_pairing)))
    }

    fn classical_multiply_by_basis(
        &self,
        left: BasisId,
        right: &crate::spaces::projective_space::CohomologyClass,
    ) -> Result<crate::spaces::projective_space::CohomologyClass, GwError> {
        let mut coefficients = vec![RatFun::zero(); self.n() + 1];
        for (right_basis, coefficient) in right.coeffs().iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            for (output, scalar) in self
                .canonical_theory
                .classical_product(left, BasisId(right_basis))?
            {
                let contribution = coefficient * &RatFun::from_rational(scalar);
                coefficients[output.0] = &coefficients[output.0] + &contribution;
            }
        }
        crate::spaces::projective_space::CohomologyClass::try_new(self.n(), coefficients)
    }

    pub fn fiber_equivariant(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.line_mode = TwistedLineMode::FiberEquivariant;
        out.fiber_parameter_names = default_fiber_parameter_names(out.twist.rank());
        Ok(out)
    }

    /// Expanded QRR provider at a fixed rational fiber-equivariant point.
    ///
    /// The line-summand weights are paired with the input `degrees` and then
    /// reordered by the canonical target theory.  Base localization weights
    /// remain `lambda_i = w_i lambda_0` until a complete correlator is summed;
    /// the fiber weights are not scaled with `lambda_0`.  Consequently every
    /// intermediate [`RatFun`] is univariate in the auxiliary base parameter,
    /// while retaining the same inverse-Euler CohFT specialization as
    /// `mu_i -> fiber_weights[i]` in the fully symbolic theory.
    pub fn qrr_fixed_fiber_lambda_line(
        n: usize,
        degrees: Vec<usize>,
        fiber_weights: Vec<Rational>,
    ) -> Result<Self, GwError> {
        if degrees.len() != fiber_weights.len() {
            return Err(GwError::ConventionMismatch(format!(
                "twist rank {} does not match {} fixed fiber weights",
                degrees.len(),
                fiber_weights.len()
            )));
        }
        let mut out = Self::new(n, degrees.clone(), false)?;
        let fiber_weights = out
            .canonical_theory
            .canonicalize_summand_payloads(degrees, fiber_weights)?;
        if let Some(index) = fiber_weights.iter().position(Rational::is_zero) {
            return Err(GwError::ConventionMismatch(format!(
                "fixed inverse-Euler fiber weight mu_{index} must be nonzero"
            )));
        }
        out.line_mode = TwistedLineMode::FixedFiberLambdaLine;
        out.fiber_parameter_names = default_fiber_parameter_names(out.twist.rank());
        out.custom_fiber_weights = Some(fiber_weights);
        Ok(out)
    }

    /// The exact symbolic-parameter point represented by a fixed-fiber QRR
    /// provider.
    pub fn fixed_fiber_parameter_assignments(&self) -> Result<BTreeMap<String, Rational>, GwError> {
        if self.line_mode != TwistedLineMode::FixedFiberLambdaLine {
            return Err(GwError::ConventionMismatch(
                "fiber-parameter assignments require a fixed-fiber QRR provider".to_string(),
            ));
        }
        Ok(self
            .fiber_parameter_names
            .iter()
            .cloned()
            .zip(self.rational_fiber_weights())
            .collect())
    }

    pub fn symbolic_lambda_line(
        n: usize,
        degrees: Vec<usize>,
        equivariant: bool,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, equivariant)?;
        out.line_mode = TwistedLineMode::SymbolicLimit;
        Ok(out)
    }

    pub fn euler_twist(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.calibration_mode = TwistedCalibrationMode::Euler;
        Ok(out)
    }

    pub fn inverse_euler_with_positive_fiber_qrr(
        n: usize,
        degrees: Vec<usize>,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.calibration_mode = TwistedCalibrationMode::InverseEulerFiberPlus;
        Ok(out)
    }

    fn descendant_s_cache_key(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> TwistedDescendantSCacheKey {
        TwistedDescendantSCacheKey {
            n: self.base.n(),
            twist_degrees: self.twist.degrees().to_vec(),
            q_degree,
            z_order,
            base_weights: self.base_weights().to_vec(),
            fiber_weights: self.rational_fiber_weights(),
            fiber_parameter_names: self.fiber_parameter_names.clone(),
            base_equivariant: self.base.is_equivariant(),
            line_mode: self.line_mode.clone(),
            calibration_mode: self.calibration_mode.clone(),
        }
    }

    /// Single construction path for the raw Birkhoff fundamental solution.
    /// Quantum-product reconstruction consumes this convention directly;
    /// graph legs instead use its metric adjoint below.
    fn compute_raw_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        let rational_fiber_weights = self.rational_fiber_weights();
        match self.line_mode {
            TwistedLineMode::EarlyRational => {
                NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
                    self.base.n(),
                    self.twist.clone(),
                    q_degree,
                    z_order,
                    self.base_weights().to_vec(),
                    rational_fiber_weights,
                )?
                .birkhoff_descendant_s_matrix(z_order)
            }
            TwistedLineMode::SymbolicLimit
            | TwistedLineMode::FiberEquivariant
            | TwistedLineMode::FixedFiberLambdaLine => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                NegativeSplitLineHypergeometricModel::from_ratfun_weights(
                    self.base.n(),
                    self.twist.clone(),
                    q_degree,
                    z_order,
                    base_weights.clone(),
                    &fiber_weights,
                )?
                .birkhoff_descendant_s_matrix(z_order)
            }
        }
    }

    fn cached_raw_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<Arc<SeriesSMatrix>, GwError> {
        static CACHE: OnceLock<
            Mutex<BoundedCache<TwistedDescendantSCacheKey, Arc<SeriesSMatrix>>>,
        > = OnceLock::new();
        let key = self.descendant_s_cache_key(q_degree, z_order);
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
        if let Some((_, matrix)) = minimal_covering_descendant_s(&cache.lock().unwrap(), &key) {
            return Ok(matrix);
        }
        let matrix = Arc::new(self.compute_raw_descendant_s_matrix(q_degree, z_order)?);
        cache.lock().unwrap().insert(key, matrix.clone());
        Ok(matrix)
    }

    /// Graph legs consume the metric adjoint of the raw Birkhoff solution.
    /// Keeping that conversion distinct is essential: the raw matrix is also
    /// the input to quantum-product reconstruction.
    fn compute_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        let raw_s = self.cached_raw_descendant_s_matrix(q_degree, z_order)?;
        let flat_metric = self.flat_metric_matrix(q_degree)?;
        metric_adjoint_descendant_s_matrix(raw_s.as_ref().clone(), &flat_metric)
    }

    fn cached_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<Arc<SeriesSMatrix>, GwError> {
        static CACHE: OnceLock<
            Mutex<BoundedCache<TwistedDescendantSCacheKey, Arc<SeriesSMatrix>>>,
        > = OnceLock::new();
        let key = self.descendant_s_cache_key(q_degree, z_order);
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
        if let Some((cached_z_order, matrix)) =
            minimal_covering_descendant_s(&cache.lock().unwrap(), &key)
        {
            if crate::env_flag("GW_PROFILE") {
                eprintln!(
                    "GW_PROFILE twisted_descendant_s_cache=hit q_degree={} z_order={} cached_z_order={}",
                    q_degree, z_order, cached_z_order
                );
            }
            return Ok(matrix);
        }

        let started = std::time::Instant::now();
        let matrix = Arc::new(self.compute_descendant_s_matrix(q_degree, z_order)?);
        if crate::env_flag("GW_PROFILE") {
            eprintln!(
                "GW_PROFILE twisted_descendant_s_cache=miss q_degree={} z_order={} build={:.3}s",
                q_degree,
                z_order,
                started.elapsed().as_secs_f64()
            );
        }
        cache.lock().unwrap().insert(key, matrix.clone());
        Ok(matrix)
    }

    fn flat_metric_matrix(&self, q_degree: usize) -> Result<SeriesMatrix, GwError> {
        let fiber_weights = self.rational_fiber_weights();
        match self.line_mode {
            TwistedLineMode::EarlyRational => twisted_inverse_euler_flat_metric_matrix(
                self.base.n(),
                q_degree,
                &self.twist,
                self.base_weights(),
                &fiber_weights,
            ),
            TwistedLineMode::SymbolicLimit
            | TwistedLineMode::FiberEquivariant
            | TwistedLineMode::FixedFiberLambdaLine => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                twisted_inverse_euler_flat_metric_matrix_ratfun(
                    self.base.n(),
                    q_degree,
                    &self.twist,
                    &base_weights,
                    &fiber_weights,
                )
            }
        }
    }

    fn genus_zero_two_point_fallback(
        &self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<Option<RatFun>, GwError> {
        let Some((descendant_idx, primary_idx, s_order)) =
            genus_zero_two_point_descendant_layout(insertions)
        else {
            return Ok(None);
        };
        let s_matrix = self.cached_descendant_s_matrix(degree, s_order)?;
        let metric = self.flat_metric_matrix(degree)?;
        let descendant = self.insertion_vector(&insertions[descendant_idx], degree)?;
        let primary = self.insertion_vector(&insertions[primary_idx], degree)?;
        genus_zero_two_point_s_matrix_pairing_coeff(
            self.colors(),
            degree,
            s_order,
            s_matrix.as_ref(),
            &metric,
            &descendant,
            &primary,
        )
        .map(Some)
    }

    pub fn rational_lambda_line_with_scale(
        n: usize,
        degrees: Vec<usize>,
        scale: Rational,
    ) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        for weight in &mut out.base.weights {
            *weight = weight.clone() * scale.clone();
        }
        out.fiber_weight_scale = scale;
        Ok(out)
    }

    pub fn rational_lambda_line_with_weights(
        n: usize,
        degrees: Vec<usize>,
        base_weights: Vec<Rational>,
        fiber_weights: Vec<Rational>,
    ) -> Result<Self, GwError> {
        if degrees.len() != fiber_weights.len() {
            return Err(GwError::ConventionMismatch(format!(
                "twist rank {} does not match {} custom fiber weights",
                degrees.len(),
                fiber_weights.len()
            )));
        }
        let mut out = Self::new(n, degrees.clone(), false)?;
        // Preserve the association between a line summand and its custom
        // equivariant weight using the target theory's canonical order.
        let fiber_weights = out
            .canonical_theory
            .canonicalize_summand_payloads(degrees, fiber_weights)?;
        validate_twisted_weights(n, &out.twist, &base_weights, &fiber_weights)?;
        out.base.weights = base_weights;
        out.fiber_weight_scale = Rational::one();
        out.custom_fiber_weights = Some(fiber_weights);
        Ok(out)
    }

    pub fn n(&self) -> usize {
        self.base.n()
    }

    pub fn twist(&self) -> &NegativeSplitBundleTwist {
        &self.twist
    }

    pub(crate) fn base_weights(&self) -> &[Rational] {
        &self.base.weights
    }

    fn rational_fiber_weights(&self) -> Vec<Rational> {
        if let Some(weights) = &self.custom_fiber_weights {
            return weights.clone();
        }
        // Compute the historical generic seed 2^(n+1)-1 in the arbitrary-
        // precision coefficient ring.  A machine-word shift panics once
        // n+1 reaches usize::BITS even though such state spaces are otherwise
        // representable by the fallible provider constructor.
        let start = Rational::from(2).pow_usize(self.base.n() + 1) - Rational::one();
        (0..self.twist.rank())
            .map(|idx| {
                (start.clone() + Rational::from(2) * Rational::from(idx))
                    * self.fiber_weight_scale.clone()
            })
            .collect()
    }

    fn ratfun_base_weights(&self) -> Vec<RatFun> {
        match self.line_mode {
            TwistedLineMode::SymbolicLimit
            | TwistedLineMode::FiberEquivariant
            | TwistedLineMode::FixedFiberLambdaLine => {
                let lambda = crate::core::algebra::lambda(0);
                self.base_weights()
                    .iter()
                    .map(|weight| lambda.clone() * RatFun::from_rational(weight.clone()))
                    .collect()
            }
            TwistedLineMode::EarlyRational => self
                .base_weights()
                .iter()
                .cloned()
                .map(RatFun::from_rational)
                .collect(),
        }
    }

    fn ratfun_fiber_weights(&self) -> Vec<RatFun> {
        match self.line_mode {
            TwistedLineMode::SymbolicLimit => {
                let lambda = crate::core::algebra::lambda(0);
                self.rational_fiber_weights()
                    .into_iter()
                    .map(|weight| lambda.clone() * RatFun::from_rational(weight))
                    .collect()
            }
            TwistedLineMode::FiberEquivariant => self
                .fiber_parameter_names
                .iter()
                .cloned()
                .map(RatFun::variable)
                .collect(),
            TwistedLineMode::FixedFiberLambdaLine => self
                .rational_fiber_weights()
                .into_iter()
                .map(RatFun::from_rational)
                .collect(),
            TwistedLineMode::EarlyRational => self
                .rational_fiber_weights()
                .into_iter()
                .map(RatFun::from_rational)
                .collect(),
        }
    }
}

fn twisted_default_base_weights(n: usize) -> Result<Vec<Rational>, GwError> {
    let size = n.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant("twisted base dimension is too large".to_string())
    })?;
    let mut weights = Vec::new();
    weights.try_reserve_exact(size).map_err(|_| {
        GwError::UnsupportedInvariant(format!("cannot allocate {size} twisted base weights"))
    })?;
    let mut weight = Rational::one();
    for _ in 0..size {
        weights.push(weight.clone());
        weight = weight * Rational::from(2);
    }
    Ok(weights)
}

pub(crate) fn genus_zero_two_point_descendant_layout(
    insertions: &[Insertion],
) -> Option<(usize, usize, usize)> {
    if insertions.len() != 2 {
        return None;
    }
    let descendant_positions = insertions
        .iter()
        .enumerate()
        .filter_map(|(idx, insertion)| (insertion.descendant_power > 0).then_some(idx))
        .collect::<Vec<_>>();
    if descendant_positions.len() != 1 {
        return None;
    }

    let descendant_idx = descendant_positions[0];
    let primary_idx = 1 - descendant_idx;
    let s_order = insertions[descendant_idx].descendant_power + 1;
    Some((descendant_idx, primary_idx, s_order))
}

pub(crate) fn genus_zero_two_point_s_matrix_pairing_coeff<C: Coeff>(
    colors: usize,
    degree: usize,
    s_order: usize,
    s_matrix: &SeriesSMatrix<C>,
    metric: &SeriesMatrix<C>,
    descendant: &[QSeries<C>],
    primary: &[QSeries<C>],
) -> Result<C, GwError> {
    let s_coeff = s_matrix
        .coefficient(s_order)
        .ok_or(GwError::TruncationTooLow)?;

    let mut transformed = vec![QSeries::<C>::zero(degree); colors];
    for (row, target) in transformed.iter_mut().enumerate() {
        let mut total = QSeries::<C>::zero(degree);
        for (col, class_coeff) in descendant.iter().enumerate() {
            total = total.add(&s_coeff.entry(row, col).mul(class_coeff));
        }
        *target = total;
    }

    let mut paired = QSeries::<C>::zero(degree);
    for (left, transformed_coeff) in transformed.iter().enumerate() {
        for (right, primary_coeff) in primary.iter().enumerate() {
            let term = transformed_coeff
                .mul(metric.entry(left, right))
                .mul(primary_coeff);
            paired = paired.add(&term);
        }
    }
    Ok(paired.coeff(degree).cloned().unwrap_or_else(C::zero))
}

pub(crate) fn genus_zero_two_point_raw_s_matrix_pairing_coeff<C: Coeff>(
    colors: usize,
    degree: usize,
    s_order: usize,
    raw_s_matrix: &SeriesSMatrix<C>,
    metric: &SeriesMatrix<C>,
    descendant: &[QSeries<C>],
    primary: &[QSeries<C>],
) -> Result<C, GwError> {
    let s_coeff = raw_s_matrix
        .coefficient(s_order)
        .ok_or(GwError::TruncationTooLow)?;

    let mut metric_descendant = vec![QSeries::<C>::zero(degree); colors];
    for (row, target) in metric_descendant.iter_mut().enumerate() {
        let mut total = QSeries::<C>::zero(degree);
        for (col, class_coeff) in descendant.iter().enumerate() {
            if metric.entry(row, col).is_structurally_zero() || class_coeff.is_structurally_zero() {
                continue;
            }
            total = total.add(&metric.entry(row, col).mul(class_coeff));
        }
        *target = total;
    }

    let mut paired = QSeries::<C>::zero(degree);
    for (row, metric_descendant_coeff) in metric_descendant.iter().enumerate() {
        if metric_descendant_coeff.is_structurally_zero() {
            continue;
        }
        for (col, primary_coeff) in primary.iter().enumerate() {
            if s_coeff.entry(row, col).is_structurally_zero()
                || primary_coeff.is_structurally_zero()
            {
                continue;
            }
            let term = metric_descendant_coeff
                .mul(s_coeff.entry(row, col))
                .mul(primary_coeff);
            paired = paired.add(&term);
        }
    }
    Ok(paired.coeff(degree).cloned().unwrap_or_else(C::zero))
}

pub(crate) fn genus_zero_three_primary_layout(insertions: &[Insertion]) -> bool {
    insertions.len() == 3
        && insertions
            .iter()
            .all(|insertion| insertion.descendant_power == 0)
}

fn twisted_genus_zero_three_primary_value_from_s_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    calibration_mode: &TwistedCalibrationMode,
    base_weights: &[C],
    fiber_weights: &[C],
    descendant_s: &SeriesSMatrix<C>,
    insertions: &[Vec<QSeries<C>>],
) -> Result<C, GwError> {
    debug_assert_eq!(insertions.len(), 3);
    let classical_h = twisted_classical_h_multiplication_matrix_coeff(n, degree, base_weights)?;
    let quantum_h =
        twisted_quantum_multiplication_from_s_coeff(descendant_s, &classical_h, calibration_mode)?;
    let metric = twisted_inverse_euler_flat_metric_matrix_coeff(
        n,
        degree,
        twist,
        base_weights,
        fiber_weights,
    )?;
    let quantum_algebra =
        CyclicQuantumAlgebra::try_new(quantum_h, "twisted quantum H multiplication")?;
    let left_cyclic_coordinates: CyclicCoordinates<C> =
        quantum_algebra.coordinates(&insertions[0])?;
    let product = quantum_algebra.left_product(&left_cyclic_coordinates, &insertions[1])?;
    pair_vectors_coeff(&metric, &product, &insertions[2], degree)
}

pub(crate) fn pair_vectors_coeff<C: Coeff>(
    metric: &SeriesMatrix<C>,
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    degree: usize,
) -> Result<C, GwError> {
    let colors = metric.rows();
    debug_assert_eq!(metric.cols(), colors);
    debug_assert_eq!(left.len(), colors);
    debug_assert_eq!(right.len(), colors);
    let mut paired = QSeries::<C>::zero(degree);
    for (left_idx, left_coeff) in left.iter().enumerate() {
        if left_coeff.is_structurally_zero() {
            continue;
        }
        for (right_idx, right_coeff) in right.iter().enumerate() {
            if right_coeff.is_structurally_zero()
                || metric.entry(left_idx, right_idx).is_structurally_zero()
            {
                continue;
            }
            paired = paired.add(
                &left_coeff
                    .mul(metric.entry(left_idx, right_idx))
                    .mul(right_coeff),
            );
        }
    }
    Ok(paired.coeff(degree).cloned().unwrap_or_else(C::zero))
}

pub(crate) fn default_fiber_parameter_names(rank: usize) -> Vec<String> {
    (0..rank).map(|idx| format!("mu_{idx}")).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoredTwistedProjectiveSpaceProvider {
    inner: TwistedProjectiveSpaceProvider,
    base_weight_mode: FactoredBaseWeightMode,
}

/// The public factored API historically fixes the auxiliary base weights at a
/// generic rational semisimple point.  QRR needs a different, internal mode:
/// keep the common base localization parameter symbolic until the complete
/// correlator has been summed, then take its lambda-line limit once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FactoredBaseWeightMode {
    RationalSpecialization,
    SymbolicLambdaLine,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FactoredTwistedDescendantSCacheKey {
    n: usize,
    twist_degrees: Vec<usize>,
    q_degree: usize,
    z_order: usize,
    base_weights: Vec<Rational>,
    fiber_weights: Vec<Rational>,
    fiber_parameter_names: Vec<String>,
    base_equivariant: bool,
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
    base_weight_mode: FactoredBaseWeightMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FactoredTwistedGraphKernelCacheKey {
    n: usize,
    twist_degrees: Vec<usize>,
    q_degree: usize,
    r_order: usize,
    graph_dimension: usize,
    base_weights: Vec<Rational>,
    fiber_weights: Vec<Rational>,
    fiber_parameter_names: Vec<String>,
    base_equivariant: bool,
    line_mode: TwistedLineMode,
    calibration_mode: TwistedCalibrationMode,
    base_weight_mode: FactoredBaseWeightMode,
    validation: TwistedCalibrationValidation,
}

/// Cache keys whose truncation may safely be restricted after construction.
///
/// The family comparison deliberately excludes only the truncation order.  In
/// particular, Novikov degree, geometry, coefficient conventions, and
/// calibration-validation mode must agree exactly.
trait DominatingDescendantSCacheKey {
    fn z_order(&self) -> usize;
    fn same_family(&self, other: &Self) -> bool;
}

impl DominatingDescendantSCacheKey for TwistedDescendantSCacheKey {
    fn z_order(&self) -> usize {
        self.z_order
    }

    fn same_family(&self, other: &Self) -> bool {
        self.n == other.n
            && self.twist_degrees == other.twist_degrees
            && self.q_degree == other.q_degree
            && self.base_weights == other.base_weights
            && self.fiber_weights == other.fiber_weights
            && self.fiber_parameter_names == other.fiber_parameter_names
            && self.base_equivariant == other.base_equivariant
            && self.line_mode == other.line_mode
            && self.calibration_mode == other.calibration_mode
    }
}

impl DominatingDescendantSCacheKey for FactoredTwistedDescendantSCacheKey {
    fn z_order(&self) -> usize {
        self.z_order
    }

    fn same_family(&self, other: &Self) -> bool {
        self.n == other.n
            && self.twist_degrees == other.twist_degrees
            && self.q_degree == other.q_degree
            && self.base_weights == other.base_weights
            && self.fiber_weights == other.fiber_weights
            && self.fiber_parameter_names == other.fiber_parameter_names
            && self.base_equivariant == other.base_equivariant
            && self.line_mode == other.line_mode
            && self.calibration_mode == other.calibration_mode
            && self.base_weight_mode == other.base_weight_mode
    }
}

fn minimal_covering_descendant_s<K, C>(
    cache: &BoundedCache<K, Arc<SeriesSMatrix<C>>>,
    requested: &K,
) -> Option<(usize, Arc<SeriesSMatrix<C>>)>
where
    K: DominatingDescendantSCacheKey,
{
    cache
        .iter()
        .filter(|(candidate, _)| {
            candidate.same_family(requested) && candidate.z_order() >= requested.z_order()
        })
        .min_by_key(|(candidate, _)| candidate.z_order() - requested.z_order())
        .map(|(key, matrix)| (key.z_order(), matrix.clone()))
}

trait DominatingGraphKernelCacheKey {
    fn r_order(&self) -> usize;
    fn graph_dimension(&self) -> usize;
    fn same_family(&self, other: &Self) -> bool;
}

impl DominatingGraphKernelCacheKey for TwistedGraphKernelCacheKey {
    fn r_order(&self) -> usize {
        self.r_order
    }

    fn graph_dimension(&self) -> usize {
        self.graph_dimension
    }

    fn same_family(&self, other: &Self) -> bool {
        self.n == other.n
            && self.twist_degrees == other.twist_degrees
            && self.q_degree == other.q_degree
            && self.base_weights == other.base_weights
            && self.fiber_weights == other.fiber_weights
            && self.fiber_parameter_names == other.fiber_parameter_names
            && self.line_mode == other.line_mode
            && self.calibration_mode == other.calibration_mode
            && self.validation == other.validation
    }
}

impl DominatingGraphKernelCacheKey for FactoredTwistedGraphKernelCacheKey {
    fn r_order(&self) -> usize {
        self.r_order
    }

    fn graph_dimension(&self) -> usize {
        self.graph_dimension
    }

    fn same_family(&self, other: &Self) -> bool {
        self.n == other.n
            && self.twist_degrees == other.twist_degrees
            && self.q_degree == other.q_degree
            && self.base_weights == other.base_weights
            && self.fiber_weights == other.fiber_weights
            && self.fiber_parameter_names == other.fiber_parameter_names
            && self.base_equivariant == other.base_equivariant
            && self.line_mode == other.line_mode
            && self.calibration_mode == other.calibration_mode
            && self.base_weight_mode == other.base_weight_mode
            && self.validation == other.validation
    }
}

fn minimal_covering_graph_kernel<K, C>(
    cache: &BoundedCache<K, Arc<GiventalGraphKernel<C>>>,
    requested: &K,
) -> Option<(usize, usize, Arc<GiventalGraphKernel<C>>)>
where
    K: DominatingGraphKernelCacheKey,
{
    cache
        .iter()
        .filter(|(candidate, _)| {
            candidate.same_family(requested)
                && candidate.r_order() >= requested.r_order()
                && candidate.graph_dimension() >= requested.graph_dimension()
        })
        // Prefer the least total overbuild; ties are broken by R-order and
        // then graph dimension, so selection does not depend on hash order.
        .min_by_key(|(candidate, _)| {
            let r_excess = candidate.r_order() - requested.r_order();
            let graph_excess = candidate.graph_dimension() - requested.graph_dimension();
            (
                r_excess.saturating_add(graph_excess),
                r_excess,
                graph_excess,
            )
        })
        .map(|(key, kernel)| (key.r_order(), key.graph_dimension(), kernel.clone()))
}

impl FactoredTwistedProjectiveSpaceProvider {
    pub fn fiber_equivariant(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        Ok(Self {
            inner: TwistedProjectiveSpaceProvider::fiber_equivariant(n, degrees)?,
            base_weight_mode: FactoredBaseWeightMode::RationalSpecialization,
        })
    }

    /// Native-factored provider for the QRR audit.  Unlike the public
    /// early-specialized factored provider, this retains
    /// `lambda_i = w_i lambda_0` throughout calibration, graph contraction,
    /// and divisor recursion.
    pub(crate) fn qrr_fiber_equivariant(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        Ok(Self {
            inner: TwistedProjectiveSpaceProvider::fiber_equivariant(n, degrees)?,
            base_weight_mode: FactoredBaseWeightMode::SymbolicLambdaLine,
        })
    }

    /// Native-factored QRR provider at an exact rational fiber-weight point.
    ///
    /// The independent `mu_i` are specialized before the hypergeometric and
    /// Birkhoff calibrations are built.  Only the auxiliary base parameter
    /// `lambda_0` remains symbolic through graph contraction and divisor
    /// recursion, and is removed once from the complete correlator.
    pub(crate) fn qrr_fixed_fiber_lambda_line(
        n: usize,
        degrees: Vec<usize>,
        fiber_weights: Vec<Rational>,
    ) -> Result<Self, GwError> {
        Ok(Self {
            inner: TwistedProjectiveSpaceProvider::qrr_fixed_fiber_lambda_line(
                n,
                degrees,
                fiber_weights,
            )?,
            base_weight_mode: FactoredBaseWeightMode::SymbolicLambdaLine,
        })
    }

    pub fn inner(&self) -> &TwistedProjectiveSpaceProvider {
        &self.inner
    }

    fn factored_base_weights(&self) -> Vec<FactoredRatFun> {
        let lambda = (self.base_weight_mode == FactoredBaseWeightMode::SymbolicLambdaLine)
            .then(|| FactoredRatFun::variable("lambda_0"));
        self.inner
            .base_weights()
            .iter()
            .cloned()
            .map(FactoredRatFun::from_rational)
            .map(|weight| {
                lambda
                    .as_ref()
                    .map(|lambda| lambda.mul(&weight))
                    .unwrap_or(weight)
            })
            .collect()
    }

    fn factored_fiber_weights(&self) -> Vec<FactoredRatFun> {
        match self.inner.line_mode {
            TwistedLineMode::FiberEquivariant => self
                .inner
                .fiber_parameter_names
                .iter()
                .cloned()
                .map(FactoredRatFun::variable)
                .collect(),
            TwistedLineMode::FixedFiberLambdaLine
            | TwistedLineMode::EarlyRational
            | TwistedLineMode::SymbolicLimit => self
                .inner
                .rational_fiber_weights()
                .into_iter()
                .map(FactoredRatFun::from_rational)
                .collect(),
        }
    }

    fn factored_flat_metric_matrix(
        &self,
        q_degree: usize,
    ) -> Result<SeriesMatrix<FactoredRatFun>, GwError> {
        let fiber_weights = self.factored_fiber_weights();
        match self.base_weight_mode {
            FactoredBaseWeightMode::RationalSpecialization => {
                let (metric, _) = twisted_inverse_euler_flat_metric_pair_from_rational_base(
                    self.inner.base.n(),
                    q_degree,
                    &self.inner.twist,
                    self.inner.base_weights(),
                    &fiber_weights,
                )?;
                Ok(metric)
            }
            FactoredBaseWeightMode::SymbolicLambdaLine => {
                twisted_inverse_euler_flat_metric_matrix_coeff(
                    self.inner.base.n(),
                    q_degree,
                    &self.inner.twist,
                    &self.factored_base_weights(),
                    &fiber_weights,
                )
            }
        }
    }

    fn descendant_s_cache_key(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> FactoredTwistedDescendantSCacheKey {
        FactoredTwistedDescendantSCacheKey {
            n: self.inner.base.n(),
            twist_degrees: self.inner.twist.degrees().to_vec(),
            q_degree,
            z_order,
            base_weights: self.inner.base_weights().to_vec(),
            fiber_weights: self.inner.rational_fiber_weights(),
            fiber_parameter_names: self.inner.fiber_parameter_names.clone(),
            base_equivariant: self.inner.base.is_equivariant(),
            line_mode: self.inner.line_mode.clone(),
            calibration_mode: self.inner.calibration_mode.clone(),
            base_weight_mode: self.base_weight_mode,
        }
    }

    fn compute_factored_raw_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
        let base_weights = self.factored_base_weights();
        let fiber_weights = self.factored_fiber_weights();
        NegativeSplitLineHypergeometricModel::<FactoredRatFun>::from_coeff_weights(
            self.inner.base.n(),
            self.inner.twist.clone(),
            q_degree,
            z_order,
            base_weights,
            &fiber_weights,
        )?
        .birkhoff_descendant_s_matrix(z_order)
    }

    fn cached_factored_raw_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<Arc<SeriesSMatrix<FactoredRatFun>>, GwError> {
        static CACHE: OnceLock<
            Mutex<
                BoundedCache<
                    FactoredTwistedDescendantSCacheKey,
                    Arc<SeriesSMatrix<FactoredRatFun>>,
                >,
            >,
        > = OnceLock::new();
        let key = self.descendant_s_cache_key(q_degree, z_order);
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
        if let Some((cached_z_order, matrix)) =
            minimal_covering_descendant_s(&cache.lock().unwrap(), &key)
        {
            if crate::env_flag("GW_PROFILE") {
                eprintln!(
                    "GW_PROFILE factored_twisted_raw_s_cache=hit q_degree={} z_order={} cached_z_order={}",
                    q_degree, z_order, cached_z_order
                );
            }
            return Ok(matrix);
        }

        let started = std::time::Instant::now();
        let matrix = Arc::new(self.compute_factored_raw_descendant_s_matrix(q_degree, z_order)?);
        if crate::env_flag("GW_PROFILE") {
            eprintln!(
                "GW_PROFILE factored_twisted_raw_s_cache=miss q_degree={} z_order={} build={:.3}s",
                q_degree,
                z_order,
                started.elapsed().as_secs_f64()
            );
        }
        cache.lock().unwrap().insert(key, matrix.clone());
        Ok(matrix)
    }

    fn compute_factored_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
        let raw_s = self.cached_factored_raw_descendant_s_matrix(q_degree, z_order)?;
        let fiber_weights = self.factored_fiber_weights();
        match self.base_weight_mode {
            FactoredBaseWeightMode::RationalSpecialization => {
                let (flat_metric, flat_metric_inverse) =
                    twisted_inverse_euler_flat_metric_pair_from_rational_base(
                        self.inner.base.n(),
                        q_degree,
                        &self.inner.twist,
                        self.inner.base_weights(),
                        &fiber_weights,
                    )?;
                metric_adjoint_descendant_s_matrix_with_inverse_coeff(
                    raw_s.as_ref().clone(),
                    &flat_metric,
                    &flat_metric_inverse,
                )
            }
            FactoredBaseWeightMode::SymbolicLambdaLine => {
                let flat_metric = twisted_inverse_euler_flat_metric_matrix_coeff(
                    self.inner.base.n(),
                    q_degree,
                    &self.inner.twist,
                    &self.factored_base_weights(),
                    &fiber_weights,
                )?;
                metric_adjoint_descendant_s_matrix_coeff(raw_s.as_ref().clone(), &flat_metric)
            }
        }
    }

    fn cached_factored_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<Arc<SeriesSMatrix<FactoredRatFun>>, GwError> {
        static CACHE: OnceLock<
            Mutex<
                BoundedCache<
                    FactoredTwistedDescendantSCacheKey,
                    Arc<SeriesSMatrix<FactoredRatFun>>,
                >,
            >,
        > = OnceLock::new();
        let key = self.descendant_s_cache_key(q_degree, z_order);
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
        if let Some((_, matrix)) = minimal_covering_descendant_s(&cache.lock().unwrap(), &key) {
            return Ok(matrix);
        }
        let matrix = Arc::new(self.compute_factored_descendant_s_matrix(q_degree, z_order)?);
        cache.lock().unwrap().insert(key, matrix.clone());
        Ok(matrix)
    }

    fn graph_kernel_cache_key(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
        validation: TwistedCalibrationValidation,
    ) -> FactoredTwistedGraphKernelCacheKey {
        FactoredTwistedGraphKernelCacheKey {
            n: self.inner.base.n(),
            twist_degrees: self.inner.twist.degrees().to_vec(),
            q_degree,
            r_order,
            graph_dimension,
            base_weights: self.inner.base_weights().to_vec(),
            fiber_weights: self.inner.rational_fiber_weights(),
            fiber_parameter_names: self.inner.fiber_parameter_names.clone(),
            base_equivariant: self.inner.base.is_equivariant(),
            line_mode: self.inner.line_mode.clone(),
            calibration_mode: self.inner.calibration_mode.clone(),
            base_weight_mode: self.base_weight_mode,
            validation,
        }
    }

    pub(crate) fn evaluate_qrr_fiber_equivariant_positive_degree(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<FactoredRatFun, GwError> {
        if self.base_weight_mode != FactoredBaseWeightMode::SymbolicLambdaLine {
            return Err(GwError::ConventionMismatch(
                "QRR factored evaluation requires symbolic base lambda-line weights".to_string(),
            ));
        }
        self.inner
            .validate_fiber_equivariant_positive_degree_request(degree, insertions)?;
        self.evaluate_qrr_positive_degree_factored(genus, degree, insertions, truncation)
    }

    /// Fixed-fiber counterpart of
    /// [`Self::evaluate_qrr_fiber_equivariant_positive_degree`].
    ///
    /// This returns the complete factored expression before its auxiliary
    /// lambda-line limit.  In particular, divisor-equation corrections are
    /// combined in the factored ring rather than after separately expanding
    /// and specializing their stable graph values.
    pub(crate) fn evaluate_qrr_fixed_fiber_positive_degree(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<FactoredRatFun, GwError> {
        if self.base_weight_mode != FactoredBaseWeightMode::SymbolicLambdaLine {
            return Err(GwError::ConventionMismatch(
                "fixed-fiber QRR factored evaluation requires symbolic base lambda-line weights"
                    .to_string(),
            ));
        }
        self.inner
            .validate_qrr_fixed_fiber_positive_degree_request(degree, insertions)?;
        self.evaluate_qrr_positive_degree_factored(genus, degree, insertions, truncation)
    }

    fn evaluate_qrr_positive_degree_factored(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<FactoredRatFun, GwError> {
        if self
            .inner
            .vanishes_above_base_virtual_dimension(genus, degree, insertions)
        {
            return Ok(FactoredRatFun::zero());
        }
        self.inner
            .evaluate_positive_degree_with_divisor_recursion_coeff(
                genus,
                degree,
                insertions.to_vec(),
                &|stable_insertions| {
                    compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
                        self,
                        genus,
                        degree,
                        stable_insertions,
                        truncation,
                    )
                },
            )
    }

    pub(crate) fn qrr_lambda_line_limit(&self, value: &FactoredRatFun) -> Result<RatFun, GwError> {
        value.lambda_line_limit_preserving_variables(self.inner.n(), self.inner.base_weights())
    }

    fn factored_genus_zero_two_point_fallback(
        &self,
        degree: usize,
        insertions: &[Insertion],
    ) -> Result<Option<FactoredRatFun>, GwError> {
        let Some((descendant_idx, primary_idx, s_order)) =
            genus_zero_two_point_descendant_layout(insertions)
        else {
            return Ok(None);
        };
        let s_matrix = self.cached_factored_raw_descendant_s_matrix(degree, s_order)?;
        let metric = self.factored_flat_metric_matrix(degree)?;
        let descendant = <Self as SemisimpleCohftProvider<FactoredRatFun>>::insertion_vector(
            self,
            &insertions[descendant_idx],
            degree,
        )?;
        let primary = <Self as SemisimpleCohftProvider<FactoredRatFun>>::insertion_vector(
            self,
            &insertions[primary_idx],
            degree,
        )?;
        genus_zero_two_point_raw_s_matrix_pairing_coeff(
            <Self as SemisimpleCohftProvider<FactoredRatFun>>::colors(self),
            degree,
            s_order,
            s_matrix.as_ref(),
            &metric,
            &descendant,
            &primary,
        )
        .map(Some)
    }
}

impl SemisimpleCohftProvider<FactoredRatFun> for FactoredTwistedProjectiveSpaceProvider {
    type Insertion = Insertion;

    fn colors(&self) -> usize {
        <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::colors(&self.inner)
    }

    fn descendant_power(&self, insertion: &Self::Insertion) -> usize {
        <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::descendant_power(
            &self.inner,
            insertion,
        )
    }

    fn insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::insertion_degree(
            &self.inner,
            insertions,
        )
    }

    fn virtual_dimension(&self, genus: usize, degree: usize, markings: usize) -> Option<isize> {
        <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::virtual_dimension(
            &self.inner,
            genus,
            degree,
            markings,
        )
    }

    fn degree_is_effective(&self, degree: usize) -> bool {
        <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::degree_is_effective(
            &self.inner,
            degree,
        )
    }

    fn vanishes_by_dimension(&self, virtual_dimension: isize, total_degree: usize) -> bool {
        <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::vanishes_by_dimension(
            &self.inner,
            virtual_dimension,
            total_degree,
        )
    }

    fn expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::expected_degree_from_dimension(
            &self.inner,
            genus,
            insertions,
        )
    }

    fn candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::candidate_degrees_from_dimension(
            &self.inner,
            genus,
            degree_max,
            insertions,
        )
    }

    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
        Ok(self
            .cached_factored_descendant_s_matrix(q_degree, z_order)?
            .as_ref()
            .clone())
    }

    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel<FactoredRatFun>>, GwError> {
        static CACHE: OnceLock<
            Mutex<
                BoundedCache<
                    FactoredTwistedGraphKernelCacheKey,
                    Arc<GiventalGraphKernel<FactoredRatFun>>,
                >,
            >,
        > = OnceLock::new();
        let validation = twisted_calibration_validation_from_env();
        let key = self.graph_kernel_cache_key(q_degree, r_order, graph_dimension, validation);
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
        if let Some((cached_r_order, cached_graph_dimension, kernel)) =
            minimal_covering_graph_kernel(&cache.lock().unwrap(), &key)
        {
            if crate::env_flag("GW_PROFILE") {
                eprintln!(
                    "GW_PROFILE factored_twisted_graph_kernel_cache=hit q_degree={} r_order={} graph_dimension={} cached_r_order={} cached_graph_dimension={}",
                    q_degree,
                    r_order,
                    graph_dimension,
                    cached_r_order,
                    cached_graph_dimension
                );
            }
            return Ok(kernel);
        }

        let profile_enabled = crate::env_flag("GW_PROFILE");
        let started = std::time::Instant::now();
        let base_weights = self.factored_base_weights();
        let fiber_weights = self.factored_fiber_weights();
        let calibration =
            negative_split_twisted_birkhoff_calibration_candidate_for_coeff_weights_with_validation(
                self.inner.base.n(),
                &self.inner.twist,
                q_degree,
                r_order,
                &base_weights,
                &fiber_weights,
                validation,
            )?;
        if profile_enabled {
            eprintln!(
                "GW_PROFILE factored_twisted_calibration={:.3}s",
                started.elapsed().as_secs_f64()
            );
        }
        let kernel_started = std::time::Instant::now();
        let kernel = Arc::new(GiventalGraphKernel::from_calibration(
            calibration,
            graph_dimension,
        )?);
        if profile_enabled {
            eprintln!(
                "GW_PROFILE factored_graph_kernel={:.3}s",
                kernel_started.elapsed().as_secs_f64()
            );
        }
        cache.lock().unwrap().insert(key, kernel.clone());
        Ok(kernel)
    }

    fn insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries<FactoredRatFun>>, GwError> {
        Ok(self
            .inner
            .insertion_vector(insertion, q_degree)?
            .iter()
            .map(qseries_to_factored)
            .collect())
    }

    fn direct_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<FactoredRatFun>, GwError> {
        if genus != 0 || !genus_zero_three_primary_layout(insertions) {
            return Ok(None);
        }
        let base_weights = self.factored_base_weights();
        let fiber_weights = self.factored_fiber_weights();
        let insertion_vectors = insertions
            .iter()
            .map(|insertion| {
                <Self as SemisimpleCohftProvider<FactoredRatFun>>::insertion_vector(
                    self, insertion, degree,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let descendant_s = self.cached_factored_raw_descendant_s_matrix(degree, 1)?;
        twisted_genus_zero_three_primary_value_from_s_coeff(
            self.inner.base.n(),
            &self.inner.twist,
            degree,
            &self.inner.calibration_mode,
            &base_weights,
            &fiber_weights,
            descendant_s.as_ref(),
            &insertion_vectors,
        )
        .map(Some)
    }

    fn scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<Option<FactoredRatFun>, GwError> {
        if genus == 0 {
            return self.factored_genus_zero_two_point_fallback(degree, insertions);
        }
        Ok(self
            .inner
            .scalar_fallback_value(genus, degree, insertions, truncation)?
            .map(FactoredRatFun::from_ratfun))
    }
}

#[cfg(test)]
pub(crate) fn series_s_matrix_to_factored(
    matrix: &SeriesSMatrix,
) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
    SeriesSMatrix::from_coefficients(
        matrix.size(),
        matrix.q_degree(),
        matrix.z_order(),
        matrix
            .coefficients()
            .iter()
            .map(series_matrix_to_factored)
            .collect(),
        matrix.calibration().clone(),
    )
}

#[cfg(test)]
pub(crate) fn series_matrix_to_factored(matrix: &SeriesMatrix) -> SeriesMatrix<FactoredRatFun> {
    SeriesMatrix::from_entries(
        matrix
            .entries()
            .iter()
            .map(|row| row.iter().map(qseries_to_factored).collect())
            .collect(),
    )
}

pub(crate) fn qseries_to_factored(series: &QSeries) -> QSeries<FactoredRatFun> {
    QSeries::from_coeffs(
        series
            .coeffs()
            .iter()
            .cloned()
            .map(FactoredRatFun::from_ratfun)
            .collect(),
    )
}

impl SemisimpleCohftProvider for TwistedProjectiveSpaceProvider {
    type Insertion = Insertion;

    fn colors(&self) -> usize {
        self.base.colors()
    }

    fn descendant_power(&self, insertion: &Self::Insertion) -> usize {
        self.base.descendant_power(insertion)
    }

    fn insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        self.base.insertion_degree(insertions)
    }

    fn virtual_dimension(&self, genus: usize, degree: usize, markings: usize) -> Option<isize> {
        self.canonical_theory()
            .ok()?
            .virtual_dimension_at_degree(genus, degree, markings)
            .ok()
    }

    fn degree_is_effective(&self, degree: usize) -> bool {
        self.canonical_theory()
            .is_ok_and(|theory| theory.degree_is_effective(degree).unwrap_or(false))
    }

    fn vanishes_by_dimension(&self, virtual_dimension: isize, total_degree: usize) -> bool {
        if matches!(
            self.line_mode,
            TwistedLineMode::FiberEquivariant | TwistedLineMode::FixedFiberLambdaLine
        ) {
            // The fiber parameters can carry excess degree.  Keep the
            // localized theory conservative here: degree-zero inverse-Euler
            // twists can also have negative parameter degree.
            false
        } else {
            usize::try_from(virtual_dimension).ok() != Some(total_degree)
        }
    }

    fn expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        if matches!(
            self.line_mode,
            TwistedLineMode::FiberEquivariant | TwistedLineMode::FixedFiberLambdaLine
        ) {
            return None;
        }
        self.canonical_theory()
            .ok()?
            .expected_degree_from_dimension(
                genus,
                insertions.len(),
                self.insertion_degree(insertions)?,
            )
            .ok()?
    }

    fn candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        let Some(theory) = self.canonical_theory().ok() else {
            return Vec::new();
        };
        let insertion_degree = (!matches!(
            self.line_mode,
            TwistedLineMode::FiberEquivariant | TwistedLineMode::FixedFiberLambdaLine
        ))
        .then(|| self.insertion_degree(insertions))
        .flatten();
        theory
            .candidate_degrees_from_dimension(genus, degree_max, insertions.len(), insertion_degree)
            .unwrap_or_default()
    }

    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        Ok(self
            .cached_descendant_s_matrix(q_degree, z_order)?
            .as_ref()
            .clone())
    }

    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel>, GwError> {
        // This is where the twisted target becomes a generic semisimple CohFT:
        // Birkhoff canonical data + QRR R-recursion are packaged into the same
        // kernel interface used by ordinary P^n.
        static CACHE: OnceLock<
            Mutex<BoundedCache<TwistedGraphKernelCacheKey, Arc<GiventalGraphKernel>>>,
        > = OnceLock::new();
        let rational_fiber_weights = self.rational_fiber_weights();
        let validation = twisted_calibration_validation_from_env();
        let key = TwistedGraphKernelCacheKey {
            n: self.base.n(),
            twist_degrees: self.twist.degrees().to_vec(),
            q_degree,
            r_order,
            graph_dimension,
            base_weights: self.base_weights().to_vec(),
            fiber_weights: rational_fiber_weights.clone(),
            fiber_parameter_names: self.fiber_parameter_names.clone(),
            line_mode: self.line_mode.clone(),
            calibration_mode: self.calibration_mode.clone(),
            validation,
        };
        let cache = CACHE
            .get_or_init(|| Mutex::new(BoundedCache::new(TARGET_RECONSTRUCTION_CACHE_CAPACITY)));
        if let Some((_, _, kernel)) = minimal_covering_graph_kernel(&cache.lock().unwrap(), &key) {
            return Ok(kernel);
        }

        let calibration = match self.line_mode {
            TwistedLineMode::EarlyRational => {
                negative_split_twisted_birkhoff_calibration_candidate_with_mode_and_validation(
                    self.base.n(),
                    &self.twist,
                    q_degree,
                    r_order,
                    self.base_weights(),
                    &rational_fiber_weights,
                    self.calibration_mode.clone(),
                    validation,
                )?
            }
            TwistedLineMode::SymbolicLimit
            | TwistedLineMode::FiberEquivariant
            | TwistedLineMode::FixedFiberLambdaLine => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                negative_split_twisted_birkhoff_calibration_candidate_for_ratfun_weights_with_validation(
                    self.base.n(),
                    &self.twist,
                    q_degree,
                    r_order,
                    &base_weights,
                    &fiber_weights,
                    validation,
                )?
            }
        };
        let kernel = Arc::new(GiventalGraphKernel::from_calibration(
            calibration,
            graph_dimension,
        )?);
        cache.lock().unwrap().insert(key, kernel.clone());
        Ok(kernel)
    }

    fn insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries>, GwError> {
        self.base.insertion_vector(insertion, q_degree)
    }

    fn direct_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        if genus != 0 || !genus_zero_three_primary_layout(insertions) {
            return Ok(None);
        }
        let base_weights = self.ratfun_base_weights();
        let fiber_weights = self.ratfun_fiber_weights();
        let insertion_vectors = insertions
            .iter()
            .map(|insertion| self.insertion_vector(insertion, degree))
            .collect::<Result<Vec<_>, _>>()?;
        let descendant_s = self.cached_raw_descendant_s_matrix(degree, 1)?;
        twisted_genus_zero_three_primary_value_from_s_coeff(
            self.base.n(),
            &self.twist,
            degree,
            &self.calibration_mode,
            &base_weights,
            &fiber_weights,
            descendant_s.as_ref(),
            &insertion_vectors,
        )
        .map(Some)
    }

    fn scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<RatFun>, GwError> {
        if genus == 0 {
            return self.genus_zero_two_point_fallback(degree, insertions);
        }
        Ok(None)
    }
}

pub(crate) fn metric_adjoint_descendant_s_matrix(
    s_matrix: SeriesSMatrix,
    flat_metric: &SeriesMatrix,
) -> Result<SeriesSMatrix, GwError> {
    metric_adjoint_descendant_s_matrix_coeff(s_matrix, flat_metric)
}

#[cfg(test)]
mod provider_tests {
    use super::*;

    #[test]
    fn stable_primary_divisor_reduction_has_positive_degree_factor_and_terminates() {
        let provider = TwistedProjectiveSpaceProvider::new(2, vec![2], false).unwrap();
        let h = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::h_power(2, 1),
        );
        let h2 = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::h_power(2, 2),
        );
        let calls = std::cell::Cell::new(0usize);
        let seen = std::cell::RefCell::new(Vec::<Insertion>::new());

        let value = provider
            .evaluate_positive_degree_with_divisor_recursion_coeff(
                2,
                3,
                vec![h.clone(), h2.clone(), h],
                &|stable_insertions| {
                    calls.set(calls.get() + 1);
                    *seen.borrow_mut() = stable_insertions.to_vec();
                    Ok::<_, GwError>(RatFun::from_rational(Rational::from(5usize)))
                },
            )
            .unwrap();

        // Each of the two primary H insertions contributes +(H.3) = +3.
        assert_eq!(value, RatFun::from_rational(Rational::from(45usize)));
        assert_eq!(calls.get(), 1);
        assert_eq!(*seen.borrow(), vec![h2]);
    }

    #[test]
    fn stable_primary_divisor_reduction_includes_positive_descendant_corrections() {
        let provider = TwistedProjectiveSpaceProvider::new(2, vec![2], false).unwrap();
        let h = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::h_power(2, 1),
        );
        let tau_one_unit = crate::tau(
            1,
            crate::spaces::projective_space::CohomologyClass::h_power(2, 0),
        );
        let seen = std::cell::RefCell::new(Vec::<Vec<Insertion>>::new());

        let value = provider
            .evaluate_positive_degree_with_divisor_recursion_coeff(
                2,
                3,
                vec![h, tau_one_unit.clone()],
                &|stable_insertions| {
                    seen.borrow_mut().push(stable_insertions.to_vec());
                    if stable_insertions.is_empty() {
                        Ok::<_, GwError>(RatFun::from_rational(Rational::from(7usize)))
                    } else {
                        assert_eq!(stable_insertions, std::slice::from_ref(&tau_one_unit));
                        Ok::<_, GwError>(RatFun::from_rational(Rational::from(5usize)))
                    }
                },
            )
            .unwrap();

        // <H,tau_1(1)>_3 = 3 <tau_1(1)>_3 + <H>_3, while the
        // nested primary relation gives <H>_3 = 3 < >_3.  The correction has
        // a positive sign, hence 3*5 + 3*7 = 36.
        assert_eq!(value, RatFun::from_rational(Rational::from(36usize)));
        assert_eq!(
            *seen.borrow(),
            vec![vec![tau_one_unit], Vec::<Insertion>::new()]
        );
    }

    #[test]
    fn stable_primary_divisor_reduction_strips_h_before_tau_three_top_class() {
        let provider = TwistedProjectiveSpaceProvider::new(2, vec![2], false).unwrap();
        let h = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::h_power(2, 1),
        );
        let tau_three_point = crate::tau(
            3,
            crate::spaces::projective_space::CohomologyClass::h_power(2, 2),
        );
        let calls = std::cell::Cell::new(0usize);

        let value = provider
            .evaluate_positive_degree_with_divisor_recursion_coeff(
                2,
                1,
                vec![h, tau_three_point.clone()],
                &|stable_insertions| {
                    calls.set(calls.get() + 1);
                    assert_eq!(stable_insertions, std::slice::from_ref(&tau_three_point));
                    Ok::<_, GwError>(RatFun::from_rational(Rational::from(11usize)))
                },
            )
            .unwrap();

        // <H,tau_3(H^2)>_{2,1} = <tau_3(H^2)>_{2,1}: the degree
        // factor is one, and the descendant correction contains H^3=0.
        assert_eq!(value, RatFun::from_rational(Rational::from(11usize)));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn stable_primary_divisor_reduction_stops_before_the_unstable_boundary() {
        let provider = TwistedProjectiveSpaceProvider::new(2, vec![2], false).unwrap();
        let h = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::h_power(2, 1),
        );
        let h2 = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::h_power(2, 2),
        );
        let seen_markings = std::cell::Cell::new(0usize);

        let value = provider
            .evaluate_positive_degree_with_divisor_recursion_coeff(
                0,
                2,
                vec![h, h2.clone(), h2],
                &|stable_insertions| {
                    seen_markings.set(stable_insertions.len());
                    Ok::<_, GwError>(RatFun::from_rational(Rational::from(7usize)))
                },
            )
            .unwrap();

        // Removing H would leave a genus-zero two-pointed curve, so the
        // stable graph evaluator receives the original boundary case once.
        assert_eq!(value, RatFun::from_rational(Rational::from(7usize)));
        assert_eq!(seen_markings.get(), 3);
    }

    #[test]
    fn twisted_providers_reject_the_rank_zero_compatibility_recipe() {
        assert!(NegativeSplitBundleTwist::new(Vec::new()).is_ok());
        assert!(matches!(
            TwistedProjectiveSpaceProvider::new(1, Vec::new(), false),
            Err(GwError::ConventionMismatch(_))
        ));
        assert!(matches!(
            TwistedProjectiveSpaceProvider::fiber_equivariant(1, Vec::new()),
            Err(GwError::ConventionMismatch(_))
        ));
        assert!(matches!(
            TwistedProjectiveSpaceProvider::symbolic_lambda_line(1, Vec::new(), true),
            Err(GwError::ConventionMismatch(_))
        ));
        assert!(matches!(
            FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(1, Vec::new()),
            Err(GwError::ConventionMismatch(_))
        ));
    }

    #[test]
    fn twisted_requests_reject_an_empty_twist_even_when_built_as_a_struct() {
        assert!(matches!(
            TwistedInvariantRequest::new(1, Vec::new(), 0, 1, Vec::new()),
            Err(GwError::ConventionMismatch(_))
        ));
        let request = TwistedInvariantRequest {
            n: 1,
            twist: NegativeSplitBundleTwist::new(Vec::new()).unwrap(),
            genus: 0,
            degree: 1,
            insertions: Vec::new(),
            equivariant: false,
            truncation: None,
        };
        assert!(matches!(
            request.validate(),
            Err(GwError::ConventionMismatch(_))
        ));
    }

    #[test]
    fn factored_twisted_provider_preserves_equivariant_grading() {
        let factored =
            FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(2, vec![1]).unwrap();

        let expected = <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<
            RatFun,
        >>::vanishes_by_dimension(factored.inner(), 3, 7);
        let canonical = <FactoredTwistedProjectiveSpaceProvider as SemisimpleCohftProvider<
            FactoredRatFun,
        >>::vanishes_by_dimension(&factored, 3, 7);
        assert!(
            !expected,
            "fiber-equivariant parameters carry excess degree"
        );
        assert_eq!(canonical, expected);
    }

    #[test]
    fn descendant_s_cache_reuses_the_exact_birkhoff_matrix() {
        let provider = TwistedProjectiveSpaceProvider::fiber_equivariant(1, vec![1]).unwrap();
        let first = provider.cached_descendant_s_matrix(0, 1).unwrap();
        let second = provider.cached_descendant_s_matrix(0, 1).unwrap();
        let uncached = provider.compute_descendant_s_matrix(0, 1).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.as_ref(), &uncached);
    }

    #[test]
    fn descendant_s_caches_reuse_a_larger_consistent_truncation() {
        let expanded = TwistedProjectiveSpaceProvider::fiber_equivariant(0, vec![17]).unwrap();
        let expanded_large = expanded.cached_raw_descendant_s_matrix(0, 2).unwrap();
        let expanded_small = expanded.cached_raw_descendant_s_matrix(0, 1).unwrap();
        let expanded_small_uncached = expanded.compute_raw_descendant_s_matrix(0, 1).unwrap();
        assert!(Arc::ptr_eq(&expanded_large, &expanded_small));
        assert_eq!(
            &expanded_large.coefficients()[..=1],
            expanded_small_uncached.coefficients()
        );

        let factored =
            FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(0, vec![19]).unwrap();
        let factored_large = factored
            .cached_factored_raw_descendant_s_matrix(0, 2)
            .unwrap();
        let factored_small = factored
            .cached_factored_raw_descendant_s_matrix(0, 1)
            .unwrap();
        let factored_small_uncached = factored
            .compute_factored_raw_descendant_s_matrix(0, 1)
            .unwrap();
        assert!(Arc::ptr_eq(&factored_large, &factored_small));
        assert_eq!(
            &factored_large.coefficients()[..=1],
            factored_small_uncached.coefficients()
        );
    }

    #[test]
    fn graph_kernel_cache_keys_only_dominate_matching_families() {
        let provider =
            FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(0, vec![23]).unwrap();
        let requested =
            provider.graph_kernel_cache_key(0, 1, 1, TwistedCalibrationValidation::Fast);
        let covering = provider.graph_kernel_cache_key(0, 3, 2, TwistedCalibrationValidation::Fast);
        assert!(covering.same_family(&requested));
        assert!(covering.r_order() >= requested.r_order());
        assert!(covering.graph_dimension() >= requested.graph_dimension());

        let other_degree =
            provider.graph_kernel_cache_key(1, 3, 2, TwistedCalibrationValidation::Fast);
        let other_validation =
            provider.graph_kernel_cache_key(0, 3, 2, TwistedCalibrationValidation::Full);
        assert!(!other_degree.same_family(&requested));
        assert!(!other_validation.same_family(&requested));
    }

    #[test]
    fn expanded_and_factored_graph_kernel_caches_reuse_covering_kernels() {
        let expanded = TwistedProjectiveSpaceProvider::rational_lambda_line_with_weights(
            0,
            vec![29],
            vec![Rational::from(31usize)],
            vec![Rational::from(37usize)],
        )
        .unwrap();
        let expanded_large =
            <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::graph_kernel(
                &expanded, 0, 3, 2,
            )
            .unwrap();
        let expanded_small =
            <TwistedProjectiveSpaceProvider as SemisimpleCohftProvider<RatFun>>::graph_kernel(
                &expanded, 0, 1, 1,
            )
            .unwrap();
        assert!(Arc::ptr_eq(&expanded_large, &expanded_small));

        let factored =
            FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(0, vec![31]).unwrap();
        let factored_large = <FactoredTwistedProjectiveSpaceProvider as SemisimpleCohftProvider<
            FactoredRatFun,
        >>::graph_kernel(&factored, 0, 3, 2)
        .unwrap();
        let factored_small = <FactoredTwistedProjectiveSpaceProvider as SemisimpleCohftProvider<
            FactoredRatFun,
        >>::graph_kernel(&factored, 0, 1, 1)
        .unwrap();
        assert!(Arc::ptr_eq(&factored_large, &factored_small));
    }

    #[test]
    fn raw_and_metric_adjoint_descendant_s_caches_keep_distinct_conventions() {
        let provider = TwistedProjectiveSpaceProvider::fiber_equivariant(1, vec![1]).unwrap();
        let first = provider.cached_raw_descendant_s_matrix(0, 1).unwrap();
        let second = provider.cached_raw_descendant_s_matrix(0, 1).unwrap();
        let uncached = provider.compute_raw_descendant_s_matrix(0, 1).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.as_ref(), &uncached);
        // Both conventions are cached, but the graph-leg cache remains the
        // metric adjoint rather than silently becoming the raw solution.
        assert!(provider.cached_descendant_s_matrix(0, 1).is_ok());
    }

    #[test]
    fn qrr_factored_mode_keeps_base_lambda_symbolic_and_caches_raw_s() {
        let public = FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(1, vec![1]).unwrap();
        let qrr =
            FactoredTwistedProjectiveSpaceProvider::qrr_fiber_equivariant(1, vec![1]).unwrap();
        let lambda = crate::core::algebra::lambda(0);

        assert!(public.factored_base_weights()[0]
            .to_ratfun()
            .equivalent(&RatFun::one()));
        assert!(qrr.factored_base_weights()[0]
            .to_ratfun()
            .equivalent(&lambda));
        assert_ne!(
            public.descendant_s_cache_key(0, 1),
            qrr.descendant_s_cache_key(0, 1)
        );

        let first = qrr.cached_factored_raw_descendant_s_matrix(0, 1).unwrap();
        let second = qrr.cached_factored_raw_descendant_s_matrix(0, 1).unwrap();
        let uncached = qrr.compute_factored_raw_descendant_s_matrix(0, 1).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.as_ref(), &uncached);
    }

    #[test]
    fn inverse_euler_base_dimension_pruning_is_an_upper_bound_not_equality() {
        let provider = TwistedProjectiveSpaceProvider::fiber_equivariant(2, vec![2]).unwrap();
        let divisor = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(2, 1).unwrap(),
        );
        let point = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(2, 2).unwrap(),
        );
        let deficit = [divisor.clone(), divisor, point];

        // For (g,d,N)=(0,1,3) on P2, V_base=5 while D=4.  The
        // codimension-one inverse-Euler term may fill this deficit.
        assert!(!provider.vanishes_above_base_virtual_dimension(0, 1, &deficit));
        let value = provider
            .evaluate_fiber_equivariant_positive_degree(0, 1, &deficit, None)
            .unwrap();
        assert!(!value.is_zero());

        let excess = [crate::tau(
            2,
            crate::spaces::projective_space::CohomologyClass::try_h_power(2, 2).unwrap(),
        )];
        assert!(provider.vanishes_above_base_virtual_dimension(0, 1, &excess));
        assert!(provider.vanishes_above_base_virtual_dimension(0, 0, &[]));
    }

    #[test]
    fn symbolic_lambda_factored_qrr_matches_expanded_degree_zero_and_positive_degree(
    ) -> Result<(), GwError> {
        let expanded = TwistedProjectiveSpaceProvider::fiber_equivariant(1, vec![2]).unwrap();
        let factored =
            FactoredTwistedProjectiveSpaceProvider::qrr_fiber_equivariant(1, vec![2]).unwrap();
        let unit = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(1, 0).unwrap(),
        );
        let divisor = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(1, 1).unwrap(),
        );

        let degree_zero_insertions = [divisor.clone(), unit.clone(), unit.clone()];
        let expanded_degree_zero = crate::givental::compute_semisimple_graph_value(
            &expanded,
            0,
            0,
            &degree_zero_insertions,
            None,
        )?
        .lambda_line_limit_preserving_variables(expanded.n(), expanded.base_weights())?;
        let factored_degree_zero = compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
            &factored,
            0,
            0,
            &degree_zero_insertions,
            None,
        )?;
        assert!(expanded_degree_zero
            .equivalent(&factored.qrr_lambda_line_limit(&factored_degree_zero)?));

        // This row is outside the genus-zero three-primary shortcut: it runs
        // the genuine S/R stable-graph path on both coefficient backends.
        // P0 keeps that regression cheap while the genus-one Hodge term makes
        // its one-marking degree-zero value nonzero.
        let graph_expanded = TwistedProjectiveSpaceProvider::fiber_equivariant(0, vec![1]).unwrap();
        let graph_factored =
            FactoredTwistedProjectiveSpaceProvider::qrr_fiber_equivariant(0, vec![1]).unwrap();
        let graph_insertions = [crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(0, 0).unwrap(),
        )];
        let expanded_graph = crate::givental::compute_semisimple_graph_value(
            &graph_expanded,
            1,
            0,
            &graph_insertions,
            None,
        )?
        .lambda_line_limit_preserving_variables(
            graph_expanded.n(),
            graph_expanded.base_weights(),
        )?;
        let factored_graph = compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
            &graph_factored,
            1,
            0,
            &graph_insertions,
            None,
        )?;
        let factored_graph = graph_factored.qrr_lambda_line_limit(&factored_graph)?;
        assert!(!expanded_graph.is_zero());
        assert!(expanded_graph.equivalent(&factored_graph));

        let positive_insertions = [unit, divisor.clone(), divisor];
        let expanded_positive = expanded.evaluate_fiber_equivariant_positive_degree(
            0,
            1,
            &positive_insertions,
            None,
        )?;
        let factored_positive = factored.evaluate_qrr_fiber_equivariant_positive_degree(
            0,
            1,
            &positive_insertions,
            None,
        )?;
        assert!(expanded_positive.equivalent(&factored.qrr_lambda_line_limit(&factored_positive)?));
        Ok(())
    }

    #[test]
    fn fixed_fiber_factored_qrr_matches_expanded_and_symbolic_rows() -> Result<(), GwError> {
        let fiber_weight = Rational::from(3usize);
        let expanded_zero = TwistedProjectiveSpaceProvider::qrr_fixed_fiber_lambda_line(
            0,
            vec![1],
            vec![fiber_weight.clone()],
        )?;
        let factored_zero = FactoredTwistedProjectiveSpaceProvider::qrr_fixed_fiber_lambda_line(
            0,
            vec![1],
            vec![fiber_weight.clone()],
        )?;
        let symbolic_zero =
            FactoredTwistedProjectiveSpaceProvider::qrr_fiber_equivariant(0, vec![1])?;
        let assignments = BTreeMap::from([("mu_0".to_string(), fiber_weight)]);
        let zero_insertion = [crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(0, 0)?,
        )];

        // This stable genus-one constant-map row bypasses the genus-zero
        // Frobenius and unstable S-matrix shortcuts, so it exercises the
        // actual graph backend while remaining a one-color calculation.
        let expanded_zero_raw = crate::givental::compute_semisimple_graph_value(
            &expanded_zero,
            1,
            0,
            &zero_insertion,
            None,
        )?;
        let expanded_zero_value =
            expanded_zero.qrr_fixed_fiber_lambda_line_limit(&expanded_zero_raw)?;
        let factored_zero_raw = compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
            &factored_zero,
            1,
            0,
            &zero_insertion,
            None,
        )?;
        let factored_zero_value = factored_zero.qrr_lambda_line_limit(&factored_zero_raw)?;
        let symbolic_zero_raw = compute_semisimple_graph_value_with_coeff::<FactoredRatFun, _>(
            &symbolic_zero,
            1,
            0,
            &zero_insertion,
            None,
        )?;
        let symbolic_zero_value = symbolic_zero
            .qrr_lambda_line_limit(&symbolic_zero_raw)?
            .evaluate_variables(&assignments)?;
        assert!(expanded_zero_value.equivalent(&factored_zero_value));
        assert_eq!(factored_zero_value.as_rational(), Some(symbolic_zero_value));
        assert!(!factored_zero_value.is_zero());

        // A separate cheap positive-degree Frobenius row checks that the fixed
        // specialization is already present in calibration coefficients.
        let expanded = TwistedProjectiveSpaceProvider::qrr_fixed_fiber_lambda_line(
            2,
            vec![2],
            vec![Rational::from(3usize)],
        )?;
        let factored = FactoredTwistedProjectiveSpaceProvider::qrr_fixed_fiber_lambda_line(
            2,
            vec![2],
            vec![Rational::from(3usize)],
        )?;
        let symbolic = FactoredTwistedProjectiveSpaceProvider::qrr_fiber_equivariant(2, vec![2])?;
        let divisor = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(2, 1)?,
        );
        let point = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(2, 2)?,
        );
        let positive_insertions = [divisor.clone(), divisor, point];
        let expanded_positive =
            expanded.evaluate_qrr_fixed_fiber_positive_degree(0, 1, &positive_insertions, None)?;
        let factored_positive_raw =
            factored.evaluate_qrr_fixed_fiber_positive_degree(0, 1, &positive_insertions, None)?;
        let factored_positive = factored.qrr_lambda_line_limit(&factored_positive_raw)?;
        let symbolic_positive_raw = symbolic.evaluate_qrr_fiber_equivariant_positive_degree(
            0,
            1,
            &positive_insertions,
            None,
        )?;
        let symbolic_positive = symbolic
            .qrr_lambda_line_limit(&symbolic_positive_raw)?
            .evaluate_variables(&assignments)?;
        assert!(expanded_positive.equivalent(&factored_positive));
        assert_eq!(factored_positive.as_rational(), Some(symbolic_positive));
        assert!(
            !factored_positive.is_zero(),
            "positive graph comparison should exercise a nonzero coefficient"
        );
        Ok(())
    }

    #[test]
    fn descendant_s_cache_key_records_truncation_and_provider_conventions() {
        let fiber = TwistedProjectiveSpaceProvider::fiber_equivariant(1, vec![1]).unwrap();
        let symbolic =
            TwistedProjectiveSpaceProvider::symbolic_lambda_line(1, vec![1], false).unwrap();

        assert_ne!(
            fiber.descendant_s_cache_key(0, 1),
            fiber.descendant_s_cache_key(1, 1)
        );
        assert_ne!(
            fiber.descendant_s_cache_key(0, 1),
            fiber.descendant_s_cache_key(0, 2)
        );
        assert_ne!(
            fiber.descendant_s_cache_key(0, 1),
            symbolic.descendant_s_cache_key(0, 1)
        );
    }

    #[test]
    fn custom_fiber_weights_follow_their_sorted_line_summands() {
        let provider = TwistedProjectiveSpaceProvider::rational_lambda_line_with_weights(
            1,
            vec![3, 1, 2],
            vec![Rational::from(2), Rational::from(5)],
            vec![Rational::from(30), Rational::from(10), Rational::from(20)],
        )
        .unwrap();
        assert_eq!(provider.twist().degrees(), &[1, 2, 3]);
        assert_eq!(
            provider.rational_fiber_weights(),
            vec![Rational::from(10), Rational::from(20), Rational::from(30)]
        );
    }

    #[test]
    fn fiber_equivariant_unstable_descendant_uses_full_divisor_equation() {
        let provider = TwistedProjectiveSpaceProvider::fiber_equivariant(1, vec![1]).unwrap();
        let unit = crate::tau(
            1,
            crate::spaces::projective_space::CohomologyClass::try_h_power(1, 0).unwrap(),
        );
        let divisor = crate::tau(
            0,
            crate::spaces::projective_space::CohomologyClass::try_h_power(1, 1).unwrap(),
        );
        let unstable = provider
            .evaluate_fiber_equivariant_positive_degree(0, 1, std::slice::from_ref(&unit), None)
            .unwrap();
        let stable_descendant = provider
            .evaluate_fiber_equivariant_positive_degree(
                0,
                1,
                &[unit, divisor.clone(), divisor.clone()],
                None,
            )
            .unwrap();
        let stable_primary = provider
            .evaluate_fiber_equivariant_positive_degree(
                0,
                1,
                &[divisor.clone(), divisor.clone(), divisor],
                None,
            )
            .unwrap();

        // At degree one, applying the descendant divisor equation twice gives
        // <tau_1(1)> = <tau_1(1),H,H> - 2 <H,H,H>.
        let twice_primary = &stable_primary * &RatFun::from_rational(Rational::from(2usize));
        assert!(unstable.equivalent(&(&stable_descendant - &twice_primary)));

        let mut request = TwistedInvariantRequest::new(
            1,
            vec![1],
            0,
            1,
            vec![crate::tau(
                1,
                crate::spaces::projective_space::CohomologyClass::try_h_power(1, 0).unwrap(),
            )],
        )
        .unwrap();
        request.equivariant = true;
        let expanded = compute_negative_split_twisted(&request).unwrap().value;
        let factored = compute_negative_split_twisted_factored(&request)
            .unwrap()
            .to_ratfun();
        assert!(expanded.equivalent(&factored));
    }

    #[test]
    fn default_fiber_weights_do_not_shift_machine_words() {
        let provider = TwistedProjectiveSpaceProvider::new(63, vec![64], false).unwrap();
        assert_eq!(
            provider.rational_fiber_weights(),
            vec![Rational::from(2).pow_usize(64) - Rational::one()]
        );
    }

    #[test]
    fn noncyclic_quantum_h_reports_a_cyclic_basis_mismatch() {
        let zero = QSeries::<Rational>::zero(0);
        let quantum_h = SeriesMatrix::from_entries(vec![
            vec![zero.clone(), zero.clone()],
            vec![zero.clone(), zero.clone()],
        ]);
        let error =
            match CyclicQuantumAlgebra::try_new(quantum_h, "twisted quantum H multiplication") {
                Ok(_) => panic!("a noncyclic multiplication matrix must be rejected"),
                Err(error) => error,
            };
        assert!(matches!(
            error,
            GwError::ConventionMismatch(message) if message.contains("invertible cyclic basis")
        ));
    }
}
