//! The twisted semisimple CohFT providers (rational and factored), the
//! TwistedInvariantRequest, the public compute entry points, and the graph
//! evaluator wiring.

use super::*;
use crate::theory::{CurveClass, CurveEffectivity, GwTheory, NegativeSplitTotalSpaceTheory};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwistedProjectiveSpaceProvider {
    base: ProjectiveSpaceProvider,
    twist: NegativeSplitBundleTwist,
    canonical_theory: Option<NegativeSplitTotalSpaceTheory>,
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

    // Nonconstant maps are stable without markings, but the CohFT graph
    // reconstruction itself needs a stable pointed curve.  Add the minimum
    // number of primary divisors and remove them again with the divisor
    // equation: three in genus zero and one in genus one.
    if req.insertions.is_empty() && req.genus <= 1 {
        let divisor_markings = if req.genus == 0 { 3 } else { 1 };
        let mut stabilized = req.clone();
        stabilized.insertions =
            vec![
                crate::tau(0, crate::geometry::CohomologyClass::h_power(req.n, 1));
                divisor_markings
            ];
        let mut result = compute_negative_split_twisted(&stabilized)?;
        let divisor_factor = Rational::from(req.degree).pow_usize(divisor_markings);
        result.value = &result.value / &RatFun::from_rational(divisor_factor);
        result.notes.push(format!(
            "unmarked genus-{} invariant reconstructed by adding {divisor_markings} hyperplane divisor marking(s) and applying the divisor equation",
            req.genus
        ));
        return Ok(result);
    }

    let provider = if req.equivariant {
        TwistedProjectiveSpaceProvider::fiber_equivariant(req.n, req.twist.degrees().to_vec())?
    } else {
        TwistedProjectiveSpaceProvider::new(req.n, req.twist.degrees().to_vec(), false)?
    };
    if !req.equivariant {
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
    }

    let unstable_two_point = req.genus == 0 && req.insertions.len() == 2;
    let primary_three_point = req.genus == 0 && genus_zero_three_primary_layout(&req.insertions);
    let raw = crate::givental::compute_semisimple_graph_value(
        &provider,
        req.genus,
        req.degree,
        &req.insertions,
        req.truncation.as_ref(),
    )?;
    let value = if req.equivariant {
        raw.lambda_line_limit_preserving_variables(req.n, provider.base_weights())?
    } else {
        match raw.as_rational() {
            Some(value) => RatFun::from_rational(value),
            None => RatFun::from_rational(raw.nonequivariant_limit_line(0, &[Rational::one()])?),
        }
    };
    Ok(InvariantResult {
        value,
        engine: if req.equivariant {
            "twisted-negative-split-fiber-equivariant-givental-birkhoff"
        } else {
            "twisted-negative-split-givental-birkhoff-early-line"
        },
        notes: vec![if unstable_two_point {
            if req.equivariant {
                "computed by the genus-zero two-point unstable S-matrix convention from the fiber-equivariant hypergeometric/Birkhoff calibration; base weights are early-specialized and fiber weights are symbolic mu_i"
                    .to_string()
            } else {
                "computed by the genus-zero two-point unstable S-matrix convention from the same early rational one-parameter lambda-line hypergeometric/Birkhoff calibration; no local oracle shortcut is used"
                    .to_string()
            }
        } else if primary_three_point {
            if req.equivariant {
                "computed by the fiber-equivariant twisted genus-zero Frobenius quantum product from the same hypergeometric/Birkhoff calibration; base weights are early-specialized and fiber weights are symbolic mu_i"
                    .to_string()
            } else {
                "computed by the twisted genus-zero Frobenius quantum product from the same early rational hypergeometric/Birkhoff calibration"
                    .to_string()
            }
        } else if req.equivariant {
            "computed by fiber-equivariant hypergeometric/Birkhoff S and QRR R stable-graph expansion; base weights are early-specialized and fiber weights are symbolic mu_i"
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
    if req.insertions.is_empty() && req.genus <= 1 {
        return compute_negative_split_twisted(req)
            .map(|result| FactoredRatFun::from_ratfun(result.value));
    }

    let provider = FactoredTwistedProjectiveSpaceProvider::fiber_equivariant(
        req.n,
        req.twist.degrees().to_vec(),
    )?;
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
    crate::givental::compute_packed_resolvent_with_provider(
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
    crate::givental::compute_packed_resolvent_with_coeff_provider(
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
        let twist = NegativeSplitBundleTwist::new(degrees.clone())?;
        let canonical_theory = (twist.rank() > 0)
            .then(|| NegativeSplitTotalSpaceTheory::new(n, degrees))
            .transpose()?;
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
        self.canonical_theory.as_ref().ok_or_else(|| {
            GwError::ConventionMismatch(
                "an empty twist is an ordinary projective-space theory, not a negative total space"
                    .to_string(),
            )
        })
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

        self.evaluate_nonequivariant_with_divisor_recursion(genus, degree, insertions.to_vec())
    }

    fn evaluate_nonequivariant_with_divisor_recursion(
        &self,
        genus: usize,
        degree: usize,
        insertions: Vec<Insertion>,
    ) -> Result<RatFun, GwError> {
        let pointed_curve_is_stable = match genus {
            0 => insertions.len() >= 3,
            1 => !insertions.is_empty(),
            _ => true,
        };
        if pointed_curve_is_stable {
            let raw = crate::givental::compute_semisimple_graph_value(
                self,
                genus,
                degree,
                &insertions,
                None,
            )?;
            return match raw.as_rational() {
                Some(value) => Ok(RatFun::from_rational(value)),
                None => Ok(RatFun::from_rational(
                    raw.nonequivariant_limit_line(0, &[Rational::one()])?,
                )),
            };
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
        let maximum = crate::MAX_UNSTABLE_DIVISOR_RECURSION_TOTAL_PSI;
        if total_descendant_power > maximum {
            return Err(GwError::UnsupportedInvariant(format!(
                "twisted unstable descendant degree {total_descendant_power} exceeds the divisor-recursion implementation bound {maximum}"
            )));
        }

        let divisor = crate::tau(
            0,
            crate::geometry::CohomologyClass::try_h_power(self.n(), 1)?,
        );
        let mut with_divisor = insertions.clone();
        with_divisor.push(divisor);
        let mut numerator =
            self.evaluate_nonequivariant_with_divisor_recursion(genus, degree, with_divisor)?;
        for index in 0..insertions.len() {
            if insertions[index].descendant_power == 0 {
                continue;
            }
            let power = insertions[index].class.pure_power().ok_or_else(|| {
                GwError::UnsupportedInvariant(
                    "twisted divisor recursion currently requires homogeneous hyperplane-basis insertions"
                        .to_string(),
                )
            })?;
            if power == self.n() {
                continue;
            }
            let mut correction = insertions.clone();
            correction[index].descendant_power -= 1;
            correction[index].class =
                crate::geometry::CohomologyClass::try_h_power(self.n(), power + 1)?;
            let correction_value =
                self.evaluate_nonequivariant_with_divisor_recursion(genus, degree, correction)?;
            numerator = &numerator - &correction_value;
        }
        Ok(&numerator / &RatFun::from_rational(Rational::from(degree)))
    }

    pub fn fiber_equivariant(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        let mut out = Self::new(n, degrees, false)?;
        out.line_mode = TwistedLineMode::FiberEquivariant;
        out.fiber_parameter_names = default_fiber_parameter_names(out.twist.rank());
        Ok(out)
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
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
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
        let s_matrix = self.descendant_s_matrix(degree, s_order)?;
        let metric = self.flat_metric_matrix(degree)?;
        let descendant = self.insertion_vector(&insertions[descendant_idx], degree)?;
        let primary = self.insertion_vector(&insertions[primary_idx], degree)?;
        genus_zero_two_point_s_matrix_pairing_coeff(
            self.colors(),
            degree,
            s_order,
            &s_matrix,
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
        // Preserve the association between a line summand and its custom
        // equivariant weight while canonicalizing direct-sum order.
        let mut summands = degrees.into_iter().zip(fiber_weights).collect::<Vec<_>>();
        summands.sort_by_key(|(degree, _)| *degree);
        let (degrees, fiber_weights): (Vec<_>, Vec<_>) = summands.into_iter().unzip();
        let mut out = Self::new(n, degrees, false)?;
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

    fn base_weights(&self) -> &[Rational] {
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
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let lambda = crate::algebra::lambda(0);
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
                let lambda = crate::algebra::lambda(0);
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

pub(crate) fn twisted_genus_zero_three_primary_value_coeff<C: Coeff>(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    degree: usize,
    calibration_mode: &TwistedCalibrationMode,
    base_weights: &[C],
    fiber_weights: &[C],
    insertions: &[Vec<QSeries<C>>],
) -> Result<C, GwError> {
    debug_assert_eq!(insertions.len(), 3);
    let model = NegativeSplitLineHypergeometricModel::<C>::from_coeff_weights(
        n,
        twist.clone(),
        degree,
        1,
        base_weights.to_vec(),
        fiber_weights,
    )?;
    let descendant_s = model.birkhoff_descendant_s_matrix(1)?;
    let classical_h = twisted_classical_h_multiplication_matrix_coeff(n, degree, base_weights)?;
    let quantum_h =
        twisted_quantum_multiplication_from_s_coeff(&descendant_s, &classical_h, calibration_mode)?;
    let metric = twisted_inverse_euler_flat_metric_matrix_coeff(
        n,
        degree,
        twist,
        base_weights,
        fiber_weights,
    )?;
    let product = quantum_product_vectors_coeff(&quantum_h, &insertions[0], &insertions[1], degree);
    pair_vectors_coeff(&metric, &product, &insertions[2], degree)
}

pub(crate) fn quantum_product_vectors_coeff<C: Coeff>(
    quantum_h: &SeriesMatrix<C>,
    left: &[QSeries<C>],
    right: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    let colors = quantum_h.rows();
    debug_assert_eq!(left.len(), colors);
    debug_assert_eq!(right.len(), colors);
    let mut result = vec![QSeries::<C>::zero(q_degree); colors];
    let mut h_power_right = right.to_vec();
    for (power, left_coeff) in left.iter().enumerate() {
        if !left_coeff.is_structurally_zero() {
            for row in 0..colors {
                result[row] = result[row].add(&h_power_right[row].mul(left_coeff));
            }
        }
        if power + 1 < left.len() {
            h_power_right =
                apply_series_matrix_to_vector_coeff(quantum_h, &h_power_right, q_degree);
        }
    }
    result
}

pub(crate) fn apply_series_matrix_to_vector_coeff<C: Coeff>(
    matrix: &SeriesMatrix<C>,
    vector: &[QSeries<C>],
    q_degree: usize,
) -> Vec<QSeries<C>> {
    debug_assert_eq!(matrix.cols(), vector.len());
    (0..matrix.rows())
        .map(|row| {
            let mut total = QSeries::<C>::zero(q_degree);
            for (col, vector_coeff) in vector.iter().enumerate() {
                if vector_coeff.is_structurally_zero()
                    || matrix.entry(row, col).is_structurally_zero()
                {
                    continue;
                }
                total = total.add(&matrix.entry(row, col).mul(vector_coeff));
            }
            total
        })
        .collect()
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
}

impl FactoredTwistedProjectiveSpaceProvider {
    pub fn fiber_equivariant(n: usize, degrees: Vec<usize>) -> Result<Self, GwError> {
        Ok(Self {
            inner: TwistedProjectiveSpaceProvider::fiber_equivariant(n, degrees)?,
        })
    }

    pub fn inner(&self) -> &TwistedProjectiveSpaceProvider {
        &self.inner
    }

    fn factored_base_weights(&self) -> Vec<FactoredRatFun> {
        self.inner
            .base_weights()
            .iter()
            .cloned()
            .map(FactoredRatFun::from_rational)
            .collect()
    }

    fn factored_fiber_weights(&self) -> Vec<FactoredRatFun> {
        self.inner
            .fiber_parameter_names
            .iter()
            .cloned()
            .map(FactoredRatFun::variable)
            .collect()
    }

    fn factored_flat_metric_matrix(
        &self,
        q_degree: usize,
    ) -> Result<SeriesMatrix<FactoredRatFun>, GwError> {
        let (metric, _) = twisted_inverse_euler_flat_metric_pair_from_rational_base(
            self.inner.base.n(),
            q_degree,
            &self.inner.twist,
            self.inner.base_weights(),
            &self.factored_fiber_weights(),
        )?;
        Ok(metric)
    }

    fn factored_raw_descendant_s_matrix(
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
        let s_matrix = self.factored_raw_descendant_s_matrix(degree, s_order)?;
        let metric = self.factored_flat_metric_matrix(degree)?;
        let descendant = self.coeff_insertion_vector(&insertions[descendant_idx], degree)?;
        let primary = self.coeff_insertion_vector(&insertions[primary_idx], degree)?;
        genus_zero_two_point_raw_s_matrix_pairing_coeff(
            self.coeff_colors(),
            degree,
            s_order,
            &s_matrix,
            &metric,
            &descendant,
            &primary,
        )
        .map(Some)
    }
}

impl CoefficientSemisimpleCohftProvider<FactoredRatFun> for FactoredTwistedProjectiveSpaceProvider {
    type Insertion = Insertion;

    fn coeff_colors(&self) -> usize {
        self.inner.colors()
    }

    fn coeff_descendant_power(&self, insertion: &Self::Insertion) -> usize {
        self.inner.descendant_power(insertion)
    }

    fn coeff_insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        self.inner.insertion_degree(insertions)
    }

    fn coeff_virtual_dimension(
        &self,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Option<isize> {
        self.inner.virtual_dimension(genus, degree, markings)
    }

    fn coeff_degree_is_effective(&self, degree: usize) -> bool {
        self.inner.degree_is_effective(degree)
    }

    fn coeff_expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        self.inner.expected_degree_from_dimension(genus, insertions)
    }

    fn coeff_candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        self.inner
            .candidate_degrees_from_dimension(genus, degree_max, insertions)
    }

    fn coeff_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<FactoredRatFun>, GwError> {
        let fiber_weights = self.factored_fiber_weights();
        let s_matrix = self.factored_raw_descendant_s_matrix(q_degree, z_order)?;
        let (flat_metric, flat_metric_inverse) =
            twisted_inverse_euler_flat_metric_pair_from_rational_base(
                self.inner.base.n(),
                q_degree,
                &self.inner.twist,
                self.inner.base_weights(),
                &fiber_weights,
            )?;
        metric_adjoint_descendant_s_matrix_with_inverse_coeff(
            s_matrix,
            &flat_metric,
            &flat_metric_inverse,
        )
    }

    fn coeff_graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel<FactoredRatFun>>, GwError> {
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
                twisted_calibration_validation_from_env(),
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
        Ok(kernel)
    }

    fn coeff_insertion_vector(
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

    fn coeff_direct_value(
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
            .map(|insertion| self.coeff_insertion_vector(insertion, degree))
            .collect::<Result<Vec<_>, _>>()?;
        twisted_genus_zero_three_primary_value_coeff(
            self.inner.base.n(),
            &self.inner.twist,
            degree,
            &self.inner.calibration_mode,
            &base_weights,
            &fiber_weights,
            &insertion_vectors,
        )
        .map(Some)
    }

    fn coeff_scalar_fallback_value(
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
        let degree = i64::try_from(degree).ok()?;
        match self.canonical_theory() {
            Ok(theory) => theory
                .virtual_dimension(genus, &CurveClass::new(vec![degree]), markings)
                .ok(),
            // Preserve the historical rank-zero "empty twist" behavior even
            // though it lies outside the negative-total-space theory contract.
            Err(_) if self.twist.rank() == 0 => {
                self.base
                    .virtual_dimension(genus, usize::try_from(degree).ok()?, markings)
            }
            Err(_) => None,
        }
    }

    fn degree_is_effective(&self, degree: usize) -> bool {
        let Ok(degree_i64) = i64::try_from(degree) else {
            return false;
        };
        match self.canonical_theory() {
            Ok(theory) => theory
                .effectivity(&CurveClass::new(vec![degree_i64]))
                .is_ok_and(|effectivity| effectivity == CurveEffectivity::Effective),
            // An empty twist is outside `NegativeSplitTotalSpaceTheory`'s
            // contract but historically behaves as the untwisted base here.
            Err(_) if self.twist.rank() == 0 => self.base.degree_is_effective(degree),
            Err(_) => false,
        }
    }

    fn vanishes_by_dimension(&self, virtual_dimension: isize, total_degree: usize) -> bool {
        if self.line_mode == TwistedLineMode::FiberEquivariant {
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
        if self.line_mode == TwistedLineMode::FiberEquivariant {
            return None;
        }
        if self.twist.rank() == 0 {
            return self.base.expected_degree_from_dimension(genus, insertions);
        }
        let theory = self.canonical_theory().ok()?;
        let insertion_degree = i128::try_from(self.insertion_degree(insertions)?).ok()?;
        let constant_dimension = theory
            .virtual_dimension(genus, &CurveClass::new(vec![0]), insertions.len())
            .ok()? as i128;
        let slope = i128::from(theory.c1_pairing(&CurveClass::new(vec![1])).ok()?);
        if slope == 0 {
            return None;
        }
        let numerator = insertion_degree - constant_dimension;
        if numerator % slope != 0 {
            return None;
        }
        usize::try_from(numerator / slope)
            .ok()
            .filter(|degree| self.degree_is_effective(*degree))
    }

    fn candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        if self.line_mode == TwistedLineMode::FiberEquivariant {
            return (0..=degree_max)
                .filter(|degree| self.degree_is_effective(*degree))
                .collect();
        }
        if self.twist.rank() == 0 {
            return self
                .base
                .candidate_degrees_from_dimension(genus, degree_max, insertions);
        }
        let Some(theory) = self.canonical_theory().ok() else {
            return Vec::new();
        };
        let Some(insertion_degree) = self
            .insertion_degree(insertions)
            .and_then(|degree| i128::try_from(degree).ok())
        else {
            return (0..=degree_max)
                .filter(|degree| self.degree_is_effective(*degree))
                .collect();
        };
        let Some(constant_dimension) = theory
            .virtual_dimension(genus, &CurveClass::new(vec![0]), insertions.len())
            .ok()
            .map(|dimension| dimension as i128)
        else {
            return Vec::new();
        };
        let Some(slope) = theory
            .c1_pairing(&CurveClass::new(vec![1]))
            .ok()
            .map(i128::from)
        else {
            return Vec::new();
        };
        let numerator = insertion_degree - constant_dimension;
        if slope == 0 {
            return if numerator == 0 {
                (0..=degree_max)
                    .filter(|degree| self.degree_is_effective(*degree))
                    .collect()
            } else {
                Vec::new()
            };
        }
        if numerator % slope != 0 {
            return Vec::new();
        }
        usize::try_from(numerator / slope)
            .ok()
            .filter(|degree| *degree <= degree_max && self.degree_is_effective(*degree))
            .into_iter()
            .collect()
    }

    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix, GwError> {
        // The Birkhoff factor produces the fundamental-solution convention.
        // The graph evaluator expects the descendant insertion operator, which
        // is the adjoint with respect to the twisted flat metric.
        let rational_fiber_weights = self.rational_fiber_weights();
        let (s_matrix, flat_metric) = match self.line_mode {
            TwistedLineMode::EarlyRational => {
                let s_matrix =
                    NegativeSplitEquivariantHypergeometricModel::with_default_z_truncation(
                        self.base.n(),
                        self.twist.clone(),
                        q_degree,
                        z_order,
                        self.base_weights().to_vec(),
                        rational_fiber_weights.clone(),
                    )?
                    .birkhoff_descendant_s_matrix(z_order)?;
                let flat_metric = twisted_inverse_euler_flat_metric_matrix(
                    self.base.n(),
                    q_degree,
                    &self.twist,
                    self.base_weights(),
                    &rational_fiber_weights,
                )?;
                (s_matrix, flat_metric)
            }
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
                let base_weights = self.ratfun_base_weights();
                let fiber_weights = self.ratfun_fiber_weights();
                let s_matrix = NegativeSplitLineHypergeometricModel::from_ratfun_weights(
                    self.base.n(),
                    self.twist.clone(),
                    q_degree,
                    z_order,
                    base_weights.clone(),
                    &fiber_weights,
                )?
                .birkhoff_descendant_s_matrix(z_order)?;
                let flat_metric = twisted_inverse_euler_flat_metric_matrix_ratfun(
                    self.base.n(),
                    q_degree,
                    &self.twist,
                    &base_weights,
                    &fiber_weights,
                )?;
                (s_matrix, flat_metric)
            }
        };
        metric_adjoint_descendant_s_matrix(s_matrix, &flat_metric)
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
            Mutex<HashMap<TwistedGraphKernelCacheKey, Arc<GiventalGraphKernel>>>,
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
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(kernel) = cache.lock().unwrap().get(&key).cloned() {
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
            TwistedLineMode::SymbolicLimit | TwistedLineMode::FiberEquivariant => {
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
        twisted_genus_zero_three_primary_value_coeff(
            self.base.n(),
            &self.twist,
            degree,
            &self.calibration_mode,
            &base_weights,
            &fiber_weights,
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

pub(crate) fn metric_adjoint_descendant_s_matrix_coeff<C: Coeff>(
    s_matrix: SeriesSMatrix<C>,
    flat_metric: &SeriesMatrix<C>,
) -> Result<SeriesSMatrix<C>, GwError> {
    // Converts S to its metric adjoint: S^* = eta^{-1} S^T eta.  This is the
    // correct action on covector insertions in the graph formula, and is
    // especially important after inverse-Euler twisting changes the pairing.
    let metric_inverse = invert_series_matrix_coeff(flat_metric)?;
    metric_adjoint_descendant_s_matrix_with_inverse_coeff(s_matrix, flat_metric, &metric_inverse)
}

pub(crate) fn metric_adjoint_descendant_s_matrix_with_inverse_coeff<C: Coeff>(
    s_matrix: SeriesSMatrix<C>,
    flat_metric: &SeriesMatrix<C>,
    metric_inverse: &SeriesMatrix<C>,
) -> Result<SeriesSMatrix<C>, GwError> {
    // Converts S to its metric adjoint: S^* = eta^{-1} S^T eta.  This overload
    // accepts eta^{-1} directly, which is important for factored twisted
    // theories where eta has a closed Vandermonde inverse and Gaussian
    // elimination would expand symbolic rational sums.
    let coefficients = s_matrix
        .coefficients()
        .iter()
        .map(|matrix| metric_inverse.mul(&matrix.transpose()).mul(flat_metric))
        .collect::<Vec<_>>();
    SeriesSMatrix::from_coefficients(
        s_matrix.size(),
        s_matrix.q_degree(),
        s_matrix.z_order(),
        coefficients,
        CalibrationId(format!("{}-metric-adjoint", s_matrix.calibration().0)),
    )
}

#[cfg(test)]
mod canonicalization_tests {
    use super::*;

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
    fn default_fiber_weights_do_not_shift_machine_words() {
        let provider = TwistedProjectiveSpaceProvider::new(63, vec![64], false).unwrap();
        assert_eq!(
            provider.rational_fiber_weights(),
            vec![Rational::from(2).pow_usize(64) - Rational::one()]
        );
    }
}
