//! Compact projective-bundle completion of a negative split total space.

use super::theory::NegativeSplitTotalSpaceTheory;
use crate::core::algebra::Rational;
use crate::core::error::GwError;
use crate::core::theory::{scan_bound_overflow, BasisId, CurveClass};
use crate::spaces::projective_bundle::ProjectiveBundleTheory;

/// Compact projective-bundle geometry containing a negative split total space.
///
/// For `V = direct_sum_i O(-a_i)` over `P^n`, let `A = max_i a_i`.
/// Tensoring `O + V` by `O(A)` gives the normalized line-projectivization
///
/// `P(O(A) + direct_sum_i O(A-a_i))`.
///
/// With the convention `xi = -c1(S)` used by [`ProjectiveBundleTheory`], the
/// section corresponding to `O(A)` has `xi|_S = -A H`.  Consequently its
/// degree-`d` curve class is `(d, -A d)`, and
/// `H^h xi^j|_S = (-A)^j H^(h+j)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeSplitProjectiveCompletion {
    local_theory: NegativeSplitTotalSpaceTheory,
    compact_theory: ProjectiveBundleTheory,
    max_degree: usize,
}

impl NegativeSplitProjectiveCompletion {
    pub fn new(local_theory: NegativeSplitTotalSpaceTheory) -> Result<Self, GwError> {
        let max_degree = *local_theory
            .degrees()
            .iter()
            .max()
            .expect("negative split theory has at least one degree");
        let twist_count = local_theory.degrees().len().checked_add(1).ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "negative-split projective-completion rank overflow".to_string(),
            )
        })?;
        let mut twists = Vec::new();
        twists.try_reserve_exact(twist_count).map_err(|_| {
            GwError::UnsupportedInvariant(
                "cannot allocate negative-split projective-completion twists".to_string(),
            )
        })?;
        twists.push(max_degree);
        twists.extend(
            local_theory
                .degrees()
                .iter()
                .map(|degree| max_degree - degree),
        );
        let compact_theory = ProjectiveBundleTheory::new(local_theory.base_dimension(), twists)?;
        Ok(Self {
            local_theory,
            compact_theory,
            max_degree,
        })
    }

    pub fn local_theory(&self) -> &NegativeSplitTotalSpaceTheory {
        &self.local_theory
    }

    pub fn compact_theory(&self) -> &ProjectiveBundleTheory {
        &self.compact_theory
    }

    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// The compact curve class of a degree-`degree` map to the distinguished
    /// section.
    pub fn section_curve(&self, degree: usize) -> Result<CurveClass, GwError> {
        let degree_i64 = i64::try_from(degree).map_err(|_| scan_bound_overflow())?;
        let max_degree_i64 = i64::try_from(self.max_degree).map_err(|_| {
            GwError::UnsupportedInvariant(
                "projective-completion normalization degree does not fit the curve lattice"
                    .to_string(),
            )
        })?;
        let fiber_degree = max_degree_i64
            .checked_mul(degree_i64)
            .and_then(i64::checked_neg)
            .ok_or_else(scan_bound_overflow)?;
        self.compact_theory.try_curve(degree, fiber_degree)
    }

    /// Return the base degree exactly when `curve` is a class supported on the
    /// distinguished section.
    pub fn section_degree(&self, curve: &CurveClass) -> Option<usize> {
        let (base_degree, fiber_degree) = self.compact_theory.bidegree(curve)?;
        let base_degree_i64 = i64::try_from(base_degree).ok()?;
        let max_degree_i64 = i64::try_from(self.max_degree).ok()?;
        let expected_fiber_degree = max_degree_i64.checked_mul(base_degree_i64)?.checked_neg()?;
        (fiber_degree == expected_fiber_degree).then_some(base_degree)
    }

    /// Restrict a compact-bundle basis element to the distinguished section.
    ///
    /// `Ok(None)` denotes the zero class, including powers `H^h xi^j` with
    /// `h + j > n`.  Valid nonzero restrictions are returned as a local basis
    /// id followed by its exact scalar coefficient; an invalid compact basis
    /// id is an error rather than a false zero.
    pub fn restrict_basis_to_section(
        &self,
        basis: BasisId,
    ) -> Result<Option<(BasisId, Rational)>, GwError> {
        let (h_power, xi_power) = self.compact_theory.basis_powers(basis).ok_or_else(|| {
            GwError::ConventionMismatch(
                "projective-completion insertion is outside the compact bundle basis".to_string(),
            )
        })?;
        let local_power = h_power.checked_add(xi_power).ok_or_else(|| {
            GwError::UnsupportedInvariant(
                "projective-completion insertion degree overflow".to_string(),
            )
        })?;
        if local_power > self.local_theory.base_dimension() {
            return Ok(None);
        }
        let coefficient = (-Rational::from(self.max_degree)).pow_usize(xi_power);
        Ok(Some((BasisId(local_power), coefficient)))
    }
}
