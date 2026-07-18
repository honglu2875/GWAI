//! Canonical negative split-bundle twist recipe.

use super::theory::{self, NegativeSplitTotalSpaceTheory};
use crate::core::error::GwError;
use crate::core::theory::{CurveClass, GwTheory};
use crate::spaces::projective_space::ProjectiveSpaceTheory;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct NegativeSplitBundleTwist {
    degrees: Vec<usize>,
}

impl NegativeSplitBundleTwist {
    /// Split bundle `O(-a_1) + ... + O(-a_r)` over `P^n`.
    ///
    /// The stored degrees are the positive integers `a_i`; the signs are part
    /// of the type convention.  Negativity is what gives the concave Euler
    /// factors in the hypergeometric `I`-function below.
    pub fn new(degrees: Vec<usize>) -> Result<Self, GwError> {
        let (degrees, _) =
            theory::canonicalize_negative_split_degrees(degrees).map_err(|issue| match issue {
                theory::NegativeSplitDegreeIssue::NonPositive => GwError::ParseError(
                    "negative split-bundle degrees must be positive".to_string(),
                ),
                theory::NegativeSplitDegreeIssue::SumOverflow => GwError::UnsupportedInvariant(
                    "negative split-bundle degree sum overflow".to_string(),
                ),
            })?;
        Ok(Self { degrees })
    }

    /// Build the hypergeometric twist recipe from canonical target geometry.
    ///
    /// Providers use this path so summand order, rank, and degree data come
    /// from [`NegativeSplitTotalSpaceTheory`] rather than a second parsing and
    /// normalization pass.
    pub fn from_theory(theory: &NegativeSplitTotalSpaceTheory) -> Self {
        Self {
            degrees: theory.degrees().to_vec(),
        }
    }

    pub fn degrees(&self) -> &[usize] {
        &self.degrees
    }

    pub fn rank(&self) -> usize {
        self.degrees.len()
    }

    pub fn degree_sum(&self) -> usize {
        self.degrees.iter().sum()
    }

    fn with_canonical_theory<T>(
        &self,
        base_n: usize,
        use_theory: impl FnOnce(&dyn GwTheory) -> Result<T, GwError>,
    ) -> Result<T, GwError> {
        if self.degrees.is_empty() {
            let theory = ProjectiveSpaceTheory::try_new(base_n)?;
            use_theory(&theory)
        } else {
            let theory = NegativeSplitTotalSpaceTheory::new(base_n, self.degrees.clone())?;
            use_theory(&theory)
        }
    }

    #[deprecated(
        note = "construct NegativeSplitTotalSpaceTheory and call target_dimension() instead"
    )]
    pub fn try_total_space_dimension(&self, base_n: usize) -> Result<usize, GwError> {
        self.with_canonical_theory(base_n, |theory| Ok(theory.target_dimension()))
    }

    #[deprecated(
        note = "construct NegativeSplitTotalSpaceTheory and call target_dimension() instead"
    )]
    #[allow(deprecated)]
    pub fn total_space_dimension(&self, base_n: usize) -> usize {
        self.try_total_space_dimension(base_n)
            .expect("negative-split total-space dimension must be representable")
    }

    #[deprecated(
        note = "construct NegativeSplitTotalSpaceTheory and query its first-Chern pairing instead"
    )]
    pub fn try_is_calabi_yau(&self, base_n: usize) -> Result<bool, GwError> {
        self.with_canonical_theory(base_n, |theory| {
            Ok(theory.c1_pairing(&CurveClass::new(vec![1]))? == 0)
        })
    }

    #[deprecated(
        note = "construct NegativeSplitTotalSpaceTheory and query its first-Chern pairing instead"
    )]
    #[allow(deprecated)]
    pub fn is_calabi_yau(&self, base_n: usize) -> bool {
        self.try_is_calabi_yau(base_n)
            .expect("negative-split canonical theory must be representable")
    }

    #[deprecated(
        note = "construct NegativeSplitTotalSpaceTheory and call virtual_dimension_at_degree() instead"
    )]
    pub fn try_virtual_dimension(
        &self,
        base_n: usize,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Result<isize, GwError> {
        self.with_canonical_theory(base_n, |theory| {
            theory::virtual_dimension_at_degree(theory, genus, degree, markings)
        })
    }

    #[deprecated(
        note = "construct NegativeSplitTotalSpaceTheory and call virtual_dimension_at_degree() instead"
    )]
    #[allow(deprecated)]
    pub fn virtual_dimension(
        &self,
        base_n: usize,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> isize {
        self.try_virtual_dimension(base_n, genus, degree, markings)
            .expect("negative-split virtual dimension must be representable")
    }

    #[deprecated(
        note = "construct NegativeSplitTotalSpaceTheory and call candidate_degrees_from_dimension() instead"
    )]
    pub fn try_candidate_degrees(
        &self,
        base_n: usize,
        genus: usize,
        degree_max: usize,
        markings: usize,
        insertion_degree: Option<usize>,
    ) -> Result<Vec<usize>, GwError> {
        self.with_canonical_theory(base_n, |theory| {
            theory::candidate_degrees_from_dimension(
                theory,
                genus,
                degree_max,
                markings,
                insertion_degree,
            )
        })
    }

    #[deprecated(
        note = "construct NegativeSplitTotalSpaceTheory and call candidate_degrees_from_dimension() instead"
    )]
    #[allow(deprecated)]
    pub fn candidate_degrees(
        &self,
        base_n: usize,
        genus: usize,
        degree_max: usize,
        markings: usize,
        insertion_degree: Option<usize>,
    ) -> Vec<usize> {
        self.try_candidate_degrees(base_n, genus, degree_max, markings, insertion_degree)
            .expect("negative-split degree candidates must be representable")
    }
}
