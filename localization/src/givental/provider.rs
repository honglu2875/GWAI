//! Universal semisimple-CohFT calibration data and provider contract.
//!
//! Concrete providers are owned by their target modules under `spaces`.

use super::{GiventalGraphKernel, SeriesRMatrix, SeriesSMatrix, Truncation};
use crate::core::algebra::{Coeff, RatFun};
use crate::core::error::GwError;
use crate::core::series::{QSeries, SeriesMatrix};
use std::sync::Arc;

/// Semisimple calibration data in a canonical idempotent frame.
///
/// The graph evaluator below only depends on this package of data, not on how
/// it was produced. For projective space it comes from the small J-function;
/// for twisted theories, equivariant theories, r-spin, or other semisimple
/// CohFTs a provider can supply a different calibration with the same shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemisimpleCalibration<C = RatFun> {
    pub r_matrix: SeriesRMatrix<C>,
    pub metric: SeriesMatrix<C>,
    pub psi: SeriesMatrix<C>,
    pub psi_inverse: SeriesMatrix<C>,
    pub connection: SeriesMatrix<C>,
    pub delta: Vec<QSeries<C>>,
    pub inverse_delta: Vec<QSeries<C>>,
    pub relative_sqrt_delta: Vec<QSeries<C>>,
    pub relative_sqrt_delta_inverse: Vec<QSeries<C>>,
}

/// Source of the semisimple data needed by the Givental-Teleman graph engine.
///
/// Coefficients live in `C` over one Novikov variable through `QSeries`.
/// `RatFun` remains the default, preserving the ordinary public provider API,
/// while factored symbolic providers implement the same canonical boundary
/// with a different coefficient type. Genuinely multi-parameter theories
/// should eventually replace `QSeries` behind this boundary rather than
/// duplicating the provider contract.
pub trait SemisimpleCohftProvider<C: Coeff = RatFun> {
    type Insertion;

    /// Number of canonical idempotents, also the number of colors in the graph
    /// sum.
    fn colors(&self) -> usize;

    /// Descendant exponent `k` in an insertion `tau_k(gamma)`.
    fn descendant_power(&self, insertion: &Self::Insertion) -> usize;

    /// Cohomological degree of the whole insertion monomial, when it is known
    /// from the target basis.
    fn insertion_degree(&self, _insertions: &[Self::Insertion]) -> Option<usize> {
        None
    }

    /// Virtual dimension in the target theory. The graph engine uses this only
    /// for pruning; the actual Givental sum is independent of this helper.
    fn virtual_dimension(&self, _genus: usize, _degree: usize, _markings: usize) -> Option<isize> {
        None
    }

    /// Whether the one-parameter Novikov degree represents an effective curve
    /// class for this target.
    ///
    /// Most providers support every nonnegative degree. Degenerate targets can
    /// override this without disguising an empty curve class as a
    /// virtual-dimension mismatch.
    fn degree_is_effective(&self, _degree: usize) -> bool {
        true
    }

    /// Whether a correlator is forced to vanish by the grading of the output
    /// coefficient ring.
    ///
    /// The default is the nonequivariant scalar rule: a known homogeneous
    /// insertion degree must equal a nonnegative virtual dimension. Providers
    /// whose output retains equivariant parameters may override this because
    /// excess insertion degree can be carried by those parameters.
    fn vanishes_by_dimension(&self, virtual_dimension: isize, total_degree: usize) -> bool {
        usize::try_from(virtual_dimension).ok() != Some(total_degree)
    }

    fn expected_degree_from_dimension(
        &self,
        _genus: usize,
        _insertions: &[Self::Insertion],
    ) -> Option<usize> {
        None
    }

    fn candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        if self.insertion_degree(insertions).is_some() {
            self.expected_degree_from_dimension(genus, insertions)
                .filter(|degree| *degree <= degree_max && self.degree_is_effective(*degree))
                .into_iter()
                .collect()
        } else {
            (0..=degree_max)
                .filter(|degree| self.degree_is_effective(*degree))
                .collect()
        }
    }

    /// Descendant-to-ancestor calibration.
    ///
    /// Algebraically, this expands each descendant insertion into ancestor
    /// powers before the `R`-matrix graph action is applied.
    fn descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<C>, GwError>;

    /// Complete reusable graph kernel for a fixed target and truncation.
    ///
    /// The kernel contains `R`, `R^{-1}`, the edge propagator, and translation
    /// coefficients. It is cached aggressively because those objects dominate
    /// repeated series computations.
    fn graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel<C>>, GwError>;

    /// Flat-basis vector for a cohomology insertion.
    ///
    /// The graph evaluator immediately applies `S` and `Psi^{-1}` after this
    /// conversion, so provider implementations should return coefficients in
    /// the same flat basis used by their `S`-matrix.
    fn insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries<C>>, GwError>;

    fn direct_value(
        &self,
        _genus: usize,
        _degree: usize,
        _insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<C>, GwError> {
        Ok(None)
    }

    /// Optional scalar fallback for intentionally small seed cases.
    ///
    /// This is used only after the graph path reports that an unstable range or
    /// missing truncation is outside the implemented graph evaluator.
    fn scalar_fallback_value(
        &self,
        _genus: usize,
        _degree: usize,
        _insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<C>, GwError> {
        Ok(None)
    }
}

/// Deprecated compatibility view of [`SemisimpleCohftProvider`].
///
/// The graph engine uses the canonical trait directly. This one-way blanket
/// adapter remains only so existing callers of the historical `coeff_*`
/// methods keep compiling; it cannot define independent provider behavior.
#[deprecated(
    since = "0.1.0",
    note = "use SemisimpleCohftProvider<C>; the graph engine no longer uses prefixed methods"
)]
pub trait CoefficientSemisimpleCohftProvider<C: Coeff> {
    type Insertion;

    fn coeff_colors(&self) -> usize;
    fn coeff_descendant_power(&self, insertion: &Self::Insertion) -> usize;

    fn coeff_insertion_degree(&self, _insertions: &[Self::Insertion]) -> Option<usize> {
        None
    }

    fn coeff_virtual_dimension(
        &self,
        _genus: usize,
        _degree: usize,
        _markings: usize,
    ) -> Option<isize> {
        None
    }

    fn coeff_degree_is_effective(&self, _degree: usize) -> bool {
        true
    }

    fn coeff_vanishes_by_dimension(&self, virtual_dimension: isize, total_degree: usize) -> bool {
        usize::try_from(virtual_dimension).ok() != Some(total_degree)
    }

    fn coeff_expected_degree_from_dimension(
        &self,
        _genus: usize,
        _insertions: &[Self::Insertion],
    ) -> Option<usize> {
        None
    }

    fn coeff_candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        if self.coeff_insertion_degree(insertions).is_some() {
            self.coeff_expected_degree_from_dimension(genus, insertions)
                .filter(|degree| *degree <= degree_max && self.coeff_degree_is_effective(*degree))
                .into_iter()
                .collect()
        } else {
            (0..=degree_max)
                .filter(|degree| self.coeff_degree_is_effective(*degree))
                .collect()
        }
    }

    fn coeff_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<C>, GwError>;

    fn coeff_graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel<C>>, GwError>;

    fn coeff_insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries<C>>, GwError>;

    fn coeff_direct_value(
        &self,
        _genus: usize,
        _degree: usize,
        _insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<C>, GwError> {
        Ok(None)
    }

    fn coeff_scalar_fallback_value(
        &self,
        _genus: usize,
        _degree: usize,
        _insertions: &[Self::Insertion],
        _truncation: Option<&Truncation>,
    ) -> Result<Option<C>, GwError> {
        Ok(None)
    }
}

#[allow(deprecated)]
impl<C, P> CoefficientSemisimpleCohftProvider<C> for P
where
    C: Coeff,
    P: SemisimpleCohftProvider<C>,
{
    type Insertion = <P as SemisimpleCohftProvider<C>>::Insertion;

    fn coeff_colors(&self) -> usize {
        <P as SemisimpleCohftProvider<C>>::colors(self)
    }

    fn coeff_descendant_power(&self, insertion: &Self::Insertion) -> usize {
        <P as SemisimpleCohftProvider<C>>::descendant_power(self, insertion)
    }

    fn coeff_insertion_degree(&self, insertions: &[Self::Insertion]) -> Option<usize> {
        <P as SemisimpleCohftProvider<C>>::insertion_degree(self, insertions)
    }

    fn coeff_virtual_dimension(
        &self,
        genus: usize,
        degree: usize,
        markings: usize,
    ) -> Option<isize> {
        <P as SemisimpleCohftProvider<C>>::virtual_dimension(self, genus, degree, markings)
    }

    fn coeff_degree_is_effective(&self, degree: usize) -> bool {
        <P as SemisimpleCohftProvider<C>>::degree_is_effective(self, degree)
    }

    fn coeff_vanishes_by_dimension(&self, virtual_dimension: isize, total_degree: usize) -> bool {
        <P as SemisimpleCohftProvider<C>>::vanishes_by_dimension(
            self,
            virtual_dimension,
            total_degree,
        )
    }

    fn coeff_expected_degree_from_dimension(
        &self,
        genus: usize,
        insertions: &[Self::Insertion],
    ) -> Option<usize> {
        <P as SemisimpleCohftProvider<C>>::expected_degree_from_dimension(self, genus, insertions)
    }

    fn coeff_candidate_degrees_from_dimension(
        &self,
        genus: usize,
        degree_max: usize,
        insertions: &[Self::Insertion],
    ) -> Vec<usize> {
        <P as SemisimpleCohftProvider<C>>::candidate_degrees_from_dimension(
            self, genus, degree_max, insertions,
        )
    }

    fn coeff_descendant_s_matrix(
        &self,
        q_degree: usize,
        z_order: usize,
    ) -> Result<SeriesSMatrix<C>, GwError> {
        <P as SemisimpleCohftProvider<C>>::descendant_s_matrix(self, q_degree, z_order)
    }

    fn coeff_graph_kernel(
        &self,
        q_degree: usize,
        r_order: usize,
        graph_dimension: usize,
    ) -> Result<Arc<GiventalGraphKernel<C>>, GwError> {
        <P as SemisimpleCohftProvider<C>>::graph_kernel(self, q_degree, r_order, graph_dimension)
    }

    fn coeff_insertion_vector(
        &self,
        insertion: &Self::Insertion,
        q_degree: usize,
    ) -> Result<Vec<QSeries<C>>, GwError> {
        <P as SemisimpleCohftProvider<C>>::insertion_vector(self, insertion, q_degree)
    }

    fn coeff_direct_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<Option<C>, GwError> {
        <P as SemisimpleCohftProvider<C>>::direct_value(self, genus, degree, insertions, truncation)
    }

    fn coeff_scalar_fallback_value(
        &self,
        genus: usize,
        degree: usize,
        insertions: &[Self::Insertion],
        truncation: Option<&Truncation>,
    ) -> Result<Option<C>, GwError> {
        <P as SemisimpleCohftProvider<C>>::scalar_fallback_value(
            self, genus, degree, insertions, truncation,
        )
    }
}
