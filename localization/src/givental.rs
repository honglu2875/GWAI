//! Givental-Teleman reconstruction and graph contraction.
//!
//! This module implements the universal semisimple CohFT part of the package.
//! Target-specific geometry enters through `SemisimpleCohftProvider`: the
//! provider supplies flat insertions, descendant-to-ancestor `S`, canonical
//! transition data, and an `R`-matrix.  The code here then performs the common
//! Givental graph sum over stable curves.
//!
//! The main mathematical transformations are:
//!
//! - descendents -> ancestors by the `S`-matrix;
//! - flat basis -> canonical idempotent basis by `Psi^{-1}`;
//! - ancestor legs -> graph legs by `R^{-1}`;
//! - internal edges -> the symplectic propagator built from `R^{-1}` and the
//!   canonical metric;
//! - unstable translations -> insertions of `T(psi) = psi(1 - R^{-1})1`;
//! - vertices -> products of point-theory psi integrals and the diagonal TFT.
//!
//! The graph code is intentionally target-agnostic. Ordinary projective
//! spaces, products, projective bundles, and negative-split twists differ in
//! how their providers construct the calibration package; exact ray
//! interpolation, Birkhoff window planning, and cyclic-basis algebra are
//! shared through the crate-private `reconstruction` module.

/// Optional caller-imposed cap on the `z`-order of the `S`/`R` calibration.
///
/// The graph engine derives the order it needs from each request and returns
/// [`GwError::TruncationTooLow`](crate::core::error::GwError::TruncationTooLow)
/// when this cap is below that. Only the `z`-order is configurable; earlier
/// revisions carried additional fields
/// (`q_degree`, `descendant_degree`, `genus`) that were never consulted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncation {
    pub z_order: usize,
}

mod classical_limit;
pub(crate) use classical_limit::*;
mod matrices;
mod r_solve;
pub use matrices::*;
mod provider;
pub use provider::*;
// Historical `givental::*` paths for target-owned projective-space providers.
pub use crate::spaces::projective_space::provider::{
    projective_space_descendant_s_matrix, projective_space_j_calibration,
    FactoredProjectiveSpaceProvider, ProjectiveSpaceJCalibration, ProjectiveSpaceProvider,
};
pub mod recipe;
pub use recipe::{calibration_from_canonical_frame, descendant_s_from_divisor_qde, CanonicalFrame};
pub mod target;
pub use crate::spaces::projective_space::ProjectiveTarget;
pub use target::{GwTarget, SemisimpleTarget, TargetProvider};
// Historical module paths; implementations are target-owned under `spaces`.
pub mod bundle;
pub mod product;
pub use crate::spaces::product_projective::{
    bidegree_dimension_matches, bidegree_dimension_matches_in_theory,
    reconstruct_bidegree_invariants, reconstruct_bidegree_invariants_in_theory,
    try_bidegree_dimension_matches, ProductInsertion, ProductProjectiveRay, ProductRayProvider,
};
pub use crate::spaces::projective_bundle::{
    bundle_dimension_matches, bundle_dimension_matches_in_theory, reconstruct_bundle_invariants,
    reconstruct_bundle_invariants_in_theory, try_bundle_dimension_matches, BundleInsertion,
    BundleRayProvider, ProjectiveBundleRay,
};
pub use crate::spaces::projective_space::api::{
    compute_by_givental_graphs, compute_givental as compute, compute_projective_resolvent_packed,
    compute_series_master, projective_graph_bounded_potential_coefficients,
};
pub use crate::spaces::projective_space::{
    compute_packed_resolvent_with_coeff_provider, compute_packed_resolvent_with_provider,
    compute_series_master_with_provider,
};
mod graph;
pub use graph::*;
pub(crate) use r_solve::*;

// Compatibility export: the implementation and guard now live with the
// target-neutral exact-ray interpolation algorithm.
pub use crate::reconstruction::MAX_EXACT_RECONSTRUCTION_RAYS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaConvention {
    MetricNorm,
    InverseMetricNorm,
}

/// Identifies the convention used to produce a calibration.
///
/// This is not used as mathematics; it is metadata that keeps tests and error
/// messages honest when several possible `R`/`S` normalizations are present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationId(pub String);

/// Basis and normalization of the semisimple frame.
///
/// The graph evaluator assumes diagonal TFT vertices.  The exact powers of the
/// canonical metric depend on whether idempotents have already been normalized
/// by square roots of metric norms, so we keep the convention explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalFrameConvention {
    FlatBasis,
    UnnormalizedCanonicalIdempotents,
    RelativeNormalizedCanonicalIdempotents,
    NormalizedCanonicalIdempotents,
}
