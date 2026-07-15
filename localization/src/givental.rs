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
//! The graph code is intentionally target-agnostic.  Projective-space and
//! twisted-projective-space code only differ in how they construct the
//! calibration package.

use crate::algebra::{Coeff, RatFun, Rational};
use crate::error::GwError;
use crate::frobenius::FrobeniusData;
use crate::geometry::elementary_symmetric_weights;
use crate::graphs::{stable_graphs, StableGraph};
use crate::resolvent::{
    enumerate_resolvent_indices, ResolventIndex, ResolventPolynomial, ResolventRequest,
    ResolventResult,
};
use crate::series::{
    integrate_q_derivative_zero_constant_matrix, QSeries, RationalQSeries, SeriesMatrix,
};
use crate::tautological::{TautologicalOracle, WittenKontsevich};
use crate::validation;
use crate::{
    ComputeMode, Insertion, InvariantRequest, InvariantResult, SeriesCoefficient, SeriesRequest,
    SeriesResult, Truncation,
};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

mod classical_limit;
use classical_limit::*;
mod matrices;
mod r_solve;
pub use matrices::*;
mod provider;
pub use provider::*;
pub mod recipe;
pub use recipe::{calibration_from_canonical_frame, descendant_s_from_divisor_qde, CanonicalFrame};
pub mod target;
pub use target::{GwTarget, ProjectiveTarget, TargetProvider};
pub mod bundle;
pub use bundle::{
    bundle_dimension_matches, bundle_dimension_matches_in_theory, reconstruct_bundle_invariants,
    reconstruct_bundle_invariants_in_theory, BundleInsertion, BundleRayProvider,
    ProjectiveBundleRay,
};
pub mod product;
pub use product::{
    bidegree_dimension_matches, bidegree_dimension_matches_in_theory,
    reconstruct_bidegree_invariants, reconstruct_bidegree_invariants_in_theory, ProductInsertion,
    ProductProjectiveRay, ProductRayProvider,
};
mod graph;
pub use graph::*;
use r_solve::*;

/// Maximum number of one-parameter Novikov rays materialized by an exact
/// multi-degree reconstruction.
///
/// Product and projective-bundle reconstruction solve a dense Vandermonde
/// system and currently run one scoped worker per ray.  Keeping this guard in
/// the shared reconstruction layer prevents a large degree from turning a
/// public fallible API into an allocation or thread-spawn abort.  The bound
/// can be raised deliberately once those algorithms use bounded parallelism
/// and a more scalable interpolation strategy.
pub const MAX_EXACT_RECONSTRUCTION_RAYS: usize = 64;

pub(crate) fn checked_reconstruction_ray_count(
    target: &str,
    total_degree: usize,
) -> Result<usize, GwError> {
    let ray_count = total_degree.checked_add(1).ok_or_else(|| {
        GwError::UnsupportedInvariant(format!("{target} reconstruction degree is too large"))
    })?;
    if ray_count > MAX_EXACT_RECONSTRUCTION_RAYS {
        return Err(GwError::UnsupportedInvariant(format!(
            "{target} reconstruction requires {ray_count} Novikov rays, exceeding the explicit limit {MAX_EXACT_RECONSTRUCTION_RAYS}"
        )));
    }
    Ok(ray_count)
}

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
