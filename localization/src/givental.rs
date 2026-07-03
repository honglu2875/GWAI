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
    bundle_dimension_matches, reconstruct_bundle_invariants, BundleInsertion, BundleRayProvider,
    ProjectiveBundleRay,
};
pub mod product;
pub use product::{
    bidegree_dimension_matches, reconstruct_bidegree_invariants, ProductInsertion,
    ProductProjectiveRay, ProductRayProvider,
};
mod graph;
pub use graph::*;
use r_solve::*;

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
