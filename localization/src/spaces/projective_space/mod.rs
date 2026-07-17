//! Ordinary projective space `P^n`.
//!
//! [`ProjectiveSpaceTheory`] is the canonical geometric record. The provider
//! and target types below are computation adapters over that record; they do
//! not restate its dimension, state space, or curve geometry.

pub use crate::constraints::virasoro::ProjectiveSpaceEvaluator;
pub use crate::geometry::{CohomologyClass, EquivariantProjectiveSpace};
pub use crate::givental::{
    projective_space_descendant_s_matrix, projective_space_j_calibration,
    FactoredProjectiveSpaceProvider, ProjectiveSpaceJCalibration, ProjectiveSpaceProvider,
    ProjectiveTarget, TargetProvider,
};
pub use crate::resolvent::{ResolventRequest, ResolventResult};
pub use crate::theory::ProjectiveSpaceTheory;
pub use crate::{
    compute, compute_series, tau, ComputeMode, Insertion, InvariantRequest, InvariantResult,
    SeriesRequest, SeriesResult, Truncation,
};
