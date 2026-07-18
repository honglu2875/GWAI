//! Ordinary projective space `P^n`.
//!
//! [`ProjectiveSpaceTheory`] is the canonical geometric record. The provider
//! and target types below are computation adapters over that record; they do
//! not restate its dimension, state space, or curve geometry.

pub mod theory;
pub use theory::*;
pub mod api;
pub use api::*;
mod batch;
pub mod seeds;
pub use batch::{
    compute_packed_resolvent_with_coeff_provider, compute_packed_resolvent_with_provider,
    compute_series_master_with_provider,
};
pub mod resolvent;
pub use resolvent::*;
pub mod provider;
pub use provider::*;
pub mod target;
pub use target::*;
pub mod virasoro;
pub use virasoro::*;

pub mod equivariant;
pub use equivariant::*;
pub mod frobenius;
pub use frobenius::*;

pub use crate::givental::TargetProvider;
