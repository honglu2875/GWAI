//! Foundational contracts and exact algebra shared by every target space.
//!
//! Concrete target geometry belongs under [`crate::spaces`]. Reconstruction,
//! constraints, and Givental contraction consume these primitives without
//! adding parallel descriptions of a space.

pub mod algebra;
pub(crate) mod bounded_cache;
pub mod error;
pub(crate) mod fused;
pub mod moduli;
pub mod series;
pub mod theory;
