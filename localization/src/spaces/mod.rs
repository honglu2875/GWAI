//! Canonical public entry points grouped by target space.
//!
//! Each peer module puts a target's canonical [`crate::theory::GwTheory`]
//! implementation next to the provider, insertion, reconstruction, and
//! Virasoro-evaluation adapters that operate on it. These modules are a
//! discovery facade only: they reexport the existing implementations and do
//! not introduce another source of geometric data. Their original paths
//! remain available for compatibility.
//!
//! Generic theory contracts live in [`crate::theory`]. Target-neutral formal
//! reconstruction algebra lives in the crate-private `reconstruction` module,
//! while [`crate::givental`] owns the universal CohFT calibration and stable-
//! graph engine. The historical root paths remain compatibility shims only.

pub mod negative_split_projective;
pub mod product_projective;
pub mod projective_bundle;
pub mod projective_space;
