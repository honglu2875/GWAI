//! Canonical public entry points grouped by target space.
//!
//! Each peer module puts a target's canonical [`crate::theory::GwTheory`]
//! implementation next to the provider, insertion, reconstruction, and
//! Virasoro-evaluation adapters that operate on it. These modules are a
//! discovery facade only: they reexport the existing implementations and do
//! not introduce another source of geometric data. Their original paths
//! remain available for compatibility.
//!
//! Generic theory data still lives in [`crate::theory`], and the universal
//! reconstruction machinery still lives in [`crate::givental`].

pub mod negative_split_projective;
pub mod product_projective;
pub mod projective_bundle;
pub mod projective_space;
