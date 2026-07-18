//! Canonical public entry points grouped by target space.
//!
//! Each peer module puts a target's canonical [`crate::core::theory::GwTheory`]
//! implementation next to the provider, insertion, reconstruction, and
//! Virasoro-evaluation adapters that operate on it. Target-specific
//! implementations live in this hierarchy; public reexports flatten each
//! peer's API without creating another source of geometric data. Historical
//! root and `givental` paths remain available for compatibility.
//!
//! Generic theory contracts live in [`crate::core::theory`]. Target-neutral formal
//! reconstruction algebra lives in the crate-private `reconstruction` module,
//! while [`crate::givental`] owns the universal CohFT calibration and stable-
//! graph engine. The historical root paths remain compatibility shims only.

pub mod negative_split_projective;
pub mod product_projective;
pub mod projective_bundle;
pub mod projective_space;

#[cfg(test)]
mod conformance_tests;
