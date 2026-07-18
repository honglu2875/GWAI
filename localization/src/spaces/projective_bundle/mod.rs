//! Split projective bundles `P(O(a_1) + ... + O(a_m))` over `P^n`.
//!
//! [`ProjectiveBundleTheory`] owns the state space and geometric curve classes
//! `(H.beta, xi.beta)`. Ray specialization and shifted Novikov coordinates
//! belong only to the reconstruction adapters implemented here.

pub mod provider;
pub mod theory;
pub mod virasoro;

pub use provider::{
    bundle_dimension_matches, bundle_dimension_matches_in_theory, reconstruct_bundle_invariants,
    reconstruct_bundle_invariants_in_theory, try_bundle_dimension_matches, BundleInsertion,
    BundleRayProvider, ProjectiveBundleRay,
};
pub use theory::*;
pub use virasoro::*;
