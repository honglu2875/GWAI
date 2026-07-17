//! Split projective bundles `P(O(a_1) + ... + O(a_m))` over `P^n`.
//!
//! [`ProjectiveBundleTheory`] owns the state space and geometric curve classes
//! `(H.beta, xi.beta)`. Ray specialization and shifted Novikov coordinates
//! belong only to the reconstruction adapters reexported here.

pub use crate::constraints::virasoro::ProjectiveBundleEvaluator;
pub use crate::givental::bundle::{
    bundle_dimension_matches, bundle_dimension_matches_in_theory, reconstruct_bundle_invariants,
    reconstruct_bundle_invariants_in_theory, try_bundle_dimension_matches, BundleInsertion,
    BundleRayProvider, ProjectiveBundleRay,
};
pub use crate::theory::ProjectiveBundleTheory;
