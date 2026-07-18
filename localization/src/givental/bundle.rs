//! Compatibility exports for the projective-bundle provider.
//!
//! The implementation lives with its target under
//! [`crate::spaces::projective_bundle`]. This historical module preserves the
//! `crate::givental::bundle` API.

pub use crate::spaces::projective_bundle::{
    bundle_dimension_matches, bundle_dimension_matches_in_theory, reconstruct_bundle_invariants,
    reconstruct_bundle_invariants_in_theory, try_bundle_dimension_matches, BundleInsertion,
    BundleRayProvider, ProjectiveBundleRay,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_module_reexports_the_canonical_bundle_types() {
        let insertion = BundleInsertion::new(1, 2, 3);
        let canonical: crate::spaces::projective_bundle::BundleInsertion = insertion;
        assert_eq!(canonical.descendant_power, 1);
        assert_eq!(canonical.h_power, 2);
        assert_eq!(canonical.xi_power, 3);
    }
}
