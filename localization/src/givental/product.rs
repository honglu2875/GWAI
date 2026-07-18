//! Compatibility path for products of projective spaces.
//!
//! The implementation lives with its target under
//! [`crate::spaces::product_projective`]. This module preserves the historical
//! `crate::givental::product` API.

pub use crate::spaces::product_projective::{
    bidegree_dimension_matches, bidegree_dimension_matches_in_theory,
    reconstruct_bidegree_invariants, reconstruct_bidegree_invariants_in_theory,
    try_bidegree_dimension_matches, ProductInsertion, ProductProjectiveRay, ProductRayProvider,
};
