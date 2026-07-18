//! Products `P^n x P^m`.
//!
//! [`ProductProjectiveTheory`] is the canonical geometric record. Ray objects
//! and providers specialize its two Novikov variables only for computation;
//! exact reconstruction returns the geometric bidegrees.

pub mod provider;
pub mod theory;
pub mod virasoro;

pub use provider::{
    bidegree_dimension_matches, bidegree_dimension_matches_in_theory,
    reconstruct_bidegree_invariants, reconstruct_bidegree_invariants_in_theory,
    try_bidegree_dimension_matches, ProductInsertion, ProductProjectiveRay, ProductRayProvider,
};
pub use theory::*;
pub use virasoro::*;
