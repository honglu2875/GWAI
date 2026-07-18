//! Negative-split target wrappers around the target-neutral cone-point
//! reconstruction algebra.

use super::{negative_split_mirror_map_coefficients, NegativeSplitBundleTwist};
use crate::core::algebra::Rational;
use crate::core::series::invert_mirror_map;

pub fn negative_split_inverse_mirror_map_coefficients(
    n: usize,
    twist: &NegativeSplitBundleTwist,
    q_degree: usize,
) -> Vec<Rational> {
    let mirror = negative_split_mirror_map_coefficients(n, twist, q_degree);
    invert_mirror_map(&mirror, q_degree)
}
