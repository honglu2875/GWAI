//! Twisted projective-space theories by negative split bundles.
//!
//! This module currently provides the target metadata, non-equivariant
//! hypergeometric `I`-function coefficients, bounded line-specialized
//! equivariant `I` coefficients with base and fiber weights, a genus-zero
//! QRR/Lefschetz factorization of the same coefficients, mirror-map
//! normalization, a formal Birkhoff extraction of the descendant `S`-factor, and
//! two semisimple skeletons.  The principal-relation skeleton is diagnostic
//! only: it fails the flat-pairing diagonalization beyond q=0.  The equivariant
//! Birkhoff skeleton validates the inverse-Euler product and low-order
//! Birkhoff/QRR `R` unitarity, including local `P^2`.  The non-equivariant
//! graph path uses an early rational specialization of the one-parameter lambda
//! line.  A fiber-equivariant mode keeps independent symbolic parameters
//! `mu_i` for the split summands while keeping the base weights
//! early-specialized; calibration-level specialization tests cover this mode.
//! The factored coefficient path keeps fiber-equivariant denominators
//! unexpanded through S/R calibration and graph-kernel construction.  Dense
//! symbolic stable-graph leg products remain the main performance frontier.
//! Fast validation currently covers several resolved-conifold rows and the
//! first local-`P^2` genus-2 row; genus-4 local curve computations are the next
//! observed performance frontier.

pub mod theory;
pub use theory::*;
pub mod completion;
pub use completion::*;
pub mod virasoro;
pub use virasoro::*;

pub mod twist;
pub use twist::*;
pub mod i_function;
mod numeric;
pub use i_function::*;
pub mod hypergeometric;
pub use hypergeometric::*;
pub mod mirror;
pub use mirror::*;
pub mod calibration;
pub use calibration::*;
pub mod provider;
pub use provider::*;

pub use crate::givental::Truncation;
pub use crate::reconstruction::{HCoeffLaurentSeries, HLaurentSeries};
pub use crate::spaces::projective_space::resolvent::{ResolventRequest, ResolventResult};
pub use crate::spaces::projective_space::{CohomologyClass, Insertion, InvariantResult};

#[cfg(test)]
mod tests;
