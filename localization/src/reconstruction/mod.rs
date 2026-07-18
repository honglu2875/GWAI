//! Target-neutral reconstruction algebra.
//!
//! This module owns coefficient-generic operations shared by ordinary,
//! twisted, product, and projective-bundle targets.  Geometry-specific
//! providers live under their target modules; Laurent Birkhoff factorization
//! and formal series-matrix algebra live here.

mod birkhoff;
pub(crate) use birkhoff::*;
mod h_laurent;
pub(crate) use h_laurent::*;
pub use h_laurent::{HCoeffLaurentSeries, HLaurentSeries};
mod cyclic;
pub(crate) use cyclic::{CyclicCoordinates, CyclicQuantumAlgebra};
mod interpolation;
pub(crate) use interpolation::ExactRayInterpolation;
pub use interpolation::MAX_EXACT_RECONSTRUCTION_RAYS;
mod linear;
pub(crate) use linear::*;
mod series_matrix;
pub(crate) use series_matrix::*;
mod truncation;
pub(crate) use truncation::*;
