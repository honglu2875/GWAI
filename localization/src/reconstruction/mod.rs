//! Target-neutral reconstruction algebra.
//!
//! This module owns coefficient-generic operations shared by ordinary,
//! twisted, product, and projective-bundle targets.  Geometry-specific
//! providers live under their target modules; Laurent Birkhoff factorization
//! and formal series-matrix algebra live here.

mod birkhoff;
pub(crate) use birkhoff::*;
mod series_matrix;
pub(crate) use series_matrix::*;
