//! Symbolic algebra support for formula-level rationalization.
//!
//! This module is intentionally separate from the numerical Givental evaluator.
//! Its job is to manipulate the algebraic expressions that appear after a raw
//! graph formula has been specialized to a concrete semisimple theory.  The
//! first supported case is a one-generator quotient such as ordinary
//! projective space, where canonical roots satisfy a monic relation
//! `P(u_i)=0`.

pub mod projective;
pub mod quotient;
pub mod univariate;

pub use projective::{
    projective_quotient, projective_relation, projective_residue_monomial,
    projective_trace_monomial,
};
pub use quotient::OneGeneratorQuotient;
pub use univariate::UniPoly;
