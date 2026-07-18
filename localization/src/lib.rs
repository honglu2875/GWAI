//! Experimental exact Gromov--Witten computations for projective spaces,
//! products, projective bundles, and negative split-bundle twists, together
//! with symbolic Virasoro generation and exact compact-theory audits.
//!
//! The crate is intentionally staged. The public computation path is the
//! Givental/S/R graph pipeline, while validation-only backends preserve older
//! convention checks and independent oracle comparisons.
//!
//! The `formula` command is the human-facing explanation path. It renders the
//! stable-graph formula in text or TeX. The default raw basis adds the current
//! backend-specific symbolic calibration dictionary:
//!
//! ```text
//! gw-pn formula --n 2 --g 2 --markings 1 --format tex-fragment
//! gw-pn formula --n 2 --g 2 --markings 1 --basis raw --format tex
//! gw-pn formula --n 2 --g 2 --markings 1 --twist -3 --basis raw --format tex
//! ```
//!
//! The `factored` module keeps denominator factors unexpanded and is the
//! default coefficient engine for symbolic equivariant graph contraction; the
//! expanded `RatFun` engine remains available as a fallback and validation
//! target (`GWAI_DISABLE_FACTORED_GRAPH`).

pub mod algebra;
pub mod constraints;
pub mod core;
pub mod error;
pub mod factored;
pub mod formula;
pub mod frobenius;
pub mod geometry;
pub mod givental;
pub mod graphs;
pub(crate) mod reconstruction;
pub mod resolvent;
pub mod series;
pub mod spaces;
pub mod symbolic;
pub mod tautological;
pub mod testsuite;
pub mod theory;
pub mod twisted;
pub mod validation;
#[doc(hidden)]
pub mod validation_backends;

// Historical crate-root API for ordinary projective space.
pub use spaces::projective_space::api::*;

/// Crate-wide boolean environment flag.
///
/// Enabled by `1`, `true`, `yes`, `on`, or `full` (case-insensitive); unset,
/// empty, or any other value — including `0` — disables. Every debug/tuning
/// flag in the crate goes through this helper so that `FLAG=0` never means
/// "on".
pub(crate) fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "full"
            )
        })
        .unwrap_or(false)
}
