//! Historical constraint-oriented paths for target-owned evaluator adapters.
//!
//! Generic Virasoro generation and evaluation do not depend on concrete
//! spaces. This facade alone preserves the pre-refactor names under
//! `constraints::virasoro`.

pub use crate::spaces::negative_split_projective::NegativeSplitCompletionEvaluator;
pub use crate::spaces::product_projective::ProductProjectiveEvaluator;
pub use crate::spaces::projective_bundle::ProjectiveBundleEvaluator;
pub use crate::spaces::projective_space::ProjectiveSpaceEvaluator;
