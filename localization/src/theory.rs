//! Compatibility facade for the historical `crate::theory` path.
//!
//! Universal theory contracts live in [`crate::core::theory`]. Concrete
//! geometry is owned by the corresponding module under [`crate::spaces`].

pub use crate::core::theory::*;
pub use crate::spaces::negative_split_projective::{
    NegativeSplitProjectiveCompletion, NegativeSplitTotalSpaceTheory,
};
pub use crate::spaces::product_projective::ProductProjectiveTheory;
pub use crate::spaces::projective_bundle::ProjectiveBundleTheory;
pub use crate::spaces::projective_space::ProjectiveSpaceTheory;
