//! Negative split-bundle theories over projective space.
//!
//! [`NegativeSplitTotalSpaceTheory`] records the noncompact total-space
//! geometry used by the twisted providers. It deliberately does not invent
//! compact pairing data. [`NegativeSplitProjectiveCompletion`] is the separate
//! compact completion used by [`NegativeSplitCompletionEvaluator`] for
//! ordinary Virasoro checks.

pub use crate::constraints::virasoro::NegativeSplitCompletionEvaluator;
pub use crate::geometry::CohomologyClass;
pub use crate::resolvent::{ResolventRequest, ResolventResult};
pub use crate::theory::{NegativeSplitProjectiveCompletion, NegativeSplitTotalSpaceTheory};
pub use crate::twisted::{
    compute_negative_split_twisted, compute_negative_split_twisted_factored,
    compute_negative_split_twisted_resolvent_packed,
    compute_negative_split_twisted_resolvent_packed_factored,
    FactoredTwistedProjectiveSpaceProvider, NegativeSplitBundleTwist, TwistedCalibrationMode,
    TwistedInvariantRequest, TwistedProjectiveSpaceProvider,
};
pub use crate::{Insertion, InvariantResult, Truncation};
