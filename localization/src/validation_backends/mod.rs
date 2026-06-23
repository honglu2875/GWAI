//! Validation-only backends and oracle tables.
//!
//! These modules are not computation shortcuts for the production engines. They
//! provide independent checks, external ground-truth rows, and diagnostic
//! scripts used to keep the Givental/S/R implementation honest.

pub mod growi;
pub mod legacy_localization;
pub mod local_cy;
pub mod zinger;
