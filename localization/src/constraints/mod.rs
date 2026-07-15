//! Backend-independent identities used to audit Gromov--Witten computations.
//!
//! Constraint formulas live here as symbolic data.  They do not know whether
//! a correlator will be evaluated by the Givental graph engine, localization,
//! or a table of known values.  Likewise, target geometry belongs to the
//! canonical theory; this module only records the formulas derived from that
//! data.

pub mod virasoro;
