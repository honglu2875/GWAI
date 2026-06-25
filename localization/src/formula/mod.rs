//! Human-facing Givental graph formula explanations.
//!
//! The fast evaluator in [`crate::givental`] contracts the graph sum directly.
//! This module keeps a separate, verbose representation of the same formal
//! ingredients for theory-level inspection: what the basis elements mean, which
//! truncation orders are finite for fixed `(g,m)`, and how stable graphs are
//! assembled before substituting concrete projective-space or twisted
//! calibration data.
//!
//! The intent is educational and diagnostic, not performance.  Code here
//! should remain readable and explicit even when the production graph evaluator
//! uses more compact data structures.
//!
//! Command examples:
//!
//! ```text
//! gw-pn formula --n 2 --g 2 --markings 1 --format tex-fragment
//! gw-pn formula --n 2 --g 2 --markings 1 --basis raw --format tex
//! gw-pn formula --n 2 --g 2 --markings 1 --twist -3 --basis raw --format tex
//! ```

pub mod basis;
pub mod expansion;
pub mod skeleton;

pub use basis::{all_basis_kinds, basis_glossary, BasisKind};
pub use expansion::FormulaExpansion;
pub use skeleton::{
    build_formula_skeleton, FormulaBasisMode, FormulaRequest, FormulaSkeleton,
    GraphFormulaSkeleton, VertexFormulaSlot,
};
