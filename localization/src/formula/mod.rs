//! Human-facing Givental graph formula explanations.
//!
//! The fast evaluator in [`crate::givental`] contracts the graph sum directly.
//! This module keeps a separate, verbose representation of the same formal
//! ingredients for theory-level inspection: what the atoms mean, which
//! truncation orders are finite for fixed `(g,m)`, and how stable graphs are
//! assembled before substituting concrete projective-space or twisted
//! calibration data.
//!
//! The intent is educational and diagnostic, not performance.  Code here
//! should remain readable and explicit even when the production graph evaluator
//! uses more compact data structures.

pub mod atoms;
pub mod skeleton;
pub mod specialization;

pub use atoms::{all_atom_kinds, atom_glossary, AtomKind};
pub use skeleton::{
    build_formula_skeleton, FormulaRequest, FormulaSkeleton, GraphFormulaSkeleton,
    VertexFormulaSlot,
};
pub use specialization::FormulaSpecialization;
