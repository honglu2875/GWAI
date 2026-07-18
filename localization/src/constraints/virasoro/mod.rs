//! Symbolic Virasoro constraints and their audit reports.
//!
//! [`VirasoroConstraint`] is the backend-independent output of the universal
//! generator: evaluators inspect its correlator keys, substitute exact values,
//! and return a [`ResidualReport`].  Keeping the representation apart from
//! generation makes it possible to review the mathematical convention before
//! any generated equation is used as a test oracle.

mod ast;
mod compat;
mod evaluator;
mod generator;
mod notation;
mod render;
mod report;
mod scan;

pub use ast::{
    CohomologicalGrading, ConstraintSector, ConstraintTerm, CorrelatorKey, Descendant,
    DilatonShift, FormulaSource, LinearTerm, PotentialConvention, QuadraticTerm,
    StateSpaceConvention, TermOrigin, TheoryLabel, TimeMonomial, TimeNormalization,
    UnstableConvention, VirasoroConstraint, VirasoroConventions, VirasoroOperator,
};
pub use compat::*;
pub(crate) use evaluator::evaluate_with_divisor_recursion;
pub use evaluator::{
    evaluate_constraint, evaluate_constraint_with_bounds, CanonicalCorrelatorEvaluator,
    CorrelatorEvaluationBounds,
};
pub use generator::{
    generate_constraint, generate_constraint_with_term_limit, getzler_bracket,
    CanonicalVirasoroConstraint, DEFAULT_GENERATED_TERM_LIMIT,
    MAX_STANDARD_VIRASORO_OPERATOR_INDEX, MAX_VIRASORO_MARKINGS,
};
pub use notation::CanonicalTheoryNotation;
pub use render::{ConstraintNotation, DisplayNotation};
pub use report::{
    EvaluatedTerm, IncompleteReason, MissingCorrelator, ResidualOutcome, ResidualReport,
    ResidualStatus,
};
pub use scan::{
    scan_constraints, VirasoroScanBounds, VirasoroScanEntry, VirasoroScanReport,
    MAX_VIRASORO_SCAN_MARKINGS,
};
