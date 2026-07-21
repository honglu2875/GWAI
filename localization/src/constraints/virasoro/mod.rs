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
mod qrr;
mod render;
mod report;
mod scan;
mod specialization;

pub use ast::{
    CohomologicalGrading, ConstraintSector, ConstraintTerm, CorrelatorKey, Descendant,
    DilatonShift, FormulaSource, LinearTerm, PotentialConvention, QuadraticTerm,
    StateSpaceConvention, TermOrigin, TheoryLabel, TimeMonomial, TimeNormalization,
    UnstableConvention, VirasoroConstraint, VirasoroConventions, VirasoroOperator,
};
pub use compat::*;
pub(crate) use evaluator::evaluate_with_divisor_recursion;
pub use evaluator::{
    evaluate_constraint, evaluate_constraint_with_bounds, evaluate_symbolic_constraint,
    evaluate_symbolic_constraint_with_bounds, CanonicalCorrelatorEvaluator,
    CorrelatorDimensionPolicy, CorrelatorEvaluationBounds,
};
pub use generator::{
    generate_constraint, generate_constraint_with_term_limit, getzler_bracket,
    CanonicalVirasoroConstraint, DEFAULT_GENERATED_TERM_LIMIT,
    MAX_STANDARD_VIRASORO_OPERATOR_INDEX, MAX_VIRASORO_MARKINGS,
};
pub use notation::CanonicalTheoryNotation;
pub use qrr::{
    qrr_bernoulli_number, QrrConjugationFormula, QrrFactor, QrrHamiltonianTerm,
    MAX_QRR_CHERN_INDEX, MAX_QRR_POSITIVE_Z_POWER,
};
pub use render::{ConstraintNotation, DisplayNotation};
pub use report::{
    EvaluatedTerm, IncompleteReason, MissingCorrelator, ResidualOutcome, ResidualReport,
    ResidualStatus,
};
pub use scan::{
    scan_constraints, VirasoroScanBounds, VirasoroScanEntry, VirasoroScanReport,
    MAX_VIRASORO_SCAN_MARKINGS,
};
pub use specialization::{
    specialize_symbolic_constraint_parameters, SpecializedVirasoroConstraint,
};

use crate::core::algebra::RatFun;
use crate::core::theory::{BasisId, CurveClass};

/// A Virasoro coefficient equation over the symbolic equivariant coefficient
/// field.  QRR-conjugated operators use this type because Euler weights occur
/// in denominators.
pub type SymbolicVirasoroConstraint = VirasoroConstraint<CurveClass, BasisId, RatFun>;
