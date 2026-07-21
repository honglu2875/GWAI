//! Exact rational specializations of symbolic constraint coefficients.
//!
//! A specialization is an operation on the coefficient field, not a change of
//! target, sector, time coefficient, or Virasoro convention.  The wrapper in
//! this module therefore retains the generated constraint's metadata verbatim
//! and records the exact point at which its rational-function coefficients
//! were evaluated.

use super::{
    CanonicalVirasoroConstraint, ConstraintTerm, LinearTerm, QuadraticTerm,
    SymbolicVirasoroConstraint, VirasoroConstraint,
};
use crate::core::algebra::Rational;
use crate::core::error::GwError;
use std::collections::{BTreeMap, BTreeSet};

/// A symbolic Virasoro equation evaluated at an exact rational parameter
/// point.
///
/// The inner equation has rational coefficients and can be passed directly to
/// [`super::evaluate_constraint`].  `assignments` is retained separately so a
/// displayed or archived numerical equation still records which point of the
/// equivariant coefficient field produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecializedVirasoroConstraint {
    constraint: CanonicalVirasoroConstraint,
    assignments: BTreeMap<String, Rational>,
}

impl SpecializedVirasoroConstraint {
    /// The rational-coefficient equation, with all non-coefficient metadata
    /// copied verbatim from the symbolic equation.
    pub fn constraint(&self) -> &CanonicalVirasoroConstraint {
        &self.constraint
    }

    /// Exact named parameter values used for the coefficient-field
    /// specialization.
    pub fn assignments(&self) -> &BTreeMap<String, Rational> {
        &self.assignments
    }

    pub fn into_constraint(self) -> CanonicalVirasoroConstraint {
        self.constraint
    }
}

/// Evaluate every rational-function coefficient of a symbolic equation at an
/// exact named rational point.
///
/// The declared `equivariant_parameters` are the source of truth: every one
/// must be assigned and unknown assignment names are rejected.  Evaluation is
/// fail-closed at poles, so a point where any retained coefficient has zero
/// denominator does not produce a purported numerical constraint.
pub fn specialize_symbolic_constraint_parameters(
    constraint: &SymbolicVirasoroConstraint,
    assignments: &BTreeMap<String, Rational>,
) -> Result<SpecializedVirasoroConstraint, GwError> {
    validate_assignment_names(constraint, assignments)?;
    let terms = constraint
        .terms
        .iter()
        .enumerate()
        .map(|(term_index, term)| {
            let specialize = |coefficient: &crate::core::algebra::RatFun| {
                coefficient
                    .evaluate_variables(assignments)
                    .map_err(|error| {
                        GwError::AlgebraFailure(format!(
                            "cannot specialize Virasoro term {term_index}: {error}"
                        ))
                    })
            };
            match term {
                ConstraintTerm::Constant {
                    coefficient,
                    origin,
                } => Ok(ConstraintTerm::Constant {
                    coefficient: specialize(coefficient)?,
                    origin: origin.clone(),
                }),
                ConstraintTerm::Linear(term) => Ok(ConstraintTerm::Linear(LinearTerm::new(
                    specialize(&term.coefficient)?,
                    term.correlator.clone(),
                    term.origin.clone(),
                ))),
                ConstraintTerm::Quadratic(term) => {
                    Ok(ConstraintTerm::Quadratic(QuadraticTerm::new(
                        specialize(&term.coefficient)?,
                        term.left.clone(),
                        term.right.clone(),
                        term.origin.clone(),
                    )))
                }
            }
        })
        .collect::<Result<Vec<_>, GwError>>()?;

    Ok(SpecializedVirasoroConstraint {
        constraint: VirasoroConstraint {
            theory: constraint.theory.clone(),
            theory_fingerprint: constraint.theory_fingerprint.clone(),
            operator: constraint.operator,
            sector: constraint.sector.clone(),
            time_coefficient: constraint.time_coefficient.clone(),
            terms,
            conventions: constraint.conventions.clone(),
            source: constraint.source.clone(),
        },
        assignments: assignments.clone(),
    })
}

fn validate_assignment_names(
    constraint: &SymbolicVirasoroConstraint,
    assignments: &BTreeMap<String, Rational>,
) -> Result<(), GwError> {
    let declared = constraint
        .conventions
        .equivariant_parameters
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if let Some(name) = declared
        .iter()
        .find(|name| !assignments.contains_key(*name))
    {
        return Err(GwError::ConventionMismatch(format!(
            "missing rational assignment for equivariant parameter `{name}`"
        )));
    }
    if let Some(name) = assignments.keys().find(|name| !declared.contains(*name)) {
        return Err(GwError::ConventionMismatch(format!(
            "assignment for undeclared equivariant parameter `{name}`"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::virasoro::{
        CohomologicalGrading, ConstraintSector, Descendant, DilatonShift, FormulaSource,
        PotentialConvention, StateSpaceConvention, TermOrigin, TheoryLabel, TimeMonomial,
        TimeNormalization, UnstableConvention, VirasoroConventions, VirasoroOperator,
    };
    use crate::core::algebra::RatFun;
    use crate::core::theory::{BasisId, CurveClass};

    fn example_constraint(coefficient: RatFun) -> SymbolicVirasoroConstraint {
        SymbolicVirasoroConstraint {
            theory: TheoryLabel::new("example", "X"),
            theory_fingerprint: "example-fingerprint".to_string(),
            operator: VirasoroOperator::new(0),
            sector: ConstraintSector::new(2, CurveClass::new(vec![1])),
            time_coefficient: TimeMonomial::from_descendants([Descendant::new(0, BasisId(0))]),
            terms: vec![ConstraintTerm::Constant {
                coefficient,
                origin: TermOrigin::Other("specialization fixture".to_string()),
            }],
            conventions: VirasoroConventions {
                potential: PotentialConvention::LogarithmicPartitionFunctionEquation,
                time_normalization: TimeNormalization::Exponential,
                dilaton_shift: DilatonShift::StandardUnit,
                grading: CohomologicalGrading::Complex,
                unstable: UnstableConvention::Excluded,
                state_space: StateSpaceConvention::EvenOnly,
                novikov_variables: vec!["q".to_string()],
                equivariant_parameters: vec!["mu_0".to_string(), "mu_1".to_string()],
                notes: vec!["metadata sentinel".to_string()],
            },
            source: FormulaSource {
                title: "source sentinel".to_string(),
                citation: Some("citation sentinel".to_string()),
                locator: None,
                derivation: None,
                notes: Vec::new(),
            },
        }
    }

    #[test]
    fn exact_specialization_records_point_and_preserves_metadata() {
        let mu_0 = RatFun::variable("mu_0");
        let mu_1 = RatFun::variable("mu_1");
        let constraint = example_constraint(&(&mu_0 + &mu_1) / &mu_0);
        let assignments = BTreeMap::from([
            ("mu_0".to_string(), Rational::from(2)),
            ("mu_1".to_string(), Rational::from(3)),
        ]);
        let specialized =
            specialize_symbolic_constraint_parameters(&constraint, &assignments).unwrap();

        assert_eq!(specialized.assignments(), &assignments);
        assert_eq!(specialized.constraint().theory, constraint.theory);
        assert_eq!(
            specialized.constraint().theory_fingerprint,
            constraint.theory_fingerprint
        );
        assert_eq!(specialized.constraint().operator, constraint.operator);
        assert_eq!(specialized.constraint().sector, constraint.sector);
        assert_eq!(
            specialized.constraint().time_coefficient,
            constraint.time_coefficient
        );
        assert_eq!(specialized.constraint().conventions, constraint.conventions);
        assert_eq!(specialized.constraint().source, constraint.source);
        assert!(matches!(
            &specialized.constraint().terms[0],
            ConstraintTerm::Constant { coefficient, .. }
                if coefficient == &Rational::new(5, 2)
        ));
    }

    #[test]
    fn missing_unknown_and_pole_assignments_fail_closed() {
        let mu_0 = RatFun::variable("mu_0");
        let mu_1 = RatFun::variable("mu_1");
        let constraint = example_constraint(&RatFun::one() / &(&mu_0 - &mu_1));

        let missing = BTreeMap::from([("mu_0".to_string(), Rational::one())]);
        assert!(matches!(
            specialize_symbolic_constraint_parameters(&constraint, &missing),
            Err(GwError::ConventionMismatch(message)) if message.contains("mu_1")
        ));

        let unknown = BTreeMap::from([
            ("mu_0".to_string(), Rational::one()),
            ("mu_1".to_string(), Rational::from(2)),
            ("typo".to_string(), Rational::from(3)),
        ]);
        assert!(matches!(
            specialize_symbolic_constraint_parameters(&constraint, &unknown),
            Err(GwError::ConventionMismatch(message)) if message.contains("typo")
        ));

        let pole = BTreeMap::from([
            ("mu_0".to_string(), Rational::from(7)),
            ("mu_1".to_string(), Rational::from(7)),
        ]);
        assert!(matches!(
            specialize_symbolic_constraint_parameters(&constraint, &pole),
            Err(GwError::AlgebraFailure(message)) if message.contains("zero denominator")
        ));
    }
}
