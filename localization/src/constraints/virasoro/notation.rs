use super::{CanonicalVirasoroConstraint, ConstraintNotation};
use crate::core::algebra::Rational;
use crate::core::error::GwError;
use crate::core::theory::{BasisId, CurveClass, GwTheory};

/// Human-readable names sourced from the same canonical theory that generated
/// the equation.  This object supplies notation only; it cannot alter any
/// coefficient or correlator key in the AST.
pub struct CanonicalTheoryNotation<'a> {
    theory: &'a dyn GwTheory,
}

impl<'a> CanonicalTheoryNotation<'a> {
    pub fn new(theory: &'a dyn GwTheory) -> Self {
        Self { theory }
    }
}

impl ConstraintNotation<CurveClass, BasisId, Rational> for CanonicalTheoryNotation<'_> {
    fn degree_text(&self, degree: &CurveClass) -> String {
        named_degree(
            degree,
            &self.theory.curve_class_space().coordinate_names,
            false,
        )
    }

    fn degree_tex(&self, degree: &CurveClass) -> String {
        named_degree(degree, &self.theory.curve_coordinate_tex_names(), true)
    }

    fn basis_text(&self, basis: &BasisId) -> String {
        self.theory
            .state_space()
            .element(*basis)
            .map(|element| element.label.clone())
            .unwrap_or_else(|| format!("e_{}", basis.0))
    }

    fn basis_tex(&self, basis: &BasisId) -> String {
        self.theory
            .basis_tex(*basis)
            .unwrap_or_else(|| format!("e_{{{}}}", basis.0))
    }

    fn coefficient_text(&self, coefficient: &Rational) -> String {
        coefficient.to_string()
    }

    fn coefficient_tex(&self, coefficient: &Rational) -> String {
        coefficient.to_string()
    }
}

fn named_degree(degree: &CurveClass, names: &[String], tex: bool) -> String {
    if degree.rank() == 1 {
        return degree.to_string();
    }
    let assignments = degree
        .coordinates()
        .iter()
        .enumerate()
        .map(|(index, coordinate)| {
            let raw_name = names
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("d{}", index + 1));
            format!("{raw_name}={coordinate}")
        })
        .collect::<Vec<_>>();
    format!("({})", assignments.join(if tex { ",\\ " } else { ", " }))
}

impl CanonicalVirasoroConstraint {
    fn checked_notation<'a>(
        &self,
        theory: &'a dyn GwTheory,
    ) -> Result<CanonicalTheoryNotation<'a>, GwError> {
        if self.theory_fingerprint != theory.theory_fingerprint() {
            return Err(GwError::ConventionMismatch(
                "cannot render a Virasoro constraint with notation from a different theory"
                    .to_string(),
            ));
        }
        Ok(CanonicalTheoryNotation::new(theory))
    }

    /// Render using labels supplied by the exact canonical theory that
    /// generated this constraint.
    pub fn render_text_for_theory(&self, theory: &dyn GwTheory) -> Result<String, GwError> {
        Ok(self.render_text_with(&self.checked_notation(theory)?))
    }

    /// Render TeX using labels supplied by the exact canonical theory that
    /// generated this constraint.
    pub fn render_tex_for_theory(&self, theory: &dyn GwTheory) -> Result<String, GwError> {
        Ok(self.render_tex_with(&self.checked_notation(theory)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::virasoro::{generate_constraint, TimeMonomial};
    use crate::spaces::product_projective::ProductProjectiveTheory;
    use crate::spaces::projective_bundle::ProjectiveBundleTheory;
    use crate::spaces::projective_space::ProjectiveSpaceTheory;

    #[test]
    fn notation_uses_canonical_theory_basis_and_curve_names() {
        let product = ProductProjectiveTheory::new(1, 2).unwrap();
        let notation = CanonicalTheoryNotation::new(&product);
        assert_eq!(notation.basis_text(&BasisId(5)), "H1 H2^2");
        assert_eq!(
            notation.degree_text(&CurveClass::new(vec![2, 3])),
            "(d1=2, d2=3)"
        );

        let bundle = ProjectiveBundleTheory::new(1, vec![0, 2]).unwrap();
        let notation = CanonicalTheoryNotation::new(&bundle);
        assert!(notation.basis_tex(&BasisId(1)).contains("\\xi"));
    }

    #[test]
    fn tex_basis_powers_brace_multidigit_exponents() {
        let projective = ProjectiveSpaceTheory::new(10);
        let notation = CanonicalTheoryNotation::new(&projective);
        assert_eq!(notation.basis_tex(&BasisId(10)), "H^{10}");

        let product = ProductProjectiveTheory::new(10, 1).unwrap();
        let notation = CanonicalTheoryNotation::new(&product);
        assert_eq!(
            notation.basis_tex(&product.basis_id(10, 0).unwrap()),
            "H_1^{10}"
        );

        let bundle = ProjectiveBundleTheory::new(1, vec![0; 11]).unwrap();
        let notation = CanonicalTheoryNotation::new(&bundle);
        assert_eq!(
            notation.basis_tex(&bundle.basis_id(0, 10).unwrap()),
            "\\xi^{10}"
        );
    }

    #[test]
    fn checked_rendering_rejects_another_theorys_notation() {
        let p1 = ProjectiveSpaceTheory::new(1);
        let p2 = ProjectiveSpaceTheory::new(2);
        let constraint = generate_constraint(&p1, 0, 0, p1.curve(0), TimeMonomial::one()).unwrap();

        assert!(constraint.render_text_for_theory(&p1).is_ok());
        assert!(matches!(
            constraint.render_tex_for_theory(&p2),
            Err(GwError::ConventionMismatch(_))
        ));
    }
}
