use super::ConstraintNotation;
use crate::algebra::Rational;
use crate::theory::{BasisId, CurveClass, GwTheory};

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
        named_degree(
            degree,
            &self.theory.curve_class_space().coordinate_names,
            true,
        )
    }

    fn basis_text(&self, basis: &BasisId) -> String {
        self.theory
            .state_space()
            .element(*basis)
            .map(|element| element.label.clone())
            .unwrap_or_else(|| format!("e_{}", basis.0))
    }

    fn basis_tex(&self, basis: &BasisId) -> String {
        let text = self.basis_text(basis);
        let symbolic = text
            .replace("H1", "H_1")
            .replace("H2", "H_2")
            .replace("xi", "\\xi")
            .replace(' ', "\\,");
        brace_numeric_superscripts(&symbolic)
    }

    fn coefficient_text(&self, coefficient: &Rational) -> String {
        coefficient.to_string()
    }

    fn coefficient_tex(&self, coefficient: &Rational) -> String {
        coefficient.to_string()
    }
}

fn brace_numeric_superscripts(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut characters = input.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '^' {
            output.push(character);
            continue;
        }
        let mut exponent = String::new();
        while characters.peek().is_some_and(|next| next.is_ascii_digit()) {
            exponent.push(characters.next().expect("peeked digit"));
        }
        if exponent.is_empty() {
            output.push('^');
        } else {
            output.push_str("^{");
            output.push_str(&exponent);
            output.push('}');
        }
    }
    output
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
            let name = if tex {
                raw_name
                    .replace("d1", "d_1")
                    .replace("d2", "d_2")
                    .replace("H.beta", "H\\!\\cdot\\!\\beta")
                    .replace("xi.beta", "\\xi\\!\\cdot\\!\\beta")
            } else {
                raw_name
            };
            format!("{name}={coordinate}")
        })
        .collect::<Vec<_>>();
    format!("({})", assignments.join(if tex { ",\\ " } else { ", " }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::{ProductProjectiveTheory, ProjectiveBundleTheory, ProjectiveSpaceTheory};

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
}
