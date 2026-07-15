use std::fmt::Display;
use std::fmt::Write;

use super::ast::{ConstraintTerm, CorrelatorKey, Descendant, TimeMonomial, VirasoroConstraint};

/// Target-specific notation used only for presentation.
///
/// Implementations must not supply mathematical data: the degree, basis key,
/// and coefficient have already been chosen by the constraint generator.  The
/// trait merely gives those existing values text and TeX spellings.
pub trait ConstraintNotation<D, B, C> {
    fn degree_text(&self, degree: &D) -> String;
    fn degree_tex(&self, degree: &D) -> String;
    fn basis_text(&self, basis: &B) -> String;
    fn basis_tex(&self, basis: &B) -> String;
    fn coefficient_text(&self, coefficient: &C) -> String;
    fn coefficient_tex(&self, coefficient: &C) -> String;
}

/// Basic notation for atom types whose `Display` spelling is already suitable
/// for both plain text and TeX math mode.
#[derive(Debug, Clone, Copy, Default)]
pub struct DisplayNotation;

impl<D: Display, B: Display, C: Display> ConstraintNotation<D, B, C> for DisplayNotation {
    fn degree_text(&self, degree: &D) -> String {
        degree.to_string()
    }

    fn degree_tex(&self, degree: &D) -> String {
        degree.to_string()
    }

    fn basis_text(&self, basis: &B) -> String {
        basis.to_string()
    }

    fn basis_tex(&self, basis: &B) -> String {
        basis.to_string()
    }

    fn coefficient_text(&self, coefficient: &C) -> String {
        coefficient.to_string()
    }

    fn coefficient_tex(&self, coefficient: &C) -> String {
        coefficient.to_string()
    }
}

impl<D, B, C> VirasoroConstraint<D, B, C> {
    pub fn render_text_with<N: ConstraintNotation<D, B, C>>(&self, notation: &N) -> String {
        let mut out = String::new();
        writeln!(out, "Virasoro coefficient constraint").unwrap();
        writeln!(out, "===============================").unwrap();
        writeln!(out, "Theory: {}", self.theory.text).unwrap();
        writeln!(out, "Operator: L_{}", self.operator.index).unwrap();
        writeln!(
            out,
            "Sector: g={}, beta={}",
            self.sector.genus,
            notation.degree_text(&self.sector.degree)
        )
        .unwrap();
        writeln!(
            out,
            "Extracted coefficient: {}",
            render_time_text(&self.time_coefficient, notation)
        )
        .unwrap();

        out.push_str("\nFormula:\n");
        if self.terms.is_empty() {
            out.push_str("  0 = 0\n");
        } else {
            for (index, term) in self.terms.iter().enumerate() {
                if index == 0 {
                    out.push_str("  0 = ");
                } else {
                    out.push_str("    + ");
                }
                out.push_str(&render_term_text(term, notation));
                out.push_str("  [");
                out.push_str(term.origin().label());
                out.push_str("]\n");
            }
        }

        out.push_str("\nConventions:\n");
        writeln!(out, "- Potential: {}", self.conventions.potential.label()).unwrap();
        writeln!(
            out,
            "- Time normalization: {}",
            self.conventions.time_normalization.label()
        )
        .unwrap();
        writeln!(
            out,
            "- Dilaton shift: {}",
            self.conventions.dilaton_shift.label()
        )
        .unwrap();
        writeln!(out, "- Grading: {}", self.conventions.grading.label()).unwrap();
        writeln!(
            out,
            "- Unstable terms: {}",
            self.conventions.unstable.label()
        )
        .unwrap();
        writeln!(
            out,
            "- State space: {}",
            self.conventions.state_space.label()
        )
        .unwrap();
        writeln!(
            out,
            "- Novikov variables: {}",
            comma_list(&self.conventions.novikov_variables)
        )
        .unwrap();
        writeln!(
            out,
            "- Equivariant parameters: {}",
            comma_list(&self.conventions.equivariant_parameters)
        )
        .unwrap();
        for note in &self.conventions.notes {
            writeln!(out, "- Note: {note}").unwrap();
        }

        out.push_str("\nSource:\n");
        writeln!(out, "- {}", self.source.title).unwrap();
        if let Some(citation) = &self.source.citation {
            writeln!(out, "- Citation: {citation}").unwrap();
        }
        if let Some(locator) = &self.source.locator {
            writeln!(out, "- Locator: {locator}").unwrap();
        }
        if let Some(derivation) = &self.source.derivation {
            writeln!(out, "- Derivation: {derivation}").unwrap();
        }
        for note in &self.source.notes {
            writeln!(out, "- Note: {note}").unwrap();
        }
        out
    }

    pub fn render_tex_with<N: ConstraintNotation<D, B, C>>(&self, notation: &N) -> String {
        let mut out = String::new();
        out.push_str("\\subsection*{Virasoro coefficient constraint}\n");
        writeln!(out, "\\textbf{{Theory:}} \\({}\\)\\\\", self.theory.tex).unwrap();
        writeln!(
            out,
            "\\textbf{{Operator:}} $L_{{{}}}$\\\\",
            self.operator.index
        )
        .unwrap();
        writeln!(
            out,
            "\\textbf{{Sector:}} $g={},\\ \\beta={}$\\\\",
            self.sector.genus,
            notation.degree_tex(&self.sector.degree)
        )
        .unwrap();
        writeln!(
            out,
            "\\textbf{{Extracted coefficient:}} ${}$\n",
            render_time_tex(&self.time_coefficient, notation)
        )
        .unwrap();

        out.push_str("\\[\n\\begin{aligned}\n");
        if self.terms.is_empty() {
            out.push_str("0 &= 0.\n");
        } else {
            for (index, term) in self.terms.iter().enumerate() {
                if index == 0 {
                    out.push_str("0 &={}");
                } else {
                    out.push_str("\\\\\n  &\\mathrel{+}{}");
                }
                write!(
                    out,
                    "{} &&\\text{{{}}}",
                    render_term_tex(term, notation),
                    tex_escape(term.origin().label())
                )
                .unwrap();
            }
            out.push_str(".\n");
        }
        out.push_str("\\end{aligned}\n\\]\n");

        out.push_str("\\paragraph{Conventions.}\n\\begin{itemize}\n");
        tex_item(&mut out, "Potential", self.conventions.potential.label());
        tex_item(
            &mut out,
            "Time normalization",
            self.conventions.time_normalization.label(),
        );
        tex_item(
            &mut out,
            "Dilaton shift",
            self.conventions.dilaton_shift.label(),
        );
        tex_item(&mut out, "Grading", self.conventions.grading.label());
        tex_item(
            &mut out,
            "Unstable terms",
            self.conventions.unstable.label(),
        );
        tex_item(
            &mut out,
            "State space",
            self.conventions.state_space.label(),
        );
        tex_item(
            &mut out,
            "Novikov variables",
            &comma_list(&self.conventions.novikov_variables),
        );
        tex_item(
            &mut out,
            "Equivariant parameters",
            &comma_list(&self.conventions.equivariant_parameters),
        );
        for note in &self.conventions.notes {
            tex_item(&mut out, "Note", note);
        }
        out.push_str("\\end{itemize}\n");

        out.push_str("\\paragraph{Source.}\n\\begin{itemize}\n");
        tex_item(&mut out, "Formula", &self.source.title);
        if let Some(citation) = &self.source.citation {
            tex_item(&mut out, "Citation", citation);
        }
        if let Some(locator) = &self.source.locator {
            tex_item(&mut out, "Locator", locator);
        }
        if let Some(derivation) = &self.source.derivation {
            tex_item(&mut out, "Derivation", derivation);
        }
        for note in &self.source.notes {
            tex_item(&mut out, "Note", note);
        }
        out.push_str("\\end{itemize}\n");
        out
    }
}

fn render_time_text<D, B, C, N: ConstraintNotation<D, B, C>>(
    monomial: &TimeMonomial<B>,
    notation: &N,
) -> String {
    if monomial.is_one() {
        return "1".to_string();
    }
    monomial
        .factors()
        .map(|(descendant, multiplicity)| {
            let factor = format!(
                "t_{}({})",
                descendant.psi_power,
                notation.basis_text(&descendant.class)
            );
            if multiplicity == 1 {
                factor
            } else {
                format!("{factor}^{multiplicity}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_time_tex<D, B, C, N: ConstraintNotation<D, B, C>>(
    monomial: &TimeMonomial<B>,
    notation: &N,
) -> String {
    if monomial.is_one() {
        return "1".to_string();
    }
    monomial
        .factors()
        .map(|(descendant, multiplicity)| {
            let factor = format!(
                "t_{{{}}}^{{{}}}",
                descendant.psi_power,
                notation.basis_tex(&descendant.class)
            );
            if multiplicity == 1 {
                factor
            } else {
                // `factor` already carries the basis-label superscript.
                // Group it before adding a multiplicity exponent, otherwise
                // TeX sees an invalid double superscript such as `t_0^1^2`.
                format!("\\left({factor}\\right)^{{{multiplicity}}}")
            }
        })
        .collect::<Vec<_>>()
        .join("\\,")
}

fn render_descendant_text<D, B, C, N: ConstraintNotation<D, B, C>>(
    descendant: &Descendant<B>,
    notation: &N,
) -> String {
    format!(
        "tau_{}({})",
        descendant.psi_power,
        notation.basis_text(&descendant.class)
    )
}

fn render_descendant_tex<D, B, C, N: ConstraintNotation<D, B, C>>(
    descendant: &Descendant<B>,
    notation: &N,
) -> String {
    format!(
        "\\tau_{{{}}}\\!\\left({}\\right)",
        descendant.psi_power,
        notation.basis_tex(&descendant.class)
    )
}

fn render_correlator_text<D, B, C, N: ConstraintNotation<D, B, C>>(
    correlator: &CorrelatorKey<D, B>,
    notation: &N,
) -> String {
    let insertions = if correlator.insertions().is_empty() {
        "1".to_string()
    } else {
        correlator
            .insertions()
            .iter()
            .map(|descendant| render_descendant_text(descendant, notation))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "<{insertions}>_{{g={},beta={}}}",
        correlator.genus,
        notation.degree_text(&correlator.degree)
    )
}

fn render_correlator_tex<D, B, C, N: ConstraintNotation<D, B, C>>(
    correlator: &CorrelatorKey<D, B>,
    notation: &N,
) -> String {
    let insertions = if correlator.insertions().is_empty() {
        "1".to_string()
    } else {
        correlator
            .insertions()
            .iter()
            .map(|descendant| render_descendant_tex(descendant, notation))
            .collect::<Vec<_>>()
            .join(",\\,")
    };
    format!(
        "\\left\\langle {insertions} \\right\\rangle_{{{},\\,{}}}",
        correlator.genus,
        notation.degree_tex(&correlator.degree)
    )
}

fn render_term_text<D, B, C, N: ConstraintNotation<D, B, C>>(
    term: &ConstraintTerm<D, B, C>,
    notation: &N,
) -> String {
    match term {
        ConstraintTerm::Constant { coefficient, .. } => {
            format!("({})", notation.coefficient_text(coefficient))
        }
        ConstraintTerm::Linear(term) => format!(
            "({}) {}",
            notation.coefficient_text(&term.coefficient),
            render_correlator_text(&term.correlator, notation)
        ),
        ConstraintTerm::Quadratic(term) => format!(
            "({}) {} * {}",
            notation.coefficient_text(&term.coefficient),
            render_correlator_text(&term.left, notation),
            render_correlator_text(&term.right, notation)
        ),
    }
}

fn render_term_tex<D, B, C, N: ConstraintNotation<D, B, C>>(
    term: &ConstraintTerm<D, B, C>,
    notation: &N,
) -> String {
    match term {
        ConstraintTerm::Constant { coefficient, .. } => {
            format!("\\left({}\\right)", notation.coefficient_tex(coefficient))
        }
        ConstraintTerm::Linear(term) => format!(
            "\\left({}\\right) {}",
            notation.coefficient_tex(&term.coefficient),
            render_correlator_tex(&term.correlator, notation)
        ),
        ConstraintTerm::Quadratic(term) => format!(
            "\\left({}\\right) {}{}",
            notation.coefficient_tex(&term.coefficient),
            render_correlator_tex(&term.left, notation),
            render_correlator_tex(&term.right, notation)
        ),
    }
}

fn comma_list(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(", ")
    }
}

fn tex_item(out: &mut String, label: &str, value: &str) {
    writeln!(
        out,
        "\\item \\textbf{{{}:}} {}",
        tex_escape(label),
        tex_prose(value)
    )
    .unwrap();
}

/// Escape prose while preserving the small, documented set of mathematical
/// fragments emitted by the convention metadata.
fn tex_prose(value: &str) -> String {
    const MATH_FRAGMENTS: [(&str, &str); 4] = [
        ("Z^{-1} L_k Z", "\\(Z^{-1}L_kZ\\)"),
        (
            "q_k^a = t_k^a - delta_(k,1) delta_(a,unit)",
            "\\(q_k^a=t_k^a-\\delta_{k,1}\\delta_{a,\\mathrm{unit}}\\)",
        ),
        ("1/n!", "\\(1/n!\\)"),
        ("t=0", "\\(t=0\\)"),
    ];
    let mut remaining = value;
    let mut out = String::new();
    while !remaining.is_empty() {
        let next = MATH_FRAGMENTS
            .iter()
            .filter_map(|(plain, tex)| remaining.find(plain).map(|index| (index, *plain, *tex)))
            .min_by_key(|(index, _, _)| *index);
        let Some((index, plain, tex)) = next else {
            out.push_str(&tex_escape(remaining));
            break;
        };
        out.push_str(&tex_escape(&remaining[..index]));
        out.push_str(tex);
        remaining = &remaining[index + plain.len()..];
    }
    out
}

fn tex_escape(value: &str) -> String {
    let mut out = String::new();
    for character in value.chars() {
        match character {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '$' => out.push_str("\\$"),
            '&' => out.push_str("\\&"),
            '#' => out.push_str("\\#"),
            '_' => out.push_str("\\_"),
            '%' => out.push_str("\\%"),
            '^' => out.push_str("\\textasciicircum{}"),
            '~' => out.push_str("\\textasciitilde{}"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use super::*;
    use crate::constraints::virasoro::{
        CohomologicalGrading, ConstraintSector, DilatonShift, FormulaSource, LinearTerm,
        PotentialConvention, QuadraticTerm, StateSpaceConvention, TermOrigin, TheoryLabel,
        TimeNormalization, UnstableConvention, VirasoroConventions, VirasoroOperator,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    enum Basis {
        Unit,
        H,
    }

    impl fmt::Display for Basis {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(match self {
                Self::Unit => "1",
                Self::H => "H",
            })
        }
    }

    fn example_constraint() -> VirasoroConstraint<usize, Basis, &'static str> {
        let unit = Descendant::new(1, Basis::Unit);
        let h = Descendant::new(0, Basis::H);
        let lhs = CorrelatorKey::new(1, 2, vec![Descendant::new(2, Basis::H)]);
        let split_left = CorrelatorKey::new(0, 0, vec![Descendant::new(0, Basis::Unit)]);
        let split_right = CorrelatorKey::new(1, 2, vec![Descendant::new(0, Basis::H)]);
        let conventions = VirasoroConventions {
            potential: PotentialConvention::ConnectedDescendant,
            time_normalization: TimeNormalization::Exponential,
            dilaton_shift: DilatonShift::StandardUnit,
            grading: CohomologicalGrading::Complex,
            unstable: UnstableConvention::Excluded,
            state_space: StateSpaceConvention::EvenOnly,
            novikov_variables: vec!["q".to_string()],
            equivariant_parameters: Vec::new(),
            notes: vec!["formal example; coefficients are test data".to_string()],
        };
        let mut source = FormulaSource::new("AST renderer fixture");
        source.derivation = Some("hand-written test equation".to_string());
        VirasoroConstraint {
            theory: TheoryLabel::new("P^1", "\\mathbb{P}^1"),
            theory_fingerprint: "renderer-fixture-v1".to_string(),
            operator: VirasoroOperator::new(1),
            sector: ConstraintSector::new(1, 2),
            time_coefficient: TimeMonomial::try_from_factors([(unit, 2), (h, 1)]).unwrap(),
            terms: vec![
                ConstraintTerm::Linear(LinearTerm::new("3", lhs, TermOrigin::LinearOperator)),
                ConstraintTerm::Quadratic(QuadraticTerm::new(
                    "1/2",
                    split_right,
                    split_left,
                    TermOrigin::DegreeSplitting,
                )),
            ],
            conventions,
            source,
        }
    }

    #[test]
    fn canonical_keys_ignore_input_order() {
        let a = Descendant::new(0, Basis::H);
        let b = Descendant::new(1, Basis::Unit);
        assert_eq!(
            TimeMonomial::from_descendants([b.clone(), a.clone(), b.clone()]),
            TimeMonomial::try_from_factors([(a.clone(), 1), (b.clone(), 2)]).unwrap()
        );
        assert_eq!(
            CorrelatorKey::new(1, 3, vec![b.clone(), a.clone()]),
            CorrelatorKey::new(1, 3, vec![a, b])
        );
    }

    #[test]
    fn text_render_is_stable_and_explicit_about_conventions() {
        let actual = example_constraint().render_text_with(&DisplayNotation);
        let expected = "Virasoro coefficient constraint\n\
===============================\n\
Theory: P^1\n\
Operator: L_1\n\
Sector: g=1, beta=2\n\
Extracted coefficient: t_0(H) t_1(1)^2\n\
\n\
Formula:\n\
\x20\x200 = (3) <tau_2(H)>_{g=1,beta=2}  [linear operator]\n\
\x20\x20\x20\x20+ (1/2) <tau_0(1)>_{g=0,beta=0} * <tau_0(H)>_{g=1,beta=2}  [degree splitting]\n\
\n\
Conventions:\n\
- Potential: connected descendant potential\n\
- Time normalization: exponential generating series (1/n!)\n\
- Dilaton shift: q_k^a = t_k^a - delta_(k,1) delta_(a,unit)\n\
- Grading: complex cohomological degree\n\
- Unstable terms: unstable correlators excluded\n\
- State space: even state space\n\
- Novikov variables: q\n\
- Equivariant parameters: none\n\
- Note: formal example; coefficients are test data\n\
\n\
Source:\n\
- AST renderer fixture\n\
- Derivation: hand-written test equation\n";
        assert_eq!(actual, expected);
    }

    #[test]
    fn tex_render_contains_auditable_formula_and_metadata() {
        let actual = example_constraint().render_tex_with(&DisplayNotation);
        assert!(actual.contains("\\textbf{Theory:} \\(\\mathbb{P}^1\\)"));
        assert!(actual.contains("t_{0}^{H}\\,\\left(t_{1}^{1}\\right)^{2}"));
        assert!(actual.contains("\\tau_{2}\\!\\left(H\\right)"));
        assert!(actual.contains("\\text{degree splitting}"));
        assert!(actual.contains("connected descendant potential"));
        assert!(actual.contains("formal example; coefficients are test data"));
        assert_eq!(
            tex_prose("coefficient of Z^{-1} L_k Z at t=0"),
            "coefficient of \\(Z^{-1}L_kZ\\) at \\(t=0\\)"
        );
    }
}
