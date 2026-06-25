//! Engine-specific expansions of the universal formula basis elements.
//!
//! The stable-graph skeleton is independent of the target CohFT.  This module
//! records how the same symbols are read when the calibration comes from one
//! of the implemented providers.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormulaExpansion {
    ProjectiveSpace {
        n: usize,
        equivariant: bool,
    },
    NegativeSplitTwisted {
        n: usize,
        degrees: Vec<usize>,
        equivariant: bool,
    },
}

impl FormulaExpansion {
    pub fn render_text(&self) -> String {
        match self {
            Self::ProjectiveSpace { n, equivariant } => render_projective_text(*n, *equivariant),
            Self::NegativeSplitTwisted {
                n,
                degrees,
                equivariant,
            } => render_twisted_text(*n, degrees, *equivariant),
        }
    }

    pub fn render_tex(&self) -> String {
        match self {
            Self::ProjectiveSpace { n, equivariant } => render_projective_tex(*n, *equivariant),
            Self::NegativeSplitTwisted {
                n,
                degrees,
                equivariant,
            } => render_twisted_tex(*n, degrees, *equivariant),
        }
    }
}

fn render_projective_text(n: usize, equivariant: bool) -> String {
    let mut out = String::new();
    out.push_str("Basis expansion: ordinary projective space\n");
    out.push_str("------------------------------------------\n");
    out.push_str(&format!("Target: P^{n}\n"));
    if equivariant {
        out.push_str("Engine: symbolic equivariant projective-space calibration.\n");
    } else {
        out.push_str(
            "Engine: ordinary projective-space Givental backend with the generic lambda-line calibration.\n",
        );
    }
    out.push_str("Smaller calibration primitives:\n");
    out.push_str("- P(x)=prod_a (x-lambda_a)-q, with canonical roots u_i(q).\n");
    out.push_str("- Evaluation matrix E_{i,alpha}=u_i^alpha in the flat basis 1,H,...,H^n.\n");
    out.push_str(
        "- Delta_i=P'(u_i); equivalently Delta_i^{-1} is the canonical idempotent norm.\n",
    );
    out.push_str("- Psi^{-1}=Delta^{-1/2} E in the relative-normalized canonical frame used by the graph engine.\n");
    out.push_str("- S is the small-J descendant calibration, computed by the quantum-minus-classical H recursion.\n");
    out.push_str(
        "- R is solved from the canonical flatness equation using U=diag(u_i), Psi, and Delta.\n",
    );
    out.push_str("- RInv, T, and edge propagators are derived universally from R^{-1}, Delta, and the canonical metric.\n");
    out
}

fn render_twisted_text(n: usize, degrees: &[usize], equivariant: bool) -> String {
    let mut out = String::new();
    out.push_str("Basis expansion: negative split twist\n");
    out.push_str("-------------------------------------\n");
    out.push_str(&format!("Base: P^{n}\n"));
    out.push_str(&format!(
        "Twist: {}\n",
        degrees
            .iter()
            .map(|degree| format!("O(-{degree})"))
            .collect::<Vec<_>>()
            .join(" + ")
    ));
    if equivariant {
        out.push_str(
            "Engine: symbolic equivariant negative-split formulas are not a compute backend yet; this document records the intended calibration primitives.\n",
        );
    } else {
        out.push_str(
            "Engine: negative-split hypergeometric/Birkhoff S plus QRR R backend with the generic rational lambda-line evaluation.\n",
        );
    }
    out.push_str("Smaller calibration primitives:\n");
    out.push_str("- I^tw(q,z): the negative-split hypergeometric I-function with the concave Euler factor.\n");
    out.push_str("- Birkhoff factorization turns I^tw into the descendant S-calibration used on insertions.\n");
    out.push_str("- eta^tw is the twisted flat metric, using the inverse Euler pairing in the local negative-split path.\n");
    out.push_str("- Quantum multiplication by H is extracted from S; its eigenvalues u_i^tw(q) are the canonical roots.\n");
    out.push_str("- Psi, Psi^{-1}, and Delta are obtained by diagonalizing that multiplication operator against eta^tw.\n");
    out.push_str("- R is then produced by the QRR/flatness recursion in the canonical frame.\n");
    out.push_str("- RInv, T, and edge propagators are derived universally from the resulting semisimple calibration.\n");
    out
}

fn render_projective_tex(n: usize, equivariant: bool) -> String {
    let mut out = String::new();
    out.push_str("\\section*{Basis Expansion: Ordinary Projective Space}\n");
    out.push_str(&format!("Target: $\\mathbb{{P}}^{{{n}}}$.\n\n"));
    if equivariant {
        out.push_str("Engine: symbolic equivariant projective-space calibration.\n\n");
    } else {
        out.push_str(
            "Engine: ordinary projective-space Givental backend with the generic $\\lambda$-line calibration.\n\n",
        );
    }
    out.push_str("The universal basis elements in the graph formulas are read through the following smaller calibration data:\n");
    out.push_str("\\begin{align*}\n");
    out.push_str(&format!(
        "P(x)&=\\prod_{{a=0}}^{{{n}}}(x-\\lambda_a)-q, &
P(u_i)&=0,\\\\\n",
    ));
    out.push_str(
        "E_{i\\alpha}&=u_i^{\\alpha}, &
\\Delta_i&=P'(u_i),\\\\\n",
    );
    out.push_str(&format!(
        "(\\Psi^{{-1}})_{{i\\alpha}}&=\\Delta_i^{{-1/2}}E_{{i\\alpha}}, &
U&=\\operatorname{{diag}}(u_0,\\ldots,u_{n}).\n",
    ));
    out.push_str("\\end{align*}\n");
    out.push_str(
        "Here $S(z)$ is the small-$J$ descendant calibration.  In the implementation it is computed by the quantum-minus-classical $H$ recursion\n",
    );
    out.push_str("\\[\n");
    out.push_str(
        "q\\partial_q S_{s+1}=M_H^{\\mathrm{quant}}S_s-S_sM_H^{\\mathrm{cl}},\\qquad S_0=1.\n",
    );
    out.push_str("\\]\n");
    out.push_str(
        "The $R$-matrix is solved from the canonical flatness equation using $U$, $\\Psi$, and $\\Delta$.  The displayed basis elements $R^{-1}$, $(T_p)_i$, and the edge propagator are then universal consequences of this $R$:\n",
    );
    out.push_str("\\[\n");
    out.push_str("T(z)=z(1-R(z)^{-1})\\mathbf 1,\\qquad \\mathbf 1_i=\\Delta_i^{-1/2}.\n");
    out.push_str("\\]\n");
    out
}

fn render_twisted_tex(n: usize, degrees: &[usize], equivariant: bool) -> String {
    let mut out = String::new();
    out.push_str("\\section*{Basis Expansion: Negative Split Twist}\n");
    out.push_str(&format!("Base: $\\mathbb{{P}}^{{{n}}}$.\n\n"));
    let twist = degrees
        .iter()
        .map(|degree| format!("\\mathcal{{O}}(-{degree})"))
        .collect::<Vec<_>>()
        .join("\\oplus ");
    out.push_str(&format!("Twist: ${twist}$.\n\n"));
    if equivariant {
        out.push_str(
            "Engine note: full symbolic equivariant negative-split output is not yet a compute backend; this document records the calibration primitives it would expand to.\n\n",
        );
    } else {
        out.push_str(
            "Engine: negative-split hypergeometric/Birkhoff $S$ plus QRR $R$ backend with the generic rational $\\lambda$-line evaluation.\n\n",
        );
    }
    out.push_str(
        "The universal basis elements are read through the following smaller calibration data:\n",
    );
    out.push_str("\\begin{align*}\n");
    out.push_str(&format!(
        "I^{{\\mathrm{{tw}}}}(q,z)&=\\sum_{{d\\ge0}}q^d I_d^{{\\mathbb P^{n}}}(z)\\,\\mathcal E_d^{{\\mathrm{{conc}}}}, &
S^{{\\mathrm{{tw}}}}(z)&=\\operatorname{{Birkhoff}}(I^{{\\mathrm{{tw}}}}),\\\\\n",
    ));
    out.push_str(&format!(
        "\\eta^{{\\mathrm{{tw}}}}(\\phi_\\alpha,\\phi_\\beta)&=\\int_{{\\mathbb P^{n}}}\\frac{{\\phi_\\alpha\\phi_\\beta}}{{e(E)}}, &
U^{{\\mathrm{{tw}}}}&=\\operatorname{{diag}}(u_0^{{\\mathrm{{tw}}}},\\ldots,u_{n}^{{\\mathrm{{tw}}}}),\\\\\n",
    ));
    out.push_str(
        "H\\star_{\\mathrm{tw}} e_i&=u_i^{\\mathrm{tw}}e_i, &
\\Delta_i^{\\mathrm{tw},-1}&=(e_i,e_i)_{\\eta^{\\mathrm{tw}}}.\n",
    );
    out.push_str("\\end{align*}\n");
    out.push_str(
        "The matrices $\\Psi^{\\mathrm{tw}}$ and $(\\Psi^{\\mathrm{tw}})^{-1}$ are obtained by diagonalizing twisted quantum multiplication against $\\eta^{\\mathrm{tw}}$.  The $R$-matrix is then produced by the QRR/flatness recursion in this canonical frame.  As in the untwisted case,\n",
    );
    out.push_str("\\[\n");
    out.push_str("T^{\\mathrm{tw}}(z)=z(1-R^{\\mathrm{tw}}(z)^{-1})\\mathbf 1^{\\mathrm{tw}},\n");
    out.push_str("\\]\n");
    out.push_str(
        "and edge propagators are universal expressions in $(R^{\\mathrm{tw}})^{-1}$ and $(\\eta^{\\mathrm{tw}})^{-1}$.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projective_expansion_mentions_canonical_roots() {
        let rendered = FormulaExpansion::ProjectiveSpace {
            n: 2,
            equivariant: false,
        }
        .render_tex();
        assert!(rendered.contains("Basis Expansion: Ordinary Projective Space"));
        assert!(rendered.contains("P(x)&=\\prod_{a=0}^{2}(x-\\lambda_a)-q"));
        assert!(rendered.contains("\\Delta_i&=P'(u_i)"));
        assert!(rendered.contains("q\\partial_q S_{s+1}"));
    }

    #[test]
    fn twisted_expansion_mentions_hypergeometric_birkhoff_qrr() {
        let rendered = FormulaExpansion::NegativeSplitTwisted {
            n: 2,
            degrees: vec![3],
            equivariant: false,
        }
        .render_tex();
        assert!(rendered.contains("Basis Expansion: Negative Split Twist"));
        assert!(rendered.contains("\\mathcal{O}(-3)"));
        assert!(rendered.contains("I^{\\mathrm{tw}}(q,z)"));
        assert!(rendered.contains("S^{\\mathrm{tw}}(z)&=\\operatorname{Birkhoff}"));
        assert!(rendered.contains("QRR"));
    }
}
