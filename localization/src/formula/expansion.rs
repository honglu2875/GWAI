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

    pub fn render_raw_text(&self) -> String {
        match self {
            Self::ProjectiveSpace { n, equivariant } => {
                render_projective_raw_text(*n, *equivariant)
            }
            Self::NegativeSplitTwisted {
                n,
                degrees,
                equivariant,
            } => render_twisted_raw_text(*n, degrees, *equivariant),
        }
    }

    pub fn render_raw_tex(&self) -> String {
        match self {
            Self::ProjectiveSpace { n, equivariant } => render_projective_raw_tex(*n, *equivariant),
            Self::NegativeSplitTwisted {
                n,
                degrees,
                equivariant,
            } => render_twisted_raw_tex(*n, degrees, *equivariant),
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
    out.push_str("Smaller calibration data:\n");
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
            "Engine: fiber-equivariant negative-split calibration keeps symbolic fiber parameters mu_i over early-specialized base weights; large graph contractions still need factored rational arithmetic.\n",
        );
    } else {
        out.push_str(
            "Engine: negative-split hypergeometric/Birkhoff S plus QRR R backend with the generic rational lambda-line evaluation.\n",
        );
    }
    out.push_str("Smaller calibration data:\n");
    out.push_str("- I^tw(q,z): the negative-split hypergeometric I-function with the concave Euler factor.\n");
    out.push_str("- Birkhoff factorization turns I^tw into the descendant S-calibration used on insertions.\n");
    out.push_str("- eta^tw is the twisted flat metric, using the inverse Euler pairing in the local negative-split path.\n");
    out.push_str("- Quantum multiplication by H is extracted from S; its eigenvalues u_i^tw(q) are the canonical roots.\n");
    out.push_str("- Psi, Psi^{-1}, and Delta are obtained by diagonalizing that multiplication operator against eta^tw.\n");
    out.push_str("- R is then produced by the QRR/flatness recursion in the canonical frame.\n");
    out.push_str("- RInv, T, and edge propagators are derived universally from the resulting semisimple calibration.\n");
    out
}

fn render_projective_raw_text(n: usize, equivariant: bool) -> String {
    let mut out = String::new();
    out.push_str("Raw basis specialization: ordinary projective space\n");
    out.push_str("---------------------------------------------------\n");
    out.push_str(&format!("Target: P^{n}\n"));
    if equivariant {
        out.push_str("Equivariant parameters lambda_0,...,lambda_n are kept symbolic.\n");
    } else {
        out.push_str("The current compute backend usually evaluates on a generic lambda line before taking the non-equivariant limit; this display keeps the symbolic root-sum form.\n");
    }
    out.push_str("Canonical roots and norms:\n");
    out.push_str("  P(x)=prod_{a=0}^n (x-lambda_a)-q,  P(u_i)=0,  Delta_i=P'(u_i).\n");
    out.push_str("Flat-to-canonical transition:\n");
    out.push_str("  E_{i,a}=u_i^a,  PsiInv_{i,a}=Delta_i^{-1/2} E_{i,a}.\n");
    out.push_str("Packed kernels:\n");
    out.push_str("  L_i^raw(z,psi)=sum_{j,a,b} RInv_ij^raw(psi) Delta_j^(-1/2) u_j^b S^raw(z)_{b,a} gamma_a/(z-psi).\n");
    out.push_str(
        "  E_ij^raw(psi,phi)=(delta_ij-sum_nu RInv_i,nu^raw(psi) RInv_j,nu^raw(phi))/(psi+phi).\n",
    );
    out.push_str("Here S is computed from the small J-function by the quantum-minus-classical H recursion, and RInv/T are solved from the canonical flatness equation.  All q-series are read only to the requested q-degree.\n");
    out
}

fn render_twisted_raw_text(n: usize, degrees: &[usize], equivariant: bool) -> String {
    let mut out = String::new();
    out.push_str("Raw basis specialization: negative split twist\n");
    out.push_str("----------------------------------------------\n");
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
        out.push_str("Fiber-equivariant negative-split calibration keeps one symbolic parameter mu_i for each split summand while base weights are early-specialized.  Large symbolic graph contractions remain a performance frontier for the expanded RatFun engine.\n");
    } else {
        out.push_str("The implemented twisted backend uses a generic rational lambda-line specialization with a non-equivariant limit when available.\n");
    }
    out.push_str("Twisted calibration:\n");
    out.push_str("  I^tw(q,z)=sum_d q^d I_d^{P^n}(z) E_d^{conc};  S^tw=Birkhoff(I^tw).\n");
    out.push_str("  eta^tw(phi_a,phi_b)=int_{P^n} phi_a phi_b / e(E).\n");
    out.push_str("  H *_tw e_i = u_i^tw e_i,  Delta_i^{tw,-1}=(e_i,e_i)_{eta^tw}.\n");
    out.push_str("The graph formula substitutes the resolvent leg and edge kernels inline, with S,R,Psi,Delta,eta replaced by their twisted versions.  All q-series are read only to the requested q-degree.\n");
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
            "Engine note: the fiber-equivariant negative-split calibration keeps symbolic fiber parameters $\\mu_i$ over early-specialized base weights.  Large symbolic graph contractions still need factored rational arithmetic.\n\n",
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

fn render_projective_raw_tex(n: usize, equivariant: bool) -> String {
    let mut out = String::new();
    out.push_str("\\section*{Raw Basis: Ordinary Projective Space}\n");
    out.push_str(&format!("Target: $\\mathbb{{P}}^{{{n}}}$.\n\n"));
    if equivariant {
        out.push_str("The parameters $\\lambda_0,\\ldots,\\lambda_n$ are kept symbolic.\n\n");
    } else {
        out.push_str("The numerical backend may evaluate on a generic $\\lambda$-line before taking a non-equivariant limit; this display keeps the symbolic root-sum form.\n\n");
    }
    out.push_str("\\begin{align*}\n");
    out.push_str(&format!(
        "P(x)&=\\prod_{{a=0}}^{{{n}}}(x-\\lambda_a)-q, & P(u_i)&=0, & \\Delta_i&=P'(u_i),\\\\\n"
    ));
    out.push_str(
        "E_{i\\alpha}&=u_i^\\alpha, & (\\Psi^{-1})_{i\\alpha}&=\\Delta_i^{-1/2}E_{i\\alpha}, & \\eta^{ij}&=\\delta^{ij}.\n",
    );
    out.push_str("\\end{align*}\n");
    out.push_str(
        "The graph formula substitutes the following $q$-truncated root-sum expressions inline:\n",
    );
    out.push_str("\\begin{align*}\n");
    out.push_str(
        "\\mathcal L_i^{\\mathrm{raw},\\gamma}(z,\\psi)&=\\sum_{j,\\alpha,\\beta}(R^{\\mathrm{raw},-1}(\\psi))_{ij}\\Delta_j^{-1/2}u_j^\\beta S^{\\mathrm{raw}}(z)_{\\beta\\alpha}\\frac{\\gamma_\\alpha}{z-\\psi},\\\\\n",
    );
    out.push_str(
        "\\mathcal E_{ij}^{\\mathrm{raw}}(\\psi,\\phi)&=\\frac{\\delta_{ij}-\\sum_\\nu(R^{\\mathrm{raw},-1}(\\psi))_{i\\nu}(R^{\\mathrm{raw},-1}(\\phi))_{j\\nu}}{\\psi+\\phi},\\\\\n",
    );
    out.push_str("\\Theta_{g,n}^{\\mathrm{raw}}(i)&=P'(u_i)^{g-1}\\bigl(P'(u_i)^{1/2}\\bigr)^n.\n");
    out.push_str("\\end{align*}\n");
    out.push_str("Here $S$ comes from the small $J$-function recursion and $R^{-1},T$ from the canonical flatness equation; all series are truncated at the requested $q$-degree.\n");
    out
}

fn render_twisted_raw_tex(n: usize, degrees: &[usize], equivariant: bool) -> String {
    let mut out = String::new();
    out.push_str("\\section*{Raw Basis: Negative Split Twist}\n");
    out.push_str(&format!("Base: $\\mathbb{{P}}^{{{n}}}$.\n\n"));
    let twist = degrees
        .iter()
        .map(|degree| format!("\\mathcal{{O}}(-{degree})"))
        .collect::<Vec<_>>()
        .join("\\oplus ");
    out.push_str(&format!("Twist: ${twist}$.\n\n"));
    if equivariant {
        out.push_str("Fiber-equivariant negative-split calibration keeps one symbolic parameter $\\mu_i$ for each split summand while base weights are early-specialized.  Large symbolic graph contractions remain a performance frontier for the expanded \\texttt{RatFun} engine.\n\n");
    } else {
        out.push_str("The implemented twisted backend uses a generic rational $\\lambda$-line specialization with a non-equivariant limit when available.\n\n");
    }
    out.push_str("\\begin{align*}\n");
    out.push_str(&format!(
        "I^{{\\mathrm{{tw}}}}(q,z)&=\\sum_{{d\\ge0}}q^d I_d^{{\\mathbb P^{n}}}(z)\\mathcal E_d^{{\\mathrm{{conc}}}}, &
S^{{\\mathrm{{tw}}}}&=\\operatorname{{Birkhoff}}(I^{{\\mathrm{{tw}}}}),\\\\\n"
    ));
    out.push_str(&format!(
        "\\eta^{{\\mathrm{{tw}}}}(\\phi_\\alpha,\\phi_\\beta)&=\\int_{{\\mathbb P^{n}}}\\frac{{\\phi_\\alpha\\phi_\\beta}}{{e(E)}}, &
H\\star_{{\\mathrm{{tw}}}}e_i&=u_i^{{\\mathrm{{tw}}}}e_i,\\\\\n"
    ));
    out.push_str("\\Delta_i^{\\mathrm{tw},-1}&=(e_i,e_i)_{\\eta^{\\mathrm{tw}}}.\n");
    out.push_str("\\end{align*}\n");
    out.push_str("The graph formula substitutes the resolvent leg and edge kernels inline, with $S,R,\\Psi,\\Delta,\\eta$ replaced by these twisted calibration data and truncated at the requested $q$-degree.\n");
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
