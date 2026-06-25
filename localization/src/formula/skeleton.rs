//! Finite formula skeletons for fixed stable `(g,m)`.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::algebra::{RatFun, Rational};
use crate::error::GwError;
use crate::graphs::{stable_graphs, StableEdge, StableGraph};
use crate::symbolic::projective_residue_monomial;

use super::basis::basis_glossary;
use super::expansion::FormulaExpansion;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormulaRequest {
    pub genus: usize,
    pub markings: usize,
    pub colors: usize,
    pub max_descendant_power: usize,
    pub q_degree: Option<usize>,
    pub include_glossary: bool,
    pub expansion: Option<FormulaExpansion>,
    pub basis: FormulaBasisMode,
}

impl FormulaRequest {
    pub fn new(genus: usize, markings: usize, colors: usize) -> Self {
        Self {
            genus,
            markings,
            colors,
            max_descendant_power: 0,
            q_degree: None,
            include_glossary: true,
            expansion: None,
            basis: FormulaBasisMode::Raw,
        }
    }

    pub fn graph_dimension(&self) -> usize {
        3 * self.genus + self.markings - 3
    }

    pub fn inverse_r_order(&self) -> usize {
        self.graph_dimension() + 1
    }

    pub fn edge_power_max(&self) -> usize {
        self.graph_dimension()
    }

    pub fn translation_power_max(&self) -> usize {
        self.graph_dimension() + 1
    }

    pub fn descendant_s_order(&self) -> usize {
        self.max_descendant_power
    }

    pub fn z_order(&self) -> usize {
        self.inverse_r_order().max(self.descendant_s_order())
    }

    pub fn validate(&self) -> Result<(), GwError> {
        if self.colors == 0 {
            return Err(GwError::ParseError(
                "formula skeleton needs at least one canonical color".to_string(),
            ));
        }
        if 2 * self.genus + self.markings <= 2 {
            return Err(GwError::UnsupportedInvariant(
                "formula skeleton is implemented for stable (g,m) ranges only".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormulaBasisMode {
    Coefficients,
    Raw,
    Resolvent,
    Rational,
}

impl FormulaBasisMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Coefficients => "coefficients",
            Self::Raw => "raw",
            Self::Resolvent => "resolvent",
            Self::Rational => "rational",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Coefficients => "Coefficient Basis",
            Self::Raw => "Raw Basis",
            Self::Resolvent => "Resolvent Basis",
            Self::Rational => "Rational Basis",
        }
    }

    pub fn tex_superscript(self) -> &'static str {
        match self {
            Self::Coefficients => "\\mathrm{coeff}",
            Self::Raw => "\\mathrm{raw}",
            Self::Resolvent => "\\mathrm{res}",
            Self::Rational => "\\mathrm{rat}",
        }
    }

    pub fn requires_expansion(self) -> bool {
        matches!(self, Self::Raw | Self::Rational)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormulaSkeleton {
    pub request: FormulaRequest,
    pub graphs: Vec<GraphFormulaSkeleton>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFormulaSkeleton {
    pub index: usize,
    pub graph: StableGraph,
    pub automorphism_order: usize,
    pub vertices: Vec<VertexFormulaSlot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexFormulaSlot {
    pub index: usize,
    pub genus: usize,
    pub valence: usize,
    pub psi_dimension_cap: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PowerVariable {
    Leg { marking: usize, vertex: usize },
    EdgeLeft { edge: usize, vertex: usize },
    EdgeRight { edge: usize, vertex: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PowerAssignment {
    leg_powers: Vec<usize>,
    edge_powers: Vec<(usize, usize)>,
    vertex_powers: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RationalPrimaryTerm {
    insertions: Vec<usize>,
    coefficient: RatFun,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TikzPoint {
    x: f64,
    y: f64,
}

pub fn build_formula_skeleton(request: FormulaRequest) -> Result<FormulaSkeleton, GwError> {
    request.validate()?;
    let graphs = stable_graphs(request.genus, request.markings)
        .into_iter()
        .enumerate()
        .map(|(index, graph)| {
            let automorphism_order = graph.automorphism_order();
            let vertices = graph
                .vertices
                .iter()
                .enumerate()
                .map(|(vertex_index, vertex)| {
                    let valence = graph.valence(vertex_index);
                    VertexFormulaSlot {
                        index: vertex_index,
                        genus: vertex.genus,
                        valence,
                        psi_dimension_cap: 3 * vertex.genus + valence - 3,
                    }
                })
                .collect();
            GraphFormulaSkeleton {
                index,
                graph,
                automorphism_order,
                vertices,
            }
        })
        .collect();
    Ok(FormulaSkeleton { request, graphs })
}

impl FormulaSkeleton {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        self.render_header(&mut out);
        self.render_finite_orders(&mut out);
        self.render_formula_convention(&mut out);
        if let Some(expansion) = &self.request.expansion {
            out.push('\n');
            if self.request.basis == FormulaBasisMode::Raw {
                out.push_str(&expansion.render_raw_text());
            } else {
                out.push_str(&expansion.render_text());
            }
        }
        if self.request.include_glossary {
            out.push('\n');
            match self.request.basis {
                FormulaBasisMode::Coefficients => out.push_str(&basis_glossary()),
                FormulaBasisMode::Raw => self.render_raw_glossary(&mut out),
                FormulaBasisMode::Resolvent => self.render_resolvent_glossary(&mut out),
                FormulaBasisMode::Rational => self.render_rational_glossary(&mut out),
            }
        }
        self.render_graphs(&mut out);
        out
    }

    pub fn render_tex(&self) -> String {
        let mut out = String::new();
        self.render_tex_header(&mut out);
        self.render_tex_finite_orders(&mut out);
        self.render_tex_formula_convention(&mut out);
        if let Some(expansion) = &self.request.expansion {
            out.push('\n');
            if self.request.basis == FormulaBasisMode::Raw {
                out.push_str(&expansion.render_raw_tex());
            } else {
                out.push_str(&expansion.render_tex());
            }
        }
        if self.request.include_glossary {
            out.push('\n');
            match self.request.basis {
                FormulaBasisMode::Coefficients => self.render_tex_glossary(&mut out),
                FormulaBasisMode::Raw => self.render_tex_raw_glossary(&mut out),
                FormulaBasisMode::Resolvent => self.render_tex_resolvent_glossary(&mut out),
                FormulaBasisMode::Rational => self.render_tex_rational_glossary(&mut out),
            }
        }
        self.render_tex_graphs(&mut out);
        out
    }

    pub fn render_tex_document(&self) -> String {
        let mut out = String::new();
        out.push_str("\\documentclass[11pt]{article}\n");
        out.push_str("\\usepackage[margin=1in]{geometry}\n");
        out.push_str("\\usepackage[T1]{fontenc}\n");
        out.push_str("\\usepackage{amsmath,amssymb,mathtools}\n");
        out.push_str("\\usepackage{microtype}\n");
        out.push_str("\\usepackage{tikz}\n");
        out.push_str("\\usetikzlibrary{calc,arrows.meta}\n");
        out.push_str("\\allowdisplaybreaks\n");
        out.push_str("\\setlength{\\parindent}{0pt}\n");
        out.push_str("\\setlength{\\parskip}{0.5\\baselineskip}\n");
        out.push_str("\\begin{document}\n\n");
        out.push_str(&self.render_tex());
        out.push_str("\\end{document}\n");
        out
    }

    fn render_header(&self, out: &mut String) {
        out.push_str("Givental graph formula skeleton\n");
        out.push_str("===============================\n");
        out.push_str(&format!(
            "Stable range: genus g={}, markings m={}\n",
            self.request.genus, self.request.markings
        ));
        out.push_str(&format!(
            "Canonical colors: i=0,...,{}\n",
            self.request.colors - 1
        ));
        out.push_str(&format!(
            "Stable-curve dimension D=3g-3+m={}\n",
            self.request.graph_dimension()
        ));
        out.push_str(&format!(
            "Formula basis mode: {}\n",
            self.request.basis.label()
        ));
        match self.request.q_degree {
            Some(degree) => out.push_str(&format!(
                "Calibration q-series should be read modulo q^{}.\n",
                degree + 1
            )),
            None if self.request.expansion.is_some() => out.push_str(
                "No q-degree was fixed; displayed calibration formulas are formal and should be truncated to the needed q-order by the reader.\n",
            ),
            None => out.push_str(
                "No q-degree was fixed here; the displayed formula is a universal graph skeleton.\n",
            ),
        }
    }

    fn render_finite_orders(&self, out: &mut String) {
        out.push('\n');
        out.push_str("Finite truncation orders\n");
        out.push_str("------------------------\n");
        out.push_str(&format!(
            "- S_s is needed for 0 <= s <= K, where K={} is the descendant-power bound.\n",
            self.request.descendant_s_order()
        ));
        out.push_str(&format!(
            "- RInv_r is needed for 0 <= r <= D+1 = {}.\n",
            self.request.inverse_r_order()
        ));
        out.push_str(&format!(
            "- Internal edge factors use endpoint powers 0 <= a,b <= D = {}; the graph sum prunes terms whose total vertex psi degree is too large.\n",
            self.request.edge_power_max()
        ));
        if self.request.translation_power_max() < 2 {
            out.push_str("- T_i^p cannot contribute in this range because D+1 < 2.\n");
        } else {
            out.push_str(&format!(
                "- T_i^p can contribute only for 2 <= p <= D+1 = {}.\n",
                self.request.translation_power_max()
            ));
        }
        out.push_str(&format!(
            "- A single z-truncation z_order >= max(D+1,K) = {} is enough for this skeleton.\n",
            self.request.z_order()
        ));
    }

    fn render_formula_convention(&self, out: &mut String) {
        out.push('\n');
        out.push_str(&format!("{} conventions\n", self.request.basis.title()));
        out.push_str(&"-".repeat(self.request.basis.title().len() + " conventions".len()));
        out.push('\n');
        if matches!(
            self.request.basis,
            FormulaBasisMode::Raw | FormulaBasisMode::Resolvent | FormulaBasisMode::Rational
        ) {
            out.push_str("Descendants are packed as gamma_ell/(z_ell-psi_ell); extracting the coefficient of z_ell^{-k-1} recovers tau_k at marking ell.\n");
            out.push_str("Leg kernel:\n");
            out.push_str("  L_i^ell(z,psi) = sum_{j,a,b} RInv_i,j(psi) * PsiInv[j,b] * S(z)[b,a] * gamma_{ell,a}/(z-psi)\n");
            out.push_str("Edge kernel:\n");
            out.push_str("  E_ij(psi,phi) = (eta^{ij} - sum_nu RInv_i,nu(psi) eta^{nu,nu} RInv_j,nu(phi))/(psi+phi)\n");
            out.push_str("Vertex TFT factor:\n");
            out.push_str("  Theta_{g,n}(i) = Delta_i^{g-1} * (Delta_i^{1/2})^n\n");
            out.push_str("The graph bracket <...>_Gamma^pt is the product of point-theory vertex integrals, including exp(T_i) translation insertions with symmetry factors.\n");
            return;
        }
        out.push_str("How the coefficient basis elements assemble\n");
        out.push_str("-------------------------------------------\n");
        out.push_str("For a formal insertion at marking ell,\n");
        out.push_str("  gamma_ell = sum_{k<=K,a} x_{ell,k,a} tau_k(phi_a),\n");
        out.push_str("each marking factor is expanded directly as finite sums of\n");
        out.push_str("  x_{ell,k,a} * S_s[b,a] * PsiInv[j,b] * RInv_r[i,j]\n");
        out.push_str("with p = k - s + r.  Internal edges are also expanded directly in\n");
        out.push_str(
            "RInv and EtaInv, so the graph terms below use only coefficient calibration symbols.\n\n",
        );
        out.push_str(
            "For a vertex of genus h and color i, the base half-edge/marking powers and\n",
        );
        out.push_str(
            "any number of translation insertions T_i^p are integrated by a point-theory\n",
        );
        out.push_str("psi integral.  The diagonal TFT factor is\n");
        out.push_str("  DeltaInv_i * RelSqrtDelta_i^N       when h=0,\n");
        out.push_str("  Delta_i^{h-1} * RelSqrtDelta_i^N    when h>0,\n");
        out.push_str(
            "where N is the total number of ordinary and translation markings at that vertex.\n",
        );
    }

    fn render_tex_header(&self, out: &mut String) {
        out.push_str("\\section*{Givental Graph Formula Skeleton}\n");
        out.push_str(&format!(
            "Stable range: genus $g={}$ with $m={}$ marking{}.\n",
            self.request.genus,
            self.request.markings,
            if self.request.markings == 1 { "" } else { "s" }
        ));
        out.push_str(&format!(
            "Canonical colors are indexed by $i=0,\\ldots,{}$, and the stable-curve dimension is\n",
            self.request.colors - 1
        ));
        out.push_str(&format!(
            "\\[\nD=3g-3+m={}.\n\\]\n",
            self.request.graph_dimension()
        ));
        match self.request.q_degree {
            Some(degree) => out.push_str(&format!(
                "Calibration series are read modulo $q^{{{}}}$.\n",
                degree + 1
            )),
            None if self.request.expansion.is_some() => out.push_str(
                "No $q$-degree is fixed; the displayed calibration formulas are formal and should be truncated to the needed $q$-order.\n",
            ),
            None => {
                out.push_str("No $q$-degree is fixed here; this is a universal graph skeleton.\n")
            }
        }
    }

    fn render_tex_finite_orders(&self, out: &mut String) {
        out.push_str("\n\\subsection*{Finite Truncation Orders}\n");
        out.push_str("\\begin{align*}\n");
        out.push_str(&format!(
            "0\\le s &\\le K={}, &&\\text{{for }} S_s,\\\\\n",
            self.request.descendant_s_order()
        ));
        out.push_str(&format!(
            "0\\le r &\\le D+1={}, &&\\text{{for }} R_r^{{-1}},\\\\\n",
            self.request.inverse_r_order()
        ));
        out.push_str(&format!(
            "0\\le a,b &\\le D={}, &&\\text{{for edge endpoint powers}},\\\\\n",
            self.request.edge_power_max()
        ));
        if self.request.translation_power_max() < 2 {
            out.push_str("T_p&=0, &&\\text{in this range after truncation},\\\\\n");
        } else {
            out.push_str(&format!(
                "2\\le p &\\le D+1={}, &&\\text{{for translation coefficients }}T_p,\\\\\n",
                self.request.translation_power_max()
            ));
        }
        out.push_str(&format!(
            "z_{{\\max}} &\\ge \\max(D+1,K)={}.\n",
            self.request.z_order()
        ));
        out.push_str("\\end{align*}\n");
    }

    fn render_tex_formula_convention(&self, out: &mut String) {
        out.push_str("\n\\subsection*{Conventions}\n");
        out.push_str(&format!(
            "Basis mode: $\\mathrm{{{}}}$.\n\n",
            self.request.basis.label()
        ));
        if matches!(
            self.request.basis,
            FormulaBasisMode::Raw | FormulaBasisMode::Resolvent | FormulaBasisMode::Rational
        ) {
            self.render_tex_resolvent_convention(out);
            return;
        }
        out.push_str("For formal descendant insertions we write\n");
        out.push_str("\\[\n");
        out.push_str("\\gamma_\\ell=\\sum_{0\\le k\\le K}\\sum_\\alpha x_{\\ell,k,\\alpha}\\,\\tau_k(\\phi_\\alpha).\n");
        out.push_str("\\]\n");
        out.push_str(
            "The leg factors are expanded in the coefficient symbols of $S(z)$, $\\Psi^{-1}$, and $R(z)^{-1}$.\n",
        );
        out.push_str(
            "Internal edges use the standard Givental propagator expanded in $R(z)^{-1}$ and the inverse canonical metric.\n",
        );
        out.push_str(
            "At a vertex of genus $h$ and color $i$, translation insertions $(T_p)_i$ are integrated against point-theory psi classes:\n",
        );
        out.push_str("\\[\n");
        out.push_str("\\langle \\tau_{p_1}\\cdots\\tau_{p_N}\\rangle_h^{\\mathrm{pt}}\n");
        out.push_str("=\n");
        out.push_str("\\int_{\\overline{\\mathcal M}_{h,N}}\\prod_{a=1}^N \\psi_a^{p_a}.\n");
        out.push_str("\\]\n");
    }

    fn render_tex_glossary(&self, out: &mut String) {
        out.push_str("\\subsection*{Coefficient Basis Elements}\n");
        out.push_str("\\begin{itemize}\n");
        out.push_str(
            "\\item $(S_s)_{\\beta\\alpha}$: coefficient of $z^{-s}$ in the descendant-to-ancestor $S$-calibration.\n",
        );
        out.push_str(
            "\\item $(\\Psi^{-1})_{j\\beta}$: transition from flat classes to canonical colors.\n",
        );
        out.push_str("\\item $(R_r^{-1})_{ij}$: coefficient of $z^r$ in $R(z)^{-1}$.\n");
        out.push_str(
            "\\item $(T_p)_i$: translation coefficient in $T(z)=z(1-R(z)^{-1})\\mathbf 1$; here $p\\ge 2$.\n",
        );
        out.push_str(
            "\\item $\\Delta_i$: canonical metric norm; $\\Delta_i^{-1}$ is the genus-zero TFT factor.\n",
        );
        out.push_str("\\item $\\eta^{ii}$: diagonal inverse metric in canonical coordinates.\n");
        out.push_str(
            "\\item $\\langle \\tau_{p_1}\\cdots\\tau_{p_N}\\rangle_h^{\\mathrm{pt}}$: Witten--Kontsevich intersection number on $\\overline{\\mathcal M}_{h,N}$.\n",
        );
        out.push_str("\\end{itemize}\n");
    }

    fn render_resolvent_glossary(&self, out: &mut String) {
        out.push_str("Resolvent basis glossary\n");
        out.push_str("------------------------\n");
        out.push_str("This view keeps descendant insertions packed as resolvents 1/(z_ell-psi).\n");
        out.push_str("Coefficient extraction in z_ell^{-k-1} recovers tau_k at marking ell.\n");
        out.push_str("- L_i^ell(z,psi): packed leg kernel R^{-1}(psi) Psi^{-1} S(z)/(z-psi).\n");
        out.push_str("- E_{ij}(psi,phi): regular edge kernel (eta^{-1}-R^{-1}(psi) eta^{-1} R^{-1}(phi)^T)/(psi+phi).\n");
        out.push_str("- Theta_{g,n}(i): diagonal TFT factor Delta_i^{g-1}(Delta_i^{1/2})^n in the stored relative frame.\n");
        out.push_str("- exp(T_i): shorthand for arbitrary translation markings with the usual 1/m! symmetry factors.\n");
        out.push_str("- <...>_Gamma^pt: product of Witten-Kontsevich point-theory integrals over the vertices of Gamma.\n");
    }

    fn render_raw_glossary(&self, out: &mut String) {
        out.push_str("Raw basis glossary\n");
        out.push_str("------------------\n");
        out.push_str("This view uses the same packed graph expression as the resolvent basis, but the kernels are read in the selected projective or twisted calibration.\n");
        out.push_str("For fixed q-degree, all displayed engine data are q-truncated root-sum expressions in canonical roots, equivariant weights, and z-variables.\n");
        self.render_resolvent_glossary(out);
    }

    fn render_rational_glossary(&self, out: &mut String) {
        out.push_str("Rational basis glossary\n");
        out.push_str("-----------------------\n");
        out.push_str("This view contracts supported raw color/root sums through quotient-ring residue identities.\n");
        out.push_str("Currently implemented: ordinary P^n, genus 0, one vertex, no edges, three primary markings.\n");
        out.push_str("For that graph, sum_{P(u)=0} f(u)/P'(u) is reduced as the H^n coefficient of f(H) modulo prod_a(H-lambda_a)-q.\n");
        out.push_str("Unsupported graphs are printed explicitly as not yet reduced instead of falling back to raw root-sum notation.\n");
    }

    fn render_tex_resolvent_convention(&self, out: &mut String) {
        out.push_str(
            "Descendants are packed by the resolvent insertion $\\gamma_\\ell/(z_\\ell-\\bar\\psi_\\ell)$; coefficient extraction in $z_\\ell^{-k-1}$ recovers $\\tau_k$ at marking $\\ell$.\n",
        );
        out.push_str("\\begin{align*}\n");
        out.push_str(
            "\\mathcal L_i^{\\gamma_\\ell}(z_\\ell,\\psi)&=\\sum_{j,\\alpha,\\beta}(R^{-1}(\\psi))_{ij}(\\Psi^{-1})_{j\\beta}S(z_\\ell)_{\\beta\\alpha}\\frac{\\gamma_{\\ell,\\alpha}}{z_\\ell-\\psi},\\\\\n",
        );
        out.push_str(
            "\\mathcal E_{ij}(\\psi,\\phi)&=\\frac{\\eta^{ij}-\\sum_\\nu(R^{-1}(\\psi))_{i\\nu}\\eta^{\\nu\\nu}(R^{-1}(\\phi))_{j\\nu}}{\\psi+\\phi},\\\\\n",
        );
        out.push_str("\\Theta_{g,n}(i)&=\\Delta_i^{g-1}\\bigl(\\Delta_i^{1/2}\\bigr)^n.\n");
        out.push_str("\\end{align*}\n");
        out.push_str(
            "The graph bracket $\\langle\\cdots\\rangle_\\Gamma^{\\mathrm{pt}}$ means the product of point-theory vertex integrals, including arbitrary translation markings from $\\exp(T_i)$ with the usual symmetry factors.\n",
        );
    }

    fn render_tex_resolvent_glossary(&self, out: &mut String) {
        out.push_str("\\subsection*{Resolvent Basis Elements}\n");
        out.push_str("\\begin{itemize}\n");
        out.push_str("\\item $\\mathcal L_i^{\\gamma_\\ell}(z_\\ell,\\psi)$: packed descendant leg kernel; $[z_\\ell^{-k-1}]$ gives the $\\tau_k$ coefficient.\n");
        out.push_str("\\item $\\mathcal E_{ij}(\\psi,\\phi)$: regular edge propagator before coefficient extraction in the two half-edge psi classes.\n");
        out.push_str("\\item $\\Theta_{g,n}(i)$: diagonal TFT factor in the canonical idempotent color $i$.\n");
        out.push_str("\\item $\\exp(T_i)$: translation insertions produced by $T(z)=z(1-R(z)^{-1})\\mathbf 1$.\n");
        out.push_str("\\item $\\langle\\cdots\\rangle_\\Gamma^{\\mathrm{pt}}$: product of Witten--Kontsevich vertex integrals.\n");
        out.push_str("\\end{itemize}\n");
        out.push_str("In graph contributions these kernel definitions are substituted inline rather than left as opaque $\\mathcal L$ or $\\mathcal E$ factors.\n");
    }

    fn render_tex_raw_glossary(&self, out: &mut String) {
        out.push_str("\\subsection*{Raw Basis Elements}\n");
        out.push_str("This view uses the resolvent graph expression with engine-specialized kernels substituted inline.  For fixed $q$-degree, these kernels are read as truncated root-sum expressions in canonical roots, equivariant weights, and insertion variables.\n");
        self.render_tex_resolvent_glossary(out);
    }

    fn render_tex_rational_glossary(&self, out: &mut String) {
        out.push_str("\\subsection*{Rational Basis Elements}\n");
        out.push_str("This view contracts supported raw color/root sums by quotient-ring residue identities.  Currently implemented: ordinary $\\mathbb P^n$, genus $0$, one vertex, no edges, three primary markings.  There the color sum is reduced by\n");
        out.push_str("\\[\n");
        out.push_str("\\sum_{P(u)=0}\\frac{f(u)}{P'(u)}=[H^n]\\,f(H)\\bmod P(H),\\qquad P(H)=\\prod_{a=0}^{n}(H-\\lambda_a)-q.\n");
        out.push_str("\\]\n");
        out.push_str("Unsupported graphs are reported as not yet reduced.\n");
    }

    fn render_tex_graphs(&self, out: &mut String) {
        out.push_str("\n\\section*{Stable Graphs}\n");
        out.push_str(&format!(
            "There are ${}$ stable graphs.\n\n",
            self.graphs.len()
        ));
        for graph in &self.graphs {
            graph.render_tex(out, &self.request);
            out.push('\n');
        }
    }

    fn render_graphs(&self, out: &mut String) {
        out.push('\n');
        out.push_str("Stable graphs\n");
        out.push_str("-------------\n");
        out.push_str(&format!(
            "Number of stable graphs: {}\n\n",
            self.graphs.len()
        ));
        for graph in &self.graphs {
            graph.render_text(out, &self.request);
            out.push('\n');
        }
    }
}

impl GraphFormulaSkeleton {
    fn render_text(&self, out: &mut String, request: &FormulaRequest) {
        out.push_str(&format!(
            "Graph #{}: |Aut|={}, h1={}, vertices={}, edges={}, markings={}\n",
            self.index,
            self.automorphism_order,
            self.graph.first_betti(),
            self.graph.vertices.len(),
            self.graph.edges.len(),
            self.graph.legs.len()
        ));
        out.push_str("  Vertices:\n");
        for vertex in &self.vertices {
            out.push_str(&format!(
                "    v{}: genus={}, valence={}, vertex psi cap=3h-3+valence={}\n",
                vertex.index, vertex.genus, vertex.valence, vertex.psi_dimension_cap
            ));
        }
        out.push_str("  Edges:\n");
        if self.graph.edges.is_empty() {
            out.push_str("    none\n");
        } else {
            for (edge_index, edge) in self.graph.edges.iter().enumerate() {
                let kind = if edge.is_loop() { "loop" } else { "edge" };
                out.push_str(&format!(
                    "    e{}: {} v{}--v{}\n",
                    edge_index, kind, edge.a, edge.b
                ));
            }
        }
        out.push_str("  Markings:\n");
        if self.graph.legs.is_empty() {
            out.push_str("    none\n");
        } else {
            for (marking, vertex) in self.graph.legs.iter().enumerate() {
                out.push_str(&format!("    ell{} -> v{}\n", marking, vertex));
            }
        }
        out.push_str("  Symbolic contribution shape:\n");
        out.push_str(&format!(
            "    (1/{}) * sum_{{color(v) in 0..{}}} [product of marking, edge, and vertex factors]\n",
            self.automorphism_order,
            request.colors - 1
        ));
        match request.basis {
            FormulaBasisMode::Coefficients => self.render_expanded_expression(out, request),
            FormulaBasisMode::Raw | FormulaBasisMode::Resolvent => {
                self.render_compact_expression(out, request)
            }
            FormulaBasisMode::Rational => self.render_rational_expression(out, request),
        }
    }

    fn render_tex(&self, out: &mut String, request: &FormulaRequest) {
        out.push_str(&format!("\\subsection*{{Graph {}}}\n", self.index));
        out.push_str("\\begin{itemize}\n");
        out.push_str(&format!(
            "\\item $|\\operatorname{{Aut}}\\Gamma|={}$, $h^1(\\Gamma)={}$, $|V|={}$, $|E|={}$, $m={}$.\n",
            self.automorphism_order,
            self.graph.first_betti(),
            self.graph.vertices.len(),
            self.graph.edges.len(),
            self.graph.legs.len()
        ));

        let vertices = self
            .vertices
            .iter()
            .map(|vertex| {
                format!(
                    "$v_{{{}}}: h={}, n={}$",
                    vertex.index, vertex.genus, vertex.valence
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("\\item Vertices: {}.\n", vertices));

        let edges = if self.graph.edges.is_empty() {
            "none".to_string()
        } else {
            self.graph
                .edges
                .iter()
                .enumerate()
                .map(|(edge_index, edge)| {
                    let kind = if edge.is_loop() {
                        "\\mathrm{loop}"
                    } else {
                        "e"
                    };
                    format!(
                        "${kind}_{{{edge_index}}}:v_{{{}}}\\text{{--}}v_{{{}}}$",
                        edge.a, edge.b
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        out.push_str(&format!("\\item Edges: {}.\n", edges));

        let markings = if self.graph.legs.is_empty() {
            "none".to_string()
        } else {
            self.graph
                .legs
                .iter()
                .enumerate()
                .map(|(marking, vertex)| format!("$\\ell_{{{marking}}}\\mapsto v_{{{vertex}}}$"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        out.push_str(&format!("\\item Markings: {}.\n", markings));
        out.push_str("\\end{itemize}\n");
        self.render_tikz(out);
        match request.basis {
            FormulaBasisMode::Coefficients => self.render_expanded_tex_expression(out, request),
            FormulaBasisMode::Raw | FormulaBasisMode::Resolvent => {
                self.render_compact_tex_expression(out, request)
            }
            FormulaBasisMode::Rational => self.render_rational_tex_expression(out, request),
        }
    }

    fn render_tikz(&self, out: &mut String) {
        let positions = tikz_vertex_positions(self.graph.vertices.len());
        out.push_str("\\begin{center}\n");
        out.push_str("\\begin{tikzpicture}[\n");
        out.push_str("  stable vertex/.style={circle,draw,thick,fill=white,inner sep=1.5pt,minimum size=20pt},\n");
        out.push_str("  stable edge/.style={line width=0.5pt},\n");
        out.push_str("  leg/.style={line width=0.45pt},\n");
        out.push_str("  marking/.style={font=\\scriptsize},\n");
        out.push_str("  edge label/.style={midway,fill=white,inner sep=1pt,font=\\scriptsize},\n");
        out.push_str("  every node/.style={font=\\scriptsize}\n");
        out.push_str("]\n");

        for (vertex_index, vertex) in self.graph.vertices.iter().enumerate() {
            let point = positions[vertex_index];
            out.push_str(&format!(
                "\\node[stable vertex] (v{vertex_index}) at ({:.3},{:.3}) {{$\\begin{{smallmatrix}}v_{{{vertex_index}}}\\\\ h={}\\end{{smallmatrix}}$}};\n",
                point.x, point.y, vertex.genus
            ));
        }

        let pair_counts = edge_pair_counts(&self.graph.edges);
        let mut pair_seen = BTreeMap::<(usize, usize), usize>::new();
        let mut loop_seen = BTreeMap::<usize, usize>::new();
        for (edge_index, edge) in self.graph.edges.iter().enumerate() {
            if edge.is_loop() {
                let ordinal = *loop_seen.get(&edge.a).unwrap_or(&0);
                loop_seen.insert(edge.a, ordinal + 1);
                let direction = tikz_loop_direction(edge.a, ordinal, positions.len());
                out.push_str(&format!(
                    "\\draw[stable edge] (v{}) edge[loop {direction}] node[edge label] {{$e_{{{edge_index}}}$}} (v{});\n",
                    edge.a, edge.a
                ));
            } else {
                let key = (edge.a, edge.b);
                let ordinal = *pair_seen.get(&key).unwrap_or(&0);
                pair_seen.insert(key, ordinal + 1);
                let total = pair_counts.get(&key).copied().unwrap_or(1);
                let bend = tikz_bend_option(ordinal, total);
                out.push_str(&format!(
                    "\\draw[stable edge] (v{}) to{bend} node[edge label] {{$e_{{{edge_index}}}$}} (v{});\n",
                    edge.a, edge.b
                ));
            }
        }

        let leg_counts = leg_counts_by_vertex(&self.graph);
        let mut leg_seen = vec![0usize; self.graph.vertices.len()];
        for (marking, &vertex) in self.graph.legs.iter().enumerate() {
            let ordinal = leg_seen[vertex];
            leg_seen[vertex] += 1;
            let total = leg_counts[vertex];
            let angle = tikz_leg_angle(vertex, ordinal, total, &positions);
            let start = positions[vertex];
            let end = start.add_polar(angle, 0.85);
            let anchor = tikz_anchor(angle);
            out.push_str(&format!(
                "\\draw[leg] (v{vertex}) -- ({:.3},{:.3}) node[marking,anchor={anchor}] {{$\\ell_{{{marking}}}$}};\n",
                end.x, end.y
            ));
        }

        out.push_str("\\end{tikzpicture}\n");
        out.push_str("\\end{center}\n");
    }

    fn render_expanded_expression(&self, out: &mut String, request: &FormulaRequest) {
        let terms = self.expanded_terms(request);
        out.push_str("  Expanded contribution in basis coefficients:\n");
        out.push_str(&format!("    C_{} = ", self.index));
        if terms.is_empty() {
            out.push_str("0\n");
        } else {
            out.push_str(&format!(
                "(1/{}) * {} (\n",
                self.automorphism_order,
                self.color_sum_label(request.colors)
            ));
            for (idx, term) in terms.iter().enumerate() {
                let sign = if idx == 0 { "  " } else { "+ " };
                out.push_str("      ");
                out.push_str(sign);
                out.push_str(term);
                out.push('\n');
            }
            out.push_str("    )\n");
        }
    }

    fn render_compact_expression(&self, out: &mut String, request: &FormulaRequest) {
        let label = request.basis.label();
        out.push_str(&format!("  Packed {label} contribution:\n"));
        out.push_str(&format!(
            "    C_{}^{}(z) = (1/{}) * {} <{}>_Gamma^pt\n",
            self.index,
            label,
            self.automorphism_order,
            self.color_sum_label(request.colors),
            self.compact_text_integrand(request)
        ));
        out.push_str(
            "    Coefficients in z_ell^{-k-1} recover individual descendant invariants.\n",
        );
    }

    fn render_rational_expression(&self, out: &mut String, request: &FormulaRequest) {
        out.push_str("  Rational residue-reduced contribution:\n");
        match self.rational_primary_three_point_terms(request) {
            Some(Ok(terms)) => {
                out.push_str(&format!("    C_{}^rational = ", self.index));
                if terms.is_empty() {
                    out.push_str("0\n");
                } else {
                    out.push_str(&rational_primary_terms_text(&terms));
                    out.push('\n');
                }
                out.push_str(
                    "    This uses QH_T(P^n)=Q[lambda,q][H]/(prod_a(H-lambda_a)-q) and residue sum f(H)/P'(H).\n",
                );
            }
            Some(Err(err)) => {
                out.push_str(&format!(
                    "    quotient reduction failed for this graph: {err}\n"
                ));
            }
            None => {
                out.push_str(
                    "    not implemented for this graph. Current rational reduction supports only ordinary P^n, genus 0, one vertex, no edges, three primary markings, and max-descendant 0.\n",
                );
            }
        }
    }

    fn compact_text_integrand(&self, request: &FormulaRequest) -> String {
        let mut factors = Vec::new();
        for vertex in &self.vertices {
            factors.push(vertex_theta_text(vertex, request));
            if request.translation_power_max() >= 2 {
                factors.push(format!("exp(T_i{})", vertex.index));
            }
        }
        for (marking, &vertex) in self.graph.legs.iter().enumerate() {
            factors.push(resolvent_leg_text(marking, vertex, request));
        }
        for (edge_index, edge) in self.graph.edges.iter().enumerate() {
            factors.push(resolvent_edge_text(edge_index, edge, request));
        }
        if factors.is_empty() {
            "1".to_string()
        } else {
            factors.join(" * ")
        }
    }

    fn render_expanded_tex_expression(&self, out: &mut String, request: &FormulaRequest) {
        let terms = self
            .expanded_tex_factor_terms(request)
            .into_iter()
            .map(|factors| {
                let nontrivial = factors
                    .into_iter()
                    .filter(|factor| factor != "1")
                    .collect::<Vec<_>>();
                if nontrivial.is_empty() {
                    vec!["1".to_string()]
                } else {
                    nontrivial
                }
            })
            .collect::<Vec<_>>();

        if terms.is_empty() {
            out.push_str(&format!("\\[\nC_{{{}}}=0\n\\]\n", self.index));
            return;
        }

        let parenthesize = terms.len() > 1;
        // The `={}&` sets the alignment column for the page-breakable `align*`.
        let head = format!(
            "C_{{{}}}={{}}&{}{}{}",
            self.index,
            tex_prefactor(self.automorphism_order),
            self.color_sum_tex(request.colors),
            if parenthesize { "\\bigl(" } else { "" }
        );
        let tail = if parenthesize { "\\bigr)" } else { "" };

        let mut items = Vec::new();
        for (term_index, factors) in terms.iter().enumerate() {
            for (factor_index, factor) in factors.iter().enumerate() {
                let connector = if term_index == 0 && factor_index == 0 {
                    String::new()
                } else if factor_index == 0 {
                    "+".to_string()
                } else {
                    "\\mathbin{\\cdot}".to_string()
                };
                items.push((connector, factor.clone()));
            }
        }
        out.push_str(&tex_aligned_display(&head, &items, tail, TEX_LINE_BUDGET));
    }

    fn render_compact_tex_expression(&self, out: &mut String, request: &FormulaRequest) {
        let factors = self.compact_tex_factors(request);
        let head = format!(
            "C_{{{}}}^{{{}}}(\\mathbf z)={}{}\\bigl\\langle ",
            self.index,
            request.basis.tex_superscript(),
            tex_prefactor(self.automorphism_order),
            self.color_sum_tex(request.colors),
        );
        let tail = "\\bigr\\rangle_\\Gamma^{\\mathrm{pt}}";

        if factors.is_empty() {
            out.push_str(&format!("\\[\n{head}1{tail}\n\\]\n"));
            return;
        }

        let items = factors
            .into_iter()
            .enumerate()
            .map(|(index, factor)| {
                let connector = if index == 0 {
                    String::new()
                } else {
                    "\\mathbin{\\cdot}".to_string()
                };
                (connector, factor)
            })
            .collect::<Vec<_>>();
        out.push_str(&tex_multlined_display(&head, &items, tail, TEX_LINE_BUDGET));
    }

    fn render_rational_tex_expression(&self, out: &mut String, request: &FormulaRequest) {
        match self.rational_primary_three_point_terms(request) {
            Some(Ok(terms)) => {
                let head = format!("C_{{{}}}^{{\\mathrm{{rat}}}}={{}}&", self.index);
                let items = rational_primary_terms_tex_items(&terms);
                if items.is_empty() {
                    out.push_str(&format!(
                        "\\[\nC_{{{}}}^{{\\mathrm{{rat}}}}=0\n\\]\n",
                        self.index
                    ));
                } else {
                    out.push_str(&tex_aligned_display_preserving_items(
                        &head,
                        &items,
                        "",
                        RATIONAL_TEX_LINE_BUDGET,
                    ));
                }
                out.push_str("\\[\n");
                out.push_str("\\sum_{P(u)=0}\\frac{f(u)}{P'(u)}=[H^n]\\,f(H)\\bmod P(H),\\qquad P(H)=\\prod_{a=0}^{n}(H-\\lambda_a)-q.\n");
                out.push_str("\\]\n");
            }
            Some(Err(err)) => {
                out.push_str("\\[\n");
                out.push_str(&format!(
                    "\\text{{Quotient reduction failed for this graph: {}}}\n",
                    escape_tex_text(&err.to_string())
                ));
                out.push_str("\\]\n");
            }
            None => {
                out.push_str("\\[\n");
                out.push_str("\\text{Rational reduction is not implemented for this graph.}\n");
                out.push_str("\\]\n");
            }
        }
    }

    fn compact_tex_factors(&self, request: &FormulaRequest) -> Vec<String> {
        let mut factors = Vec::new();
        for vertex in &self.vertices {
            factors.push(vertex_theta_tex(vertex, request));
            if request.translation_power_max() >= 2 {
                factors.push(format!("\\exp(T_{{i_{{{}}}}})", vertex.index));
            }
        }
        for (marking, &vertex) in self.graph.legs.iter().enumerate() {
            factors.push(resolvent_leg_tex(marking, vertex, request));
        }
        for (edge_index, edge) in self.graph.edges.iter().enumerate() {
            factors.push(resolvent_edge_tex(edge_index, edge, request));
        }
        factors
    }

    fn rational_primary_three_point_terms(
        &self,
        request: &FormulaRequest,
    ) -> Option<Result<Vec<RationalPrimaryTerm>, GwError>> {
        let n = rational_projective_n(request)?;
        if request.genus != 0
            || request.markings != 3
            || request.colors != n + 1
            || request.max_descendant_power != 0
            || self.graph.vertices.len() != 1
            || self.graph.edges.len() != 0
            || self.graph.legs.len() != 3
            || self.vertices.len() != 1
            || self.vertices[0].genus != 0
        {
            return None;
        }

        let mut terms = Vec::new();
        for alpha0 in 0..=n {
            for alpha1 in 0..=n {
                for alpha2 in 0..=n {
                    let insertions = vec![alpha0, alpha1, alpha2];
                    let coefficient = match projective_residue_monomial(n, alpha0 + alpha1 + alpha2)
                    {
                        Ok(value) => value,
                        Err(err) => return Some(Err(err)),
                    };
                    if !coefficient.is_zero() {
                        terms.push(RationalPrimaryTerm {
                            insertions,
                            coefficient,
                        });
                    }
                }
            }
        }
        Some(Ok(terms))
    }

    fn color_sum_label(&self, colors: usize) -> String {
        if self.graph.vertices.len() == 1 {
            format!("sum_{{i0=0..{}}}", colors - 1)
        } else {
            let variables = (0..self.graph.vertices.len())
                .map(|vertex| format!("i{vertex}=0..{}", colors - 1))
                .collect::<Vec<_>>()
                .join(", ");
            format!("sum_{{{variables}}}")
        }
    }

    fn color_sum_tex(&self, colors: usize) -> String {
        (0..self.graph.vertices.len())
            .map(|vertex| format!("\\sum_{{i_{{{vertex}}}=0}}^{{{}}}", colors - 1))
            .collect::<Vec<_>>()
            .join("")
    }

    fn expanded_terms(&self, request: &FormulaRequest) -> Vec<String> {
        let assignments = self.power_assignments(request);
        let mut terms = Vec::new();
        for assignment in assignments {
            let mut fixed_factors = Vec::new();
            for (marking, &power) in assignment.leg_powers.iter().enumerate() {
                let vertex = self.graph.legs[marking];
                fixed_factors.push(leg_factor(marking, &format!("i{vertex}"), power, request));
            }
            for (edge_index, &(left_power, right_power)) in
                assignment.edge_powers.iter().enumerate()
            {
                let edge = &self.graph.edges[edge_index];
                fixed_factors.push(edge_factor(
                    &format!("i{}", edge.a),
                    &format!("i{}", edge.b),
                    left_power,
                    right_power,
                    request.colors,
                ));
            }

            let vertex_terms = self
                .vertices
                .iter()
                .map(|vertex| {
                    vertex_expanded_terms(
                        vertex.genus,
                        &format!("i{}", vertex.index),
                        &assignment.vertex_powers[vertex.index],
                    )
                })
                .collect::<Vec<_>>();
            append_distributed_terms(
                &fixed_factors,
                &vertex_terms,
                0,
                &mut Vec::new(),
                &mut terms,
            );
        }
        terms
    }

    fn expanded_tex_factor_terms(&self, request: &FormulaRequest) -> Vec<Vec<String>> {
        let assignments = self.power_assignments(request);
        let mut terms = Vec::new();
        for assignment in assignments {
            let mut fixed_factors = Vec::new();
            for (marking, &power) in assignment.leg_powers.iter().enumerate() {
                let vertex = self.graph.legs[marking];
                fixed_factors.push(leg_factor_tex(
                    marking,
                    &format!("i_{{{vertex}}}"),
                    power,
                    request,
                ));
            }
            for (edge_index, &(left_power, right_power)) in
                assignment.edge_powers.iter().enumerate()
            {
                let edge = &self.graph.edges[edge_index];
                fixed_factors.push(edge_factor_tex(
                    &format!("i_{{{}}}", edge.a),
                    &format!("i_{{{}}}", edge.b),
                    left_power,
                    right_power,
                    request.colors,
                ));
            }

            let vertex_terms = self
                .vertices
                .iter()
                .map(|vertex| {
                    vertex_expanded_factor_terms_tex(
                        vertex.genus,
                        &format!("i_{{{}}}", vertex.index),
                        &assignment.vertex_powers[vertex.index],
                    )
                })
                .collect::<Vec<_>>();
            append_distributed_tex_factor_terms(
                &fixed_factors,
                &vertex_terms,
                0,
                &mut Vec::new(),
                &mut terms,
            );
        }
        terms
    }

    fn power_assignments(&self, request: &FormulaRequest) -> Vec<PowerAssignment> {
        let variables = self.power_variables();
        let mut assignment = PowerAssignment {
            leg_powers: vec![0; self.graph.legs.len()],
            edge_powers: vec![(0, 0); self.graph.edges.len()],
            vertex_powers: vec![Vec::new(); self.graph.vertices.len()],
        };
        let mut vertex_sums = vec![0usize; self.graph.vertices.len()];
        let mut out = Vec::new();
        self.collect_power_assignments(
            request,
            &variables,
            0,
            0,
            &mut vertex_sums,
            &mut assignment,
            &mut out,
        );
        out
    }

    fn power_variables(&self) -> Vec<PowerVariable> {
        let mut variables = Vec::new();
        for (marking, &leg_vertex) in self.graph.legs.iter().enumerate() {
            variables.push(PowerVariable::Leg {
                marking,
                vertex: leg_vertex,
            });
        }
        for (edge_index, edge) in self.graph.edges.iter().enumerate() {
            variables.push(PowerVariable::EdgeLeft {
                edge: edge_index,
                vertex: edge.a,
            });
            variables.push(PowerVariable::EdgeRight {
                edge: edge_index,
                vertex: edge.b,
            });
        }
        variables
    }

    fn collect_power_assignments(
        &self,
        request: &FormulaRequest,
        variables: &[PowerVariable],
        variable_index: usize,
        total_power: usize,
        vertex_sums: &mut [usize],
        assignment: &mut PowerAssignment,
        out: &mut Vec<PowerAssignment>,
    ) {
        if variable_index == variables.len() {
            out.push(assignment.clone());
            return;
        }

        let variable = variables[variable_index];
        let vertex = variable.vertex();
        let vertex_cap = self.vertices[vertex].psi_dimension_cap;
        let remaining_vertex = vertex_cap - vertex_sums[vertex];
        let remaining_total = request.graph_dimension() - total_power;
        let max_power = remaining_vertex.min(remaining_total);
        for power in 0..=max_power {
            variable.write_power(power, assignment);
            vertex_sums[vertex] += power;
            assignment.vertex_powers[vertex].push(power);
            self.collect_power_assignments(
                request,
                variables,
                variable_index + 1,
                total_power + power,
                vertex_sums,
                assignment,
                out,
            );
            assignment.vertex_powers[vertex].pop();
            vertex_sums[vertex] -= power;
        }
    }
}

impl PowerVariable {
    fn vertex(self) -> usize {
        match self {
            PowerVariable::Leg { vertex, .. }
            | PowerVariable::EdgeLeft { vertex, .. }
            | PowerVariable::EdgeRight { vertex, .. } => vertex,
        }
    }

    fn write_power(self, power: usize, assignment: &mut PowerAssignment) {
        match self {
            PowerVariable::Leg { marking, .. } => assignment.leg_powers[marking] = power,
            PowerVariable::EdgeLeft { edge, .. } => assignment.edge_powers[edge].0 = power,
            PowerVariable::EdgeRight { edge, .. } => assignment.edge_powers[edge].1 = power,
        }
    }
}

fn append_distributed_terms(
    fixed_factors: &[String],
    vertex_terms: &[Vec<String>],
    vertex_index: usize,
    current_vertex_factors: &mut Vec<String>,
    out: &mut Vec<String>,
) {
    if vertex_index == vertex_terms.len() {
        let mut factors = fixed_factors.to_vec();
        factors.extend(current_vertex_factors.iter().cloned());
        out.push(join_factors(&factors));
        return;
    }

    for term in &vertex_terms[vertex_index] {
        current_vertex_factors.push(term.clone());
        append_distributed_terms(
            fixed_factors,
            vertex_terms,
            vertex_index + 1,
            current_vertex_factors,
            out,
        );
        current_vertex_factors.pop();
    }
}

fn append_distributed_tex_factor_terms(
    fixed_factors: &[String],
    vertex_terms: &[Vec<Vec<String>>],
    vertex_index: usize,
    current_vertex_factors: &mut Vec<String>,
    out: &mut Vec<Vec<String>>,
) {
    if vertex_index == vertex_terms.len() {
        let mut factors = fixed_factors.to_vec();
        factors.extend(current_vertex_factors.iter().cloned());
        out.push(factors);
        return;
    }

    for term in &vertex_terms[vertex_index] {
        let previous_len = current_vertex_factors.len();
        current_vertex_factors.extend(term.iter().cloned());
        append_distributed_tex_factor_terms(
            fixed_factors,
            vertex_terms,
            vertex_index + 1,
            current_vertex_factors,
            out,
        );
        current_vertex_factors.truncate(previous_len);
    }
}

fn vertex_theta_text(vertex: &VertexFormulaSlot, request: &FormulaRequest) -> String {
    let delta = delta_text(&format!("i{}", vertex.index), request);
    let mut factors = Vec::new();
    if vertex.genus == 0 {
        factors.push(format!("{delta}^-1"));
    } else if vertex.genus > 1 {
        factors.push(format!("{delta}^{}", vertex.genus - 1));
    }
    factors.push(format!("({delta}^(1/2))^{}", vertex.valence));
    join_factors(&factors)
}

fn resolvent_leg_text(marking: usize, vertex: usize, request: &FormulaRequest) -> String {
    let max = request.colors - 1;
    let psi = format!("barpsi_ell{marking}");
    let r_inv = r_inv_series_text(&psi, request);
    let s = s_series_text(&format!("z{marking}"), request);
    match raw_projective_n(request) {
        Some(_) => format!(
            "sum_{{j,alpha,beta=0..{max}}} {r_inv}[i{vertex},j] * Delta_j^(-1/2) * u_j^beta * {s}[beta,alpha] * gamma_{{{marking},alpha}}/(z{marking}-{psi})"
        ),
        _ => {
            let psi_inv = psi_inverse_text(request);
            format!(
                "sum_{{j,alpha,beta=0..{max}}} {r_inv}[i{vertex},j] * {psi_inv}[j,beta] * {s}[beta,alpha] * gamma_{{{marking},alpha}}/(z{marking}-{psi})"
            )
        }
    }
}

fn resolvent_edge_text(edge_index: usize, edge: &StableEdge, request: &FormulaRequest) -> String {
    let max = request.colors - 1;
    let left = format!("barpsi_e{edge_index}_plus");
    let right = format!("barpsi_e{edge_index}_minus");
    let left_r = r_inv_series_text(&left, request);
    let right_r = r_inv_series_text(&right, request);
    let denominator = format!("({left}+{right})");
    match raw_projective_n(request) {
        Some(_) => format!(
            "(delta_{{i{},i{}}} - sum_{{nu=0..{max}}} {left_r}[i{},nu] * {right_r}[i{},nu]) / {denominator}",
            edge.a, edge.b, edge.a, edge.b
        ),
        _ => {
            let eta = eta_inverse_text(request);
            format!(
                "({eta}[i{},i{}] - sum_{{nu=0..{max}}} {left_r}[i{},nu] * {eta}[nu,nu] * {right_r}[i{},nu]) / {denominator}",
                edge.a, edge.b, edge.a, edge.b
            )
        }
    }
}

fn vertex_theta_tex(vertex: &VertexFormulaSlot, request: &FormulaRequest) -> String {
    let delta = delta_tex(&format!("i_{{{}}}", vertex.index), request);
    let mut factors = Vec::new();
    if vertex.genus == 0 {
        factors.push(format!("\\bigl({delta}\\bigr)^{{-1}}"));
    } else if vertex.genus > 1 {
        factors.push(powered_factor_tex(&delta, vertex.genus - 1));
    }
    let sqrt_delta = format!("\\bigl({delta}\\bigr)^{{1/2}}");
    factors.push(powered_factor_tex(&sqrt_delta, vertex.valence));
    join_tex_product(&factors)
}

fn resolvent_leg_tex(marking: usize, vertex: usize, request: &FormulaRequest) -> String {
    let max = request.colors - 1;
    let psi = format!("\\bar\\psi_{{\\ell_{{{marking}}}}}");
    let z = format!("z_{{{marking}}}");
    let r_inv = r_inv_series_tex(&psi, request);
    let s = s_series_tex(&z, request);
    match raw_projective_n(request) {
        Some(_) => format!(
            "\\sum_{{j,\\alpha,\\beta=0}}^{{{max}}}{r_inv}_{{i_{{{vertex}}},j}}\\mathbin{{\\cdot}}\\Delta_j^{{-1/2}}u_j^\\beta\\mathbin{{\\cdot}}{s}_{{\\beta,\\alpha}}\\mathbin{{\\cdot}}\\frac{{\\gamma_{{{marking},\\alpha}}}}{{{z}-{psi}}}"
        ),
        _ => {
            let psi_inv = psi_inverse_tex(request);
            format!(
                "\\sum_{{j,\\alpha,\\beta=0}}^{{{max}}}{r_inv}_{{i_{{{vertex}}},j}}\\mathbin{{\\cdot}}{psi_inv}_{{j,\\beta}}\\mathbin{{\\cdot}}{s}_{{\\beta,\\alpha}}\\mathbin{{\\cdot}}\\frac{{\\gamma_{{{marking},\\alpha}}}}{{{z}-{psi}}}"
            )
        }
    }
}

fn resolvent_edge_tex(edge_index: usize, edge: &StableEdge, request: &FormulaRequest) -> String {
    let max = request.colors - 1;
    let left = format!("\\bar\\psi_{{e_{{{edge_index}}},+}}");
    let right = format!("\\bar\\psi_{{e_{{{edge_index}}},-}}");
    let left_r = r_inv_series_tex(&left, request);
    let right_r = r_inv_series_tex(&right, request);
    let denominator = format!("\\bigl({left}+{right}\\bigr)");
    match raw_projective_n(request) {
        Some(_) => format!(
            "\\bigl(\\delta_{{i_{{{}}},i_{{{}}}}} - \\sum_{{\\nu=0}}^{{{max}}}{left_r}_{{i_{{{}}},\\nu}}\\mathbin{{\\cdot}}{right_r}_{{i_{{{}}},\\nu}}\\bigr)\\mathbin{{/}}{denominator}",
            edge.a, edge.b, edge.a, edge.b
        ),
        _ => {
            let eta = eta_inverse_tex(request);
            format!(
                "\\bigl({eta}^{{i_{{{}}}i_{{{}}}}} - \\sum_{{\\nu=0}}^{{{max}}}{left_r}_{{i_{{{}}},\\nu}}\\mathbin{{\\cdot}}{eta}^{{\\nu\\nu}}\\mathbin{{\\cdot}}{right_r}_{{i_{{{}}},\\nu}}\\bigr)\\mathbin{{/}}{denominator}",
                edge.a, edge.b, edge.a, edge.b
            )
        }
    }
}

fn raw_projective_n(request: &FormulaRequest) -> Option<usize> {
    match &request.expansion {
        Some(FormulaExpansion::ProjectiveSpace { n, .. })
            if request.basis == FormulaBasisMode::Raw =>
        {
            Some(*n)
        }
        _ => None,
    }
}

fn rational_projective_n(request: &FormulaRequest) -> Option<usize> {
    match &request.expansion {
        Some(FormulaExpansion::ProjectiveSpace { n, .. })
            if request.basis == FormulaBasisMode::Rational =>
        {
            Some(*n)
        }
        _ => None,
    }
}

fn raw_twisted(request: &FormulaRequest) -> bool {
    matches!(
        (&request.expansion, request.basis),
        (
            Some(FormulaExpansion::NegativeSplitTwisted { .. }),
            FormulaBasisMode::Raw
        )
    )
}

fn delta_text(color: &str, request: &FormulaRequest) -> String {
    if raw_projective_n(request).is_some() {
        format!("P'({color})")
    } else if raw_twisted(request) {
        format!("Delta_tw_{color}")
    } else {
        format!("Delta_{color}")
    }
}

fn delta_tex(color: &str, request: &FormulaRequest) -> String {
    if raw_projective_n(request).is_some() {
        format!("P'(u_{{{color}}})")
    } else if raw_twisted(request) {
        format!("\\Delta_{{{color}}}^{{\\mathrm{{tw}}}}")
    } else {
        format!("\\Delta_{{{color}}}")
    }
}

fn r_inv_series_text(argument: &str, request: &FormulaRequest) -> String {
    if raw_projective_n(request).is_some() {
        format!("RInv_raw({argument})")
    } else if raw_twisted(request) {
        format!("RInv_tw({argument})")
    } else {
        format!("RInv({argument})")
    }
}

fn r_inv_series_tex(argument: &str, request: &FormulaRequest) -> String {
    if raw_projective_n(request).is_some() {
        format!("(R^{{\\mathrm{{raw}},-1}}({argument}))")
    } else if raw_twisted(request) {
        format!("(R^{{\\mathrm{{tw}},-1}}({argument}))")
    } else {
        format!("(R^{{-1}}({argument}))")
    }
}

fn s_series_text(argument: &str, request: &FormulaRequest) -> String {
    if raw_projective_n(request).is_some() {
        format!("S_raw({argument})")
    } else if raw_twisted(request) {
        format!("S_tw({argument})")
    } else {
        format!("S({argument})")
    }
}

fn s_series_tex(argument: &str, request: &FormulaRequest) -> String {
    if raw_projective_n(request).is_some() {
        format!("S^{{\\mathrm{{raw}}}}({argument})")
    } else if raw_twisted(request) {
        format!("S^{{\\mathrm{{tw}}}}({argument})")
    } else {
        format!("S({argument})")
    }
}

fn psi_inverse_text(request: &FormulaRequest) -> &'static str {
    if raw_twisted(request) {
        "PsiInv_tw"
    } else {
        "PsiInv"
    }
}

fn psi_inverse_tex(request: &FormulaRequest) -> &'static str {
    if raw_twisted(request) {
        "(\\Psi^{\\mathrm{tw},-1})"
    } else {
        "(\\Psi^{-1})"
    }
}

fn eta_inverse_text(request: &FormulaRequest) -> &'static str {
    if raw_twisted(request) {
        "EtaInv_tw"
    } else {
        "EtaInv"
    }
}

fn eta_inverse_tex(request: &FormulaRequest) -> &'static str {
    if raw_twisted(request) {
        "(\\eta^{\\mathrm{tw}})"
    } else {
        "\\eta"
    }
}

fn rational_primary_terms_text(terms: &[RationalPrimaryTerm]) -> String {
    terms
        .iter()
        .map(|term| {
            let gamma = term
                .insertions
                .iter()
                .enumerate()
                .map(|(marking, power)| format!("gamma_{{{marking},{power}}}"))
                .collect::<Vec<_>>()
                .join(" * ");
            if term.coefficient.is_one() {
                gamma
            } else {
                format!("({}) * {gamma}", term.coefficient)
            }
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

fn rational_primary_terms_tex_items(terms: &[RationalPrimaryTerm]) -> Vec<(String, String)> {
    let mut items = Vec::new();
    for term in terms {
        append_rational_primary_term_tex_items(term, &mut items);
    }
    items
}

fn append_rational_primary_term_tex_items(
    term: &RationalPrimaryTerm,
    items: &mut Vec<(String, String)>,
) {
    let gamma = term
        .insertions
        .iter()
        .enumerate()
        .map(|(marking, power)| format!("\\gamma_{{{marking},{power}}}"))
        .collect::<Vec<_>>()
        .join("\\mathbin{\\cdot}");
    if term.coefficient.is_one() {
        let connector = if items.is_empty() { "" } else { "+" };
        items.push((connector.to_string(), gamma));
    } else if term.coefficient.as_rational() == Some(-Rational::one()) {
        items.push(("-".to_string(), gamma));
    } else if term.coefficient.den.is_one() {
        for (sign, piece) in split_plain_signed_sum(&term.coefficient.num.to_string()) {
            let connector = rational_tex_connector(items.is_empty(), &sign);
            let factor = if piece == "1" {
                gamma.clone()
            } else {
                format!("{}\\mathbin{{\\cdot}}{gamma}", algebra_fragment_tex(&piece))
            };
            items.push((connector, factor));
        }
    } else {
        let connector = if items.is_empty() { "" } else { "+" };
        let factor = format!(
            "\\bigl({}\\bigr)\\mathbin{{\\cdot}}{gamma}",
            algebra_fragment_tex(&term.coefficient.to_string())
        );
        items.push((connector.to_string(), factor));
    }
}

fn rational_tex_connector(first: bool, sign: &str) -> String {
    match (first, sign) {
        (true, "+") | (true, "") => String::new(),
        (_, "-") => "-".to_string(),
        _ => "+".to_string(),
    }
}

fn split_plain_signed_sum(value: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut current = value.trim();
    let first_sign = if let Some(stripped) = current.strip_prefix('-') {
        current = stripped.trim_start();
        "-"
    } else {
        ""
    };
    let mut sign = first_sign.to_string();
    while let Some((split, next_sign)) = next_plain_sum_split(current) {
        let piece = current[..split].trim();
        if !piece.is_empty() {
            out.push((std::mem::take(&mut sign), piece.to_string()));
        }
        sign = next_sign.to_string();
        current = current[split + 3..].trim();
    }
    if !current.is_empty() {
        out.push((sign, current.to_string()));
    }
    out
}

fn next_plain_sum_split(value: &str) -> Option<(usize, &'static str)> {
    let plus = value.find(" + ").map(|index| (index, "+"));
    let minus = value.find(" - ").map(|index| (index, "-"));
    match (plus, minus) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(found), None) | (None, Some(found)) => Some(found),
        (None, None) => None,
    }
}

fn algebra_fragment_tex(value: &str) -> String {
    value.replace("lambda_", "\\lambda_").replace('*', "\\,")
}

fn escape_tex_text(value: &str) -> String {
    value
        .replace('\\', "\\textbackslash{}")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('_', "\\_")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('$', "\\$")
        .replace('#', "\\#")
}

fn join_tex_product(factors: &[String]) -> String {
    let nontrivial = factors
        .iter()
        .filter(|factor| factor.as_str() != "1")
        .cloned()
        .collect::<Vec<_>>();
    if nontrivial.is_empty() {
        "1".to_string()
    } else {
        nontrivial.join("\\mathbin{\\cdot}")
    }
}

/// Target visual width (in rough character units) for a single displayed line.
/// Lines are wrapped conservatively so the largest graph contributions stay
/// inside the text block instead of running off the right margin.
const TEX_LINE_BUDGET: usize = 52;
const RATIONAL_TEX_LINE_BUDGET: usize = 36;

/// A trivial automorphism factor reads as a bare `1`; suppress it so the display
/// shows `\sum\langle\cdots\rangle` rather than the noisy `\frac{1}{1}`.
fn tex_prefactor(order: usize) -> String {
    if order <= 1 {
        String::new()
    } else {
        format!("\\frac{{1}}{{{order}}}")
    }
}

/// Greedily break a `head … tail` display into lines no wider than `budget`.
/// Each item pairs a factor with the connector (`\cdot`, `+`, …) that precedes
/// it; when a break falls on a connector it leads the continuation line so the
/// operator stays visible.  Line 0 already carries `head`; `tail` is appended to
/// the final line.
fn wrap_display_lines(
    head: &str,
    items: &[(String, String)],
    tail: &str,
    budget: usize,
) -> Vec<String> {
    let items = expand_wide_items(items, budget);
    let mut lines: Vec<String> = Vec::new();
    let mut current = head.to_string();
    let mut current_width = tex_visual_width(head);

    for (index, (connector, factor)) in items.iter().enumerate() {
        let piece_width = tex_visual_width(connector) + tex_visual_width(factor);
        if index == 0 || current_width + piece_width <= budget {
            current.push_str(connector);
            current.push_str(factor);
            current_width += piece_width;
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(connector);
            current.push_str(factor);
            current_width = piece_width;
        }
    }
    current.push_str(tail);
    lines.push(current);
    lines
}

/// Same as `wrap_display_lines`, but never splits inside an item.  This is
/// useful for explicit rational sums where each item is a coefficient times a
/// gamma monomial: breaking inside the coefficient polynomial would detach part
/// of the coefficient from the monomial it multiplies.
fn wrap_display_lines_preserving_items(
    head: &str,
    items: &[(String, String)],
    tail: &str,
    budget: usize,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = head.to_string();
    let mut current_width = tex_visual_width(head);

    for (index, (connector, factor)) in items.iter().enumerate() {
        let piece_width = tex_visual_width(connector) + tex_visual_width(factor);
        if index == 0 || current_width + piece_width <= budget {
            current.push_str(connector);
            current.push_str(factor);
            current_width += piece_width;
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(connector);
            current.push_str(factor);
            current_width = piece_width;
        }
    }
    current.push_str(tail);
    lines.push(current);
    lines
}

/// A single factor can still be wider than a whole line — most often an edge
/// propagator that expands into a parenthesized sum of several signed root-sums.
/// Such factors are split at their top-level `+`/`-` boundaries so the inner
/// sum can break across lines; the enclosing `\bigl(`/`\bigr)` simply ride along
/// on the first and last pieces.  Factors that already fit are left untouched.
fn expand_wide_items(items: &[(String, String)], budget: usize) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (connector, factor) in items {
        if tex_visual_width(factor) <= budget {
            out.push((connector.clone(), factor.clone()));
            continue;
        }
        let pieces = split_signed_sum(factor);
        if pieces.len() <= 1 {
            out.push((connector.clone(), factor.clone()));
            continue;
        }
        for (piece_index, (sign, text)) in pieces.into_iter().enumerate() {
            let piece_connector = if piece_index == 0 {
                connector.clone()
            } else {
                sign
            };
            out.push((piece_connector, text));
        }
    }
    out
}

/// Split a fragment on its top-level (brace depth zero) ` + ` / ` - ` separators,
/// returning each summand paired with the sign that precedes it (empty for the
/// first).  Used only to break an over-wide parenthesized sum across lines.
fn split_signed_sum(fragment: &str) -> Vec<(String, String)> {
    let chars: Vec<char> = fragment.chars().collect();
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut connector = String::new();
    let mut depth = 0i32;
    let mut index = 0;
    while index < chars.len() {
        let c = chars[index];
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
        }
        if depth == 0
            && c == ' '
            && index + 2 < chars.len()
            && (chars[index + 1] == '+' || chars[index + 1] == '-')
            && chars[index + 2] == ' '
        {
            parts.push((std::mem::take(&mut connector), std::mem::take(&mut current)));
            connector = chars[index + 1].to_string();
            index += 3;
            continue;
        }
        current.push(c);
        index += 1;
    }
    parts.push((connector, current));
    parts
}

/// Render a self-contained product (a compact graph bracket) with `multlined`.
/// These displays are short and never span a page, so the centered continuation
/// lines of `multlined` read nicely.  A display that already fits on one line is
/// emitted as an ordinary centered equation instead of a left-flushed singleton.
fn tex_multlined_display(
    head: &str,
    items: &[(String, String)],
    tail: &str,
    budget: usize,
) -> String {
    let lines = wrap_display_lines(head, items, tail, budget);
    let mut out = String::new();
    if lines.len() == 1 {
        out.push_str("\\[\n");
        out.push_str(&lines[0]);
        out.push_str("\n\\]\n");
    } else {
        out.push_str("\\[\n\\begin{multlined}[b]\n");
        out.push_str(&lines.join("\\\\\n"));
        out.push_str("\n\\end{multlined}\n\\]\n");
    }
    out
}

/// Render a long sum (the fully expanded basis-coefficient terms) with `align*`.
/// Unlike `multlined`, `align*` rows break across pages under
/// `\allowdisplaybreaks`, which matters because one expanded contribution can be
/// many pages tall.  `head` must already contain the alignment `&` (just after
/// the `=`); continuation lines are flushed to that column.
fn tex_aligned_display(
    head: &str,
    items: &[(String, String)],
    tail: &str,
    budget: usize,
) -> String {
    let lines = wrap_display_lines(head, items, tail, budget);
    tex_align_lines(&lines)
}

fn tex_aligned_display_preserving_items(
    head: &str,
    items: &[(String, String)],
    tail: &str,
    budget: usize,
) -> String {
    let lines = wrap_display_lines_preserving_items(head, items, tail, budget);
    tex_align_lines(&lines)
}

fn tex_align_lines(lines: &[String]) -> String {
    let body = lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            if index == 0 {
                line.clone()
            } else {
                format!("&{line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\\\\\n");
    format!("\\begin{{align*}}\n{body}\n\\end{{align*}}\n")
}

/// Rough estimate of the rendered width of a TeX fragment, in character-ish
/// units.  Used only to decide line breaks: control sequences and grouping or
/// scripting delimiters are dropped, while the visible glyphs they wrap are
/// counted.  A deliberate over-count of scripts keeps the estimate on the safe
/// (slightly narrow) side.
fn tex_visual_width(fragment: &str) -> usize {
    let mut width = 0usize;
    let mut chars = fragment.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.peek() {
                Some(d) if d.is_ascii_alphabetic() => {
                    let mut name = String::new();
                    while let Some(&d) = chars.peek() {
                        if d.is_ascii_alphabetic() {
                            name.push(d);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    width += match name.as_str() {
                        // Big operators occupy roughly an em plus inter-script
                        // slack, far more than a single glyph.
                        "sum" | "prod" | "int" => 4,
                        "frac" => 2,
                        "bigl" | "bigr" | "langle" | "rangle" => 1,
                        "cdot" | "Delta" | "Theta" | "Psi" | "Gamma" | "eta" | "tau" | "gamma"
                        | "psi" | "phi" | "nu" | "alpha" | "beta" | "ldots" | "exp" | "bar"
                        | "mathbf" | "mathrm" => 1,
                        _ => 0,
                    };
                }
                Some(_) => {
                    chars.next();
                    width += 1;
                }
                None => {}
            },
            '{' | '}' | '_' | '^' | ' ' => {}
            _ => width += 1,
        }
    }
    width
}

fn leg_factor(marking: usize, color: &str, power: usize, request: &FormulaRequest) -> String {
    let mut terms = Vec::new();
    for k in 0..=request.max_descendant_power {
        for s in 0..=k {
            for r in 0..=request.inverse_r_order() {
                if k - s + r != power {
                    continue;
                }
                terms.push(format!(
                    "sum_{{alpha,beta,j=0..{}}} x_{{{marking},{k},alpha}} * RInv_{r}[{color},j] * PsiInv[j,beta] * S_{s}[beta,alpha]",
                    request.colors - 1
                ));
            }
        }
    }
    parenthesized_sum(&terms)
}

fn leg_factor_tex(marking: usize, color: &str, power: usize, request: &FormulaRequest) -> String {
    let mut terms = Vec::new();
    for k in 0..=request.max_descendant_power {
        for s in 0..=k {
            for r in 0..=request.inverse_r_order() {
                if k - s + r != power {
                    continue;
                }
                terms.push(format!(
                    "\\sum_{{\\alpha,\\beta,j=0}}^{{{}}} x_{{{marking},{k},\\alpha}}\\,(R^{{-1}}_{{{r}}})_{{{color},j}}\\mathbin{{\\cdot}}(\\Psi^{{-1}})_{{j,\\beta}}\\,(S_{{{s}}})_{{\\beta,\\alpha}}",
                    request.colors - 1
                ));
            }
        }
    }
    parenthesized_sum_tex(&terms)
}

fn edge_factor(
    left_color: &str,
    right_color: &str,
    left_power: usize,
    right_power: usize,
    colors: usize,
) -> String {
    let mut out = String::new();
    out.push('(');
    for t in 0..=right_power {
        let sign = if t % 2 == 0 { "-" } else { "+" };
        if t == 0 {
            out.push_str(sign);
        } else {
            out.push(' ');
            out.push_str(sign);
            out.push(' ');
        }
        out.push_str(&format!(
            "sum_{{nu=0..{}}} RInv_{}[{left_color},nu] * EtaInv_nu * RInv_{}[{right_color},nu]",
            colors - 1,
            left_power + 1 + t,
            right_power - t
        ));
    }
    out.push(')');
    out
}

fn edge_factor_tex(
    left_color: &str,
    right_color: &str,
    left_power: usize,
    right_power: usize,
    colors: usize,
) -> String {
    let mut terms = Vec::new();
    for t in 0..=right_power {
        let sign = if t % 2 == 0 { '-' } else { '+' };
        terms.push((
            sign,
            format!(
                "\\sum_{{\\nu=0}}^{{{}}}(R^{{-1}}_{{{}}})_{{{},\\nu}}\\mathbin{{\\cdot}}\\eta^{{\\nu\\nu}}\\,(R^{{-1}}_{{{}}})_{{{},\\nu}}",
                colors - 1,
                left_power + 1 + t,
                left_color,
                right_power - t,
                right_color
            ),
        ));
    }
    parenthesized_signed_sum_tex(&terms)
}

fn parenthesized_sum(terms: &[String]) -> String {
    match terms {
        [] => "0".to_string(),
        [term] => format!("({term})"),
        _ => format!("({})", terms.join(" + ")),
    }
}

fn parenthesized_sum_tex(terms: &[String]) -> String {
    match terms {
        [] => "0".to_string(),
        [term] => term.clone(),
        _ => format!("\\bigl({}\\bigr)", terms.join(" + ")),
    }
}

fn parenthesized_signed_sum_tex(terms: &[(char, String)]) -> String {
    match terms {
        [] => "0".to_string(),
        _ => {
            let mut out = "\\bigl(".to_string();
            for (idx, (sign, term)) in terms.iter().enumerate() {
                match (idx, sign) {
                    (0, '-') => out.push('-'),
                    (0, _) => {}
                    (_, '-') => out.push_str(" - "),
                    (_, _) => out.push_str(" + "),
                }
                out.push_str(term);
            }
            out.push_str("\\bigr)");
            out
        }
    }
}

fn vertex_expanded_terms(genus: usize, color: &str, base_powers: &[usize]) -> Vec<String> {
    let dimension = 3 * genus + base_powers.len() - 3;
    let power_sum = base_powers.iter().sum::<usize>();
    if power_sum > dimension {
        return Vec::new();
    }

    let excess = dimension - power_sum;
    if excess == 0 {
        return vec![join_factors(&[
            tft_factor(genus, color, base_powers.len()),
            psi_integral_factor(genus, base_powers),
        ])];
    }

    translation_partitions(excess)
        .into_iter()
        .map(|partition| {
            let translation_count = partition
                .iter()
                .map(|(_, multiplicity)| *multiplicity)
                .sum::<usize>();
            let mut powers = base_powers.to_vec();
            let mut factors = Vec::new();
            let mut symmetry = 1usize;
            for (translation_excess, multiplicity) in partition {
                let translation_power = translation_excess + 1;
                powers.extend(std::iter::repeat(translation_power).take(multiplicity));
                factors.push(powered_factor(
                    &format!("T_{{{color}}}^{translation_power}"),
                    multiplicity,
                ));
                symmetry *= factorial(multiplicity);
            }
            if symmetry > 1 {
                factors.push(format!("1/{symmetry}"));
            }
            factors.push(tft_factor(
                genus,
                color,
                base_powers.len() + translation_count,
            ));
            factors.push(psi_integral_factor(genus, &powers));
            join_factors(&factors)
        })
        .collect()
}

fn vertex_expanded_factor_terms_tex(
    genus: usize,
    color: &str,
    base_powers: &[usize],
) -> Vec<Vec<String>> {
    let dimension = 3 * genus + base_powers.len() - 3;
    let power_sum = base_powers.iter().sum::<usize>();
    if power_sum > dimension {
        return Vec::new();
    }

    let excess = dimension - power_sum;
    if excess == 0 {
        let mut factors = tft_factor_tex_factors(genus, color, base_powers.len());
        factors.push(psi_integral_factor_tex(genus, base_powers));
        return vec![factors];
    }

    translation_partitions(excess)
        .into_iter()
        .map(|partition| {
            let translation_count = partition
                .iter()
                .map(|(_, multiplicity)| *multiplicity)
                .sum::<usize>();
            let mut powers = base_powers.to_vec();
            let mut factors = Vec::new();
            let mut symmetry = 1usize;
            for (translation_excess, multiplicity) in partition {
                let translation_power = translation_excess + 1;
                powers.extend(std::iter::repeat(translation_power).take(multiplicity));
                factors.push(powered_factor_tex(
                    &format!("(T_{{{translation_power}}})_{{{color}}}"),
                    multiplicity,
                ));
                symmetry *= factorial(multiplicity);
            }
            if symmetry > 1 {
                factors.push(format!("\\frac{{1}}{{{symmetry}}}"));
            }
            factors.extend(tft_factor_tex_factors(
                genus,
                color,
                base_powers.len() + translation_count,
            ));
            factors.push(psi_integral_factor_tex(genus, &powers));
            factors
        })
        .collect()
}

fn tft_factor(genus: usize, color: &str, valence: usize) -> String {
    let mut factors = Vec::new();
    if genus == 0 {
        factors.push(format!("DeltaInv_{{{color}}}"));
    } else if genus > 1 {
        factors.push(powered_factor(&format!("Delta_{{{color}}}"), genus - 1));
    }
    factors.push(powered_factor(
        &format!("RelSqrtDelta_{{{color}}}"),
        valence,
    ));
    join_factors(&factors)
}

fn tft_factor_tex_factors(genus: usize, color: &str, valence: usize) -> Vec<String> {
    let mut factors = Vec::new();
    if genus == 0 {
        factors.push(format!("\\Delta_{{{color}}}^{{-1}}"));
    } else if genus > 1 {
        factors.push(powered_factor_tex(
            &format!("\\Delta_{{{color}}}"),
            genus - 1,
        ));
    }
    factors.push(powered_factor_tex(
        &format!("\\Delta_{{{color}}}^{{1/2}}"),
        valence,
    ));
    factors
}

fn psi_integral_factor(genus: usize, powers: &[usize]) -> String {
    if powers.is_empty() {
        format!("PsiInt({genus};)")
    } else {
        let powers = powers
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!("PsiInt({genus};{powers})")
    }
}

fn psi_integral_factor_tex(genus: usize, powers: &[usize]) -> String {
    let insertions = if powers.is_empty() {
        "1".to_string()
    } else {
        powers
            .iter()
            .map(|power| format!("\\tau_{{{power}}}"))
            .collect::<Vec<_>>()
            .join("")
    };
    format!("\\langle {insertions}\\rangle_{{{genus}}}^{{\\mathrm{{pt}}}}")
}

fn powered_factor(base: &str, exponent: usize) -> String {
    match exponent {
        0 => "1".to_string(),
        1 => base.to_string(),
        _ => format!("({base})^{exponent}"),
    }
}

fn powered_factor_tex(base: &str, exponent: usize) -> String {
    match exponent {
        0 => "1".to_string(),
        1 => base.to_string(),
        _ => format!("\\bigl({base}\\bigr)^{{{exponent}}}"),
    }
}

fn join_factors(factors: &[String]) -> String {
    let nontrivial = factors
        .iter()
        .filter(|factor| factor.as_str() != "1")
        .cloned()
        .collect::<Vec<_>>();
    if nontrivial.is_empty() {
        "1".to_string()
    } else {
        nontrivial.join(" * ")
    }
}

impl TikzPoint {
    fn from_polar(angle: f64, radius: f64) -> Self {
        Self {
            x: radius * angle.cos(),
            y: radius * angle.sin(),
        }
    }

    fn add_polar(self, angle: f64, radius: f64) -> Self {
        Self {
            x: self.x + radius * angle.cos(),
            y: self.y + radius * angle.sin(),
        }
    }
}

fn tikz_vertex_positions(vertex_count: usize) -> Vec<TikzPoint> {
    match vertex_count {
        0 => Vec::new(),
        1 => vec![TikzPoint { x: 0.0, y: 0.0 }],
        2 => vec![
            TikzPoint { x: -1.35, y: 0.0 },
            TikzPoint { x: 1.35, y: 0.0 },
        ],
        count => {
            let radius = 1.65 + 0.12 * count as f64;
            (0..count)
                .map(|idx| {
                    let angle = PI / 2.0 + 2.0 * PI * idx as f64 / count as f64;
                    TikzPoint::from_polar(angle, radius)
                })
                .collect()
        }
    }
}

fn edge_pair_counts(edges: &[StableEdge]) -> BTreeMap<(usize, usize), usize> {
    let mut counts = BTreeMap::new();
    for edge in edges {
        if !edge.is_loop() {
            *counts.entry((edge.a, edge.b)).or_insert(0) += 1;
        }
    }
    counts
}

fn tikz_bend_option(ordinal: usize, total: usize) -> String {
    if total <= 1 {
        return String::new();
    }
    let midpoint = (total - 1) as f64 / 2.0;
    let offset = ordinal as f64 - midpoint;
    if offset.abs() < 0.1 {
        String::new()
    } else {
        let direction = if offset > 0.0 { "left" } else { "right" };
        let angle = 16 + (offset.abs() * 12.0).round() as usize;
        format!("[bend {direction}={angle}]")
    }
}

fn tikz_loop_direction(vertex: usize, ordinal: usize, vertex_count: usize) -> &'static str {
    // Only the four cardinal `loop <dir>` keys are predefined by TikZ; the
    // diagonal variants are not valid pgf keys, so we never emit them.
    const DIRECTIONS: [&str; 4] = ["above", "right", "left", "below"];
    if vertex_count == 1 {
        // Markings on an isolated vertex fan out downward (see `tikz_leg_angle`),
        // so keep loops in the upper half-plane first to avoid overlapping them.
        return DIRECTIONS[ordinal % DIRECTIONS.len()];
    }
    DIRECTIONS[(vertex + ordinal) % DIRECTIONS.len()]
}

fn leg_counts_by_vertex(graph: &StableGraph) -> Vec<usize> {
    let mut counts = vec![0usize; graph.vertices.len()];
    for &vertex in &graph.legs {
        counts[vertex] += 1;
    }
    counts
}

fn tikz_leg_angle(
    vertex: usize,
    ordinal: usize,
    total_at_vertex: usize,
    positions: &[TikzPoint],
) -> f64 {
    if positions.len() == 1 {
        // Fan the markings out across the lower half-plane, centered straight
        // down, leaving the top free for loop edges.
        let spread = PI / 6.0;
        let midpoint = total_at_vertex.saturating_sub(1) as f64 / 2.0;
        return -PI / 2.0 + (ordinal as f64 - midpoint) * spread;
    }

    let point = positions[vertex];
    let base = point.y.atan2(point.x);
    let spread = PI / 9.0;
    let midpoint = (total_at_vertex.saturating_sub(1)) as f64 / 2.0;
    base + (ordinal as f64 - midpoint) * spread
}

fn tikz_anchor(angle: f64) -> &'static str {
    let x = angle.cos();
    let y = angle.sin();
    match (x > 0.35, x < -0.35, y > 0.35, y < -0.35) {
        (true, _, true, _) => "south west",
        (true, _, _, true) => "north west",
        (_, true, true, _) => "south east",
        (_, true, _, true) => "north east",
        (true, _, _, _) => "west",
        (_, true, _, _) => "east",
        (_, _, true, _) => "south",
        (_, _, _, true) => "north",
        _ => "center",
    }
}

fn translation_partitions(total: usize) -> Vec<Vec<(usize, usize)>> {
    fn rec(
        next_excess: usize,
        remaining: usize,
        current: &mut Vec<(usize, usize)>,
        out: &mut Vec<Vec<(usize, usize)>>,
    ) {
        if remaining == 0 {
            out.push(current.clone());
            return;
        }
        for excess in next_excess..=remaining {
            let max_multiplicity = remaining / excess;
            for multiplicity in 1..=max_multiplicity {
                current.push((excess, multiplicity));
                rec(excess + 1, remaining - excess * multiplicity, current, out);
                current.pop();
            }
        }
    }

    let mut out = Vec::new();
    rec(1, total, &mut Vec::new(), &mut out);
    out
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formula_request_records_expected_truncation_bounds() {
        let mut req = FormulaRequest::new(2, 1, 3);
        req.max_descendant_power = 5;
        assert_eq!(req.graph_dimension(), 4);
        assert_eq!(req.inverse_r_order(), 5);
        assert_eq!(req.edge_power_max(), 4);
        assert_eq!(req.translation_power_max(), 5);
        assert_eq!(req.z_order(), 5);
    }

    #[test]
    fn genus_zero_three_marking_skeleton_has_one_graph() {
        let skeleton = build_formula_skeleton(FormulaRequest::new(0, 3, 2)).unwrap();
        assert_eq!(skeleton.graphs.len(), 1);
        assert_eq!(skeleton.graphs[0].automorphism_order, 1);
        assert!(skeleton.render_text().contains("Raw basis glossary"));
        assert!(!skeleton
            .render_text()
            .contains("Expanded contribution in basis coefficients"));
    }

    #[test]
    fn graph_renderer_unravels_basis_coefficients() {
        let mut request = FormulaRequest::new(0, 3, 2);
        request.basis = FormulaBasisMode::Coefficients;
        let skeleton = build_formula_skeleton(request).unwrap();
        let rendered = skeleton.render_text();
        assert!(rendered.contains("Expanded contribution in basis coefficients"));
        assert!(rendered.contains("x_{0,0,alpha}"));
        assert!(rendered.contains("RInv_0[i0,j]"));
        assert!(rendered.contains("S_0[beta,alpha]"));
        assert!(rendered.contains("DeltaInv_{i0}"));
        assert!(rendered.contains("PsiInt(0;0,0,0)"));
        assert!(!rendered.contains("Here L"));
        assert!(!rendered.contains("L_{"));
    }

    #[test]
    fn tex_renderer_uses_standard_givental_symbols() {
        let mut request = FormulaRequest::new(0, 3, 2);
        request.basis = FormulaBasisMode::Coefficients;
        let skeleton = build_formula_skeleton(request).unwrap();
        let rendered = skeleton.render_tex();
        assert!(rendered.contains("\\section*{Givental Graph Formula Skeleton}"));
        assert!(rendered.contains("\\begin{tikzpicture}"));
        assert!(rendered.contains("\\draw[leg]"));
        assert!(rendered.contains("\\mathbin{\\cdot}"));
        assert!(!rendered.contains("\\left["));
        assert!(!rendered.contains("\\right]"));
        assert!(rendered.contains("(R^{-1}_{0})_{i_{0},j}"));
        assert!(rendered.contains("(\\Psi^{-1})_{j,\\beta}"));
        assert!(rendered.contains("(S_{0})_{\\beta,\\alpha}"));
        assert!(rendered.contains("\\Delta_{i_{0}}^{-1}"));
        assert!(rendered.contains("\\langle \\tau_{0}\\tau_{0}\\tau_{0}\\rangle_{0}"));
    }

    #[test]
    fn tex_renderer_wraps_expanded_sums_in_pagebreakable_align() {
        let mut request = FormulaRequest::new(1, 1, 2);
        request.basis = FormulaBasisMode::Coefficients;
        let skeleton = build_formula_skeleton(request).unwrap();
        let rendered = skeleton.render_tex();
        // Expanded sums can run many pages tall, so they use `align*` (whose rows
        // break across pages) rather than the old `breqn`/`dmath*` route that
        // overflowed both horizontally and vertically.
        assert!(rendered.contains("\\begin{align*}"));
        assert!(!rendered.contains("dmath"));
        assert!(rendered.contains("\\mathbin{\\cdot}(\\Psi^{-1})"));
        assert!(rendered.contains("\\mathbin{\\cdot}\\eta^{\\nu\\nu}"));
        assert!(rendered.contains("\\mathbin{\\cdot}(T_{2})_{i_{0}}"));
        assert!(rendered.contains("\\mathbin{\\cdot}\\langle \\tau_{1}\\rangle_{1}^{\\mathrm{pt}}"));
        assert!(!rendered.contains("\\begin{aligned}[t]"));
        assert!(!rendered.contains("&\\qquad {}\\cdot"));
    }

    #[test]
    fn tex_renderer_wraps_compact_brackets_in_multlined() {
        let mut request = FormulaRequest::new(2, 1, 3);
        request.basis = FormulaBasisMode::Resolvent;
        let skeleton = build_formula_skeleton(request).unwrap();
        let rendered = skeleton.render_tex();
        // The compact graph brackets are short and use `multlined`.
        assert!(rendered.contains("\\begin{multlined}[b]"));
        assert!(rendered.contains("\\bigl\\langle "));
        assert!(rendered.contains("\\bigr\\rangle_\\Gamma^{\\mathrm{pt}}"));
        assert!(!rendered.contains("dmath"));
    }

    #[test]
    fn resolvent_basis_packs_descendants_into_kernels() {
        let mut request = FormulaRequest::new(1, 1, 2);
        request.basis = FormulaBasisMode::Resolvent;
        let skeleton = build_formula_skeleton(request).unwrap();
        let rendered = skeleton.render_tex();
        assert!(rendered.contains("Resolvent Basis Elements"));
        assert!(rendered.contains("\\mathcal L_i^{\\gamma_\\ell}(z_\\ell,\\psi)"));
        assert!(rendered.contains("C_{0}^{\\mathrm{res}}"));
        assert!(rendered.contains("(R^{-1}(\\bar\\psi_{\\ell_{0}}))_{i_{0},j}"));
        assert!(rendered.contains("(\\Psi^{-1})_{j,\\beta}"));
        assert!(rendered.contains("S(z_{0})_{\\beta,\\alpha}"));
        assert!(rendered.contains("\\frac{\\gamma_{0,\\alpha}}{z_{0}-\\bar\\psi_{\\ell_{0}}}"));
        assert!(rendered.contains("(R^{-1}(\\bar\\psi_{e_{0},+}))_{i_{0},\\nu}"));
        assert!(!rendered.contains("\\mathcal L_{i_{0}}^{\\gamma_{0}}"));
        assert!(!rendered.contains("\\mathcal E_{i_{0}i_{0}}"));
        assert!(rendered.contains("\\langle"));
        assert!(!rendered.contains("Expanded contribution in basis coefficients"));
    }

    #[test]
    fn raw_basis_uses_engine_specialized_kernel_notation() {
        let mut request = FormulaRequest::new(0, 3, 3);
        request.basis = FormulaBasisMode::Raw;
        request.expansion = Some(FormulaExpansion::ProjectiveSpace {
            n: 2,
            equivariant: false,
        });
        let skeleton = build_formula_skeleton(request).unwrap();
        let rendered = skeleton.render_tex();
        assert!(rendered.contains("Raw Basis: Ordinary Projective Space"));
        assert!(rendered.contains("P(u_i)&=0"));
        assert!(rendered.contains("C_{0}^{\\mathrm{raw}}"));
        assert!(rendered.contains("\\bigl(P'(u_{i_{0}})\\bigr)^{-1}"));
        assert!(rendered.contains("(R^{\\mathrm{raw},-1}(\\bar\\psi_{\\ell_{0}}))_{i_{0},j}"));
        assert!(rendered.contains("\\Delta_j^{-1/2}u_j^\\beta"));
        assert!(rendered.contains("S^{\\mathrm{raw}}(z_{0})_{\\beta,\\alpha}"));
        assert!(!rendered.contains("\\mathcal L_{i_{0}}^{\\mathrm{raw},\\gamma_{0}}"));
        assert!(!rendered.contains("\\mathcal L^{\\mathrm{raw}}_{i_{0}}^{\\gamma_{0}}"));
        assert!(!rendered.contains("\\Theta_{0, 3}^{\\mathrm{raw}}(i_{0})"));
    }

    #[test]
    fn rational_basis_reduces_projective_primary_three_point_graph() {
        let mut request = FormulaRequest::new(0, 3, 3);
        request.basis = FormulaBasisMode::Rational;
        request.expansion = Some(FormulaExpansion::ProjectiveSpace {
            n: 2,
            equivariant: false,
        });
        let skeleton = build_formula_skeleton(request).unwrap();
        let rendered = skeleton.render_text();
        assert!(rendered.contains("Rational residue-reduced contribution"));
        assert!(rendered.contains("gamma_{0,0} * gamma_{1,0} * gamma_{2,2}"));
        assert!(rendered.contains("lambda_0"));
        assert!(rendered.contains("lambda_1"));
        assert!(rendered.contains("lambda_2"));
        assert!(rendered.contains("residue sum f(H)/P'(H)"));
        assert!(!rendered.contains("concrete graph-wise q-series"));

        let rendered_tex = skeleton.render_tex();
        assert!(rendered_tex.contains("C_{0}^{\\mathrm{rat}}"));
        assert!(rendered_tex.contains(
            "\\lambda_0\\mathbin{\\cdot}\\gamma_{0,0}\\mathbin{\\cdot}\\gamma_{1,1}\\mathbin{\\cdot}\\gamma_{2,2}"
        ));
    }

    #[test]
    fn rational_basis_reports_unreduced_graphs_explicitly() {
        let mut request = FormulaRequest::new(1, 1, 2);
        request.basis = FormulaBasisMode::Rational;
        request.expansion = Some(FormulaExpansion::ProjectiveSpace {
            n: 1,
            equivariant: false,
        });
        let skeleton = build_formula_skeleton(request).unwrap();
        let rendered = skeleton.render_text();
        assert!(rendered.contains("not implemented for this graph"));
        assert!(!rendered.contains("Packed rational contribution"));
    }

    #[test]
    fn tex_document_renderer_is_standalone() {
        let skeleton = build_formula_skeleton(FormulaRequest::new(0, 3, 2)).unwrap();
        let rendered = skeleton.render_tex_document();
        assert!(rendered.starts_with("\\documentclass[11pt]{article}"));
        assert!(rendered.contains("\\usepackage{amsmath,amssymb,mathtools}"));
        assert!(rendered.contains("\\usepackage{tikz}"));
        assert!(rendered.contains("\\begin{document}"));
        assert!(rendered.contains("\\begin{tikzpicture}"));
        assert!(rendered.ends_with("\\end{document}\n"));
    }

    #[test]
    fn tikz_renderer_draws_loop_edges() {
        let skeleton = build_formula_skeleton(FormulaRequest::new(1, 1, 2)).unwrap();
        let rendered = skeleton.render_tex();
        assert!(rendered.contains("edge[loop"));
        assert!(rendered.contains("node[edge label] {$e_{0}$}"));
    }

    #[test]
    fn vertex_terms_expand_translation_partitions() {
        let terms = vertex_expanded_terms(1, "i0", &[0]);
        assert_eq!(terms.len(), 1);
        assert!(terms[0].contains("T_{i0}^2"));
        assert!(terms[0].contains("PsiInt(1;0,2)"));
    }
}
