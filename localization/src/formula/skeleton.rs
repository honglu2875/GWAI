//! Finite formula skeletons for fixed stable `(g,m)`.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::error::GwError;
use crate::graphs::{stable_graphs, StableEdge, StableGraph};

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
            out.push_str(&expansion.render_text());
        }
        if self.request.include_glossary {
            out.push('\n');
            out.push_str(&basis_glossary());
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
            out.push_str(&expansion.render_tex());
        }
        if self.request.include_glossary {
            out.push('\n');
            self.render_tex_glossary(&mut out);
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
        out.push_str("\\usepackage{tikz}\n");
        out.push_str("\\usetikzlibrary{calc,arrows.meta}\n");
        out.push_str("\\allowdisplaybreaks\n");
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
        match self.request.q_degree {
            Some(degree) => out.push_str(&format!(
                "Calibration q-series should be read modulo q^{}.\n",
                degree + 1
            )),
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
        out.push_str("How the basis elements assemble\n");
        out.push_str("--------------------------------\n");
        out.push_str("For a formal insertion at marking ell,\n");
        out.push_str("  gamma_ell = sum_{k<=K,a} x_{ell,k,a} tau_k(phi_a),\n");
        out.push_str("each marking factor is expanded directly as finite sums of\n");
        out.push_str("  x_{ell,k,a} * S_s[b,a] * PsiInv[j,b] * RInv_r[i,j]\n");
        out.push_str("with p = k - s + r.  Internal edges are also expanded directly in\n");
        out.push_str(
            "RInv and EtaInv, so the graph terms below use only primitive calibration basis elements.\n\n",
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
            "Stable range: $g={}$, $m={}$.\n\n",
            self.request.genus, self.request.markings
        ));
        out.push_str(&format!(
            "Canonical colors are indexed by $i=0,\\ldots,{}$.\n\n",
            self.request.colors - 1
        ));
        out.push_str(&format!(
            "\\[\nD=3g-3+m={}\n\\]\n",
            self.request.graph_dimension()
        ));
        match self.request.q_degree {
            Some(degree) => out.push_str(&format!(
                "Calibration series are read modulo $q^{{{}}}$.\n",
                degree + 1
            )),
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
        out.push_str("For formal descendant insertions we write\n");
        out.push_str("\\[\n");
        out.push_str("\\gamma_\\ell=\\sum_{0\\le k\\le K}\\sum_\\alpha x_{\\ell,k,\\alpha}\\,\\tau_k(\\phi_\\alpha).\n");
        out.push_str("\\]\n");
        out.push_str(
            "The leg factors are expanded in the primitive coefficients of $S(z)$, $\\Psi^{-1}$, and $R(z)^{-1}$.\n",
        );
        out.push_str(
            "Internal edges use the standard Givental propagator expanded in $R(z)^{-1}$ and the inverse canonical metric.\n",
        );
        out.push_str(
            "At a vertex of genus $h$ and color $i$, translation insertions $(T_p)_i$ are integrated against point-theory psi classes:\n",
        );
        out.push_str("\\[\n");
        out.push_str(
            "\\left\\langle \\tau_{p_1}\\cdots\\tau_{p_N}\\right\\rangle_h^{\\mathrm{pt}}\n",
        );
        out.push_str("=\n");
        out.push_str("\\int_{\\overline{\\mathcal M}_{h,N}}\\prod_{a=1}^N \\psi_a^{p_a}.\n");
        out.push_str("\\]\n");
    }

    fn render_tex_glossary(&self, out: &mut String) {
        out.push_str("\\subsection*{Primitive Basis Elements}\n");
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
            "\\item $\\left\\langle \\tau_{p_1}\\cdots\\tau_{p_N}\\right\\rangle_h^{\\mathrm{pt}}$: Witten--Kontsevich intersection number on $\\overline{\\mathcal M}_{h,N}$.\n",
        );
        out.push_str("\\end{itemize}\n");
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
        self.render_expanded_expression(out, request);
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
        self.render_expanded_tex_expression(out, request);
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

    fn render_expanded_tex_expression(&self, out: &mut String, request: &FormulaRequest) {
        let terms = self.expanded_tex_terms(request);
        out.push_str("\\[\n");
        out.push_str(&format!("C_{{{}}}=", self.index));
        if terms.is_empty() {
            out.push_str("0\n\\]\n");
            return;
        }

        out.push_str(&format!(
            "\\frac{{1}}{{{}}}{}\\left[\\begin{{aligned}}\n",
            self.automorphism_order,
            self.color_sum_tex(request.colors)
        ));
        for (idx, term) in terms.iter().enumerate() {
            if idx == 0 {
                out.push_str("&");
            } else {
                out.push_str("&+");
            }
            out.push_str(term);
            out.push_str("\\\\\n");
        }
        out.push_str("\\end{aligned}\\right]\n");
        out.push_str("\\]\n");
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

    fn expanded_tex_terms(&self, request: &FormulaRequest) -> Vec<String> {
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
                    vertex_expanded_terms_tex(
                        vertex.genus,
                        &format!("i_{{{}}}", vertex.index),
                        &assignment.vertex_powers[vertex.index],
                    )
                })
                .collect::<Vec<_>>();
            append_distributed_tex_terms(
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

fn append_distributed_tex_terms(
    fixed_factors: &[String],
    vertex_terms: &[Vec<String>],
    vertex_index: usize,
    current_vertex_factors: &mut Vec<String>,
    out: &mut Vec<String>,
) {
    if vertex_index == vertex_terms.len() {
        let mut factors = fixed_factors.to_vec();
        factors.extend(current_vertex_factors.iter().cloned());
        out.push(join_factors_tex(&factors));
        return;
    }

    for term in &vertex_terms[vertex_index] {
        current_vertex_factors.push(term.clone());
        append_distributed_tex_terms(
            fixed_factors,
            vertex_terms,
            vertex_index + 1,
            current_vertex_factors,
            out,
        );
        current_vertex_factors.pop();
    }
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
                    "\\sum_{{\\alpha,\\beta,j=0}}^{{{}}} x_{{{marking},{k},\\alpha}}\\,(R^{{-1}}_{{{r}}})_{{{color},j}}\\,(\\Psi^{{-1}})_{{j,\\beta}}\\,(S_{{{s}}})_{{\\beta,\\alpha}}",
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
    let mut out = String::new();
    out.push_str("\\left(");
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
            "\\sum_{{\\nu=0}}^{{{}}}(R^{{-1}}_{{{}}})_{{{},\\nu}}\\,\\eta^{{\\nu\\nu}}\\,(R^{{-1}}_{{{}}})_{{{},\\nu}}",
            colors - 1,
            left_power + 1 + t,
            left_color,
            right_power - t,
            right_color
        ));
    }
    out.push_str("\\right)");
    out
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
        [term] => format!("\\left({term}\\right)"),
        _ => format!("\\left({}\\right)", terms.join(" + ")),
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

fn vertex_expanded_terms_tex(genus: usize, color: &str, base_powers: &[usize]) -> Vec<String> {
    let dimension = 3 * genus + base_powers.len() - 3;
    let power_sum = base_powers.iter().sum::<usize>();
    if power_sum > dimension {
        return Vec::new();
    }

    let excess = dimension - power_sum;
    if excess == 0 {
        return vec![join_factors_tex(&[
            tft_factor_tex(genus, color, base_powers.len()),
            psi_integral_factor_tex(genus, base_powers),
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
                factors.push(powered_factor_tex(
                    &format!("(T_{{{translation_power}}})_{{{color}}}"),
                    multiplicity,
                ));
                symmetry *= factorial(multiplicity);
            }
            if symmetry > 1 {
                factors.push(format!("\\frac{{1}}{{{symmetry}}}"));
            }
            factors.push(tft_factor_tex(
                genus,
                color,
                base_powers.len() + translation_count,
            ));
            factors.push(psi_integral_factor_tex(genus, &powers));
            join_factors_tex(&factors)
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

fn tft_factor_tex(genus: usize, color: &str, valence: usize) -> String {
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
    join_factors_tex(&factors)
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
    format!("\\left\\langle {insertions}\\right\\rangle_{{{genus}}}^{{\\mathrm{{pt}}}}")
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
        _ => format!("\\left({base}\\right)^{{{exponent}}}"),
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

fn join_factors_tex(factors: &[String]) -> String {
    let nontrivial = factors
        .iter()
        .filter(|factor| factor.as_str() != "1")
        .cloned()
        .collect::<Vec<_>>();
    if nontrivial.is_empty() {
        "1".to_string()
    } else {
        nontrivial.join("\\,")
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
    const DIRECTIONS: [&str; 8] = [
        "above",
        "right",
        "below",
        "left",
        "above right",
        "below right",
        "below left",
        "above left",
    ];
    if vertex_count == 1 {
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
        return PI / 2.0 + 2.0 * PI * ordinal as f64 / total_at_vertex.max(1) as f64;
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
        assert!(skeleton.render_text().contains("Basis glossary"));
    }

    #[test]
    fn graph_renderer_unravels_basis_coefficients() {
        let skeleton = build_formula_skeleton(FormulaRequest::new(0, 3, 2)).unwrap();
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
        let skeleton = build_formula_skeleton(FormulaRequest::new(0, 3, 2)).unwrap();
        let rendered = skeleton.render_tex();
        assert!(rendered.contains("\\section*{Givental Graph Formula Skeleton}"));
        assert!(rendered.contains("\\begin{tikzpicture}"));
        assert!(rendered.contains("\\draw[leg]"));
        assert!(rendered.contains("(R^{-1}_{0})_{i_{0},j}"));
        assert!(rendered.contains("(\\Psi^{-1})_{j,\\beta}"));
        assert!(rendered.contains("(S_{0})_{\\beta,\\alpha}"));
        assert!(rendered.contains("\\Delta_{i_{0}}^{-1}"));
        assert!(rendered.contains("\\left\\langle \\tau_{0}\\tau_{0}\\tau_{0}\\right\\rangle_{0}"));
    }

    #[test]
    fn tex_document_renderer_is_standalone() {
        let skeleton = build_formula_skeleton(FormulaRequest::new(0, 3, 2)).unwrap();
        let rendered = skeleton.render_tex_document();
        assert!(rendered.starts_with("\\documentclass[11pt]{article}"));
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
