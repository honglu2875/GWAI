//! Finite formula skeletons for fixed stable `(g,m)`.

use crate::error::GwError;
use crate::graphs::{stable_graphs, StableGraph};

use super::atoms::atom_glossary;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormulaRequest {
    pub genus: usize,
    pub markings: usize,
    pub colors: usize,
    pub max_descendant_power: usize,
    pub q_degree: Option<usize>,
    pub include_glossary: bool,
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
        if self.request.include_glossary {
            out.push('\n');
            out.push_str(&atom_glossary());
        }
        self.render_graphs(&mut out);
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
            "- Edge_{{i,j}}^{{a,b}} is materialized for 0 <= a,b <= D = {}; the graph sum prunes terms whose total vertex psi degree is too large.\n",
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
        out.push_str("How the atoms assemble\n");
        out.push_str("----------------------\n");
        out.push_str("For a formal insertion at marking ell,\n");
        out.push_str("  gamma_ell = sum_{k<=K,a} x_{ell,k,a} tau_k(phi_a),\n");
        out.push_str("the descendant leg of final color i and ancestor psi power p is\n");
        out.push_str("  Leg_{ell,i}^p = sum x_{ell,k,a} RInv_r[i,j] PsiInv[j,b] S_s[b,a]\n");
        out.push_str(
            "where p = k - s + r, 0 <= s <= k, and repeated flat/color indices are summed.\n\n",
        );
        out.push_str("For an internal edge between colors i and j, Edge_{i,j}^{a,b} is the\n");
        out.push_str("regularized symplectic propagator coefficient carrying psi powers a and b\n");
        out.push_str("to the two endpoint vertices.\n\n");
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
            "    (1/{}) * sum_{{color(v) in 0..{}}} [product of leg, edge, and vertex atoms]\n",
            self.automorphism_order,
            request.colors - 1
        ));
        self.render_expanded_expression(out, request);
    }

    fn render_expanded_expression(&self, out: &mut String, request: &FormulaRequest) {
        let terms = self.expanded_terms(request);
        out.push_str("  Expanded contribution in atom coefficients:\n");
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
        out.push_str("    Here L_{ell,i}^p is the descendant leg coefficient\n");
        out.push_str(&format!(
            "      L_{{ell,i}}^p = sum_{{0<=k<=K={}, 0<=alpha,beta,j<{}, 0<=s<=k, 0<=r<=D+1={}, p=k-s+r}}\n",
            request.max_descendant_power,
            request.colors,
            request.inverse_r_order()
        ));
        out.push_str("        x_{ell,k,alpha} * RInv_r[i,j] * PsiInv[j,beta] * S_s[beta,alpha].\n");
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

    fn expanded_terms(&self, request: &FormulaRequest) -> Vec<String> {
        let assignments = self.power_assignments(request);
        let mut terms = Vec::new();
        for assignment in assignments {
            let mut fixed_factors = Vec::new();
            for (marking, &power) in assignment.leg_powers.iter().enumerate() {
                let vertex = self.graph.legs[marking];
                fixed_factors.push(format!("L_{{{marking},i{vertex}}}^{power}"));
            }
            for (edge_index, &(left_power, right_power)) in
                assignment.edge_powers.iter().enumerate()
            {
                let edge = &self.graph.edges[edge_index];
                fixed_factors.push(format!(
                    "Edge_{{i{},i{}}}^{{{left_power},{right_power}}}",
                    edge.a, edge.b
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

fn powered_factor(base: &str, exponent: usize) -> String {
    match exponent {
        0 => "1".to_string(),
        1 => base.to_string(),
        _ => format!("({base})^{exponent}"),
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
        assert!(skeleton.render_text().contains("Atom glossary"));
    }

    #[test]
    fn graph_renderer_unravels_atom_coefficients() {
        let skeleton = build_formula_skeleton(FormulaRequest::new(0, 3, 2)).unwrap();
        let rendered = skeleton.render_text();
        assert!(rendered.contains("Expanded contribution in atom coefficients"));
        assert!(rendered.contains("L_{0,i0}^0"));
        assert!(rendered.contains("DeltaInv_{i0}"));
        assert!(rendered.contains("PsiInt(0;0,0,0)"));
        assert!(rendered.contains("L_{ell,i}^p"));
    }

    #[test]
    fn vertex_terms_expand_translation_partitions() {
        let terms = vertex_expanded_terms(1, "i0", &[0]);
        assert_eq!(terms.len(), 1);
        assert!(terms[0].contains("T_{i0}^2"));
        assert!(terms[0].contains("PsiInt(1;0,2)"));
    }
}
