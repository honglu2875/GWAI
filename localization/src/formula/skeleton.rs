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
            graph.render_text(out, self.request.colors);
            out.push('\n');
        }
    }
}

impl GraphFormulaSkeleton {
    fn render_text(&self, out: &mut String, colors: usize) {
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
            colors - 1
        ));
    }
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
}
