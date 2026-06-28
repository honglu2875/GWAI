//! Legacy direct stable-map localization validation backend.
//!
//! This module is intentionally not a production computation path. It preserves
//! the old direct fixed-locus code for narrow tests and convention checks. It is
//! not actively maintained for new features, and it deliberately does not fall
//! back to seed formulas or the Givental backend.

use crate::algebra::{lambda, LaurentSeries, RatFun, Rational};
use crate::error::GwError;
use crate::geometry::EquivariantProjectiveSpace;
use crate::tautological::{TautologicalOracle, WittenKontsevich};
use crate::{InvariantRequest, InvariantResult};
use std::collections::BTreeSet;

pub fn compute(req: &InvariantRequest) -> Result<InvariantResult, GwError> {
    if req.equivariant {
        if let Some(value) = genus_zero_primary_localization(req)? {
            return Ok(InvariantResult {
                value,
                engine: "localization-primary-tree",
                notes: vec![
                    "computed by genus-zero primary tree localization; result remains equivariant"
                        .to_string(),
                ],
            });
        }
    } else if let Some(value) = genus_zero_primary_localization_nonequivariant(req)? {
        return Ok(InvariantResult {
            value: RatFun::from_rational(value),
            engine: "localization-primary-tree-limit",
            notes: vec![
                "computed by genus-zero primary tree localization using lambda-line Laurent summation"
                    .to_string(),
            ],
        });
    }

    Err(GwError::UnsupportedInvariant(format!(
        "legacy direct localization only implements degree-one genus-zero primary tree checks; requested n={}, g={}, d={}, markings={}",
        req.n,
        req.genus,
        req.degree,
        req.insertions.len()
    )))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocVertex {
    pub fixed_point: usize,
    pub genus: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocEdge {
    pub from: usize,
    pub to: usize,
    pub degree: usize,
}

impl LocEdge {
    pub fn new(from: usize, to: usize, degree: usize) -> Self {
        if from <= to {
            Self { from, to, degree }
        } else {
            Self {
                from: to,
                to: from,
                degree,
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MarkedLeg {
    pub vertex: usize,
    pub marking: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocGraph {
    pub vertices: Vec<LocVertex>,
    pub edges: Vec<LocEdge>,
    pub legs: Vec<MarkedLeg>,
}

impl LocGraph {
    pub fn degree(&self) -> usize {
        self.edges.iter().map(|edge| edge.degree).sum()
    }

    pub fn first_betti(&self) -> usize {
        if self.vertices.is_empty() {
            0
        } else {
            self.edges.len() + 1 - self.vertices.len()
        }
    }

    pub fn genus(&self) -> usize {
        self.vertices.iter().map(|v| v.genus).sum::<usize>() + self.first_betti()
    }

    pub fn valence(&self, vertex: usize) -> usize {
        let edge_valence = self
            .edges
            .iter()
            .map(|edge| usize::from(edge.from == vertex) + usize::from(edge.to == vertex))
            .sum::<usize>();
        let leg_valence = self.legs.iter().filter(|leg| leg.vertex == vertex).count();
        edge_valence + leg_valence
    }

    pub fn is_connected(&self) -> bool {
        if self.vertices.is_empty() {
            return false;
        }
        let mut dsu = DisjointSet::new(self.vertices.len());
        for edge in &self.edges {
            dsu.union(edge.from, edge.to);
        }
        let root = dsu.find(0);
        (1..self.vertices.len()).all(|vertex| dsu.find(vertex) == root)
    }

    pub fn canonical_label(&self) -> String {
        let mut best = None::<String>;
        for permutation in permutations(self.vertices.len()) {
            let label = self.label_with_permutation(&permutation);
            if best.as_ref().is_none_or(|current| label < *current) {
                best = Some(label);
            }
        }
        best.unwrap_or_default()
    }

    pub fn graph_automorphism_order(&self) -> usize {
        let identity = (0..self.vertices.len()).collect::<Vec<_>>();
        let base_label = self.label_with_permutation(&identity);
        permutations(self.vertices.len())
            .into_iter()
            .filter(|permutation| self.label_with_permutation(permutation) == base_label)
            .count()
    }

    pub fn cover_automorphism_order(&self) -> usize {
        self.edges
            .iter()
            .map(|edge| edge.degree)
            .product::<usize>()
            .max(1)
            * self.graph_automorphism_order()
    }

    fn label_with_permutation(&self, permutation: &[usize]) -> String {
        let mut inverse = vec![0usize; permutation.len()];
        for (new, &old) in permutation.iter().enumerate() {
            inverse[old] = new;
        }

        let vertices = permutation
            .iter()
            .map(|&old| {
                format!(
                    "{}:{}",
                    self.vertices[old].fixed_point, self.vertices[old].genus
                )
            })
            .collect::<Vec<_>>()
            .join(",");

        let mut edges = self
            .edges
            .iter()
            .map(|edge| {
                let mapped = LocEdge::new(inverse[edge.from], inverse[edge.to], edge.degree);
                format!("{}-{}:{}", mapped.from, mapped.to, mapped.degree)
            })
            .collect::<Vec<_>>();
        edges.sort();

        let mut legs = self
            .legs
            .iter()
            .map(|leg| format!("{}:{}", leg.marking, inverse[leg.vertex]))
            .collect::<Vec<_>>();
        legs.sort();

        format!("v:{vertices}|e:{}|l:{}", edges.join(","), legs.join(","))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightedLocGraph {
    pub graph: LocGraph,
    pub automorphism_order: usize,
}

pub fn localization_graphs(req: &InvariantRequest) -> Result<Vec<WeightedLocGraph>, GwError> {
    if req.genus != 0 {
        return Err(GwError::UnsupportedInvariant(
            "localization graph enumeration currently supports genus zero".to_string(),
        ));
    }
    Ok(genus_zero_localization_graphs(
        req.n,
        req.degree,
        req.insertions.len(),
    ))
}

pub fn genus_zero_localization_graphs(
    target_n: usize,
    degree: usize,
    markings: usize,
) -> Vec<WeightedLocGraph> {
    if degree == 0 {
        return (0..=target_n)
            .map(|fixed_point| {
                let graph = LocGraph {
                    vertices: vec![LocVertex {
                        fixed_point,
                        genus: 0,
                    }],
                    edges: Vec::new(),
                    legs: (0..markings)
                        .map(|marking| MarkedLeg { vertex: 0, marking })
                        .collect(),
                };
                WeightedLocGraph {
                    graph,
                    automorphism_order: 1,
                }
            })
            .collect();
    }

    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for edge_count in 1..=degree {
        let vertex_count = edge_count + 1;
        for abstract_edges in labelled_trees(vertex_count) {
            for fixed_points in colorings(vertex_count, target_n + 1, &abstract_edges) {
                for degrees in positive_compositions(degree, edge_count) {
                    for leg_vertices in labelled_leg_assignments(markings, vertex_count) {
                        let graph = LocGraph {
                            vertices: fixed_points
                                .iter()
                                .map(|&fixed_point| LocVertex {
                                    fixed_point,
                                    genus: 0,
                                })
                                .collect(),
                            edges: abstract_edges
                                .iter()
                                .zip(degrees.iter())
                                .map(|(&(from, to), &edge_degree)| {
                                    LocEdge::new(from, to, edge_degree)
                                })
                                .collect(),
                            legs: leg_vertices
                                .iter()
                                .enumerate()
                                .map(|(marking, &vertex)| MarkedLeg { vertex, marking })
                                .collect(),
                        };
                        debug_assert!(graph.is_connected());
                        debug_assert_eq!(graph.genus(), 0);
                        let label = graph.canonical_label();
                        if seen.insert(label) {
                            let automorphism_order = graph.cover_automorphism_order();
                            out.push(WeightedLocGraph {
                                graph,
                                automorphism_order,
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn genus_zero_primary_one_edge_localization(
    req: &InvariantRequest,
) -> Result<Option<RatFun>, GwError> {
    let Some(value) = genus_zero_primary_localization(req)? else {
        return Ok(None);
    };
    if req.degree == 0 {
        return Ok(None);
    }
    Ok(Some(value))
}

pub fn genus_zero_primary_localization(req: &InvariantRequest) -> Result<Option<RatFun>, GwError> {
    if req.genus != 0
        || req.degree == 0
        || req.degree != 1
        || !req
            .insertions
            .iter()
            .all(|insertion| insertion.descendant_power == 0)
    {
        return Ok(None);
    }

    let target = EquivariantProjectiveSpace::new(req.n);
    let psi = WittenKontsevich::new();
    let mut total = RatFun::zero();
    for weighted in localization_graphs(req)?
        .into_iter()
        .filter(|weighted| weighted.graph.edges.len() == 1)
    {
        let contribution = primary_graph_contribution(&target, &psi, &weighted, &req.insertions);
        total = &total + &contribution;
    }

    Ok(Some(total))
}

pub fn genus_zero_primary_localization_nonequivariant(
    req: &InvariantRequest,
) -> Result<Option<Rational>, GwError> {
    if req.genus != 0
        || req.degree == 0
        || req.degree != 1
        || !req
            .insertions
            .iter()
            .all(|insertion| insertion.descendant_power == 0)
    {
        return Ok(None);
    }

    if req
        .insertion_degree()
        .is_some_and(|actual| actual as isize != req.virtual_dimension())
    {
        return Ok(Some(Rational::zero()));
    }

    let weights = default_lambda_line_weights(req.n);
    let target = EquivariantProjectiveSpace::new(req.n);
    let psi = WittenKontsevich::new();
    let mut total = LaurentSeries::zero();
    for weighted in localization_graphs(req)?
        .into_iter()
        .filter(|weighted| weighted.graph.edges.len() == 1)
    {
        let contribution = primary_graph_contribution(&target, &psi, &weighted, &req.insertions);
        let series = contribution.lambda_line_laurent_series(req.n, &weights, 0)?;
        total = total.add(&series);
    }
    Ok(Some(total.finite_limit()?))
}

fn default_lambda_line_weights(n: usize) -> Vec<Rational> {
    let mut weights = Vec::with_capacity(n + 1);
    let mut value = 1usize;
    for _ in 0..=n {
        weights.push(Rational::from(value));
        value = value.saturating_mul(2);
    }
    weights
}

fn primary_graph_contribution(
    target: &EquivariantProjectiveSpace,
    psi: &dyn TautologicalOracle,
    weighted: &WeightedLocGraph,
    insertions: &[crate::Insertion],
) -> RatFun {
    let mut contribution = RatFun::one();
    for edge in &weighted.graph.edges {
        contribution = &contribution * &edge_factor(target, &weighted.graph, edge);
    }
    contribution = &contribution / &RatFun::from(weighted.automorphism_order);

    for vertex_id in 0..weighted.graph.vertices.len() {
        let factor = primary_vertex_factor(target, psi, &weighted.graph, vertex_id, insertions);
        contribution = &contribution * &factor;
    }
    contribution
}

fn primary_vertex_factor(
    target: &EquivariantProjectiveSpace,
    psi: &dyn TautologicalOracle,
    graph: &LocGraph,
    vertex_id: usize,
    insertions: &[crate::Insertion],
) -> RatFun {
    let vertex = &graph.vertices[vertex_id];
    let flags = incident_flags(graph, vertex_id);
    let mut marking_factor = RatFun::one();
    let mut markings = 0usize;
    for leg in graph.legs.iter().filter(|leg| leg.vertex == vertex_id) {
        markings += 1;
        let restriction = insertions[leg.marking]
            .class
            .restrict_to_fixed_point(vertex.fixed_point);
        marking_factor = &marking_factor * &restriction;
    }

    let euler = target.fixed_point_euler(vertex.fixed_point);
    let normal_factor = match graph.valence(vertex_id) {
        1 => &flags[0].omega / &euler,
        2 if markings == 1 => &RatFun::one() / &euler,
        2 if markings == 0 => &(&RatFun::one() / &euler) / &(&flags[0].omega + &flags[1].omega),
        valence => stable_primary_vertex_factor(psi, &euler, &flags, markings, valence),
    };
    &normal_factor * &marking_factor
}

#[derive(Debug, Clone)]
struct FlagWeight {
    omega: RatFun,
}

fn incident_flags(graph: &LocGraph, vertex_id: usize) -> Vec<FlagWeight> {
    graph
        .edges
        .iter()
        .filter_map(|edge| {
            if edge.from == vertex_id {
                Some((edge.to, edge.degree))
            } else if edge.to == vertex_id {
                Some((edge.from, edge.degree))
            } else {
                None
            }
        })
        .map(|(neighbor, degree)| FlagWeight {
            omega: tangent_weight(
                graph.vertices[vertex_id].fixed_point,
                graph.vertices[neighbor].fixed_point,
                degree,
            ),
        })
        .collect()
}

fn stable_primary_vertex_factor(
    psi: &dyn TautologicalOracle,
    euler: &RatFun,
    flags: &[FlagWeight],
    markings: usize,
    valence: usize,
) -> RatFun {
    debug_assert!(valence >= 3);
    let target_degree = valence - 3;
    let mut total = RatFun::zero();
    for powers in weak_compositions(target_degree, flags.len()) {
        let mut powers_with_markings = powers.clone();
        powers_with_markings.extend(std::iter::repeat_n(0, markings));
        let integral = psi.psi_integral(0, &powers_with_markings);
        if integral.is_zero() {
            continue;
        }
        let mut term = RatFun::from_rational(integral);
        for (flag, power) in flags.iter().zip(powers.iter()) {
            term = &term / &flag.omega.pow_usize(power + 1);
        }
        total = &total + &term;
    }
    &total / euler
}

fn weak_compositions(total: usize, parts: usize) -> Vec<Vec<usize>> {
    fn rec(total: usize, parts: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == parts {
            if total == 0 {
                out.push(current.clone());
            }
            return;
        }
        for value in 0..=total {
            current.push(value);
            rec(total - value, parts, current, out);
            current.pop();
        }
    }
    if parts == 0 {
        return if total == 0 {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
    }
    let mut out = Vec::new();
    rec(total, parts, &mut Vec::new(), &mut out);
    out
}

fn edge_factor(target: &EquivariantProjectiveSpace, graph: &LocGraph, edge: &LocEdge) -> RatFun {
    let i = graph.vertices[edge.from].fixed_point;
    let j = graph.vertices[edge.to].fixed_point;
    debug_assert!(i <= target.n && j <= target.n);
    let degree = edge.degree;
    let li = lambda(i);
    let lj = lambda(j);
    let diff = &li - &lj;

    let sign = if degree.is_multiple_of(2) { 1 } else { -1 };
    let mut factor = RatFun::from_rational(Rational::new(
        sign * (factorial(degree) * factorial(degree)) as i128,
        degree.pow((2 * degree) as u32) as i128,
    ));
    factor = &factor * &target.fixed_point_euler(i);
    factor = &factor * &target.fixed_point_euler(j);
    factor = &factor / &diff.pow_usize(2 * degree);

    for k in 0..=target.n {
        if k == i || k == j {
            continue;
        }
        for a in 0..=degree {
            let left_coeff = RatFun::from(degree - a);
            let right_coeff = RatFun::from(a);
            let numerator =
                &(&left_coeff * &(&li - &lambda(k))) + &(&right_coeff * &(&lj - &lambda(k)));
            let denominator = &numerator / &RatFun::from(degree);
            factor = &factor / &denominator;
        }
    }

    factor
}

fn tangent_weight(from_fixed_point: usize, to_fixed_point: usize, degree: usize) -> RatFun {
    &(&lambda(from_fixed_point) - &lambda(to_fixed_point)) / &RatFun::from(degree)
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

fn labelled_trees(vertex_count: usize) -> Vec<Vec<(usize, usize)>> {
    if vertex_count == 1 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    let pairs = vertex_pairs(vertex_count);
    for edges in edge_subsets(&pairs, vertex_count - 1) {
        let graph = LocGraph {
            vertices: vec![
                LocVertex {
                    fixed_point: 0,
                    genus: 0,
                };
                vertex_count
            ],
            edges: edges
                .iter()
                .map(|&(from, to)| LocEdge::new(from, to, 1))
                .collect(),
            legs: Vec::new(),
        };
        if graph.is_connected() {
            out.push(edges);
        }
    }
    out
}

fn vertex_pairs(vertex_count: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for a in 0..vertex_count {
        for b in a + 1..vertex_count {
            pairs.push((a, b));
        }
    }
    pairs
}

fn edge_subsets(pairs: &[(usize, usize)], size: usize) -> Vec<Vec<(usize, usize)>> {
    fn rec(
        pairs: &[(usize, usize)],
        size: usize,
        start: usize,
        current: &mut Vec<(usize, usize)>,
        out: &mut Vec<Vec<(usize, usize)>>,
    ) {
        if current.len() == size {
            out.push(current.clone());
            return;
        }
        for idx in start..pairs.len() {
            current.push(pairs[idx]);
            rec(pairs, size, idx + 1, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    rec(pairs, size, 0, &mut Vec::new(), &mut out);
    out
}

fn colorings(vertex_count: usize, color_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    fn rec(
        vertex_count: usize,
        color_count: usize,
        edges: &[(usize, usize)],
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == vertex_count {
            out.push(current.clone());
            return;
        }
        let vertex = current.len();
        for color in 0..color_count {
            let valid = edges.iter().all(|&(a, b)| {
                if a == vertex && b < current.len() {
                    current[b] != color
                } else if b == vertex && a < current.len() {
                    current[a] != color
                } else {
                    true
                }
            });
            if valid {
                current.push(color);
                rec(vertex_count, color_count, edges, current, out);
                current.pop();
            }
        }
    }
    let mut out = Vec::new();
    rec(vertex_count, color_count, edges, &mut Vec::new(), &mut out);
    out
}

fn positive_compositions(total: usize, parts: usize) -> Vec<Vec<usize>> {
    fn rec(total: usize, parts: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() + 1 == parts {
            if total > 0 {
                current.push(total);
                out.push(current.clone());
                current.pop();
            }
            return;
        }
        let remaining_parts = parts - current.len() - 1;
        for value in 1..=total.saturating_sub(remaining_parts) {
            current.push(value);
            rec(total - value, parts, current, out);
            current.pop();
        }
    }
    if parts == 0 {
        return if total == 0 {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
    }
    let mut out = Vec::new();
    rec(total, parts, &mut Vec::new(), &mut out);
    out
}

fn labelled_leg_assignments(legs: usize, vertex_count: usize) -> Vec<Vec<usize>> {
    fn rec(legs: usize, vertex_count: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == legs {
            out.push(current.clone());
            return;
        }
        for vertex in 0..vertex_count {
            current.push(vertex);
            rec(legs, vertex_count, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    rec(legs, vertex_count, &mut Vec::new(), &mut out);
    out
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, values: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if start == values.len() {
            out.push(values.clone());
            return;
        }
        for idx in start..values.len() {
            values.swap(start, idx);
            rec(start + 1, values, out);
            values.swap(start, idx);
        }
    }
    let mut values = (0..n).collect::<Vec<_>>();
    let mut out = Vec::new();
    rec(0, &mut values, &mut out);
    out
}

#[derive(Debug, Clone)]
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[rb] = ra;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{CohomologyClass, EquivariantProjectiveSpace};
    use crate::{tau, ComputeMode};

    #[test]
    fn degree_zero_graphs_are_one_vertex_per_fixed_point() {
        let graphs = genus_zero_localization_graphs(2, 0, 3);
        assert_eq!(graphs.len(), 3);
        assert!(graphs.iter().all(|weighted| {
            weighted.graph.vertices.len() == 1
                && weighted.graph.edges.is_empty()
                && weighted.graph.legs.len() == 3
                && weighted.automorphism_order == 1
        }));
    }

    #[test]
    fn p1_degree_one_has_single_unmarked_fixed_locus() {
        let graphs = genus_zero_localization_graphs(1, 1, 0);
        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].graph.degree(), 1);
        assert_eq!(graphs[0].automorphism_order, 1);
    }

    #[test]
    fn p2_degree_one_unmarked_graphs_are_fixed_lines() {
        let graphs = genus_zero_localization_graphs(2, 1, 0);
        assert_eq!(graphs.len(), 3);
    }

    #[test]
    fn request_graphs_reject_high_genus_for_now() {
        let req = InvariantRequest {
            n: 1,
            genus: 1,
            degree: 1,
            insertions: vec![tau(0, CohomologyClass::one(1))],
            equivariant: true,
            mode: ComputeMode::Givental,
            truncation: None,
        };
        assert!(matches!(
            localization_graphs(&req),
            Err(GwError::UnsupportedInvariant(_))
        ));
    }

    #[test]
    fn p1_degree_one_fixed_point_primary_evaluates_equivariantly() {
        let target = EquivariantProjectiveSpace::new(1);
        let phi0 = target.fixed_point_idempotent(0);
        let phi1 = target.fixed_point_idempotent(1);
        let req = InvariantRequest {
            n: 1,
            genus: 0,
            degree: 1,
            insertions: vec![tau(0, phi0), tau(0, phi1)],
            equivariant: true,
            mode: ComputeMode::Givental,
            truncation: None,
        };
        let value = genus_zero_primary_one_edge_localization(&req)
            .unwrap()
            .unwrap();
        let expected =
            &RatFun::one() / &(&target.fixed_point_euler(0) * &target.fixed_point_euler(1));
        assert_eq!(value, expected);
    }

    #[test]
    fn legacy_compute_uses_one_edge_evaluator_for_equivariant_request() {
        let target = EquivariantProjectiveSpace::new(1);
        let req = InvariantRequest {
            n: 1,
            genus: 0,
            degree: 1,
            insertions: vec![
                tau(0, target.fixed_point_idempotent(0)),
                tau(0, target.fixed_point_idempotent(1)),
            ],
            equivariant: true,
            mode: ComputeMode::Givental,
            truncation: None,
        };
        let result = compute(&req).unwrap();
        assert_eq!(result.engine, "localization-primary-tree");
    }

    #[test]
    fn legacy_compute_does_not_fall_back_to_seed_formulas() {
        let req = InvariantRequest::new(2, 0, 2, vec![tau(0, CohomologyClass::h_power(2, 2)); 5]);
        assert!(matches!(
            compute(&req),
            Err(GwError::UnsupportedInvariant(_))
        ));
    }
}
